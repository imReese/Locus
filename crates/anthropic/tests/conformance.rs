use std::collections::BTreeSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use locus_anthropic::{ApiConfig, TenantCredential, router_with_config};
use locus_core::{
    EngineCapabilities, EngineFinishReason, EngineInstance, EngineInstanceId, EngineInstanceRef,
    ExecutionRole, ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OutputSemanticIdentity, ParallelLayout,
    RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
};
use locus_engine::{EngineRegistry, FakeEngineAdapter, FakeEngineOutput, FakeToolCall};
use locus_model_io::{
    BasicModelIo, ByteDecoder, ByteTokenizer, ModelProfile, ModelRegistry, SimpleTemplateRenderer,
};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_store::{NullStateStore, StateStore};
use serde_json::{Value, json};
use tower::ServiceExt;

fn component(kind: &str) -> SemanticComponentIdentity {
    SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: "v1".to_owned(),
        fingerprint: format!("{kind}-v1"),
    }
}

fn service(output: FakeEngineOutput) -> (Arc<dyn InferenceService>, Arc<FakeEngineAdapter>) {
    let model = ModelExecutionIdentity {
        model_revision: "model-v1".to_owned(),
        adapter_revision: None,
        execution_profile: "default".to_owned(),
    };
    let semantics = SemanticIdentity {
        input: InputSemanticIdentity {
            tokenizer: component("tokenizer"),
            template: component("template"),
            multimodal_preprocessing: None,
        },
        generation: GenerationSemanticIdentity {
            sampling_normalization: component("sampling"),
            stop_behavior: component("stop"),
            constrained_generation: None,
        },
        output: OutputSemanticIdentity {
            detokenizer: component("detokenizer"),
            reasoning_parser: Some(component("reasoning")),
            tool_parser: Some(component("tool")),
        },
        umbrella_fingerprint: None,
    };
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelIo::new(
            ModelProfile {
                public_aliases: vec!["locus-test".to_owned()],
                model: model.clone(),
                semantic_identity: semantics.clone(),
            },
            Arc::new(ByteTokenizer::new(semantics.input.tokenizer.clone())),
            Arc::new(SimpleTemplateRenderer::new(
                semantics.input.template.clone(),
            )),
            Arc::new(ByteDecoder),
        )))
        .expect("register model");
    let engine_ref = EngineInstanceRef {
        id: EngineInstanceId::new("fake-engine"),
        generation: 1,
    };
    let adapter = Arc::new(
        FakeEngineAdapter::new(
            EngineInstance {
                reference: engine_ref.clone(),
                runtime: RuntimeIdentity {
                    kind: "fake".to_owned(),
                    runtime_version: "v1".to_owned(),
                    adapter_version: "v1".to_owned(),
                },
                topology: "local".to_owned(),
                hardware: "cpu".to_owned(),
                health_endpoint: None,
            },
            ExecutionTarget {
                id: ExecutionTargetId::new("fake-target"),
                engine: engine_ref,
                model,
                role: ExecutionRole::Combined,
                parallel_layout: ParallelLayout {
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    expert_parallel: 1,
                    layout_revision: "v1".to_owned(),
                },
                residency: "resident".to_owned(),
                capability_revision: "v1".to_owned(),
            },
            EngineCapabilities {
                supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                emits_token_deltas: false,
                emits_text_deltas: true,
                emits_reasoning_deltas: true,
                emits_tool_calls: true,
                supports_structured_output: false,
                supported_state_kinds: BTreeSet::new(),
            },
        )
        .with_output(output),
    );
    let engines = EngineRegistry::new();
    engines.register(adapter.clone()).expect("register engine");
    let store: Arc<dyn StateStore> = Arc::new(NullStateStore::default());
    (
        Arc::new(DefaultInferenceService::new(models, engines, store)),
        adapter,
    )
}

fn app(output: FakeEngineOutput) -> (axum::Router, Arc<FakeEngineAdapter>) {
    let (service, adapter) = service(output);
    (
        router_with_config(service, ApiConfig::default()).expect("Anthropic router"),
        adapter,
    )
}

async fn send(app: axum::Router, body: Value) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header(header::CONTENT_TYPE, "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let headers = response.headers().clone();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes()
        .to_vec();
    (status, headers, body)
}

#[tokio::test]
async fn non_streaming_message_matches_anthropic_shape() {
    let (app, adapter) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hello".to_owned(), " world".to_owned()],
        ..FakeEngineOutput::default()
    });
    let (status, headers, body) = send(
        app,
        json!({
            "model": "locus-test",
            "max_tokens": 32,
            "system": "be concise",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.contains_key("request-id"));
    let body: Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "hello world");
    assert_eq!(body["stop_reason"], "end_turn");
    assert!(body["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0);
    assert_eq!(adapter.call_counts().execute, 1);
}

#[tokio::test]
async fn streaming_message_emits_ordered_text_and_tool_events() {
    let (app, _) = app(FakeEngineOutput {
        text_deltas: vec!["checking".to_owned()],
        tool_calls: vec![FakeToolCall {
            call_id: "toolu_1".to_owned(),
            name: "weather".to_owned(),
            argument_deltas: vec!["{\"city\":".to_owned(), "\"Paris\"}".to_owned()],
        }],
        finish_reason: EngineFinishReason::RuntimeSpecific {
            namespace: "fake".to_owned(),
            value: "tool_calls".to_owned(),
        },
        ..FakeEngineOutput::default()
    });
    let (status, _, body) = send(
        app,
        json!({
            "model": "locus-test", "max_tokens": 32, "stream": true,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = String::from_utf8(body).expect("SSE body");
    let event_types = [
        "event: message_start",
        "event: content_block_start",
        "event: content_block_delta",
        "event: content_block_stop",
        "event: message_delta",
        "event: message_stop",
    ];
    let mut cursor = 0;
    for event_type in event_types {
        let found = body[cursor..].find(event_type).expect("ordered SSE event");
        cursor += found + event_type.len();
    }
    assert!(body.contains("input_json_delta"));
    assert!(body.contains("toolu_1"));
}

#[tokio::test]
async fn unsupported_multimodal_input_fails_before_inference() {
    let (app, adapter) = app(FakeEngineOutput::default());
    let (status, _, body) = send(
        app,
        json!({
            "model": "locus-test", "max_tokens": 32,
            "messages": [{"role": "user", "content": [{
                "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}
            }]}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(adapter.call_counts().execute, 0);
}

#[tokio::test]
async fn version_and_api_key_are_required_when_configured() {
    let (service, adapter) = service(FakeEngineOutput::default());
    let app = router_with_config(
        service,
        ApiConfig {
            tenant_credentials: vec![TenantCredential {
                tenant_id: "default".to_owned(),
                bearer_token: "secret".to_owned(),
            }],
            ..ApiConfig::default()
        },
    )
    .expect("Anthropic router");
    let request = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01")
        .header("x-api-key", "wrong")
        .body(Body::from(
            json!({"model": "locus-test", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}).to_string(),
        ))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(response.headers().contains_key("request-id"));
    assert_eq!(adapter.call_counts().execute, 0);
}
