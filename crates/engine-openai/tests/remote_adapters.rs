use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, header};
use axum::routing::{get, post};
use futures::TryStreamExt;
use locus_core::{
    CanonicalRequest, CapabilityRequirements, EngineEvent, EngineInstance, EngineInstanceId,
    EngineInstanceRef, ExecutionRole, ExecutionTarget, ExecutionTargetId,
    GenerationSemanticIdentity, InputBundle, InputSemanticIdentity, ModelExecutionIdentity,
    OperationContext, OutputSemanticIdentity, ParallelLayout, RequestId, RuntimeIdentity,
    SamplingParameters, SemanticComponentIdentity, SemanticIdentity, TypedMetadata,
};
use locus_engine::EngineAdapter;
use locus_engine_openai::{RemoteEngineConfig, SglangEngineAdapter, VllmEngineAdapter};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Capture {
    completions: Arc<Mutex<Vec<Value>>>,
    aborts: Arc<Mutex<Vec<Value>>>,
    health_calls: Arc<AtomicUsize>,
}

async fn health(State(capture): State<Capture>) -> &'static str {
    capture.health_calls.fetch_add(1, Ordering::AcqRel);
    "ok"
}

async fn completions(State(capture): State<Capture>, Json(body): Json<Value>) -> Response<Body> {
    capture
        .completions
        .lock()
        .expect("completion lock")
        .push(body);
    let body = concat!(
        "data: {\"choices\":[{\"index\":0,\"text\":\"hel\",\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"text\":\"lo\",\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(body))
        .expect("SSE response")
}

async fn abort(State(capture): State<Capture>, Json(body): Json<Value>) {
    capture.aborts.lock().expect("abort lock").push(body);
}

async fn server() -> (String, Capture) {
    let capture = Capture::default();
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/completions", post(completions))
        .route("/abort_request", post(abort))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let address = listener.local_addr().expect("mock address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve mock engine");
    });
    (format!("http://{address}"), capture)
}

fn component(kind: &str) -> SemanticComponentIdentity {
    SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: "v1".to_owned(),
        fingerprint: format!("{kind}-v1"),
    }
}

fn identities() -> (ModelExecutionIdentity, SemanticIdentity) {
    (
        ModelExecutionIdentity {
            model_revision: "model-v1".to_owned(),
            adapter_revision: None,
            execution_profile: "default".to_owned(),
        },
        SemanticIdentity {
            input: InputSemanticIdentity {
                tokenizer: component("tokenizer"),
                template: component("template"),
                multimodal_preprocessing: None,
            },
            generation: GenerationSemanticIdentity {
                sampling_normalization: component("sampling"),
                stop_behavior: component("stop"),
                constrained_generation: Some(component("json-schema")),
            },
            output: OutputSemanticIdentity {
                detokenizer: component("detokenizer"),
                reasoning_parser: None,
                tool_parser: None,
            },
            umbrella_fingerprint: None,
        },
    )
}

fn config(base_url: String, kind: &str) -> RemoteEngineConfig {
    let (model, _) = identities();
    let engine = EngineInstanceRef {
        id: EngineInstanceId::new(format!("{kind}-engine")),
        generation: 1,
    };
    RemoteEngineConfig {
        base_url,
        api_key: Some("secret".to_owned()),
        served_model: "served-model".to_owned(),
        instance: EngineInstance {
            reference: engine.clone(),
            runtime: RuntimeIdentity {
                kind: kind.to_owned(),
                runtime_version: "test".to_owned(),
                adapter_version: "locus-v1".to_owned(),
            },
            topology: "local".to_owned(),
            hardware: "mock".to_owned(),
            health_endpoint: None,
        },
        target: ExecutionTarget {
            id: ExecutionTargetId::new(format!("{kind}-target")),
            engine,
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
    }
}

fn request(id: &str) -> CanonicalRequest {
    let (model, semantic_identity) = identities();
    let mut input = InputBundle::token_sequence("prompt", vec![1, 2, 3]);
    input.annotations.push(TypedMetadata {
        type_url: "locus.generation-contract.v1".to_owned(),
        fields: BTreeMap::from([
            ("response_format".to_owned(), "json_schema".to_owned()),
            ("json_schema_name".to_owned(), "answer".to_owned()),
            (
                "json_schema".to_owned(),
                json!({"type": "object"}).to_string(),
            ),
            ("strict".to_owned(), "true".to_owned()),
        ]),
    });
    CanonicalRequest {
        id: RequestId::new(id),
        model,
        semantic_identity,
        requirements: CapabilityRequirements::for_input(&input),
        input,
        sampling: SamplingParameters {
            max_output_tokens: Some(8),
            stop_sequences: vec!["END".to_owned(), "DONE".to_owned()],
            ..SamplingParameters::default()
        },
    }
}

fn tool_request(id: &str, with_profile_parser: bool) -> CanonicalRequest {
    let mut request = request(id);
    request.input.annotations.push(TypedMetadata {
        type_url: "locus.tool.v1".to_owned(),
        fields: BTreeMap::from([
            ("name".to_owned(), "weather".to_owned()),
            (
                "parameters".to_owned(),
                json!({"type": "object"}).to_string(),
            ),
        ]),
    });
    if with_profile_parser {
        request.semantic_identity.output.tool_parser = Some(component("tool-parser"));
    }
    request
}

async fn assert_completion(
    adapter: &dyn EngineAdapter,
    expected_request_field: &str,
    capture: &Capture,
) {
    let target = adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover")))
        .await
        .expect("targets")
        .remove(0);
    let snapshot = adapter
        .snapshot(&target, &OperationContext::new(RequestId::new("health")))
        .await
        .expect("snapshot");
    assert!(snapshot.ready);
    let events = adapter
        .execute(
            &target,
            request("req-remote"),
            None,
            OperationContext::new(RequestId::new("req-remote")),
        )
        .await
        .expect("execute")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect events");
    let text = events
        .iter()
        .filter_map(|event| match event {
            EngineEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "hello");
    assert!(
        matches!(events.last(), Some(EngineEvent::Finished { usage, .. }) if usage.input_tokens == 3)
    );
    let bodies = capture.completions.lock().expect("completion lock");
    let body = bodies.last().expect("completion body");
    assert_eq!(body["prompt"], json!([1, 2, 3]));
    assert_eq!(body["stop"], json!(["END", "DONE"]));
    assert_eq!(body[expected_request_field], "req-remote");
}

async fn assert_profile_parser_tool_transport(adapter: &dyn EngineAdapter, capture: &Capture) {
    let target = adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover-tools")))
        .await
        .expect("targets")
        .remove(0);
    let capabilities = adapter
        .capabilities(
            &target,
            &OperationContext::new(RequestId::new("capabilities-tools")),
        )
        .await
        .expect("capabilities");
    assert!(!capabilities.emits_tool_calls);
    let parser_request = tool_request("req-tools", true);
    assert!(capabilities.satisfies(&parser_request.requirements));
    adapter
        .execute(
            &target,
            parser_request,
            None,
            OperationContext::new(RequestId::new("req-tools")),
        )
        .await
        .expect("profile parser permits tool transport")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect events");
    assert_eq!(
        capture.completions.lock().expect("completion lock").len(),
        1
    );

    let error = adapter
        .execute(
            &target,
            tool_request("req-tools-no-parser", false),
            None,
            OperationContext::new(RequestId::new("req-tools-no-parser")),
        )
        .await
        .err()
        .expect("missing profile parser must reject tools");
    assert!(error.to_string().contains("model-profile tool parser"));

    let mut native_tool_request = tool_request("req-native-tools", true);
    native_tool_request.requirements.requires_tool_calls = true;
    let error = adapter
        .execute(
            &target,
            native_tool_request,
            None,
            OperationContext::new(RequestId::new("req-native-tools")),
        )
        .await
        .err()
        .expect("native tool event requirement must be rejected");
    assert!(error.to_string().contains("does not emit native tool-call"));

    let mut native_reasoning_request = request("req-native-reasoning");
    native_reasoning_request
        .semantic_identity
        .output
        .reasoning_parser = Some(component("reasoning-parser"));
    native_reasoning_request
        .requirements
        .requires_reasoning_deltas = true;
    let error = adapter
        .execute(
            &target,
            native_reasoning_request,
            None,
            OperationContext::new(RequestId::new("req-native-reasoning")),
        )
        .await
        .err()
        .expect("native reasoning event requirement must be rejected");
    assert!(error.to_string().contains("does not emit native reasoning"));
}

#[tokio::test]
async fn sglang_adapter_streams_pretokenized_completions_and_aborts_by_rid() {
    let (base_url, capture) = server().await;
    let adapter = SglangEngineAdapter::new(config(base_url, "sglang")).expect("adapter");
    assert_completion(&adapter, "rid", &capture).await;
    adapter
        .cancel(
            &RequestId::new("req-remote"),
            &OperationContext::new(RequestId::new("cancel")),
        )
        .await
        .expect("cancel");
    assert_eq!(
        capture.aborts.lock().expect("abort lock").as_slice(),
        &[json!({"rid": "req-remote", "abort_all": false})]
    );
}

#[tokio::test]
async fn sglang_and_vllm_transport_tool_prompts_only_with_a_profile_parser() {
    let (sglang_url, sglang_capture) = server().await;
    let sglang = SglangEngineAdapter::new(config(sglang_url, "sglang")).expect("adapter");
    assert_profile_parser_tool_transport(&sglang, &sglang_capture).await;

    let (vllm_url, vllm_capture) = server().await;
    let vllm = VllmEngineAdapter::new(config(vllm_url, "vllm")).expect("adapter");
    assert_profile_parser_tool_transport(&vllm, &vllm_capture).await;
}

#[tokio::test]
async fn vllm_adapter_streams_pretokenized_completions_with_request_id() {
    let (base_url, capture) = server().await;
    let adapter = VllmEngineAdapter::new(config(base_url, "vllm")).expect("adapter");
    assert_completion(&adapter, "request_id", &capture).await;
    let body = capture
        .completions
        .lock()
        .expect("completion lock")
        .last()
        .cloned()
        .expect("completion body");
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(body["add_special_tokens"], false);
}

#[test]
fn remote_adapter_test_contract_uses_only_locus_types() {
    let supported = BTreeSet::from([locus_core::InputKind::TokenSequence]);
    assert_eq!(supported.len(), 1);
}
