use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use locus_core::{
    EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole,
    ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OutputSemanticIdentity, ParallelLayout,
    RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
};
use locus_engine::{EngineRegistry, FakeEngineAdapter, FakeEngineOutput, FakeToolCall};
use locus_model_io::{
    BasicModelIo, ByteDecoder, ByteTokenizer, ModelProfile, ModelRegistry, SimpleTemplateRenderer,
};
use locus_openai::{ApiConfig, TenantCredential, router, router_with_config};
use locus_runtime::{DefaultInferenceService, InferenceService, TrafficController, TrafficPolicy};
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
                reasoning_parser: Some(component("reasoning")),
                tool_parser: Some(component("tool")),
            },
            umbrella_fingerprint: None,
        },
    )
}

fn service(output: FakeEngineOutput) -> (Arc<dyn InferenceService>, Arc<FakeEngineAdapter>) {
    service_with_traffic(output, None)
}

fn service_with_traffic(
    output: FakeEngineOutput,
    traffic: Option<TrafficController>,
) -> (Arc<dyn InferenceService>, Arc<FakeEngineAdapter>) {
    let (model, semantics_identity) = identities();
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelIo::new(
            ModelProfile {
                public_aliases: vec!["locus-test".to_owned(), "locus-test-latest".to_owned()],
                model: model.clone(),
                semantic_identity: semantics_identity.clone(),
            },
            Arc::new(ByteTokenizer::new(
                semantics_identity.input.tokenizer.clone(),
            )),
            Arc::new(SimpleTemplateRenderer::new(
                semantics_identity.input.template.clone(),
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
                supports_structured_output: true,
                supported_state_kinds: BTreeSet::new(),
            },
        )
        .with_output(output),
    );
    let engines = EngineRegistry::new();
    engines.register(adapter.clone()).expect("register engine");
    let store: Arc<dyn StateStore> = Arc::new(NullStateStore::default());
    let service = DefaultInferenceService::new(models, engines, store);
    let service: Arc<dyn InferenceService> = Arc::new(match traffic {
        Some(traffic) => service.with_traffic_control(traffic),
        None => service,
    });
    (service, adapter)
}

fn app(output: FakeEngineOutput) -> (axum::Router, Arc<FakeEngineAdapter>) {
    let (service, adapter) = service(output);
    (router(service), adapter)
}

async fn json_response(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Value,
) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("response body")
        .to_bytes();
    let value = serde_json::from_slice(&bytes).expect("JSON response");
    (status, value)
}

#[tokio::test]
async fn health_models_and_non_streaming_response_are_openai_shaped() {
    let (app, adapter) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hello".to_owned()],
        ..FakeEngineOutput::default()
    });
    let (health_status, health) = json_response(app.clone(), "GET", "/healthz", json!({})).await;
    assert_eq!(health_status, StatusCode::OK);
    assert_eq!(health["status"], "ok");
    let (models_status, models) = json_response(app.clone(), "GET", "/v1/models", json!({})).await;
    assert_eq!(models_status, StatusCode::OK);
    assert_eq!(models["object"], "list");
    assert_eq!(models["data"].as_array().expect("model list").len(), 2);

    let (status, response) = json_response(
        app,
        "POST",
        "/v1/responses",
        json!({"model": "locus-test", "input": "Say hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["object"], "response");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["output"][0]["type"], "message");
    assert_eq!(response["output"][0]["content"][0]["text"], "hello");
    assert_eq!(adapter.call_counts().execute, 1);
}

#[tokio::test]
async fn legacy_completions_use_raw_text_or_token_prompts_without_a_chat_template() {
    let (app, adapter) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["raw completion".to_owned()],
        ..FakeEngineOutput::default()
    });
    let (status, response) = json_response(
        app.clone(),
        "POST",
        "/v1/completions",
        json!({
            "model": "locus-test",
            "prompt": "abc",
            "max_tokens": 8,
            "temperature": 0.25,
            "top_p": 0.9,
            "seed": 7,
            "stop": ["END", "DONE"],
            "n": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["object"], "text_completion");
    assert_eq!(response["choices"][0]["text"], "raw completion");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(response["usage"]["completion_tokens"], 14);

    let (status, response) = json_response(
        app,
        "POST",
        "/v1/completions",
        json!({"model": "locus-test", "prompt": [65, 66, 67]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["usage"]["prompt_tokens"], 3);
    assert_eq!(adapter.call_counts().execute, 2);
}

#[tokio::test]
async fn legacy_completion_stream_emits_text_finish_usage_and_done() {
    let (app, _) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hel".to_owned(), "lo".to_owned()],
        finish_reason: locus_core::EngineFinishReason::Length,
        ..FakeEngineOutput::default()
    });
    let response = app
        .oneshot(
            Request::post("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "locus-test",
                        "prompt": "raw",
                        "stream": true,
                        "stream_options": {"include_usage": true}
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("stream body")
            .to_bytes()
            .to_vec(),
    )
    .expect("UTF-8 stream");
    assert!(body.contains("\"object\":\"text_completion\""));
    assert!(body.contains("\"text\":\"hel\""));
    assert!(body.contains("\"text\":\"lo\""));
    assert!(body.contains("\"finish_reason\":\"length\""));
    assert!(body.contains("\"prompt_tokens\":3"));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn legacy_completions_reject_batch_and_unsupported_options_explicitly() {
    let cases = [
        (json!({"prompt": ["one", "two"]}), "prompt"),
        (json!({"prompt": [[1, 2], [3, 4]]}), "prompt"),
        (json!({"prompt": "x", "n": 2}), "n"),
        (json!({"prompt": "x", "best_of": 2}), "best_of"),
        (json!({"prompt": "x", "echo": true}), "echo"),
        (json!({"prompt": "x", "logprobs": 1}), "logprobs"),
        (json!({"prompt": "x", "suffix": "y"}), "suffix"),
    ];
    for (mut body, parameter) in cases {
        body["model"] = json!("locus-test");
        let (app, adapter) = app(FakeEngineOutput::default());
        let (status, response) = json_response(app, "POST", "/v1/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "parameter {parameter}");
        assert_eq!(response["error"]["param"], parameter);
        assert_eq!(adapter.call_counts().execute, 0);
    }
}

#[tokio::test]
async fn readiness_observes_registered_model_and_live_target() {
    let (app, _) = app(FakeEngineOutput::default());
    let (status, readiness) = json_response(app, "GET", "/readyz", json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(readiness["status"], "ready");
    assert_eq!(readiness["model_profiles"], 1);
    assert_eq!(readiness["routable_models"], 1);
    assert_eq!(readiness["required_models"], 0);
    assert_eq!(readiness["ready_targets"], 1);
    assert_eq!(readiness["observed_targets"], 1);
}

#[tokio::test]
async fn bearer_auth_protects_api_routes_but_not_probes() {
    let (service, _) = service(FakeEngineOutput::default());
    let app = router_with_config(
        service,
        ApiConfig {
            bearer_token: Some("test-secret".to_owned()),
            ..ApiConfig::default()
        },
    )
    .expect("configured router");
    let health = app
        .clone()
        .oneshot(
            Request::get("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("health response");
    assert_eq!(health.status(), StatusCode::OK);

    let unauthorized = app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("unauthorized response");
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(unauthorized.headers()[header::WWW_AUTHENTICATE], "Bearer");
    let error: Value = serde_json::from_slice(
        &unauthorized
            .into_body()
            .collect()
            .await
            .expect("error body")
            .to_bytes(),
    )
    .expect("error JSON");
    assert_eq!(error["error"]["code"], "invalid_api_key");

    let authorized = app
        .oneshot(
            Request::get("/v1/models")
                .header(header::AUTHORIZATION, "Bearer test-secret")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("authorized response");
    assert_eq!(authorized.status(), StatusCode::OK);
}

#[tokio::test]
async fn bearer_credentials_are_the_only_source_of_tenant_policy() {
    let mut policy = TrafficPolicy::default();
    let mut alpha = policy.tenants[0].clone();
    alpha.id = "alpha".to_owned();
    let mut beta = alpha.clone();
    beta.id = "beta".to_owned();
    policy.tenants = vec![alpha, beta];
    let traffic = TrafficController::new(policy).expect("traffic controller");
    let (service, _) = service_with_traffic(FakeEngineOutput::default(), Some(traffic.clone()));
    let app = router_with_config(
        service,
        ApiConfig {
            bearer_token: None,
            tenant_credentials: vec![
                TenantCredential {
                    tenant_id: "alpha".to_owned(),
                    bearer_token: "alpha-secret".to_owned(),
                },
                TenantCredential {
                    tenant_id: "beta".to_owned(),
                    bearer_token: "beta-secret".to_owned(),
                },
            ],
            anonymous_tenant: None,
            traffic,
            ..ApiConfig::default()
        },
    )
    .expect("configured router");

    let response = app
        .clone()
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::AUTHORIZATION, "Bearer alpha-secret")
                .header("x-tenant-id", "beta")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "trusted tenant"}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    response
        .into_body()
        .collect()
        .await
        .expect("consume response");

    let metrics = app
        .clone()
        .oneshot(
            Request::get("/metrics")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("metrics")
        .into_body()
        .collect()
        .await
        .expect("metrics body")
        .to_bytes();
    let metrics = std::str::from_utf8(&metrics).expect("UTF-8 metrics");
    assert!(metrics.contains(
        "locus_admission_requests_total{class=\"standard\",tenant=\"alpha\",outcome=\"admitted\"} 1"
    ));
    assert!(!metrics.contains(
        "locus_admission_requests_total{class=\"standard\",tenant=\"beta\",outcome=\"admitted\"}"
    ));

    let unauthorized = app
        .oneshot(
            Request::get("/v1/models")
                .header(header::AUTHORIZATION, "Bearer unknown")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("unauthorized response");
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn tenant_deadline_clamps_untrusted_header_and_cancels_stream() {
    let mut policy = TrafficPolicy::default();
    policy.tenants[0].default_request_timeout = Duration::from_millis(25);
    policy.tenants[0].max_request_timeout = Duration::from_millis(25);
    let traffic = TrafficController::new(policy).expect("traffic controller");
    let metrics_traffic = traffic.clone();
    let (service, adapter) = service_with_traffic(
        FakeEngineOutput {
            event_delay: Duration::from_millis(100),
            ..FakeEngineOutput::default()
        },
        Some(traffic.clone()),
    );
    let app = router_with_config(
        service,
        ApiConfig {
            traffic,
            ..ApiConfig::default()
        },
    )
    .expect("configured router");
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-request-timeout-ms", "10000")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "deadline"}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("deadline response");
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    tokio::task::yield_now().await;
    assert_eq!(adapter.call_counts().cancel, 1);
    let metrics = metrics_traffic.prometheus().expect("metrics");
    assert!(metrics.contains("reason=\"deadline\""));
    assert!(!metrics.contains("reason=\"client_cancelled\""));
}

#[tokio::test]
async fn tenant_deadline_covers_request_body_ingress_before_inference() {
    let mut policy = TrafficPolicy::default();
    policy.tenants[0].default_request_timeout = Duration::from_millis(20);
    policy.tenants[0].max_request_timeout = Duration::from_millis(20);
    let traffic = TrafficController::new(policy).expect("traffic controller");
    let metrics_traffic = traffic.clone();
    let (service, adapter) =
        service_with_traffic(FakeEngineOutput::default(), Some(traffic.clone()));
    let app = router_with_config(
        service,
        ApiConfig {
            traffic,
            ..ApiConfig::default()
        },
    )
    .expect("configured router");
    let payload = json!({"model": "locus-test", "input": "slow body"}).to_string();
    let body = Body::from_stream(futures::stream::once(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok::<_, Infallible>(Bytes::from(payload))
    }));

    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-request-timeout-ms", "10000")
                .body(body)
                .expect("request"),
        )
        .await
        .expect("deadline response");

    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(adapter.call_counts().execute, 0);
    assert!(
        !metrics_traffic
            .prometheus()
            .expect("metrics")
            .contains("outcome=\"admitted\"")
    );
}

#[tokio::test]
async fn configured_body_limit_rejects_oversized_requests() {
    let (service, _) = service(FakeEngineOutput::default());
    let app = router_with_config(
        service,
        ApiConfig {
            max_request_bytes: 32,
            ..ApiConfig::default()
        },
    )
    .expect("configured router");
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "this body is deliberately too large"})
                        .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn concurrency_permit_is_held_until_stream_body_is_dropped() {
    let mut policy = TrafficPolicy {
        max_active_requests: 1,
        ..TrafficPolicy::default()
    };
    policy.classes[0].max_active_requests = 1;
    policy.tenants[0].max_active_requests = 1;
    let traffic = TrafficController::new(policy).expect("traffic controller");
    let metrics_traffic = traffic.clone();
    let (service, _) = service_with_traffic(FakeEngineOutput::default(), Some(traffic.clone()));
    let app = router_with_config(
        service,
        ApiConfig {
            traffic,
            ..ApiConfig::default()
        },
    )
    .expect("configured router");
    let first = app
        .clone()
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "hold", "stream": true}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("first response");
    let blocked = tokio::time::timeout(
        Duration::from_millis(20),
        app.clone().oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "wait", "stream": true}).to_string(),
                ))
                .expect("request"),
        ),
    )
    .await;
    assert!(blocked.is_err(), "second request must wait for the stream");
    drop(first);
    tokio::task::yield_now().await;
    assert!(
        metrics_traffic
            .prometheus()
            .expect("metrics")
            .contains("reason=\"client_cancelled\"")
    );
    let admitted = tokio::time::timeout(
        Duration::from_secs(1),
        app.oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "admit", "stream": true}).to_string(),
                ))
                .expect("request"),
        ),
    )
    .await
    .expect("second request should be admitted")
    .expect("models response");
    assert_eq!(admitted.status(), StatusCode::OK);
}

#[tokio::test]
async fn responses_sse_has_created_delta_done_and_completed_order() {
    let (app, _) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hel".to_owned(), "lo".to_owned()],
        ..FakeEngineOutput::default()
    });
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "hello", "stream": true}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()[header::CONTENT_TYPE]
            .to_str()
            .expect("content type")
            .starts_with("text/event-stream")
    );
    let body = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("SSE body")
            .to_bytes()
            .to_vec(),
    )
    .expect("UTF-8 SSE");
    let event_types = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|value| value["type"].as_str().map(str::to_owned))
        .collect::<Vec<_>>();
    let created = event_types
        .iter()
        .position(|event| event == "response.created")
        .expect("created event");
    let delta = event_types
        .iter()
        .position(|event| event == "response.output_text.delta")
        .expect("delta event");
    let done = event_types
        .iter()
        .position(|event| event == "response.output_text.done")
        .expect("done event");
    let completed = event_types
        .iter()
        .position(|event| event == "response.completed")
        .expect("completed event");
    assert!(created < delta && delta < done && done < completed);
}

#[tokio::test]
async fn function_reasoning_and_structured_output_share_the_model_io_pipeline() {
    let (app, _) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["{\"answer\":\"ok\"}".to_owned()],
        reasoning_deltas: vec!["checked constraints".to_owned()],
        tool_calls: vec![FakeToolCall {
            call_id: "call_weather".to_owned(),
            name: "weather".to_owned(),
            argument_deltas: vec!["{\"city\":".to_owned(), "\"Beijing\"}".to_owned()],
        }],
        ..FakeEngineOutput::default()
    });
    let (status, response) = json_response(
        app,
        "POST",
        "/v1/responses",
        json!({
            "model": "locus-test",
            "input": "answer and call a tool",
            "reasoning": {"effort": "high"},
            "tools": [{"type": "function", "name": "weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]
            }, "strict": true}],
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": {
                "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]
            }, "strict": true}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let output = response["output"].as_array().expect("output array");
    assert!(output.iter().any(|item| item["type"] == "reasoning"));
    assert!(output.iter().any(|item| item["type"] == "message"));
    let tool = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call");
    assert_eq!(tool["name"], "weather");
    assert_eq!(tool["arguments"], "{\"city\":\"Beijing\"}");
}

#[tokio::test]
async fn chat_completions_reuses_service_and_emits_tool_calls() {
    let (app, _) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        tool_calls: vec![FakeToolCall {
            call_id: "call_1".to_owned(),
            name: "lookup".to_owned(),
            argument_deltas: vec!["{}".to_owned()],
        }],
        ..FakeEngineOutput::default()
    });
    let (status, response) = json_response(
        app,
        "POST",
        "/v1/chat/completions",
        json!({
            "model": "locus-test",
            "messages": [{"role": "user", "content": "look it up"}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object"}
            }}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(response["object"], "chat.completion");
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup"
    );
}

#[tokio::test]
async fn chat_completions_stream_uses_chunks_and_done_sentinel() {
    let (app, _) = app(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hello".to_owned()],
        ..FakeEngineOutput::default()
    });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "locus-test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": true
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    let body = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("stream body")
            .to_bytes()
            .to_vec(),
    )
    .expect("UTF-8 stream");
    assert!(body.contains("\"object\":\"chat.completion.chunk\""));
    assert!(body.contains("\"content\":\"hello\""));
    assert!(body.contains("data: [DONE]"));
}

#[tokio::test]
async fn errors_use_standard_envelope_and_unknown_fields_are_not_ignored() {
    let (app, _) = app(FakeEngineOutput::default());
    let (unknown_status, unknown) = json_response(
        app.clone(),
        "POST",
        "/v1/responses",
        json!({"model": "locus-test", "input": "hi", "mystery": true}),
    )
    .await;
    assert_eq!(unknown_status, StatusCode::BAD_REQUEST);
    assert_eq!(unknown["error"]["type"], "invalid_request_error");
    assert_eq!(unknown["error"]["code"], "invalid_json");

    let (missing_status, missing) = json_response(
        app,
        "POST",
        "/v1/responses",
        json!({"model": "missing", "input": "hi"}),
    )
    .await;
    assert_eq!(missing_status, StatusCode::NOT_FOUND);
    assert_eq!(missing["error"]["code"], "model_not_found");
    assert_eq!(missing["error"]["param"], "model");
}

#[tokio::test]
async fn dropping_sse_body_cancels_the_selected_engine_request() {
    let (app, adapter) = app(FakeEngineOutput::default());
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "locus-test", "input": "hi", "stream": true}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    drop(response);
    tokio::task::yield_now().await;
    assert_eq!(adapter.call_counts().cancel, 1);
}
