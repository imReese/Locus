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
use locus_engine_openai::{
    RemoteEngineConfig, RemoteExecutionTarget, RemoteTelemetryConfig, SglangEngineAdapter,
    VllmEngineAdapter,
};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Capture {
    completions: Arc<Mutex<Vec<Value>>>,
    aborts: Arc<Mutex<Vec<Value>>>,
    health_calls: Arc<AtomicUsize>,
    models: Arc<Mutex<BTreeSet<String>>>,
    metrics: Arc<Mutex<String>>,
}

async fn health(State(capture): State<Capture>) -> &'static str {
    capture.health_calls.fetch_add(1, Ordering::AcqRel);
    "ok"
}

async fn models(State(capture): State<Capture>) -> Json<Value> {
    let data = capture
        .models
        .lock()
        .expect("models lock")
        .iter()
        .map(|id| json!({"id": id, "object": "model"}))
        .collect::<Vec<_>>();
    Json(json!({"object": "list", "data": data}))
}

async fn metrics(State(capture): State<Capture>) -> String {
    capture.metrics.lock().expect("metrics lock").clone()
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
    capture
        .models
        .lock()
        .expect("models lock")
        .insert("served-model".to_owned());
    *capture.metrics.lock().expect("metrics lock") = concat!(
        "sglang:num_running_reqs{model_name=\"served-model\"} 2\n",
        "sglang:num_queue_reqs{model_name=\"served-model\"} 3\n",
        "sglang:token_usage{model_name=\"served-model\"} 0.25\n",
        "sglang:gen_throughput{model_name=\"served-model\"} 40\n",
        "vllm:num_requests_running{model_name=\"served-model\"} 2\n",
        "vllm:num_requests_waiting{model_name=\"served-model\"} 3\n",
        "vllm:kv_cache_usage_perc{model_name=\"served-model\"} 0.25\n",
        "vllm:prompt_tokens_total{model_name=\"served-model\"} 100\n",
        "vllm:generation_tokens_total{model_name=\"served-model\"} 50\n",
    )
    .to_owned();
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/models", get(models))
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
        targets: vec![RemoteExecutionTarget {
            served_model: "served-model".to_owned(),
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
        }],
        telemetry: RemoteTelemetryConfig {
            min_scrape_interval_millis: 1,
            ..RemoteTelemetryConfig::default()
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
    assert_eq!(
        snapshot.telemetry_status,
        locus_core::TelemetryStatus::Fresh
    );
    assert_eq!(snapshot.running_requests, Some(2));
    assert_eq!(snapshot.waiting_requests, Some(3));
    assert_eq!(snapshot.kv_cache_usage_permyriad, Some(2_500));
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

#[tokio::test]
async fn execution_targets_follow_the_remote_model_inventory() {
    let (base_url, capture) = server().await;
    let mut remote = config(base_url, "sglang");
    let mut second = remote.targets[0].clone();
    second.served_model = "second-model".to_owned();
    second.target.id = ExecutionTargetId::new("sglang-second-target");
    second.target.model.model_revision = "model-v2".to_owned();
    remote.targets.push(second);
    let adapter = SglangEngineAdapter::new(remote).expect("adapter");
    let context = OperationContext::new(RequestId::new("discover-dynamic"));

    let initial = adapter
        .execution_targets(&context)
        .await
        .expect("initial targets");
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].model.model_revision, "model-v1");

    {
        let mut models = capture.models.lock().expect("models lock");
        models.clear();
        models.insert("second-model".to_owned());
        models.insert("unconfigured-model".to_owned());
    }
    let refreshed = adapter
        .execution_targets(&context)
        .await
        .expect("refreshed targets");
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].model.model_revision, "model-v2");
}

#[tokio::test]
async fn malformed_or_required_telemetry_is_not_fabricated_as_zero() {
    let (base_url, capture) = server().await;
    *capture.metrics.lock().expect("metrics lock") = "malformed metric".to_owned();
    let adapter = SglangEngineAdapter::new(config(base_url.clone(), "sglang")).expect("adapter");
    let target = adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover-malformed")))
        .await
        .expect("targets")
        .remove(0);
    let snapshot = adapter
        .snapshot(
            &target,
            &OperationContext::new(RequestId::new("snapshot-malformed")),
        )
        .await
        .expect("health-only snapshot");
    assert!(snapshot.ready);
    assert_eq!(
        snapshot.telemetry_status,
        locus_core::TelemetryStatus::Unavailable
    );
    assert_eq!(snapshot.running_requests, None);
    assert_eq!(snapshot.waiting_requests, None);
    assert_eq!(snapshot.estimated_queue_micros, None);
    assert!(snapshot.degraded_reason.is_some());

    let mut strict = config(base_url, "sglang");
    strict.telemetry.require_fresh_metrics = true;
    let strict_adapter = SglangEngineAdapter::new(strict).expect("strict adapter");
    let strict_target = strict_adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover-strict")))
        .await
        .expect("targets")
        .remove(0);
    let strict_snapshot = strict_adapter
        .snapshot(
            &strict_target,
            &OperationContext::new(RequestId::new("snapshot-strict")),
        )
        .await
        .expect("strict snapshot");
    assert!(!strict_snapshot.ready);
}

#[tokio::test]
async fn failed_refresh_preserves_but_marks_expired_telemetry_stale() {
    let (base_url, capture) = server().await;
    let mut remote = config(base_url, "sglang");
    remote.telemetry.valid_for_millis = 1;
    let adapter = SglangEngineAdapter::new(remote).expect("adapter");
    let target = adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover-stale")))
        .await
        .expect("targets")
        .remove(0);
    let fresh = adapter
        .snapshot(
            &target,
            &OperationContext::new(RequestId::new("snapshot-fresh")),
        )
        .await
        .expect("fresh snapshot");
    assert_eq!(fresh.telemetry_status, locus_core::TelemetryStatus::Fresh);

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    *capture.metrics.lock().expect("metrics lock") = "broken".to_owned();
    let stale = adapter
        .snapshot(
            &target,
            &OperationContext::new(RequestId::new("snapshot-stale")),
        )
        .await
        .expect("stale snapshot");
    assert_eq!(stale.telemetry_status, locus_core::TelemetryStatus::Stale);
    assert_eq!(stale.waiting_requests, Some(3));
    assert_eq!(
        stale.telemetry_confidence,
        locus_core::TelemetryConfidence::Unknown
    );
}

#[tokio::test]
async fn vllm_counter_deltas_produce_version_tolerant_service_rates() {
    let (base_url, capture) = server().await;
    let adapter = VllmEngineAdapter::new(config(base_url, "vllm")).expect("adapter");
    let target = adapter
        .execution_targets(&OperationContext::new(RequestId::new("discover-rates")))
        .await
        .expect("targets")
        .remove(0);
    let first = adapter
        .snapshot(
            &target,
            &OperationContext::new(RequestId::new("snapshot-rates-1")),
        )
        .await
        .expect("first snapshot");
    assert_eq!(first.prefill_tokens_per_second, None);
    assert_eq!(first.decode_tokens_per_second, None);

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    *capture.metrics.lock().expect("metrics lock") = concat!(
        "vllm_num_requests_running{model_name=\"served-model\"} 1\n",
        "vllm_num_requests_waiting{model_name=\"served-model\"} 2\n",
        "vllm_kv_cache_usage_perc{model_name=\"served-model\"} 0.5\n",
        "vllm_prompt_tokens_total{model_name=\"served-model\"} 200\n",
        "vllm_generation_tokens_total{model_name=\"served-model\"} 100\n",
    )
    .to_owned();
    let second = adapter
        .snapshot(
            &target,
            &OperationContext::new(RequestId::new("snapshot-rates-2")),
        )
        .await
        .expect("second snapshot");
    assert!(
        second
            .prefill_tokens_per_second
            .is_some_and(|rate| rate > 0)
    );
    assert!(second.decode_tokens_per_second.is_some_and(|rate| rate > 0));
    assert_eq!(second.running_requests, Some(1));
    assert_eq!(second.waiting_requests, Some(2));
    assert_eq!(second.kv_cache_usage_permyriad, Some(5_000));
}

#[test]
fn remote_adapter_test_contract_uses_only_locus_types() {
    let supported = BTreeSet::from([locus_core::InputKind::TokenSequence]);
    assert_eq!(supported.len(), 1);
}
