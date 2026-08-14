use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use futures::TryStreamExt;
use locus_core::{
    EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole,
    ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OperationContext, OutputSemanticIdentity,
    ParallelLayout, RequestId, RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
    StateKind, StateRequirement,
};
use locus_engine::{EngineRegistry, FakeEngineAdapter, FakeEngineOutput};
use locus_model_io::{
    BasicModelIo, ByteDecoder, ByteTokenizer, Conversation, ConversationMessage, ConversationRole,
    ModelProfile, ModelRegistry, ModelRequest, SimpleTemplateRenderer,
};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_store::{StateStore, StoreError};
use locus_store_nexuskv::{NexusKvStore, NexusKvStoreConfig};
use serde_json::{Value, json};

#[derive(Clone, Default)]
struct Capture {
    lookups: Arc<Mutex<Vec<Value>>>,
    estimates: Arc<Mutex<Vec<Value>>>,
    materializations: Arc<Mutex<Vec<Value>>>,
    mismatch_validation: bool,
    fail_materialization: bool,
}

async fn lookup(State(capture): State<Capture>, Json(body): Json<Value>) -> Json<Value> {
    capture
        .lookups
        .lock()
        .expect("lookup capture")
        .push(body.clone());
    let input_semantics = if capture.mismatch_validation {
        json!({
            "tokenizer": {"kind": "wrong", "revision": "v1", "fingerprint": "wrong"},
            "template": body["locus_input_semantic_identity"]["template"],
            "multimodal_preprocessing": null
        })
    } else {
        body["locus_input_semantic_identity"].clone()
    };
    Json(json!({
        "schema_version": "locus.nexuskv-bridge.v1",
        "nexuskv_schema_version": "nexuskv.contract.v1",
        "match_result": {
            "classification": "partial",
            "matched_extent": {"units": 5, "granularity": "token"},
            "entry": {
                "identity": {
                    "key": {"model": "model-v1"},
                    "entry_id": "nexus-state-1",
                    "version": {"generation": 7, "lineage": "lineage-a"}
                },
                "descriptor": {
                    "schema_version": "nexuskv.contract.v1",
                    "descriptor_id": "descriptor-v1",
                    "semantic_type": "mha_kv",
                    "granularity": "token"
                },
                "location": {"tier": "host_dram", "locator": "shm://nexus-state-1"}
            },
            "compatibility": {
                "reusable": true, "fallback_to_recompute": false, "reason": "prefix match"
            },
            "validation": {
                "model_identity": body["locus_model_identity"],
                "input_semantic_identity": input_semantics,
                "source_handle": "source-capability-1"
            }
        }
    }))
}

async fn estimate(State(capture): State<Capture>, Json(body): Json<Value>) -> Json<Value> {
    capture
        .estimates
        .lock()
        .expect("estimate capture")
        .push(body);
    Json(json!({
        "schema_version": "locus.nexuskv-bridge.v1",
        "option_id": "nexus-option-1",
        "option_handle": "transfer-plan-1",
        "locality": "local",
        "topology_path": null,
        "estimated_transfer_micros": 0
    }))
}

async fn materialize(
    State(capture): State<Capture>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    capture
        .materializations
        .lock()
        .expect("materialization capture")
        .push(body);
    if capture.fail_materialization {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "schema_version": "locus.nexuskv-bridge.v1",
                "error": {
                    "code": "materialization_unavailable",
                    "message": "injected bridge materialization failure"
                }
            })),
        );
    }
    (
        StatusCode::OK,
        Json(json!({
            "schema_version": "locus.nexuskv-bridge.v1",
            "bytes_transferred": 0,
            "receipt": {
                "namespace": "nexuskv.protocol-transfer-receipt.v1",
                "value": "receipt-1"
            },
            "evidence": {
                "level": "protocol",
                "physical_transfer_verified": false
            }
        })),
    )
}

async fn bridge(mismatch_validation: bool, fail_materialization: bool) -> (String, Capture) {
    let capture = Capture {
        mismatch_validation,
        fail_materialization,
        ..Capture::default()
    };
    let app = Router::new()
        .route("/locus/v1/lookup", post(lookup))
        .route("/locus/v1/estimate", post(estimate))
        .route("/locus/v1/materialize", post(materialize))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind bridge");
    let address = listener.local_addr().expect("bridge address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve bridge");
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
                constrained_generation: None,
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

fn inference_service(
    base_url: String,
    api_key: Option<String>,
) -> (DefaultInferenceService, Arc<FakeEngineAdapter>) {
    let store: Arc<dyn StateStore> = Arc::new(
        NexusKvStore::new(NexusKvStoreConfig {
            base_url,
            api_key,
            tenant: "tenant-a".to_owned(),
            namespace: "default".to_owned(),
            engine_family: "sglang".to_owned(),
            semantic_type: "mha_kv".to_owned(),
        })
        .expect("store"),
    );
    let (model, semantic_identity) = identities();
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelIo::new(
            ModelProfile {
                public_aliases: vec!["nexus-model".to_owned()],
                model: model.clone(),
                semantic_identity: semantic_identity.clone(),
            },
            Arc::new(ByteTokenizer::new(
                semantic_identity.input.tokenizer.clone(),
            )),
            Arc::new(SimpleTemplateRenderer::new(
                semantic_identity.input.template.clone(),
            )),
            Arc::new(ByteDecoder),
        )))
        .expect("register model");
    let engine_ref = EngineInstanceRef {
        id: EngineInstanceId::new("engine-1"),
        generation: 1,
    };
    let adapter = Arc::new(
        FakeEngineAdapter::new(
            EngineInstance {
                reference: engine_ref.clone(),
                runtime: RuntimeIdentity {
                    kind: "fake-state-aware-engine".to_owned(),
                    runtime_version: "v1".to_owned(),
                    adapter_version: "v1".to_owned(),
                },
                topology: "local".to_owned(),
                hardware: "cpu".to_owned(),
                health_endpoint: None,
            },
            ExecutionTarget {
                id: ExecutionTargetId::new("target-1"),
                engine: engine_ref,
                model,
                role: ExecutionRole::Combined,
                parallel_layout: ParallelLayout {
                    tensor_parallel: 1,
                    pipeline_parallel: 1,
                    expert_parallel: 1,
                    layout_revision: "v1".to_owned(),
                },
                residency: "node-a".to_owned(),
                capability_revision: "v1".to_owned(),
            },
            EngineCapabilities {
                supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                emits_token_deltas: false,
                emits_text_deltas: true,
                emits_reasoning_deltas: false,
                emits_tool_calls: false,
                supports_structured_output: false,
                supported_state_kinds: BTreeSet::from([StateKind::new("nexuskv.mha_kv")]),
            },
        )
        .with_output(FakeEngineOutput {
            token_deltas: Vec::new(),
            text_deltas: vec!["ok".to_owned()],
            ..FakeEngineOutput::default()
        }),
    );
    let engines = EngineRegistry::new();
    engines.register(adapter.clone()).expect("register engine");
    (
        DefaultInferenceService::new(models, engines, store),
        adapter,
    )
}

async fn run_inference(service: &DefaultInferenceService, request_id: &str) {
    service
        .infer(
            ModelRequest {
                model: "nexus-model".to_owned(),
                input: locus_model_io::ModelInput::Conversation(Conversation {
                    messages: vec![ConversationMessage {
                        role: ConversationRole::User,
                        content: "use cached prefix".to_owned(),
                        tool_call_id: None,
                    }],
                }),
                ..ModelRequest::default()
            },
            OperationContext::new(RequestId::new(request_id)),
        )
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect inference");
}

#[tokio::test]
async fn nexuskv_bridge_runs_lookup_estimate_and_import_handshake_end_to_end() {
    let (base_url, capture) = bridge(false, false).await;
    let (service, adapter) = inference_service(base_url, Some("bridge-secret".to_owned()));
    run_inference(&service, "req-nexus").await;

    assert_eq!(adapter.call_counts().prepare, 1);
    assert_eq!(adapter.call_counts().commit, 1);
    assert_eq!(adapter.call_counts().execute, 1);
    assert_eq!(capture.lookups.lock().expect("lookups").len(), 1);
    assert_eq!(capture.estimates.lock().expect("estimates").len(), 1);
    let materializations = capture.materializations.lock().expect("materializations");
    assert_eq!(materializations.len(), 1);
    assert_eq!(
        materializations[0]["sink_namespace"],
        "locus.fake.engine-sink.v1"
    );
    assert_eq!(materializations[0]["target_engine_generation"], 1);
    let estimates = capture.estimates.lock().expect("estimates");
    assert_eq!(estimates[0]["source_state"], "nexus-state-1");
    assert_eq!(estimates[0]["source_handle"], "source-capability-1");
}

#[tokio::test]
async fn nexuskv_bridge_materialization_failure_aborts_and_falls_back_cold() {
    let (base_url, capture) = bridge(false, true).await;
    let (service, adapter) = inference_service(base_url, None);
    run_inference(&service, "req-fallback").await;

    let calls = adapter.call_counts();
    assert_eq!(calls.prepare, 1);
    assert_eq!(calls.commit, 0);
    assert_eq!(calls.abort, 1);
    assert_eq!(calls.execute, 1);
    assert_eq!(
        capture
            .materializations
            .lock()
            .expect("materializations")
            .len(),
        1
    );
}

#[tokio::test]
async fn nexuskv_bridge_fails_closed_on_semantic_validation_mismatch() {
    let (base_url, _) = bridge(true, false).await;
    let store = NexusKvStore::new(NexusKvStoreConfig {
        base_url,
        api_key: None,
        tenant: "tenant-a".to_owned(),
        namespace: "default".to_owned(),
        engine_family: "sglang".to_owned(),
        semantic_type: "mha_kv".to_owned(),
    })
    .expect("store");
    let (model, semantic_identity) = identities();
    let error = store
        .lookup(
            &StateRequirement {
                model,
                input_semantics: semantic_identity.input,
                accepted_state_kinds: BTreeSet::from([StateKind::new("nexuskv.mha_kv")]),
                input_fingerprint: "input-v1".to_owned(),
                query_token_ids: Some(vec![1, 2, 3]),
                tenant_scope: None,
            },
            &OperationContext::new(RequestId::new("req-mismatch")),
        )
        .await
        .expect_err("semantic mismatch must fail closed");
    assert!(matches!(error, StoreError::Incompatible(_)));
}
