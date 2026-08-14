use std::collections::BTreeSet;
use std::sync::Arc;

use futures::TryStreamExt;
use locus_core::{
    EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole,
    ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OperationContext, OutputSemanticIdentity,
    ParallelLayout, RequestId, RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
    StateKind,
};
use locus_engine::{EngineRegistry, FakeEngineAdapter, FakeEngineOutput};
use locus_model_io::{
    BasicModelIo, ByteDecoder, ByteTokenizer, Conversation, ConversationMessage, ConversationRole,
    ModelProfile, ModelRegistry, ModelRequest, SimpleTemplateRenderer,
};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_state::StateProvider;
use locus_state_nexuskv::{NexusKvBridgeConfig, NexusKvStateProvider};

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

fn inference_service(base_url: String) -> (DefaultInferenceService, Arc<FakeEngineAdapter>) {
    let provider: Arc<dyn StateProvider> = Arc::new(
        NexusKvStateProvider::new(NexusKvBridgeConfig {
            base_url,
            api_key: None,
            tenant: "tenant-a".to_owned(),
            namespace: "default".to_owned(),
            engine_family: "sglang".to_owned(),
            semantic_type: "mha_kv".to_owned(),
        })
        .expect("provider"),
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
        DefaultInferenceService::new(models, engines, provider),
        adapter,
    )
}

#[tokio::test]
#[ignore = "requires a separately running NexusKV bridge process"]
async fn real_nexuskv_process_completes_locus_plan_and_import_handshake() {
    let base_url = std::env::var("LOCUS_NEXUSKV_BRIDGE_URL")
        .expect("LOCUS_NEXUSKV_BRIDGE_URL must name the external bridge process");
    let (service, adapter) = inference_service(base_url);

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
            OperationContext::new(RequestId::new("req-real-nexuskv")),
        )
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect inference");

    let calls = adapter.call_counts();
    assert_eq!(calls.prepare, 1);
    assert_eq!(calls.commit, 1);
    assert_eq!(calls.abort, 0);
    assert_eq!(calls.execute, 1);
}
