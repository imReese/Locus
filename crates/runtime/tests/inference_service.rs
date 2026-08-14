use std::collections::BTreeSet;
use std::sync::Arc;

use futures::TryStreamExt;
use locus_core::{
    EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole,
    ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OperationContext, OutputSemanticIdentity,
    ParallelLayout, RequestId, RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
};
use locus_engine::{EngineRegistry, FakeEngineAdapter, FakeEngineOutput};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_semantics::{
    BasicModelSemantics, ByteDecoder, ByteTokenizer, Conversation, ConversationMessage,
    ConversationRole, ModelProfile, ModelRegistry, SemanticEvent, SemanticRequest,
    SimpleTemplateRenderer,
};
use locus_state::{NullStateProvider, StateProvider};

fn component(kind: &str) -> SemanticComponentIdentity {
    SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: "v1".to_owned(),
        fingerprint: format!("{kind}-v1"),
    }
}

fn model_and_semantics() -> (ModelExecutionIdentity, SemanticIdentity) {
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
                reasoning_parser: Some(component("reasoning")),
                tool_parser: Some(component("tool")),
            },
            umbrella_fingerprint: Some("semantic-v1".to_owned()),
        },
    )
}

fn service(output: FakeEngineOutput) -> (DefaultInferenceService, Arc<FakeEngineAdapter>) {
    let (model, semantic_identity) = model_and_semantics();
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelSemantics::new(
            ModelProfile {
                public_aliases: vec!["test-model".to_owned()],
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
    let instance = EngineInstance {
        reference: engine_ref.clone(),
        runtime: RuntimeIdentity {
            kind: "fake".to_owned(),
            runtime_version: "v1".to_owned(),
            adapter_version: "v1".to_owned(),
        },
        topology: "local".to_owned(),
        hardware: "cpu".to_owned(),
        health_endpoint: None,
    };
    let target = ExecutionTarget {
        id: ExecutionTargetId::new("target-1"),
        engine: engine_ref,
        model,
        role: ExecutionRole::Combined,
        parallel_layout: ParallelLayout {
            tensor_parallel: 1,
            pipeline_parallel: 1,
            expert_parallel: 1,
            layout_revision: "layout-v1".to_owned(),
        },
        residency: "resident".to_owned(),
        capability_revision: "cap-v1".to_owned(),
    };
    let adapter = Arc::new(
        FakeEngineAdapter::new(
            instance,
            target,
            EngineCapabilities {
                supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                emits_token_deltas: true,
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
    let state_provider: Arc<dyn StateProvider> = Arc::new(NullStateProvider::default());
    (
        DefaultInferenceService::new(models, engines, state_provider),
        adapter,
    )
}

fn request() -> SemanticRequest {
    SemanticRequest {
        model: "test-model".to_owned(),
        input: locus_semantics::SemanticInput::Conversation(Conversation {
            messages: vec![ConversationMessage {
                role: ConversationRole::User,
                content: "hello".to_owned(),
                tool_call_id: None,
            }],
        }),
        ..SemanticRequest::default()
    }
}

#[tokio::test]
async fn inference_runs_through_semantics_planner_and_executor() {
    let (service, adapter) = service(FakeEngineOutput {
        token_deltas: Vec::new(),
        text_deltas: vec!["hel".to_owned(), "lo".to_owned()],
        ..FakeEngineOutput::default()
    });
    let events = service
        .infer(request(), OperationContext::new(RequestId::new("req-1")))
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect semantic events");
    let text = events
        .iter()
        .filter_map(|event| match event {
            SemanticEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "hello");
    assert!(matches!(
        events.first(),
        Some(SemanticEvent::Accepted { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(SemanticEvent::Finished { .. })
    ));
    assert_eq!(adapter.call_counts().execute, 1);
    assert_eq!(
        service.models().await.expect("models")[0].public_aliases[0],
        "test-model"
    );
}

#[tokio::test]
async fn dropping_semantic_stream_propagates_cancellation_to_engine() {
    let (service, adapter) = service(FakeEngineOutput::default());
    let stream = service
        .infer(
            request(),
            OperationContext::new(RequestId::new("req-cancel")),
        )
        .await
        .expect("start inference");
    drop(stream);
    tokio::task::yield_now().await;
    assert_eq!(adapter.call_counts().cancel, 1);
}
