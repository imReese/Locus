use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::TryStreamExt;
use locus_core::{
    ContextError, EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef,
    ExecutionRole, ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity, InputKind,
    InputSemanticIdentity, ModelExecutionIdentity, OperationContext, OutputSemanticIdentity,
    ParallelLayout, RequestId, RuntimeIdentity, SemanticComponentIdentity, SemanticIdentity,
};
use locus_engine::{EngineLifecycle, EngineRegistry, FakeEngineAdapter, FakeEngineOutput};
use locus_model_io::{
    BasicModelIo, ByteDecoder, ByteTokenizer, Conversation, ConversationMessage, ConversationRole,
    ModelEvent, ModelProfile, ModelRegistry, ModelRequest, SimpleTemplateRenderer,
};
use locus_planner::{
    ACTIVE_CONFIRMATION, CalibrationKey, CalibrationObservation, CalibrationPolicy,
    PersistentCalibrator, PlacementMode, PlanExecutionError,
};
use locus_runtime::{
    DefaultInferenceService, InferenceError, InferenceService, PlacementConfigurationError,
    PlacementControl,
};
use locus_store::{NullStateStore, StateStore};

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
    let (service, adapter, _) = service_with_registry(output);
    (service, adapter)
}

fn service_with_registry(
    output: FakeEngineOutput,
) -> (
    DefaultInferenceService,
    Arc<FakeEngineAdapter>,
    EngineRegistry,
) {
    let (model, semantic_identity) = model_and_semantics();
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelIo::new(
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
    let store: Arc<dyn StateStore> = Arc::new(NullStateStore::default());
    (
        DefaultInferenceService::new(models, engines.clone(), store),
        adapter,
        engines,
    )
}

fn request() -> ModelRequest {
    ModelRequest {
        model: "test-model".to_owned(),
        input: locus_model_io::ModelInput::Conversation(Conversation {
            messages: vec![ConversationMessage {
                role: ConversationRole::User,
                content: "hello".to_owned(),
                tool_call_id: None,
            }],
        }),
        ..ModelRequest::default()
    }
}

#[tokio::test]
async fn inference_runs_through_model_io_planner_and_executor() {
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
        .expect("collect model events");
    let text = events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "hello");
    assert!(matches!(events.first(), Some(ModelEvent::Accepted { .. })));
    assert!(matches!(events.last(), Some(ModelEvent::Finished { .. })));
    assert_eq!(adapter.call_counts().execute, 1);
    assert_eq!(
        service.models().await.expect("models")[0].public_aliases[0],
        "test-model"
    );
}

#[tokio::test]
async fn dropping_model_event_stream_propagates_cancellation_to_engine() {
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

#[tokio::test]
async fn deadline_cancels_a_live_engine_stream_and_maps_to_context_error() {
    let (service, adapter) = service(FakeEngineOutput {
        event_delay: Duration::from_millis(100),
        ..FakeEngineOutput::default()
    });
    let context = OperationContext::new(RequestId::new("req-deadline"))
        .with_deadline(Instant::now() + Duration::from_millis(25));
    let error = service
        .infer(request(), context)
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect_err("deadline must terminate stream");
    assert!(matches!(
        error,
        InferenceError::Context(ContextError::DeadlineExceeded)
    ));
    tokio::task::yield_now().await;
    assert_eq!(adapter.call_counts().cancel, 1);
}

#[tokio::test]
async fn deadline_while_engine_stream_is_starting_cancels_engine() {
    let (service, adapter) = service(FakeEngineOutput {
        execute_delay: Duration::from_millis(100),
        ..FakeEngineOutput::default()
    });
    let context = OperationContext::new(RequestId::new("deadline-during-execute"))
        .with_deadline(Instant::now() + Duration::from_millis(20));

    let error = match service.infer(request(), context).await {
        Ok(_) => panic!("deadline must stop engine stream establishment"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        InferenceError::Execution(PlanExecutionError::Context(ContextError::DeadlineExceeded))
    ));
    assert_eq!(adapter.call_counts().execute, 1);
    assert_eq!(adapter.call_counts().cancel, 1);
}

#[tokio::test]
async fn engine_drain_blocks_new_work_then_forces_and_releases_active_execution() {
    let (service, adapter, engines) = service_with_registry(FakeEngineOutput {
        event_delay: Duration::from_secs(1),
        ..FakeEngineOutput::default()
    });
    let context = OperationContext::new(RequestId::new("req-drain"));
    let cancellation = context.cancellation.clone();
    let stream = service
        .infer(request(), context)
        .await
        .expect("start inference");
    assert_eq!(engines.active_executions().expect("active executions"), 1);

    let report = engines
        .drain_all(Duration::from_millis(1))
        .await
        .expect("drain engines");
    assert!(!report.completed);
    assert_eq!(report.forced_cancellations, 1);
    assert!(cancellation.is_cancelled());
    assert!(service.readiness().await.is_err());

    drop(stream);
    tokio::task::yield_now().await;
    assert_eq!(engines.active_executions().expect("released execution"), 0);
    assert_eq!(
        engines
            .lifecycle(&EngineInstanceId::new("engine-1"))
            .expect("engine lifecycle"),
        EngineLifecycle::Stopped
    );
    assert_eq!(adapter.call_counts().cancel, 1);
}

#[test]
fn active_placement_requires_confirmation_and_persistence() {
    let in_memory = PersistentCalibrator::load(CalibrationPolicy::default(), None)
        .expect("in-memory calibrator");
    let persistence_error =
        PlacementControl::new(PlacementMode::Active, in_memory, Some(ACTIVE_CONFIRMATION))
            .err()
            .expect("active mode must require persistence");
    assert_eq!(
        persistence_error,
        PlacementConfigurationError::PersistenceRequired
    );

    let temporary = tempfile::tempdir().expect("temporary directory");
    let persistent = PersistentCalibrator::load(
        CalibrationPolicy::default(),
        Some(temporary.path().join("calibration.json")),
    )
    .expect("persistent calibrator");
    let confirmation_error = PlacementControl::new(PlacementMode::Active, persistent, Some("yes"))
        .err()
        .expect("active mode must require exact confirmation");
    assert_eq!(
        confirmation_error,
        PlacementConfigurationError::ActiveConfirmation
    );
}

#[tokio::test]
async fn active_mode_falls_back_safely_before_calibration_is_qualified() {
    let (service, adapter) = service(FakeEngineOutput::default());
    let temporary = tempfile::tempdir().expect("temporary directory");
    let calibrator = PersistentCalibrator::load(
        CalibrationPolicy::default(),
        Some(temporary.path().join("calibration.json")),
    )
    .expect("persistent calibrator");
    let placement =
        PlacementControl::new(PlacementMode::Active, calibrator, Some(ACTIVE_CONFIRMATION))
            .expect("active placement configuration");
    let service = service.with_placement_control(placement);

    service
        .infer(
            request(),
            OperationContext::new(RequestId::new("req-active-gated")),
        )
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect response");

    assert_eq!(adapter.call_counts().execute, 1);
    let readiness = service.readiness().await.expect("readiness");
    assert_eq!(readiness.placement_mode, PlacementMode::Active);
    assert!(readiness.calibration_persistent);
    assert!(readiness.calibration_persistence_healthy);
    assert!(readiness.calibration_revision >= 1);
}

#[tokio::test]
async fn qualified_active_placement_selects_the_calibrated_target() {
    let (model, semantic_identity) = model_and_semantics();
    let models = ModelRegistry::new();
    models
        .register(Arc::new(BasicModelIo::new(
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

    let capabilities = EngineCapabilities {
        supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
        emits_token_deltas: true,
        emits_text_deltas: true,
        emits_reasoning_deltas: true,
        emits_tool_calls: true,
        supports_structured_output: true,
        supported_state_kinds: BTreeSet::new(),
    };
    let make_engine = |suffix: &str| {
        let engine_ref = EngineInstanceRef {
            id: EngineInstanceId::new(format!("engine-{suffix}")),
            generation: 1,
        };
        let instance = EngineInstance {
            reference: engine_ref.clone(),
            runtime: RuntimeIdentity {
                kind: "fake".to_owned(),
                runtime_version: "v1".to_owned(),
                adapter_version: "v1".to_owned(),
            },
            topology: format!("node-{suffix}"),
            hardware: "cpu".to_owned(),
            health_endpoint: None,
        };
        let target = ExecutionTarget {
            id: ExecutionTargetId::new(format!("target-{suffix}")),
            engine: engine_ref,
            model: model.clone(),
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
        (
            Arc::new(FakeEngineAdapter::new(
                instance,
                target.clone(),
                capabilities.clone(),
            )),
            target,
        )
    };
    let (engine_a, target_a) = make_engine("a");
    let (engine_b, target_b) = make_engine("b");
    let engines = EngineRegistry::new();
    engines
        .register(engine_a.clone())
        .expect("register engine a");
    engines
        .register(engine_b.clone())
        .expect("register engine b");

    let temporary = tempfile::tempdir().expect("temporary directory");
    let policy = CalibrationPolicy {
        ewma_alpha_bps: 10_000,
        min_samples_per_metric: 1,
        max_mape_bps: 10_000,
        min_shadow_decisions: 10,
        min_shadow_agreement_bps: 9_000,
        persistence_flush_every_updates: 1_000,
        ..CalibrationPolicy::default()
    };
    let calibrator =
        PersistentCalibrator::load(policy, Some(temporary.path().join("calibration.json")))
            .expect("persistent calibrator");
    for _ in 0..10 {
        calibrator
            .record_shadow_decision(true, "same", "same")
            .expect("seed shadow agreement");
    }
    for (target, ttft, generation) in [(&target_a, 100_000, 100_000), (&target_b, 100, 100)] {
        calibrator
            .record_observation(&CalibrationObservation {
                key: CalibrationKey::from_target(target),
                waiting_requests: Some(0),
                input_tokens: 10,
                output_tokens: 10,
                time_to_first_token_micros: Some(ttft),
                generation_micros: Some(generation),
                topology_micros: None,
                materialization: None,
                completed: true,
            })
            .expect("seed target calibration");
    }
    let placement =
        PlacementControl::new(PlacementMode::Active, calibrator, Some(ACTIVE_CONFIRMATION))
            .expect("active placement");
    let store: Arc<dyn StateStore> = Arc::new(NullStateStore::default());
    let service =
        DefaultInferenceService::new(models, engines, store).with_placement_control(placement);

    service
        .infer(
            request(),
            OperationContext::new(RequestId::new("req-active-qualified")),
        )
        .await
        .expect("start inference")
        .try_collect::<Vec<_>>()
        .await
        .expect("collect response");

    assert_eq!(engine_a.call_counts().execute, 0);
    assert_eq!(engine_b.call_counts().execute, 1);
}
