use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use futures::TryStreamExt;
use locus_core::{
    BoundaryCompleteness, CanonicalRequest, CapabilityRequirements, CompatibilityResult,
    ComponentCoverage, EngineCapabilities, EngineInstance, EngineInstanceId, EngineInstanceRef,
    EngineSnapshot, ExecutionRole, ExecutionTarget, ExecutionTargetId, GenerationSemanticIdentity,
    InputBundle, InputItem, InputItemId, InputItemValue, InputKind, InputSemanticIdentity,
    MaterializationOption, MaterializationOptionId, MediaReference, ModelExecutionIdentity,
    OpaqueHandle, OperationContext, OutputSemanticIdentity, ParallelLayout, ProviderId, RequestId,
    ResumeCoordinate, ReusableBoundary, RuntimeIdentity, SamplingParameters,
    SemanticComponentIdentity, SemanticIdentity, StateDescriptor, StateId, StateImportSpec,
    StateKind, StateLocality, TransferReceipt,
};
use locus_engine::{EngineAdapter, EngineError, EngineRegistry, FakeEngineAdapter};
use locus_planner::{
    CostBasedPlanner, DefaultPlanExecutor, ExecutionEstimate, ExecutionPath, FallbackAction,
    PlanExecutor, Planner, PlanningCandidate, PlanningInput, RoutingPolicy, StateObservation,
    StatePathCandidate,
};
use locus_state::FakeStateProvider;

fn component(kind: &str) -> SemanticComponentIdentity {
    SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: "v1".to_owned(),
        fingerprint: format!("{kind}-fingerprint"),
    }
}

fn semantics() -> SemanticIdentity {
    SemanticIdentity {
        input: InputSemanticIdentity {
            tokenizer: component("tokenizer"),
            template: component("template"),
            multimodal_preprocessing: Some(component("media")),
        },
        generation: GenerationSemanticIdentity {
            sampling_normalization: component("sampling"),
            stop_behavior: component("stop"),
            constrained_generation: None,
        },
        output: OutputSemanticIdentity {
            detokenizer: component("detokenizer"),
            reasoning_parser: Some(component("reasoning")),
            tool_parser: Some(component("tools")),
        },
        umbrella_fingerprint: Some("semantic-profile-v1".to_owned()),
    }
}

fn model() -> ModelExecutionIdentity {
    ModelExecutionIdentity {
        model_revision: "model@abc123".to_owned(),
        adapter_revision: None,
        execution_profile: "bf16".to_owned(),
    }
}

fn token_request(id: &str) -> CanonicalRequest {
    let input = InputBundle::token_sequence("prompt", vec![1, 2, 3, 4]);
    CanonicalRequest {
        id: RequestId::new(id),
        model: model(),
        semantic_identity: semantics(),
        requirements: CapabilityRequirements::for_input(&input),
        input,
        sampling: SamplingParameters::default(),
    }
}

fn media_request(id: &str) -> CanonicalRequest {
    let input = InputBundle {
        items: vec![InputItem {
            id: InputItemId::new("image"),
            value: InputItemValue::MediaReference(MediaReference {
                media_type: "image/png".to_owned(),
                digest: "sha256:123".to_owned(),
                access_reference: "memory://image".to_owned(),
            }),
        }],
        ..InputBundle::default()
    };
    CanonicalRequest {
        id: RequestId::new(id),
        model: model(),
        semantic_identity: semantics(),
        requirements: CapabilityRequirements::for_input(&input),
        input,
        sampling: SamplingParameters::default(),
    }
}

fn engine_and_target(name: &str) -> (EngineInstance, ExecutionTarget) {
    let reference = EngineInstanceRef {
        id: EngineInstanceId::new(format!("engine-{name}")),
        generation: 1,
    };
    let instance = EngineInstance {
        reference: reference.clone(),
        runtime: RuntimeIdentity {
            kind: "fake".to_owned(),
            runtime_version: "1".to_owned(),
            adapter_version: "1".to_owned(),
        },
        topology: format!("rack-{name}"),
        hardware: "fake-accelerator".to_owned(),
        health_endpoint: None,
    };
    let target = ExecutionTarget {
        id: ExecutionTargetId::new(format!("target-{name}")),
        engine: reference,
        model: model(),
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
    (instance, target)
}

fn capabilities(input_kind: InputKind, with_state: bool) -> EngineCapabilities {
    EngineCapabilities {
        supported_input_kinds: BTreeSet::from([input_kind]),
        emits_token_deltas: true,
        emits_text_deltas: false,
        emits_reasoning_deltas: false,
        emits_tool_calls: false,
        supports_structured_output: false,
        supported_state_kinds: if with_state {
            BTreeSet::from([StateKind::new("kv")])
        } else {
            BTreeSet::new()
        },
    }
}

fn estimate(unmatched_prefill_micros: u64, topology_micros: u64) -> ExecutionEstimate {
    ExecutionEstimate {
        unmatched_prefill_micros,
        topology_micros,
        ..ExecutionEstimate::default()
    }
}

struct StateFixtureSpec<'a> {
    id: &'a str,
    coverage: u64,
    resume: u64,
    transfer_micros: u64,
    unmatched_prefill_micros: u64,
    topology_micros: u64,
    compatible: bool,
    locality: StateLocality,
}

fn state_path(
    target: &ExecutionTarget,
    provider: &ProviderId,
    spec: StateFixtureSpec<'_>,
) -> StatePathCandidate {
    let state_kind = StateKind::new("kv");
    let state = StateDescriptor {
        id: StateId::new(spec.id),
        provider: provider.clone(),
        kind: state_kind.clone(),
        model: model(),
        relevant_input_semantics: Some(semantics().input),
        representation_revision: "kv-layout-v1".to_owned(),
        positional_semantics: Some("absolute".to_owned()),
        runtime_compatibility: Some("fake-runtime-v1".to_owned()),
        boundary: ReusableBoundary {
            covered_components: vec![ComponentCoverage {
                item_id: InputItemId::new("prompt"),
                covered_units: spec.coverage,
            }],
            resume_coordinate: ResumeCoordinate::TokenOffset {
                item_id: InputItemId::new("prompt"),
                offset: spec.resume,
            },
            completeness: BoundaryCompleteness::Checkpointed,
            validation_digest: format!("boundary-{}", spec.id),
        },
        locations: vec!["fake://state".to_owned()],
        provider_reference: OpaqueHandle {
            namespace: "locus.fake.state.v1".to_owned(),
            value: spec.id.to_owned(),
        },
    };
    let option = MaterializationOption {
        id: MaterializationOptionId::new(format!("option-{}", spec.id)),
        provider: provider.clone(),
        source_state: state.id.clone(),
        target_id: target.id.clone(),
        target_engine: target.engine.clone(),
        state_kind,
        locality: spec.locality,
        estimated_transfer_micros: spec.transfer_micros,
        option_handle: OpaqueHandle {
            namespace: "locus.fake.option.v1".to_owned(),
            value: spec.id.to_owned(),
        },
    };
    StatePathCandidate {
        state,
        compatibility: if spec.compatible {
            CompatibilityResult::compatible("fixture identities match")
        } else {
            CompatibilityResult::incompatible("fixture identities differ")
        },
        option,
        estimate: estimate(spec.unmatched_prefill_micros, spec.topology_micros),
    }
}

fn candidate(
    target: &ExecutionTarget,
    engine_capabilities: EngineCapabilities,
    queue_micros: u64,
    cold_prefill_micros: u64,
    state_paths: Vec<StatePathCandidate>,
) -> PlanningCandidate {
    PlanningCandidate {
        target: target.clone(),
        capabilities: engine_capabilities,
        snapshot: EngineSnapshot {
            target_id: target.id.clone(),
            ready: true,
            queue_depth: 0,
            estimated_queue_micros: Some(queue_micros),
            observation_revision: 1,
        },
        cold_estimate: estimate(cold_prefill_micros, 0),
        state_paths,
    }
}

fn planning_input(request: CanonicalRequest, candidates: Vec<PlanningCandidate>) -> PlanningInput {
    PlanningInput {
        request,
        candidates,
        state_observation: StateObservation::default(),
        policy: RoutingPolicy::default(),
    }
}

fn assert_reuse_on(plan: &locus_planner::PlacementPlan, target: &ExecutionTarget, state: &str) {
    assert_eq!(plan.target.id, target.id);
    match &plan.path {
        ExecutionPath::Reuse(reuse) => assert_eq!(reuse.state.id, StateId::new(state)),
        ExecutionPath::Cold => panic!("expected a reusable-state path"),
    }
}

#[tokio::test]
async fn fake_vertical_slice_selects_reuse_and_executor_owns_side_effects() {
    let request = token_request("vertical-slice");
    let provider_id = ProviderId::new("fake-provider");
    let (instance_a, target_a) = engine_and_target("a");
    let (instance_b, target_b) = engine_and_target("b");
    let engine_a = Arc::new(FakeEngineAdapter::new(
        instance_a,
        target_a.clone(),
        capabilities(InputKind::TokenSequence, false),
    ));
    let engine_b = Arc::new(FakeEngineAdapter::new(
        instance_b,
        target_b.clone(),
        capabilities(InputKind::TokenSequence, true),
    ));
    let reusable = state_path(
        &target_b,
        &provider_id,
        StateFixtureSpec {
            id: "state-b",
            coverage: 900,
            resume: 900,
            transfer_micros: 50,
            unmatched_prefill_micros: 100,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let provider = Arc::new(FakeStateProvider::new(
        provider_id,
        vec![reusable.state.clone()],
        vec![reusable.option.clone()],
    ));
    let input = planning_input(
        request.clone(),
        vec![
            candidate(
                &target_a,
                capabilities(InputKind::TokenSequence, false),
                500,
                2_000,
                vec![],
            ),
            candidate(
                &target_b,
                capabilities(InputKind::TokenSequence, true),
                800,
                2_000,
                vec![reusable],
            ),
        ],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert_reuse_on(&plan, &target_b, "state-b");
    assert_eq!(engine_a.call_counts().execute, 0);
    assert_eq!(engine_b.call_counts().prepare, 0);
    assert_eq!(provider.call_counts().materialize, 0);

    let registry = EngineRegistry::new();
    registry
        .register(engine_a.clone())
        .expect("register engine a");
    registry
        .register(engine_b.clone())
        .expect("register engine b");
    let executor = DefaultPlanExecutor::new(registry, provider.clone());
    let events = executor
        .execute(
            plan,
            request.clone(),
            OperationContext::new(request.id.clone()),
        )
        .await
        .expect("execute")
        .try_collect::<Vec<_>>()
        .await
        .expect("event stream");

    assert_eq!(events.len(), 3);
    assert_eq!(engine_a.call_counts().execute, 0);
    assert_eq!(engine_b.call_counts().prepare, 1);
    assert_eq!(engine_b.call_counts().commit, 1);
    assert_eq!(engine_b.call_counts().execute, 1);
    assert_eq!(provider.call_counts().materialize, 1);
}

#[tokio::test]
async fn transfer_cost_can_lose_to_cold_recompute() {
    let request = token_request("transfer-loses");
    let provider = ProviderId::new("fake-provider");
    let (_, target) = engine_and_target("cold-wins");
    let remote = state_path(
        &target,
        &provider,
        StateFixtureSpec {
            id: "expensive-remote",
            coverage: 1_000,
            resume: 1_000,
            transfer_micros: 500,
            unmatched_prefill_micros: 0,
            topology_micros: 100,
            compatible: true,
            locality: StateLocality::Remote {
                topology_path: "rack-a/rack-b".to_owned(),
            },
        },
    );
    let input = planning_input(
        request,
        vec![candidate(
            &target,
            capabilities(InputKind::TokenSequence, true),
            0,
            300,
            vec![remote],
        )],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert!(matches!(plan.path, ExecutionPath::Cold));
    assert_eq!(plan.predicted_cost.total_micros(), 300);
}

#[tokio::test]
async fn incompatible_state_is_filtered_before_cost_scoring() {
    let request = token_request("incompatible");
    let provider = ProviderId::new("fake-provider");
    let (_, target) = engine_and_target("compatibility");
    let incompatible = state_path(
        &target,
        &provider,
        StateFixtureSpec {
            id: "cheap-but-wrong",
            coverage: 1_000,
            resume: 1_000,
            transfer_micros: 0,
            unmatched_prefill_micros: 0,
            topology_micros: 0,
            compatible: false,
            locality: StateLocality::Local,
        },
    );
    let input = planning_input(
        request,
        vec![candidate(
            &target,
            capabilities(InputKind::TokenSequence, true),
            0,
            100,
            vec![incompatible],
        )],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert!(matches!(plan.path, ExecutionPath::Cold));
}

#[tokio::test]
async fn capability_rejection_happens_before_scoring() {
    let request = media_request("media");
    let (_, unsupported) = engine_and_target("unsupported-cheap");
    let (_, supported) = engine_and_target("supported-expensive");
    let input = planning_input(
        request,
        vec![
            candidate(
                &unsupported,
                capabilities(InputKind::TokenSequence, false),
                0,
                0,
                vec![],
            ),
            candidate(
                &supported,
                capabilities(InputKind::MediaReference, false),
                100,
                100,
                vec![],
            ),
        ],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert_eq!(plan.target.id, supported.id);
}

#[tokio::test]
async fn complete_paths_compare_local_shorter_and_remote_longer_state() {
    let request = token_request("local-vs-remote");
    let provider = ProviderId::new("fake-provider");
    let (_, local_target) = engine_and_target("local");
    let (_, remote_target) = engine_and_target("remote");
    let local_shorter = state_path(
        &local_target,
        &provider,
        StateFixtureSpec {
            id: "local-shorter",
            coverage: 600,
            resume: 600,
            transfer_micros: 0,
            unmatched_prefill_micros: 100,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let remote_longer = state_path(
        &remote_target,
        &provider,
        StateFixtureSpec {
            id: "remote-longer",
            coverage: 900,
            resume: 900,
            transfer_micros: 200,
            unmatched_prefill_micros: 10,
            topology_micros: 25,
            compatible: true,
            locality: StateLocality::Remote {
                topology_path: "remote".to_owned(),
            },
        },
    );
    let input = planning_input(
        request,
        vec![
            candidate(
                &local_target,
                capabilities(InputKind::TokenSequence, true),
                0,
                1_000,
                vec![local_shorter],
            ),
            candidate(
                &remote_target,
                capabilities(InputKind::TokenSequence, true),
                0,
                1_000,
                vec![remote_longer],
            ),
        ],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert_reuse_on(&plan, &local_target, "local-shorter");
}

#[tokio::test]
async fn remote_longer_state_wins_when_total_transfer_path_is_cheaper() {
    let request = token_request("remote-wins");
    let provider = ProviderId::new("fake-provider");
    let (_, local_target) = engine_and_target("local-costly");
    let (_, remote_target) = engine_and_target("remote-cheap");
    let local = state_path(
        &local_target,
        &provider,
        StateFixtureSpec {
            id: "local",
            coverage: 600,
            resume: 600,
            transfer_micros: 0,
            unmatched_prefill_micros: 150,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let remote = state_path(
        &remote_target,
        &provider,
        StateFixtureSpec {
            id: "remote",
            coverage: 900,
            resume: 900,
            transfer_micros: 20,
            unmatched_prefill_micros: 10,
            topology_micros: 10,
            compatible: true,
            locality: StateLocality::Remote {
                topology_path: "fast-link".to_owned(),
            },
        },
    );
    let input = planning_input(
        request,
        vec![
            candidate(
                &local_target,
                capabilities(InputKind::TokenSequence, true),
                0,
                1_000,
                vec![local],
            ),
            candidate(
                &remote_target,
                capabilities(InputKind::TokenSequence, true),
                0,
                1_000,
                vec![remote],
            ),
        ],
    );

    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert_reuse_on(&plan, &remote_target, "remote");
}

#[tokio::test]
async fn stale_engine_generation_invalidates_import_handle() {
    let request = token_request("generation-fence");
    let provider = ProviderId::new("fake-provider");
    let (instance, target) = engine_and_target("restart");
    let engine = FakeEngineAdapter::new(
        instance,
        target.clone(),
        capabilities(InputKind::TokenSequence, true),
    );
    let reusable = state_path(
        &target,
        &provider,
        StateFixtureSpec {
            id: "state-before-restart",
            coverage: 100,
            resume: 100,
            transfer_micros: 0,
            unmatched_prefill_micros: 0,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let context = OperationContext::new(request.id);
    let spec = StateImportSpec::from_plan(&reusable.state, reusable.compatibility);
    let import = engine
        .prepare_state_import(&target, &spec, &context)
        .await
        .expect("prepare import");
    engine.restart();
    let receipt = TransferReceipt {
        import_id: import.import_id.clone(),
        provider,
        bytes_transferred: 1,
        receipt: OpaqueHandle {
            namespace: "test".to_owned(),
            value: "receipt".to_owned(),
        },
    };

    let error = engine
        .commit_state_import(&import, &receipt, &context)
        .await
        .expect_err("stale generation must fail");
    assert!(matches!(error, EngineError::StaleGeneration));
}

#[tokio::test]
async fn materialization_failure_aborts_import_and_uses_encoded_cold_fallback() {
    let request = token_request("materialization-fallback");
    let provider_id = ProviderId::new("fake-provider");
    let (instance, target) = engine_and_target("fallback");
    let engine = Arc::new(FakeEngineAdapter::new(
        instance,
        target.clone(),
        capabilities(InputKind::TokenSequence, true),
    ));
    let reusable = state_path(
        &target,
        &provider_id,
        StateFixtureSpec {
            id: "state-fails",
            coverage: 900,
            resume: 900,
            transfer_micros: 1,
            unmatched_prefill_micros: 0,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let provider = Arc::new(FakeStateProvider::new(
        provider_id,
        vec![reusable.state.clone()],
        vec![reusable.option.clone()],
    ));
    provider.set_fail_materialization(true);
    let input = planning_input(
        request.clone(),
        vec![candidate(
            &target,
            capabilities(InputKind::TokenSequence, true),
            0,
            100,
            vec![reusable],
        )],
    );
    let plan = CostBasedPlanner.plan(&input).await.expect("plan");
    assert_eq!(plan.fallback, FallbackAction::ColdOnSameTarget);

    let registry = EngineRegistry::new();
    registry.register(engine.clone()).expect("register engine");
    let executor = DefaultPlanExecutor::new(registry, provider.clone());
    let events = executor
        .execute(
            plan,
            request.clone(),
            OperationContext::new(request.id.clone()),
        )
        .await
        .expect("cold fallback")
        .try_collect::<Vec<_>>()
        .await
        .expect("event stream");

    assert_eq!(events.len(), 3);
    assert_eq!(engine.call_counts().prepare, 1);
    assert_eq!(engine.call_counts().commit, 0);
    assert_eq!(engine.call_counts().abort, 1);
    assert_eq!(engine.call_counts().execute, 1);
    assert_eq!(provider.call_counts().materialize, 1);
}

#[tokio::test]
async fn observed_provider_outage_degrades_to_cold_when_policy_allows() {
    let request = token_request("provider-outage");
    let provider_id = ProviderId::new("outage-provider");
    let (instance, target) = engine_and_target("outage");
    let engine = Arc::new(FakeEngineAdapter::new(
        instance,
        target.clone(),
        capabilities(InputKind::TokenSequence, true),
    ));
    let reusable = state_path(
        &target,
        &provider_id,
        StateFixtureSpec {
            id: "unavailable-state",
            coverage: 900,
            resume: 900,
            transfer_micros: 0,
            unmatched_prefill_micros: 0,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let provider = Arc::new(FakeStateProvider::new(
        provider_id.clone(),
        vec![reusable.state.clone()],
        vec![reusable.option.clone()],
    ));
    provider.set_unavailable(true);
    let mut input = planning_input(
        request.clone(),
        vec![candidate(
            &target,
            capabilities(InputKind::TokenSequence, true),
            0,
            100,
            vec![reusable],
        )],
    );
    input.state_observation.unavailable_providers =
        BTreeMap::from([(provider_id, "health check failed".to_owned())]);

    let plan = CostBasedPlanner.plan(&input).await.expect("cold plan");
    assert!(matches!(plan.path, ExecutionPath::Cold));
    let registry = EngineRegistry::new();
    registry.register(engine.clone()).expect("register engine");
    let executor = DefaultPlanExecutor::new(registry, provider.clone());
    let events = executor
        .execute(
            plan,
            request.clone(),
            OperationContext::new(request.id.clone()),
        )
        .await
        .expect("execute cold")
        .try_collect::<Vec<_>>()
        .await
        .expect("event stream");

    assert_eq!(events.len(), 3);
    assert_eq!(provider.call_counts().materialize, 0);
    assert_eq!(engine.call_counts().prepare, 0);
    assert_eq!(engine.call_counts().execute, 1);
}

#[tokio::test]
async fn planner_performs_no_state_provider_mutation() {
    let request = token_request("pure-planner");
    let provider_id = ProviderId::new("fake-provider");
    let (_, target) = engine_and_target("pure");
    let reusable = state_path(
        &target,
        &provider_id,
        StateFixtureSpec {
            id: "pure-state",
            coverage: 500,
            resume: 500,
            transfer_micros: 0,
            unmatched_prefill_micros: 1,
            topology_micros: 0,
            compatible: true,
            locality: StateLocality::Local,
        },
    );
    let provider = FakeStateProvider::new(
        provider_id,
        vec![reusable.state.clone()],
        vec![reusable.option.clone()],
    );
    let input = planning_input(
        request,
        vec![candidate(
            &target,
            capabilities(InputKind::TokenSequence, true),
            0,
            100,
            vec![reusable],
        )],
    );

    CostBasedPlanner.plan(&input).await.expect("plan");
    assert_eq!(provider.call_counts(), Default::default());
}

#[test]
fn recurrent_coverage_can_exceed_discrete_resume_checkpoint() {
    let boundary = ReusableBoundary {
        covered_components: vec![ComponentCoverage {
            item_id: InputItemId::new("recurrent-input"),
            covered_units: 100,
        }],
        resume_coordinate: ResumeCoordinate::Checkpoint {
            namespace: "recurrent-step".to_owned(),
            step: 64,
        },
        completeness: BoundaryCompleteness::Checkpointed,
        validation_digest: "checkpoint-64".to_owned(),
    };

    assert_eq!(boundary.covered_components[0].covered_units, 100);
    assert!(matches!(
        boundary.resume_coordinate,
        ResumeCoordinate::Checkpoint { step: 64, .. }
    ));
}

#[test]
fn core_source_has_no_backend_or_framework_domain_type_leaks() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let forbidden = [
        "sglang", "vllm", "tensorrt", "nexuskv", "axum::", "tonic::", "pyo3::",
    ];

    for crate_name in [
        "locus-core",
        "locus-semantics",
        "locus-engine",
        "locus-state",
        "locus-planner",
    ] {
        assert_no_forbidden_source(
            &workspace.join("crates").join(crate_name).join("src"),
            &forbidden,
        );
    }
}

fn assert_no_forbidden_source(directory: &Path, forbidden: &[&str]) {
    for entry in fs::read_dir(directory).expect("read source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            assert_no_forbidden_source(&path, forbidden);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let source = fs::read_to_string(&path).expect("read Rust source");
            let lowercase = source.to_ascii_lowercase();
            for term in forbidden {
                assert!(
                    !lowercase.contains(term),
                    "{} leaked forbidden term {term}",
                    path.display()
                );
            }
        }
    }
}
