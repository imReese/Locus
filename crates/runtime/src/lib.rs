use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::{Stream, StreamExt, stream::BoxStream};
use locus_core::{
    CanonicalRequest, CompatibilityResult, CompatibilityVerdict, EngineCapabilities, EngineEvent,
    EngineFinishReason, EngineSnapshot, ExecutionTarget, InputItemValue, OperationContext,
    RequestId, StateDescriptor, StateRequirement,
};
use locus_engine::{EngineAdapter, EngineError, EngineRegistry};
use locus_planner::{
    ACTIVE_CONFIRMATION, CalibrationError, CalibrationKey, CalibrationObservation,
    CalibrationPolicy, CostBasedPlanner, CostBreakdown, DefaultPlanExecutor, ExecutionEstimate,
    MaterializationObservation, PersistentCalibrator, PlacementMode, PlacementPlan,
    PlanExecutionError, PlanExecutor, Planner, PlanningCandidate, PlanningError, PlanningInput,
    RoutingPolicy, StateObservation, StatePathCandidate, plan_decision_fingerprint,
    plan_fingerprint,
};
use locus_semantics::{
    ModelCatalog, ModelProfile, ModelRegistry, SemanticError, SemanticEvent, SemanticRequest,
};
use locus_state::{StateError, StateProvider};
use thiserror::Error;

pub type SemanticEventStream = BoxStream<'static, Result<SemanticEvent, InferenceError>>;

const MAX_PLACEMENT_AUDIT_PATHS: usize = 256;

#[derive(Clone, Debug)]
pub struct DiscoveredTarget {
    pub target: ExecutionTarget,
    pub capabilities: EngineCapabilities,
    pub snapshot: EngineSnapshot,
}

#[async_trait]
pub trait TargetDiscovery: Send + Sync {
    async fn discover(
        &self,
        request: &CanonicalRequest,
        context: &OperationContext,
    ) -> Result<Vec<DiscoveredTarget>, InferenceError>;
}

#[derive(Clone)]
pub struct EngineTargetDiscovery {
    engines: EngineRegistry,
}

impl EngineTargetDiscovery {
    #[must_use]
    pub fn new(engines: EngineRegistry) -> Self {
        Self { engines }
    }
}

#[async_trait]
impl TargetDiscovery for EngineTargetDiscovery {
    async fn discover(
        &self,
        request: &CanonicalRequest,
        context: &OperationContext,
    ) -> Result<Vec<DiscoveredTarget>, InferenceError> {
        let mut discovered = Vec::new();
        let mut failures = Vec::new();
        for adapter in self.engines.adapters()? {
            let targets = match adapter.execution_targets(context).await {
                Ok(targets) => targets,
                Err(error) => {
                    failures.push(error.to_string());
                    continue;
                }
            };
            for target in targets {
                if target.model != request.model {
                    continue;
                }
                let capabilities = match adapter.capabilities(&target, context).await {
                    Ok(capabilities) => capabilities,
                    Err(error) => {
                        failures.push(error.to_string());
                        continue;
                    }
                };
                let snapshot = match adapter.snapshot(&target, context).await {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        failures.push(error.to_string());
                        continue;
                    }
                };
                discovered.push(DiscoveredTarget {
                    target,
                    capabilities,
                    snapshot,
                });
            }
        }
        discovered.sort_by(|left, right| left.target.id.cmp(&right.target.id));
        if discovered.is_empty() && !failures.is_empty() {
            return Err(InferenceError::Discovery(failures.join("; ")));
        }
        Ok(discovered)
    }
}

#[async_trait]
pub trait InferenceService: Send + Sync {
    async fn infer(
        &self,
        request: SemanticRequest,
        context: OperationContext,
    ) -> Result<SemanticEventStream, InferenceError>;

    async fn models(&self) -> Result<Vec<ModelProfile>, InferenceError>;

    async fn readiness(&self) -> Result<ReadinessReport, InferenceError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadinessReport {
    pub model_profiles: usize,
    pub routable_models: usize,
    pub required_models: usize,
    pub ready_targets: usize,
    pub observed_targets: usize,
    pub placement_mode: PlacementMode,
    pub calibration_revision: u64,
    pub calibration_persistent: bool,
    pub calibration_persistence_healthy: bool,
}

#[derive(Clone)]
pub struct PlacementControl {
    mode: PlacementMode,
    calibrator: PersistentCalibrator,
}

impl PlacementControl {
    pub fn new(
        mode: PlacementMode,
        calibrator: PersistentCalibrator,
        active_confirmation: Option<&str>,
    ) -> Result<Self, PlacementConfigurationError> {
        if mode == PlacementMode::Active {
            if active_confirmation != Some(ACTIVE_CONFIRMATION) {
                return Err(PlacementConfigurationError::ActiveConfirmation);
            }
            if !calibrator.is_persistent() {
                return Err(PlacementConfigurationError::PersistenceRequired);
            }
        }
        Ok(Self { mode, calibrator })
    }

    #[must_use]
    pub fn shadow(calibrator: PersistentCalibrator) -> Self {
        Self {
            mode: PlacementMode::Shadow,
            calibrator,
        }
    }

    #[must_use]
    pub fn mode(&self) -> PlacementMode {
        self.mode
    }
}

pub struct DefaultInferenceService {
    catalog: Arc<dyn ModelCatalog>,
    discovery: Arc<dyn TargetDiscovery>,
    planner: Arc<dyn Planner>,
    executor: Arc<dyn PlanExecutor>,
    engines: EngineRegistry,
    state_provider: Arc<dyn StateProvider>,
    policy: RoutingPolicy,
    required_models: BTreeSet<String>,
    placement: PlacementControl,
}

struct TargetInventory {
    profiles: Vec<ModelProfile>,
    ready_models: BTreeSet<locus_core::ModelExecutionIdentity>,
    observed_targets: usize,
    ready_targets: usize,
    failures: Vec<String>,
}

impl DefaultInferenceService {
    #[must_use]
    pub fn new(
        models: ModelRegistry,
        engines: EngineRegistry,
        state_provider: Arc<dyn StateProvider>,
    ) -> Self {
        let discovery = Arc::new(EngineTargetDiscovery::new(engines.clone()));
        let executor = Arc::new(DefaultPlanExecutor::new(
            engines.clone(),
            Arc::clone(&state_provider),
        ));
        Self {
            catalog: Arc::new(models),
            discovery,
            planner: Arc::new(CostBasedPlanner),
            executor,
            engines,
            state_provider,
            policy: RoutingPolicy::default(),
            required_models: BTreeSet::new(),
            placement: PlacementControl::shadow(default_calibrator()),
        }
    }

    #[must_use]
    pub fn with_components(
        models: ModelRegistry,
        engines: EngineRegistry,
        state_provider: Arc<dyn StateProvider>,
        discovery: Arc<dyn TargetDiscovery>,
        planner: Arc<dyn Planner>,
        executor: Arc<dyn PlanExecutor>,
        policy: RoutingPolicy,
    ) -> Self {
        Self {
            catalog: Arc::new(models),
            discovery,
            planner,
            executor,
            engines,
            state_provider,
            policy,
            required_models: BTreeSet::new(),
            placement: PlacementControl::shadow(default_calibrator()),
        }
    }

    #[must_use]
    pub fn with_required_models(
        mut self,
        required_models: impl IntoIterator<Item = String>,
    ) -> Self {
        self.required_models = required_models.into_iter().collect();
        self
    }

    #[must_use]
    pub fn with_placement_control(mut self, placement: PlacementControl) -> Self {
        self.placement = placement;
        self
    }

    async fn target_inventory(&self) -> Result<TargetInventory, InferenceError> {
        let profiles = self.catalog.profiles()?;
        let catalog_models = profiles
            .iter()
            .map(|profile| profile.model.clone())
            .collect::<BTreeSet<_>>();
        let context = OperationContext::new(RequestId::new("locus-model-inventory"));
        let mut ready_models = BTreeSet::new();
        let mut observed_targets = 0_usize;
        let mut ready_targets = 0_usize;
        let mut failures = Vec::new();
        for adapter in self.engines.adapters()? {
            let targets = match adapter.execution_targets(&context).await {
                Ok(targets) => targets,
                Err(error) => {
                    failures.push(error.to_string());
                    continue;
                }
            };
            for target in targets {
                observed_targets += 1;
                if !catalog_models.contains(&target.model) {
                    failures.push(format!(
                        "target {} has no matching semantic profile",
                        target.id
                    ));
                    continue;
                }
                match adapter.snapshot(&target, &context).await {
                    Ok(snapshot) if snapshot.ready => {
                        ready_targets += 1;
                        ready_models.insert(target.model);
                    }
                    Ok(_) => failures.push(format!("target {} is not ready", target.id)),
                    Err(error) => failures.push(error.to_string()),
                }
            }
        }
        Ok(TargetInventory {
            profiles,
            ready_models,
            observed_targets,
            ready_targets,
            failures,
        })
    }

    async fn planning_input(
        &self,
        request: CanonicalRequest,
        context: &OperationContext,
    ) -> Result<PlanningInput, InferenceError> {
        let discovered = self.discovery.discover(&request, context).await?;
        let accepted_state_kinds = discovered
            .iter()
            .flat_map(|candidate| candidate.capabilities.supported_state_kinds.iter().cloned())
            .collect::<BTreeSet<_>>();
        let mut state_observation = StateObservation::default();
        let states = if accepted_state_kinds.is_empty() {
            Vec::new()
        } else {
            let requirement =
                StateRequirement {
                    model: request.model.clone(),
                    input_semantics: request.semantic_identity.input.clone(),
                    accepted_state_kinds,
                    input_fingerprint: input_fingerprint(&request),
                    query_token_ids: request.input.items.iter().find_map(|item| {
                        match &item.value {
                            InputItemValue::TokenSequence(tokens) => Some(tokens.token_ids.clone()),
                            _ => None,
                        }
                    }),
                    tenant_scope: None,
                };
            match self.state_provider.lookup(&requirement, context).await {
                Ok(states) => states,
                Err(StateError::Unavailable(reason)) => {
                    state_observation
                        .unavailable_providers
                        .insert(self.state_provider.identity().clone(), reason);
                    Vec::new()
                }
                Err(error) => return Err(error.into()),
            }
        };

        let mut candidates = Vec::new();
        for discovered in discovered {
            let mut state_paths = Vec::new();
            for state in &states {
                if !discovered
                    .capabilities
                    .supported_state_kinds
                    .contains(&state.kind)
                {
                    continue;
                }
                let compatibility = compatibility_for(state, &request);
                if !compatibility.is_compatible() {
                    continue;
                }
                let options = match self
                    .state_provider
                    .estimate(state, &discovered.target, context)
                    .await
                {
                    Ok(options) => options,
                    Err(StateError::Unavailable(reason)) => {
                        state_observation
                            .unavailable_providers
                            .insert(self.state_provider.identity().clone(), reason);
                        Vec::new()
                    }
                    Err(error) => return Err(error.into()),
                };
                for option in options {
                    state_paths.push(StatePathCandidate {
                        state: state.clone(),
                        compatibility: compatibility.clone(),
                        option,
                        materialization_estimate_micros: None,
                        estimate: reuse_estimate(state, &request),
                    });
                }
            }
            candidates.push(PlanningCandidate {
                target: discovered.target,
                capabilities: discovered.capabilities,
                snapshot: discovered.snapshot,
                cold_estimate: cold_estimate(&request),
                state_paths,
            });
        }

        Ok(PlanningInput {
            request,
            candidates,
            state_observation,
            policy: self.policy.clone(),
        })
    }

    async fn select_plan(
        &self,
        input: &PlanningInput,
    ) -> Result<PlacementSelection, InferenceError> {
        let legacy_plan = self.planner.plan(input).await?;
        let now_unix_millis = unix_millis();
        let application = match self.placement.calibrator.apply(input, now_unix_millis) {
            Ok(application) => application,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "placement calibration failed; using legacy plan"
                );
                return Ok(PlacementSelection::legacy(legacy_plan, "calibration_error"));
            }
        };
        let calibrated_plan = match self.planner.plan(&application.input).await {
            Ok(plan) => plan,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "calibrated planning failed; using legacy plan"
                );
                return Ok(PlacementSelection::legacy(
                    legacy_plan,
                    "calibrated_planning_error",
                ));
            }
        };
        trace_planning_paths(input, "legacy");
        trace_planning_paths(&application.input, "calibrated");
        let replay = self.planner.plan(&application.input).await;
        let replay_consistent = replay
            .as_ref()
            .is_ok_and(|plan| plan_fingerprint(plan) == plan_fingerprint(&calibrated_plan));
        let calibrator = self.placement.calibrator.clone();
        let legacy_decision = plan_decision_fingerprint(&legacy_plan);
        let calibrated_decision = plan_decision_fingerprint(&calibrated_plan);
        match tokio::task::spawn_blocking(move || {
            calibrator.record_shadow_decision(
                replay_consistent,
                &legacy_decision,
                &calibrated_decision,
            )
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                tracing::warn!(
                    error = %error,
                    "failed to persist shadow evidence; using legacy plan"
                );
                return Ok(PlacementSelection::legacy(
                    legacy_plan,
                    "shadow_persistence_error",
                ));
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "shadow evidence task failed; using legacy plan"
                );
                return Ok(PlacementSelection::legacy(legacy_plan, "shadow_task_error"));
            }
        }

        if self.placement.mode == PlacementMode::Shadow {
            return Ok(PlacementSelection {
                plan: legacy_plan,
                source: "shadow_mode",
                calibration_revision: application.revision,
                promotion_reasons: Vec::new(),
            });
        }
        let evidence = application
            .evidence
            .get(calibrated_plan.target.id.as_str())
            .cloned();
        let status_calibrator = self.placement.calibrator.clone();
        let status_plan = calibrated_plan.clone();
        let status_evidence = evidence.clone();
        let status = match tokio::task::spawn_blocking(move || {
            status_calibrator.promotion_status(&status_plan, status_evidence.as_ref())
        })
        .await
        {
            Ok(status) => status,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "active promotion task failed; using legacy plan"
                );
                return Ok(PlacementSelection::legacy(
                    legacy_plan,
                    "promotion_task_error",
                ));
            }
        };
        match status {
            Ok(status) if status.qualified => Ok(PlacementSelection {
                plan: calibrated_plan,
                source: "calibrated_active",
                calibration_revision: status.revision,
                promotion_reasons: Vec::new(),
            }),
            Ok(status) => Ok(PlacementSelection {
                plan: legacy_plan,
                source: "active_gate_fallback",
                calibration_revision: status.revision,
                promotion_reasons: status.reasons,
            }),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "active promotion check failed; using legacy plan"
                );
                Ok(PlacementSelection::legacy(
                    legacy_plan,
                    "promotion_check_error",
                ))
            }
        }
    }
}

struct PlacementSelection {
    plan: PlacementPlan,
    source: &'static str,
    calibration_revision: u64,
    promotion_reasons: Vec<String>,
}

impl PlacementSelection {
    fn legacy(plan: PlacementPlan, source: &'static str) -> Self {
        Self {
            plan,
            source,
            calibration_revision: 0,
            promotion_reasons: Vec::new(),
        }
    }
}

#[async_trait]
impl InferenceService for DefaultInferenceService {
    async fn infer(
        &self,
        request: SemanticRequest,
        context: OperationContext,
    ) -> Result<SemanticEventStream, InferenceError> {
        context.ensure_active()?;
        let semantics = self.catalog.resolve(&request.model)?;
        let normalized = semantics.normalize(&request, context.request_id.clone())?;
        let mut pipeline = semantics.output_pipeline(&normalized.output_contract)?;
        let planning_input = self
            .planning_input(normalized.canonical.clone(), &context)
            .await?;
        let selection = self.select_plan(&planning_input).await?;
        let plan = selection.plan;
        let adapter = self.engines.adapter_for(&plan.target)?;
        let selected_target_id = plan.target.id.to_string();
        let observation_key = CalibrationKey::from_target(&plan.target);
        let waiting_requests = planning_input
            .candidates
            .iter()
            .find(|candidate| candidate.target.id == plan.target.id)
            .and_then(|candidate| {
                candidate
                    .snapshot
                    .telemetry_is_fresh_at(unix_millis())
                    .then_some(candidate.snapshot.waiting_requests)
                    .flatten()
            });
        let selected_snapshot = planning_input
            .candidates
            .iter()
            .find(|candidate| candidate.target.id == plan.target.id)
            .map(|candidate| &candidate.snapshot);
        tracing::info!(
            placement_mode = ?self.placement.mode,
            placement_source = selection.source,
            target_id = %plan.target.id,
            engine_id = %plan.target.engine.id,
            engine_generation = plan.target.engine.generation,
            calibration_revision = selection.calibration_revision,
            promotion_blockers = selection.promotion_reasons.len(),
            promotion_reasons = %selection.promotion_reasons.join(","),
            telemetry_status = ?selected_snapshot.map(|snapshot| snapshot.telemetry_status),
            telemetry_confidence = ?selected_snapshot.map(|snapshot| snapshot.telemetry_confidence),
            telemetry_source = selected_snapshot.map_or("missing", |snapshot| snapshot.telemetry_source.as_str()),
            telemetry_revision = selected_snapshot.map_or(0, |snapshot| snapshot.observation_revision),
            telemetry_valid_until_unix_millis = selected_snapshot.map_or(0, |snapshot| snapshot.valid_until_unix_millis),
            queue_micros = plan.predicted_cost.queue_micros,
            prefill_micros = plan.predicted_cost.unmatched_prefill_micros,
            materialization_micros = plan.predicted_cost.state_materialization_micros,
            decode_micros = plan.predicted_cost.decode_micros,
            topology_micros = plan.predicted_cost.topology_micros,
            policy_micros = plan.predicted_cost.policy_micros,
            "placement decision"
        );
        let execution_started = Instant::now();
        let execution = self
            .executor
            .execute(plan, normalized.canonical, context)
            .await?;
        let execution_metadata = execution.metadata;
        let engine_stream = execution.stream;
        let calibrator = self.placement.calibrator.clone();
        let stream = async_stream::try_stream! {
            let mut engine_stream = engine_stream;
            let mut first_output_at = None;
            while let Some(event) = engine_stream.next().await {
                let event = event?;
                if first_output_at.is_none() && is_output_event(&event) {
                    first_output_at = Some(Instant::now());
                }
                if let EngineEvent::Finished { reason, usage, .. } = &event {
                    let completed = !matches!(reason, EngineFinishReason::Cancelled | EngineFinishReason::Error);
                    let materialization = execution_metadata.materialization.as_ref().map(|timing| {
                        MaterializationObservation {
                            provider: timing.provider.clone(),
                            state_kind: timing.state_kind.clone(),
                            target_id: timing.target_id.clone(),
                            estimated_micros: timing.provider_estimated_micros,
                            actual_micros: timing.actual_micros,
                        }
                    });
                    let time_to_first_token_micros = if execution_metadata.fallback_used {
                        None
                    } else {
                        first_output_at.map(|first| {
                            elapsed_between_micros(execution_started, first)
                                .saturating_sub(
                                    execution_metadata
                                        .materialization
                                        .as_ref()
                                        .map_or(0, |timing| timing.actual_micros),
                                )
                        })
                    };
                    let generation_micros = first_output_at.map(elapsed_micros);
                    let observation = CalibrationObservation {
                        key: observation_key.clone(),
                        waiting_requests,
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens.saturating_sub(1),
                        time_to_first_token_micros,
                        generation_micros,
                        topology_micros: execution_metadata.topology_micros,
                        materialization,
                        completed,
                    };
                    tracing::info!(
                        target_id = %selected_target_id,
                        engine_id = %observation_key.engine_id,
                        engine_generation = observation_key.engine_generation,
                        completed,
                        fallback_used = execution_metadata.fallback_used,
                        executed_path = ?execution_metadata.executed_path,
                        waiting_requests = ?waiting_requests,
                        input_tokens = usage.input_tokens,
                        output_tokens = usage.output_tokens,
                        time_to_first_token_micros = ?time_to_first_token_micros,
                        generation_micros = ?generation_micros,
                        materialization_micros = ?execution_metadata.materialization.as_ref().map(|timing| timing.actual_micros),
                        topology_micros = ?execution_metadata.topology_micros,
                        "placement outcome"
                    );
                    let observation_calibrator = calibrator.clone();
                    match tokio::task::spawn_blocking(move || {
                        observation_calibrator.record_observation(&observation)
                    }).await {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => tracing::warn!(error = %error, "failed to record calibration observation"),
                        Err(error) => tracing::warn!(error = %error, "calibration observation task failed"),
                    }
                }
                for event in pipeline.process(event)? {
                    yield event;
                }
            }
        };
        Ok(CancelOnDropStream::new(Box::pin(stream), planning_input.request.id, adapter).boxed())
    }

    async fn models(&self) -> Result<Vec<ModelProfile>, InferenceError> {
        let inventory = self.target_inventory().await?;
        Ok(inventory
            .profiles
            .into_iter()
            .filter(|profile| inventory.ready_models.contains(&profile.model))
            .collect())
    }

    async fn readiness(&self) -> Result<ReadinessReport, InferenceError> {
        let inventory = self.target_inventory().await?;
        if inventory.profiles.is_empty() {
            return Err(InferenceError::Discovery(
                "no model profiles are registered".to_owned(),
            ));
        }
        if inventory.ready_models.is_empty() {
            let evidence = if inventory.failures.is_empty() {
                "no execution target matched a registered semantic profile".to_owned()
            } else {
                inventory.failures.join("; ")
            };
            return Err(InferenceError::Discovery(format!(
                "no model has a ready execution target; {evidence}"
            )));
        }
        let mut missing = Vec::new();
        for alias in &self.required_models {
            let semantics = self.catalog.resolve(alias).map_err(|error| {
                InferenceError::Discovery(format!(
                    "required model {alias} has no semantic profile: {error}"
                ))
            })?;
            if !inventory.ready_models.contains(&semantics.profile().model) {
                missing.push(alias.clone());
            }
        }
        if !missing.is_empty() {
            let evidence = if inventory.failures.is_empty() {
                "no matching ready execution target was observed".to_owned()
            } else {
                inventory.failures.join("; ")
            };
            return Err(InferenceError::Discovery(format!(
                "required models without a ready target: {}; {evidence}",
                missing.join(", ")
            )));
        }
        let routable_models = inventory
            .profiles
            .iter()
            .filter(|profile| inventory.ready_models.contains(&profile.model))
            .count();
        let calibration = self.placement.calibrator.status()?;
        Ok(ReadinessReport {
            model_profiles: inventory.profiles.len(),
            routable_models,
            required_models: self.required_models.len(),
            ready_targets: inventory.ready_targets,
            observed_targets: inventory.observed_targets,
            placement_mode: self.placement.mode,
            calibration_revision: calibration.revision,
            calibration_persistent: calibration.persistent,
            calibration_persistence_healthy: calibration.persistence_healthy,
        })
    }
}

fn compatibility_for(state: &StateDescriptor, request: &CanonicalRequest) -> CompatibilityResult {
    if state.model != request.model {
        return CompatibilityResult::incompatible("model execution identity mismatch");
    }
    match &state.relevant_input_semantics {
        Some(identity) if identity != &request.semantic_identity.input => {
            CompatibilityResult::incompatible("input semantic identity mismatch")
        }
        Some(_) => CompatibilityResult::compatible("model and input semantics match"),
        None => CompatibilityResult {
            verdict: CompatibilityVerdict::Unknown,
            checked_dimensions: vec!["model_execution".to_owned()],
            evidence: vec!["provider omitted input semantic identity".to_owned()],
            required_conversion: None,
        },
    }
}

fn input_fingerprint(request: &CanonicalRequest) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    mix_fingerprint(&mut hash, request.model.model_revision.as_bytes());
    mix_fingerprint(
        &mut hash,
        request
            .model
            .adapter_revision
            .as_deref()
            .unwrap_or("")
            .as_bytes(),
    );
    mix_fingerprint(&mut hash, request.model.execution_profile.as_bytes());
    mix_fingerprint(
        &mut hash,
        request
            .semantic_identity
            .input
            .tokenizer
            .fingerprint
            .as_bytes(),
    );
    mix_fingerprint(
        &mut hash,
        request
            .semantic_identity
            .input
            .template
            .fingerprint
            .as_bytes(),
    );
    for item in &request.input.items {
        mix_fingerprint(&mut hash, item.id.as_str().as_bytes());
        match &item.value {
            InputItemValue::TokenSequence(tokens) => {
                for token in &tokens.token_ids {
                    mix_fingerprint(&mut hash, &token.to_le_bytes());
                }
            }
            other => mix_fingerprint(&mut hash, format!("{other:?}").as_bytes()),
        }
    }
    format!("locus-input-fnv1a64-v1:{hash:016x}")
}

fn mix_fingerprint(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn input_token_count(request: &CanonicalRequest) -> u64 {
    request
        .input
        .items
        .iter()
        .filter_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens.token_ids.len() as u64),
            _ => None,
        })
        .sum()
}

fn is_output_event(event: &EngineEvent) -> bool {
    matches!(
        event,
        EngineEvent::TokenDelta { .. }
            | EngineEvent::TextDelta { .. }
            | EngineEvent::ReasoningDelta { .. }
            | EngineEvent::ToolCallStarted { .. }
            | EngineEvent::ToolCallArgumentsDelta { .. }
            | EngineEvent::ToolCallCompleted { .. }
    )
}

struct PathAudit {
    target_id: String,
    engine_id: String,
    path_kind: &'static str,
    state_kind: Option<String>,
    provider: Option<String>,
    feasible: bool,
    exclusions: Vec<&'static str>,
    cost: CostBreakdown,
    stable_key: String,
    rank: Option<usize>,
    telemetry_status: locus_core::TelemetryStatus,
    telemetry_confidence: locus_core::TelemetryConfidence,
    telemetry_source: String,
    telemetry_revision: u64,
    telemetry_valid_until_unix_millis: u64,
}

fn trace_planning_paths(input: &PlanningInput, variant: &'static str) {
    let mut audits = Vec::new();
    for candidate in &input.candidates {
        let candidate_exclusions = candidate_exclusions(input, candidate);
        let queue_micros = candidate
            .snapshot
            .estimated_queue_micros
            .unwrap_or(u64::MAX);
        audits.push(PathAudit {
            target_id: candidate.target.id.to_string(),
            engine_id: candidate.target.engine.id.to_string(),
            path_kind: "cold",
            state_kind: None,
            provider: None,
            feasible: candidate_exclusions.is_empty(),
            exclusions: candidate_exclusions.clone(),
            cost: audit_cost(queue_micros, &candidate.cold_estimate, 0),
            stable_key: format!("{}:0:cold", candidate.target.id),
            rank: None,
            telemetry_status: candidate.snapshot.telemetry_status,
            telemetry_confidence: candidate.snapshot.telemetry_confidence,
            telemetry_source: candidate.snapshot.telemetry_source.clone(),
            telemetry_revision: candidate.snapshot.observation_revision,
            telemetry_valid_until_unix_millis: candidate.snapshot.valid_until_unix_millis,
        });
        for state_path in &candidate.state_paths {
            let mut exclusions = candidate_exclusions.clone();
            exclusions.extend(state_path_exclusions(input, candidate, state_path));
            let materialization_micros = state_path
                .materialization_estimate_micros
                .unwrap_or(state_path.option.estimated_transfer_micros);
            audits.push(PathAudit {
                target_id: candidate.target.id.to_string(),
                engine_id: candidate.target.engine.id.to_string(),
                path_kind: "reuse",
                state_kind: Some(state_path.state.kind.as_str().to_owned()),
                provider: Some(state_path.option.provider.as_str().to_owned()),
                feasible: exclusions.is_empty(),
                exclusions,
                cost: audit_cost(queue_micros, &state_path.estimate, materialization_micros),
                stable_key: format!(
                    "{}:1:{}:{}",
                    candidate.target.id, state_path.state.id, state_path.option.id
                ),
                rank: None,
                telemetry_status: candidate.snapshot.telemetry_status,
                telemetry_confidence: candidate.snapshot.telemetry_confidence,
                telemetry_source: candidate.snapshot.telemetry_source.clone(),
                telemetry_revision: candidate.snapshot.observation_revision,
                telemetry_valid_until_unix_millis: candidate.snapshot.valid_until_unix_millis,
            });
        }
    }
    let mut feasible = audits
        .iter()
        .enumerate()
        .filter(|(_, audit)| audit.feasible)
        .map(|(index, audit)| (index, audit.cost.total_micros(), audit.stable_key.clone()))
        .collect::<Vec<_>>();
    feasible.sort_by(|left, right| (left.1, &left.2).cmp(&(right.1, &right.2)));
    for (rank, (index, _, _)) in feasible.into_iter().enumerate() {
        audits[index].rank = Some(rank + 1);
    }
    let total_paths = audits.len();
    for audit in audits.iter().take(MAX_PLACEMENT_AUDIT_PATHS) {
        tracing::debug!(
            decision_variant = variant,
            target_id = %audit.target_id,
            engine_id = %audit.engine_id,
            path_kind = audit.path_kind,
            state_kind = audit.state_kind.as_deref().unwrap_or("none"),
            provider = audit.provider.as_deref().unwrap_or("none"),
            feasible = audit.feasible,
            exclusion_reasons = %audit.exclusions.join(","),
            rank = ?audit.rank,
            queue_micros = audit.cost.queue_micros,
            prefill_micros = audit.cost.unmatched_prefill_micros,
            materialization_micros = audit.cost.state_materialization_micros,
            decode_micros = audit.cost.decode_micros,
            topology_micros = audit.cost.topology_micros,
            policy_micros = audit.cost.policy_micros,
            total_micros = audit.cost.total_micros(),
            telemetry_status = ?audit.telemetry_status,
            telemetry_confidence = ?audit.telemetry_confidence,
            telemetry_source = %audit.telemetry_source,
            telemetry_revision = audit.telemetry_revision,
            telemetry_valid_until_unix_millis = audit.telemetry_valid_until_unix_millis,
            "placement candidate"
        );
    }
    if total_paths > MAX_PLACEMENT_AUDIT_PATHS {
        tracing::warn!(
            decision_variant = variant,
            total_paths,
            logged_paths = MAX_PLACEMENT_AUDIT_PATHS,
            "placement candidate audit was truncated"
        );
    }
}

fn candidate_exclusions(input: &PlanningInput, candidate: &PlanningCandidate) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if candidate.target.model != input.request.model {
        reasons.push("model_identity_mismatch");
    }
    if candidate.snapshot.target_id != candidate.target.id {
        reasons.push("snapshot_target_mismatch");
    }
    if !candidate.snapshot.ready {
        reasons.push("target_not_ready");
    }
    if !candidate
        .capabilities
        .satisfies(&input.request.requirements)
    {
        reasons.push("capability_mismatch");
    }
    reasons
}

fn state_path_exclusions(
    input: &PlanningInput,
    candidate: &PlanningCandidate,
    state_path: &StatePathCandidate,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if input
        .state_observation
        .unavailable_providers
        .contains_key(&state_path.option.provider)
    {
        reasons.push("state_provider_unavailable");
    }
    if !state_path.compatibility.is_compatible() {
        reasons.push("state_incompatible");
    }
    if state_path.state.provider != state_path.option.provider {
        reasons.push("state_provider_mismatch");
    }
    if state_path.state.id != state_path.option.source_state {
        reasons.push("source_state_mismatch");
    }
    if state_path.state.kind != state_path.option.state_kind {
        reasons.push("state_kind_mismatch");
    }
    if state_path.state.model != input.request.model {
        reasons.push("state_model_mismatch");
    }
    if state_path.option.target_id != candidate.target.id {
        reasons.push("state_target_mismatch");
    }
    if state_path.option.target_engine != candidate.target.engine {
        reasons.push("state_engine_generation_mismatch");
    }
    if !candidate
        .capabilities
        .supported_state_kinds
        .contains(&state_path.state.kind)
    {
        reasons.push("state_kind_unsupported");
    }
    if state_path
        .state
        .relevant_input_semantics
        .as_ref()
        .is_some_and(|identity| identity != &input.request.semantic_identity.input)
    {
        reasons.push("state_input_semantics_mismatch");
    }
    reasons
}

fn audit_cost(
    queue_micros: u64,
    estimate: &ExecutionEstimate,
    materialization_micros: u64,
) -> CostBreakdown {
    CostBreakdown {
        queue_micros,
        unmatched_prefill_micros: estimate.unmatched_prefill_micros,
        state_materialization_micros: materialization_micros,
        decode_micros: estimate.decode_micros,
        topology_micros: estimate.topology_micros,
        policy_micros: estimate.policy_micros,
    }
}

fn elapsed_between_micros(started: Instant, finished: Instant) -> u64 {
    u64::try_from(finished.duration_since(started).as_micros()).unwrap_or(u64::MAX)
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
        })
}

fn default_calibrator() -> PersistentCalibrator {
    PersistentCalibrator::load(CalibrationPolicy::default(), None)
        .unwrap_or_else(|_| unreachable!("default calibration policy is valid"))
}

fn cold_estimate(request: &CanonicalRequest) -> ExecutionEstimate {
    ExecutionEstimate {
        unmatched_prefill_micros: input_token_count(request).saturating_mul(10),
        decode_micros: u64::from(request.sampling.max_output_tokens.unwrap_or(16))
            .saturating_mul(20),
        topology_micros: 0,
        policy_micros: 0,
    }
}

fn reuse_estimate(state: &StateDescriptor, request: &CanonicalRequest) -> ExecutionEstimate {
    let covered = state
        .boundary
        .covered_components
        .iter()
        .map(|component| component.covered_units)
        .sum::<u64>();
    let unmatched = input_token_count(request).saturating_sub(covered);
    ExecutionEstimate {
        unmatched_prefill_micros: unmatched.saturating_mul(10),
        ..cold_estimate(request)
    }
}

struct CancelOnDropStream {
    inner: SemanticEventStream,
    request_id: RequestId,
    adapter: Arc<dyn EngineAdapter>,
    completed: bool,
}

impl CancelOnDropStream {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<SemanticEvent, InferenceError>> + Send>>,
        request_id: RequestId,
        adapter: Arc<dyn EngineAdapter>,
    ) -> Self {
        Self {
            inner,
            request_id,
            adapter,
            completed: false,
        }
    }
}

impl Stream for CancelOnDropStream {
    type Item = Result<SemanticEvent, InferenceError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = self.inner.as_mut().poll_next(context);
        if matches!(
            &result,
            Poll::Ready(None) | Poll::Ready(Some(Ok(SemanticEvent::Finished { .. })))
        ) {
            self.completed = true;
        }
        result
    }
}

impl Drop for CancelOnDropStream {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let adapter = Arc::clone(&self.adapter);
        let request_id = self.request_id.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let context = OperationContext::new(request_id.clone());
                let _ = adapter.cancel(&request_id, &context).await;
            });
        }
    }
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error(transparent)]
    Context(#[from] locus_core::ContextError),
    #[error(transparent)]
    Semantic(#[from] SemanticError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    Planning(#[from] PlanningError),
    #[error(transparent)]
    Execution(#[from] PlanExecutionError),
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
    #[error("target discovery failed: {0}")]
    Discovery(String),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlacementConfigurationError {
    #[error("active placement requires confirmation value {ACTIVE_CONFIRMATION}")]
    ActiveConfirmation,
    #[error("active placement requires a persistent calibration state path")]
    PersistenceRequired,
}
