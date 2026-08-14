mod calibration;
mod executor;

use std::collections::BTreeMap;

use async_trait::async_trait;
use locus_core::{
    CanonicalRequest, CompatibilityResult, ExecutionTarget, MaterializationOption, ProviderId,
    StateDescriptor,
};
use thiserror::Error;

pub use calibration::{
    ACTIVE_CONFIRMATION, CalibrationApplication, CalibrationError, CalibrationKey,
    CalibrationObservation, CalibrationPolicy, CalibrationStatus, CandidateCalibrationEvidence,
    MaterializationObservation, PersistentCalibrator, PlacementMode, PromotionStatus,
    plan_decision_fingerprint, plan_fingerprint,
};
pub use executor::{
    DefaultPlanExecutor, ExecutedPath, MaterializationTiming, PlanExecution, PlanExecutionError,
    PlanExecutionMetadata, PlanExecutor,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CostBreakdown {
    pub queue_micros: u64,
    pub unmatched_prefill_micros: u64,
    pub state_materialization_micros: u64,
    pub decode_micros: u64,
    pub topology_micros: u64,
    pub policy_micros: u64,
}

impl CostBreakdown {
    #[must_use]
    pub fn total_micros(&self) -> u64 {
        self.queue_micros
            .saturating_add(self.unmatched_prefill_micros)
            .saturating_add(self.state_materialization_micros)
            .saturating_add(self.decode_micros)
            .saturating_add(self.topology_micros)
            .saturating_add(self.policy_micros)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionEstimate {
    pub unmatched_prefill_micros: u64,
    pub decode_micros: u64,
    pub topology_micros: u64,
    pub policy_micros: u64,
}

#[derive(Clone, Debug)]
pub struct StatePathCandidate {
    pub state: StateDescriptor,
    pub compatibility: CompatibilityResult,
    pub option: MaterializationOption,
    /// Planner-only override. The provider estimate in `option` remains immutable so
    /// observed transfer time is always calibrated against the provider's raw baseline.
    pub materialization_estimate_micros: Option<u64>,
    pub estimate: ExecutionEstimate,
}

#[derive(Clone, Debug)]
pub struct PlanningCandidate {
    pub target: ExecutionTarget,
    pub capabilities: locus_core::EngineCapabilities,
    pub snapshot: locus_core::EngineSnapshot,
    pub cold_estimate: ExecutionEstimate,
    pub state_paths: Vec<StatePathCandidate>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateObservation {
    pub unavailable_providers: BTreeMap<ProviderId, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackAction {
    ColdOnSameTarget,
    Replan,
    Fail,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingPolicy {
    pub allow_cold_on_provider_outage: bool,
    pub import_failure_fallback: FallbackAction,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            allow_cold_on_provider_outage: true,
            import_failure_fallback: FallbackAction::ColdOnSameTarget,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlanningInput {
    pub request: CanonicalRequest,
    pub candidates: Vec<PlanningCandidate>,
    pub state_observation: StateObservation,
    pub policy: RoutingPolicy,
}

#[derive(Clone, Debug)]
pub enum ExecutionPath {
    Cold,
    Reuse(Box<StateReusePlan>),
}

#[derive(Clone, Debug)]
pub struct StateReusePlan {
    pub state: StateDescriptor,
    pub compatibility: CompatibilityResult,
    pub option: MaterializationOption,
}

#[derive(Clone, Debug)]
pub struct PlacementPlan {
    pub target: ExecutionTarget,
    pub path: ExecutionPath,
    pub predicted_cost: CostBreakdown,
    pub fallback: FallbackAction,
    pub rationale: Vec<String>,
}

#[async_trait]
pub trait Planner: Send + Sync {
    async fn plan(&self, input: &PlanningInput) -> Result<PlacementPlan, PlanningError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CostBasedPlanner;

#[derive(Clone, Debug)]
struct FeasiblePath {
    target: ExecutionTarget,
    path: ExecutionPath,
    cost: CostBreakdown,
    fallback: FallbackAction,
    stable_key: String,
    rationale: Vec<String>,
}

impl FeasiblePath {
    fn is_better_than(&self, other: &Self) -> bool {
        (self.cost.total_micros(), &self.stable_key)
            < (other.cost.total_micros(), &other.stable_key)
    }
}

#[async_trait]
impl Planner for CostBasedPlanner {
    async fn plan(&self, input: &PlanningInput) -> Result<PlacementPlan, PlanningError> {
        let mut best: Option<FeasiblePath> = None;
        let mut unavailable_path_seen = false;

        for candidate in &input.candidates {
            if candidate.target.model != input.request.model
                || candidate.snapshot.target_id != candidate.target.id
                || !candidate.snapshot.ready
                || !candidate
                    .capabilities
                    .satisfies(&input.request.requirements)
            {
                continue;
            }

            let queue_micros = candidate
                .snapshot
                .estimated_queue_micros
                .unwrap_or(u64::MAX);
            let cold_cost = cost_for(queue_micros, &candidate.cold_estimate, 0);
            consider(
                &mut best,
                FeasiblePath {
                    target: candidate.target.clone(),
                    path: ExecutionPath::Cold,
                    cost: cold_cost,
                    fallback: FallbackAction::Fail,
                    stable_key: format!("{}:0:cold", candidate.target.id),
                    rationale: vec!["feasible cold execution path".to_owned()],
                },
            );

            for state_path in &candidate.state_paths {
                if input
                    .state_observation
                    .unavailable_providers
                    .contains_key(&state_path.option.provider)
                {
                    unavailable_path_seen = true;
                    continue;
                }
                if !state_path.compatibility.is_compatible()
                    || state_path.state.provider != state_path.option.provider
                    || state_path.state.id != state_path.option.source_state
                    || state_path.state.kind != state_path.option.state_kind
                    || state_path.state.model != input.request.model
                    || state_path.option.target_id != candidate.target.id
                    || state_path.option.target_engine != candidate.target.engine
                    || !candidate
                        .capabilities
                        .supported_state_kinds
                        .contains(&state_path.state.kind)
                    || state_path
                        .state
                        .relevant_input_semantics
                        .as_ref()
                        .is_some_and(|identity| identity != &input.request.semantic_identity.input)
                {
                    continue;
                }

                let cost = cost_for(
                    queue_micros,
                    &state_path.estimate,
                    state_path
                        .materialization_estimate_micros
                        .unwrap_or(state_path.option.estimated_transfer_micros),
                );
                consider(
                    &mut best,
                    FeasiblePath {
                        target: candidate.target.clone(),
                        path: ExecutionPath::Reuse(Box::new(StateReusePlan {
                            state: state_path.state.clone(),
                            compatibility: state_path.compatibility.clone(),
                            option: state_path.option.clone(),
                        })),
                        cost,
                        fallback: input.policy.import_failure_fallback,
                        stable_key: format!(
                            "{}:1:{}:{}",
                            candidate.target.id, state_path.state.id, state_path.option.id
                        ),
                        rationale: vec![
                            "compatible state path passed correctness and capability filters"
                                .to_owned(),
                        ],
                    },
                );
            }
        }

        let selected = best.ok_or(PlanningError::NoFeasibleCandidate)?;
        if unavailable_path_seen
            && !input.policy.allow_cold_on_provider_outage
            && matches!(&selected.path, ExecutionPath::Cold)
        {
            return Err(PlanningError::StateProviderUnavailable);
        }
        Ok(PlacementPlan {
            target: selected.target,
            path: selected.path,
            predicted_cost: selected.cost,
            fallback: selected.fallback,
            rationale: selected.rationale,
        })
    }
}

fn cost_for(
    queue_micros: u64,
    estimate: &ExecutionEstimate,
    state_materialization_micros: u64,
) -> CostBreakdown {
    CostBreakdown {
        queue_micros,
        unmatched_prefill_micros: estimate.unmatched_prefill_micros,
        state_materialization_micros,
        decode_micros: estimate.decode_micros,
        topology_micros: estimate.topology_micros,
        policy_micros: estimate.policy_micros,
    }
}

fn consider(best: &mut Option<FeasiblePath>, candidate: FeasiblePath) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.is_better_than(current))
    {
        *best = Some(candidate);
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PlanningError {
    #[error("no feasible execution target satisfies the request")]
    NoFeasibleCandidate,
    #[error("a required state provider is unavailable and cold degradation is disabled")]
    StateProviderUnavailable,
}
