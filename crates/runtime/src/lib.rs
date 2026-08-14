use std::collections::BTreeSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::{Stream, StreamExt, stream::BoxStream};
use locus_core::{
    CanonicalRequest, CompatibilityResult, CompatibilityVerdict, EngineCapabilities,
    EngineSnapshot, ExecutionTarget, InputItemValue, OperationContext, RequestId, StateDescriptor,
    StateRequirement,
};
use locus_engine::{EngineAdapter, EngineError, EngineRegistry};
use locus_planner::{
    CostBasedPlanner, DefaultPlanExecutor, ExecutionEstimate, PlanExecutionError, PlanExecutor,
    Planner, PlanningCandidate, PlanningError, PlanningInput, RoutingPolicy, StateObservation,
    StatePathCandidate,
};
use locus_semantics::{
    ModelCatalog, ModelProfile, ModelRegistry, SemanticError, SemanticEvent, SemanticRequest,
};
use locus_state::{StateError, StateProvider};
use thiserror::Error;

pub type SemanticEventStream = BoxStream<'static, Result<SemanticEvent, InferenceError>>;

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
        let plan = self.planner.plan(&planning_input).await?;
        let adapter = self.engines.adapter_for(&plan.target)?;
        let engine_stream = self
            .executor
            .execute(plan, normalized.canonical, context)
            .await?;
        let stream = async_stream::try_stream! {
            let mut engine_stream = engine_stream;
            while let Some(event) = engine_stream.next().await {
                for event in pipeline.process(event?)? {
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
        Ok(ReadinessReport {
            model_profiles: inventory.profiles.len(),
            routable_models,
            required_models: self.required_models.len(),
            ready_targets: inventory.ready_targets,
            observed_targets: inventory.observed_targets,
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
    #[error("target discovery failed: {0}")]
    Discovery(String),
}
