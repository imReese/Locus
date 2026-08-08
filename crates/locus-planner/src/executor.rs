use std::sync::Arc;

use async_trait::async_trait;
use locus_core::{CanonicalRequest, OperationContext, StateImportSpec, StateImportTarget};
use locus_engine::{EngineError, EngineEventStream, EngineRegistry};
use locus_state::{StateError, StateProvider};
use thiserror::Error;

use crate::{ExecutionPath, FallbackAction, PlacementPlan};

#[async_trait]
pub trait PlanExecutor: Send + Sync {
    async fn execute(
        &self,
        plan: PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
    ) -> Result<EngineEventStream, PlanExecutionError>;
}

pub struct DefaultPlanExecutor {
    engines: EngineRegistry,
    state_provider: Arc<dyn StateProvider>,
}

impl DefaultPlanExecutor {
    #[must_use]
    pub fn new(engines: EngineRegistry, state_provider: Arc<dyn StateProvider>) -> Self {
        Self {
            engines,
            state_provider,
        }
    }

    async fn apply_fallback(
        &self,
        fallback: FallbackAction,
        plan: &PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
        cause: String,
    ) -> Result<EngineEventStream, PlanExecutionError> {
        match fallback {
            FallbackAction::ColdOnSameTarget => {
                let adapter = self.engines.adapter_for(&plan.target)?;
                adapter
                    .execute(&plan.target, request, None, context)
                    .await
                    .map_err(PlanExecutionError::Engine)
            }
            FallbackAction::Replan => Err(PlanExecutionError::ReplanRequired(cause)),
            FallbackAction::Fail => Err(PlanExecutionError::PlannedPathFailed(cause)),
        }
    }

    async fn abort_then_fallback(
        &self,
        import: &StateImportTarget,
        plan: &PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
        cause: String,
    ) -> Result<EngineEventStream, PlanExecutionError> {
        let adapter = self.engines.adapter_for(&plan.target)?;
        if let Err(cleanup) = adapter.abort_state_import(import, &context).await {
            return Err(PlanExecutionError::CleanupFailed { cause, cleanup });
        }
        self.apply_fallback(plan.fallback, plan, request, context, cause)
            .await
    }
}

#[async_trait]
impl PlanExecutor for DefaultPlanExecutor {
    async fn execute(
        &self,
        plan: PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
    ) -> Result<EngineEventStream, PlanExecutionError> {
        context.ensure_active()?;
        if request.model != plan.target.model {
            return Err(PlanExecutionError::InvalidPlan(
                "request model does not match the selected execution target".to_owned(),
            ));
        }

        let adapter = self.engines.adapter_for(&plan.target)?;
        let ExecutionPath::Reuse(reuse) = &plan.path else {
            return adapter
                .execute(&plan.target, request, None, context)
                .await
                .map_err(PlanExecutionError::Engine);
        };

        let spec = StateImportSpec::from_plan(&reuse.state, reuse.compatibility.clone());
        let import = match adapter
            .prepare_state_import(&plan.target, &spec, &context)
            .await
        {
            Ok(import) => import,
            Err(error) => {
                return self
                    .apply_fallback(plan.fallback, &plan, request, context, error.to_string())
                    .await;
            }
        };

        let receipt = match self
            .state_provider
            .materialize(&reuse.option, &import, &context)
            .await
        {
            Ok(receipt) => receipt,
            Err(error) => {
                return self
                    .abort_then_fallback(&import, &plan, request, context, error.to_string())
                    .await;
            }
        };

        let attachment = match adapter
            .commit_state_import(&import, &receipt, &context)
            .await
        {
            Ok(attachment) => attachment,
            Err(error) => {
                return self
                    .abort_then_fallback(&import, &plan, request, context, error.to_string())
                    .await;
            }
        };

        adapter
            .execute(&plan.target, request, Some(attachment), context)
            .await
            .map_err(PlanExecutionError::Engine)
    }
}

#[derive(Debug, Error)]
pub enum PlanExecutionError {
    #[error(transparent)]
    Context(#[from] locus_core::ContextError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error("invalid placement plan: {0}")]
    InvalidPlan(String),
    #[error("planned path failed: {0}")]
    PlannedPathFailed(String),
    #[error("planned path failed and requires replanning: {0}")]
    ReplanRequired(String),
    #[error("planned path failed ({cause}); state import cleanup also failed: {cleanup}")]
    CleanupFailed { cause: String, cleanup: EngineError },
}
