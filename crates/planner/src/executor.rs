use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use async_trait::async_trait;
use locus_core::{CanonicalRequest, OperationContext, StateImportSpec, StateImportTarget};
use locus_engine::{EngineError, EngineEventStream, EngineRegistry};
use locus_store::{StateStore, StoreError};
use thiserror::Error;

use crate::{ExecutionPath, FallbackAction, PlacementPlan};

#[async_trait]
pub trait PlanExecutor: Send + Sync {
    async fn execute(
        &self,
        plan: PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
    ) -> Result<PlanExecution, PlanExecutionError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutedPath {
    Cold,
    Reuse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializationTiming {
    pub store: String,
    pub state_kind: String,
    pub target_id: String,
    pub store_estimated_micros: u64,
    pub actual_micros: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanExecutionMetadata {
    pub executed_path: ExecutedPath,
    pub fallback_used: bool,
    pub materialization: Option<MaterializationTiming>,
    pub topology_micros: Option<u64>,
}

pub struct PlanExecution {
    pub stream: EngineEventStream,
    pub metadata: PlanExecutionMetadata,
}

impl futures::Stream for PlanExecution {
    type Item = Result<locus_core::EngineEvent, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.stream.as_mut().poll_next(context)
    }
}

pub struct DefaultPlanExecutor {
    engines: EngineRegistry,
    store: Arc<dyn StateStore>,
}

impl DefaultPlanExecutor {
    #[must_use]
    pub fn new(engines: EngineRegistry, store: Arc<dyn StateStore>) -> Self {
        Self { engines, store }
    }

    async fn apply_fallback(
        &self,
        fallback: FallbackAction,
        plan: &PlacementPlan,
        request: CanonicalRequest,
        context: OperationContext,
        cause: String,
        materialization: Option<MaterializationTiming>,
    ) -> Result<PlanExecution, PlanExecutionError> {
        match fallback {
            FallbackAction::ColdOnSameTarget => {
                let adapter = self.engines.adapter_for(&plan.target)?;
                let stream = adapter
                    .execute(&plan.target, request, None, context)
                    .await
                    .map_err(PlanExecutionError::Engine)?;
                Ok(PlanExecution {
                    stream,
                    metadata: PlanExecutionMetadata {
                        executed_path: ExecutedPath::Cold,
                        fallback_used: true,
                        materialization,
                        topology_micros: None,
                    },
                })
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
        materialization: Option<MaterializationTiming>,
    ) -> Result<PlanExecution, PlanExecutionError> {
        let adapter = self.engines.adapter_for(&plan.target)?;
        if let Err(cleanup) = adapter.abort_state_import(import, &context).await {
            return Err(PlanExecutionError::CleanupFailed { cause, cleanup });
        }
        self.apply_fallback(
            plan.fallback,
            plan,
            request,
            context,
            cause,
            materialization,
        )
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
    ) -> Result<PlanExecution, PlanExecutionError> {
        context.ensure_active()?;
        if request.model != plan.target.model {
            return Err(PlanExecutionError::InvalidPlan(
                "request model does not match the selected execution target".to_owned(),
            ));
        }

        let adapter = self.engines.adapter_for(&plan.target)?;
        let ExecutionPath::Reuse(reuse) = &plan.path else {
            let stream = adapter
                .execute(&plan.target, request, None, context)
                .await
                .map_err(PlanExecutionError::Engine)?;
            return Ok(PlanExecution {
                stream,
                metadata: PlanExecutionMetadata {
                    executed_path: ExecutedPath::Cold,
                    fallback_used: false,
                    materialization: None,
                    topology_micros: None,
                },
            });
        };

        let spec = StateImportSpec::from_plan(&reuse.state, reuse.compatibility.clone());
        let activation_started = Instant::now();
        let import = match adapter
            .prepare_state_import(&plan.target, &spec, &context)
            .await
        {
            Ok(import) => import,
            Err(error) => {
                return self
                    .apply_fallback(
                        plan.fallback,
                        &plan,
                        request,
                        context,
                        error.to_string(),
                        None,
                    )
                    .await;
            }
        };

        let materialization_started = Instant::now();
        let receipt = match self
            .store
            .materialize(&reuse.option, &import, &context)
            .await
        {
            Ok(receipt) => receipt,
            Err(error) => {
                return self
                    .abort_then_fallback(&import, &plan, request, context, error.to_string(), None)
                    .await;
            }
        };
        let materialization = MaterializationTiming {
            store: reuse.option.store.as_str().to_owned(),
            state_kind: reuse.option.state_kind.as_str().to_owned(),
            target_id: plan.target.id.to_string(),
            store_estimated_micros: reuse.option.estimated_transfer_micros,
            actual_micros: elapsed_micros(materialization_started),
        };

        let attachment = match adapter
            .commit_state_import(&import, &receipt, &context)
            .await
        {
            Ok(attachment) => attachment,
            Err(error) => {
                return self
                    .abort_then_fallback(
                        &import,
                        &plan,
                        request,
                        context,
                        error.to_string(),
                        Some(materialization),
                    )
                    .await;
            }
        };

        let stream = adapter
            .execute(&plan.target, request, Some(attachment), context)
            .await
            .map_err(PlanExecutionError::Engine)?;
        let topology_micros =
            elapsed_micros(activation_started).saturating_sub(materialization.actual_micros);
        Ok(PlanExecution {
            stream,
            metadata: PlanExecutionMetadata {
                executed_path: ExecutedPath::Reuse,
                fallback_used: false,
                materialization: Some(materialization),
                topology_micros: Some(topology_micros),
            },
        })
    }
}

fn elapsed_micros(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[derive(Debug, Error)]
pub enum PlanExecutionError {
    #[error(transparent)]
    Context(#[from] locus_core::ContextError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("invalid placement plan: {0}")]
    InvalidPlan(String),
    #[error("planned path failed: {0}")]
    PlannedPathFailed(String),
    #[error("planned path failed and requires replanning: {0}")]
    ReplanRequired(String),
    #[error("planned path failed ({cause}); state import cleanup also failed: {cleanup}")]
    CleanupFailed { cause: String, cleanup: EngineError },
}
