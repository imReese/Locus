mod fake;

use async_trait::async_trait;
use locus_core::{
    ContextError, ExecutionTarget, MaterializationOption, OperationContext, ProviderId,
    StateDescriptor, StateImportTarget, StateRequirement, TransferReceipt,
};
use thiserror::Error;

pub use fake::{FakeStateCallCounts, FakeStateProvider};

#[async_trait]
pub trait StateProvider: Send + Sync {
    fn identity(&self) -> &ProviderId;

    async fn lookup(
        &self,
        requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StateError>;

    async fn estimate(
        &self,
        state: &StateDescriptor,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StateError>;

    async fn materialize(
        &self,
        option: &MaterializationOption,
        target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StateError>;
}

pub struct NullStateProvider {
    identity: ProviderId,
}

impl Default for NullStateProvider {
    fn default() -> Self {
        Self {
            identity: ProviderId::new("locus.null-state-provider"),
        }
    }
}

#[async_trait]
impl StateProvider for NullStateProvider {
    fn identity(&self) -> &ProviderId {
        &self.identity
    }

    async fn lookup(
        &self,
        _requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StateError> {
        context.ensure_active()?;
        Ok(Vec::new())
    }

    async fn estimate(
        &self,
        _state: &StateDescriptor,
        _target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StateError> {
        context.ensure_active()?;
        Ok(Vec::new())
    }

    async fn materialize(
        &self,
        _option: &MaterializationOption,
        _target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StateError> {
        context.ensure_active()?;
        Err(StateError::Unsupported(
            "null provider cannot materialize state".to_owned(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error("state provider is unavailable: {0}")]
    Unavailable(String),
    #[error("state operation is unsupported: {0}")]
    Unsupported(String),
    #[error("state compatibility failed: {0}")]
    Incompatible(String),
    #[error("state provider protocol failed: {0}")]
    Protocol(String),
    #[error("state materialization failed: {0}")]
    Materialization(String),
}
