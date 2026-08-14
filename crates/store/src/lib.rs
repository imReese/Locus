mod fake;

use async_trait::async_trait;
use locus_core::{
    ContextError, ExecutionTarget, MaterializationOption, OperationContext, StateDescriptor,
    StateImportTarget, StateRequirement, StoreId, TransferReceipt,
};
use thiserror::Error;

pub use fake::{FakeStateCallCounts, FakeStateStore};

#[async_trait]
pub trait StateStore: Send + Sync {
    fn identity(&self) -> &StoreId;

    async fn lookup(
        &self,
        requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StoreError>;

    async fn estimate(
        &self,
        state: &StateDescriptor,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StoreError>;

    async fn materialize(
        &self,
        option: &MaterializationOption,
        target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StoreError>;
}

pub struct NullStateStore {
    identity: StoreId,
}

impl Default for NullStateStore {
    fn default() -> Self {
        Self {
            identity: StoreId::new("locus.null-state-store"),
        }
    }
}

#[async_trait]
impl StateStore for NullStateStore {
    fn identity(&self) -> &StoreId {
        &self.identity
    }

    async fn lookup(
        &self,
        _requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StoreError> {
        context.ensure_active()?;
        Ok(Vec::new())
    }

    async fn estimate(
        &self,
        _state: &StateDescriptor,
        _target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StoreError> {
        context.ensure_active()?;
        Ok(Vec::new())
    }

    async fn materialize(
        &self,
        _option: &MaterializationOption,
        _target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StoreError> {
        context.ensure_active()?;
        Err(StoreError::Unsupported(
            "null store cannot materialize state".to_owned(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error("state store is unavailable: {0}")]
    Unavailable(String),
    #[error("state operation is unsupported: {0}")]
    Unsupported(String),
    #[error("state compatibility failed: {0}")]
    Incompatible(String),
    #[error("state store protocol failed: {0}")]
    Protocol(String),
    #[error("state materialization failed: {0}")]
    Materialization(String),
}
