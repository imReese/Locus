mod fake;

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::stream::BoxStream;
use locus_core::{
    CanonicalRequest, ContextError, EngineCapabilities, EngineEvent, EngineInstance,
    EngineInstanceId, EngineSnapshot, ExecutionTarget, OperationContext, PreparedStateAttachment,
    RequestId, StateImportSpec, StateImportTarget, TransferReceipt,
};
use thiserror::Error;

pub use fake::{FakeEngineAdapter, FakeEngineCallCounts, FakeEngineOutput, FakeToolCall};

pub type EngineEventStream = BoxStream<'static, Result<EngineEvent, EngineError>>;

#[async_trait]
pub trait EngineAdapter: Send + Sync {
    fn instance(&self) -> EngineInstance;

    async fn execution_targets(
        &self,
        context: &OperationContext,
    ) -> Result<Vec<ExecutionTarget>, EngineError>;

    async fn capabilities(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineCapabilities, EngineError>;

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, EngineError>;

    async fn prepare_state_import(
        &self,
        target: &ExecutionTarget,
        spec: &StateImportSpec,
        context: &OperationContext,
    ) -> Result<StateImportTarget, EngineError>;

    async fn commit_state_import(
        &self,
        import: &StateImportTarget,
        receipt: &TransferReceipt,
        context: &OperationContext,
    ) -> Result<PreparedStateAttachment, EngineError>;

    async fn abort_state_import(
        &self,
        import: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<(), EngineError>;

    async fn execute(
        &self,
        target: &ExecutionTarget,
        request: CanonicalRequest,
        state: Option<PreparedStateAttachment>,
        context: OperationContext,
    ) -> Result<EngineEventStream, EngineError>;

    async fn cancel(
        &self,
        request_id: &RequestId,
        context: &OperationContext,
    ) -> Result<(), EngineError>;
}

#[derive(Clone, Default)]
pub struct EngineRegistry {
    adapters: Arc<RwLock<BTreeMap<EngineInstanceId, Arc<dyn EngineAdapter>>>>,
}

impl EngineRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, adapter: Arc<dyn EngineAdapter>) -> Result<(), EngineError> {
        self.adapters
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .insert(adapter.instance().reference.id, adapter);
        Ok(())
    }

    pub fn adapters(&self) -> Result<Vec<Arc<dyn EngineAdapter>>, EngineError> {
        Ok(self
            .adapters
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .values()
            .cloned()
            .collect())
    }

    pub fn adapter_for(
        &self,
        target: &ExecutionTarget,
    ) -> Result<Arc<dyn EngineAdapter>, EngineError> {
        self.adapters
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .get(&target.engine.id)
            .cloned()
            .ok_or_else(|| EngineError::TargetNotFound(target.id.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Context(#[from] ContextError),
    #[error("execution target was not found: {0}")]
    TargetNotFound(String),
    #[error("engine registry failed: {0}")]
    Registry(String),
    #[error("engine generation is stale")]
    StaleGeneration,
    #[error("engine capability is unsupported: {0}")]
    Unsupported(String),
    #[error("state import failed: {0}")]
    StateImport(String),
    #[error("engine execution failed: {0}")]
    Execution(String),
}
