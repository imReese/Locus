mod fake;

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use locus_core::{
    CancellationToken, CanonicalRequest, ContextError, EngineCapabilities, EngineEvent,
    EngineInstance, EngineInstanceId, EngineSnapshot, ExecutionTarget, OperationContext,
    PreparedStateAttachment, RequestId, StateImportSpec, StateImportTarget, TransferReceipt,
};
use thiserror::Error;
use tokio::sync::Notify;

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
    entries: Arc<RwLock<BTreeMap<EngineInstanceId, RegisteredEngine>>>,
    idle: Arc<Notify>,
}

struct RegisteredEngine {
    adapter: Arc<dyn EngineAdapter>,
    lifecycle: EngineLifecycle,
    active: BTreeMap<RequestId, CancellationToken>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineLifecycle {
    #[default]
    Ready,
    Draining,
    Stopped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineDrainReport {
    pub completed: bool,
    pub forced_cancellations: usize,
}

impl EngineRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, adapter: Arc<dyn EngineAdapter>) -> Result<(), EngineError> {
        let engine_id = adapter.instance().reference.id;
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        if entries.contains_key(&engine_id) {
            return Err(EngineError::Registry(format!(
                "duplicate engine instance id: {engine_id}"
            )));
        }
        entries.insert(
            engine_id,
            RegisteredEngine {
                adapter,
                lifecycle: EngineLifecycle::Ready,
                active: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub fn adapters(&self) -> Result<Vec<Arc<dyn EngineAdapter>>, EngineError> {
        Ok(self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .values()
            .map(|entry| Arc::clone(&entry.adapter))
            .collect())
    }

    pub fn routable_adapters(&self) -> Result<Vec<Arc<dyn EngineAdapter>>, EngineError> {
        Ok(self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .values()
            .filter(|entry| entry.lifecycle == EngineLifecycle::Ready)
            .map(|entry| Arc::clone(&entry.adapter))
            .collect())
    }

    pub fn adapter_for(
        &self,
        target: &ExecutionTarget,
    ) -> Result<Arc<dyn EngineAdapter>, EngineError> {
        let entries = self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        let entry = entries
            .get(&target.engine.id)
            .ok_or_else(|| EngineError::TargetNotFound(target.id.to_string()))?;
        if entry.adapter.instance().reference.generation != target.engine.generation {
            return Err(EngineError::StaleGeneration);
        }
        Ok(Arc::clone(&entry.adapter))
    }

    pub fn acquire_execution(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineExecutionPermit, EngineError> {
        context.ensure_active()?;
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        let entry = entries
            .get_mut(&target.engine.id)
            .ok_or_else(|| EngineError::TargetNotFound(target.id.to_string()))?;
        if entry.adapter.instance().reference.generation != target.engine.generation {
            return Err(EngineError::StaleGeneration);
        }
        match entry.lifecycle {
            EngineLifecycle::Ready => {}
            EngineLifecycle::Draining => return Err(EngineError::Draining),
            EngineLifecycle::Stopped => return Err(EngineError::Stopped),
        }
        if entry.active.contains_key(&context.request_id) {
            return Err(EngineError::Registry(format!(
                "duplicate active request id: {}",
                context.request_id.as_str()
            )));
        }
        entry
            .active
            .insert(context.request_id.clone(), context.cancellation.clone());
        Ok(EngineExecutionPermit {
            registry: self.clone(),
            engine_id: target.engine.id.clone(),
            request_id: context.request_id.clone(),
            released: false,
        })
    }

    pub fn begin_drain_all(&self) -> Result<(), EngineError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        for entry in entries.values_mut() {
            if entry.lifecycle == EngineLifecycle::Ready {
                entry.lifecycle = EngineLifecycle::Draining;
            }
        }
        Ok(())
    }

    pub fn begin_drain(&self, engine_id: &EngineInstanceId) -> Result<(), EngineError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        let entry = entries
            .get_mut(engine_id)
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))?;
        if entry.lifecycle == EngineLifecycle::Ready {
            entry.lifecycle = EngineLifecycle::Draining;
        }
        Ok(())
    }

    pub fn lifecycle(&self, engine_id: &EngineInstanceId) -> Result<EngineLifecycle, EngineError> {
        self.entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .get(engine_id)
            .map(|entry| entry.lifecycle)
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))
    }

    pub fn resume(&self, engine_id: &EngineInstanceId) -> Result<(), EngineError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        let entry = entries
            .get_mut(engine_id)
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))?;
        if entry.lifecycle == EngineLifecycle::Draining {
            entry.lifecycle = EngineLifecycle::Ready;
        }
        Ok(())
    }

    pub async fn drain_all(&self, grace: Duration) -> Result<EngineDrainReport, EngineError> {
        self.begin_drain_all()?;
        let deadline = tokio::time::Instant::now()
            .checked_add(grace)
            .ok_or_else(|| {
                EngineError::Registry("engine drain grace exceeds timer range".to_owned())
            })?;
        loop {
            let notified = self.idle.notified();
            if self.active_executions()? == 0 {
                self.stop_drained()?;
                return Ok(EngineDrainReport {
                    completed: true,
                    forced_cancellations: 0,
                });
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                break;
            }
        }
        let cancellations = self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .values()
            .flat_map(|entry| entry.active.values().cloned())
            .collect::<Vec<_>>();
        for cancellation in &cancellations {
            cancellation.cancel();
        }
        Ok(EngineDrainReport {
            completed: false,
            forced_cancellations: cancellations.len(),
        })
    }

    pub async fn drain(
        &self,
        engine_id: &EngineInstanceId,
        grace: Duration,
    ) -> Result<EngineDrainReport, EngineError> {
        self.begin_drain(engine_id)?;
        let deadline = tokio::time::Instant::now()
            .checked_add(grace)
            .ok_or_else(|| {
                EngineError::Registry("engine drain grace exceeds timer range".to_owned())
            })?;
        loop {
            let notified = self.idle.notified();
            if self.active_executions_for(engine_id)? == 0 {
                self.stop_drained_engine(engine_id)?;
                return Ok(EngineDrainReport {
                    completed: true,
                    forced_cancellations: 0,
                });
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                break;
            }
        }
        let cancellations = self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .get(engine_id)
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))?
            .active
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for cancellation in &cancellations {
            cancellation.cancel();
        }
        Ok(EngineDrainReport {
            completed: false,
            forced_cancellations: cancellations.len(),
        })
    }

    pub fn active_executions(&self) -> Result<usize, EngineError> {
        Ok(self
            .entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .values()
            .map(|entry| entry.active.len())
            .sum())
    }

    pub fn active_executions_for(
        &self,
        engine_id: &EngineInstanceId,
    ) -> Result<usize, EngineError> {
        self.entries
            .read()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?
            .get(engine_id)
            .map(|entry| entry.active.len())
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))
    }

    fn stop_drained(&self) -> Result<(), EngineError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        for entry in entries.values_mut() {
            if entry.lifecycle == EngineLifecycle::Draining && entry.active.is_empty() {
                entry.lifecycle = EngineLifecycle::Stopped;
            }
        }
        Ok(())
    }

    fn stop_drained_engine(&self, engine_id: &EngineInstanceId) -> Result<(), EngineError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|_| EngineError::Registry("engine registry lock poisoned".to_owned()))?;
        let entry = entries
            .get_mut(engine_id)
            .ok_or_else(|| EngineError::TargetNotFound(engine_id.to_string()))?;
        if entry.lifecycle == EngineLifecycle::Draining && entry.active.is_empty() {
            entry.lifecycle = EngineLifecycle::Stopped;
        }
        Ok(())
    }

    fn release_execution(&self, engine_id: &EngineInstanceId, request_id: &RequestId) {
        let Ok(mut entries) = self.entries.write() else {
            return;
        };
        let Some(entry) = entries.get_mut(engine_id) else {
            return;
        };
        if entry.active.remove(request_id).is_some() {
            if entry.lifecycle == EngineLifecycle::Draining && entry.active.is_empty() {
                entry.lifecycle = EngineLifecycle::Stopped;
            }
            self.idle.notify_waiters();
        }
    }
}

pub struct EngineExecutionPermit {
    registry: EngineRegistry,
    engine_id: EngineInstanceId,
    request_id: RequestId,
    released: bool,
}

impl Drop for EngineExecutionPermit {
    fn drop(&mut self) {
        if !self.released {
            self.released = true;
            self.registry
                .release_execution(&self.engine_id, &self.request_id);
        }
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
    #[error("engine is draining and accepts no new execution")]
    Draining,
    #[error("engine is stopped")]
    Stopped,
    #[error("engine capability is unsupported: {0}")]
    Unsupported(String),
    #[error("state import failed: {0}")]
    StateImport(String),
    #[error("engine execution failed: {0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use locus_core::{
        EngineInstanceRef, ExecutionRole, ExecutionTargetId, InputKind, ModelExecutionIdentity,
        ParallelLayout, RuntimeIdentity,
    };

    use super::*;

    fn fake_engine(id: &str) -> (Arc<FakeEngineAdapter>, ExecutionTarget) {
        let engine = EngineInstanceRef {
            id: EngineInstanceId::new(id),
            generation: 1,
        };
        let target = ExecutionTarget {
            id: ExecutionTargetId::new(format!("{id}/model")),
            engine: engine.clone(),
            model: ModelExecutionIdentity {
                model_revision: "model-v1".to_owned(),
                adapter_revision: None,
                execution_profile: "default".to_owned(),
            },
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
        let adapter = Arc::new(FakeEngineAdapter::new(
            EngineInstance {
                reference: engine,
                runtime: RuntimeIdentity {
                    kind: "fake".to_owned(),
                    runtime_version: "v1".to_owned(),
                    adapter_version: "v1".to_owned(),
                },
                topology: "local".to_owned(),
                hardware: "cpu".to_owned(),
                health_endpoint: None,
            },
            target.clone(),
            EngineCapabilities {
                supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                emits_token_deltas: true,
                emits_text_deltas: false,
                emits_reasoning_deltas: false,
                emits_tool_calls: false,
                supports_structured_output: false,
                supported_state_kinds: BTreeSet::new(),
            },
        ));
        (adapter, target)
    }

    #[tokio::test]
    async fn targeted_drain_cancels_only_the_selected_engine() {
        let registry = EngineRegistry::new();
        let (first, first_target) = fake_engine("engine-a");
        let (second, second_target) = fake_engine("engine-b");
        registry.register(first).expect("register first");
        registry.register(second).expect("register second");
        let first_context = OperationContext::new(RequestId::new("first"));
        let second_context = OperationContext::new(RequestId::new("second"));
        let first_permit = registry
            .acquire_execution(&first_target, &first_context)
            .expect("first execution");
        let second_permit = registry
            .acquire_execution(&second_target, &second_context)
            .expect("second execution");

        let report = registry
            .drain(&EngineInstanceId::new("engine-a"), Duration::ZERO)
            .await
            .expect("targeted drain");

        assert!(!report.completed);
        assert_eq!(report.forced_cancellations, 1);
        assert!(first_context.cancellation.is_cancelled());
        assert!(!second_context.cancellation.is_cancelled());
        assert_eq!(
            registry
                .lifecycle(&EngineInstanceId::new("engine-b"))
                .expect("second lifecycle"),
            EngineLifecycle::Ready
        );
        drop(first_permit);
        assert_eq!(
            registry
                .lifecycle(&EngineInstanceId::new("engine-a"))
                .expect("first lifecycle"),
            EngineLifecycle::Stopped
        );
        drop(second_permit);
    }
}
