use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use async_trait::async_trait;
use locus_core::{
    ExecutionTarget, MaterializationOption, OpaqueHandle, OperationContext, ProviderId,
    StateDescriptor, StateImportTarget, StateRequirement, TransferReceipt,
};

use crate::{StateError, StateProvider};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FakeStateCallCounts {
    pub lookup: usize,
    pub estimate: usize,
    pub materialize: usize,
}

pub struct FakeStateProvider {
    identity: ProviderId,
    states: Vec<StateDescriptor>,
    options: Vec<MaterializationOption>,
    unavailable: AtomicBool,
    fail_materialization: AtomicBool,
    lookup_calls: AtomicUsize,
    estimate_calls: AtomicUsize,
    materialize_calls: AtomicUsize,
}

impl FakeStateProvider {
    #[must_use]
    pub fn new(
        identity: ProviderId,
        states: Vec<StateDescriptor>,
        options: Vec<MaterializationOption>,
    ) -> Self {
        Self {
            identity,
            states,
            options,
            unavailable: AtomicBool::new(false),
            fail_materialization: AtomicBool::new(false),
            lookup_calls: AtomicUsize::new(0),
            estimate_calls: AtomicUsize::new(0),
            materialize_calls: AtomicUsize::new(0),
        }
    }

    pub fn set_unavailable(&self, unavailable: bool) {
        self.unavailable.store(unavailable, Ordering::Release);
    }

    pub fn set_fail_materialization(&self, fail: bool) {
        self.fail_materialization.store(fail, Ordering::Release);
    }

    #[must_use]
    pub fn call_counts(&self) -> FakeStateCallCounts {
        FakeStateCallCounts {
            lookup: self.lookup_calls.load(Ordering::Acquire),
            estimate: self.estimate_calls.load(Ordering::Acquire),
            materialize: self.materialize_calls.load(Ordering::Acquire),
        }
    }
}

#[async_trait]
impl StateProvider for FakeStateProvider {
    fn identity(&self) -> &ProviderId {
        &self.identity
    }

    async fn lookup(
        &self,
        requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StateError> {
        context.ensure_active()?;
        self.lookup_calls.fetch_add(1, Ordering::AcqRel);
        if self.unavailable.load(Ordering::Acquire) {
            return Err(StateError::Unavailable("configured fake outage".to_owned()));
        }
        Ok(self
            .states
            .iter()
            .filter(|state| {
                state.model == requirement.model
                    && requirement.accepted_state_kinds.contains(&state.kind)
                    && state
                        .relevant_input_semantics
                        .as_ref()
                        .is_none_or(|identity| identity == &requirement.input_semantics)
            })
            .cloned()
            .collect())
    }

    async fn estimate(
        &self,
        state: &StateDescriptor,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StateError> {
        context.ensure_active()?;
        self.estimate_calls.fetch_add(1, Ordering::AcqRel);
        if self.unavailable.load(Ordering::Acquire) {
            return Err(StateError::Unavailable("configured fake outage".to_owned()));
        }
        Ok(self
            .options
            .iter()
            .filter(|option| option.source_state == state.id && option.target_id == target.id)
            .cloned()
            .collect())
    }

    async fn materialize(
        &self,
        option: &MaterializationOption,
        target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StateError> {
        context.ensure_active()?;
        self.materialize_calls.fetch_add(1, Ordering::AcqRel);
        if self.unavailable.load(Ordering::Acquire) {
            return Err(StateError::Unavailable("configured fake outage".to_owned()));
        }
        if self.fail_materialization.load(Ordering::Acquire) {
            return Err(StateError::Materialization(
                "configured fake materialization failure".to_owned(),
            ));
        }
        if option.provider != self.identity {
            return Err(StateError::Materialization(
                "materialization option belongs to another provider".to_owned(),
            ));
        }
        if option.target_id != target.target_id
            || option.target_engine != target.engine
            || option.state_kind != target.state_kind
        {
            return Err(StateError::Materialization(
                "import target does not match materialization option".to_owned(),
            ));
        }
        if Instant::now() >= target.expires_at {
            return Err(StateError::Materialization(
                "import target expired before transfer".to_owned(),
            ));
        }

        Ok(TransferReceipt {
            import_id: target.import_id.clone(),
            provider: self.identity.clone(),
            bytes_transferred: 4096,
            receipt: OpaqueHandle {
                namespace: "locus.fake.transfer-receipt.v1".to_owned(),
                value: option.id.to_string(),
            },
        })
    }
}
