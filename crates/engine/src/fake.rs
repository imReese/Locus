use std::collections::BTreeMap;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::{StreamExt, stream};
use locus_core::{
    AttachmentId, CanonicalRequest, EngineCapabilities, EngineEvent, EngineFinishReason,
    EngineInstance, EngineInstanceRef, EngineSnapshot, ExecutionTarget, ImportId, OpaqueHandle,
    OperationContext, PreparedStateAttachment, RequestId, StateImportSpec, StateImportTarget,
    TransferReceipt, Usage,
};

use crate::{EngineAdapter, EngineError, EngineEventStream};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FakeEngineCallCounts {
    pub prepare: usize,
    pub commit: usize,
    pub abort: usize,
    pub execute: usize,
    pub cancel: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeToolCall {
    pub call_id: String,
    pub name: String,
    pub argument_deltas: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FakeEngineOutput {
    pub token_deltas: Vec<Vec<u32>>,
    pub text_deltas: Vec<String>,
    pub reasoning_deltas: Vec<String>,
    pub tool_calls: Vec<FakeToolCall>,
    pub finish_reason: EngineFinishReason,
    pub execute_delay: Duration,
    pub event_delay: Duration,
}

impl Default for FakeEngineOutput {
    fn default() -> Self {
        Self {
            token_deltas: vec![vec![42]],
            text_deltas: Vec::new(),
            reasoning_deltas: Vec::new(),
            tool_calls: Vec::new(),
            finish_reason: EngineFinishReason::Stop,
            execute_delay: Duration::ZERO,
            event_delay: Duration::ZERO,
        }
    }
}

pub struct FakeEngineAdapter {
    instance_template: EngineInstance,
    target_template: ExecutionTarget,
    capabilities: EngineCapabilities,
    output: FakeEngineOutput,
    generation: AtomicU64,
    next_import: AtomicU64,
    imports: Mutex<BTreeMap<ImportId, StateImportSpec>>,
    prepare_calls: AtomicUsize,
    commit_calls: AtomicUsize,
    abort_calls: AtomicUsize,
    execute_calls: AtomicUsize,
    cancel_calls: AtomicUsize,
}

impl FakeEngineAdapter {
    #[must_use]
    pub fn new(
        instance: EngineInstance,
        target: ExecutionTarget,
        capabilities: EngineCapabilities,
    ) -> Self {
        Self {
            generation: AtomicU64::new(instance.reference.generation),
            instance_template: instance,
            target_template: target,
            capabilities,
            output: FakeEngineOutput::default(),
            next_import: AtomicU64::new(1),
            imports: Mutex::new(BTreeMap::new()),
            prepare_calls: AtomicUsize::new(0),
            commit_calls: AtomicUsize::new(0),
            abort_calls: AtomicUsize::new(0),
            execute_calls: AtomicUsize::new(0),
            cancel_calls: AtomicUsize::new(0),
        }
    }

    #[must_use]
    pub fn with_output(mut self, output: FakeEngineOutput) -> Self {
        self.output = output;
        self
    }

    #[must_use]
    pub fn current_target(&self) -> ExecutionTarget {
        let mut target = self.target_template.clone();
        target.engine.generation = self.generation.load(Ordering::Acquire);
        target
    }

    pub fn restart(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    #[must_use]
    pub fn call_counts(&self) -> FakeEngineCallCounts {
        FakeEngineCallCounts {
            prepare: self.prepare_calls.load(Ordering::Acquire),
            commit: self.commit_calls.load(Ordering::Acquire),
            abort: self.abort_calls.load(Ordering::Acquire),
            execute: self.execute_calls.load(Ordering::Acquire),
            cancel: self.cancel_calls.load(Ordering::Acquire),
        }
    }

    fn validate_target(&self, target: &ExecutionTarget) -> Result<(), EngineError> {
        if target.id != self.target_template.id
            || target.engine.id != self.instance_template.reference.id
        {
            return Err(EngineError::TargetNotFound(target.id.to_string()));
        }
        if target.engine.generation != self.generation.load(Ordering::Acquire) {
            return Err(EngineError::StaleGeneration);
        }
        Ok(())
    }

    fn validate_engine_ref(&self, engine: &EngineInstanceRef) -> Result<(), EngineError> {
        if engine.id != self.instance_template.reference.id {
            return Err(EngineError::TargetNotFound(engine.id.to_string()));
        }
        if engine.generation != self.generation.load(Ordering::Acquire) {
            return Err(EngineError::StaleGeneration);
        }
        Ok(())
    }
}

#[async_trait]
impl EngineAdapter for FakeEngineAdapter {
    fn instance(&self) -> EngineInstance {
        let mut instance = self.instance_template.clone();
        instance.reference.generation = self.generation.load(Ordering::Acquire);
        instance
    }

    async fn execution_targets(
        &self,
        context: &OperationContext,
    ) -> Result<Vec<ExecutionTarget>, EngineError> {
        context.ensure_active()?;
        Ok(vec![self.current_target()])
    }

    async fn capabilities(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineCapabilities, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        Ok(self.capabilities.clone())
    }

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        Ok(EngineSnapshot {
            target_id: target.id.clone(),
            ready: true,
            telemetry_status: locus_core::TelemetryStatus::Fresh,
            telemetry_confidence: locus_core::TelemetryConfidence::High,
            telemetry_source: "locus.fake.v1".to_owned(),
            observed_at_unix_millis: 1,
            valid_until_unix_millis: u64::MAX,
            running_requests: Some(0),
            waiting_requests: Some(0),
            estimated_queue_micros: Some(0),
            kv_cache_usage_permyriad: Some(0),
            prefill_tokens_per_second: Some(100_000),
            decode_tokens_per_second: Some(50_000),
            observation_revision: 1,
            degraded_reason: None,
        })
    }

    async fn prepare_state_import(
        &self,
        target: &ExecutionTarget,
        spec: &StateImportSpec,
        context: &OperationContext,
    ) -> Result<StateImportTarget, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        if !spec.compatibility.is_compatible() {
            return Err(EngineError::StateImport(
                "incompatible state reached import preparation".to_owned(),
            ));
        }
        if !self
            .capabilities
            .supported_state_kinds
            .contains(&spec.state_kind)
        {
            return Err(EngineError::Unsupported(format!(
                "state kind {}",
                spec.state_kind.as_str()
            )));
        }

        self.prepare_calls.fetch_add(1, Ordering::AcqRel);
        let import_id = ImportId::new(format!(
            "import-{}",
            self.next_import.fetch_add(1, Ordering::AcqRel)
        ));
        self.imports
            .lock()
            .map_err(|_| EngineError::StateImport("fake import registry poisoned".to_owned()))?
            .insert(import_id.clone(), spec.clone());

        Ok(StateImportTarget {
            import_id,
            target_id: target.id.clone(),
            engine: target.engine.clone(),
            state_kind: spec.state_kind.clone(),
            sink: OpaqueHandle {
                namespace: "locus.fake.engine-sink.v1".to_owned(),
                value: target.id.to_string(),
            },
            expires_at: Instant::now() + Duration::from_secs(60),
        })
    }

    async fn commit_state_import(
        &self,
        import: &StateImportTarget,
        receipt: &TransferReceipt,
        context: &OperationContext,
    ) -> Result<PreparedStateAttachment, EngineError> {
        context.ensure_active()?;
        self.validate_engine_ref(&import.engine)?;
        if Instant::now() >= import.expires_at {
            return Err(EngineError::StateImport("import target expired".to_owned()));
        }
        if receipt.import_id != import.import_id {
            return Err(EngineError::StateImport(
                "transfer receipt does not match import".to_owned(),
            ));
        }

        self.commit_calls.fetch_add(1, Ordering::AcqRel);
        let spec = self
            .imports
            .lock()
            .map_err(|_| EngineError::StateImport("fake import registry poisoned".to_owned()))?
            .remove(&import.import_id)
            .ok_or_else(|| EngineError::StateImport("unknown import handle".to_owned()))?;

        Ok(PreparedStateAttachment {
            id: AttachmentId::new(format!("attachment-{}", import.import_id)),
            target_id: import.target_id.clone(),
            engine: import.engine.clone(),
            state_kind: import.state_kind.clone(),
            boundary: spec.boundary,
            expires_at: Instant::now() + Duration::from_secs(60),
        })
    }

    async fn abort_state_import(
        &self,
        import: &StateImportTarget,
        _context: &OperationContext,
    ) -> Result<(), EngineError> {
        // Cleanup must remain possible after cancellation or deadline expiry.
        self.abort_calls.fetch_add(1, Ordering::AcqRel);
        self.imports
            .lock()
            .map_err(|_| EngineError::StateImport("fake import registry poisoned".to_owned()))?
            .remove(&import.import_id);
        Ok(())
    }

    async fn execute(
        &self,
        target: &ExecutionTarget,
        request: CanonicalRequest,
        state: Option<PreparedStateAttachment>,
        context: OperationContext,
    ) -> Result<EngineEventStream, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        if state
            .as_ref()
            .is_some_and(|attachment| !attachment.is_valid_for(target))
        {
            return Err(EngineError::StateImport(
                "prepared state is not valid for target generation".to_owned(),
            ));
        }
        self.execute_calls.fetch_add(1, Ordering::AcqRel);
        if !self.output.execute_delay.is_zero() {
            tokio::time::sleep(self.output.execute_delay).await;
        }

        let request_id = request.id;
        let output_tokens = self
            .output
            .token_deltas
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .saturating_add(
                self.output
                    .text_deltas
                    .iter()
                    .map(|delta| delta.chars().count())
                    .sum(),
            ) as u64;
        let usage = Usage {
            input_tokens: request
                .input
                .items
                .iter()
                .filter_map(|item| match &item.value {
                    locus_core::InputItemValue::TokenSequence(tokens) => {
                        Some(tokens.token_ids.len() as u64)
                    }
                    _ => None,
                })
                .sum(),
            output_tokens,
        };
        let mut events = vec![Ok(EngineEvent::Accepted {
            request_id: request_id.clone(),
        })];
        let mut sequence_number = 1;
        for token_ids in &self.output.token_deltas {
            events.push(Ok(EngineEvent::TokenDelta {
                request_id: request_id.clone(),
                sequence_number,
                token_ids: token_ids.clone(),
            }));
            sequence_number += 1;
        }
        for text in &self.output.text_deltas {
            events.push(Ok(EngineEvent::TextDelta {
                request_id: request_id.clone(),
                sequence_number,
                text: text.clone(),
            }));
            sequence_number += 1;
        }
        for text in &self.output.reasoning_deltas {
            events.push(Ok(EngineEvent::ReasoningDelta {
                request_id: request_id.clone(),
                sequence_number,
                text: text.clone(),
            }));
            sequence_number += 1;
        }
        for tool_call in &self.output.tool_calls {
            events.push(Ok(EngineEvent::ToolCallStarted {
                request_id: request_id.clone(),
                sequence_number,
                call_id: tool_call.call_id.clone(),
                name: tool_call.name.clone(),
            }));
            sequence_number += 1;
            for delta in &tool_call.argument_deltas {
                events.push(Ok(EngineEvent::ToolCallArgumentsDelta {
                    request_id: request_id.clone(),
                    sequence_number,
                    call_id: tool_call.call_id.clone(),
                    delta: delta.clone(),
                }));
                sequence_number += 1;
            }
            events.push(Ok(EngineEvent::ToolCallCompleted {
                request_id: request_id.clone(),
                sequence_number,
                call_id: tool_call.call_id.clone(),
            }));
            sequence_number += 1;
        }
        events.push(Ok(EngineEvent::Finished {
            request_id,
            reason: self.output.finish_reason.clone(),
            usage,
        }));
        let delay = self.output.event_delay;
        if delay.is_zero() {
            Ok(stream::iter(events).boxed())
        } else {
            Ok(stream::iter(events)
                .then(move |event| async move {
                    tokio::time::sleep(delay).await;
                    event
                })
                .boxed())
        }
    }

    async fn cancel(
        &self,
        _request_id: &RequestId,
        context: &OperationContext,
    ) -> Result<(), EngineError> {
        context.ensure_active()?;
        self.cancel_calls.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}
