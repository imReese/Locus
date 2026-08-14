use std::collections::BTreeSet;

use crate::{
    ExecutionTargetId, InputBundle, InputKind, ModelExecutionIdentity, RequestId, SemanticIdentity,
    StateKind,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingParameters {
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapabilityRequirements {
    pub input_kinds: BTreeSet<InputKind>,
    pub requires_incremental_output: bool,
    pub requires_reasoning_deltas: bool,
    pub requires_tool_calls: bool,
    pub requires_structured_output: bool,
}

impl CapabilityRequirements {
    #[must_use]
    pub fn for_input(input: &InputBundle) -> Self {
        Self {
            input_kinds: input.kinds().collect(),
            requires_incremental_output: true,
            requires_reasoning_deltas: false,
            requires_tool_calls: false,
            requires_structured_output: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalRequest {
    pub id: RequestId,
    pub model: ModelExecutionIdentity,
    pub semantic_identity: SemanticIdentity,
    pub input: InputBundle,
    pub sampling: SamplingParameters,
    pub requirements: CapabilityRequirements,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineCapabilities {
    pub supported_input_kinds: BTreeSet<InputKind>,
    pub emits_token_deltas: bool,
    pub emits_text_deltas: bool,
    pub emits_reasoning_deltas: bool,
    pub emits_tool_calls: bool,
    pub supports_structured_output: bool,
    pub supported_state_kinds: BTreeSet<StateKind>,
}

impl EngineCapabilities {
    #[must_use]
    pub fn satisfies(&self, requirements: &CapabilityRequirements) -> bool {
        requirements
            .input_kinds
            .is_subset(&self.supported_input_kinds)
            && (!requirements.requires_incremental_output
                || self.emits_token_deltas
                || self.emits_text_deltas)
            && (!requirements.requires_reasoning_deltas || self.emits_reasoning_deltas)
            && (!requirements.requires_tool_calls || self.emits_tool_calls)
            && (!requirements.requires_structured_output || self.supports_structured_output)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineSnapshot {
    pub target_id: ExecutionTargetId,
    pub ready: bool,
    pub queue_depth: u64,
    pub estimated_queue_micros: Option<u64>,
    pub observation_revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFinishReason {
    Stop,
    Length,
    Cancelled,
    Error,
    RuntimeSpecific { namespace: String, value: String },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineEvent {
    Accepted {
        request_id: RequestId,
    },
    TokenDelta {
        request_id: RequestId,
        sequence_number: u64,
        token_ids: Vec<u32>,
    },
    TextDelta {
        request_id: RequestId,
        sequence_number: u64,
        text: String,
    },
    ReasoningDelta {
        request_id: RequestId,
        sequence_number: u64,
        text: String,
    },
    ToolCallStarted {
        request_id: RequestId,
        sequence_number: u64,
        call_id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        request_id: RequestId,
        sequence_number: u64,
        call_id: String,
        delta: String,
    },
    ToolCallCompleted {
        request_id: RequestId,
        sequence_number: u64,
        call_id: String,
    },
    UsageUpdate {
        request_id: RequestId,
        usage: Usage,
        final_update: bool,
    },
    Finished {
        request_id: RequestId,
        reason: EngineFinishReason,
        usage: Usage,
    },
}
