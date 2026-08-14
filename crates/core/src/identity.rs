use std::fmt;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            #[must_use]
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

string_id!(RequestId);
string_id!(EngineInstanceId);
string_id!(ExecutionTargetId);
string_id!(ProviderId);
string_id!(StateId);
string_id!(MaterializationOptionId);
string_id!(ImportId);
string_id!(AttachmentId);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpaqueHandle {
    pub namespace: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeIdentity {
    pub kind: String,
    pub runtime_version: String,
    pub adapter_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EngineInstanceRef {
    pub id: EngineInstanceId,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineInstance {
    pub reference: EngineInstanceRef,
    pub runtime: RuntimeIdentity,
    pub topology: String,
    pub hardware: String,
    pub health_endpoint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelExecutionIdentity {
    pub model_revision: String,
    pub adapter_revision: Option<String>,
    pub execution_profile: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionRole {
    Combined,
    Prefill,
    Decode,
    Namespaced(String),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParallelLayout {
    pub tensor_parallel: u16,
    pub pipeline_parallel: u16,
    pub expert_parallel: u16,
    pub layout_revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionTarget {
    pub id: ExecutionTargetId,
    pub engine: EngineInstanceRef,
    pub model: ModelExecutionIdentity,
    pub role: ExecutionRole,
    pub parallel_layout: ParallelLayout,
    pub residency: String,
    pub capability_revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticComponentIdentity {
    pub kind: String,
    pub revision: String,
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputSemanticIdentity {
    pub tokenizer: SemanticComponentIdentity,
    pub template: SemanticComponentIdentity,
    pub multimodal_preprocessing: Option<SemanticComponentIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationSemanticIdentity {
    pub sampling_normalization: SemanticComponentIdentity,
    pub stop_behavior: SemanticComponentIdentity,
    pub constrained_generation: Option<SemanticComponentIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputSemanticIdentity {
    pub detokenizer: SemanticComponentIdentity,
    pub reasoning_parser: Option<SemanticComponentIdentity>,
    pub tool_parser: Option<SemanticComponentIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SemanticIdentity {
    pub input: InputSemanticIdentity,
    pub generation: GenerationSemanticIdentity,
    pub output: OutputSemanticIdentity,
    pub umbrella_fingerprint: Option<String>,
}
