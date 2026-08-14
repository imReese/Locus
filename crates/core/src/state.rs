use std::collections::BTreeSet;
use std::time::Instant;

use crate::{
    AttachmentId, EngineInstanceRef, ExecutionTarget, ExecutionTargetId, ImportId, InputItemId,
    InputSemanticIdentity, MaterializationOptionId, ModelExecutionIdentity, OpaqueHandle,
    ProviderId, StateId,
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateKind(String);

impl StateKind {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentCoverage {
    pub item_id: InputItemId,
    pub covered_units: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResumeCoordinate {
    TokenOffset { item_id: InputItemId, offset: u64 },
    Checkpoint { namespace: String, step: u64 },
    ItemBoundary { item_id: InputItemId },
    Typed { type_url: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryCompleteness {
    Complete,
    Checkpointed,
    Partial,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReusableBoundary {
    pub covered_components: Vec<ComponentCoverage>,
    pub resume_coordinate: ResumeCoordinate,
    pub completeness: BoundaryCompleteness,
    pub validation_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompatibilityVerdict {
    Compatible,
    Incompatible,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatibilityResult {
    pub verdict: CompatibilityVerdict,
    pub checked_dimensions: Vec<String>,
    pub evidence: Vec<String>,
    pub required_conversion: Option<String>,
}

impl CompatibilityResult {
    #[must_use]
    pub fn compatible(evidence: impl Into<String>) -> Self {
        Self {
            verdict: CompatibilityVerdict::Compatible,
            checked_dimensions: vec!["model_execution".to_owned(), "input_semantics".to_owned()],
            evidence: vec![evidence.into()],
            required_conversion: None,
        }
    }

    #[must_use]
    pub fn incompatible(evidence: impl Into<String>) -> Self {
        Self {
            verdict: CompatibilityVerdict::Incompatible,
            checked_dimensions: vec!["model_execution".to_owned(), "input_semantics".to_owned()],
            evidence: vec![evidence.into()],
            required_conversion: None,
        }
    }

    #[must_use]
    pub fn is_compatible(&self) -> bool {
        self.verdict == CompatibilityVerdict::Compatible
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateRequirement {
    pub model: ModelExecutionIdentity,
    pub input_semantics: InputSemanticIdentity,
    pub accepted_state_kinds: BTreeSet<StateKind>,
    pub input_fingerprint: String,
    pub query_token_ids: Option<Vec<u32>>,
    pub tenant_scope: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateDescriptor {
    pub id: StateId,
    pub provider: ProviderId,
    pub kind: StateKind,
    pub model: ModelExecutionIdentity,
    pub relevant_input_semantics: Option<InputSemanticIdentity>,
    pub representation_revision: String,
    pub positional_semantics: Option<String>,
    pub runtime_compatibility: Option<String>,
    pub boundary: ReusableBoundary,
    pub locations: Vec<String>,
    pub provider_reference: OpaqueHandle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateLocality {
    Local,
    Remote { topology_path: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializationOption {
    pub id: MaterializationOptionId,
    pub provider: ProviderId,
    pub source_state: StateId,
    pub target_id: ExecutionTargetId,
    pub target_engine: EngineInstanceRef,
    pub state_kind: StateKind,
    pub locality: StateLocality,
    pub estimated_transfer_micros: u64,
    pub option_handle: OpaqueHandle,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateImportSpec {
    pub source_state: StateId,
    pub state_kind: StateKind,
    pub boundary: ReusableBoundary,
    pub compatibility: CompatibilityResult,
}

impl StateImportSpec {
    #[must_use]
    pub fn from_plan(state: &StateDescriptor, compatibility: CompatibilityResult) -> Self {
        Self {
            source_state: state.id.clone(),
            state_kind: state.kind.clone(),
            boundary: state.boundary.clone(),
            compatibility,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StateImportTarget {
    pub import_id: ImportId,
    pub target_id: ExecutionTargetId,
    pub engine: EngineInstanceRef,
    pub state_kind: StateKind,
    pub sink: OpaqueHandle,
    pub expires_at: Instant,
}

impl StateImportTarget {
    #[must_use]
    pub fn is_valid_for(&self, target: &ExecutionTarget) -> bool {
        self.target_id == target.id
            && self.engine == target.engine
            && Instant::now() < self.expires_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferReceipt {
    pub import_id: ImportId,
    pub provider: ProviderId,
    pub bytes_transferred: u64,
    pub receipt: OpaqueHandle,
}

#[derive(Clone, Debug)]
pub struct PreparedStateAttachment {
    pub id: AttachmentId,
    pub target_id: ExecutionTargetId,
    pub engine: EngineInstanceRef,
    pub state_kind: StateKind,
    pub boundary: ReusableBoundary,
    pub expires_at: Instant,
}

impl PreparedStateAttachment {
    #[must_use]
    pub fn is_valid_for(&self, target: &ExecutionTarget) -> bool {
        self.target_id == target.id
            && self.engine == target.engine
            && Instant::now() < self.expires_at
    }
}
