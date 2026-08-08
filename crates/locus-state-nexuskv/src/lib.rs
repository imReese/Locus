use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use locus_core::{
    BoundaryCompleteness, ComponentCoverage, ExecutionTarget, InputItemId, MaterializationOption,
    MaterializationOptionId, OpaqueHandle, OperationContext, ProviderId, ResumeCoordinate,
    ReusableBoundary, StateDescriptor, StateId, StateImportTarget, StateKind, StateLocality,
    StateRequirement, TransferReceipt,
};
use locus_state::{StateError, StateProvider};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const BRIDGE_SCHEMA: &str = "locus.nexuskv-bridge.v1";
const NEXUSKV_SCHEMA: &str = "nexuskv.contract.v1";

#[derive(Clone, Debug)]
pub struct NexusKvBridgeConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub tenant: String,
    pub namespace: String,
    pub engine_family: String,
    pub semantic_type: String,
}

#[derive(Clone)]
pub struct NexusKvStateProvider {
    identity: ProviderId,
    config: NexusKvBridgeConfig,
    client: Client,
    records: Arc<RwLock<BTreeMap<StateId, NexusStateRecord>>>,
}

#[derive(Clone)]
struct NexusStateRecord {
    source_locator: String,
    source_tier: String,
}

impl NexusKvStateProvider {
    pub fn new(config: NexusKvBridgeConfig) -> Result<Self, StateError> {
        if config.base_url.trim().is_empty() {
            return Err(StateError::Protocol(
                "NexusKV bridge base URL must not be empty".to_owned(),
            ));
        }
        if config.tenant.is_empty() || config.namespace.is_empty() {
            return Err(StateError::Protocol(
                "NexusKV tenant and namespace must not be empty".to_owned(),
            ));
        }
        Ok(Self {
            identity: ProviderId::new("locus.nexuskv-bridge"),
            config,
            client: Client::new(),
            records: Arc::new(RwLock::new(BTreeMap::new())),
        })
    }

    fn endpoint(&self, operation: &str) -> String {
        format!(
            "{}/locus/v1/{operation}",
            self.config.base_url.trim_end_matches('/')
        )
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.config.api_key {
            request.bearer_auth(api_key)
        } else {
            request
        }
    }

    async fn post<Request, Response>(
        &self,
        operation: &str,
        request: &Request,
    ) -> Result<Response, StateError>
    where
        Request: Serialize + Sync,
        Response: for<'de> Deserialize<'de>,
    {
        let response = self
            .authorized(self.client.post(self.endpoint(operation)))
            .json(request)
            .send()
            .await
            .map_err(|error| StateError::Unavailable(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "response body was unavailable".to_owned());
            return Err(match status {
                StatusCode::CONFLICT => StateError::Incompatible(detail),
                StatusCode::UNPROCESSABLE_ENTITY => StateError::Unsupported(detail),
                status if status.is_server_error() => StateError::Unavailable(detail),
                _ => StateError::Protocol(format!("bridge returned {status}: {detail}")),
            });
        }
        response
            .json()
            .await
            .map_err(|error| StateError::Protocol(format!("invalid bridge response: {error}")))
    }
}

#[async_trait]
impl StateProvider for NexusKvStateProvider {
    fn identity(&self) -> &ProviderId {
        &self.identity
    }

    async fn lookup(
        &self,
        requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StateError> {
        context.ensure_active()?;
        let tokens = requirement.query_token_ids.as_ref().ok_or_else(|| {
            StateError::Unsupported("NexusKV lookup requires canonical token IDs".to_owned())
        })?;
        let response: LookupResponse = self
            .post(
                "lookup",
                &LookupRequest {
                    schema_version: BRIDGE_SCHEMA,
                    nexuskv_schema_version: NEXUSKV_SCHEMA,
                    identity: LookupIdentity {
                        tenant: requirement
                            .tenant_scope
                            .as_deref()
                            .unwrap_or(&self.config.tenant),
                        namespace: &self.config.namespace,
                        model: &requirement.model.model_revision,
                        engine_family: &self.config.engine_family,
                        semantic_type: &self.config.semantic_type,
                        tokens,
                    },
                    locus_model_identity: LookupModelIdentity {
                        model_revision: &requirement.model.model_revision,
                        adapter_revision: requirement.model.adapter_revision.as_deref(),
                        execution_profile: &requirement.model.execution_profile,
                    },
                    locus_input_semantic_identity: LookupInputSemanticIdentity {
                        tokenizer: component_ref(&requirement.input_semantics.tokenizer),
                        template: component_ref(&requirement.input_semantics.template),
                        multimodal_preprocessing: requirement
                            .input_semantics
                            .multimodal_preprocessing
                            .as_ref()
                            .map(component_ref),
                    },
                    input_fingerprint: &requirement.input_fingerprint,
                },
            )
            .await?;
        validate_schema(&response.schema_version, &response.nexuskv_schema_version)?;
        let Some(hit) = response.match_result else {
            return Ok(Vec::new());
        };
        if !hit.validation.matches(requirement) {
            return Err(StateError::Incompatible(
                "NexusKV bridge did not validate the requested model and input semantics"
                    .to_owned(),
            ));
        }
        if !hit.compatibility.reusable || hit.compatibility.fallback_to_recompute {
            return Ok(Vec::new());
        }
        if hit.entry.identity.key.model != requirement.model.model_revision {
            return Err(StateError::Incompatible(
                "NexusKV match model does not equal the requested revision".to_owned(),
            ));
        }
        if hit.entry.descriptor.schema_version != NEXUSKV_SCHEMA {
            return Err(StateError::Protocol(format!(
                "unsupported NexusKV descriptor schema: {}",
                hit.entry.descriptor.schema_version
            )));
        }
        if hit.matched_extent.units == 0 {
            return Ok(Vec::new());
        }
        let state_id = StateId::new(hit.entry.identity.entry_id.clone());
        let state_kind = StateKind::new(format!("nexuskv.{}", hit.entry.descriptor.semantic_type));
        if !requirement.accepted_state_kinds.contains(&state_kind) {
            return Ok(Vec::new());
        }
        self.records
            .write()
            .map_err(|_| StateError::Protocol("NexusKV record lock poisoned".to_owned()))?
            .insert(
                state_id.clone(),
                NexusStateRecord {
                    source_locator: hit.entry.location.locator.clone(),
                    source_tier: hit.entry.location.tier.clone(),
                },
            );
        Ok(vec![StateDescriptor {
            id: state_id,
            provider: self.identity.clone(),
            kind: state_kind,
            model: requirement.model.clone(),
            relevant_input_semantics: Some(requirement.input_semantics.clone()),
            representation_revision: hit.entry.descriptor.descriptor_id.clone(),
            positional_semantics: Some(hit.entry.descriptor.granularity.clone()),
            runtime_compatibility: Some(self.config.engine_family.clone()),
            boundary: ReusableBoundary {
                covered_components: vec![ComponentCoverage {
                    item_id: InputItemId::new("prompt"),
                    covered_units: u64::from(hit.matched_extent.units),
                }],
                resume_coordinate: ResumeCoordinate::TokenOffset {
                    item_id: InputItemId::new("prompt"),
                    offset: u64::from(hit.matched_extent.units),
                },
                completeness: BoundaryCompleteness::Complete,
                validation_digest: format!(
                    "{}:{}",
                    hit.entry.identity.version.lineage, hit.entry.identity.version.generation
                ),
            },
            locations: vec![hit.entry.location.locator.clone()],
            provider_reference: OpaqueHandle {
                namespace: "nexuskv.contract.v1.entry".to_owned(),
                value: hit.entry.identity.entry_id,
            },
        }])
    }

    async fn estimate(
        &self,
        state: &StateDescriptor,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StateError> {
        context.ensure_active()?;
        if state.provider != self.identity {
            return Err(StateError::Incompatible(
                "state belongs to another provider".to_owned(),
            ));
        }
        let record = self
            .records
            .read()
            .map_err(|_| StateError::Protocol("NexusKV record lock poisoned".to_owned()))?
            .get(&state.id)
            .cloned()
            .ok_or_else(|| StateError::Protocol("unknown NexusKV state record".to_owned()))?;
        let response: EstimateResponse = self
            .post(
                "estimate",
                &EstimateRequest {
                    schema_version: BRIDGE_SCHEMA,
                    source_state: state.id.as_str(),
                    source_locator: &record.source_locator,
                    source_tier: &record.source_tier,
                    target_id: target.id.as_str(),
                    target_engine_id: target.engine.id.as_str(),
                    target_engine_generation: target.engine.generation,
                    target_residency: &target.residency,
                },
            )
            .await?;
        if response.schema_version != BRIDGE_SCHEMA {
            return Err(StateError::Protocol(format!(
                "unsupported estimate schema: {}",
                response.schema_version
            )));
        }
        Ok(vec![MaterializationOption {
            id: MaterializationOptionId::new(response.option_id),
            provider: self.identity.clone(),
            source_state: state.id.clone(),
            target_id: target.id.clone(),
            target_engine: target.engine.clone(),
            state_kind: state.kind.clone(),
            locality: if response.locality == "local" {
                StateLocality::Local
            } else {
                StateLocality::Remote {
                    topology_path: response
                        .topology_path
                        .unwrap_or_else(|| "unknown".to_owned()),
                }
            },
            estimated_transfer_micros: response.estimated_transfer_micros,
            option_handle: OpaqueHandle {
                namespace: "locus.nexuskv-bridge.materialization.v1".to_owned(),
                value: response.option_handle,
            },
        }])
    }

    async fn materialize(
        &self,
        option: &MaterializationOption,
        target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StateError> {
        context.ensure_active()?;
        if option.provider != self.identity
            || option.target_id != target.target_id
            || option.target_engine != target.engine
            || option.state_kind != target.state_kind
            || option.option_handle.namespace != "locus.nexuskv-bridge.materialization.v1"
        {
            return Err(StateError::Incompatible(
                "NexusKV materialization option does not match the import target".to_owned(),
            ));
        }
        if !self
            .records
            .read()
            .map_err(|_| StateError::Protocol("NexusKV record lock poisoned".to_owned()))?
            .contains_key(&option.source_state)
        {
            return Err(StateError::Protocol(
                "unknown NexusKV materialization source".to_owned(),
            ));
        }
        let response: MaterializeResponse = self
            .post(
                "materialize",
                &MaterializeRequest {
                    schema_version: BRIDGE_SCHEMA,
                    option_id: option.id.as_str(),
                    option_handle: &option.option_handle.value,
                    import_id: target.import_id.as_str(),
                    target_id: target.target_id.as_str(),
                    target_engine_id: target.engine.id.as_str(),
                    target_engine_generation: target.engine.generation,
                    sink_namespace: &target.sink.namespace,
                    sink_value: &target.sink.value,
                },
            )
            .await?;
        if response.schema_version != BRIDGE_SCHEMA {
            return Err(StateError::Protocol(format!(
                "unsupported materialization schema: {}",
                response.schema_version
            )));
        }
        Ok(TransferReceipt {
            import_id: target.import_id.clone(),
            provider: self.identity.clone(),
            bytes_transferred: response.bytes_transferred,
            receipt: OpaqueHandle {
                namespace: response.receipt.namespace,
                value: response.receipt.value,
            },
        })
    }
}

fn validate_schema(bridge: &str, nexuskv: &str) -> Result<(), StateError> {
    if bridge != BRIDGE_SCHEMA || nexuskv != NEXUSKV_SCHEMA {
        return Err(StateError::Protocol(format!(
            "unsupported bridge schemas: bridge={bridge}, nexuskv={nexuskv}"
        )));
    }
    Ok(())
}

fn component_ref(component: &locus_core::SemanticComponentIdentity) -> ComponentIdentityRef<'_> {
    ComponentIdentityRef {
        kind: &component.kind,
        revision: &component.revision,
        fingerprint: &component.fingerprint,
    }
}

#[derive(Serialize)]
struct LookupRequest<'a> {
    schema_version: &'static str,
    nexuskv_schema_version: &'static str,
    identity: LookupIdentity<'a>,
    locus_model_identity: LookupModelIdentity<'a>,
    locus_input_semantic_identity: LookupInputSemanticIdentity<'a>,
    input_fingerprint: &'a str,
}

#[derive(Serialize)]
struct LookupModelIdentity<'a> {
    model_revision: &'a str,
    adapter_revision: Option<&'a str>,
    execution_profile: &'a str,
}

#[derive(Serialize)]
struct LookupInputSemanticIdentity<'a> {
    tokenizer: ComponentIdentityRef<'a>,
    template: ComponentIdentityRef<'a>,
    multimodal_preprocessing: Option<ComponentIdentityRef<'a>>,
}

#[derive(Serialize)]
struct ComponentIdentityRef<'a> {
    kind: &'a str,
    revision: &'a str,
    fingerprint: &'a str,
}

#[derive(Serialize)]
struct LookupIdentity<'a> {
    tenant: &'a str,
    namespace: &'a str,
    model: &'a str,
    engine_family: &'a str,
    semantic_type: &'a str,
    tokens: &'a [u32],
}

#[derive(Deserialize)]
struct LookupResponse {
    schema_version: String,
    nexuskv_schema_version: String,
    match_result: Option<NexusMatchResult>,
}

#[derive(Deserialize)]
struct NexusMatchResult {
    matched_extent: NexusMatchExtent,
    entry: NexusEntry,
    compatibility: NexusCompatibility,
    validation: LocusValidation,
}

#[derive(Deserialize)]
struct NexusMatchExtent {
    units: u32,
}

#[derive(Deserialize)]
struct NexusEntry {
    identity: NexusEntryIdentity,
    descriptor: NexusDescriptor,
    location: NexusLocation,
}

#[derive(Deserialize)]
struct NexusEntryIdentity {
    key: NexusKeyIdentity,
    entry_id: String,
    version: NexusEntryVersion,
}

#[derive(Deserialize)]
struct NexusKeyIdentity {
    model: String,
}

#[derive(Deserialize)]
struct NexusEntryVersion {
    generation: u32,
    lineage: String,
}

#[derive(Deserialize)]
struct NexusDescriptor {
    schema_version: String,
    descriptor_id: String,
    semantic_type: String,
    granularity: String,
    #[serde(flatten)]
    _remaining: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct NexusLocation {
    tier: String,
    locator: String,
}

#[derive(Deserialize)]
struct NexusCompatibility {
    reusable: bool,
    fallback_to_recompute: bool,
    #[serde(default)]
    _reason: String,
}

#[derive(Deserialize)]
struct LocusValidation {
    model_identity: ValidatedModelIdentity,
    input_semantic_identity: ValidatedInputSemanticIdentity,
}

impl LocusValidation {
    fn matches(&self, requirement: &StateRequirement) -> bool {
        self.model_identity.model_revision == requirement.model.model_revision
            && self.model_identity.adapter_revision == requirement.model.adapter_revision
            && self.model_identity.execution_profile == requirement.model.execution_profile
            && self
                .input_semantic_identity
                .matches(&requirement.input_semantics)
    }
}

#[derive(Deserialize)]
struct ValidatedModelIdentity {
    model_revision: String,
    adapter_revision: Option<String>,
    execution_profile: String,
}

#[derive(Deserialize)]
struct ValidatedInputSemanticIdentity {
    tokenizer: ValidatedComponentIdentity,
    template: ValidatedComponentIdentity,
    multimodal_preprocessing: Option<ValidatedComponentIdentity>,
}

impl ValidatedInputSemanticIdentity {
    fn matches(&self, input: &locus_core::InputSemanticIdentity) -> bool {
        self.tokenizer.matches(&input.tokenizer)
            && self.template.matches(&input.template)
            && match (
                &self.multimodal_preprocessing,
                &input.multimodal_preprocessing,
            ) {
                (Some(validated), Some(required)) => validated.matches(required),
                (None, None) => true,
                _ => false,
            }
    }
}

#[derive(Deserialize)]
struct ValidatedComponentIdentity {
    kind: String,
    revision: String,
    fingerprint: String,
}

impl ValidatedComponentIdentity {
    fn matches(&self, component: &locus_core::SemanticComponentIdentity) -> bool {
        self.kind == component.kind
            && self.revision == component.revision
            && self.fingerprint == component.fingerprint
    }
}

#[derive(Serialize)]
struct EstimateRequest<'a> {
    schema_version: &'static str,
    source_state: &'a str,
    source_locator: &'a str,
    source_tier: &'a str,
    target_id: &'a str,
    target_engine_id: &'a str,
    target_engine_generation: u64,
    target_residency: &'a str,
}

#[derive(Deserialize)]
struct EstimateResponse {
    schema_version: String,
    option_id: String,
    option_handle: String,
    locality: String,
    topology_path: Option<String>,
    estimated_transfer_micros: u64,
}

#[derive(Serialize)]
struct MaterializeRequest<'a> {
    schema_version: &'static str,
    option_id: &'a str,
    option_handle: &'a str,
    import_id: &'a str,
    target_id: &'a str,
    target_engine_id: &'a str,
    target_engine_generation: u64,
    sink_namespace: &'a str,
    sink_value: &'a str,
}

#[derive(Deserialize)]
struct MaterializeResponse {
    schema_version: String,
    bytes_transferred: u64,
    receipt: ReceiptHandle,
}

#[derive(Deserialize)]
struct ReceiptHandle {
    namespace: String,
    value: String,
}
