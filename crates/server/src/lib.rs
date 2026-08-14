use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderName, HeaderValue, Request};
use locus_core::{
    EngineInstance, EngineInstanceId, EngineInstanceRef, ExecutionRole, ExecutionTarget,
    ExecutionTargetId, ModelExecutionIdentity, ParallelLayout, RuntimeIdentity,
};
use locus_engine::{EngineAdapter, EngineRegistry};
use locus_engine_openai::{
    RemoteEngineConfig, RemoteExecutionTarget, RemoteTelemetryConfig, SglangEngineAdapter,
    VllmEngineAdapter,
};
use locus_openai::{ApiConfig, router_with_config};
use locus_planner::{CalibrationPolicy, PersistentCalibrator, PlacementMode};
use locus_runtime::{DefaultInferenceService, InferenceService, PlacementControl};
use locus_semantics::ModelRegistry;
use locus_semantics_hf::{
    HuggingFaceProfileSpec, TaggedJsonToolParserSpec, TaggedReasoningParserSpec,
    load_huggingface_semantics,
};
use locus_state::{NullStateProvider, StateProvider};
use locus_state_nexuskv::{NexusKvBridgeConfig, NexusKvStateProvider};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::request_id::{
    MakeRequestId, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::{Level, info_span};

const REQUEST_ID_HEADER: &str = "x-request-id";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: SocketAddr,
    #[serde(default)]
    pub api: ApiSettings,
    pub models: Vec<ModelSettings>,
    #[serde(default)]
    pub required_models: Vec<String>,
    pub engines: Vec<EngineSettings>,
    #[serde(default)]
    pub state: StateSettings,
    #[serde(default)]
    pub observability: ObservabilitySettings,
    #[serde(default)]
    pub placement: PlacementSettings,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiSettings {
    #[serde(default)]
    pub bearer_token_env: Option<String>,
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes: usize,
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,
}

impl Default for ApiSettings {
    fn default() -> Self {
        Self {
            bearer_token_env: None,
            max_request_bytes: default_max_request_bytes(),
            max_concurrent_requests: default_max_concurrent_requests(),
        }
    }
}

const fn default_max_request_bytes() -> usize {
    2 * 1024 * 1024
}

const fn default_max_concurrent_requests() -> usize {
    128
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSettings {
    pub aliases: Vec<String>,
    pub model_revision: String,
    #[serde(default)]
    pub adapter_revision: Option<String>,
    #[serde(default = "default_execution_profile")]
    pub execution_profile: String,
    pub tokenizer_json: PathBuf,
    pub tokenizer_revision: String,
    pub chat_template: PathBuf,
    pub template_revision: String,
    #[serde(default)]
    pub template_context: BTreeMap<String, Value>,
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    #[serde(default = "default_max_rendered_bytes")]
    pub max_rendered_bytes: usize,
    #[serde(default)]
    pub reasoning_parser: Option<ReasoningParserSettings>,
    #[serde(default)]
    pub tool_parser: Option<ToolParserSettings>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningParserSettings {
    Tagged {
        revision: String,
        start_delimiter: String,
        end_delimiter: String,
    },
}

impl ReasoningParserSettings {
    fn profile_spec(&self) -> TaggedReasoningParserSpec {
        match self {
            Self::Tagged {
                revision,
                start_delimiter,
                end_delimiter,
            } => TaggedReasoningParserSpec {
                revision: revision.clone(),
                start_delimiter: start_delimiter.clone(),
                end_delimiter: end_delimiter.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ToolParserSettings {
    TaggedJson {
        revision: String,
        start_delimiter: String,
        end_delimiter: String,
        #[serde(default = "default_max_tool_call_bytes")]
        max_buffered_bytes: usize,
    },
}

impl ToolParserSettings {
    fn profile_spec(&self) -> TaggedJsonToolParserSpec {
        match self {
            Self::TaggedJson {
                revision,
                start_delimiter,
                end_delimiter,
                max_buffered_bytes,
            } => TaggedJsonToolParserSpec {
                revision: revision.clone(),
                start_delimiter: start_delimiter.clone(),
                end_delimiter: end_delimiter.clone(),
                max_buffered_bytes: *max_buffered_bytes,
            },
        }
    }
}

fn default_execution_profile() -> String {
    "default".to_owned()
}

const fn default_true() -> bool {
    true
}

const fn default_max_rendered_bytes() -> usize {
    4 * 1024 * 1024
}

const fn default_max_tool_call_bytes() -> usize {
    64 * 1024
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    Sglang,
    Vllm,
}

impl EngineKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSettings {
    pub id: String,
    #[serde(default = "default_generation")]
    pub generation: u64,
    pub kind: EngineKind,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Optional explicit upstream-name to semantic-profile mappings. When this
    /// is empty, every public catalog alias is a discovery candidate.
    #[serde(default)]
    pub model_mappings: Vec<EngineModelSettings>,
    /// Legacy single-model fields retained for configuration compatibility.
    #[serde(default)]
    pub served_model: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    pub runtime_version: String,
    #[serde(default = "default_adapter_version")]
    pub adapter_version: String,
    #[serde(default = "default_topology")]
    pub topology: String,
    #[serde(default = "default_hardware")]
    pub hardware: String,
    #[serde(default)]
    pub health_endpoint: Option<String>,
    #[serde(default)]
    pub target_id: Option<String>,
    #[serde(default = "default_residency")]
    pub residency: String,
    #[serde(default = "default_capability_revision")]
    pub capability_revision: String,
    #[serde(default = "default_parallel_degree")]
    pub tensor_parallel: u16,
    #[serde(default = "default_parallel_degree")]
    pub pipeline_parallel: u16,
    #[serde(default = "default_parallel_degree")]
    pub expert_parallel: u16,
    #[serde(default = "default_layout_revision")]
    pub layout_revision: String,
    #[serde(default)]
    pub telemetry: EngineTelemetrySettings,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineTelemetrySettings {
    pub metrics_path: String,
    pub request_timeout_millis: u64,
    pub min_scrape_interval_millis: u64,
    pub valid_for_millis: u64,
    pub max_response_bytes: usize,
    pub max_samples: usize,
    pub require_fresh_metrics: bool,
}

impl Default for EngineTelemetrySettings {
    fn default() -> Self {
        let defaults = RemoteTelemetryConfig::default();
        Self {
            metrics_path: defaults.metrics_path,
            request_timeout_millis: defaults.request_timeout_millis,
            min_scrape_interval_millis: defaults.min_scrape_interval_millis,
            valid_for_millis: defaults.valid_for_millis,
            max_response_bytes: defaults.max_response_bytes,
            max_samples: defaults.max_samples,
            require_fresh_metrics: defaults.require_fresh_metrics,
        }
    }
}

impl EngineTelemetrySettings {
    fn remote_config(&self) -> RemoteTelemetryConfig {
        RemoteTelemetryConfig {
            metrics_path: self.metrics_path.clone(),
            request_timeout_millis: self.request_timeout_millis,
            min_scrape_interval_millis: self.min_scrape_interval_millis,
            valid_for_millis: self.valid_for_millis,
            max_response_bytes: self.max_response_bytes,
            max_samples: self.max_samples,
            require_fresh_metrics: self.require_fresh_metrics,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineModelSettings {
    pub upstream_model: String,
    pub profile: String,
    #[serde(default)]
    pub target_id: Option<String>,
}

struct ResolvedEngineModel {
    upstream_model: String,
    profile: String,
    target_id: String,
}

const fn default_generation() -> u64 {
    1
}

fn default_adapter_version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

fn default_topology() -> String {
    "unspecified".to_owned()
}

fn default_hardware() -> String {
    "unspecified".to_owned()
}

fn default_residency() -> String {
    "resident".to_owned()
}

fn default_capability_revision() -> String {
    "v1".to_owned()
}

const fn default_parallel_degree() -> u16 {
    1
}

fn default_layout_revision() -> String {
    "v1".to_owned()
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementModeSettings {
    #[default]
    Shadow,
    Active,
}

impl PlacementModeSettings {
    fn runtime_mode(self) -> PlacementMode {
        match self {
            Self::Shadow => PlacementMode::Shadow,
            Self::Active => PlacementMode::Active,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlacementSettings {
    pub mode: PlacementModeSettings,
    pub state_path: Option<PathBuf>,
    pub active_confirmation: Option<String>,
    pub calibration: CalibrationSettings,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CalibrationSettings {
    pub ewma_alpha_bps: u32,
    pub min_samples_per_metric: u64,
    pub max_mape_bps: u64,
    pub min_shadow_decisions: u64,
    pub min_shadow_agreement_bps: u64,
    pub conservative_queue_micros: u64,
    pub conservative_queue_micros_per_waiting_request: u64,
    pub conservative_prefill_micros_per_token: u64,
    pub conservative_decode_micros_per_token: u64,
    pub conservative_materialization_ratio_bps: u64,
    pub conservative_topology_micros: u64,
    pub max_unit_micros: u64,
    pub max_materialization_ratio_bps: u64,
    pub persistence_flush_every_updates: u64,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self::from(CalibrationPolicy::default())
    }
}

impl From<CalibrationPolicy> for CalibrationSettings {
    fn from(policy: CalibrationPolicy) -> Self {
        Self {
            ewma_alpha_bps: policy.ewma_alpha_bps,
            min_samples_per_metric: policy.min_samples_per_metric,
            max_mape_bps: policy.max_mape_bps,
            min_shadow_decisions: policy.min_shadow_decisions,
            min_shadow_agreement_bps: policy.min_shadow_agreement_bps,
            conservative_queue_micros: policy.conservative_queue_micros,
            conservative_queue_micros_per_waiting_request: policy
                .conservative_queue_micros_per_waiting_request,
            conservative_prefill_micros_per_token: policy.conservative_prefill_micros_per_token,
            conservative_decode_micros_per_token: policy.conservative_decode_micros_per_token,
            conservative_materialization_ratio_bps: policy.conservative_materialization_ratio_bps,
            conservative_topology_micros: policy.conservative_topology_micros,
            max_unit_micros: policy.max_unit_micros,
            max_materialization_ratio_bps: policy.max_materialization_ratio_bps,
            persistence_flush_every_updates: policy.persistence_flush_every_updates,
        }
    }
}

impl CalibrationSettings {
    fn policy(&self) -> CalibrationPolicy {
        CalibrationPolicy {
            ewma_alpha_bps: self.ewma_alpha_bps,
            min_samples_per_metric: self.min_samples_per_metric,
            max_mape_bps: self.max_mape_bps,
            min_shadow_decisions: self.min_shadow_decisions,
            min_shadow_agreement_bps: self.min_shadow_agreement_bps,
            conservative_queue_micros: self.conservative_queue_micros,
            conservative_queue_micros_per_waiting_request: self
                .conservative_queue_micros_per_waiting_request,
            conservative_prefill_micros_per_token: self.conservative_prefill_micros_per_token,
            conservative_decode_micros_per_token: self.conservative_decode_micros_per_token,
            conservative_materialization_ratio_bps: self.conservative_materialization_ratio_bps,
            conservative_topology_micros: self.conservative_topology_micros,
            max_unit_micros: self.max_unit_micros,
            max_materialization_ratio_bps: self.max_materialization_ratio_bps,
            persistence_flush_every_updates: self.persistence_flush_every_updates,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum StateSettings {
    #[default]
    Disabled,
    Nexuskv {
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
        tenant: String,
        namespace: String,
        engine_family: String,
        semantic_type: String,
    },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservabilitySettings {
    #[serde(default)]
    pub json: bool,
    #[serde(default = "default_log_filter")]
    pub filter: String,
}

impl Default for ObservabilitySettings {
    fn default() -> Self {
        Self {
            json: false,
            filter: default_log_filter(),
        }
    }
}

fn default_log_filter() -> String {
    "locus=info,tower_http=info".to_owned()
}

pub struct ConfiguredServer {
    pub listen: SocketAddr,
    pub app: Router,
    pub observability: ObservabilitySettings,
}

pub fn load_config(path: &Path) -> Result<ServerConfig, ServerError> {
    let bytes = fs::read(path).map_err(|source| ServerError::ReadConfig {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ServerError::ParseConfig {
        path: path.to_path_buf(),
        source,
    })
}

pub fn build_server(
    config: ServerConfig,
    config_directory: &Path,
) -> Result<ConfiguredServer, ServerError> {
    if config.models.is_empty() {
        return Err(ServerError::InvalidConfig(
            "at least one model must be configured".to_owned(),
        ));
    }
    if config.engines.is_empty() {
        return Err(ServerError::InvalidConfig(
            "at least one engine must be configured".to_owned(),
        ));
    }
    let models = ModelRegistry::new();
    let mut model_by_alias = BTreeMap::new();
    for model in &config.models {
        let execution_identity = ModelExecutionIdentity {
            model_revision: required("models.model_revision", &model.model_revision)?,
            adapter_revision: model.adapter_revision.clone(),
            execution_profile: required("models.execution_profile", &model.execution_profile)?,
        };
        let mut spec = HuggingFaceProfileSpec::new(
            model.aliases.clone(),
            execution_identity.clone(),
            resolve_path(config_directory, &model.tokenizer_json),
            resolve_path(config_directory, &model.chat_template),
        );
        spec.tokenizer_revision = model.tokenizer_revision.clone();
        spec.template_revision = model.template_revision.clone();
        spec.template_context.clone_from(&model.template_context);
        spec.add_generation_prompt = model.add_generation_prompt;
        spec.max_rendered_bytes = model.max_rendered_bytes;
        spec.reasoning_parser = model
            .reasoning_parser
            .as_ref()
            .map(ReasoningParserSettings::profile_spec);
        spec.tool_parser = model
            .tool_parser
            .as_ref()
            .map(ToolParserSettings::profile_spec);
        models.register(load_huggingface_semantics(spec)?)?;
        for alias in &model.aliases {
            model_by_alias.insert(alias.clone(), execution_identity.clone());
        }
    }

    let required_models = config
        .required_models
        .iter()
        .map(|alias| required("required_models", alias))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for alias in &required_models {
        if !model_by_alias.contains_key(alias) {
            return Err(ServerError::InvalidConfig(format!(
                "required model references unknown profile alias: {alias}"
            )));
        }
    }

    let engines = EngineRegistry::new();
    let mut engine_ids = BTreeSet::new();
    let mut target_ids = BTreeSet::new();
    for engine in &config.engines {
        if !engine_ids.insert(engine.id.clone()) {
            return Err(ServerError::InvalidConfig(format!(
                "duplicate engine id: {}",
                engine.id
            )));
        }
        if engine.generation == 0 {
            return Err(ServerError::InvalidConfig(format!(
                "engine {} generation must be greater than zero",
                engine.id
            )));
        }
        let engine_ref = EngineInstanceRef {
            id: EngineInstanceId::new(required("engines.id", &engine.id)?),
            generation: engine.generation,
        };
        let instance = EngineInstance {
            reference: engine_ref.clone(),
            runtime: RuntimeIdentity {
                kind: engine.kind.as_str().to_owned(),
                runtime_version: required("engines.runtime_version", &engine.runtime_version)?,
                adapter_version: required("engines.adapter_version", &engine.adapter_version)?,
            },
            topology: required("engines.topology", &engine.topology)?,
            hardware: required("engines.hardware", &engine.hardware)?,
            health_endpoint: engine.health_endpoint.clone(),
        };
        let bindings = resolve_engine_models(engine, &model_by_alias)?;
        let mut targets = Vec::with_capacity(bindings.len());
        for binding in bindings {
            if !target_ids.insert(binding.target_id.clone()) {
                return Err(ServerError::InvalidConfig(format!(
                    "duplicate execution target id: {}",
                    binding.target_id
                )));
            }
            let model = model_by_alias
                .get(&binding.profile)
                .cloned()
                .ok_or_else(|| {
                    ServerError::InvalidConfig(format!(
                        "engine {} references unknown model profile {}",
                        engine.id, binding.profile
                    ))
                })?;
            targets.push(RemoteExecutionTarget {
                served_model: binding.upstream_model,
                target: ExecutionTarget {
                    id: ExecutionTargetId::new(binding.target_id),
                    engine: engine_ref.clone(),
                    model,
                    role: ExecutionRole::Combined,
                    parallel_layout: ParallelLayout {
                        tensor_parallel: nonzero(
                            "engines.tensor_parallel",
                            engine.tensor_parallel,
                        )?,
                        pipeline_parallel: nonzero(
                            "engines.pipeline_parallel",
                            engine.pipeline_parallel,
                        )?,
                        expert_parallel: nonzero(
                            "engines.expert_parallel",
                            engine.expert_parallel,
                        )?,
                        layout_revision: required(
                            "engines.layout_revision",
                            &engine.layout_revision,
                        )?,
                    },
                    residency: required("engines.residency", &engine.residency)?,
                    capability_revision: required(
                        "engines.capability_revision",
                        &engine.capability_revision,
                    )?,
                },
            });
        }
        let remote = RemoteEngineConfig {
            base_url: required("engines.base_url", &engine.base_url)?,
            api_key: resolve_optional_secret(engine.api_key_env.as_deref())?,
            instance,
            targets,
            telemetry: engine.telemetry.remote_config(),
        };
        let adapter: Arc<dyn EngineAdapter> = match engine.kind {
            EngineKind::Sglang => Arc::new(SglangEngineAdapter::new(remote)?),
            EngineKind::Vllm => Arc::new(VllmEngineAdapter::new(remote)?),
        };
        engines.register(adapter)?;
    }

    let state_provider: Arc<dyn StateProvider> = match &config.state {
        StateSettings::Disabled => Arc::new(NullStateProvider::default()),
        StateSettings::Nexuskv {
            base_url,
            api_key_env,
            tenant,
            namespace,
            engine_family,
            semantic_type,
        } => Arc::new(NexusKvStateProvider::new(NexusKvBridgeConfig {
            base_url: required("state.base_url", base_url)?,
            api_key: resolve_optional_secret(api_key_env.as_deref())?,
            tenant: required("state.tenant", tenant)?,
            namespace: required("state.namespace", namespace)?,
            engine_family: required("state.engine_family", engine_family)?,
            semantic_type: required("state.semantic_type", semantic_type)?,
        })?),
    };
    let calibration_path = config
        .placement
        .state_path
        .as_deref()
        .map(|path| resolve_path(config_directory, path));
    let calibrator =
        PersistentCalibrator::load(config.placement.calibration.policy(), calibration_path)?;
    let placement = PlacementControl::new(
        config.placement.mode.runtime_mode(),
        calibrator,
        config.placement.active_confirmation.as_deref(),
    )?;
    let service: Arc<dyn InferenceService> = Arc::new(
        DefaultInferenceService::new(models, engines, state_provider)
            .with_required_models(required_models)
            .with_placement_control(placement),
    );
    let api = ApiConfig {
        bearer_token: resolve_optional_secret(config.api.bearer_token_env.as_deref())?,
        max_request_bytes: config.api.max_request_bytes,
        max_concurrent_requests: config.api.max_concurrent_requests,
    };
    let request_id_header = HeaderName::from_static(REQUEST_ID_HEADER);
    let app = router_with_config(service, api)?
        .layer(CatchPanicLayer::new())
        .layer(PropagateRequestIdLayer::new(request_id_header.clone()))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<Body>| {
                    let request_id = request
                        .headers()
                        .get(REQUEST_ID_HEADER)
                        .and_then(|value| value.to_str().ok())
                        .unwrap_or("invalid");
                    info_span!(
                        "http.request",
                        method = %request.method(),
                        uri = %request.uri(),
                        request_id,
                    )
                })
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(SetRequestIdLayer::new(
            request_id_header,
            LocusRequestId::default(),
        ));
    Ok(ConfiguredServer {
        listen: config.listen,
        app,
        observability: config.observability,
    })
}

fn resolve_engine_models(
    engine: &EngineSettings,
    model_by_alias: &BTreeMap<String, ModelExecutionIdentity>,
) -> Result<Vec<ResolvedEngineModel>, ServerError> {
    let legacy_fields = [
        engine.served_model.is_some(),
        engine.model.is_some(),
        engine.target_id.is_some(),
    ];
    let legacy_count = legacy_fields.into_iter().filter(|present| *present).count();
    if !engine.model_mappings.is_empty() && legacy_count > 0 {
        return Err(ServerError::InvalidConfig(format!(
            "engine {} cannot combine model_mappings with legacy served_model/model/target_id fields",
            engine.id
        )));
    }

    let candidates = if !engine.model_mappings.is_empty() {
        engine
            .model_mappings
            .iter()
            .map(|mapping| {
                let upstream_model = required(
                    "engines.model_mappings.upstream_model",
                    &mapping.upstream_model,
                )?;
                let profile = required("engines.model_mappings.profile", &mapping.profile)?;
                let target_id = mapping
                    .target_id
                    .as_deref()
                    .map(|value| required("engines.model_mappings.target_id", value))
                    .transpose()?
                    .unwrap_or_else(|| format!("{}/{upstream_model}", engine.id));
                Ok(ResolvedEngineModel {
                    upstream_model,
                    profile,
                    target_id,
                })
            })
            .collect::<Result<Vec<_>, ServerError>>()?
    } else if legacy_count > 0 {
        if legacy_count != legacy_fields.len() {
            return Err(ServerError::InvalidConfig(format!(
                "engine {} legacy served_model, model, and target_id fields must be configured together",
                engine.id
            )));
        }
        vec![ResolvedEngineModel {
            upstream_model: required(
                "engines.served_model",
                engine.served_model.as_deref().unwrap_or_default(),
            )?,
            profile: required("engines.model", engine.model.as_deref().unwrap_or_default())?,
            target_id: required(
                "engines.target_id",
                engine.target_id.as_deref().unwrap_or_default(),
            )?,
        }]
    } else {
        model_by_alias
            .keys()
            .map(|alias| ResolvedEngineModel {
                upstream_model: alias.clone(),
                profile: alias.clone(),
                target_id: format!("{}/{alias}", engine.id),
            })
            .collect()
    };

    let mut upstream_models = BTreeSet::new();
    for candidate in &candidates {
        if !upstream_models.insert(candidate.upstream_model.clone()) {
            return Err(ServerError::InvalidConfig(format!(
                "engine {} has duplicate upstream model mapping: {}",
                engine.id, candidate.upstream_model
            )));
        }
    }
    Ok(candidates)
}

fn required(field: &'static str, value: &str) -> Result<String, ServerError> {
    if value.trim().is_empty() {
        Err(ServerError::InvalidConfig(format!(
            "{field} must not be empty"
        )))
    } else {
        Ok(value.to_owned())
    }
}

fn nonzero(field: &'static str, value: u16) -> Result<u16, ServerError> {
    if value == 0 {
        Err(ServerError::InvalidConfig(format!(
            "{field} must be greater than zero"
        )))
    } else {
        Ok(value)
    }
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn resolve_optional_secret(variable: Option<&str>) -> Result<Option<String>, ServerError> {
    variable.map(resolve_secret).transpose()
}

fn resolve_secret(variable: &str) -> Result<String, ServerError> {
    if variable.trim().is_empty() {
        return Err(ServerError::InvalidConfig(
            "secret environment variable name must not be empty".to_owned(),
        ));
    }
    match env::var(variable) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) => Err(ServerError::MissingSecret(variable.to_owned())),
        Err(env::VarError::NotPresent) => Err(ServerError::MissingSecret(variable.to_owned())),
        Err(env::VarError::NotUnicode(_)) => Err(ServerError::InvalidSecret(variable.to_owned())),
    }
}

#[derive(Clone, Default)]
struct LocusRequestId {
    next: Arc<AtomicU64>,
}

impl MakeRequestId for LocusRequestId {
    fn make_request_id<B>(&mut self, _request: &Request<B>) -> Option<RequestId> {
        let value = format!(
            "locus-http-{:016x}",
            self.next.fetch_add(1, Ordering::Relaxed)
        );
        HeaderValue::from_str(&value).ok().map(RequestId::new)
    }
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("failed to read server config {path}: {source}")]
    ReadConfig {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse server config {path}: {source}")]
    ParseConfig {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid server config: {0}")]
    InvalidConfig(String),
    #[error("required secret environment variable is missing or empty: {0}")]
    MissingSecret(String),
    #[error("secret environment variable is not valid Unicode: {0}")]
    InvalidSecret(String),
    #[error(transparent)]
    Semantics(#[from] locus_semantics_hf::HuggingFaceSemanticsError),
    #[error(transparent)]
    SemanticRegistry(#[from] locus_semantics::SemanticError),
    #[error(transparent)]
    Engine(#[from] locus_engine::EngineError),
    #[error(transparent)]
    State(#[from] locus_state::StateError),
    #[error(transparent)]
    Api(#[from] locus_openai::ApiConfigError),
    #[error(transparent)]
    Calibration(#[from] locus_planner::CalibrationError),
    #[error(transparent)]
    Placement(#[from] locus_runtime::PlacementConfigurationError),
}
