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
use locus_engine_openai::{RemoteEngineConfig, SglangEngineAdapter, VllmEngineAdapter};
use locus_openai::{ApiConfig, router_with_config};
use locus_runtime::{DefaultInferenceService, InferenceService};
use locus_semantics::ModelRegistry;
use locus_semantics_hf::{HuggingFaceProfileSpec, load_huggingface_semantics};
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
    pub engines: Vec<EngineSettings>,
    #[serde(default)]
    pub state: StateSettings,
    #[serde(default)]
    pub observability: ObservabilitySettings,
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
    pub served_model: String,
    pub model: String,
    pub runtime_version: String,
    #[serde(default = "default_adapter_version")]
    pub adapter_version: String,
    #[serde(default = "default_topology")]
    pub topology: String,
    #[serde(default = "default_hardware")]
    pub hardware: String,
    #[serde(default)]
    pub health_endpoint: Option<String>,
    pub target_id: String,
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
        models.register(load_huggingface_semantics(spec)?)?;
        for alias in &model.aliases {
            model_by_alias.insert(alias.clone(), execution_identity.clone());
        }
    }

    let engines = EngineRegistry::new();
    let mut models_with_engines = BTreeSet::new();
    let mut engine_ids = BTreeSet::new();
    let mut target_ids = BTreeSet::new();
    for engine in &config.engines {
        if !engine_ids.insert(engine.id.clone()) {
            return Err(ServerError::InvalidConfig(format!(
                "duplicate engine id: {}",
                engine.id
            )));
        }
        if !target_ids.insert(engine.target_id.clone()) {
            return Err(ServerError::InvalidConfig(format!(
                "duplicate execution target id: {}",
                engine.target_id
            )));
        }
        if engine.generation == 0 {
            return Err(ServerError::InvalidConfig(format!(
                "engine {} generation must be greater than zero",
                engine.id
            )));
        }
        let model = model_by_alias.get(&engine.model).cloned().ok_or_else(|| {
            ServerError::InvalidConfig(format!(
                "engine {} references unknown model alias {}",
                engine.id, engine.model
            ))
        })?;
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
        let target = ExecutionTarget {
            id: ExecutionTargetId::new(required("engines.target_id", &engine.target_id)?),
            engine: engine_ref,
            model: model.clone(),
            role: ExecutionRole::Combined,
            parallel_layout: ParallelLayout {
                tensor_parallel: nonzero("engines.tensor_parallel", engine.tensor_parallel)?,
                pipeline_parallel: nonzero("engines.pipeline_parallel", engine.pipeline_parallel)?,
                expert_parallel: nonzero("engines.expert_parallel", engine.expert_parallel)?,
                layout_revision: required("engines.layout_revision", &engine.layout_revision)?,
            },
            residency: required("engines.residency", &engine.residency)?,
            capability_revision: required(
                "engines.capability_revision",
                &engine.capability_revision,
            )?,
        };
        let remote = RemoteEngineConfig {
            base_url: required("engines.base_url", &engine.base_url)?,
            api_key: resolve_optional_secret(engine.api_key_env.as_deref())?,
            served_model: required("engines.served_model", &engine.served_model)?,
            instance,
            target,
        };
        let adapter: Arc<dyn EngineAdapter> = match engine.kind {
            EngineKind::Sglang => Arc::new(SglangEngineAdapter::new(remote)?),
            EngineKind::Vllm => Arc::new(VllmEngineAdapter::new(remote)?),
        };
        engines.register(adapter)?;
        models_with_engines.insert(model);
    }
    let uncovered = config
        .models
        .iter()
        .filter_map(|model| model.aliases.first())
        .filter(|alias| {
            model_by_alias
                .get(*alias)
                .is_some_and(|identity| !models_with_engines.contains(identity))
        })
        .cloned()
        .collect::<Vec<_>>();
    if !uncovered.is_empty() {
        return Err(ServerError::InvalidConfig(format!(
            "models without a configured engine: {}",
            uncovered.join(", ")
        )));
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
    let service: Arc<dyn InferenceService> = Arc::new(DefaultInferenceService::new(
        models,
        engines,
        state_provider,
    ));
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
}
