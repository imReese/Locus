use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{
    DefaultBodyLimit, Extension, FromRef, Request, State, rejection::JsonRejection,
};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use locus_core::{OperationContext, RequestId, SamplingParameters, Usage};
use locus_http::{CredentialIndex, CredentialIndexError, TransportMetrics};
use locus_model_io::{
    Conversation, ConversationMessage, ConversationRole, ModelEvent, ModelFinishReason, ModelInput,
    ModelIoError, ModelRequest, PromptInput, ReasoningEffort, ResponseFormat, ToolChoice,
    ToolDefinition,
};
use locus_runtime::{
    AdmissionError, InferenceError, InferenceService, ModelEventStream, TrafficController,
};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

pub use locus_http::TenantCredential;

const DEFAULT_MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const REQUEST_TIMEOUT_HEADER: &str = "x-request-timeout-ms";

#[derive(Clone)]
pub struct ApiConfig {
    pub bearer_token: Option<String>,
    pub tenant_credentials: Vec<TenantCredential>,
    pub anonymous_tenant: Option<String>,
    pub traffic: TrafficController,
    pub transport_metrics: TransportMetrics,
    pub max_request_bytes: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bearer_token: None,
            tenant_credentials: Vec::new(),
            anonymous_tenant: None,
            traffic: TrafficController::default(),
            transport_metrics: TransportMetrics::default(),
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
        }
    }
}

#[derive(Clone)]
struct AuthState {
    credentials: CredentialIndex,
    anonymous_tenant: Option<Arc<str>>,
    traffic: TrafficController,
    next_request: Arc<AtomicU64>,
    max_request_bytes: usize,
}

#[derive(Clone)]
struct TrustedRequest {
    response_id: String,
    context: OperationContext,
}

#[derive(Clone)]
struct ApiState {
    service: Arc<dyn InferenceService>,
    traffic: TrafficController,
    transport_metrics: TransportMetrics,
}

impl FromRef<ApiState> for Arc<dyn InferenceService> {
    fn from_ref(state: &ApiState) -> Self {
        Arc::clone(&state.service)
    }
}

pub fn router(service: Arc<dyn InferenceService>) -> Router {
    router_with_config(service, ApiConfig::default()).expect("default API config must be valid")
}

pub fn router_with_config(
    service: Arc<dyn InferenceService>,
    config: ApiConfig,
) -> Result<Router, ApiConfigError> {
    if config.max_request_bytes == 0 {
        return Err(ApiConfigError::ZeroRequestBodyLimit);
    }
    if config
        .bearer_token
        .as_ref()
        .is_some_and(|token| token.is_empty())
    {
        return Err(ApiConfigError::EmptyBearerToken);
    }
    if config.bearer_token.is_some() && !config.tenant_credentials.is_empty() {
        return Err(ApiConfigError::AmbiguousCredentials);
    }
    let mut credentials = config.tenant_credentials;
    if let Some(bearer_token) = config.bearer_token {
        credentials.push(TenantCredential {
            tenant_id: "default".to_owned(),
            bearer_token,
        });
    }
    for credential in &credentials {
        if credential.bearer_token.is_empty() {
            return Err(ApiConfigError::EmptyBearerToken);
        }
        if !config.traffic.tenant_exists(&credential.tenant_id) {
            return Err(ApiConfigError::UnknownTenant(credential.tenant_id.clone()));
        }
    }
    let mut anonymous_tenant = config.anonymous_tenant;
    if credentials.is_empty() && anonymous_tenant.is_none() {
        anonymous_tenant = Some("default".to_owned());
    }
    if let Some(tenant) = &anonymous_tenant
        && !config.traffic.tenant_exists(tenant)
    {
        return Err(ApiConfigError::UnknownTenant(tenant.clone()));
    }
    let credentials = CredentialIndex::new(credentials).map_err(|error| match error {
        CredentialIndexError::EmptyCredential => ApiConfigError::EmptyBearerToken,
        CredentialIndexError::DuplicateCredential => ApiConfigError::DuplicateCredential,
    })?;
    let next_request = Arc::new(AtomicU64::new(1));
    let state = ApiState {
        service,
        traffic: config.traffic.clone(),
        transport_metrics: config.transport_metrics,
    };
    let mut api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/responses", post(responses))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions));
    api = api
        .layer(DefaultBodyLimit::max(config.max_request_bytes))
        .layer(from_fn_with_state(
            AuthState {
                credentials,
                anonymous_tenant: anonymous_tenant.map(Into::into),
                traffic: config.traffic,
                next_request,
                max_request_bytes: config.max_request_bytes,
            },
            authenticate,
        ));
    Ok(Router::new()
        .route("/healthz", get(health))
        .route("/readyz", get(readiness))
        .route("/metrics", get(metrics))
        .merge(api)
        .with_state(state))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn readiness(State(service): State<Arc<dyn InferenceService>>) -> Response {
    match service.readiness().await {
        Ok(report) if !report.traffic_draining => Json(json!({
            "status": "ready",
            "model_profiles": report.model_profiles,
            "routable_models": report.routable_models,
            "required_models": report.required_models,
            "ready_targets": report.ready_targets,
            "observed_targets": report.observed_targets,
            "placement_mode": match report.placement_mode {
                locus_planner::PlacementMode::Shadow => "shadow",
                locus_planner::PlacementMode::Active => "active",
            },
            "calibration_revision": report.calibration_revision,
            "calibration_persistent": report.calibration_persistent,
            "calibration_persistence_healthy": report.calibration_persistence_healthy,
            "traffic_draining": false,
        }))
        .into_response(),
        Ok(report) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "error": "traffic controller is draining",
                "traffic_draining": report.traffic_draining,
            })),
        )
            .into_response(),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "not_ready", "error": error.to_string()})),
        )
            .into_response(),
    }
}

async fn authenticate(State(auth): State<AuthState>, mut request: Request, next: Next) -> Response {
    let candidate = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let tenant = if let Some(candidate) = candidate {
        auth.credentials.authenticate(candidate)
    } else {
        auth.anonymous_tenant.clone()
    };
    if let Some(tenant) = tenant {
        let requested_timeout = match requested_timeout(request.headers()) {
            Ok(timeout) => timeout,
            Err(error) => return error.into_response(),
        };
        let prefix = request_id_prefix(request.uri().path());
        let response_id = format!(
            "{prefix}_{:016x}",
            auth.next_request.fetch_add(1, Ordering::AcqRel)
        );
        let context = match auth.traffic.operation_context(
            RequestId::new(response_id.clone()),
            tenant.as_ref(),
            requested_timeout,
        ) {
            Ok(context) => context,
            Err(error) => return api_error_from_admission(&error).into_response(),
        };
        if request.method() != Method::GET && request.method() != Method::HEAD {
            let (parts, body) = request.into_parts();
            let body = match context
                .run(read_bounded_body(body, auth.max_request_bytes))
                .await
            {
                Ok(Ok(body)) => body,
                Ok(Err(RequestBodyReadError::TooLarge)) => {
                    return ApiError::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "request body exceeded the configured byte limit",
                        "invalid_request_error",
                        None,
                        Some("request_too_large"),
                    )
                    .into_response();
                }
                Ok(Err(RequestBodyReadError::Transport(error))) => {
                    return ApiError::new(
                        StatusCode::BAD_REQUEST,
                        error,
                        "invalid_request_error",
                        None,
                        Some("invalid_request_body"),
                    )
                    .into_response();
                }
                Err(error) => {
                    return api_error_from_inference(&InferenceError::Context(error))
                        .into_response();
                }
            };
            request = Request::from_parts(parts, Body::from(body));
        }
        request.extensions_mut().insert(TrustedRequest {
            response_id,
            context: context.clone(),
        });
        if request.method() == Method::GET || request.method() == Method::HEAD {
            return match context.run(next.run(request)).await {
                Ok(response) => response,
                Err(error) => {
                    api_error_from_inference(&InferenceError::Context(error)).into_response()
                }
            };
        }
        return next.run(request).await;
    }
    let mut response = ApiError::new(
        StatusCode::UNAUTHORIZED,
        "missing or invalid bearer token",
        "authentication_error",
        None,
        Some("invalid_api_key"),
    )
    .into_response();
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

fn request_id_prefix(path: &str) -> &'static str {
    match path {
        "/v1/responses" => "resp",
        "/v1/completions" => "cmpl",
        "/v1/chat/completions" => "chatcmpl",
        "/v1/models" => "models",
        _ => "req",
    }
}

fn requested_timeout(headers: &axum::http::HeaderMap) -> Result<Option<Duration>, ApiError> {
    headers
        .get(REQUEST_TIMEOUT_HEADER)
        .map(|value| {
            let value = value.to_str().map_err(|_| {
                ApiError::invalid(
                    REQUEST_TIMEOUT_HEADER,
                    "request timeout header must be ASCII milliseconds",
                )
            })?;
            let millis = value.parse::<u64>().map_err(|_| {
                ApiError::invalid(
                    REQUEST_TIMEOUT_HEADER,
                    "request timeout header must be an integer number of milliseconds",
                )
            })?;
            if millis == 0 {
                return Err(ApiError::invalid(
                    REQUEST_TIMEOUT_HEADER,
                    "request timeout must be greater than zero",
                ));
            }
            Ok(Duration::from_millis(millis))
        })
        .transpose()
}

enum RequestBodyReadError {
    TooLarge,
    Transport(String),
}

async fn read_bounded_body(body: Body, max_bytes: usize) -> Result<Vec<u8>, RequestBodyReadError> {
    let mut stream = body.into_data_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| RequestBodyReadError::Transport(error.to_string()))?;
        if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
            return Err(RequestBodyReadError::TooLarge);
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[derive(Debug, Error)]
pub enum ApiConfigError {
    #[error("API bearer token must not be empty")]
    EmptyBearerToken,
    #[error("API request body limit must be greater than zero")]
    ZeroRequestBodyLimit,
    #[error("legacy bearer_token and tenant_credentials cannot be configured together")]
    AmbiguousCredentials,
    #[error("tenant credential references unknown traffic policy: {0}")]
    UnknownTenant(String),
    #[error("the same bearer credential cannot authorize multiple tenant mappings")]
    DuplicateCredential,
}

async fn metrics(State(state): State<ApiState>) -> Response {
    match state.traffic.prometheus() {
        Ok(mut body) => {
            body.push_str(&state.transport_metrics.prometheus());
            (
                [(
                    header::CONTENT_TYPE,
                    "text/plain; version=0.0.4; charset=utf-8",
                )],
                body,
            )
                .into_response()
        }
        Err(error) => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            error.to_string(),
            "server_error",
            None,
            Some("metrics_unavailable"),
        )
        .into_response(),
    }
}

async fn models(State(service): State<Arc<dyn InferenceService>>) -> Response {
    match service.models().await {
        Ok(profiles) => {
            let data = profiles
                .into_iter()
                .flat_map(|profile| profile.public_aliases)
                .map(|alias| {
                    json!({
                        "id": alias,
                        "object": "model",
                        "created": 0,
                        "owned_by": "locus"
                    })
                })
                .collect::<Vec<_>>();
            Json(json!({"object": "list", "data": data})).into_response()
        }
        Err(error) => api_error_from_inference(&error).into_response(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponsesRequest {
    model: String,
    input: ResponsesInput,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<ResponseTool>,
    #[serde(default)]
    tool_choice: Option<ToolChoiceDto>,
    #[serde(default)]
    text: Option<ResponseTextConfig>,
    #[serde(default)]
    reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    Text(String),
    Items(Vec<ResponseInputMessage>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseInputMessage {
    role: String,
    content: ResponseInputContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponseInputContent {
    Text(String),
    Parts(Vec<ResponseInputPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseInputPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ResponseTool {
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        parameters: Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolChoiceDto {
    Mode(String),
    Function {
        #[serde(rename = "type")]
        kind: String,
        name: String,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseTextConfig {
    #[serde(default)]
    format: Option<ResponseFormatDto>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ResponseFormatDto {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        #[serde(default)]
        description: Option<String>,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReasoningConfig {
    #[serde(default)]
    effort: Option<String>,
    #[serde(default)]
    summary: Option<String>,
}

impl ResponsesRequest {
    fn into_semantic(self) -> Result<ModelRequest, ApiError> {
        if self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.summary.as_ref())
            .is_some()
        {
            return Err(ApiError::unsupported("reasoning.summary"));
        }
        let mut messages = Vec::new();
        if let Some(instructions) = self.instructions {
            messages.push(ConversationMessage {
                role: ConversationRole::Developer,
                content: instructions,
                tool_call_id: None,
            });
        }
        match self.input {
            ResponsesInput::Text(content) => messages.push(ConversationMessage {
                role: ConversationRole::User,
                content,
                tool_call_id: None,
            }),
            ResponsesInput::Items(items) => {
                for item in items {
                    messages.push(ConversationMessage {
                        role: parse_role(&item.role)?,
                        content: flatten_response_content(item.content),
                        tool_call_id: None,
                    });
                }
            }
        }
        Ok(ModelRequest {
            model: self.model,
            input: ModelInput::Conversation(Conversation { messages }),
            sampling: SamplingParameters {
                max_output_tokens: self.max_output_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                seed: None,
                stop_sequences: Vec::new(),
            },
            tools: self
                .tools
                .into_iter()
                .map(response_tool)
                .collect::<Vec<_>>(),
            tool_choice: parse_tool_choice(self.tool_choice)?,
            response_format: parse_response_format(self.text.and_then(|text| text.format)),
            reasoning_effort: parse_reasoning_effort(
                self.reasoning.and_then(|reasoning| reasoning.effort),
            )?,
            metadata: self.metadata,
        })
    }
}

fn flatten_response_content(content: ResponseInputContent) -> String {
    match content {
        ResponseInputContent::Text(text) => text,
        ResponseInputContent::Parts(parts) => parts
            .into_iter()
            .map(|part| match part {
                ResponseInputPart::InputText { text } => text,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn response_tool(tool: ResponseTool) -> ToolDefinition {
    match tool {
        ResponseTool::Function {
            name,
            description,
            parameters,
            strict,
        } => ToolDefinition {
            name,
            description,
            parameters_schema: parameters.to_string(),
            strict,
        },
    }
}

fn parse_tool_choice(choice: Option<ToolChoiceDto>) -> Result<ToolChoice, ApiError> {
    match choice {
        None => Ok(ToolChoice::Auto),
        Some(ToolChoiceDto::Mode(mode)) if mode == "auto" => Ok(ToolChoice::Auto),
        Some(ToolChoiceDto::Mode(mode)) if mode == "none" => Ok(ToolChoice::None),
        Some(ToolChoiceDto::Mode(mode)) if mode == "required" => Ok(ToolChoice::Required),
        Some(ToolChoiceDto::Function { kind, name }) if kind == "function" => {
            Ok(ToolChoice::Function(name))
        }
        Some(_) => Err(ApiError::invalid(
            "tool_choice",
            "tool_choice must be auto, none, required, or a named function",
        )),
    }
}

fn parse_response_format(format: Option<ResponseFormatDto>) -> ResponseFormat {
    match format {
        None | Some(ResponseFormatDto::Text) => ResponseFormat::Text,
        Some(ResponseFormatDto::JsonObject) => ResponseFormat::JsonObject,
        Some(ResponseFormatDto::JsonSchema {
            name,
            description,
            schema,
            strict,
        }) => ResponseFormat::JsonSchema {
            name,
            description,
            schema: schema.to_string(),
            strict,
        },
    }
}

fn parse_reasoning_effort(effort: Option<String>) -> Result<Option<ReasoningEffort>, ApiError> {
    match effort.as_deref() {
        None => Ok(None),
        Some("low") => Ok(Some(ReasoningEffort::Low)),
        Some("medium") => Ok(Some(ReasoningEffort::Medium)),
        Some("high") => Ok(Some(ReasoningEffort::High)),
        Some(_) => Err(ApiError::invalid(
            "reasoning.effort",
            "reasoning effort must be low, medium, or high",
        )),
    }
}

async fn responses(
    State(state): State<ApiState>,
    Extension(trusted): Extension<TrustedRequest>,
    payload: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = trusted.context.ensure_active() {
        return api_error_from_inference(&InferenceError::Context(error)).into_response();
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return api_error_from_json_rejection(&error).into_response(),
    };
    let stream_requested = request.stream;
    let model = request.model.clone();
    let request = match request.into_semantic() {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let stream = match state.service.infer(request, trusted.context).await {
        Ok(stream) => stream,
        Err(error) => return api_error_from_inference(&error).into_response(),
    };
    if stream_requested {
        responses_sse(stream, trusted.response_id, model)
    } else {
        collect_response(stream, trusted.response_id, model).await
    }
}

async fn collect_response(
    mut stream: ModelEventStream,
    request_id: String,
    model: String,
) -> Response {
    let mut output = OutputAccumulator::default();
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => output.apply(&event),
            Err(error) => return api_error_from_inference(&error).into_response(),
        }
    }
    Json(output.response_value(&request_id, &model)).into_response()
}

fn responses_sse(stream: ModelEventStream, request_id: String, model: String) -> Response {
    let event_stream = async_stream::stream! {
        let mut stream = stream;
        let mut output = OutputAccumulator::default();
        let mut text_started = false;
        let mut reasoning_started = false;
        let mut next_output_index = 0_usize;
        let mut text_output_index = None;
        let mut reasoning_output_index = None;
        let mut tool_output_indices = BTreeMap::new();
        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(error) => {
                    let error = api_error_from_inference(&error).body;
                    yield Ok::<Event, Infallible>(sse_event("error", json!({"type": "error", "error": error})));
                    break;
                }
            };
            match &event {
                ModelEvent::Accepted { .. } => {
                    yield Ok(sse_event("response.created", json!({
                        "type": "response.created",
                        "response": response_shell(&request_id, &model, "in_progress")
                    })));
                }
                ModelEvent::TextDelta { text, .. } => {
                    if !text_started {
                        text_started = true;
                        let output_index = next_output_index;
                        next_output_index += 1;
                        text_output_index = Some(output_index);
                        yield Ok(sse_event("response.output_item.added", json!({
                            "type": "response.output_item.added", "output_index": output_index,
                            "item": message_item(&request_id, "in_progress", "")
                        })));
                        yield Ok(sse_event("response.content_part.added", json!({
                            "type": "response.content_part.added", "item_id": message_id(&request_id),
                            "output_index": output_index, "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []}
                        })));
                    }
                    yield Ok(sse_event("response.output_text.delta", json!({
                        "type": "response.output_text.delta", "item_id": message_id(&request_id),
                        "output_index": text_output_index, "content_index": 0, "delta": text
                    })));
                }
                ModelEvent::ReasoningDelta { text, .. } => {
                    if !reasoning_started {
                        reasoning_started = true;
                        let output_index = next_output_index;
                        next_output_index += 1;
                        reasoning_output_index = Some(output_index);
                        yield Ok(sse_event("response.output_item.added", json!({
                            "type": "response.output_item.added", "output_index": output_index,
                            "item": reasoning_item(&request_id, "in_progress", "")
                        })));
                    }
                    yield Ok(sse_event("response.reasoning_summary_text.delta", json!({
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": reasoning_id(&request_id), "delta": text
                    })));
                }
                ModelEvent::ToolCallStarted { call_id, name, .. } => {
                    let index = next_output_index;
                    next_output_index += 1;
                    tool_output_indices.insert(call_id.clone(), index);
                    yield Ok(sse_event("response.output_item.added", json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": {"type": "function_call", "id": call_id, "call_id": call_id,
                            "name": name, "arguments": "", "status": "in_progress"}
                    })));
                }
                ModelEvent::ToolCallArgumentsDelta { call_id, delta, .. } => {
                    yield Ok(sse_event("response.function_call_arguments.delta", json!({
                        "type": "response.function_call_arguments.delta", "item_id": call_id,
                        "output_index": tool_output_indices.get(call_id), "delta": delta
                    })));
                }
                ModelEvent::ToolCallCompleted { call_id, arguments, .. } => {
                    yield Ok(sse_event("response.function_call_arguments.done", json!({
                        "type": "response.function_call_arguments.done", "item_id": call_id,
                        "output_index": tool_output_indices.get(call_id), "arguments": arguments
                    })));
                }
                ModelEvent::Finished { .. } => {
                    if text_started {
                        yield Ok(sse_event("response.output_text.done", json!({
                            "type": "response.output_text.done", "item_id": message_id(&request_id),
                            "output_index": text_output_index, "content_index": 0, "text": output.text
                        })));
                        yield Ok(sse_event("response.content_part.done", json!({
                            "type": "response.content_part.done", "item_id": message_id(&request_id),
                            "output_index": text_output_index, "content_index": 0,
                            "part": {"type": "output_text", "text": output.text, "annotations": []}
                        })));
                        yield Ok(sse_event("response.output_item.done", json!({
                            "type": "response.output_item.done", "output_index": text_output_index,
                            "item": message_item(&request_id, "completed", &output.text)
                        })));
                    }
                    if reasoning_started {
                        yield Ok(sse_event("response.reasoning_summary_text.done", json!({
                            "type": "response.reasoning_summary_text.done",
                            "item_id": reasoning_id(&request_id), "text": output.reasoning
                        })));
                        yield Ok(sse_event("response.output_item.done", json!({
                            "type": "response.output_item.done", "output_index": reasoning_output_index,
                            "item": reasoning_item(&request_id, "completed", &output.reasoning)
                        })));
                    }
                }
                ModelEvent::Usage { .. } => {}
            }
            output.apply(&event);
            if matches!(event, ModelEvent::ToolCallCompleted { .. }) {
                if let Some(tool) = output.tools.last() {
                    yield Ok(sse_event("response.output_item.done", json!({
                        "type": "response.output_item.done", "output_index": tool_output_indices.get(&tool.call_id),
                        "item": tool.value("completed")
                    })));
                }
            }
            if matches!(event, ModelEvent::Finished { .. }) {
                yield Ok(sse_event("response.completed", json!({
                    "type": "response.completed", "response": output.response_value(&request_id, &model)
                })));
                break;
            }
        }
    };
    Sse::new(event_stream.boxed())
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn sse_event(event_type: &'static str, value: Value) -> Event {
    Event::default().event(event_type).data(value.to_string())
}

#[derive(Default)]
struct OutputAccumulator {
    text: String,
    reasoning: String,
    tools: Vec<ToolOutput>,
    output_order: Vec<OutputItemKey>,
    usage: Usage,
    finish: Option<ModelFinishReason>,
}

enum OutputItemKey {
    Text,
    Reasoning,
    Tool(String),
}

struct ToolOutput {
    call_id: String,
    name: String,
    arguments: String,
}

impl ToolOutput {
    fn value(&self, status: &str) -> Value {
        json!({
            "type": "function_call", "id": self.call_id, "call_id": self.call_id,
            "name": self.name, "arguments": self.arguments, "status": status
        })
    }
}

impl OutputAccumulator {
    fn apply(&mut self, event: &ModelEvent) {
        match event {
            ModelEvent::TextDelta { text, .. } => {
                if self.text.is_empty() {
                    self.output_order.push(OutputItemKey::Text);
                }
                self.text.push_str(text);
            }
            ModelEvent::ReasoningDelta { text, .. } => {
                if self.reasoning.is_empty() {
                    self.output_order.push(OutputItemKey::Reasoning);
                }
                self.reasoning.push_str(text);
            }
            ModelEvent::ToolCallStarted { call_id, name, .. } => {
                self.output_order.push(OutputItemKey::Tool(call_id.clone()));
                self.tools.push(ToolOutput {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
            }
            ModelEvent::ToolCallArgumentsDelta { call_id, delta, .. } => {
                if let Some(tool) = self.tools.iter_mut().find(|tool| &tool.call_id == call_id) {
                    tool.arguments.push_str(delta);
                }
            }
            ModelEvent::ToolCallCompleted {
                call_id, arguments, ..
            } => {
                if let Some(tool) = self.tools.iter_mut().find(|tool| &tool.call_id == call_id) {
                    tool.arguments.clone_from(arguments);
                }
            }
            ModelEvent::Usage { usage, .. } => self.usage = usage.clone(),
            ModelEvent::Finished { reason, usage, .. } => {
                self.finish = Some(reason.clone());
                self.usage = usage.clone();
            }
            ModelEvent::Accepted { .. } => {}
        }
    }

    fn response_value(&self, request_id: &str, model: &str) -> Value {
        let status = if matches!(self.finish, Some(ModelFinishReason::Length)) {
            "incomplete"
        } else {
            "completed"
        };
        let output = self
            .output_order
            .iter()
            .filter_map(|item| match item {
                OutputItemKey::Text => Some(message_item(request_id, "completed", &self.text)),
                OutputItemKey::Reasoning => {
                    Some(reasoning_item(request_id, "completed", &self.reasoning))
                }
                OutputItemKey::Tool(call_id) => self
                    .tools
                    .iter()
                    .find(|tool| &tool.call_id == call_id)
                    .map(|tool| tool.value("completed")),
            })
            .collect::<Vec<_>>();
        let mut response = response_shell(request_id, model, status);
        response["output"] = Value::Array(output);
        response["usage"] = usage_value(&self.usage);
        response["incomplete_details"] = if status == "incomplete" {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        };
        response
    }
}

fn response_shell(request_id: &str, model: &str, status: &str) -> Value {
    json!({
        "id": request_id,
        "object": "response",
        "created_at": unix_timestamp(),
        "status": status,
        "model": model,
        "output": [],
        "error": null,
        "incomplete_details": null,
        "usage": null
    })
}

fn message_id(request_id: &str) -> String {
    format!("msg_{request_id}")
}

fn reasoning_id(request_id: &str) -> String {
    format!("rs_{request_id}")
}

fn message_item(request_id: &str, status: &str, text: &str) -> Value {
    json!({
        "type": "message", "id": message_id(request_id), "status": status,
        "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

fn reasoning_item(request_id: &str, status: &str, text: &str) -> Value {
    json!({
        "type": "reasoning", "id": reasoning_id(request_id), "status": status,
        "summary": [{"type": "summary_text", "text": text}]
    })
}

fn usage_value(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens),
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0}
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    model: String,
    prompt: CompletionPrompt,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<StopSequencesDto>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<CompletionStreamOptions>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    logprobs: Option<u32>,
    #[serde(default)]
    suffix: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompletionPrompt {
    Text(String),
    TokenIds(Vec<u32>),
    TextBatch(Vec<String>),
    TokenBatch(Vec<Vec<u32>>),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequencesDto {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionStreamOptions {
    #[serde(default)]
    include_usage: bool,
}

impl CompletionRequest {
    fn into_semantic(self) -> Result<ModelRequest, ApiError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::invalid("n", "n must be 1"));
        }
        if self.best_of.is_some() {
            return Err(ApiError::unsupported("best_of"));
        }
        if self.echo.unwrap_or(false) {
            return Err(ApiError::unsupported("echo"));
        }
        if self.logprobs.is_some() {
            return Err(ApiError::unsupported("logprobs"));
        }
        if self.suffix.is_some() {
            return Err(ApiError::unsupported("suffix"));
        }
        if !self.stream && self.stream_options.is_some() {
            return Err(ApiError::invalid(
                "stream_options",
                "stream_options requires stream=true",
            ));
        }
        let input = match self.prompt {
            CompletionPrompt::Text(text) => PromptInput::Text(text),
            CompletionPrompt::TokenIds(token_ids) => PromptInput::TokenIds(token_ids),
            CompletionPrompt::TextBatch(prompts) => {
                return Err(ApiError::invalid(
                    "prompt",
                    format!(
                        "batched text prompts are unsupported (received {})",
                        prompts.len()
                    ),
                ));
            }
            CompletionPrompt::TokenBatch(prompts) => {
                return Err(ApiError::invalid(
                    "prompt",
                    format!(
                        "batched token prompts are unsupported (received {})",
                        prompts.len()
                    ),
                ));
            }
        };
        let stop_sequences = match self.stop {
            None => Vec::new(),
            Some(StopSequencesDto::One(stop)) => vec![stop],
            Some(StopSequencesDto::Many(stops)) => stops,
        };
        if stop_sequences.iter().any(String::is_empty) {
            return Err(ApiError::invalid(
                "stop",
                "stop sequences must not be empty",
            ));
        }
        Ok(ModelRequest {
            model: self.model,
            input: ModelInput::Prompt(input),
            sampling: SamplingParameters {
                max_output_tokens: self.max_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                seed: self.seed,
                stop_sequences,
            },
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            response_format: ResponseFormat::Text,
            reasoning_effort: None,
            metadata: BTreeMap::new(),
        })
    }
}

async fn completions(
    State(state): State<ApiState>,
    Extension(trusted): Extension<TrustedRequest>,
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = trusted.context.ensure_active() {
        return api_error_from_inference(&InferenceError::Context(error)).into_response();
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return api_error_from_json_rejection(&error).into_response(),
    };
    let stream_requested = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let model = request.model.clone();
    let request = match request.into_semantic() {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let stream = match state.service.infer(request, trusted.context).await {
        Ok(stream) => stream,
        Err(error) => return api_error_from_inference(&error).into_response(),
    };
    if stream_requested {
        completion_sse(stream, trusted.response_id, model, include_usage)
    } else {
        collect_completion(stream, trusted.response_id, model).await
    }
}

async fn collect_completion(
    mut stream: ModelEventStream,
    request_id: String,
    model: String,
) -> Response {
    let mut output = OutputAccumulator::default();
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => output.apply(&event),
            Err(error) => return api_error_from_inference(&error).into_response(),
        }
    }
    Json(json!({
        "id": request_id,
        "object": "text_completion",
        "created": unix_timestamp(),
        "model": model,
        "choices": [{
            "text": output.text,
            "index": 0,
            "logprobs": null,
            "finish_reason": completion_finish_reason(output.finish.as_ref())
        }],
        "usage": completion_usage_value(&output.usage),
        "system_fingerprint": null
    }))
    .into_response()
}

fn completion_sse(
    stream: ModelEventStream,
    request_id: String,
    model: String,
    include_usage: bool,
) -> Response {
    let event_stream = async_stream::stream! {
        let mut stream = stream;
        let mut output = OutputAccumulator::default();
        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(error) => {
                    yield Ok::<Event, Infallible>(Event::default().data(json!({"error": api_error_from_inference(&error).body}).to_string()));
                    break;
                }
            };
            match &event {
                ModelEvent::TextDelta { text, .. } => {
                    yield Ok(completion_chunk(
                        &request_id,
                        &model,
                        vec![json!({
                            "text": text,
                            "index": 0,
                            "logprobs": null,
                            "finish_reason": null
                        })],
                        Value::Null,
                    ));
                }
                ModelEvent::Finished { reason, .. } => {
                    yield Ok(completion_chunk(
                        &request_id,
                        &model,
                        vec![json!({
                            "text": "",
                            "index": 0,
                            "logprobs": null,
                            "finish_reason": completion_finish_reason(Some(reason))
                        })],
                        Value::Null,
                    ));
                }
                ModelEvent::Accepted { .. }
                | ModelEvent::ReasoningDelta { .. }
                | ModelEvent::ToolCallStarted { .. }
                | ModelEvent::ToolCallArgumentsDelta { .. }
                | ModelEvent::ToolCallCompleted { .. }
                | ModelEvent::Usage { .. } => {}
            }
            output.apply(&event);
            if matches!(event, ModelEvent::Finished { .. }) {
                if include_usage {
                    yield Ok(completion_chunk(
                        &request_id,
                        &model,
                        Vec::new(),
                        completion_usage_value(&output.usage),
                    ));
                }
                yield Ok(Event::default().data("[DONE]"));
                break;
            }
        }
    };
    Sse::new(event_stream.boxed())
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn completion_chunk(request_id: &str, model: &str, choices: Vec<Value>, usage: Value) -> Event {
    Event::default().data(
        json!({
            "id": request_id,
            "object": "text_completion",
            "created": unix_timestamp(),
            "model": model,
            "choices": choices,
            "usage": usage,
            "system_fingerprint": null
        })
        .to_string(),
    )
}

fn completion_usage_value(usage: &Usage) -> Value {
    json!({
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens.saturating_add(usage.output_tokens)
    })
}

fn completion_finish_reason(reason: Option<&ModelFinishReason>) -> &'static str {
    match reason {
        Some(ModelFinishReason::Length) => "length",
        Some(ModelFinishReason::ContentFilter) => "content_filter",
        Some(ModelFinishReason::Cancelled | ModelFinishReason::Error) => "error",
        _ => "stop",
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    tools: Vec<ChatTool>,
    #[serde(default)]
    tool_choice: Option<ToolChoiceDto>,
    #[serde(default)]
    response_format: Option<ChatResponseFormat>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTool {
    #[serde(rename = "type")]
    kind: String,
    function: ChatFunction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatFunction {
    name: String,
    #[serde(default)]
    description: Option<String>,
    parameters: Value,
    #[serde(default)]
    strict: bool,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum ChatResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: ChatJsonSchema },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatJsonSchema {
    name: String,
    #[serde(default)]
    description: Option<String>,
    schema: Value,
    #[serde(default)]
    strict: bool,
}

impl ChatRequest {
    fn into_semantic(self) -> Result<ModelRequest, ApiError> {
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err(ApiError::invalid(
                "max_tokens",
                "max_tokens and max_completion_tokens cannot both be set",
            ));
        }
        let messages = self
            .messages
            .into_iter()
            .map(|message| {
                Ok(ConversationMessage {
                    role: parse_role(&message.role)?,
                    content: message.content,
                    tool_call_id: message.tool_call_id,
                })
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        let tools = self
            .tools
            .into_iter()
            .map(|tool| {
                if tool.kind != "function" {
                    return Err(ApiError::invalid(
                        "tools",
                        "only function tools are supported",
                    ));
                }
                Ok(ToolDefinition {
                    name: tool.function.name,
                    description: tool.function.description,
                    parameters_schema: tool.function.parameters.to_string(),
                    strict: tool.function.strict,
                })
            })
            .collect::<Result<Vec<_>, ApiError>>()?;
        let response_format = match self.response_format {
            None | Some(ChatResponseFormat::Text) => ResponseFormat::Text,
            Some(ChatResponseFormat::JsonObject) => ResponseFormat::JsonObject,
            Some(ChatResponseFormat::JsonSchema { json_schema }) => ResponseFormat::JsonSchema {
                name: json_schema.name,
                description: json_schema.description,
                schema: json_schema.schema.to_string(),
                strict: json_schema.strict,
            },
        };
        Ok(ModelRequest {
            model: self.model,
            input: ModelInput::Conversation(Conversation { messages }),
            sampling: SamplingParameters {
                max_output_tokens: self.max_completion_tokens.or(self.max_tokens),
                temperature: self.temperature,
                top_p: self.top_p,
                seed: None,
                stop_sequences: Vec::new(),
            },
            tools,
            tool_choice: parse_tool_choice(self.tool_choice)?,
            response_format,
            reasoning_effort: parse_reasoning_effort(self.reasoning_effort)?,
            metadata: BTreeMap::new(),
        })
    }
}

async fn chat_completions(
    State(state): State<ApiState>,
    Extension(trusted): Extension<TrustedRequest>,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = trusted.context.ensure_active() {
        return api_error_from_inference(&InferenceError::Context(error)).into_response();
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => return api_error_from_json_rejection(&error).into_response(),
    };
    let stream_requested = request.stream;
    let model = request.model.clone();
    let request = match request.into_semantic() {
        Ok(request) => request,
        Err(error) => return error.into_response(),
    };
    let stream = match state.service.infer(request, trusted.context).await {
        Ok(stream) => stream,
        Err(error) => return api_error_from_inference(&error).into_response(),
    };
    if stream_requested {
        chat_sse(stream, trusted.response_id, model)
    } else {
        collect_chat(stream, trusted.response_id, model).await
    }
}

async fn collect_chat(mut stream: ModelEventStream, request_id: String, model: String) -> Response {
    let mut output = OutputAccumulator::default();
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => output.apply(&event),
            Err(error) => return api_error_from_inference(&error).into_response(),
        }
    }
    let tool_calls = output
        .tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            json!({
                "index": index, "id": tool.call_id, "type": "function",
                "function": {"name": tool.name, "arguments": tool.arguments}
            })
        })
        .collect::<Vec<_>>();
    let finish_reason = chat_finish_reason(output.finish.as_ref());
    Json(json!({
        "id": request_id, "object": "chat.completion", "created": unix_timestamp(),
        "model": model,
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": if output.text.is_empty() { Value::Null } else { json!(output.text) },
            "reasoning_content": if output.reasoning.is_empty() { Value::Null } else { json!(output.reasoning) },
            "tool_calls": tool_calls
        }, "finish_reason": finish_reason}],
        "usage": usage_value(&output.usage)
    })).into_response()
}

fn chat_sse(stream: ModelEventStream, request_id: String, model: String) -> Response {
    let event_stream = async_stream::stream! {
        let mut stream = stream;
        let mut output = OutputAccumulator::default();
        let mut role_sent = false;
        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(error) => {
                    yield Ok::<Event, Infallible>(Event::default().data(json!({"error": api_error_from_inference(&error).body}).to_string()));
                    break;
                }
            };
            if !role_sent && matches!(event, ModelEvent::Accepted { .. }) {
                role_sent = true;
                yield Ok(chat_chunk(&request_id, &model, json!({"role": "assistant", "content": ""}), Value::Null));
            }
            match &event {
                ModelEvent::TextDelta { text, .. } => {
                    yield Ok(chat_chunk(&request_id, &model, json!({"content": text}), Value::Null));
                }
                ModelEvent::ReasoningDelta { text, .. } => {
                    yield Ok(chat_chunk(&request_id, &model, json!({"reasoning_content": text}), Value::Null));
                }
                ModelEvent::ToolCallStarted { call_id, name, .. } => {
                    let index = output.tools.len();
                    yield Ok(chat_chunk(&request_id, &model, json!({"tool_calls": [{
                        "index": index, "id": call_id, "type": "function",
                        "function": {"name": name, "arguments": ""}
                    }]}), Value::Null));
                }
                ModelEvent::ToolCallArgumentsDelta { call_id, delta, .. } => {
                    let index = output.tools.iter().position(|tool| &tool.call_id == call_id).unwrap_or(0);
                    yield Ok(chat_chunk(&request_id, &model, json!({"tool_calls": [{
                        "index": index, "function": {"arguments": delta}
                    }]}), Value::Null));
                }
                ModelEvent::Finished { reason, .. } => {
                    yield Ok(chat_chunk(&request_id, &model, json!({}), json!(chat_finish_reason(Some(reason)))));
                }
                ModelEvent::Accepted { .. }
                | ModelEvent::ToolCallCompleted { .. }
                | ModelEvent::Usage { .. } => {}
            }
            output.apply(&event);
            if matches!(event, ModelEvent::Finished { .. }) {
                yield Ok(Event::default().data("[DONE]"));
                break;
            }
        }
    };
    Sse::new(event_stream.boxed())
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn chat_chunk(request_id: &str, model: &str, delta: Value, finish_reason: Value) -> Event {
    Event::default().data(json!({
        "id": request_id, "object": "chat.completion.chunk", "created": unix_timestamp(),
        "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }).to_string())
}

fn chat_finish_reason(reason: Option<&ModelFinishReason>) -> &'static str {
    match reason {
        Some(ModelFinishReason::Length) => "length",
        Some(ModelFinishReason::ToolCall) => "tool_calls",
        Some(ModelFinishReason::ContentFilter) => "content_filter",
        Some(ModelFinishReason::Cancelled | ModelFinishReason::Error) => "error",
        _ => "stop",
    }
}

fn parse_role(role: &str) -> Result<ConversationRole, ApiError> {
    match role {
        "developer" => Ok(ConversationRole::Developer),
        "system" => Ok(ConversationRole::System),
        "user" => Ok(ConversationRole::User),
        "assistant" => Ok(ConversationRole::Assistant),
        "tool" => Ok(ConversationRole::Tool),
        _ => Err(ApiError::invalid(
            "messages.role",
            "role must be developer, system, user, assistant, or tool",
        )),
    }
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

struct ApiError {
    status: StatusCode,
    body: Value,
}

impl ApiError {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        error_type: &str,
        param: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        Self {
            status,
            body: json!({
                "message": message.into(), "type": error_type, "param": param, "code": code
            }),
        }
    }

    fn invalid(param: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            Some(param),
            Some("invalid_parameter"),
        )
    }

    fn invalid_json(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            None,
            Some("invalid_json"),
        )
    }

    fn unsupported(param: &'static str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            format!("unsupported parameter: {param}"),
            "invalid_request_error",
            Some(param),
            Some("unsupported_parameter"),
        )
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status;
        let mut response = (status, Json(json!({"error": self.body}))).into_response();
        if matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
        ) {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

fn api_error_from_inference(error: &InferenceError) -> ApiError {
    match error {
        InferenceError::ModelIo(ModelIoError::ModelNotFound(model)) => ApiError::new(
            StatusCode::NOT_FOUND,
            format!("model was not found: {model}"),
            "invalid_request_error",
            Some("model"),
            Some("model_not_found"),
        ),
        InferenceError::ModelIo(ModelIoError::InvalidInput(message)) => ApiError::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            None,
            Some("invalid_parameter"),
        ),
        InferenceError::Context(locus_core::ContextError::Cancelled)
        | InferenceError::Engine(locus_engine::EngineError::Context(
            locus_core::ContextError::Cancelled,
        ))
        | InferenceError::Execution(locus_planner::PlanExecutionError::Context(
            locus_core::ContextError::Cancelled,
        ))
        | InferenceError::Execution(locus_planner::PlanExecutionError::Engine(
            locus_engine::EngineError::Context(locus_core::ContextError::Cancelled),
        )) => ApiError::new(
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            error.to_string(),
            "request_cancelled",
            None,
            Some("cancelled"),
        ),
        InferenceError::Context(locus_core::ContextError::DeadlineExceeded)
        | InferenceError::Engine(locus_engine::EngineError::Context(
            locus_core::ContextError::DeadlineExceeded,
        ))
        | InferenceError::Execution(locus_planner::PlanExecutionError::Context(
            locus_core::ContextError::DeadlineExceeded,
        ))
        | InferenceError::Execution(locus_planner::PlanExecutionError::Engine(
            locus_engine::EngineError::Context(locus_core::ContextError::DeadlineExceeded),
        )) => ApiError::new(
            StatusCode::REQUEST_TIMEOUT,
            error.to_string(),
            "request_timeout",
            None,
            Some("deadline_exceeded"),
        ),
        InferenceError::Admission(error) => api_error_from_admission(error),
        InferenceError::Planning(_)
        | InferenceError::Discovery(_)
        | InferenceError::Engine(
            locus_engine::EngineError::Draining | locus_engine::EngineError::Stopped,
        ) => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            error.to_string(),
            "server_error",
            None,
            Some("no_available_target"),
        ),
        _ => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            error.to_string(),
            "server_error",
            None,
            Some("internal_error"),
        ),
    }
}

fn api_error_from_admission(error: &AdmissionError) -> ApiError {
    match error {
        AdmissionError::Cancelled => ApiError::new(
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            error.to_string(),
            "request_cancelled",
            None,
            Some("cancelled"),
        ),
        AdmissionError::DeadlineExceeded => ApiError::new(
            StatusCode::REQUEST_TIMEOUT,
            error.to_string(),
            "request_timeout",
            None,
            Some("deadline_exceeded"),
        ),
        AdmissionError::RequestTokenLimit { .. }
        | AdmissionError::RequestExceedsCapacity
        | AdmissionError::InvalidDeadline => ApiError::new(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            "invalid_request_error",
            Some("max_output_tokens"),
            Some("token_budget_exceeded"),
        ),
        AdmissionError::QueueFull | AdmissionError::OverloadShed => ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            error.to_string(),
            "rate_limit_error",
            None,
            Some("overloaded"),
        ),
        AdmissionError::Draining | AdmissionError::Unavailable => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            error.to_string(),
            "server_error",
            None,
            Some("admission_unavailable"),
        ),
        AdmissionError::MissingTrustedTenant | AdmissionError::UnknownTenant(_) => ApiError::new(
            StatusCode::UNAUTHORIZED,
            error.to_string(),
            "authentication_error",
            None,
            Some("invalid_tenant"),
        ),
    }
}

fn api_error_from_json_rejection(error: &JsonRejection) -> ApiError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            error.body_text(),
            "invalid_request_error",
            None,
            Some("request_too_large"),
        )
    } else {
        ApiError::invalid_json(error.body_text())
    }
}
