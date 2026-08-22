use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Extension, Request, State, rejection::JsonRejection};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{Next, from_fn_with_state};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use locus_core::{OperationContext, RequestId, SamplingParameters, Usage};
use locus_http::{CredentialIndex, CredentialIndexError};
use locus_model_io::{
    Conversation, ConversationMessage, ConversationRole, ModelEvent, ModelFinishReason, ModelInput,
    ModelIoError, ModelRequest, ResponseFormat, ToolChoice, ToolDefinition,
};
use locus_runtime::{
    AdmissionError, InferenceError, InferenceService, ModelEventStream, TrafficController,
};
use serde::Deserialize;
use serde_json::{Value, json};
use thiserror::Error;

pub use locus_http::TenantCredential;

const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";
const API_KEY_HEADER: &str = "x-api-key";
const REQUEST_ID_HEADER: &str = "request-id";
const REQUEST_TIMEOUT_HEADER: &str = "x-request-timeout-ms";
const DEFAULT_MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone)]
pub struct ApiConfig {
    pub bearer_token: Option<String>,
    pub tenant_credentials: Vec<TenantCredential>,
    pub anonymous_tenant: Option<String>,
    pub traffic: TrafficController,
    pub max_request_bytes: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            bearer_token: None,
            tenant_credentials: Vec::new(),
            anonymous_tenant: None,
            traffic: TrafficController::default(),
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
}

/// Builds the Anthropic Messages compatibility surface.
///
/// This router owns only `/v1/messages`, allowing it to be merged with the
/// OpenAI router without ambiguous model or health routes.
pub fn router_with_config(
    service: Arc<dyn InferenceService>,
    config: ApiConfig,
) -> Result<Router, ApiConfigError> {
    if config.max_request_bytes == 0 {
        return Err(ApiConfigError::ZeroRequestBodyLimit);
    }
    if config.bearer_token.as_ref().is_some_and(String::is_empty) {
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
    let auth = AuthState {
        credentials,
        anonymous_tenant: anonymous_tenant.map(Into::into),
        traffic: config.traffic,
        next_request: Arc::new(AtomicU64::new(1)),
        max_request_bytes: config.max_request_bytes,
    };
    Ok(Router::new()
        .route("/v1/messages", post(messages))
        .layer(DefaultBodyLimit::max(config.max_request_bytes))
        .layer(from_fn_with_state(auth, authenticate))
        .with_state(ApiState { service }))
}

async fn authenticate(State(auth): State<AuthState>, mut request: Request, next: Next) -> Response {
    let response_id = format!(
        "msg_{:016x}",
        auth.next_request.fetch_add(1, Ordering::AcqRel)
    );
    let version = request
        .headers()
        .get(ANTHROPIC_VERSION_HEADER)
        .and_then(|value| value.to_str().ok());
    if version != Some(ANTHROPIC_VERSION) {
        return ApiError::invalid(format!(
            "{ANTHROPIC_VERSION_HEADER} must be {ANTHROPIC_VERSION}"
        ))
        .with_request_id(response_id)
        .into_response();
    }
    let candidate = api_key(request.headers());
    let tenant = if let Some(candidate) = candidate {
        auth.credentials.authenticate(candidate)
    } else {
        auth.anonymous_tenant.clone()
    };
    let Some(tenant) = tenant else {
        return ApiError::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "missing or invalid API key",
        )
        .with_request_id(response_id)
        .into_response();
    };
    let requested_timeout = match requested_timeout(request.headers()) {
        Ok(timeout) => timeout,
        Err(error) => {
            return error.with_request_id(response_id).into_response();
        }
    };
    let context = match auth.traffic.operation_context(
        RequestId::new(response_id.clone()),
        tenant.as_ref(),
        requested_timeout,
    ) {
        Ok(context) => context,
        Err(error) => {
            return api_error_from_admission(&error)
                .with_request_id(response_id)
                .into_response();
        }
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
                    "request_too_large",
                    "request body exceeded the configured byte limit",
                )
                .with_request_id(response_id)
                .into_response();
            }
            Ok(Err(RequestBodyReadError::Transport(error))) => {
                return ApiError::invalid(error)
                    .with_request_id(response_id)
                    .into_response();
            }
            Err(error) => {
                return api_error_from_inference(&InferenceError::Context(error))
                    .with_request_id(response_id)
                    .into_response();
            }
        };
        request = Request::from_parts(parts, Body::from(body));
    }
    request.extensions_mut().insert(TrustedRequest {
        response_id: response_id.clone(),
        context,
    });
    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&response_id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

fn api_key(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(API_KEY_HEADER)
        .and_then(|value| value.to_str().ok())
        .or_else(|| {
            headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.strip_prefix("Bearer "))
        })
}

fn requested_timeout(headers: &HeaderMap) -> Result<Option<Duration>, ApiError> {
    headers
        .get(REQUEST_TIMEOUT_HEADER)
        .map(|value| {
            let value = value.to_str().map_err(|_| {
                ApiError::invalid("request timeout header must be ASCII milliseconds")
            })?;
            let millis = value.parse::<u64>().map_err(|_| {
                ApiError::invalid("request timeout header must be integer milliseconds")
            })?;
            if millis == 0 {
                return Err(ApiError::invalid(
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<InputMessage>,
    #[serde(default)]
    system: Option<TextContent>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    metadata: Option<Metadata>,
    #[serde(default)]
    tools: Vec<AnthropicTool>,
    #[serde(default)]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    thinking: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputMessage {
    role: InputRole,
    content: MessageContent,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputRole {
    User,
    Assistant,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Blocks(Vec<InputContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum InputContentBlock {
    Text {
        text: String,
    },
    ToolResult {
        tool_use_id: String,
        content: TextContent,
        #[serde(default)]
        is_error: bool,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Image {
        source: Value,
    },
    Document {
        source: Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TextContent {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum TextBlock {
    Text { text: String },
}

impl TextContent {
    fn flatten(self) -> String {
        match self {
            Self::Text(text) => text,
            Self::Blocks(blocks) => blocks
                .into_iter()
                .map(|block| match block {
                    TextBlock::Text { text } => text,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Metadata {
    user_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnthropicTool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

impl MessagesRequest {
    fn into_semantic(self) -> Result<(ModelRequest, bool), ApiError> {
        if self.max_tokens == 0 {
            return Err(ApiError::invalid("max_tokens must be greater than zero"));
        }
        if self.messages.is_empty() {
            return Err(ApiError::invalid("messages must not be empty"));
        }
        if self.top_k.is_some() {
            return Err(ApiError::unsupported("top_k"));
        }
        if self.thinking.is_some() {
            return Err(ApiError::unsupported("thinking"));
        }
        let mut messages = Vec::new();
        if let Some(system) = self.system {
            messages.push(ConversationMessage {
                role: ConversationRole::System,
                content: system.flatten(),
                tool_call_id: None,
            });
        }
        for message in self.messages {
            append_message(&mut messages, message)?;
        }
        let tools = self
            .tools
            .into_iter()
            .map(|tool| ToolDefinition {
                name: tool.name,
                description: tool.description,
                parameters_schema: tool.input_schema.to_string(),
                strict: false,
            })
            .collect();
        let tool_choice = match self.tool_choice {
            None | Some(AnthropicToolChoice::Auto) => ToolChoice::Auto,
            Some(AnthropicToolChoice::Any) => ToolChoice::Required,
            Some(AnthropicToolChoice::Tool { name }) => ToolChoice::Function(name),
            Some(AnthropicToolChoice::None) => ToolChoice::None,
        };
        let mut metadata = BTreeMap::new();
        if let Some(user_id) = self.metadata.and_then(|metadata| metadata.user_id) {
            metadata.insert("user_id".to_owned(), user_id);
        }
        Ok((
            ModelRequest {
                model: self.model,
                input: ModelInput::Conversation(Conversation { messages }),
                sampling: SamplingParameters {
                    max_output_tokens: Some(self.max_tokens),
                    temperature: self.temperature,
                    top_p: self.top_p,
                    seed: None,
                    stop_sequences: self.stop_sequences,
                },
                tools,
                tool_choice,
                response_format: ResponseFormat::Text,
                reasoning_effort: None,
                metadata,
            },
            self.stream,
        ))
    }
}

fn append_message(
    output: &mut Vec<ConversationMessage>,
    message: InputMessage,
) -> Result<(), ApiError> {
    let role = match message.role {
        InputRole::User => ConversationRole::User,
        InputRole::Assistant => ConversationRole::Assistant,
    };
    match message.content {
        MessageContent::Text(content) => output.push(ConversationMessage {
            role,
            content,
            tool_call_id: None,
        }),
        MessageContent::Blocks(blocks) => {
            for block in blocks {
                match block {
                    InputContentBlock::Text { text } => output.push(ConversationMessage {
                        role: role.clone(),
                        content: text,
                        tool_call_id: None,
                    }),
                    InputContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } if matches!(message.role, InputRole::User) => {
                        let mut content = content.flatten();
                        if is_error {
                            content.insert_str(0, "Error: ");
                        }
                        output.push(ConversationMessage {
                            role: ConversationRole::Tool,
                            content,
                            tool_call_id: Some(tool_use_id),
                        });
                    }
                    InputContentBlock::ToolResult { .. } => {
                        return Err(ApiError::invalid(
                            "tool_result blocks are valid only in user messages",
                        ));
                    }
                    InputContentBlock::ToolUse { id, name, input } => {
                        let _ = (id, name, input);
                        return Err(ApiError::unsupported("messages.content.tool_use"));
                    }
                    InputContentBlock::Image { source } => {
                        let _ = source;
                        return Err(ApiError::unsupported("messages.content.image"));
                    }
                    InputContentBlock::Document { source } => {
                        let _ = source;
                        return Err(ApiError::unsupported("messages.content.document"));
                    }
                }
            }
        }
    }
    Ok(())
}

async fn messages(
    State(state): State<ApiState>,
    Extension(trusted): Extension<TrustedRequest>,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    if let Err(error) = trusted.context.ensure_active() {
        return api_error_from_inference(&InferenceError::Context(error))
            .with_request_id(trusted.response_id)
            .into_response();
    }
    let Json(request) = match payload {
        Ok(payload) => payload,
        Err(error) => {
            return api_error_from_json_rejection(&error)
                .with_request_id(trusted.response_id)
                .into_response();
        }
    };
    let model = request.model.clone();
    let (request, stream_requested) = match request.into_semantic() {
        Ok(request) => request,
        Err(error) => {
            return error.with_request_id(trusted.response_id).into_response();
        }
    };
    let stream = match state.service.infer(request, trusted.context).await {
        Ok(stream) => stream,
        Err(error) => {
            return api_error_from_inference(&error)
                .with_request_id(trusted.response_id)
                .into_response();
        }
    };
    if stream_requested {
        messages_sse(stream, trusted.response_id, model)
    } else {
        collect_message(stream, trusted.response_id, model).await
    }
}

#[derive(Default)]
struct MessageAccumulator {
    content: Vec<OutputBlock>,
    usage: Usage,
    finish: Option<ModelFinishReason>,
}

enum OutputBlock {
    Text(String),
    Tool {
        id: String,
        name: String,
        arguments: String,
    },
}

impl MessageAccumulator {
    fn apply(&mut self, event: &ModelEvent) {
        match event {
            ModelEvent::TextDelta { text, .. } => match self.content.last_mut() {
                Some(OutputBlock::Text(current)) => current.push_str(text),
                _ => self.content.push(OutputBlock::Text(text.clone())),
            },
            ModelEvent::ToolCallStarted { call_id, name, .. } => {
                self.content.push(OutputBlock::Tool {
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
            }
            ModelEvent::ToolCallArgumentsDelta { call_id, delta, .. } => {
                if let Some(OutputBlock::Tool { id, arguments, .. }) = self.content.last_mut()
                    && id == call_id
                {
                    arguments.push_str(delta);
                }
            }
            ModelEvent::Usage { usage, .. } => self.usage = usage.clone(),
            ModelEvent::Finished { reason, usage, .. } => {
                self.finish = Some(reason.clone());
                self.usage = usage.clone();
            }
            ModelEvent::Accepted { .. }
            | ModelEvent::ReasoningDelta { .. }
            | ModelEvent::ToolCallCompleted { .. } => {}
        }
    }

    fn content_value(&self) -> Result<Vec<Value>, ApiError> {
        self.content
            .iter()
            .map(|block| match block {
                OutputBlock::Text(text) => Ok(json!({"type": "text", "text": text})),
                OutputBlock::Tool {
                    id,
                    name,
                    arguments,
                } => {
                    let input = parse_tool_input(arguments)?;
                    Ok(json!({"type": "tool_use", "id": id, "name": name, "input": input}))
                }
            })
            .collect()
    }
}

async fn collect_message(
    mut stream: ModelEventStream,
    request_id: String,
    model: String,
) -> Response {
    let mut output = MessageAccumulator::default();
    while let Some(event) = stream.next().await {
        match event {
            Ok(event) => output.apply(&event),
            Err(error) => {
                return api_error_from_inference(&error)
                    .with_request_id(request_id)
                    .into_response();
            }
        }
    }
    if matches!(
        output.finish,
        Some(ModelFinishReason::Cancelled | ModelFinishReason::Error)
    ) {
        return ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "model execution did not complete successfully",
        )
        .with_request_id(request_id)
        .into_response();
    }
    let content = match output.content_value() {
        Ok(content) => content,
        Err(error) => return error.with_request_id(request_id).into_response(),
    };
    let stop_reason = anthropic_stop_reason(output.finish.as_ref());
    Json(json!({
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage_value(&output.usage),
    }))
    .into_response()
}

#[derive(Default)]
struct StreamState {
    started: bool,
    next_index: usize,
    active: Option<ActiveBlock>,
    usage: Usage,
}

enum ActiveBlock {
    Text {
        index: usize,
    },
    Tool {
        index: usize,
        id: String,
        arguments: String,
    },
}

impl StreamState {
    fn close_active(&mut self) -> Result<Vec<Event>, ApiError> {
        let Some(active) = self.active.take() else {
            return Ok(Vec::new());
        };
        let index = match active {
            ActiveBlock::Text { index } => index,
            ActiveBlock::Tool {
                index, arguments, ..
            } => {
                let _ = parse_tool_input(&arguments)?;
                index
            }
        };
        Ok(vec![anthropic_event(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        )])
    }
}

fn messages_sse(stream: ModelEventStream, request_id: String, model: String) -> Response {
    let event_stream = async_stream::stream! {
        let mut stream = stream;
        let mut state = StreamState::default();
        while let Some(result) = stream.next().await {
            let event = match result {
                Ok(event) => event,
                Err(error) => {
                    yield Ok::<Event, Infallible>(stream_error(&api_error_from_inference(&error)));
                    break;
                }
            };
            if !state.started {
                state.started = true;
                yield Ok(anthropic_event("message_start", json!({
                    "type": "message_start",
                    "message": {
                        "id": request_id, "type": "message", "role": "assistant",
                        "content": [], "model": model, "stop_reason": null,
                        "stop_sequence": null, "usage": usage_value(&Usage::default()),
                    }
                })));
            }
            match event {
                ModelEvent::Accepted { .. } | ModelEvent::ReasoningDelta { .. } => {}
                ModelEvent::TextDelta { text, .. } => {
                    if !matches!(state.active, Some(ActiveBlock::Text { .. })) {
                        match state.close_active() {
                            Ok(events) => for event in events { yield Ok(event); },
                            Err(error) => { yield Ok(stream_error(&error)); break; }
                        }
                        let index = state.next_index;
                        state.next_index += 1;
                        state.active = Some(ActiveBlock::Text { index });
                        yield Ok(anthropic_event("content_block_start", json!({
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""}
                        })));
                    }
                    let index = match state.active { Some(ActiveBlock::Text { index }) => index, _ => 0 };
                    yield Ok(anthropic_event("content_block_delta", json!({
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": text}
                    })));
                }
                ModelEvent::ToolCallStarted { call_id, name, .. } => {
                    match state.close_active() {
                        Ok(events) => for event in events { yield Ok(event); },
                        Err(error) => { yield Ok(stream_error(&error)); break; }
                    }
                    let index = state.next_index;
                    state.next_index += 1;
                    state.active = Some(ActiveBlock::Tool {
                        index,
                        id: call_id.clone(),
                        arguments: String::new(),
                    });
                    yield Ok(anthropic_event("content_block_start", json!({
                        "type": "content_block_start", "index": index,
                        "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}}
                    })));
                }
                ModelEvent::ToolCallArgumentsDelta { call_id, delta, .. } => {
                    let Some(ActiveBlock::Tool { index, id, arguments }) = state.active.as_mut() else {
                        yield Ok(stream_error(&ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", "tool delta arrived without an active tool block")));
                        break;
                    };
                    if *id != call_id {
                        yield Ok(stream_error(&ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", "tool delta did not match the active tool block")));
                        break;
                    }
                    arguments.push_str(&delta);
                    yield Ok(anthropic_event("content_block_delta", json!({
                        "type": "content_block_delta", "index": *index,
                        "delta": {"type": "input_json_delta", "partial_json": delta}
                    })));
                }
                ModelEvent::ToolCallCompleted { call_id, .. } => {
                    if !matches!(&state.active, Some(ActiveBlock::Tool { id, .. }) if *id == call_id) {
                        yield Ok(stream_error(&ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", "tool completion did not match the active tool block")));
                        break;
                    }
                    match state.close_active() {
                        Ok(events) => for event in events { yield Ok(event); },
                        Err(error) => { yield Ok(stream_error(&error)); break; }
                    }
                }
                ModelEvent::Usage { usage, .. } => state.usage = usage,
                ModelEvent::Finished { reason, usage, .. } => {
                    state.usage = usage;
                    match state.close_active() {
                        Ok(events) => for event in events { yield Ok(event); },
                        Err(error) => { yield Ok(stream_error(&error)); break; }
                    }
                    if matches!(reason, ModelFinishReason::Cancelled | ModelFinishReason::Error) {
                        yield Ok(stream_error(&ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", "model execution did not complete successfully")));
                        break;
                    }
                    yield Ok(anthropic_event("message_delta", json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": anthropic_stop_reason(Some(&reason)), "stop_sequence": null},
                        "usage": {"output_tokens": state.usage.output_tokens}
                    })));
                    yield Ok(anthropic_event("message_stop", json!({"type": "message_stop"})));
                    break;
                }
            }
        }
    };
    Sse::new(event_stream.boxed())
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn parse_tool_input(arguments: &str) -> Result<Value, ApiError> {
    let value = serde_json::from_str::<Value>(arguments).map_err(|_| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "tool input was not valid JSON",
        )
    })?;
    if !value.is_object() {
        return Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            "tool input must be a JSON object",
        ));
    }
    Ok(value)
}

fn anthropic_stop_reason(reason: Option<&ModelFinishReason>) -> &'static str {
    match reason {
        Some(ModelFinishReason::Length) => "max_tokens",
        Some(ModelFinishReason::ToolCall) => "tool_use",
        Some(ModelFinishReason::ContentFilter) => "refusal",
        _ => "end_turn",
    }
}

fn usage_value(usage: &Usage) -> Value {
    json!({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})
}

fn anthropic_event(event_type: &'static str, value: Value) -> Event {
    Event::default().event(event_type).data(value.to_string())
}

fn stream_error(error: &ApiError) -> Event {
    anthropic_event(
        "error",
        json!({"type": "error", "error": {"type": error.error_type, "message": error.message}}),
    )
}

struct ApiError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
    request_id: Option<String>,
}

impl ApiError {
    fn new(status: StatusCode, error_type: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type,
            message: message.into(),
            request_id: None,
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    fn unsupported(parameter: &str) -> Self {
        Self::invalid(format!("unsupported parameter: {parameter}"))
    }

    fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status;
        let request_id = self.request_id.unwrap_or_else(|| "req_unknown".to_owned());
        let mut response = (
            status,
            Json(json!({
                "type": "error",
                "error": {"type": self.error_type, "message": self.message},
                "request_id": request_id,
            })),
        )
            .into_response();
        if let Ok(value) = HeaderValue::from_str(&request_id) {
            response.headers_mut().insert(REQUEST_ID_HEADER, value);
        }
        if matches!(status.as_u16(), 429 | 503 | 529) {
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
            "not_found_error",
            format!("model was not found: {model}"),
        ),
        InferenceError::ModelIo(ModelIoError::InvalidInput(message)) => {
            ApiError::invalid(message.clone())
        }
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
            "invalid_request_error",
            error.to_string(),
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
            "timeout_error",
            error.to_string(),
        ),
        InferenceError::Admission(error) => api_error_from_admission(error),
        InferenceError::Planning(_)
        | InferenceError::Discovery(_)
        | InferenceError::Engine(
            locus_engine::EngineError::Draining | locus_engine::EngineError::Stopped,
        ) => ApiError::new(
            StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
            "overloaded_error",
            error.to_string(),
        ),
        _ => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            error.to_string(),
        ),
    }
}

fn api_error_from_admission(error: &AdmissionError) -> ApiError {
    match error {
        AdmissionError::Cancelled => ApiError::new(
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            "invalid_request_error",
            error.to_string(),
        ),
        AdmissionError::DeadlineExceeded => ApiError::new(
            StatusCode::REQUEST_TIMEOUT,
            "timeout_error",
            error.to_string(),
        ),
        AdmissionError::RequestTokenLimit { .. }
        | AdmissionError::RequestExceedsCapacity
        | AdmissionError::InvalidDeadline => ApiError::invalid(error.to_string()),
        AdmissionError::QueueFull => ApiError::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            error.to_string(),
        ),
        AdmissionError::OverloadShed => ApiError::new(
            StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
            "overloaded_error",
            error.to_string(),
        ),
        AdmissionError::Draining | AdmissionError::Unavailable => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "api_error",
            error.to_string(),
        ),
        AdmissionError::MissingTrustedTenant | AdmissionError::UnknownTenant(_) => ApiError::new(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            error.to_string(),
        ),
    }
}

fn api_error_from_json_rejection(error: &JsonRejection) -> ApiError {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            error.body_text(),
        )
    } else {
        ApiError::invalid(error.body_text())
    }
}

#[derive(Debug, Error)]
pub enum ApiConfigError {
    #[error("Anthropic API bearer token must not be empty")]
    EmptyBearerToken,
    #[error("Anthropic API request body limit must be greater than zero")]
    ZeroRequestBodyLimit,
    #[error("legacy bearer_token and tenant_credentials cannot be configured together")]
    AmbiguousCredentials,
    #[error("tenant credential references unknown traffic policy: {0}")]
    UnknownTenant(String),
    #[error("tenant credentials must be unique")]
    DuplicateCredential,
}
