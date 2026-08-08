use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use locus_core::{
    CanonicalRequest, CapabilityRequirements, EngineEvent, EngineFinishReason, InputBundle,
    InputItem, InputItemId, InputItemValue, ModelExecutionIdentity, RequestId, SamplingParameters,
    SemanticComponentIdentity, SemanticIdentity, TokenSequence, TypedMetadata, Usage,
};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelProfile {
    pub public_aliases: Vec<String>,
    pub model: ModelExecutionIdentity,
    pub semantic_identity: SemanticIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConversationRole {
    Developer,
    System,
    User,
    Assistant,
    Tool,
}

impl ConversationRole {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Developer => "developer",
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    pub content: String,
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conversation {
    pub messages: Vec<ConversationMessage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters_schema: String,
    pub strict: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Function(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: String,
        strict: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticRequest {
    pub model: String,
    pub conversation: Conversation,
    pub sampling: SamplingParameters,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub response_format: ResponseFormat,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OutputContract {
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub response_format: ResponseFormat,
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedSemanticRequest {
    pub canonical: CanonicalRequest,
    pub output_contract: OutputContract,
}

pub trait TokenizerProvider: Send + Sync {
    fn identity(&self) -> &SemanticComponentIdentity;

    fn encode(&self, input: &str) -> Result<TokenSequence, SemanticError>;
}

pub trait TokenDecoder: Send + Sync {
    fn decode(&self, token_ids: &[u32]) -> Result<String, SemanticError>;
}

pub trait TemplateRenderer: Send + Sync {
    fn identity(&self) -> &SemanticComponentIdentity;

    fn render(&self, conversation: &Conversation) -> Result<String, SemanticError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticFinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Cancelled,
    Error,
    Namespaced { namespace: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticEvent {
    Accepted {
        request_id: RequestId,
    },
    TextDelta {
        request_id: RequestId,
        text: String,
    },
    ReasoningDelta {
        request_id: RequestId,
        text: String,
    },
    ToolCallStarted {
        request_id: RequestId,
        call_id: String,
        name: String,
    },
    ToolCallArgumentsDelta {
        request_id: RequestId,
        call_id: String,
        delta: String,
    },
    ToolCallCompleted {
        request_id: RequestId,
        call_id: String,
        arguments: String,
    },
    Usage {
        request_id: RequestId,
        usage: Usage,
        final_update: bool,
    },
    Finished {
        request_id: RequestId,
        reason: SemanticFinishReason,
        usage: Usage,
    },
}

pub trait SemanticOutputPipeline: Send {
    fn process(&mut self, event: EngineEvent) -> Result<Vec<SemanticEvent>, SemanticError>;
}

pub trait ModelSemantics: Send + Sync {
    fn profile(&self) -> &ModelProfile;

    fn normalize(
        &self,
        request: &SemanticRequest,
        request_id: RequestId,
    ) -> Result<NormalizedSemanticRequest, SemanticError>;

    fn output_pipeline(
        &self,
        contract: &OutputContract,
    ) -> Result<Box<dyn SemanticOutputPipeline>, SemanticError>;
}

#[derive(Clone, Default)]
pub struct ModelRegistry {
    inner: Arc<RwLock<ModelRegistryInner>>,
}

#[derive(Default)]
struct ModelRegistryInner {
    aliases: BTreeMap<String, Arc<dyn ModelSemantics>>,
    profiles: BTreeMap<String, ModelProfile>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, semantics: Arc<dyn ModelSemantics>) -> Result<(), SemanticError> {
        let profile = semantics.profile().clone();
        let primary = profile
            .public_aliases
            .first()
            .ok_or_else(|| SemanticError::InvalidInput("model has no public alias".to_owned()))?
            .clone();
        let mut inner = self
            .inner
            .write()
            .map_err(|_| SemanticError::Processing("model registry lock poisoned".to_owned()))?;
        for alias in &profile.public_aliases {
            if inner.aliases.contains_key(alias) {
                return Err(SemanticError::InvalidInput(format!(
                    "model alias is already registered: {alias}"
                )));
            }
        }
        for alias in &profile.public_aliases {
            inner.aliases.insert(alias.clone(), Arc::clone(&semantics));
        }
        inner.profiles.insert(primary, profile);
        Ok(())
    }

    pub fn resolve(&self, alias: &str) -> Result<Arc<dyn ModelSemantics>, SemanticError> {
        self.inner
            .read()
            .map_err(|_| SemanticError::Processing("model registry lock poisoned".to_owned()))?
            .aliases
            .get(alias)
            .cloned()
            .ok_or_else(|| SemanticError::ModelNotFound(alias.to_owned()))
    }

    pub fn profiles(&self) -> Result<Vec<ModelProfile>, SemanticError> {
        Ok(self
            .inner
            .read()
            .map_err(|_| SemanticError::Processing("model registry lock poisoned".to_owned()))?
            .profiles
            .values()
            .cloned()
            .collect())
    }
}

pub struct BasicModelSemantics {
    profile: ModelProfile,
    tokenizer: Arc<dyn TokenizerProvider>,
    template: Arc<dyn TemplateRenderer>,
    decoder: Arc<dyn TokenDecoder>,
}

impl BasicModelSemantics {
    #[must_use]
    pub fn new(
        profile: ModelProfile,
        tokenizer: Arc<dyn TokenizerProvider>,
        template: Arc<dyn TemplateRenderer>,
        decoder: Arc<dyn TokenDecoder>,
    ) -> Self {
        Self {
            profile,
            tokenizer,
            template,
            decoder,
        }
    }
}

impl ModelSemantics for BasicModelSemantics {
    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    fn normalize(
        &self,
        request: &SemanticRequest,
        request_id: RequestId,
    ) -> Result<NormalizedSemanticRequest, SemanticError> {
        if request.conversation.messages.is_empty() {
            return Err(SemanticError::InvalidInput(
                "conversation must contain at least one message".to_owned(),
            ));
        }
        validate_tool_contract(request)?;
        let rendered = self.template.render(&request.conversation)?;
        let tokens = self.tokenizer.encode(&rendered)?;
        let mut input = InputBundle {
            items: vec![InputItem {
                id: InputItemId::new("prompt"),
                value: InputItemValue::TokenSequence(tokens),
            }],
            ..InputBundle::default()
        };
        input.annotations = contract_annotations(request)?;
        let contract = OutputContract {
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            response_format: request.response_format.clone(),
            reasoning_effort: request.reasoning_effort,
        };
        let mut requirements = CapabilityRequirements::for_input(&input);
        requirements.requires_reasoning_deltas = request.reasoning_effort.is_some();
        requirements.requires_tool_calls = !request.tools.is_empty();
        requirements.requires_structured_output =
            !matches!(request.response_format, ResponseFormat::Text);
        Ok(NormalizedSemanticRequest {
            canonical: CanonicalRequest {
                id: request_id,
                model: self.profile.model.clone(),
                semantic_identity: self.profile.semantic_identity.clone(),
                requirements,
                input,
                sampling: request.sampling.clone(),
            },
            output_contract: contract,
        })
    }

    fn output_pipeline(
        &self,
        contract: &OutputContract,
    ) -> Result<Box<dyn SemanticOutputPipeline>, SemanticError> {
        Ok(Box::new(BasicOutputPipeline {
            decoder: Arc::clone(&self.decoder),
            contract: contract.clone(),
            text: String::new(),
            tool_arguments: BTreeMap::new(),
            saw_tool_call: false,
        }))
    }
}

fn contract_annotations(request: &SemanticRequest) -> Result<Vec<TypedMetadata>, SemanticError> {
    let mut annotations = Vec::new();
    let mut contract = BTreeMap::new();
    contract.insert(
        "tool_choice".to_owned(),
        match &request.tool_choice {
            ToolChoice::Auto => "auto".to_owned(),
            ToolChoice::None => "none".to_owned(),
            ToolChoice::Required => "required".to_owned(),
            ToolChoice::Function(name) => format!("function:{name}"),
        },
    );
    match &request.response_format {
        ResponseFormat::Text => {
            contract.insert("response_format".to_owned(), "text".to_owned());
        }
        ResponseFormat::JsonObject => {
            contract.insert("response_format".to_owned(), "json_object".to_owned());
        }
        ResponseFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => {
            serde_json::from_str::<serde_json::Value>(schema).map_err(|error| {
                SemanticError::InvalidInput(format!("invalid JSON schema: {error}"))
            })?;
            contract.insert("response_format".to_owned(), "json_schema".to_owned());
            contract.insert("json_schema_name".to_owned(), name.clone());
            contract.insert("json_schema".to_owned(), schema.clone());
            contract.insert("strict".to_owned(), strict.to_string());
            if let Some(description) = description {
                contract.insert("json_schema_description".to_owned(), description.clone());
            }
        }
    }
    if let Some(effort) = request.reasoning_effort {
        contract.insert("reasoning_effort".to_owned(), effort.as_str().to_owned());
    }
    annotations.push(TypedMetadata {
        type_url: "locus.generation-contract.v1".to_owned(),
        fields: contract,
    });
    for tool in &request.tools {
        serde_json::from_str::<serde_json::Value>(&tool.parameters_schema).map_err(|error| {
            SemanticError::InvalidInput(format!(
                "invalid parameters schema for tool {}: {error}",
                tool.name
            ))
        })?;
        let mut fields = BTreeMap::new();
        fields.insert("name".to_owned(), tool.name.clone());
        fields.insert("parameters".to_owned(), tool.parameters_schema.clone());
        fields.insert("strict".to_owned(), tool.strict.to_string());
        if let Some(description) = &tool.description {
            fields.insert("description".to_owned(), description.clone());
        }
        annotations.push(TypedMetadata {
            type_url: "locus.tool.v1".to_owned(),
            fields,
        });
    }
    Ok(annotations)
}

fn validate_tool_contract(request: &SemanticRequest) -> Result<(), SemanticError> {
    let mut names = std::collections::BTreeSet::new();
    for tool in &request.tools {
        if tool.name.is_empty() {
            return Err(SemanticError::InvalidInput(
                "tool name must not be empty".to_owned(),
            ));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(SemanticError::InvalidInput(format!(
                "duplicate tool name: {}",
                tool.name
            )));
        }
    }
    match &request.tool_choice {
        ToolChoice::Required if request.tools.is_empty() => Err(SemanticError::InvalidInput(
            "tool_choice required needs at least one tool".to_owned(),
        )),
        ToolChoice::Function(name) if !names.contains(name.as_str()) => {
            Err(SemanticError::InvalidInput(format!(
                "tool_choice references an unknown function: {name}"
            )))
        }
        _ => Ok(()),
    }
}

struct BasicOutputPipeline {
    decoder: Arc<dyn TokenDecoder>,
    contract: OutputContract,
    text: String,
    tool_arguments: BTreeMap<String, String>,
    saw_tool_call: bool,
}

impl SemanticOutputPipeline for BasicOutputPipeline {
    fn process(&mut self, event: EngineEvent) -> Result<Vec<SemanticEvent>, SemanticError> {
        let semantic = match event {
            EngineEvent::Accepted { request_id } => vec![SemanticEvent::Accepted { request_id }],
            EngineEvent::TokenDelta {
                request_id,
                token_ids,
                ..
            } => {
                let text = self.decoder.decode(&token_ids)?;
                self.text.push_str(&text);
                vec![SemanticEvent::TextDelta { request_id, text }]
            }
            EngineEvent::TextDelta {
                request_id, text, ..
            } => {
                self.text.push_str(&text);
                vec![SemanticEvent::TextDelta { request_id, text }]
            }
            EngineEvent::ReasoningDelta {
                request_id, text, ..
            } => vec![SemanticEvent::ReasoningDelta { request_id, text }],
            EngineEvent::ToolCallStarted {
                request_id,
                call_id,
                name,
                ..
            } => {
                if matches!(self.contract.tool_choice, ToolChoice::None)
                    || !self.contract.tools.iter().any(|tool| tool.name == name)
                {
                    return Err(SemanticError::Processing(format!(
                        "engine emitted an unrequested tool call: {name}"
                    )));
                }
                self.saw_tool_call = true;
                self.tool_arguments.insert(call_id.clone(), String::new());
                vec![SemanticEvent::ToolCallStarted {
                    request_id,
                    call_id,
                    name,
                }]
            }
            EngineEvent::ToolCallArgumentsDelta {
                request_id,
                call_id,
                delta,
                ..
            } => {
                self.tool_arguments
                    .entry(call_id.clone())
                    .or_default()
                    .push_str(&delta);
                vec![SemanticEvent::ToolCallArgumentsDelta {
                    request_id,
                    call_id,
                    delta,
                }]
            }
            EngineEvent::ToolCallCompleted {
                request_id,
                call_id,
                ..
            } => {
                let arguments = self.tool_arguments.remove(&call_id).ok_or_else(|| {
                    SemanticError::Processing(format!(
                        "tool call completed before it was started: {call_id}"
                    ))
                })?;
                serde_json::from_str::<serde_json::Value>(&arguments).map_err(|error| {
                    SemanticError::Processing(format!(
                        "tool call {call_id} produced invalid JSON arguments: {error}"
                    ))
                })?;
                vec![SemanticEvent::ToolCallCompleted {
                    request_id,
                    call_id,
                    arguments,
                }]
            }
            EngineEvent::UsageUpdate {
                request_id,
                usage,
                final_update,
            } => vec![SemanticEvent::Usage {
                request_id,
                usage,
                final_update,
            }],
            EngineEvent::Finished {
                request_id,
                reason,
                usage,
            } => {
                self.validate_completed_output()?;
                let reason = if self.saw_tool_call && reason == EngineFinishReason::Stop {
                    SemanticFinishReason::ToolCall
                } else {
                    interpret_finish(reason)
                };
                vec![SemanticEvent::Finished {
                    request_id,
                    reason,
                    usage,
                }]
            }
        };
        Ok(semantic)
    }
}

impl BasicOutputPipeline {
    fn validate_completed_output(&self) -> Result<(), SemanticError> {
        if matches!(
            self.contract.response_format,
            ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
        ) {
            let value = serde_json::from_str::<serde_json::Value>(&self.text).map_err(|error| {
                SemanticError::Processing(format!("structured output is not valid JSON: {error}"))
            })?;
            if !value.is_object() {
                return Err(SemanticError::Processing(
                    "structured output must be a JSON object".to_owned(),
                ));
            }
        }
        if !self.tool_arguments.is_empty() {
            return Err(SemanticError::Processing(
                "engine finished with incomplete tool calls".to_owned(),
            ));
        }
        Ok(())
    }
}

fn interpret_finish(reason: EngineFinishReason) -> SemanticFinishReason {
    match reason {
        EngineFinishReason::Stop => SemanticFinishReason::Stop,
        EngineFinishReason::Length => SemanticFinishReason::Length,
        EngineFinishReason::Cancelled => SemanticFinishReason::Cancelled,
        EngineFinishReason::Error => SemanticFinishReason::Error,
        EngineFinishReason::RuntimeSpecific { namespace, value } => {
            SemanticFinishReason::Namespaced { namespace, value }
        }
    }
}

#[derive(Clone)]
pub struct ByteTokenizer {
    identity: SemanticComponentIdentity,
}

impl ByteTokenizer {
    #[must_use]
    pub fn new(identity: SemanticComponentIdentity) -> Self {
        Self { identity }
    }
}

impl TokenizerProvider for ByteTokenizer {
    fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    fn encode(&self, input: &str) -> Result<TokenSequence, SemanticError> {
        Ok(TokenSequence {
            token_ids: input
                .as_bytes()
                .iter()
                .map(|byte| u32::from(*byte))
                .collect(),
            tokenizer_fingerprint: self.identity.fingerprint.clone(),
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ByteDecoder;

impl TokenDecoder for ByteDecoder {
    fn decode(&self, token_ids: &[u32]) -> Result<String, SemanticError> {
        let bytes = token_ids
            .iter()
            .map(|token| {
                u8::try_from(*token).map_err(|_| {
                    SemanticError::Processing(format!("token {token} is not a byte token"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        String::from_utf8(bytes).map_err(|error| {
            SemanticError::Processing(format!("token delta is not valid UTF-8: {error}"))
        })
    }
}

#[derive(Clone)]
pub struct SimpleTemplateRenderer {
    identity: SemanticComponentIdentity,
}

impl SimpleTemplateRenderer {
    #[must_use]
    pub fn new(identity: SemanticComponentIdentity) -> Self {
        Self { identity }
    }
}

impl TemplateRenderer for SimpleTemplateRenderer {
    fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    fn render(&self, conversation: &Conversation) -> Result<String, SemanticError> {
        let mut rendered = String::new();
        for message in &conversation.messages {
            if message.content.is_empty() {
                return Err(SemanticError::InvalidInput(format!(
                    "{} message content must not be empty",
                    message.role.as_str()
                )));
            }
            rendered.push('<');
            rendered.push_str(message.role.as_str());
            rendered.push_str(">\n");
            rendered.push_str(&message.content);
            rendered.push('\n');
        }
        rendered.push_str("<assistant>\n");
        Ok(rendered)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SemanticError {
    #[error("model was not found: {0}")]
    ModelNotFound(String),
    #[error("invalid semantic input: {0}")]
    InvalidInput(String),
    #[error("semantic capability is unsupported: {0}")]
    Unsupported(String),
    #[error("semantic processing failed: {0}")]
    Processing(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_argument_deltas_are_aggregated_and_validated() {
        let mut pipeline = BasicOutputPipeline {
            decoder: Arc::new(ByteDecoder),
            contract: OutputContract {
                tools: vec![ToolDefinition {
                    name: "weather".to_owned(),
                    description: None,
                    parameters_schema: "{\"type\":\"object\"}".to_owned(),
                    strict: true,
                }],
                ..OutputContract::default()
            },
            text: String::new(),
            tool_arguments: BTreeMap::new(),
            saw_tool_call: false,
        };
        let id = RequestId::new("req-1");
        pipeline
            .process(EngineEvent::ToolCallStarted {
                request_id: id.clone(),
                sequence_number: 1,
                call_id: "call-1".to_owned(),
                name: "weather".to_owned(),
            })
            .unwrap();
        pipeline
            .process(EngineEvent::ToolCallArgumentsDelta {
                request_id: id.clone(),
                sequence_number: 2,
                call_id: "call-1".to_owned(),
                delta: "{\"city\":".to_owned(),
            })
            .unwrap();
        pipeline
            .process(EngineEvent::ToolCallArgumentsDelta {
                request_id: id.clone(),
                sequence_number: 3,
                call_id: "call-1".to_owned(),
                delta: "\"Beijing\"}".to_owned(),
            })
            .unwrap();
        let events = pipeline
            .process(EngineEvent::ToolCallCompleted {
                request_id: id,
                sequence_number: 4,
                call_id: "call-1".to_owned(),
            })
            .unwrap();
        assert!(matches!(
            &events[0],
            SemanticEvent::ToolCallCompleted { arguments, .. }
                if arguments == "{\"city\":\"Beijing\"}"
        ));
    }

    #[test]
    fn structured_output_rejects_non_json_at_completion() {
        let mut pipeline = BasicOutputPipeline {
            decoder: Arc::new(ByteDecoder),
            contract: OutputContract {
                response_format: ResponseFormat::JsonObject,
                ..OutputContract::default()
            },
            text: String::new(),
            tool_arguments: BTreeMap::new(),
            saw_tool_call: false,
        };
        let id = RequestId::new("req-2");
        pipeline
            .process(EngineEvent::TextDelta {
                request_id: id.clone(),
                sequence_number: 1,
                text: "not-json".to_owned(),
            })
            .unwrap();
        let error = pipeline
            .process(EngineEvent::Finished {
                request_id: id,
                reason: EngineFinishReason::Stop,
                usage: Usage::default(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("not valid JSON"));
    }
}
