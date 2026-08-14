use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use locus_core::{
    CanonicalRequest, CapabilityRequirements, EngineEvent, EngineFinishReason, InputBundle,
    InputItem, InputItemId, InputItemValue, ModelExecutionIdentity, RequestId, SamplingParameters,
    SemanticComponentIdentity, SemanticIdentity, TokenSequence, TypedMetadata, Usage,
};
use locus_parser::{
    ParserError, ReasoningSegment, TaggedJsonToolParserDefinition, TaggedJsonToolParserState,
    TaggedReasoningParserDefinition, TaggedReasoningParserState, ToolSegment,
};
use thiserror::Error;

pub mod hf;

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
pub enum PromptInput {
    Text(String),
    TokenIds(Vec<u32>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelInput {
    Conversation(Conversation),
    Prompt(PromptInput),
}

impl Default for ModelInput {
    fn default() -> Self {
        Self::Conversation(Conversation::default())
    }
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
pub struct ModelRequest {
    pub model: String,
    pub input: ModelInput,
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
pub struct NormalizedModelRequest {
    pub canonical: CanonicalRequest,
    pub output_contract: OutputContract,
}

pub trait TokenizerProvider: Send + Sync {
    fn identity(&self) -> &SemanticComponentIdentity;

    fn encode(&self, input: &str) -> Result<TokenSequence, ModelIoError>;
}

pub trait TokenDecoder: Send + Sync {
    fn decode(&self, token_ids: &[u32]) -> Result<String, ModelIoError>;
}

pub trait TemplateRenderer: Send + Sync {
    fn identity(&self) -> &SemanticComponentIdentity;

    fn render(&self, request: &ModelRequest) -> Result<String, ModelIoError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelFinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Cancelled,
    Error,
    Namespaced { namespace: String, value: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelEvent {
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
        reason: ModelFinishReason,
        usage: Usage,
    },
}

pub trait ModelOutputPipeline: Send {
    fn process(&mut self, event: EngineEvent) -> Result<Vec<ModelEvent>, ModelIoError>;
}

pub trait ModelIo: Send + Sync {
    fn profile(&self) -> &ModelProfile;

    fn normalize(
        &self,
        request: &ModelRequest,
        request_id: RequestId,
    ) -> Result<NormalizedModelRequest, ModelIoError>;

    fn output_pipeline(
        &self,
        contract: &OutputContract,
    ) -> Result<Box<dyn ModelOutputPipeline>, ModelIoError>;
}

/// Resolves public model names to immutable semantic profiles.
///
/// A catalog is intentionally independent from engine inventory: knowing how to
/// normalize a model does not imply that an execution target is currently
/// available for it.
pub trait ModelCatalog: Send + Sync {
    fn resolve(&self, alias: &str) -> Result<Arc<dyn ModelIo>, ModelIoError>;

    fn profiles(&self) -> Result<Vec<ModelProfile>, ModelIoError>;
}

#[derive(Clone, Default)]
pub struct ModelRegistry {
    inner: Arc<RwLock<ModelRegistryInner>>,
}

#[derive(Default)]
struct ModelRegistryInner {
    aliases: BTreeMap<String, Arc<dyn ModelIo>>,
    profiles: BTreeMap<String, ModelProfile>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, model_io: Arc<dyn ModelIo>) -> Result<(), ModelIoError> {
        let profile = model_io.profile().clone();
        let primary = profile
            .public_aliases
            .first()
            .ok_or_else(|| ModelIoError::InvalidInput("model has no public alias".to_owned()))?
            .clone();
        let mut inner = self
            .inner
            .write()
            .map_err(|_| ModelIoError::Processing("model registry lock poisoned".to_owned()))?;
        for alias in &profile.public_aliases {
            if inner.aliases.contains_key(alias) {
                return Err(ModelIoError::InvalidInput(format!(
                    "model alias is already registered: {alias}"
                )));
            }
        }
        for alias in &profile.public_aliases {
            inner.aliases.insert(alias.clone(), Arc::clone(&model_io));
        }
        inner.profiles.insert(primary, profile);
        Ok(())
    }

    pub fn resolve(&self, alias: &str) -> Result<Arc<dyn ModelIo>, ModelIoError> {
        self.inner
            .read()
            .map_err(|_| ModelIoError::Processing("model registry lock poisoned".to_owned()))?
            .aliases
            .get(alias)
            .cloned()
            .ok_or_else(|| ModelIoError::ModelNotFound(alias.to_owned()))
    }

    pub fn profiles(&self) -> Result<Vec<ModelProfile>, ModelIoError> {
        Ok(self
            .inner
            .read()
            .map_err(|_| ModelIoError::Processing("model registry lock poisoned".to_owned()))?
            .profiles
            .values()
            .cloned()
            .collect())
    }
}

impl ModelCatalog for ModelRegistry {
    fn resolve(&self, alias: &str) -> Result<Arc<dyn ModelIo>, ModelIoError> {
        Self::resolve(self, alias)
    }

    fn profiles(&self) -> Result<Vec<ModelProfile>, ModelIoError> {
        Self::profiles(self)
    }
}

pub struct BasicModelIo {
    profile: ModelProfile,
    tokenizer: Arc<dyn TokenizerProvider>,
    template: Arc<dyn TemplateRenderer>,
    decoder: Arc<dyn TokenDecoder>,
    reasoning_parser: Option<TaggedReasoningParserDefinition>,
    tool_parser: Option<TaggedJsonToolParserDefinition>,
}

impl BasicModelIo {
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
            reasoning_parser: None,
            tool_parser: None,
        }
    }

    pub fn with_output_parsers(
        mut self,
        reasoning_parser: Option<TaggedReasoningParserDefinition>,
        tool_parser: Option<TaggedJsonToolParserDefinition>,
    ) -> Result<Self, ModelIoError> {
        validate_parser_identity(
            "reasoning",
            self.profile
                .semantic_identity
                .output
                .reasoning_parser
                .as_ref(),
            reasoning_parser
                .as_ref()
                .map(TaggedReasoningParserDefinition::identity),
        )?;
        validate_parser_identity(
            "tool",
            self.profile.semantic_identity.output.tool_parser.as_ref(),
            tool_parser
                .as_ref()
                .map(TaggedJsonToolParserDefinition::identity),
        )?;
        self.reasoning_parser = reasoning_parser;
        self.tool_parser = tool_parser;
        Ok(self)
    }
}

fn validate_parser_identity(
    kind: &str,
    profile: Option<&SemanticComponentIdentity>,
    parser: Option<&SemanticComponentIdentity>,
) -> Result<(), ModelIoError> {
    if profile != parser {
        return Err(ModelIoError::InvalidInput(format!(
            "{kind} parser definition does not match the model profile identity"
        )));
    }
    Ok(())
}

impl ModelIo for BasicModelIo {
    fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    fn normalize(
        &self,
        request: &ModelRequest,
        request_id: RequestId,
    ) -> Result<NormalizedModelRequest, ModelIoError> {
        validate_tool_contract(request)?;
        let (tokens, semantic_identity) = match &request.input {
            ModelInput::Conversation(conversation) => {
                if conversation.messages.is_empty() {
                    return Err(ModelIoError::InvalidInput(
                        "conversation must contain at least one message".to_owned(),
                    ));
                }
                let rendered = self.template.render(request)?;
                (
                    self.tokenizer.encode(&rendered)?,
                    self.profile.semantic_identity.clone(),
                )
            }
            ModelInput::Prompt(prompt) => {
                validate_prompt_contract(request)?;
                let tokens = match prompt {
                    PromptInput::Text(text) => self.tokenizer.encode(text)?,
                    PromptInput::TokenIds(token_ids) => TokenSequence {
                        token_ids: token_ids.clone(),
                        tokenizer_fingerprint: self.tokenizer.identity().fingerprint.clone(),
                    },
                };
                (
                    tokens,
                    raw_prompt_semantic_identity(&self.profile.semantic_identity),
                )
            }
        };
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
        requirements.requires_reasoning_deltas =
            request.reasoning_effort.is_some() && self.reasoning_parser.is_none();
        requirements.requires_tool_calls = !request.tools.is_empty() && self.tool_parser.is_none();
        requirements.requires_structured_output =
            !matches!(request.response_format, ResponseFormat::Text);
        Ok(NormalizedModelRequest {
            canonical: CanonicalRequest {
                id: request_id,
                model: self.profile.model.clone(),
                semantic_identity,
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
    ) -> Result<Box<dyn ModelOutputPipeline>, ModelIoError> {
        Ok(Box::new(BasicOutputPipeline {
            decoder: Arc::clone(&self.decoder),
            contract: contract.clone(),
            text: String::new(),
            tool_arguments: BTreeMap::new(),
            saw_tool_call: false,
            reasoning_parser: self
                .reasoning_parser
                .as_ref()
                .map(TaggedReasoningParserDefinition::state),
            tool_parser: self
                .tool_parser
                .as_ref()
                .map(TaggedJsonToolParserDefinition::state),
            next_tool_index: 0,
        }))
    }
}

fn validate_prompt_contract(request: &ModelRequest) -> Result<(), ModelIoError> {
    if !request.tools.is_empty()
        || !matches!(request.tool_choice, ToolChoice::Auto | ToolChoice::None)
    {
        return Err(ModelIoError::Unsupported(
            "raw prompts do not support function tools".to_owned(),
        ));
    }
    if !matches!(request.response_format, ResponseFormat::Text) {
        return Err(ModelIoError::Unsupported(
            "raw prompts do not support structured output".to_owned(),
        ));
    }
    if request.reasoning_effort.is_some() {
        return Err(ModelIoError::Unsupported(
            "raw prompts do not support reasoning controls".to_owned(),
        ));
    }
    Ok(())
}

fn raw_prompt_semantic_identity(profile: &SemanticIdentity) -> SemanticIdentity {
    let mut identity = profile.clone();
    identity.input.template = SemanticComponentIdentity {
        kind: "locus-raw-prompt".to_owned(),
        revision: "v1".to_owned(),
        fingerprint: "locus-raw-prompt-v1".to_owned(),
    };
    identity.umbrella_fingerprint = None;
    identity
}

fn contract_annotations(request: &ModelRequest) -> Result<Vec<TypedMetadata>, ModelIoError> {
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
                ModelIoError::InvalidInput(format!("invalid JSON schema: {error}"))
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
            ModelIoError::InvalidInput(format!(
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

fn validate_tool_contract(request: &ModelRequest) -> Result<(), ModelIoError> {
    let mut names = std::collections::BTreeSet::new();
    for tool in &request.tools {
        if tool.name.trim().is_empty() {
            return Err(ModelIoError::InvalidInput(
                "tool name must not be empty".to_owned(),
            ));
        }
        let schema = serde_json::from_str::<serde_json::Value>(&tool.parameters_schema).map_err(
            |error| {
                ModelIoError::InvalidInput(format!(
                    "invalid parameters schema for tool {}: {error}",
                    tool.name
                ))
            },
        )?;
        if !schema.is_object() {
            return Err(ModelIoError::InvalidInput(format!(
                "parameters schema for tool {} must be a JSON object",
                tool.name
            )));
        }
        if !names.insert(tool.name.as_str()) {
            return Err(ModelIoError::InvalidInput(format!(
                "duplicate tool name: {}",
                tool.name
            )));
        }
    }
    match &request.tool_choice {
        ToolChoice::Required if request.tools.is_empty() => Err(ModelIoError::InvalidInput(
            "tool_choice required needs at least one tool".to_owned(),
        )),
        ToolChoice::Function(name) if !names.contains(name.as_str()) => {
            Err(ModelIoError::InvalidInput(format!(
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
    reasoning_parser: Option<TaggedReasoningParserState>,
    tool_parser: Option<TaggedJsonToolParserState>,
    next_tool_index: usize,
}

impl ModelOutputPipeline for BasicOutputPipeline {
    fn process(&mut self, event: EngineEvent) -> Result<Vec<ModelEvent>, ModelIoError> {
        let semantic = match event {
            EngineEvent::Accepted { request_id } => vec![ModelEvent::Accepted { request_id }],
            EngineEvent::TokenDelta {
                request_id,
                token_ids,
                ..
            } => {
                let text = self.decoder.decode(&token_ids)?;
                self.process_text_delta(request_id, text)?
            }
            EngineEvent::TextDelta {
                request_id, text, ..
            } => self.process_text_delta(request_id, text)?,
            EngineEvent::ReasoningDelta {
                request_id, text, ..
            } => {
                if self.reasoning_parser.is_some() {
                    return Err(ModelIoError::Processing(
                        "engine emitted native reasoning while a profile parser is selected"
                            .to_owned(),
                    ));
                }
                vec![ModelEvent::ReasoningDelta { request_id, text }]
            }
            EngineEvent::ToolCallStarted {
                request_id,
                call_id,
                name,
                ..
            } => {
                if self.tool_parser.is_some() {
                    return Err(ModelIoError::Processing(
                        "engine emitted a native tool call while a profile parser is selected"
                            .to_owned(),
                    ));
                }
                vec![self.start_tool_call(request_id, call_id, name)?]
            }
            EngineEvent::ToolCallArgumentsDelta {
                request_id,
                call_id,
                delta,
                ..
            } => vec![self.append_tool_arguments(request_id, call_id, delta)?],
            EngineEvent::ToolCallCompleted {
                request_id,
                call_id,
                ..
            } => vec![self.complete_tool_call(request_id, call_id)?],
            EngineEvent::UsageUpdate {
                request_id,
                usage,
                final_update,
            } => vec![ModelEvent::Usage {
                request_id,
                usage,
                final_update,
            }],
            EngineEvent::Finished {
                request_id,
                reason,
                usage,
            } => {
                let mut events = self.finish_parsers(&request_id)?;
                self.validate_completed_output()?;
                let runtime_reported_tool_call = matches!(
                    &reason,
                    EngineFinishReason::RuntimeSpecific { value, .. } if value == "tool_calls"
                );
                if runtime_reported_tool_call && !self.saw_tool_call {
                    return Err(ModelIoError::Processing(
                        "engine reported tool_calls without a completed tool call".to_owned(),
                    ));
                }
                let reason = if self.saw_tool_call
                    && (reason == EngineFinishReason::Stop || runtime_reported_tool_call)
                {
                    ModelFinishReason::ToolCall
                } else {
                    interpret_finish(reason)
                };
                events.push(ModelEvent::Finished {
                    request_id,
                    reason,
                    usage,
                });
                events
            }
        };
        Ok(semantic)
    }
}

impl BasicOutputPipeline {
    fn process_text_delta(
        &mut self,
        request_id: RequestId,
        text: String,
    ) -> Result<Vec<ModelEvent>, ModelIoError> {
        let segments = if let Some(parser) = self.reasoning_parser.as_mut() {
            parser.push(&text)?
        } else {
            vec![ReasoningSegment::Text(text)]
        };
        self.process_reasoning_segments(&request_id, segments)
    }

    fn process_reasoning_segments(
        &mut self,
        request_id: &RequestId,
        segments: Vec<ReasoningSegment>,
    ) -> Result<Vec<ModelEvent>, ModelIoError> {
        let mut events = Vec::new();
        for segment in segments {
            match segment {
                ReasoningSegment::Reasoning(text) => {
                    events.push(ModelEvent::ReasoningDelta {
                        request_id: request_id.clone(),
                        text,
                    });
                }
                ReasoningSegment::Text(text) => {
                    events.extend(self.process_visible_text(request_id, &text)?);
                }
            }
        }
        Ok(events)
    }

    fn process_visible_text(
        &mut self,
        request_id: &RequestId,
        text: &str,
    ) -> Result<Vec<ModelEvent>, ModelIoError> {
        let segments = if let Some(parser) = self.tool_parser.as_mut() {
            parser.push(text)?
        } else {
            vec![ToolSegment::Text(text.to_owned())]
        };
        let mut events = Vec::new();
        for segment in segments {
            match segment {
                ToolSegment::Text(text) => {
                    self.text.push_str(&text);
                    events.push(ModelEvent::TextDelta {
                        request_id: request_id.clone(),
                        text,
                    });
                }
                ToolSegment::Call { name, arguments } => {
                    let call_id = format!("call_{}_{}", request_id.as_str(), self.next_tool_index);
                    self.next_tool_index += 1;
                    events.push(self.start_tool_call(request_id.clone(), call_id.clone(), name)?);
                    events.push(self.append_tool_arguments(
                        request_id.clone(),
                        call_id.clone(),
                        arguments,
                    )?);
                    events.push(self.complete_tool_call(request_id.clone(), call_id)?);
                }
            }
        }
        Ok(events)
    }

    fn start_tool_call(
        &mut self,
        request_id: RequestId,
        call_id: String,
        name: String,
    ) -> Result<ModelEvent, ModelIoError> {
        let requested = self.contract.tools.iter().any(|tool| tool.name == name);
        let selected = match &self.contract.tool_choice {
            ToolChoice::None => false,
            ToolChoice::Function(selected) => selected == &name,
            ToolChoice::Auto | ToolChoice::Required => requested,
        };
        if !requested || !selected {
            return Err(ModelIoError::Processing(format!(
                "engine emitted an unrequested tool call: {name}"
            )));
        }
        if self.tool_arguments.contains_key(&call_id) {
            return Err(ModelIoError::Processing(format!(
                "engine emitted duplicate tool call id: {call_id}"
            )));
        }
        self.saw_tool_call = true;
        self.tool_arguments.insert(call_id.clone(), String::new());
        Ok(ModelEvent::ToolCallStarted {
            request_id,
            call_id,
            name,
        })
    }

    fn append_tool_arguments(
        &mut self,
        request_id: RequestId,
        call_id: String,
        delta: String,
    ) -> Result<ModelEvent, ModelIoError> {
        let arguments = self.tool_arguments.get_mut(&call_id).ok_or_else(|| {
            ModelIoError::Processing(format!(
                "tool arguments arrived before the call was started: {call_id}"
            ))
        })?;
        arguments.push_str(&delta);
        Ok(ModelEvent::ToolCallArgumentsDelta {
            request_id,
            call_id,
            delta,
        })
    }

    fn complete_tool_call(
        &mut self,
        request_id: RequestId,
        call_id: String,
    ) -> Result<ModelEvent, ModelIoError> {
        let arguments = self.tool_arguments.remove(&call_id).ok_or_else(|| {
            ModelIoError::Processing(format!(
                "tool call completed before it was started: {call_id}"
            ))
        })?;
        let value = serde_json::from_str::<serde_json::Value>(&arguments).map_err(|error| {
            ModelIoError::Processing(format!(
                "tool call {call_id} produced invalid JSON arguments: {error}"
            ))
        })?;
        if !value.is_object() {
            return Err(ModelIoError::Processing(format!(
                "tool call {call_id} arguments must be a JSON object"
            )));
        }
        Ok(ModelEvent::ToolCallCompleted {
            request_id,
            call_id,
            arguments,
        })
    }

    fn finish_parsers(&mut self, request_id: &RequestId) -> Result<Vec<ModelEvent>, ModelIoError> {
        let reasoning = if let Some(parser) = self.reasoning_parser.as_mut() {
            parser.finish()?
        } else {
            Vec::new()
        };
        let mut events = self.process_reasoning_segments(request_id, reasoning)?;
        let tools = if let Some(parser) = self.tool_parser.as_mut() {
            parser.finish()?
        } else {
            Vec::new()
        };
        for segment in tools {
            match segment {
                ToolSegment::Text(text) => {
                    self.text.push_str(&text);
                    events.push(ModelEvent::TextDelta {
                        request_id: request_id.clone(),
                        text,
                    });
                }
                ToolSegment::Call { .. } => {
                    return Err(ModelIoError::Processing(
                        "tool parser committed a call after finalization".to_owned(),
                    ));
                }
            }
        }
        Ok(events)
    }

    fn validate_completed_output(&self) -> Result<(), ModelIoError> {
        if matches!(
            self.contract.response_format,
            ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. }
        ) {
            let value = serde_json::from_str::<serde_json::Value>(&self.text).map_err(|error| {
                ModelIoError::Processing(format!("structured output is not valid JSON: {error}"))
            })?;
            if !value.is_object() {
                return Err(ModelIoError::Processing(
                    "structured output must be a JSON object".to_owned(),
                ));
            }
        }
        if !self.tool_arguments.is_empty() {
            return Err(ModelIoError::Processing(
                "engine finished with incomplete tool calls".to_owned(),
            ));
        }
        if matches!(
            self.contract.tool_choice,
            ToolChoice::Required | ToolChoice::Function(_)
        ) && !self.saw_tool_call
        {
            return Err(ModelIoError::Processing(
                "engine finished without the required tool call".to_owned(),
            ));
        }
        Ok(())
    }
}

fn interpret_finish(reason: EngineFinishReason) -> ModelFinishReason {
    match reason {
        EngineFinishReason::Stop => ModelFinishReason::Stop,
        EngineFinishReason::Length => ModelFinishReason::Length,
        EngineFinishReason::Cancelled => ModelFinishReason::Cancelled,
        EngineFinishReason::Error => ModelFinishReason::Error,
        EngineFinishReason::RuntimeSpecific { namespace, value } => {
            ModelFinishReason::Namespaced { namespace, value }
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

    fn encode(&self, input: &str) -> Result<TokenSequence, ModelIoError> {
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
    fn decode(&self, token_ids: &[u32]) -> Result<String, ModelIoError> {
        let bytes = token_ids
            .iter()
            .map(|token| {
                u8::try_from(*token).map_err(|_| {
                    ModelIoError::Processing(format!("token {token} is not a byte token"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        String::from_utf8(bytes).map_err(|error| {
            ModelIoError::Processing(format!("token delta is not valid UTF-8: {error}"))
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

    fn render(&self, request: &ModelRequest) -> Result<String, ModelIoError> {
        let ModelInput::Conversation(conversation) = &request.input else {
            return Err(ModelIoError::InvalidInput(
                "chat template requires conversation input".to_owned(),
            ));
        };
        let mut rendered = String::new();
        for message in &conversation.messages {
            if message.content.is_empty() {
                return Err(ModelIoError::InvalidInput(format!(
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
pub enum ModelIoError {
    #[error("model was not found: {0}")]
    ModelNotFound(String),
    #[error("invalid model input: {0}")]
    InvalidInput(String),
    #[error("model I/O capability is unsupported: {0}")]
    Unsupported(String),
    #[error("model I/O processing failed: {0}")]
    Processing(String),
}

impl From<ParserError> for ModelIoError {
    fn from(error: ParserError) -> Self {
        match error {
            ParserError::InvalidConfig(message) => Self::InvalidInput(message),
            ParserError::Processing(message) => Self::Processing(message),
        }
    }
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
            reasoning_parser: None,
            tool_parser: None,
            next_tool_index: 0,
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
            ModelEvent::ToolCallCompleted { arguments, .. }
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
            reasoning_parser: None,
            tool_parser: None,
            next_tool_index: 0,
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
