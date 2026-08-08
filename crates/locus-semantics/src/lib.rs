use locus_core::{
    EngineFinishReason, InputBundle, ModelExecutionIdentity, SemanticComponentIdentity,
    SemanticIdentity, TokenSequence,
};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelProfile {
    pub public_aliases: Vec<String>,
    pub model: ModelExecutionIdentity,
    pub semantic_identity: SemanticIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conversation {
    pub messages: Vec<ConversationMessage>,
}

pub trait TokenizerProvider: Send + Sync {
    fn identity(&self) -> &SemanticComponentIdentity;

    fn encode(&self, input: &str) -> Result<TokenSequence, SemanticError>;
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

pub trait ModelSemantics: Send + Sync {
    fn profile(&self) -> &ModelProfile;

    fn normalize_input(&self, conversation: &Conversation) -> Result<InputBundle, SemanticError>;

    fn interpret_finish(&self, reason: &EngineFinishReason) -> SemanticFinishReason;
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SemanticError {
    #[error("invalid semantic input: {0}")]
    InvalidInput(String),
    #[error("semantic capability is unsupported: {0}")]
    Unsupported(String),
    #[error("semantic processing failed: {0}")]
    Processing(String),
}
