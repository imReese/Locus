use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use locus_core::{
    GenerationSemanticIdentity, InputSemanticIdentity, ModelExecutionIdentity,
    OutputSemanticIdentity, SemanticComponentIdentity, SemanticIdentity, TokenSequence,
};
use locus_semantics::{
    BasicModelSemantics, ModelProfile, ModelSemantics, SemanticError, SemanticInput,
    SemanticRequest, TaggedJsonToolParserDefinition, TaggedReasoningParserDefinition,
    TemplateRenderer, TokenDecoder, TokenizerProvider, ToolChoice,
};
use minijinja::{Environment, Error as TemplateError, ErrorKind, UndefinedBehavior};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

const DEFAULT_MAX_RENDERED_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct HuggingFaceProfileSpec {
    pub public_aliases: Vec<String>,
    pub model: ModelExecutionIdentity,
    pub tokenizer_json: PathBuf,
    pub tokenizer_revision: String,
    pub chat_template: PathBuf,
    pub template_revision: String,
    pub template_context: BTreeMap<String, Value>,
    pub add_generation_prompt: bool,
    pub max_rendered_bytes: usize,
    pub reasoning_parser: Option<TaggedReasoningParserSpec>,
    pub tool_parser: Option<TaggedJsonToolParserSpec>,
}

#[derive(Clone, Debug)]
pub struct TaggedReasoningParserSpec {
    pub revision: String,
    pub start_delimiter: String,
    pub end_delimiter: String,
}

#[derive(Clone, Debug)]
pub struct TaggedJsonToolParserSpec {
    pub revision: String,
    pub start_delimiter: String,
    pub end_delimiter: String,
    pub max_buffered_bytes: usize,
}

impl HuggingFaceProfileSpec {
    #[must_use]
    pub fn new(
        public_aliases: Vec<String>,
        model: ModelExecutionIdentity,
        tokenizer_json: impl Into<PathBuf>,
        chat_template: impl Into<PathBuf>,
    ) -> Self {
        Self {
            public_aliases,
            model,
            tokenizer_json: tokenizer_json.into(),
            tokenizer_revision: "unversioned".to_owned(),
            chat_template: chat_template.into(),
            template_revision: "unversioned".to_owned(),
            template_context: BTreeMap::new(),
            add_generation_prompt: true,
            max_rendered_bytes: DEFAULT_MAX_RENDERED_BYTES,
            reasoning_parser: None,
            tool_parser: None,
        }
    }
}

pub fn load_huggingface_semantics(
    spec: HuggingFaceProfileSpec,
) -> Result<Arc<dyn ModelSemantics>, HuggingFaceSemanticsError> {
    validate_spec(&spec)?;
    let tokenizer_bytes = read(&spec.tokenizer_json, "tokenizer JSON")?;
    let template_bytes = read(&spec.chat_template, "chat template")?;
    let template_source = String::from_utf8(template_bytes.clone()).map_err(|error| {
        HuggingFaceSemanticsError::InvalidTemplate(format!("chat template is not UTF-8: {error}"))
    })?;
    let tokenizer_fingerprint = sha256(&tokenizer_bytes);
    let template_fingerprint = sha256(&template_bytes);
    let tokenizer_identity = SemanticComponentIdentity {
        kind: "huggingface-tokenizer-json".to_owned(),
        revision: spec.tokenizer_revision.clone(),
        fingerprint: tokenizer_fingerprint.clone(),
    };
    let template_identity = SemanticComponentIdentity {
        kind: "huggingface-chat-template".to_owned(),
        revision: spec.template_revision.clone(),
        fingerprint: template_fingerprint.clone(),
    };
    let (reasoning_identity, reasoning_parser) = build_reasoning_parser(&spec)?;
    let (tool_identity, tool_parser) = build_tool_parser(&spec)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|error| {
        HuggingFaceSemanticsError::InvalidTokenizer(format!(
            "failed to load {}: {error}",
            spec.tokenizer_json.display()
        ))
    })?;
    validate_template(&template_source)?;
    let semantic_identity = semantic_identity(
        tokenizer_identity.clone(),
        template_identity.clone(),
        &tokenizer_fingerprint,
        &template_fingerprint,
        reasoning_identity,
        tool_identity,
    );
    let tokenizer = Arc::new(HuggingFaceTokenizer {
        identity: tokenizer_identity,
        tokenizer,
    });
    let renderer = Arc::new(HuggingFaceTemplateRenderer {
        identity: template_identity,
        source: template_source,
        extra_context: spec.template_context,
        add_generation_prompt: spec.add_generation_prompt,
        max_rendered_bytes: spec.max_rendered_bytes,
    });
    let semantics = BasicModelSemantics::new(
        ModelProfile {
            public_aliases: spec.public_aliases,
            model: spec.model,
            semantic_identity,
        },
        tokenizer.clone(),
        renderer,
        tokenizer,
    )
    .with_output_parsers(reasoning_parser, tool_parser)
    .map_err(|error| HuggingFaceSemanticsError::InvalidProfile(error.to_string()))?;
    Ok(Arc::new(semantics))
}

fn build_reasoning_parser(
    spec: &HuggingFaceProfileSpec,
) -> Result<
    (
        Option<SemanticComponentIdentity>,
        Option<TaggedReasoningParserDefinition>,
    ),
    HuggingFaceSemanticsError,
> {
    let Some(parser) = &spec.reasoning_parser else {
        return Ok((None, None));
    };
    let identity = parser_identity(
        "locus-tagged-reasoning-parser",
        &parser.revision,
        &[
            parser.start_delimiter.as_str(),
            parser.end_delimiter.as_str(),
        ],
    )?;
    let definition = TaggedReasoningParserDefinition::new(
        identity.clone(),
        parser.start_delimiter.clone(),
        parser.end_delimiter.clone(),
    )
    .map_err(|error| HuggingFaceSemanticsError::InvalidProfile(error.to_string()))?;
    Ok((Some(identity), Some(definition)))
}

fn build_tool_parser(
    spec: &HuggingFaceProfileSpec,
) -> Result<
    (
        Option<SemanticComponentIdentity>,
        Option<TaggedJsonToolParserDefinition>,
    ),
    HuggingFaceSemanticsError,
> {
    let Some(parser) = &spec.tool_parser else {
        return Ok((None, None));
    };
    let max_buffered_bytes = parser.max_buffered_bytes.to_string();
    let identity = parser_identity(
        "locus-tagged-json-tool-parser",
        &parser.revision,
        &[
            parser.start_delimiter.as_str(),
            parser.end_delimiter.as_str(),
            max_buffered_bytes.as_str(),
        ],
    )?;
    let definition = TaggedJsonToolParserDefinition::new(
        identity.clone(),
        parser.start_delimiter.clone(),
        parser.end_delimiter.clone(),
        parser.max_buffered_bytes,
    )
    .map_err(|error| HuggingFaceSemanticsError::InvalidProfile(error.to_string()))?;
    Ok((Some(identity), Some(definition)))
}

fn parser_identity(
    kind: &str,
    revision: &str,
    fields: &[&str],
) -> Result<SemanticComponentIdentity, HuggingFaceSemanticsError> {
    if revision.trim().is_empty() {
        return Err(HuggingFaceSemanticsError::InvalidProfile(format!(
            "{kind} revision must not be empty"
        )));
    }
    let mut material = String::new();
    for field in std::iter::once(kind)
        .chain(std::iter::once(revision))
        .chain(fields.iter().copied())
    {
        write!(&mut material, "{}:", field.len()).expect("writing to String cannot fail");
        material.push_str(field);
    }
    Ok(SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: revision.to_owned(),
        fingerprint: sha256(material.as_bytes()),
    })
}

fn validate_spec(spec: &HuggingFaceProfileSpec) -> Result<(), HuggingFaceSemanticsError> {
    if spec.public_aliases.is_empty() {
        return Err(HuggingFaceSemanticsError::InvalidProfile(
            "at least one public model alias is required".to_owned(),
        ));
    }
    if spec
        .public_aliases
        .iter()
        .any(|alias| alias.trim().is_empty())
    {
        return Err(HuggingFaceSemanticsError::InvalidProfile(
            "public model aliases must not be empty".to_owned(),
        ));
    }
    if spec.tokenizer_revision.trim().is_empty() || spec.template_revision.trim().is_empty() {
        return Err(HuggingFaceSemanticsError::InvalidProfile(
            "tokenizer and template revisions must not be empty".to_owned(),
        ));
    }
    if spec.max_rendered_bytes == 0 {
        return Err(HuggingFaceSemanticsError::InvalidProfile(
            "max_rendered_bytes must be greater than zero".to_owned(),
        ));
    }
    for reserved in ["messages", "tools", "tool_choice", "add_generation_prompt"] {
        if spec.template_context.contains_key(reserved) {
            return Err(HuggingFaceSemanticsError::InvalidProfile(format!(
                "template_context cannot replace reserved field {reserved}"
            )));
        }
    }
    Ok(())
}

fn read(path: &Path, description: &str) -> Result<Vec<u8>, HuggingFaceSemanticsError> {
    fs::read(path).map_err(|error| HuggingFaceSemanticsError::Read {
        description: description.to_owned(),
        path: path.to_path_buf(),
        source: error,
    })
}

fn semantic_identity(
    tokenizer: SemanticComponentIdentity,
    template: SemanticComponentIdentity,
    tokenizer_fingerprint: &str,
    template_fingerprint: &str,
    reasoning_parser: Option<SemanticComponentIdentity>,
    tool_parser: Option<SemanticComponentIdentity>,
) -> SemanticIdentity {
    let component = |kind: &str, revision: &str| SemanticComponentIdentity {
        kind: kind.to_owned(),
        revision: revision.to_owned(),
        fingerprint: format!("{kind}:{revision}"),
    };
    SemanticIdentity {
        input: InputSemanticIdentity {
            tokenizer: tokenizer.clone(),
            template,
            multimodal_preprocessing: None,
        },
        generation: GenerationSemanticIdentity {
            sampling_normalization: component("locus-sampling-normalization", "v1"),
            stop_behavior: component("locus-stop-behavior", "v1"),
            constrained_generation: Some(component("locus-structured-output", "v1")),
        },
        output: OutputSemanticIdentity {
            detokenizer: tokenizer,
            reasoning_parser: reasoning_parser.clone(),
            tool_parser: tool_parser.clone(),
        },
        umbrella_fingerprint: Some(sha256(
            format!(
                "{tokenizer_fingerprint}\n{template_fingerprint}\n{}\n{}\nlocus-profile-v2",
                reasoning_parser
                    .as_ref()
                    .map_or("none", |identity| identity.fingerprint.as_str()),
                tool_parser
                    .as_ref()
                    .map_or("none", |identity| identity.fingerprint.as_str())
            )
            .as_bytes(),
        )),
    }
}

struct HuggingFaceTokenizer {
    identity: SemanticComponentIdentity,
    tokenizer: Tokenizer,
}

impl TokenizerProvider for HuggingFaceTokenizer {
    fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    fn encode(&self, input: &str) -> Result<TokenSequence, SemanticError> {
        let encoding = self
            .tokenizer
            .encode(input, false)
            .map_err(|error| SemanticError::Processing(format!("tokenization failed: {error}")))?;
        Ok(TokenSequence {
            token_ids: encoding.get_ids().to_vec(),
            tokenizer_fingerprint: self.identity.fingerprint.clone(),
        })
    }
}

impl TokenDecoder for HuggingFaceTokenizer {
    fn decode(&self, token_ids: &[u32]) -> Result<String, SemanticError> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|error| SemanticError::Processing(format!("detokenization failed: {error}")))
    }
}

struct HuggingFaceTemplateRenderer {
    identity: SemanticComponentIdentity,
    source: String,
    extra_context: BTreeMap<String, Value>,
    add_generation_prompt: bool,
    max_rendered_bytes: usize,
}

impl TemplateRenderer for HuggingFaceTemplateRenderer {
    fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    fn render(&self, request: &SemanticRequest) -> Result<String, SemanticError> {
        let SemanticInput::Conversation(conversation) = &request.input else {
            return Err(SemanticError::InvalidInput(
                "chat template requires conversation input".to_owned(),
            ));
        };
        let mut context = self.extra_context.clone();
        context.insert(
            "messages".to_owned(),
            Value::Array(
                conversation
                    .messages
                    .iter()
                    .map(|message| {
                        json!({
                            "role": message.role.as_str(),
                            "content": message.content,
                            "tool_call_id": message.tool_call_id,
                        })
                    })
                    .collect(),
            ),
        );
        let tools = request
            .tools
            .iter()
            .map(|tool| {
                let parameters =
                    serde_json::from_str::<Value>(&tool.parameters_schema).map_err(|error| {
                        SemanticError::InvalidInput(format!(
                            "invalid parameters schema for tool {}: {error}",
                            tool.name
                        ))
                    })?;
                let mut function = serde_json::Map::new();
                function.insert("name".to_owned(), Value::String(tool.name.clone()));
                function.insert("parameters".to_owned(), parameters);
                function.insert("strict".to_owned(), Value::Bool(tool.strict));
                if let Some(description) = &tool.description {
                    function.insert("description".to_owned(), Value::String(description.clone()));
                }
                Ok(json!({"type": "function", "function": function}))
            })
            .collect::<Result<Vec<_>, SemanticError>>()?;
        context.insert("tools".to_owned(), Value::Array(tools));
        context.insert(
            "tool_choice".to_owned(),
            match &request.tool_choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Function(name) => {
                    json!({"type": "function", "function": {"name": name}})
                }
            },
        );
        context.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(self.add_generation_prompt),
        );
        let environment = template_environment();
        let rendered = environment
            .render_str(&self.source, context)
            .map_err(|error| SemanticError::Processing(format!("chat template failed: {error}")))?;
        if rendered.len() > self.max_rendered_bytes {
            return Err(SemanticError::InvalidInput(format!(
                "rendered prompt exceeds configured {} byte limit",
                self.max_rendered_bytes
            )));
        }
        Ok(rendered)
    }
}

fn validate_template(source: &str) -> Result<(), HuggingFaceSemanticsError> {
    template_environment()
        .template_from_str(source)
        .map(|_| ())
        .map_err(|error| HuggingFaceSemanticsError::InvalidTemplate(error.to_string()))
}

fn template_environment<'source>() -> Environment<'source> {
    let mut environment = Environment::new();
    environment.set_undefined_behavior(UndefinedBehavior::Strict);
    environment.add_function(
        "raise_exception",
        |message: String| -> Result<String, TemplateError> {
            Err(TemplateError::new(ErrorKind::InvalidOperation, message))
        },
    );
    environment
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(7 + digest.len() * 2);
    result.push_str("sha256:");
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

#[derive(Debug, Error)]
pub enum HuggingFaceSemanticsError {
    #[error("invalid Hugging Face model profile: {0}")]
    InvalidProfile(String),
    #[error("failed to read {description} at {path}: {source}")]
    Read {
        description: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid tokenizer: {0}")]
    InvalidTokenizer(String),
    #[error("invalid chat template: {0}")]
    InvalidTemplate(String),
}
