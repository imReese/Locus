use std::collections::BTreeMap;

use ahash::AHashMap;
use locus_core::{
    EngineEvent, EngineFinishReason, InputItemValue, ModelExecutionIdentity, RequestId, Usage,
};
use locus_model_io::hf::{HuggingFaceProfileSpec, load_huggingface_model_io};
use locus_model_io::{
    Conversation, ConversationMessage, ConversationRole, ModelEvent, ModelFinishReason, ModelInput,
    ModelRequest, PromptInput, ReasoningEffort, ToolChoice, ToolDefinition,
};
use locus_parser::{TaggedJsonToolParserSpec, TaggedReasoningParserSpec};
use serde_json::json;
use tempfile::tempdir;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

fn write_fixture() -> (tempfile::TempDir, HuggingFaceProfileSpec) {
    let directory = tempdir().expect("temp directory");
    let tokenizer_path = directory.path().join("tokenizer.json");
    let template_path = directory.path().join("chat_template.jinja");
    let vocabulary = AHashMap::from_iter([
        ("<unk>".to_owned(), 0),
        ("<bos>".to_owned(), 1),
        ("user".to_owned(), 2),
        ("hello".to_owned(), 3),
        ("assistant".to_owned(), 4),
        ("tool".to_owned(), 5),
        ("weather".to_owned(), 6),
    ]);
    let model = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("<unk>".to_owned())
        .build()
        .expect("word-level model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(&tokenizer_path, false)
        .expect("save tokenizer");
    std::fs::write(
        &template_path,
        "{{ bos_token }} {% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}{% if tools %}tool {{ tools[0].function.name }} {% endif %}{% if add_generation_prompt %}assistant{% endif %}",
    )
    .expect("write template");
    let mut spec = HuggingFaceProfileSpec::new(
        vec!["fixture".to_owned()],
        ModelExecutionIdentity {
            model_revision: "fixture-model-v1".to_owned(),
            adapter_revision: None,
            execution_profile: "default".to_owned(),
        },
        tokenizer_path,
        template_path,
    );
    spec.tokenizer_revision = "fixture-tokenizer-v1".to_owned();
    spec.template_revision = "fixture-template-v1".to_owned();
    spec.template_context = BTreeMap::from([("bos_token".to_owned(), json!("<bos>"))]);
    (directory, spec)
}

fn tool() -> ToolDefinition {
    ToolDefinition {
        name: "weather".to_owned(),
        description: Some("Get the weather".to_owned()),
        parameters_schema: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })
        .to_string(),
        strict: true,
    }
}

#[test]
fn production_profile_renders_and_tokenizes_once_with_content_identities() {
    let (_directory, spec) = write_fixture();
    let model_io = load_huggingface_model_io(spec).expect("load profile");
    let normalized = model_io
        .normalize(
            &ModelRequest {
                model: "fixture".to_owned(),
                input: ModelInput::Conversation(Conversation {
                    messages: vec![ConversationMessage {
                        role: ConversationRole::User,
                        content: "hello".to_owned(),
                        tool_call_id: None,
                    }],
                }),
                ..ModelRequest::default()
            },
            RequestId::new("request-1"),
        )
        .expect("normalize");
    let tokens = normalized
        .canonical
        .input
        .items
        .iter()
        .find_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens),
            _ => None,
        })
        .expect("token sequence");
    assert_eq!(tokens.token_ids, vec![1, 2, 3, 4]);
    assert!(tokens.tokenizer_fingerprint.starts_with("sha256:"));
    assert_eq!(
        tokens.tokenizer_fingerprint,
        normalized
            .canonical
            .semantic_identity
            .input
            .tokenizer
            .fingerprint
    );
    assert!(
        normalized
            .canonical
            .semantic_identity
            .input
            .template
            .fingerprint
            .starts_with("sha256:")
    );
}

#[test]
fn raw_text_and_token_prompts_bypass_the_chat_template_with_distinct_identity() {
    let (_directory, spec) = write_fixture();
    let model_io = load_huggingface_model_io(spec).expect("load profile");
    let text = model_io
        .normalize(
            &ModelRequest {
                model: "fixture".to_owned(),
                input: ModelInput::Prompt(PromptInput::Text("hello".to_owned())),
                sampling: locus_core::SamplingParameters {
                    stop_sequences: vec!["END".to_owned()],
                    ..locus_core::SamplingParameters::default()
                },
                ..ModelRequest::default()
            },
            RequestId::new("request-raw-text"),
        )
        .expect("normalize raw text");
    let text_tokens = text
        .canonical
        .input
        .items
        .iter()
        .find_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens),
            _ => None,
        })
        .expect("text token sequence");
    assert_eq!(text_tokens.token_ids, vec![3]);
    assert_eq!(
        text.canonical.semantic_identity.input.template.kind,
        "locus-raw-prompt"
    );
    assert!(
        text.canonical
            .semantic_identity
            .umbrella_fingerprint
            .is_none()
    );
    assert_eq!(
        text.canonical.sampling.stop_sequences,
        vec!["END".to_owned()]
    );

    let tokens = model_io
        .normalize(
            &ModelRequest {
                model: "fixture".to_owned(),
                input: ModelInput::Prompt(PromptInput::TokenIds(vec![91, 92])),
                ..ModelRequest::default()
            },
            RequestId::new("request-raw-tokens"),
        )
        .expect("normalize raw tokens");
    let token_ids = tokens
        .canonical
        .input
        .items
        .iter()
        .find_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens.token_ids.as_slice()),
            _ => None,
        })
        .expect("token sequence");
    assert_eq!(token_ids, &[91, 92]);
}

#[test]
fn profile_bound_parsers_render_tools_and_emit_typed_events_from_fragmented_text() {
    let (_directory, mut spec) = write_fixture();
    spec.reasoning_parser = Some(TaggedReasoningParserSpec {
        revision: "think-tags-v1".to_owned(),
        start_delimiter: "<think>".to_owned(),
        end_delimiter: "</think>".to_owned(),
    });
    spec.tool_parser = Some(TaggedJsonToolParserSpec {
        revision: "tool-envelope-v1".to_owned(),
        start_delimiter: "<tool_call>".to_owned(),
        end_delimiter: "</tool_call>".to_owned(),
        max_buffered_bytes: 4096,
    });
    let model_io = load_huggingface_model_io(spec).expect("load parser profile");
    let normalized = model_io
        .normalize(
            &ModelRequest {
                model: "fixture".to_owned(),
                input: ModelInput::Conversation(Conversation {
                    messages: vec![ConversationMessage {
                        role: ConversationRole::User,
                        content: "hello".to_owned(),
                        tool_call_id: None,
                    }],
                }),
                tools: vec![tool()],
                tool_choice: ToolChoice::Required,
                reasoning_effort: Some(ReasoningEffort::High),
                ..ModelRequest::default()
            },
            RequestId::new("request-parser"),
        )
        .expect("normalize parser request");
    let tokens = normalized
        .canonical
        .input
        .items
        .iter()
        .find_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens.token_ids.as_slice()),
            _ => None,
        })
        .expect("token sequence");
    assert_eq!(tokens, &[1, 2, 3, 5, 6, 4]);
    assert!(
        normalized
            .canonical
            .semantic_identity
            .output
            .reasoning_parser
            .is_some()
    );
    assert!(
        normalized
            .canonical
            .semantic_identity
            .output
            .tool_parser
            .is_some()
    );
    assert!(!normalized.canonical.requirements.requires_reasoning_deltas);
    assert!(!normalized.canonical.requirements.requires_tool_calls);

    let mut pipeline = model_io
        .output_pipeline(&normalized.output_contract)
        .expect("output pipeline");
    let request_id = RequestId::new("request-parser");
    let mut events = Vec::new();
    for (sequence_number, text) in [
        "<thi",
        "nk>checked",
        " constraints</think><tool_",
        "call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Beijing\"}}",
        "</tool_call>",
    ]
    .into_iter()
    .enumerate()
    {
        events.extend(
            pipeline
                .process(EngineEvent::TextDelta {
                    request_id: request_id.clone(),
                    sequence_number: sequence_number as u64,
                    text: text.to_owned(),
                })
                .expect("parse text delta"),
        );
    }
    events.extend(
        pipeline
            .process(EngineEvent::Finished {
                request_id,
                reason: EngineFinishReason::Stop,
                usage: Usage::default(),
            })
            .expect("finish parsed output"),
    );

    let reasoning = events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::ReasoningDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(reasoning, "checked constraints");
    assert!(!events.iter().any(
        |event| matches!(event, ModelEvent::TextDelta { text, .. } if text.contains("think") || text.contains("tool_call"))
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::ToolCallStarted { call_id, name, .. }
            if call_id == "call_request-parser_0" && name == "weather"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::ToolCallCompleted { arguments, .. }
            if arguments == "{\"city\":\"Beijing\"}"
    )));
    assert!(matches!(
        events.last(),
        Some(ModelEvent::Finished {
            reason: ModelFinishReason::ToolCall,
            ..
        })
    ));
}

#[test]
fn parser_configuration_changes_output_and_umbrella_fingerprints() {
    let (_directory, mut first) = write_fixture();
    first.reasoning_parser = Some(TaggedReasoningParserSpec {
        revision: "think-tags-v1".to_owned(),
        start_delimiter: "<think>".to_owned(),
        end_delimiter: "</think>".to_owned(),
    });
    let mut second = first.clone();
    second
        .reasoning_parser
        .as_mut()
        .expect("reasoning parser")
        .end_delimiter = "</reasoning>".to_owned();
    let first = load_huggingface_model_io(first).expect("first profile");
    let second = load_huggingface_model_io(second).expect("second profile");
    assert_ne!(
        first
            .profile()
            .semantic_identity
            .output
            .reasoning_parser
            .as_ref()
            .expect("first parser identity")
            .fingerprint,
        second
            .profile()
            .semantic_identity
            .output
            .reasoning_parser
            .as_ref()
            .expect("second parser identity")
            .fingerprint
    );
    assert_ne!(
        first.profile().semantic_identity.umbrella_fingerprint,
        second.profile().semantic_identity.umbrella_fingerprint
    );
}

#[test]
fn malformed_template_is_rejected_at_profile_load() {
    let (_directory, spec) = write_fixture();
    std::fs::write(&spec.chat_template, "{% if messages %}").expect("write bad template");
    let error = match load_huggingface_model_io(spec) {
        Ok(_) => panic!("malformed template must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("invalid chat template"));
}

#[test]
fn reserved_template_context_is_rejected() {
    let (_directory, mut spec) = write_fixture();
    spec.template_context
        .insert("messages".to_owned(), json!([]));
    let error = match load_huggingface_model_io(spec) {
        Ok(_) => panic!("reserved context must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("reserved field messages"));
}
