use std::collections::BTreeMap;

use ahash::AHashMap;
use locus_core::{InputItemValue, ModelExecutionIdentity, RequestId};
use locus_semantics::{Conversation, ConversationMessage, ConversationRole, SemanticRequest};
use locus_semantics_hf::{HuggingFaceProfileSpec, load_huggingface_semantics};
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
        "{{ bos_token }} {% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}",
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

#[test]
fn production_profile_renders_and_tokenizes_once_with_content_identities() {
    let (_directory, spec) = write_fixture();
    let semantics = load_huggingface_semantics(spec).expect("load profile");
    let normalized = semantics
        .normalize(
            &SemanticRequest {
                model: "fixture".to_owned(),
                conversation: Conversation {
                    messages: vec![ConversationMessage {
                        role: ConversationRole::User,
                        content: "hello".to_owned(),
                        tool_call_id: None,
                    }],
                },
                ..SemanticRequest::default()
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
fn malformed_template_is_rejected_at_profile_load() {
    let (_directory, spec) = write_fixture();
    std::fs::write(&spec.chat_template, "{% if messages %}").expect("write bad template");
    let error = match load_huggingface_semantics(spec) {
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
    let error = match load_huggingface_semantics(spec) {
        Ok(_) => panic!("reserved context must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("reserved field messages"));
}
