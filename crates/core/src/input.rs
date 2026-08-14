use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InputItemId(String);

impl InputItemId {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputKind {
    TokenSequence,
    MediaReference,
    TensorReference,
    PreparedInputReference,
    TypedMetadata,
    Extension(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenSequence {
    pub token_ids: Vec<u32>,
    pub tokenizer_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaReference {
    pub media_type: String,
    pub digest: String,
    pub access_reference: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorReference {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub layout: String,
    pub access_reference: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedInputReference {
    pub artifact_kind: String,
    pub artifact_id: String,
    pub compatibility_revision: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedMetadata {
    pub type_url: String,
    pub fields: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputItemValue {
    TokenSequence(TokenSequence),
    MediaReference(MediaReference),
    TensorReference(TensorReference),
    PreparedInputReference(PreparedInputReference),
    TypedMetadata(TypedMetadata),
    Extension { type_url: String, payload: Vec<u8> },
}

impl InputItemValue {
    #[must_use]
    pub fn kind(&self) -> InputKind {
        match self {
            Self::TokenSequence(_) => InputKind::TokenSequence,
            Self::MediaReference(_) => InputKind::MediaReference,
            Self::TensorReference(_) => InputKind::TensorReference,
            Self::PreparedInputReference(_) => InputKind::PreparedInputReference,
            Self::TypedMetadata(_) => InputKind::TypedMetadata,
            Self::Extension { type_url, .. } => InputKind::Extension(type_url.clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputItem {
    pub id: InputItemId,
    pub value: InputItemValue,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputRelation {
    pub source: InputItemId,
    pub target: InputItemId,
    pub relation_type: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputBundle {
    pub items: Vec<InputItem>,
    pub relations: Vec<InputRelation>,
    pub annotations: Vec<TypedMetadata>,
}

impl InputBundle {
    #[must_use]
    pub fn token_sequence(id: impl Into<String>, token_ids: Vec<u32>) -> Self {
        Self {
            items: vec![InputItem {
                id: InputItemId::new(id),
                value: InputItemValue::TokenSequence(TokenSequence {
                    token_ids,
                    tokenizer_fingerprint: "test-tokenizer".to_owned(),
                }),
            }],
            ..Self::default()
        }
    }

    pub fn kinds(&self) -> impl Iterator<Item = InputKind> + '_ {
        self.items.iter().map(|item| item.value.kind())
    }
}
