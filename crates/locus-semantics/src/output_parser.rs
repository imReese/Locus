use locus_core::SemanticComponentIdentity;
use serde_json::Value;

use crate::SemanticError;

const MAX_DELIMITER_BYTES: usize = 256;

#[derive(Clone, Debug)]
pub struct TaggedReasoningParserDefinition {
    identity: SemanticComponentIdentity,
    start_delimiter: String,
    end_delimiter: String,
}

impl TaggedReasoningParserDefinition {
    pub fn new(
        identity: SemanticComponentIdentity,
        start_delimiter: impl Into<String>,
        end_delimiter: impl Into<String>,
    ) -> Result<Self, SemanticError> {
        let start_delimiter = start_delimiter.into();
        let end_delimiter = end_delimiter.into();
        validate_delimiters("reasoning", &start_delimiter, &end_delimiter)?;
        Ok(Self {
            identity,
            start_delimiter,
            end_delimiter,
        })
    }

    #[must_use]
    pub fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    pub(crate) fn state(&self) -> TaggedReasoningParserState {
        TaggedReasoningParserState {
            definition: self.clone(),
            mode: ReasoningMode::Text,
            pending: String::new(),
            saw_reasoning: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TaggedJsonToolParserDefinition {
    identity: SemanticComponentIdentity,
    start_delimiter: String,
    end_delimiter: String,
    max_buffered_bytes: usize,
}

impl TaggedJsonToolParserDefinition {
    pub fn new(
        identity: SemanticComponentIdentity,
        start_delimiter: impl Into<String>,
        end_delimiter: impl Into<String>,
        max_buffered_bytes: usize,
    ) -> Result<Self, SemanticError> {
        let start_delimiter = start_delimiter.into();
        let end_delimiter = end_delimiter.into();
        validate_delimiters("tool", &start_delimiter, &end_delimiter)?;
        if max_buffered_bytes == 0 {
            return Err(SemanticError::InvalidInput(
                "tool parser max_buffered_bytes must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            identity,
            start_delimiter,
            end_delimiter,
            max_buffered_bytes,
        })
    }

    #[must_use]
    pub fn identity(&self) -> &SemanticComponentIdentity {
        &self.identity
    }

    pub(crate) fn state(&self) -> TaggedJsonToolParserState {
        TaggedJsonToolParserState {
            definition: self.clone(),
            mode: ToolMode::Text,
            pending: String::new(),
        }
    }
}

fn validate_delimiters(kind: &str, start: &str, end: &str) -> Result<(), SemanticError> {
    if start.is_empty() || end.is_empty() {
        return Err(SemanticError::InvalidInput(format!(
            "{kind} parser delimiters must not be empty"
        )));
    }
    if start == end {
        return Err(SemanticError::InvalidInput(format!(
            "{kind} parser delimiters must be distinct"
        )));
    }
    if start.len() > MAX_DELIMITER_BYTES || end.len() > MAX_DELIMITER_BYTES {
        return Err(SemanticError::InvalidInput(format!(
            "{kind} parser delimiters must not exceed {MAX_DELIMITER_BYTES} bytes"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReasoningMode {
    Text,
    Reasoning,
}

pub(crate) enum ReasoningSegment {
    Text(String),
    Reasoning(String),
}

pub(crate) struct TaggedReasoningParserState {
    definition: TaggedReasoningParserDefinition,
    mode: ReasoningMode,
    pending: String,
    saw_reasoning: bool,
}

impl TaggedReasoningParserState {
    pub(crate) fn push(&mut self, delta: &str) -> Result<Vec<ReasoningSegment>, SemanticError> {
        self.pending.push_str(delta);
        let mut output = Vec::new();
        loop {
            let (delimiter, forbidden) = match self.mode {
                ReasoningMode::Text => (
                    self.definition.start_delimiter.as_str(),
                    self.definition.end_delimiter.as_str(),
                ),
                ReasoningMode::Reasoning => (
                    self.definition.end_delimiter.as_str(),
                    self.definition.start_delimiter.as_str(),
                ),
            };
            let delimiter_position = self.pending.find(delimiter);
            let forbidden_position = self.pending.find(forbidden);
            if forbidden_position.is_some_and(|position| {
                delimiter_position.is_none_or(|delimiter_position| position < delimiter_position)
            }) {
                return Err(SemanticError::Processing(format!(
                    "reasoning parser observed unexpected delimiter {forbidden:?}"
                )));
            }
            if let Some(position) = delimiter_position {
                let prefix = self.pending[..position].to_owned();
                push_reasoning_segment(&mut output, self.mode, prefix);
                self.pending.drain(..position + delimiter.len());
                match self.mode {
                    ReasoningMode::Text => {
                        if self.saw_reasoning {
                            return Err(SemanticError::Processing(
                                "reasoning parser observed multiple reasoning sections".to_owned(),
                            ));
                        }
                        self.saw_reasoning = true;
                        self.mode = ReasoningMode::Reasoning;
                    }
                    ReasoningMode::Reasoning => self.mode = ReasoningMode::Text,
                }
                continue;
            }

            let holdback = longest_delimiter_prefix_suffix(&self.pending, &[delimiter, forbidden]);
            let emit_bytes = self.pending.len().saturating_sub(holdback);
            if emit_bytes > 0 {
                let text = self.pending[..emit_bytes].to_owned();
                self.pending.drain(..emit_bytes);
                push_reasoning_segment(&mut output, self.mode, text);
            }
            break;
        }
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<ReasoningSegment>, SemanticError> {
        if self.mode == ReasoningMode::Reasoning {
            return Err(SemanticError::Processing(
                "engine finished with an unterminated reasoning section".to_owned(),
            ));
        }
        if !self.pending.is_empty() {
            return Err(SemanticError::Processing(
                "engine finished with an incomplete reasoning delimiter".to_owned(),
            ));
        }
        Ok(Vec::new())
    }
}

fn push_reasoning_segment(output: &mut Vec<ReasoningSegment>, mode: ReasoningMode, text: String) {
    if text.is_empty() {
        return;
    }
    output.push(match mode {
        ReasoningMode::Text => ReasoningSegment::Text(text),
        ReasoningMode::Reasoning => ReasoningSegment::Reasoning(text),
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolMode {
    Text,
    Tool,
}

pub(crate) enum ToolSegment {
    Text(String),
    Call { name: String, arguments: String },
}

pub(crate) struct TaggedJsonToolParserState {
    definition: TaggedJsonToolParserDefinition,
    mode: ToolMode,
    pending: String,
}

impl TaggedJsonToolParserState {
    pub(crate) fn push(&mut self, delta: &str) -> Result<Vec<ToolSegment>, SemanticError> {
        self.pending.push_str(delta);
        let mut output = Vec::new();
        loop {
            match self.mode {
                ToolMode::Text => {
                    let start = self.definition.start_delimiter.as_str();
                    let end = self.definition.end_delimiter.as_str();
                    let start_position = self.pending.find(start);
                    let end_position = self.pending.find(end);
                    if end_position.is_some_and(|position| {
                        start_position.is_none_or(|start_position| position < start_position)
                    }) {
                        return Err(SemanticError::Processing(format!(
                            "tool parser observed unexpected delimiter {end:?}"
                        )));
                    }
                    if let Some(position) = start_position {
                        let prefix = self.pending[..position].to_owned();
                        if !prefix.is_empty() {
                            output.push(ToolSegment::Text(prefix));
                        }
                        self.pending.drain(..position + start.len());
                        self.mode = ToolMode::Tool;
                        continue;
                    }
                    let holdback = longest_delimiter_prefix_suffix(&self.pending, &[start, end]);
                    let emit_bytes = self.pending.len().saturating_sub(holdback);
                    if emit_bytes > 0 {
                        output.push(ToolSegment::Text(self.pending[..emit_bytes].to_owned()));
                        self.pending.drain(..emit_bytes);
                    }
                    break;
                }
                ToolMode::Tool => {
                    let Some((end_position, name, arguments)) = self.complete_call() else {
                        if self.pending.len() > self.definition.max_buffered_bytes {
                            return Err(SemanticError::Processing(format!(
                                "tool call exceeds configured {} byte parser limit",
                                self.definition.max_buffered_bytes
                            )));
                        }
                        break;
                    };
                    if end_position > self.definition.max_buffered_bytes {
                        return Err(SemanticError::Processing(format!(
                            "tool call exceeds configured {} byte parser limit",
                            self.definition.max_buffered_bytes
                        )));
                    }
                    let consumed = end_position + self.definition.end_delimiter.len();
                    self.pending.drain(..consumed);
                    self.mode = ToolMode::Text;
                    output.push(ToolSegment::Call { name, arguments });
                }
            }
        }
        Ok(output)
    }

    fn complete_call(&self) -> Option<(usize, String, String)> {
        let mut search_from = 0;
        while let Some(relative) = self.pending[search_from..].find(&self.definition.end_delimiter)
        {
            let position = search_from + relative;
            if let Ok((name, arguments)) = parse_tool_envelope(&self.pending[..position]) {
                return Some((position, name, arguments));
            }
            search_from = position + self.definition.end_delimiter.len();
        }
        None
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<ToolSegment>, SemanticError> {
        if self.mode == ToolMode::Tool {
            return Err(SemanticError::Processing(
                "engine finished with an unterminated or malformed tool call".to_owned(),
            ));
        }
        if !self.pending.is_empty() {
            return Err(SemanticError::Processing(
                "engine finished with an incomplete tool delimiter".to_owned(),
            ));
        }
        Ok(Vec::new())
    }
}

fn parse_tool_envelope(input: &str) -> Result<(String, String), SemanticError> {
    let value = serde_json::from_str::<Value>(input)
        .map_err(|error| SemanticError::Processing(format!("invalid tool call JSON: {error}")))?;
    let Value::Object(mut object) = value else {
        return Err(SemanticError::Processing(
            "tool call envelope must be a JSON object".to_owned(),
        ));
    };
    if object.len() != 2 || !object.contains_key("name") || !object.contains_key("arguments") {
        return Err(SemanticError::Processing(
            "tool call envelope must contain exactly name and arguments".to_owned(),
        ));
    }
    let name = object
        .remove("name")
        .and_then(|value| value.as_str().map(str::to_owned))
        .filter(|name| !name.is_empty())
        .ok_or_else(|| SemanticError::Processing("tool call name must be a string".to_owned()))?;
    let arguments = object.remove("arguments").expect("key checked above");
    if !arguments.is_object() {
        return Err(SemanticError::Processing(
            "tool call arguments must be a JSON object".to_owned(),
        ));
    }
    let arguments = serde_json::to_string(&arguments).map_err(|error| {
        SemanticError::Processing(format!("failed to normalize tool call arguments: {error}"))
    })?;
    Ok((name, arguments))
}

fn longest_delimiter_prefix_suffix(value: &str, delimiters: &[&str]) -> usize {
    value
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(value.len()))
        .filter_map(|index| {
            let suffix = &value[index..];
            (!suffix.is_empty()
                && delimiters
                    .iter()
                    .any(|delimiter| delimiter.starts_with(suffix)))
            .then_some(suffix.len())
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(kind: &str) -> SemanticComponentIdentity {
        SemanticComponentIdentity {
            kind: kind.to_owned(),
            revision: "test-v1".to_owned(),
            fingerprint: format!("{kind}-test-v1"),
        }
    }

    #[test]
    fn reasoning_markers_are_safe_across_every_byte_boundary() {
        let input = "<think>check 北京</think>answer";
        for boundary in 0..=input.len() {
            if !input.is_char_boundary(boundary) {
                continue;
            }
            let definition =
                TaggedReasoningParserDefinition::new(identity("reasoning"), "<think>", "</think>")
                    .expect("definition");
            let mut parser = definition.state();
            let mut events = parser.push(&input[..boundary]).expect("first chunk");
            events.extend(parser.push(&input[boundary..]).expect("second chunk"));
            events.extend(parser.finish().expect("finish"));
            let reasoning = events
                .iter()
                .filter_map(|event| match event {
                    ReasoningSegment::Reasoning(text) => Some(text.as_str()),
                    ReasoningSegment::Text(_) => None,
                })
                .collect::<String>();
            let text = events
                .iter()
                .filter_map(|event| match event {
                    ReasoningSegment::Text(text) => Some(text.as_str()),
                    ReasoningSegment::Reasoning(_) => None,
                })
                .collect::<String>();
            assert_eq!(reasoning, "check 北京", "boundary {boundary}");
            assert_eq!(text, "answer", "boundary {boundary}");
        }
    }

    #[test]
    fn tagged_json_calls_commit_only_after_a_valid_complete_envelope() {
        let input = "before<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"北京\"}}</tool_call>after";
        for boundary in 0..=input.len() {
            if !input.is_char_boundary(boundary) {
                continue;
            }
            let definition = TaggedJsonToolParserDefinition::new(
                identity("tool"),
                "<tool_call>",
                "</tool_call>",
                1024,
            )
            .expect("definition");
            let mut parser = definition.state();
            let mut events = parser.push(&input[..boundary]).expect("first chunk");
            events.extend(parser.push(&input[boundary..]).expect("second chunk"));
            events.extend(parser.finish().expect("finish"));
            let reconstructed = events
                .iter()
                .map(|event| match event {
                    ToolSegment::Text(text) => text.clone(),
                    ToolSegment::Call { name, arguments } => {
                        format!("<call:{name}:{arguments}>")
                    }
                })
                .collect::<String>();
            assert_eq!(
                reconstructed, "before<call:weather:{\"city\":\"北京\"}>after",
                "boundary {boundary}"
            );
        }
    }

    #[test]
    fn malformed_or_unterminated_reserved_syntax_fails_closed() {
        let reasoning =
            TaggedReasoningParserDefinition::new(identity("reasoning"), "<think>", "</think>")
                .expect("definition");
        let mut reasoning = reasoning.state();
        reasoning.push("<think>private").expect("partial input");
        assert!(reasoning.finish().is_err());

        let reasoning =
            TaggedReasoningParserDefinition::new(identity("reasoning"), "<think>", "</think>")
                .expect("definition");
        let mut reasoning = reasoning.state();
        assert!(
            reasoning
                .push("<think>first</think><think>duplicate</think>")
                .is_err()
        );

        let tool = TaggedJsonToolParserDefinition::new(
            identity("tool"),
            "<tool_call>",
            "</tool_call>",
            1024,
        )
        .expect("definition");
        let mut tool = tool.state();
        tool.push("<tool_call>{\"name\":\"weather\"}</tool_call>")
            .expect("parser buffers malformed envelope");
        assert!(tool.finish().is_err());
    }

    #[test]
    fn tool_buffer_limit_applies_to_each_envelope_not_trailing_answer_text() {
        let definition = TaggedJsonToolParserDefinition::new(
            identity("tool"),
            "<tool_call>",
            "</tool_call>",
            64,
        )
        .expect("definition");
        let mut parser = definition.state();
        let input = format!(
            "<tool_call>{{\"name\":\"x\",\"arguments\":{{}}}}</tool_call>{}",
            "answer".repeat(100)
        );
        parser.push(&input).expect("bounded envelope");
        parser.finish().expect("finish");

        let definition =
            TaggedJsonToolParserDefinition::new(identity("tool"), "<tool_call>", "</tool_call>", 8)
                .expect("definition");
        let mut parser = definition.state();
        assert!(parser.push(&input).is_err());
    }
}
