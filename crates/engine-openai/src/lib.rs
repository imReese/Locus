use std::collections::BTreeSet;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use locus_core::{
    CanonicalRequest, EngineCapabilities, EngineEvent, EngineFinishReason, EngineInstance,
    EngineSnapshot, ExecutionTarget, InputItemValue, InputKind, OperationContext,
    PreparedStateAttachment, RequestId, StateImportSpec, StateImportTarget, TransferReceipt, Usage,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use locus_engine::{EngineAdapter, EngineError, EngineEventStream};

#[derive(Clone, Debug)]
pub struct RemoteEngineConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub served_model: String,
    pub instance: EngineInstance,
    pub target: ExecutionTarget,
}

#[derive(Clone, Copy)]
enum RuntimeFlavor {
    Sglang,
    Vllm,
}

struct RemoteCompletionAdapter {
    config: RemoteEngineConfig,
    flavor: RuntimeFlavor,
    client: Client,
}

impl RemoteCompletionAdapter {
    fn new(config: RemoteEngineConfig, flavor: RuntimeFlavor) -> Result<Self, EngineError> {
        if config.base_url.trim().is_empty() {
            return Err(EngineError::Execution(
                "remote engine base URL must not be empty".to_owned(),
            ));
        }
        if config.target.engine != config.instance.reference {
            return Err(EngineError::Execution(
                "remote target generation does not match engine instance".to_owned(),
            ));
        }
        Ok(Self {
            config,
            flavor,
            client: Client::new(),
        })
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.config.base_url.trim_end_matches('/'), path)
    }

    fn validate_target(&self, target: &ExecutionTarget) -> Result<(), EngineError> {
        if target != &self.config.target {
            return Err(EngineError::TargetNotFound(target.id.to_string()));
        }
        Ok(())
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.config.api_key {
            request.bearer_auth(api_key)
        } else {
            request
        }
    }

    fn completion_body(&self, request: &CanonicalRequest) -> Result<Value, EngineError> {
        let mut token_sequences = request.input.items.iter().filter_map(|item| {
            if let InputItemValue::TokenSequence(tokens) = &item.value {
                Some(tokens.token_ids.clone())
            } else {
                None
            }
        });
        let prompt = token_sequences.next().ok_or_else(|| {
            EngineError::Unsupported("remote completion requires a token sequence".to_owned())
        })?;
        if token_sequences.next().is_some() {
            return Err(EngineError::Unsupported(
                "remote completion supports exactly one token sequence".to_owned(),
            ));
        }
        let has_tools = request
            .input
            .annotations
            .iter()
            .any(|annotation| annotation.type_url == "locus.tool.v1");
        if has_tools && request.semantic_identity.output.tool_parser.is_none() {
            return Err(EngineError::Unsupported(
                "function tools require a model-profile tool parser".to_owned(),
            ));
        }
        if request.requirements.requires_tool_calls {
            return Err(EngineError::Unsupported(
                "remote completion adapter does not emit native tool-call events".to_owned(),
            ));
        }
        if request.requirements.requires_reasoning_deltas {
            return Err(EngineError::Unsupported(
                "remote completion adapter does not emit native reasoning events".to_owned(),
            ));
        }

        let mut body = json!({
            "model": self.config.served_model,
            "prompt": prompt,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": request.sampling.max_output_tokens.unwrap_or(16),
            "temperature": request.sampling.temperature.unwrap_or(1.0),
            "top_p": request.sampling.top_p.unwrap_or(1.0)
        });
        if let Some(seed) = request.sampling.seed {
            body["seed"] = json!(seed);
        }
        match self.flavor {
            RuntimeFlavor::Sglang => body["rid"] = json!(request.id.as_str()),
            RuntimeFlavor::Vllm => {
                body["request_id"] = json!(request.id.as_str());
                body["add_special_tokens"] = json!(false);
            }
        }
        apply_output_contract(&mut body, request, self.flavor)?;
        Ok(body)
    }

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        let health_endpoint = self
            .config
            .instance
            .health_endpoint
            .clone()
            .unwrap_or_else(|| self.endpoint("/health"));
        let response = self
            .authorized(self.client.get(health_endpoint))
            .send()
            .await
            .map_err(|error| EngineError::Execution(format!("health request failed: {error}")))?;
        Ok(EngineSnapshot {
            target_id: target.id.clone(),
            ready: response.status().is_success(),
            queue_depth: 0,
            estimated_queue_micros: None,
            observation_revision: 1,
        })
    }

    async fn execute(
        &self,
        target: &ExecutionTarget,
        request: CanonicalRequest,
        state: Option<PreparedStateAttachment>,
        context: OperationContext,
    ) -> Result<EngineEventStream, EngineError> {
        context.ensure_active()?;
        self.validate_target(target)?;
        if state.is_some() {
            return Err(EngineError::Unsupported(
                "remote completion adapter does not import reusable state".to_owned(),
            ));
        }
        let body = self.completion_body(&request)?;
        let response = self
            .authorized(self.client.post(self.endpoint("/v1/completions")))
            .header("x-request-id", request.id.as_str())
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                EngineError::Execution(format!("completion request failed: {error}"))
            })?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "response body was unavailable".to_owned());
            return Err(EngineError::Execution(format!(
                "completion endpoint returned {status}: {detail}"
            )));
        }

        let request_id = request.id;
        let stream = async_stream::try_stream! {
            yield EngineEvent::Accepted { request_id: request_id.clone() };
            let mut events = response.bytes_stream().eventsource();
            let mut sequence_number = 1_u64;
            let mut usage = Usage::default();
            let mut finish_reason = None;
            while let Some(event) = events.next().await {
                let event = event.map_err(|error| {
                    EngineError::Execution(format!("invalid completion event stream: {error}"))
                })?;
                if event.data == "[DONE]" {
                    break;
                }
                let chunk = serde_json::from_str::<CompletionChunk>(&event.data).map_err(|error| {
                    EngineError::Execution(format!("invalid completion chunk: {error}"))
                })?;
                if let Some(error) = chunk.error {
                    Err(EngineError::Execution(format!("remote engine error: {}", error.message)))?;
                }
                if let Some(remote_usage) = chunk.usage {
                    usage = Usage {
                        input_tokens: remote_usage.prompt_tokens,
                        output_tokens: remote_usage.completion_tokens,
                    };
                    yield EngineEvent::UsageUpdate {
                        request_id: request_id.clone(),
                        usage: usage.clone(),
                        final_update: true,
                    };
                }
                for choice in chunk.choices {
                    if choice.index != 0 {
                        continue;
                    }
                    if !choice.text.is_empty() {
                        yield EngineEvent::TextDelta {
                            request_id: request_id.clone(),
                            sequence_number,
                            text: choice.text,
                        };
                        sequence_number += 1;
                    }
                    if let Some(reason) = choice.finish_reason {
                        finish_reason = Some(map_finish_reason(&reason));
                    }
                }
            }
            yield EngineEvent::Finished {
                request_id,
                reason: finish_reason.unwrap_or(EngineFinishReason::Stop),
                usage,
            };
        };
        Ok(Box::pin(stream))
    }

    async fn cancel(
        &self,
        request_id: &RequestId,
        context: &OperationContext,
    ) -> Result<(), EngineError> {
        context.ensure_active()?;
        if matches!(self.flavor, RuntimeFlavor::Sglang) {
            let response = self
                .authorized(self.client.post(self.endpoint("/abort_request")))
                .json(&json!({"rid": request_id.as_str(), "abort_all": false}))
                .send()
                .await
                .map_err(|error| {
                    EngineError::Execution(format!("abort request failed: {error}"))
                })?;
            if !response.status().is_success() {
                return Err(EngineError::Execution(format!(
                    "abort endpoint returned {}",
                    response.status()
                )));
            }
        }
        // Dropping reqwest's response stream closes the in-flight HTTP request. vLLM's
        // completion endpoint has no separate public cancellation endpoint to call here.
        Ok(())
    }
}

fn apply_output_contract(
    body: &mut Value,
    request: &CanonicalRequest,
    flavor: RuntimeFlavor,
) -> Result<(), EngineError> {
    let Some(contract) = request
        .input
        .annotations
        .iter()
        .find(|annotation| annotation.type_url == "locus.generation-contract.v1")
    else {
        return Ok(());
    };
    match contract.fields.get("response_format").map(String::as_str) {
        Some("json_object") => body["response_format"] = json!({"type": "json_object"}),
        Some("json_schema") => {
            let schema_text = contract.fields.get("json_schema").ok_or_else(|| {
                EngineError::Execution("JSON schema contract omitted schema".to_owned())
            })?;
            let schema = serde_json::from_str::<Value>(schema_text).map_err(|error| {
                EngineError::Execution(format!("invalid JSON schema contract: {error}"))
            })?;
            if matches!(flavor, RuntimeFlavor::Sglang) {
                body["json_schema"] = json!(schema_text);
            } else {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": contract.fields.get("json_schema_name").map_or("response", String::as_str),
                        "schema": schema,
                        "strict": contract.fields.get("strict").is_some_and(|value| value == "true")
                    }
                });
            }
        }
        Some("text") | None => {}
        Some(other) => {
            return Err(EngineError::Unsupported(format!(
                "unknown response format: {other}"
            )));
        }
    }
    Ok(())
}

fn map_finish_reason(reason: &str) -> EngineFinishReason {
    match reason {
        "stop" => EngineFinishReason::Stop,
        "length" => EngineFinishReason::Length,
        "abort" | "cancelled" => EngineFinishReason::Cancelled,
        value => EngineFinishReason::RuntimeSpecific {
            namespace: "openai.completion.finish_reason".to_owned(),
            value: value.to_owned(),
        },
    }
}

#[derive(Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
    #[serde(default)]
    error: Option<RemoteError>,
}

#[derive(Deserialize)]
struct CompletionChoice {
    index: usize,
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct RemoteError {
    message: String,
}

pub struct SglangEngineAdapter {
    inner: RemoteCompletionAdapter,
}

impl SglangEngineAdapter {
    pub fn new(config: RemoteEngineConfig) -> Result<Self, EngineError> {
        Ok(Self {
            inner: RemoteCompletionAdapter::new(config, RuntimeFlavor::Sglang)?,
        })
    }
}

pub struct VllmEngineAdapter {
    inner: RemoteCompletionAdapter,
}

impl VllmEngineAdapter {
    pub fn new(config: RemoteEngineConfig) -> Result<Self, EngineError> {
        Ok(Self {
            inner: RemoteCompletionAdapter::new(config, RuntimeFlavor::Vllm)?,
        })
    }
}

macro_rules! impl_remote_adapter {
    ($adapter:ty) => {
        #[async_trait]
        impl EngineAdapter for $adapter {
            fn instance(&self) -> EngineInstance {
                self.inner.config.instance.clone()
            }

            async fn execution_targets(
                &self,
                context: &OperationContext,
            ) -> Result<Vec<ExecutionTarget>, EngineError> {
                context.ensure_active()?;
                Ok(vec![self.inner.config.target.clone()])
            }

            async fn capabilities(
                &self,
                target: &ExecutionTarget,
                context: &OperationContext,
            ) -> Result<EngineCapabilities, EngineError> {
                context.ensure_active()?;
                self.inner.validate_target(target)?;
                Ok(EngineCapabilities {
                    supported_input_kinds: BTreeSet::from([InputKind::TokenSequence]),
                    emits_token_deltas: false,
                    emits_text_deltas: true,
                    emits_reasoning_deltas: false,
                    emits_tool_calls: false,
                    supports_structured_output: true,
                    supported_state_kinds: BTreeSet::new(),
                })
            }

            async fn snapshot(
                &self,
                target: &ExecutionTarget,
                context: &OperationContext,
            ) -> Result<EngineSnapshot, EngineError> {
                self.inner.snapshot(target, context).await
            }

            async fn prepare_state_import(
                &self,
                _target: &ExecutionTarget,
                _spec: &StateImportSpec,
                _context: &OperationContext,
            ) -> Result<StateImportTarget, EngineError> {
                Err(EngineError::Unsupported(
                    "remote completion adapter does not import reusable state".to_owned(),
                ))
            }

            async fn commit_state_import(
                &self,
                _import: &StateImportTarget,
                _receipt: &TransferReceipt,
                _context: &OperationContext,
            ) -> Result<PreparedStateAttachment, EngineError> {
                Err(EngineError::Unsupported(
                    "remote completion adapter does not import reusable state".to_owned(),
                ))
            }

            async fn abort_state_import(
                &self,
                _import: &StateImportTarget,
                _context: &OperationContext,
            ) -> Result<(), EngineError> {
                Ok(())
            }

            async fn execute(
                &self,
                target: &ExecutionTarget,
                request: CanonicalRequest,
                state: Option<PreparedStateAttachment>,
                context: OperationContext,
            ) -> Result<EngineEventStream, EngineError> {
                self.inner.execute(target, request, state, context).await
            }

            async fn cancel(
                &self,
                request_id: &RequestId,
                context: &OperationContext,
            ) -> Result<(), EngineError> {
                self.inner.cancel(request_id, context).await
            }
        }
    };
}

impl_remote_adapter!(SglangEngineAdapter);
impl_remote_adapter!(VllmEngineAdapter);
