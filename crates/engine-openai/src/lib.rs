mod metrics;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use locus_core::{
    CanonicalRequest, EngineCapabilities, EngineEvent, EngineFinishReason, EngineInstance,
    EngineSnapshot, ExecutionTarget, InputItemValue, InputKind, OperationContext,
    PreparedStateAttachment, RequestId, StateImportSpec, StateImportTarget, TelemetryConfidence,
    TelemetryStatus, TransferReceipt, Usage,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use locus_engine::{EngineAdapter, EngineError, EngineEventStream};
use metrics::{PrometheusSample, parse_prometheus};

#[derive(Clone, Debug)]
pub struct RemoteEngineConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub instance: EngineInstance,
    pub targets: Vec<RemoteExecutionTarget>,
    pub telemetry: RemoteTelemetryConfig,
}

#[derive(Clone, Debug)]
pub struct RemoteTelemetryConfig {
    pub metrics_path: String,
    pub request_timeout_millis: u64,
    pub min_scrape_interval_millis: u64,
    pub valid_for_millis: u64,
    pub max_response_bytes: usize,
    pub max_samples: usize,
    pub require_fresh_metrics: bool,
}

impl Default for RemoteTelemetryConfig {
    fn default() -> Self {
        Self {
            metrics_path: "/metrics".to_owned(),
            request_timeout_millis: 2_000,
            min_scrape_interval_millis: 500,
            valid_for_millis: 5_000,
            max_response_bytes: 2 * 1024 * 1024,
            max_samples: 20_000,
            require_fresh_metrics: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RemoteExecutionTarget {
    /// Model name accepted by the downstream OpenAI-compatible API.
    pub served_model: String,
    /// Immutable Locus identity and planner metadata for this candidate.
    pub target: ExecutionTarget,
}

#[derive(Clone, Copy)]
enum RuntimeFlavor {
    Sglang,
    Vllm,
}

impl RuntimeFlavor {
    fn as_str(self) -> &'static str {
        match self {
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }
}

struct RemoteCompletionAdapter {
    config: RemoteEngineConfig,
    flavor: RuntimeFlavor,
    client: Client,
    telemetry: Mutex<TelemetryState>,
}

#[derive(Default)]
struct TelemetryState {
    revision: u64,
    cache: Option<CachedTelemetry>,
    counters: BTreeMap<String, CounterBaseline>,
}

#[derive(Clone)]
struct CachedTelemetry {
    observed_at_unix_millis: u64,
    revision: u64,
    targets: BTreeMap<String, TargetTelemetry>,
}

#[derive(Clone, Default)]
struct TargetTelemetry {
    running_requests: Option<u64>,
    waiting_requests: Option<u64>,
    estimated_queue_micros: Option<u64>,
    kv_cache_usage_permyriad: Option<u16>,
    prefill_tokens_per_second: Option<u64>,
    decode_tokens_per_second: Option<u64>,
}

#[derive(Clone)]
struct CounterBaseline {
    observed_at_unix_millis: u64,
    prompt_tokens: Option<f64>,
    generation_tokens: Option<f64>,
}

impl RemoteCompletionAdapter {
    fn new(config: RemoteEngineConfig, flavor: RuntimeFlavor) -> Result<Self, EngineError> {
        if config.base_url.trim().is_empty() {
            return Err(EngineError::Execution(
                "remote engine base URL must not be empty".to_owned(),
            ));
        }
        if config.targets.is_empty() {
            return Err(EngineError::Execution(
                "remote engine must have at least one model candidate".to_owned(),
            ));
        }
        if config.telemetry.metrics_path.trim().is_empty()
            || config.telemetry.request_timeout_millis == 0
            || config.telemetry.min_scrape_interval_millis == 0
            || config.telemetry.valid_for_millis == 0
            || config.telemetry.max_response_bytes == 0
            || config.telemetry.max_samples == 0
        {
            return Err(EngineError::Execution(
                "remote telemetry limits and metrics path must be non-empty and non-zero"
                    .to_owned(),
            ));
        }
        let mut target_ids = BTreeSet::new();
        let mut served_models = BTreeSet::new();
        for candidate in &config.targets {
            if candidate.target.engine != config.instance.reference {
                return Err(EngineError::Execution(
                    "remote target generation does not match engine instance".to_owned(),
                ));
            }
            if candidate.served_model.trim().is_empty() {
                return Err(EngineError::Execution(
                    "remote served model must not be empty".to_owned(),
                ));
            }
            if !target_ids.insert(candidate.target.id.clone()) {
                return Err(EngineError::Execution(format!(
                    "duplicate remote target id: {}",
                    candidate.target.id
                )));
            }
            if !served_models.insert(candidate.served_model.clone()) {
                return Err(EngineError::Execution(format!(
                    "duplicate remote served model: {}",
                    candidate.served_model
                )));
            }
        }
        Ok(Self {
            config,
            flavor,
            client: Client::new(),
            telemetry: Mutex::new(TelemetryState::default()),
        })
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}{}", self.config.base_url.trim_end_matches('/'), path)
    }

    fn metrics_endpoint(&self) -> String {
        if self.config.telemetry.metrics_path.starts_with("http://")
            || self.config.telemetry.metrics_path.starts_with("https://")
        {
            self.config.telemetry.metrics_path.clone()
        } else {
            self.endpoint(&format!(
                "/{}",
                self.config.telemetry.metrics_path.trim_start_matches('/')
            ))
        }
    }

    fn target_config(
        &self,
        target: &ExecutionTarget,
    ) -> Result<&RemoteExecutionTarget, EngineError> {
        self.config
            .targets
            .iter()
            .find(|candidate| candidate.target == *target)
            .ok_or_else(|| EngineError::TargetNotFound(target.id.to_string()))
    }

    fn authorized(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(api_key) = &self.config.api_key {
            request.bearer_auth(api_key)
        } else {
            request
        }
    }

    fn completion_body(
        &self,
        request: &CanonicalRequest,
        served_model: &str,
    ) -> Result<Value, EngineError> {
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
            "model": served_model,
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
        if !request.sampling.stop_sequences.is_empty() {
            body["stop"] = json!(&request.sampling.stop_sequences);
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

    async fn execution_targets(
        &self,
        context: &OperationContext,
    ) -> Result<Vec<ExecutionTarget>, EngineError> {
        context.ensure_active()?;
        let response = self
            .authorized(self.client.get(self.endpoint("/v1/models")))
            .send()
            .await
            .map_err(|error| {
                EngineError::Execution(format!("model discovery request failed: {error}"))
            })?;
        if !response.status().is_success() {
            let status = response.status();
            let detail = response
                .text()
                .await
                .unwrap_or_else(|_| "response body was unavailable".to_owned());
            return Err(EngineError::Execution(format!(
                "model discovery endpoint returned {status}: {detail}"
            )));
        }
        let inventory = response.json::<RemoteModelList>().await.map_err(|error| {
            EngineError::Execution(format!("invalid model discovery response: {error}"))
        })?;
        let active = inventory
            .data
            .into_iter()
            .map(|model| model.id)
            .collect::<BTreeSet<_>>();
        let mut targets = self
            .config
            .targets
            .iter()
            .filter(|candidate| active.contains(&candidate.served_model))
            .map(|candidate| candidate.target.clone())
            .collect::<Vec<_>>();
        targets.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(targets)
    }

    fn cached_telemetry(
        &self,
        served_model: &str,
        now_unix_millis: u64,
    ) -> Result<Option<(TargetTelemetry, u64, u64, TelemetryStatus)>, EngineError> {
        let state = self
            .telemetry
            .lock()
            .map_err(|_| EngineError::Execution("telemetry cache lock poisoned".to_owned()))?;
        let Some(cache) = &state.cache else {
            return Ok(None);
        };
        let Some(telemetry) = cache.targets.get(served_model).cloned() else {
            return Ok(None);
        };
        let valid_until = cache
            .observed_at_unix_millis
            .saturating_add(self.config.telemetry.valid_for_millis);
        let status =
            if cache.observed_at_unix_millis <= now_unix_millis && now_unix_millis <= valid_until {
                TelemetryStatus::Fresh
            } else {
                TelemetryStatus::Stale
            };
        Ok(Some((
            telemetry,
            cache.observed_at_unix_millis,
            cache.revision,
            status,
        )))
    }

    async fn observe_telemetry(
        &self,
        served_model: &str,
    ) -> Result<(TargetTelemetry, u64, u64), EngineError> {
        let now_unix_millis = unix_millis()?;
        if let Some((telemetry, observed_at, revision, _)) =
            self.cached_telemetry(served_model, now_unix_millis)?
            && now_unix_millis.saturating_sub(observed_at)
                < self.config.telemetry.min_scrape_interval_millis
        {
            return Ok((telemetry, observed_at, revision));
        }

        let response =
            self.authorized(self.client.get(self.metrics_endpoint()).timeout(
                Duration::from_millis(self.config.telemetry.request_timeout_millis),
            ))
            .send()
            .await
            .map_err(|error| {
                EngineError::Execution(format!("telemetry request failed: {error}"))
            })?;
        if !response.status().is_success() {
            return Err(EngineError::Execution(format!(
                "telemetry endpoint returned {}",
                response.status()
            )));
        }
        let bytes = response.bytes().await.map_err(|error| {
            EngineError::Execution(format!("telemetry response read failed: {error}"))
        })?;
        if bytes.len() > self.config.telemetry.max_response_bytes {
            return Err(EngineError::Execution(format!(
                "telemetry response exceeds the {}-byte limit",
                self.config.telemetry.max_response_bytes
            )));
        }
        let text = std::str::from_utf8(&bytes).map_err(|error| {
            EngineError::Execution(format!("telemetry response is not UTF-8: {error}"))
        })?;
        let samples = parse_prometheus(text, self.config.telemetry.max_samples)
            .map_err(|error| EngineError::Execution(format!("invalid telemetry: {error}")))?;
        let observed_at_unix_millis = unix_millis()?;
        let mut state = self
            .telemetry
            .lock()
            .map_err(|_| EngineError::Execution("telemetry cache lock poisoned".to_owned()))?;
        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let mut targets = BTreeMap::new();
        for candidate in &self.config.targets {
            let counters = metric_counters(self.flavor, &samples, &candidate.served_model);
            let previous = state.counters.get(&candidate.served_model);
            let telemetry = target_telemetry(
                self.flavor,
                &samples,
                &candidate.served_model,
                previous,
                &counters,
                observed_at_unix_millis,
            );
            state.counters.insert(
                candidate.served_model.clone(),
                CounterBaseline {
                    observed_at_unix_millis,
                    prompt_tokens: counters.prompt_tokens,
                    generation_tokens: counters.generation_tokens,
                },
            );
            targets.insert(candidate.served_model.clone(), telemetry);
        }
        let result = targets.get(served_model).cloned().ok_or_else(|| {
            EngineError::Execution(format!(
                "telemetry has no configured candidate for {served_model}"
            ))
        })?;
        state.cache = Some(CachedTelemetry {
            observed_at_unix_millis,
            revision,
            targets,
        });
        Ok((result, observed_at_unix_millis, revision))
    }

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, EngineError> {
        context.ensure_active()?;
        let served_model = self.target_config(target)?.served_model.clone();
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
        let health_ready = response.status().is_success();
        let now_unix_millis = unix_millis()?;
        let (telemetry, observed_at_unix_millis, observation_revision, telemetry_status, reason) =
            match self.observe_telemetry(&served_model).await {
                Ok((telemetry, observed_at, revision)) => (
                    telemetry,
                    observed_at,
                    revision,
                    TelemetryStatus::Fresh,
                    None,
                ),
                Err(error) => match self.cached_telemetry(&served_model, now_unix_millis)? {
                    Some((telemetry, observed_at, revision, status)) => (
                        telemetry,
                        observed_at,
                        revision,
                        status,
                        Some(error.to_string()),
                    ),
                    None => (
                        TargetTelemetry::default(),
                        now_unix_millis,
                        0,
                        TelemetryStatus::Unavailable,
                        Some(error.to_string()),
                    ),
                },
            };
        let fresh = telemetry_status == TelemetryStatus::Fresh;
        let confidence = telemetry.confidence();
        Ok(EngineSnapshot {
            target_id: target.id.clone(),
            ready: health_ready && (!self.config.telemetry.require_fresh_metrics || fresh),
            telemetry_status,
            telemetry_confidence: if fresh {
                confidence
            } else {
                TelemetryConfidence::Unknown
            },
            telemetry_source: format!("prometheus:{}", self.flavor.as_str()),
            observed_at_unix_millis,
            valid_until_unix_millis: observed_at_unix_millis
                .saturating_add(self.config.telemetry.valid_for_millis),
            running_requests: telemetry.running_requests,
            waiting_requests: telemetry.waiting_requests,
            estimated_queue_micros: telemetry.estimated_queue_micros,
            kv_cache_usage_permyriad: telemetry.kv_cache_usage_permyriad,
            prefill_tokens_per_second: telemetry.prefill_tokens_per_second,
            decode_tokens_per_second: telemetry.decode_tokens_per_second,
            observation_revision,
            degraded_reason: reason,
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
        let served_model = self.target_config(target)?.served_model.clone();
        if state.is_some() {
            return Err(EngineError::Unsupported(
                "remote completion adapter does not import reusable state".to_owned(),
            ));
        }
        let body = self.completion_body(&request, &served_model)?;
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

impl TargetTelemetry {
    fn confidence(&self) -> TelemetryConfidence {
        let scheduler = self.running_requests.is_some() && self.waiting_requests.is_some();
        let rates =
            self.prefill_tokens_per_second.is_some() || self.decode_tokens_per_second.is_some();
        if scheduler && rates {
            TelemetryConfidence::High
        } else if scheduler {
            TelemetryConfidence::Medium
        } else if rates || self.kv_cache_usage_permyriad.is_some() {
            TelemetryConfidence::Low
        } else {
            TelemetryConfidence::Unknown
        }
    }
}

struct MetricCounters {
    prompt_tokens: Option<f64>,
    generation_tokens: Option<f64>,
}

fn unix_millis() -> Result<u64, EngineError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            EngineError::Execution(format!("system clock precedes Unix epoch: {error}"))
        })?;
    u64::try_from(duration.as_millis())
        .map_err(|_| EngineError::Execution("system clock exceeds telemetry range".to_owned()))
}

fn target_telemetry(
    flavor: RuntimeFlavor,
    samples: &[PrometheusSample],
    served_model: &str,
    previous: Option<&CounterBaseline>,
    counters: &MetricCounters,
    observed_at_unix_millis: u64,
) -> TargetTelemetry {
    let (running_names, waiting_names, kv_names, direct_decode_names) = match flavor {
        RuntimeFlavor::Sglang => (
            &["sglang:num_running_reqs", "sglang_num_running_reqs"][..],
            &["sglang:num_queue_reqs", "sglang_num_queue_reqs"][..],
            &[
                "sglang:token_usage",
                "sglang_token_usage",
                "sglang:full_token_usage",
                "sglang_full_token_usage",
            ][..],
            &["sglang:gen_throughput", "sglang_gen_throughput"][..],
        ),
        RuntimeFlavor::Vllm => (
            &["vllm:num_requests_running", "vllm_num_requests_running"][..],
            &["vllm:num_requests_waiting", "vllm_num_requests_waiting"][..],
            &[
                "vllm:kv_cache_usage_perc",
                "vllm_kv_cache_usage_perc",
                "vllm:gpu_cache_usage_perc",
                "vllm_gpu_cache_usage_perc",
            ][..],
            &[][..],
        ),
    };
    let direct_queue_seconds = metric_max(
        samples,
        &[
            "locus:estimated_queue_seconds",
            "sglang:estimated_queue_time_seconds",
            "vllm:estimated_queue_time_seconds",
        ],
        served_model,
    );
    let elapsed_millis = previous
        .map(|baseline| observed_at_unix_millis.saturating_sub(baseline.observed_at_unix_millis));
    let prefill_rate = counter_rate(
        counters.prompt_tokens,
        previous.and_then(|baseline| baseline.prompt_tokens),
        elapsed_millis,
    );
    let decode_rate = metric_sum(samples, direct_decode_names, served_model)
        .and_then(nonnegative_u64)
        .or_else(|| {
            counter_rate(
                counters.generation_tokens,
                previous.and_then(|baseline| baseline.generation_tokens),
                elapsed_millis,
            )
        });
    TargetTelemetry {
        running_requests: metric_sum(samples, running_names, served_model)
            .and_then(nonnegative_u64),
        waiting_requests: metric_sum(samples, waiting_names, served_model)
            .and_then(nonnegative_u64),
        estimated_queue_micros: direct_queue_seconds.and_then(seconds_to_micros),
        kv_cache_usage_permyriad: metric_max(samples, kv_names, served_model)
            .and_then(ratio_to_permyriad),
        prefill_tokens_per_second: prefill_rate,
        decode_tokens_per_second: decode_rate,
    }
}

fn metric_counters(
    flavor: RuntimeFlavor,
    samples: &[PrometheusSample],
    served_model: &str,
) -> MetricCounters {
    let (prompt_names, generation_names) = match flavor {
        RuntimeFlavor::Sglang => (
            &[
                "sglang:prompt_tokens_total",
                "sglang_prompt_tokens_total",
                "sglang:input_tokens_total",
                "sglang_input_tokens_total",
            ][..],
            &[
                "sglang:generation_tokens_total",
                "sglang_generation_tokens_total",
                "sglang:output_tokens_total",
                "sglang_output_tokens_total",
            ][..],
        ),
        RuntimeFlavor::Vllm => (
            &[
                "vllm:prompt_tokens_total",
                "vllm_prompt_tokens_total",
                "vllm:prompt_tokens",
                "vllm_prompt_tokens",
            ][..],
            &[
                "vllm:generation_tokens_total",
                "vllm_generation_tokens_total",
                "vllm:generation_tokens",
                "vllm_generation_tokens",
            ][..],
        ),
    };
    MetricCounters {
        prompt_tokens: metric_sum(samples, prompt_names, served_model),
        generation_tokens: metric_sum(samples, generation_names, served_model),
    }
}

fn metric_sum(samples: &[PrometheusSample], names: &[&str], served_model: &str) -> Option<f64> {
    for name in names {
        let matching = samples
            .iter()
            .filter(|sample| sample.name == *name && sample_matches_model(sample, served_model))
            .map(|sample| sample.value)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .collect::<Vec<_>>();
        if !matching.is_empty() {
            return Some(matching.into_iter().sum());
        }
    }
    None
}

fn metric_max(samples: &[PrometheusSample], names: &[&str], served_model: &str) -> Option<f64> {
    for name in names {
        let maximum = samples
            .iter()
            .filter(|sample| sample.name == *name && sample_matches_model(sample, served_model))
            .map(|sample| sample.value)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .reduce(f64::max);
        if maximum.is_some() {
            return maximum;
        }
    }
    None
}

fn sample_matches_model(sample: &PrometheusSample, served_model: &str) -> bool {
    let mut observed_model_label = false;
    for key in ["model_name", "model", "served_model_name"] {
        if let Some(value) = sample.labels.get(key) {
            observed_model_label = true;
            if value == served_model {
                return true;
            }
        }
    }
    !observed_model_label
}

fn nonnegative_u64(value: f64) -> Option<u64> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        None
    } else {
        Some(value.round() as u64)
    }
}

fn seconds_to_micros(value: f64) -> Option<u64> {
    nonnegative_u64(value * 1_000_000.0)
}

fn ratio_to_permyriad(value: f64) -> Option<u16> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        None
    } else {
        Some((value * 10_000.0).round() as u16)
    }
}

fn counter_rate(
    current: Option<f64>,
    previous: Option<f64>,
    elapsed_millis: Option<u64>,
) -> Option<u64> {
    let (current, previous, elapsed_millis) = (current?, previous?, elapsed_millis?);
    if !current.is_finite() || !previous.is_finite() || current < previous || elapsed_millis == 0 {
        return None;
    }
    nonnegative_u64((current - previous) * 1_000.0 / elapsed_millis as f64)
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

#[derive(Deserialize)]
struct RemoteModelList {
    #[serde(default)]
    data: Vec<RemoteModel>,
}

#[derive(Deserialize)]
struct RemoteModel {
    id: String,
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
                self.inner.execution_targets(context).await
            }

            async fn capabilities(
                &self,
                target: &ExecutionTarget,
                context: &OperationContext,
            ) -> Result<EngineCapabilities, EngineError> {
                context.ensure_active()?;
                self.inner.target_config(target)?;
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
