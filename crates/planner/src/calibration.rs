use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use locus_core::{ExecutionTarget, InputItemValue};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ExecutionPath, PlacementPlan, PlanningInput};

const SCHEMA_VERSION: &str = "locus.calibration-state.v1";
pub const ACTIVE_CONFIRMATION: &str = "enable-calibrated-placement";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlacementMode {
    #[default]
    Shadow,
    Active,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationPolicy {
    pub ewma_alpha_bps: u32,
    pub min_samples_per_metric: u64,
    pub max_mape_bps: u64,
    pub min_shadow_decisions: u64,
    pub min_shadow_agreement_bps: u64,
    pub conservative_queue_micros: u64,
    pub conservative_queue_micros_per_waiting_request: u64,
    pub conservative_prefill_micros_per_token: u64,
    pub conservative_decode_micros_per_token: u64,
    pub conservative_materialization_ratio_bps: u64,
    pub conservative_topology_micros: u64,
    pub max_unit_micros: u64,
    pub max_materialization_ratio_bps: u64,
}

impl Default for CalibrationPolicy {
    fn default() -> Self {
        Self {
            ewma_alpha_bps: 1_250,
            min_samples_per_metric: 32,
            max_mape_bps: 2_500,
            min_shadow_decisions: 128,
            min_shadow_agreement_bps: 9_500,
            conservative_queue_micros: 5_000_000,
            conservative_queue_micros_per_waiting_request: 1_000_000,
            conservative_prefill_micros_per_token: 100,
            conservative_decode_micros_per_token: 1_000,
            conservative_materialization_ratio_bps: 15_000,
            conservative_topology_micros: 0,
            max_unit_micros: 60_000_000,
            max_materialization_ratio_bps: 100_000,
        }
    }
}

impl CalibrationPolicy {
    pub fn validate(&self) -> Result<(), CalibrationError> {
        if self.ewma_alpha_bps == 0
            || self.ewma_alpha_bps > 10_000
            || self.min_samples_per_metric == 0
            || self.max_mape_bps == 0
            || self.min_shadow_decisions == 0
            || self.min_shadow_agreement_bps == 0
            || self.min_shadow_agreement_bps > 10_000
            || self.conservative_queue_micros == 0
            || self.conservative_queue_micros_per_waiting_request == 0
            || self.conservative_prefill_micros_per_token == 0
            || self.conservative_decode_micros_per_token == 0
            || self.conservative_materialization_ratio_bps < 10_000
            || self.max_unit_micros == 0
            || self.max_materialization_ratio_bps < self.conservative_materialization_ratio_bps
        {
            return Err(CalibrationError::InvalidPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationKey {
    pub engine_id: String,
    pub engine_generation: u64,
    pub model_revision: String,
    pub adapter_revision: Option<String>,
    pub execution_profile: String,
}

impl CalibrationKey {
    #[must_use]
    pub fn from_target(target: &ExecutionTarget) -> Self {
        Self {
            engine_id: target.engine.id.as_str().to_owned(),
            engine_generation: target.engine.generation,
            model_revision: target.model.model_revision.clone(),
            adapter_revision: target.model.adapter_revision.clone(),
            execution_profile: target.model.execution_profile.clone(),
        }
    }

    fn stable_id(&self) -> String {
        serde_json::to_string(&(
            &self.engine_id,
            self.engine_generation,
            &self.model_revision,
            &self.adapter_revision,
            &self.execution_profile,
        ))
        .expect("calibration key serialization is infallible")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializationObservation {
    pub provider: String,
    pub state_kind: String,
    pub target_id: String,
    pub estimated_micros: u64,
    pub actual_micros: u64,
}

impl MaterializationObservation {
    fn stable_id(&self) -> String {
        materialization_key(&self.provider, &self.state_kind, &self.target_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationObservation {
    pub key: CalibrationKey,
    pub waiting_requests: Option<u64>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub time_to_first_token_micros: Option<u64>,
    pub generation_micros: Option<u64>,
    pub materialization: Option<MaterializationObservation>,
    pub completed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateCalibrationEvidence {
    pub key: CalibrationKey,
    pub snapshot_fresh: bool,
    pub queue_calibration_required: bool,
    pub used_conservative_queue: bool,
    pub used_conservative_prefill: bool,
    pub used_conservative_decode: bool,
    pub revision: u64,
}

#[derive(Clone, Debug)]
pub struct CalibrationApplication {
    pub input: PlanningInput,
    pub evidence: BTreeMap<String, CandidateCalibrationEvidence>,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromotionStatus {
    pub qualified: bool,
    pub revision: u64,
    pub reasons: Vec<String>,
}

#[derive(Clone)]
pub struct PersistentCalibrator {
    policy: CalibrationPolicy,
    state: Arc<Mutex<CalibrationState>>,
    state_path: Option<PathBuf>,
    persistence_healthy: Arc<AtomicBool>,
}

impl PersistentCalibrator {
    pub fn load(
        policy: CalibrationPolicy,
        state_path: Option<PathBuf>,
    ) -> Result<Self, CalibrationError> {
        policy.validate()?;
        let state = match state_path.as_deref() {
            Some(path) if path.exists() => load_state(path)?,
            _ => CalibrationState::default(),
        };
        Ok(Self {
            policy,
            state: Arc::new(Mutex::new(state)),
            state_path,
            persistence_healthy: Arc::new(AtomicBool::new(true)),
        })
    }

    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.state_path.is_some()
    }

    pub fn apply(
        &self,
        input: &PlanningInput,
        now_unix_millis: u64,
    ) -> Result<CalibrationApplication, CalibrationError> {
        let state = self.lock_state()?;
        let mut calibrated = input.clone();
        let input_tokens = input_token_count(input);
        let output_tokens = u64::from(input.request.sampling.max_output_tokens.unwrap_or(16));
        let mut evidence = BTreeMap::new();
        for candidate in &mut calibrated.candidates {
            let key = CalibrationKey::from_target(&candidate.target);
            let record = state.records.get(&key.stable_id());
            let snapshot_fresh = candidate.snapshot.telemetry_is_fresh_at(now_unix_millis);
            let waiting = if snapshot_fresh {
                candidate.snapshot.waiting_requests
            } else {
                None
            };
            let direct_queue_estimate = if snapshot_fresh {
                candidate.snapshot.estimated_queue_micros
            } else {
                None
            };
            let queue_calibration_required =
                direct_queue_estimate.is_none() && waiting.is_none_or(|waiting| waiting > 0);
            let (queue_micros, used_conservative_queue) =
                queue_estimate(&self.policy, record, waiting, direct_queue_estimate);
            let (prefill_unit, used_conservative_prefill) = unit_estimate(
                &self.policy,
                record.map(|record| &record.prefill_micros_per_token),
                if snapshot_fresh {
                    candidate.snapshot.prefill_tokens_per_second
                } else {
                    None
                },
                self.policy.conservative_prefill_micros_per_token,
            );
            let (decode_unit, used_conservative_decode) = unit_estimate(
                &self.policy,
                record.map(|record| &record.decode_micros_per_token),
                if snapshot_fresh {
                    candidate.snapshot.decode_tokens_per_second
                } else {
                    None
                },
                self.policy.conservative_decode_micros_per_token,
            );
            candidate.snapshot.estimated_queue_micros = Some(queue_micros);
            candidate.cold_estimate.unmatched_prefill_micros =
                input_tokens.saturating_mul(prefill_unit);
            candidate.cold_estimate.decode_micros = output_tokens.saturating_mul(decode_unit);
            candidate.cold_estimate.topology_micros = self.policy.conservative_topology_micros;
            for state_path in &mut candidate.state_paths {
                let covered = state_path
                    .state
                    .boundary
                    .covered_components
                    .iter()
                    .map(|component| component.covered_units)
                    .sum::<u64>();
                state_path.estimate.unmatched_prefill_micros = input_tokens
                    .saturating_sub(covered)
                    .saturating_mul(prefill_unit);
                state_path.estimate.decode_micros = output_tokens.saturating_mul(decode_unit);
                state_path.estimate.topology_micros = self.policy.conservative_topology_micros;
                let path_key = materialization_key(
                    state_path.option.provider.as_str(),
                    state_path.option.state_kind.as_str(),
                    &candidate.target.id.to_string(),
                );
                let ratio = record
                    .and_then(|record| record.materialization_ratio_bps.get(&path_key))
                    .filter(|metric| metric.samples > 0)
                    .map_or(
                        self.policy.conservative_materialization_ratio_bps,
                        |metric| metric.estimate,
                    );
                state_path.option.estimated_transfer_micros = state_path
                    .option
                    .estimated_transfer_micros
                    .saturating_mul(ratio)
                    / 10_000;
            }
            evidence.insert(
                candidate.target.id.to_string(),
                CandidateCalibrationEvidence {
                    key,
                    snapshot_fresh,
                    queue_calibration_required,
                    used_conservative_queue,
                    used_conservative_prefill,
                    used_conservative_decode,
                    revision: state.revision,
                },
            );
        }
        Ok(CalibrationApplication {
            input: calibrated,
            evidence,
            revision: state.revision,
        })
    }

    pub fn record_observation(
        &self,
        observation: &CalibrationObservation,
    ) -> Result<(), CalibrationError> {
        if !observation.completed {
            return Ok(());
        }
        let mut state = self.lock_state()?;
        let record = state
            .records
            .entry(observation.key.stable_id())
            .or_default();
        let mut changed = false;
        if let Some(ttft) = observation.time_to_first_token_micros
            && observation.input_tokens > 0
        {
            match observation.waiting_requests {
                Some(0) => {
                    let sample = ttft / observation.input_tokens.max(1);
                    record.prefill_micros_per_token.update(
                        sample,
                        &self.policy,
                        self.policy.max_unit_micros,
                    );
                    changed = true;
                }
                Some(waiting) if waiting > 0 => {
                    let prefill_unit = record
                        .prefill_micros_per_token
                        .value_or(self.policy.conservative_prefill_micros_per_token);
                    let prefill = observation.input_tokens.saturating_mul(prefill_unit);
                    let residual = ttft.saturating_sub(prefill);
                    record.queue_micros_per_waiting_request.update(
                        residual / waiting,
                        &self.policy,
                        self.policy.max_unit_micros,
                    );
                    changed = true;
                }
                _ => {}
            }
        }
        if let Some(generation_micros) = observation.generation_micros
            && observation.output_tokens > 0
        {
            record.decode_micros_per_token.update(
                generation_micros / observation.output_tokens,
                &self.policy,
                self.policy.max_unit_micros,
            );
            changed = true;
        }
        if let Some(materialization) = &observation.materialization
            && materialization.estimated_micros > 0
        {
            let ratio_bps = materialization.actual_micros.saturating_mul(10_000)
                / materialization.estimated_micros;
            record
                .materialization_ratio_bps
                .entry(materialization.stable_id())
                .or_default()
                .update(
                    ratio_bps,
                    &self.policy,
                    self.policy.max_materialization_ratio_bps,
                );
            changed = true;
        }
        if changed {
            state.revision = state.revision.saturating_add(1);
            self.persist_locked(&state)?;
        }
        Ok(())
    }

    pub fn record_shadow_decision(
        &self,
        replay_consistent: bool,
        legacy_fingerprint: &str,
        calibrated_fingerprint: &str,
    ) -> Result<(), CalibrationError> {
        let mut state = self.lock_state()?;
        state.shadow.decisions = state.shadow.decisions.saturating_add(1);
        if legacy_fingerprint == calibrated_fingerprint {
            state.shadow.agreements = state.shadow.agreements.saturating_add(1);
        }
        if !replay_consistent {
            state.shadow.replay_mismatches = state.shadow.replay_mismatches.saturating_add(1);
        }
        state.revision = state.revision.saturating_add(1);
        self.persist_locked(&state)
    }

    pub fn promotion_status(
        &self,
        plan: &PlacementPlan,
        evidence: Option<&CandidateCalibrationEvidence>,
    ) -> Result<PromotionStatus, CalibrationError> {
        let state = self.lock_state()?;
        let mut reasons = Vec::new();
        if !self.is_persistent() {
            reasons.push("calibration state is not persistent".to_owned());
        }
        if !self.persistence_healthy.load(Ordering::Acquire) {
            reasons.push("calibration persistence is unhealthy".to_owned());
        }
        if state.shadow.decisions < self.policy.min_shadow_decisions {
            reasons.push(format!(
                "shadow decisions {}/{}",
                state.shadow.decisions, self.policy.min_shadow_decisions
            ));
        }
        if let Some(agreement_bps) = state
            .shadow
            .agreements
            .saturating_mul(10_000)
            .checked_div(state.shadow.decisions)
        {
            if agreement_bps < self.policy.min_shadow_agreement_bps {
                reasons.push(format!(
                    "shadow agreement {agreement_bps} bps is below {} bps",
                    self.policy.min_shadow_agreement_bps
                ));
            }
        }
        if state.shadow.replay_mismatches > 0 {
            reasons.push(format!(
                "deterministic replay mismatches: {}",
                state.shadow.replay_mismatches
            ));
        }
        let Some(evidence) = evidence else {
            reasons.push("selected target has no calibration evidence".to_owned());
            return Ok(PromotionStatus {
                qualified: false,
                revision: state.revision,
                reasons,
            });
        };
        if !evidence.snapshot_fresh {
            reasons.push("selected target telemetry is stale or unavailable".to_owned());
        }
        let record = state.records.get(&evidence.key.stable_id());
        qualify_metric(
            record.map(|record| &record.prefill_micros_per_token),
            "prefill",
            &self.policy,
            &mut reasons,
        );
        qualify_metric(
            record.map(|record| &record.decode_micros_per_token),
            "decode",
            &self.policy,
            &mut reasons,
        );
        if evidence.queue_calibration_required {
            qualify_metric(
                record.map(|record| &record.queue_micros_per_waiting_request),
                "queue",
                &self.policy,
                &mut reasons,
            );
        }
        if let ExecutionPath::Reuse(reuse) = &plan.path {
            let key = materialization_key(
                reuse.option.provider.as_str(),
                reuse.option.state_kind.as_str(),
                &plan.target.id.to_string(),
            );
            qualify_metric(
                record.and_then(|record| record.materialization_ratio_bps.get(&key)),
                "materialization",
                &self.policy,
                &mut reasons,
            );
        }
        Ok(PromotionStatus {
            qualified: reasons.is_empty(),
            revision: state.revision,
            reasons,
        })
    }

    pub fn state_json(&self) -> Result<String, CalibrationError> {
        let state = self.lock_state()?;
        serde_json::to_string_pretty(&*state).map_err(CalibrationError::Serialize)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, CalibrationState>, CalibrationError> {
        self.state
            .lock()
            .map_err(|_| CalibrationError::LockPoisoned)
    }

    fn persist_locked(&self, state: &CalibrationState) -> Result<(), CalibrationError> {
        let Some(path) = &self.state_path else {
            return Ok(());
        };
        if let Err(error) = persist_state(path, state) {
            self.persistence_healthy.store(false, Ordering::Release);
            return Err(error);
        }
        self.persistence_healthy.store(true, Ordering::Release);
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CalibrationState {
    schema_version: String,
    revision: u64,
    records: BTreeMap<String, CalibrationRecord>,
    shadow: ShadowEvidence,
}

impl Default for CalibrationState {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            revision: 0,
            records: BTreeMap::new(),
            shadow: ShadowEvidence::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CalibrationRecord {
    queue_micros_per_waiting_request: MetricEstimate,
    prefill_micros_per_token: MetricEstimate,
    decode_micros_per_token: MetricEstimate,
    materialization_ratio_bps: BTreeMap<String, MetricEstimate>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct MetricEstimate {
    estimate: u64,
    samples: u64,
    mape_bps: u64,
}

impl MetricEstimate {
    fn update(&mut self, sample: u64, policy: &CalibrationPolicy, maximum: u64) {
        let sample = sample.clamp(1, maximum);
        if self.samples == 0 {
            self.estimate = sample;
            self.mape_bps = 0;
            self.samples = 1;
            return;
        }
        let error_bps = self.estimate.abs_diff(sample).saturating_mul(10_000) / sample.max(1);
        self.estimate = weighted_average(self.estimate, sample, u64::from(policy.ewma_alpha_bps));
        self.mape_bps =
            weighted_average(self.mape_bps, error_bps, u64::from(policy.ewma_alpha_bps));
        self.samples = self.samples.saturating_add(1);
    }

    fn value_or(&self, fallback: u64) -> u64 {
        if self.samples == 0 {
            fallback
        } else {
            self.estimate
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ShadowEvidence {
    decisions: u64,
    agreements: u64,
    replay_mismatches: u64,
}

fn weighted_average(previous: u64, sample: u64, alpha_bps: u64) -> u64 {
    let retained = 10_000_u64.saturating_sub(alpha_bps);
    let total = u128::from(previous)
        .saturating_mul(u128::from(retained))
        .saturating_add(u128::from(sample).saturating_mul(u128::from(alpha_bps)))
        / 10_000;
    u64::try_from(total).unwrap_or(u64::MAX)
}

fn queue_estimate(
    policy: &CalibrationPolicy,
    record: Option<&CalibrationRecord>,
    waiting_requests: Option<u64>,
    direct_estimate: Option<u64>,
) -> (u64, bool) {
    if let Some(direct) = direct_estimate {
        return (direct, false);
    }
    match waiting_requests {
        Some(0) => (0, false),
        Some(waiting) => {
            let metric = record.map(|record| &record.queue_micros_per_waiting_request);
            let unit = metric.map_or(
                policy.conservative_queue_micros_per_waiting_request,
                |metric| metric.value_or(policy.conservative_queue_micros_per_waiting_request),
            );
            (
                waiting.saturating_mul(unit),
                metric.is_none_or(|metric| metric.samples == 0),
            )
        }
        None => (policy.conservative_queue_micros, true),
    }
}

fn unit_estimate(
    policy: &CalibrationPolicy,
    metric: Option<&MetricEstimate>,
    telemetry_rate: Option<u64>,
    conservative: u64,
) -> (u64, bool) {
    if let Some(metric) = metric.filter(|metric| metric.samples > 0) {
        return (metric.estimate, false);
    }
    if let Some(rate) = telemetry_rate.filter(|rate| *rate > 0) {
        return ((1_000_000 / rate).clamp(1, policy.max_unit_micros), false);
    }
    (conservative, true)
}

fn qualify_metric(
    metric: Option<&MetricEstimate>,
    name: &str,
    policy: &CalibrationPolicy,
    reasons: &mut Vec<String>,
) {
    let Some(metric) = metric else {
        reasons.push(format!("{name} calibration is missing"));
        return;
    };
    if metric.samples < policy.min_samples_per_metric {
        reasons.push(format!(
            "{name} samples {}/{}",
            metric.samples, policy.min_samples_per_metric
        ));
    }
    if metric.mape_bps > policy.max_mape_bps {
        reasons.push(format!(
            "{name} MAPE {} bps exceeds {} bps",
            metric.mape_bps, policy.max_mape_bps
        ));
    }
}

fn input_token_count(input: &PlanningInput) -> u64 {
    input
        .request
        .input
        .items
        .iter()
        .filter_map(|item| match &item.value {
            InputItemValue::TokenSequence(tokens) => Some(tokens.token_ids.len() as u64),
            _ => None,
        })
        .sum()
}

fn materialization_key(provider: &str, state_kind: &str, target_id: &str) -> String {
    serde_json::to_string(&(provider, state_kind, target_id))
        .expect("materialization key serialization is infallible")
}

pub fn plan_fingerprint(plan: &PlacementPlan) -> String {
    let path = match &plan.path {
        ExecutionPath::Cold => "cold".to_owned(),
        ExecutionPath::Reuse(reuse) => format!(
            "reuse:{}:{}",
            reuse.state.id.as_str(),
            reuse.option.id.as_str()
        ),
    };
    format!(
        "{}|{}|{}",
        plan.target.id,
        path,
        plan.predicted_cost.total_micros()
    )
}

fn load_state(path: &Path) -> Result<CalibrationState, CalibrationError> {
    let bytes = fs::read(path).map_err(|source| CalibrationError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let state = serde_json::from_slice::<CalibrationState>(&bytes).map_err(|source| {
        CalibrationError::Parse {
            path: path.to_path_buf(),
            source,
        }
    })?;
    if state.schema_version != SCHEMA_VERSION {
        return Err(CalibrationError::Schema {
            expected: SCHEMA_VERSION.to_owned(),
            actual: state.schema_version,
        });
    }
    Ok(state)
}

fn persist_state(path: &Path, state: &CalibrationState) -> Result<(), CalibrationError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| CalibrationError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temporary = path.with_extension("locus-calibration.tmp");
    let bytes = serde_json::to_vec_pretty(state).map_err(CalibrationError::Serialize)?;
    fs::write(&temporary, bytes).map_err(|source| CalibrationError::Write {
        path: temporary.clone(),
        source,
    })?;
    File::open(&temporary)
        .and_then(|file| file.sync_all())
        .map_err(|source| CalibrationError::Write {
            path: temporary.clone(),
            source,
        })?;
    fs::rename(&temporary, path).map_err(|source| CalibrationError::Write {
        path: path.to_path_buf(),
        source,
    })?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|source| CalibrationError::Write {
                path: parent.to_path_buf(),
                source,
            })?;
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("calibration policy is invalid")]
    InvalidPolicy,
    #[error("calibration state lock is poisoned")]
    LockPoisoned,
    #[error("failed to read calibration state {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse calibration state {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("calibration state schema mismatch: expected {expected}, got {actual}")]
    Schema { expected: String, actual: String },
    #[error("failed to serialize calibration state: {0}")]
    Serialize(serde_json::Error),
    #[error("failed to write calibration state {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
