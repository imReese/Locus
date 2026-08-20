use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use locus_core::{CancellationToken, OperationContext, RequestId};
use thiserror::Error;
use tokio::sync::{Notify, oneshot};

const VIRTUAL_RUNTIME_SCALE: u128 = 1_000_000;
const MAX_SERVICE_CLASSES: usize = 64;
const MAX_TENANTS: usize = 1_024;

#[derive(Clone, Debug)]
pub struct ServiceClassPolicy {
    pub id: String,
    pub weight: u32,
    pub max_active_requests: usize,
    pub max_active_tokens: u64,
    /// When active-token utilization reaches this threshold, this class is
    /// shed before it consumes queue capacity. `None` disables early shedding.
    pub shed_at_global_utilization_bps: Option<u16>,
}

#[derive(Clone, Debug)]
pub struct TenantPolicy {
    pub id: String,
    pub service_class: String,
    pub weight: u32,
    pub max_active_requests: usize,
    pub max_active_tokens: u64,
    pub max_queued_requests: usize,
    pub max_tokens_per_request: u64,
    pub default_request_timeout: Duration,
    pub max_request_timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct TrafficPolicy {
    pub max_active_requests: usize,
    pub max_active_tokens: u64,
    pub max_queued_requests: usize,
    pub default_output_tokens: u64,
    pub classes: Vec<ServiceClassPolicy>,
    pub tenants: Vec<TenantPolicy>,
}

impl Default for TrafficPolicy {
    fn default() -> Self {
        Self {
            max_active_requests: 128,
            max_active_tokens: 1_048_576,
            max_queued_requests: 1_024,
            default_output_tokens: 256,
            classes: vec![ServiceClassPolicy {
                id: "standard".to_owned(),
                weight: 1,
                max_active_requests: 128,
                max_active_tokens: 1_048_576,
                shed_at_global_utilization_bps: None,
            }],
            tenants: vec![TenantPolicy {
                id: "default".to_owned(),
                service_class: "standard".to_owned(),
                weight: 1,
                max_active_requests: 128,
                max_active_tokens: 1_048_576,
                max_queued_requests: 1_024,
                max_tokens_per_request: 65_536,
                default_request_timeout: Duration::from_secs(120),
                max_request_timeout: Duration::from_secs(600),
            }],
        }
    }
}

impl TrafficPolicy {
    pub fn validate(&self) -> Result<(), TrafficConfigurationError> {
        if self.max_active_requests == 0
            || self.max_active_tokens == 0
            || self.max_queued_requests == 0
            || self.default_output_tokens == 0
        {
            return Err(TrafficConfigurationError::Invalid(
                "global traffic limits must be greater than zero".to_owned(),
            ));
        }
        if self.classes.is_empty() || self.tenants.is_empty() {
            return Err(TrafficConfigurationError::Invalid(
                "at least one service class and tenant must be configured".to_owned(),
            ));
        }
        if self.classes.len() > MAX_SERVICE_CLASSES || self.tenants.len() > MAX_TENANTS {
            return Err(TrafficConfigurationError::Invalid(format!(
                "traffic policy exceeds cardinality limits of {MAX_SERVICE_CLASSES} classes and {MAX_TENANTS} tenants"
            )));
        }
        let mut classes = BTreeMap::new();
        for class in &self.classes {
            validate_label("service class", &class.id)?;
            if class.weight == 0
                || class.max_active_requests == 0
                || class.max_active_tokens == 0
                || class
                    .shed_at_global_utilization_bps
                    .is_some_and(|threshold| threshold > 10_000)
            {
                return Err(TrafficConfigurationError::Invalid(format!(
                    "service class {} has invalid limits, weight, or shed threshold",
                    class.id
                )));
            }
            if classes.insert(class.id.as_str(), ()).is_some() {
                return Err(TrafficConfigurationError::Invalid(format!(
                    "duplicate service class: {}",
                    class.id
                )));
            }
        }
        let mut tenants = BTreeMap::new();
        for tenant in &self.tenants {
            validate_label("tenant", &tenant.id)?;
            if !classes.contains_key(tenant.service_class.as_str()) {
                return Err(TrafficConfigurationError::Invalid(format!(
                    "tenant {} references unknown service class {}",
                    tenant.id, tenant.service_class
                )));
            }
            if tenant.weight == 0
                || tenant.max_active_requests == 0
                || tenant.max_active_tokens == 0
                || tenant.max_queued_requests == 0
                || tenant.max_tokens_per_request == 0
                || tenant.default_request_timeout.is_zero()
                || tenant.max_request_timeout < tenant.default_request_timeout
            {
                return Err(TrafficConfigurationError::Invalid(format!(
                    "tenant {} has invalid limits, weight, or deadlines",
                    tenant.id
                )));
            }
            if tenants.insert(tenant.id.as_str(), ()).is_some() {
                return Err(TrafficConfigurationError::Invalid(format!(
                    "duplicate tenant: {}",
                    tenant.id
                )));
            }
        }
        Ok(())
    }
}

fn validate_label(kind: &str, value: &str) -> Result<(), TrafficConfigurationError> {
    let valid = !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
    if valid {
        Ok(())
    } else {
        Err(TrafficConfigurationError::Invalid(format!(
            "{kind} must be 1-64 ASCII letters, digits, '.', '_' or '-'"
        )))
    }
}

#[derive(Clone)]
pub struct TrafficController {
    inner: Arc<TrafficControllerInner>,
}

struct TrafficControllerInner {
    policy: TrafficPolicy,
    classes: BTreeMap<String, ServiceClassPolicy>,
    tenants: BTreeMap<String, TenantPolicy>,
    state: Mutex<TrafficState>,
    idle: Notify,
}

#[derive(Default)]
struct TrafficState {
    next_ticket: u64,
    draining: bool,
    waiting: BTreeMap<String, BTreeMap<String, VecDeque<QueuedRequest>>>,
    active: BTreeMap<u64, ActiveRequest>,
    class_active_requests: BTreeMap<String, usize>,
    class_active_tokens: BTreeMap<String, u64>,
    tenant_active_requests: BTreeMap<String, usize>,
    tenant_active_tokens: BTreeMap<String, u64>,
    tenant_queued_requests: BTreeMap<String, usize>,
    tenant_queued_tokens: BTreeMap<String, u64>,
    class_vruntime: BTreeMap<String, u128>,
    tenant_vruntime: BTreeMap<String, u128>,
    admissions: BTreeMap<(String, String, &'static str), u64>,
    rejections: BTreeMap<(String, String, &'static str), u64>,
    terminations: BTreeMap<(String, String, &'static str), u64>,
    queue_wait_micros: BTreeMap<(String, String), u128>,
    queue_wait_count: BTreeMap<(String, String), u64>,
    forced_cancellations: u64,
}

struct QueuedRequest {
    ticket: u64,
    token_cost: u64,
    enqueued_at: Instant,
    cancellation: CancellationToken,
    sender: oneshot::Sender<Result<AdmissionPermit, AdmissionError>>,
}

struct ActiveRequest {
    tenant: String,
    class: String,
    token_cost: u64,
    cancellation: CancellationToken,
}

impl TrafficController {
    pub fn new(policy: TrafficPolicy) -> Result<Self, TrafficConfigurationError> {
        policy.validate()?;
        let classes = policy
            .classes
            .iter()
            .cloned()
            .map(|class| (class.id.clone(), class))
            .collect();
        let tenants = policy
            .tenants
            .iter()
            .cloned()
            .map(|tenant| (tenant.id.clone(), tenant))
            .collect();
        Ok(Self {
            inner: Arc::new(TrafficControllerInner {
                policy,
                classes,
                tenants,
                state: Mutex::new(TrafficState::default()),
                idle: Notify::new(),
            }),
        })
    }

    #[must_use]
    pub fn tenant_exists(&self, tenant: &str) -> bool {
        self.inner.tenants.contains_key(tenant)
    }

    #[must_use]
    pub fn default_output_tokens(&self) -> u64 {
        self.inner.policy.default_output_tokens
    }

    pub fn operation_context(
        &self,
        request_id: RequestId,
        tenant: &str,
        requested_timeout: Option<Duration>,
    ) -> Result<OperationContext, AdmissionError> {
        let policy = self
            .inner
            .tenants
            .get(tenant)
            .ok_or_else(|| AdmissionError::UnknownTenant(tenant.to_owned()))?;
        let timeout = requested_timeout
            .unwrap_or(policy.default_request_timeout)
            .min(policy.max_request_timeout);
        if timeout.is_zero() {
            return Err(AdmissionError::InvalidDeadline);
        }
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(AdmissionError::InvalidDeadline)?;
        Ok(OperationContext::new(request_id)
            .with_tenant_id(tenant)
            .with_deadline(deadline))
    }

    pub async fn admit(
        &self,
        token_cost: u64,
        context: &OperationContext,
    ) -> Result<AdmissionPermit, AdmissionError> {
        context.ensure_active().map_err(AdmissionError::from)?;
        let tenant_id = context
            .tenant_id
            .as_deref()
            .ok_or(AdmissionError::MissingTrustedTenant)?;
        let tenant = self
            .inner
            .tenants
            .get(tenant_id)
            .ok_or_else(|| AdmissionError::UnknownTenant(tenant_id.to_owned()))?;
        let class = self
            .inner
            .classes
            .get(&tenant.service_class)
            .expect("validated tenant class");
        if token_cost == 0 || token_cost > tenant.max_tokens_per_request {
            self.record_rejection(tenant, "request_token_limit");
            return Err(AdmissionError::RequestTokenLimit {
                requested: token_cost,
                limit: tenant.max_tokens_per_request,
            });
        }
        if token_cost > self.inner.policy.max_active_tokens
            || token_cost > class.max_active_tokens
            || token_cost > tenant.max_active_tokens
        {
            self.record_rejection(tenant, "active_token_capacity");
            return Err(AdmissionError::RequestExceedsCapacity);
        }

        let (ticket, mut receiver) = {
            let mut state = self.lock_state()?;
            if state.draining {
                record_rejection_locked(&mut state, tenant, "draining");
                return Err(AdmissionError::Draining);
            }
            let utilization_bps = active_token_utilization_bps(&state, &self.inner.policy);
            if class
                .shed_at_global_utilization_bps
                .is_some_and(|threshold| utilization_bps >= u64::from(threshold))
            {
                record_rejection_locked(&mut state, tenant, "overload_shed");
                return Err(AdmissionError::OverloadShed);
            }
            let queued = queued_requests(&state);
            let tenant_queued = count(&state.tenant_queued_requests, &tenant.id);
            if queued >= self.inner.policy.max_queued_requests
                || tenant_queued >= tenant.max_queued_requests
            {
                record_rejection_locked(&mut state, tenant, "queue_full");
                return Err(AdmissionError::QueueFull);
            }
            state.next_ticket = state.next_ticket.saturating_add(1);
            let ticket = state.next_ticket;
            let (sender, receiver) = oneshot::channel();
            self.normalize_idle_vruntime_locked(&mut state, tenant);
            state
                .waiting
                .entry(tenant.service_class.clone())
                .or_default()
                .entry(tenant.id.clone())
                .or_default()
                .push_back(QueuedRequest {
                    ticket,
                    token_cost,
                    enqueued_at: Instant::now(),
                    cancellation: context.cancellation.clone(),
                    sender,
                });
            increment(&mut state.tenant_queued_requests, &tenant.id, 1);
            increment(&mut state.tenant_queued_tokens, &tenant.id, token_cost);
            *state
                .admissions
                .entry((tenant.service_class.clone(), tenant.id.clone(), "queued"))
                .or_default() += 1;
            self.schedule_locked(&mut state);
            (ticket, receiver)
        };

        let deadline = context.deadline;
        tokio::select! {
            biased;
            () = context.cancellation.cancelled() => {
                self.cancel_waiter(ticket, tenant, "cancelled")?;
                Err(AdmissionError::Cancelled)
            }
            () = wait_for_deadline(deadline) => {
                self.cancel_waiter(ticket, tenant, "deadline")?;
                Err(AdmissionError::DeadlineExceeded)
            }
            result = &mut receiver => {
                result.map_err(|_| AdmissionError::Unavailable)?
            }
        }
    }

    pub fn begin_drain(&self) -> Result<(), AdmissionError> {
        let mut state = self.lock_state()?;
        if state.draining {
            return Ok(());
        }
        state.draining = true;
        let waiting = std::mem::take(&mut state.waiting);
        for tenants in waiting.into_values() {
            for (tenant_id, queue) in tenants {
                let Some(tenant) = self.inner.tenants.get(&tenant_id) else {
                    continue;
                };
                for queued in queue {
                    decrement(&mut state.tenant_queued_requests, &tenant_id, 1);
                    decrement(
                        &mut state.tenant_queued_tokens,
                        &tenant_id,
                        queued.token_cost,
                    );
                    record_rejection_locked(&mut state, tenant, "draining");
                    let _ = queued.sender.send(Err(AdmissionError::Draining));
                }
            }
        }
        Ok(())
    }

    pub fn resume(&self) -> Result<(), AdmissionError> {
        self.lock_state()?.draining = false;
        Ok(())
    }

    #[must_use]
    pub fn is_draining(&self) -> bool {
        self.inner.state.lock().map_or(true, |state| state.draining)
    }

    pub async fn drain(&self, grace: Duration) -> Result<DrainReport, AdmissionError> {
        self.begin_drain()?;
        let deadline = tokio::time::Instant::now()
            .checked_add(grace)
            .ok_or(AdmissionError::Unavailable)?;
        loop {
            let notified = self.inner.idle.notified();
            if self.active_requests()? == 0 {
                return Ok(DrainReport {
                    completed: true,
                    forced_cancellations: 0,
                });
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                break;
            }
        }
        let cancellations = {
            let mut state = self.lock_state()?;
            let cancellations = state
                .active
                .values()
                .map(|active| active.cancellation.clone())
                .collect::<Vec<_>>();
            state.forced_cancellations = state
                .forced_cancellations
                .saturating_add(cancellations.len() as u64);
            cancellations
        };
        for cancellation in &cancellations {
            cancellation.cancel();
        }
        Ok(DrainReport {
            completed: false,
            forced_cancellations: cancellations.len(),
        })
    }

    pub(crate) fn record_termination(&self, context: &OperationContext, reason: &'static str) {
        let Some(tenant_id) = context.tenant_id.as_deref() else {
            return;
        };
        let Some(tenant) = self.inner.tenants.get(tenant_id) else {
            return;
        };
        if let Ok(mut state) = self.inner.state.lock() {
            *state
                .terminations
                .entry((tenant.service_class.clone(), tenant.id.clone(), reason))
                .or_default() += 1;
        }
    }

    pub fn prometheus(&self) -> Result<String, AdmissionError> {
        let state = self.lock_state()?;
        let mut output = String::new();
        output.push_str("# HELP locus_admission_requests_total Admission decisions by configured class and tenant.\n");
        output.push_str("# TYPE locus_admission_requests_total counter\n");
        for ((class, tenant, outcome), value) in &state.admissions {
            metric_line(
                &mut output,
                "locus_admission_requests_total",
                &[("class", class), ("tenant", tenant), ("outcome", outcome)],
                *value,
            );
        }
        output.push_str(
            "# HELP locus_admission_rejections_total Rejected requests by bounded reason.\n",
        );
        output.push_str("# TYPE locus_admission_rejections_total counter\n");
        for ((class, tenant, reason), value) in &state.rejections {
            metric_line(
                &mut output,
                "locus_admission_rejections_total",
                &[("class", class), ("tenant", tenant), ("reason", reason)],
                *value,
            );
        }
        output.push_str(
            "# HELP locus_request_terminations_total Request terminations by bounded reason.\n",
        );
        output.push_str("# TYPE locus_request_terminations_total counter\n");
        for ((class, tenant, reason), value) in &state.terminations {
            metric_line(
                &mut output,
                "locus_request_terminations_total",
                &[("class", class), ("tenant", tenant), ("reason", reason)],
                *value,
            );
        }
        output.push_str("# HELP locus_admission_active_requests Active admitted requests.\n");
        output.push_str("# TYPE locus_admission_active_requests gauge\n");
        output
            .push_str("# HELP locus_admission_active_tokens Reserved prompt and output tokens.\n");
        output.push_str("# TYPE locus_admission_active_tokens gauge\n");
        output.push_str("# HELP locus_admission_queued_requests Requests waiting for admission.\n");
        output.push_str("# TYPE locus_admission_queued_requests gauge\n");
        output.push_str(
            "# HELP locus_admission_queued_tokens Tokens represented by queued requests.\n",
        );
        output.push_str("# TYPE locus_admission_queued_tokens gauge\n");
        for tenant in self.inner.tenants.values() {
            let labels = [
                ("class", tenant.service_class.as_str()),
                ("tenant", tenant.id.as_str()),
            ];
            metric_line(
                &mut output,
                "locus_admission_active_requests",
                &labels,
                count(&state.tenant_active_requests, &tenant.id),
            );
            metric_line(
                &mut output,
                "locus_admission_active_tokens",
                &labels,
                count(&state.tenant_active_tokens, &tenant.id),
            );
            metric_line(
                &mut output,
                "locus_admission_queued_requests",
                &labels,
                count(&state.tenant_queued_requests, &tenant.id),
            );
            metric_line(
                &mut output,
                "locus_admission_queued_tokens",
                &labels,
                count(&state.tenant_queued_tokens, &tenant.id),
            );
        }
        output.push_str(
            "# HELP locus_admission_queue_wait_seconds Total time spent waiting for admission.\n",
        );
        output.push_str("# TYPE locus_admission_queue_wait_seconds summary\n");
        for ((class, tenant), micros) in &state.queue_wait_micros {
            let labels = [("class", class.as_str()), ("tenant", tenant.as_str())];
            metric_float_line(
                &mut output,
                "locus_admission_queue_wait_seconds_sum",
                &labels,
                *micros as f64 / 1_000_000.0,
            );
            metric_line(
                &mut output,
                "locus_admission_queue_wait_seconds_count",
                &labels,
                count(&state.queue_wait_count, &(class.clone(), tenant.clone())),
            );
        }
        output.push_str("# HELP locus_traffic_controller_draining Whether new inference admission is disabled.\n");
        output.push_str("# TYPE locus_traffic_controller_draining gauge\n");
        let _ = writeln!(
            output,
            "locus_traffic_controller_draining {}",
            u8::from(state.draining)
        );
        output.push_str("# HELP locus_traffic_forced_cancellations_total Requests cancelled after drain grace expired.\n");
        output.push_str("# TYPE locus_traffic_forced_cancellations_total counter\n");
        let _ = writeln!(
            output,
            "locus_traffic_forced_cancellations_total {}",
            state.forced_cancellations
        );
        Ok(output)
    }

    fn active_requests(&self) -> Result<usize, AdmissionError> {
        Ok(self.lock_state()?.active.len())
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, TrafficState>, AdmissionError> {
        self.inner
            .state
            .lock()
            .map_err(|_| AdmissionError::Unavailable)
    }

    fn record_rejection(&self, tenant: &TenantPolicy, reason: &'static str) {
        if let Ok(mut state) = self.inner.state.lock() {
            record_rejection_locked(&mut state, tenant, reason);
        }
    }

    fn cancel_waiter(
        &self,
        ticket: u64,
        tenant: &TenantPolicy,
        reason: &'static str,
    ) -> Result<(), AdmissionError> {
        let mut state = self.lock_state()?;
        let Some(classes) = state.waiting.get_mut(&tenant.service_class) else {
            return Ok(());
        };
        let Some(queue) = classes.get_mut(&tenant.id) else {
            return Ok(());
        };
        let Some(position) = queue.iter().position(|request| request.ticket == ticket) else {
            return Ok(());
        };
        let queued = queue.remove(position).expect("queued request position");
        decrement(&mut state.tenant_queued_requests, &tenant.id, 1);
        decrement(
            &mut state.tenant_queued_tokens,
            &tenant.id,
            queued.token_cost,
        );
        record_rejection_locked(&mut state, tenant, reason);
        self.schedule_locked(&mut state);
        Ok(())
    }

    fn schedule_locked(&self, state: &mut TrafficState) {
        while state.active.len() < self.inner.policy.max_active_requests {
            let Some((class_id, tenant_id)) = self.choose_candidate(state) else {
                break;
            };
            let queued = state
                .waiting
                .get_mut(&class_id)
                .and_then(|tenants| tenants.get_mut(&tenant_id))
                .and_then(VecDeque::pop_front)
                .expect("chosen admission candidate");
            let class = self.inner.classes.get(&class_id).expect("configured class");
            let tenant = self
                .inner
                .tenants
                .get(&tenant_id)
                .expect("configured tenant");
            decrement(&mut state.tenant_queued_requests, &tenant_id, 1);
            decrement(
                &mut state.tenant_queued_tokens,
                &tenant_id,
                queued.token_cost,
            );
            increment(&mut state.class_active_requests, &class_id, 1);
            increment(&mut state.class_active_tokens, &class_id, queued.token_cost);
            increment(&mut state.tenant_active_requests, &tenant_id, 1);
            increment(
                &mut state.tenant_active_tokens,
                &tenant_id,
                queued.token_cost,
            );
            state.active.insert(
                queued.ticket,
                ActiveRequest {
                    tenant: tenant_id.clone(),
                    class: class_id.clone(),
                    token_cost: queued.token_cost,
                    cancellation: queued.cancellation,
                },
            );
            let class_vruntime = state.class_vruntime.entry(class_id.clone()).or_default();
            *class_vruntime = class_vruntime
                .saturating_add(virtual_runtime_delta(queued.token_cost, class.weight));
            let tenant_vruntime = state.tenant_vruntime.entry(tenant_id.clone()).or_default();
            *tenant_vruntime = tenant_vruntime
                .saturating_add(virtual_runtime_delta(queued.token_cost, tenant.weight));
            *state
                .admissions
                .entry((class_id.clone(), tenant_id.clone(), "admitted"))
                .or_default() += 1;
            let wait_micros = queued.enqueued_at.elapsed().as_micros();
            *state
                .queue_wait_micros
                .entry((class_id.clone(), tenant_id.clone()))
                .or_default() += wait_micros;
            *state
                .queue_wait_count
                .entry((class_id.clone(), tenant_id.clone()))
                .or_default() += 1;
            let permit = AdmissionPermit {
                controller: self.clone(),
                ticket: queued.ticket,
                released: false,
            };
            if let Err(Ok(mut permit)) = queued.sender.send(Ok(permit)) {
                permit.release_locked(state);
            }
        }
    }

    fn normalize_idle_vruntime_locked(&self, state: &mut TrafficState, tenant: &TenantPolicy) {
        let class_is_idle = count(&state.class_active_requests, &tenant.service_class) == 0
            && state
                .waiting
                .get(&tenant.service_class)
                .is_none_or(|queues| queues.values().all(VecDeque::is_empty));
        if class_is_idle {
            let floor =
                self.inner
                    .classes
                    .keys()
                    .filter(|class_id| class_id.as_str() != tenant.service_class)
                    .filter(|class_id| {
                        count(&state.class_active_requests, class_id) > 0
                            || state.waiting.get(*class_id).is_some_and(|queues| {
                                queues.values().any(|queue| !queue.is_empty())
                            })
                    })
                    .map(|class_id| count(&state.class_vruntime, class_id))
                    .min()
                    .unwrap_or_default();
            let current = state
                .class_vruntime
                .entry(tenant.service_class.clone())
                .or_default();
            *current = (*current).max(floor);
        }
        let tenant_is_idle = count(&state.tenant_active_requests, &tenant.id) == 0
            && state
                .waiting
                .get(&tenant.service_class)
                .and_then(|queues| queues.get(&tenant.id))
                .is_none_or(VecDeque::is_empty);
        if tenant_is_idle {
            let floor = self
                .inner
                .tenants
                .values()
                .filter(|candidate| {
                    candidate.service_class == tenant.service_class && candidate.id != tenant.id
                })
                .filter(|candidate| {
                    count(&state.tenant_active_requests, &candidate.id) > 0
                        || state
                            .waiting
                            .get(&candidate.service_class)
                            .and_then(|queues| queues.get(&candidate.id))
                            .is_some_and(|queue| !queue.is_empty())
                })
                .map(|candidate| count(&state.tenant_vruntime, &candidate.id))
                .min()
                .unwrap_or_default();
            let current = state.tenant_vruntime.entry(tenant.id.clone()).or_default();
            *current = (*current).max(floor);
        }
    }

    fn choose_candidate(&self, state: &TrafficState) -> Option<(String, String)> {
        let active_tokens = state
            .active
            .values()
            .map(|active| active.token_cost)
            .sum::<u64>();
        let mut class_candidates = Vec::new();
        for (class_id, tenants) in &state.waiting {
            let class = self.inner.classes.get(class_id)?;
            if count(&state.class_active_requests, class_id) >= class.max_active_requests {
                continue;
            }
            let mut tenant_candidates = Vec::new();
            for (tenant_id, queue) in tenants {
                let Some(head) = queue.front() else {
                    continue;
                };
                let tenant = self.inner.tenants.get(tenant_id)?;
                let fits = active_tokens.saturating_add(head.token_cost)
                    <= self.inner.policy.max_active_tokens
                    && count(&state.class_active_tokens, class_id).saturating_add(head.token_cost)
                        <= class.max_active_tokens
                    && count(&state.tenant_active_requests, tenant_id) < tenant.max_active_requests
                    && count(&state.tenant_active_tokens, tenant_id)
                        .saturating_add(head.token_cost)
                        <= tenant.max_active_tokens;
                if fits {
                    tenant_candidates
                        .push((count(&state.tenant_vruntime, tenant_id), tenant_id.clone()));
                }
            }
            tenant_candidates.sort();
            if let Some((_, tenant_id)) = tenant_candidates.into_iter().next() {
                class_candidates.push((
                    count(&state.class_vruntime, class_id),
                    class_id.clone(),
                    tenant_id,
                ));
            }
        }
        class_candidates.sort();
        class_candidates
            .into_iter()
            .next()
            .map(|(_, class, tenant)| (class, tenant))
    }

    fn release(&self, ticket: u64) {
        let Ok(mut state) = self.inner.state.lock() else {
            return;
        };
        if release_active_locked(&mut state, ticket).is_some() {
            self.schedule_locked(&mut state);
            if state.active.is_empty() {
                self.inner.idle.notify_waiters();
            }
        }
    }
}

impl Default for TrafficController {
    fn default() -> Self {
        Self::new(TrafficPolicy::default()).expect("default traffic policy")
    }
}

pub struct AdmissionPermit {
    controller: TrafficController,
    ticket: u64,
    released: bool,
}

impl AdmissionPermit {
    fn release_locked(&mut self, state: &mut TrafficState) {
        if !self.released {
            let _ = release_active_locked(state, self.ticket);
            self.released = true;
        }
    }
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        if !self.released {
            self.released = true;
            self.controller.release(self.ticket);
        }
    }
}

fn release_active_locked(state: &mut TrafficState, ticket: u64) -> Option<ActiveRequest> {
    let active = state.active.remove(&ticket)?;
    decrement(&mut state.class_active_requests, &active.class, 1);
    decrement(
        &mut state.class_active_tokens,
        &active.class,
        active.token_cost,
    );
    decrement(&mut state.tenant_active_requests, &active.tenant, 1);
    decrement(
        &mut state.tenant_active_tokens,
        &active.tenant,
        active.token_cost,
    );
    Some(active)
}

fn record_rejection_locked(state: &mut TrafficState, tenant: &TenantPolicy, reason: &'static str) {
    *state
        .rejections
        .entry((tenant.service_class.clone(), tenant.id.clone(), reason))
        .or_default() += 1;
}

fn active_token_utilization_bps(state: &TrafficState, policy: &TrafficPolicy) -> u64 {
    state
        .active
        .values()
        .map(|active| active.token_cost)
        .sum::<u64>()
        .saturating_mul(10_000)
        / policy.max_active_tokens
}

fn queued_requests(state: &TrafficState) -> usize {
    state
        .tenant_queued_requests
        .values()
        .copied()
        .sum::<usize>()
}

fn virtual_runtime_delta(tokens: u64, weight: u32) -> u128 {
    (u128::from(tokens)
        .saturating_mul(VIRTUAL_RUNTIME_SCALE)
        .checked_div(u128::from(weight))
        .unwrap_or(u128::MAX))
    .max(1)
}

fn count<K, V>(map: &BTreeMap<K, V>, key: &K) -> V
where
    K: Ord,
    V: Copy + Default,
{
    map.get(key).copied().unwrap_or_default()
}

fn increment<K, V>(map: &mut BTreeMap<K, V>, key: &K, value: V)
where
    K: Ord + Clone,
    V: Copy + Default + std::ops::AddAssign,
{
    *map.entry(key.clone()).or_default() += value;
}

fn decrement<K, V>(map: &mut BTreeMap<K, V>, key: &K, value: V)
where
    K: Ord + Clone,
    V: Copy + Default + std::ops::SubAssign,
{
    if let Some(current) = map.get_mut(key) {
        *current -= value;
    }
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await,
        None => std::future::pending().await,
    }
}

fn metric_line<V: std::fmt::Display>(
    output: &mut String,
    name: &str,
    labels: &[(&str, &str)],
    value: V,
) {
    metric_float_line(output, name, labels, value);
}

fn metric_float_line<V: std::fmt::Display>(
    output: &mut String,
    name: &str,
    labels: &[(&str, &str)],
    value: V,
) {
    let _ = write!(output, "{name}{{");
    for (index, (key, value)) in labels.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(output, "{key}=\"{}\"", escape_label(value));
    }
    let _ = writeln!(output, "}} {value}");
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrainReport {
    pub completed: bool,
    pub forced_cancellations: usize,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TrafficConfigurationError {
    #[error("invalid traffic policy: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AdmissionError {
    #[error("request has no trusted tenant identity")]
    MissingTrustedTenant,
    #[error("tenant policy was not found: {0}")]
    UnknownTenant(String),
    #[error("request deadline must be greater than zero")]
    InvalidDeadline,
    #[error("request token cost {requested} exceeds tenant limit {limit}")]
    RequestTokenLimit { requested: u64, limit: u64 },
    #[error("request token cost exceeds active admission capacity")]
    RequestExceedsCapacity,
    #[error("admission queue is full")]
    QueueFull,
    #[error("request was shed by overload policy")]
    OverloadShed,
    #[error("traffic controller is draining")]
    Draining,
    #[error("request was cancelled while waiting for admission")]
    Cancelled,
    #[error("request deadline expired while waiting for admission")]
    DeadlineExceeded,
    #[error("request admission is unavailable")]
    Unavailable,
}

impl From<locus_core::ContextError> for AdmissionError {
    fn from(error: locus_core::ContextError) -> Self {
        match error {
            locus_core::ContextError::Cancelled => Self::Cancelled,
            locus_core::ContextError::DeadlineExceeded => Self::DeadlineExceeded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn policy() -> TrafficPolicy {
        TrafficPolicy {
            max_active_requests: 1,
            max_active_tokens: 100,
            max_queued_requests: 8,
            default_output_tokens: 10,
            classes: vec![ServiceClassPolicy {
                id: "batch".to_owned(),
                weight: 1,
                max_active_requests: 1,
                max_active_tokens: 100,
                shed_at_global_utilization_bps: None,
            }],
            tenants: vec![
                TenantPolicy {
                    id: "a".to_owned(),
                    service_class: "batch".to_owned(),
                    weight: 1,
                    max_active_requests: 1,
                    max_active_tokens: 100,
                    max_queued_requests: 4,
                    max_tokens_per_request: 100,
                    default_request_timeout: Duration::from_secs(1),
                    max_request_timeout: Duration::from_secs(1),
                },
                TenantPolicy {
                    id: "b".to_owned(),
                    service_class: "batch".to_owned(),
                    weight: 2,
                    max_active_requests: 1,
                    max_active_tokens: 100,
                    max_queued_requests: 4,
                    max_tokens_per_request: 100,
                    default_request_timeout: Duration::from_secs(1),
                    max_request_timeout: Duration::from_secs(1),
                },
            ],
        }
    }

    fn context(controller: &TrafficController, tenant: &str, id: &str) -> OperationContext {
        controller
            .operation_context(RequestId::new(id), tenant, None)
            .expect("context")
    }

    async fn wait_for_queued(controller: &TrafficController, expected: usize) {
        for _ in 0..100 {
            let queued = controller
                .lock_state()
                .map(|state| queued_requests(&state))
                .expect("traffic state");
            if queued == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("timed out waiting for {expected} queued requests");
    }

    #[tokio::test]
    async fn queued_permit_is_released_when_waiter_is_cancelled() {
        let controller = TrafficController::new(policy()).expect("controller");
        let first = controller
            .admit(100, &context(&controller, "a", "first"))
            .await
            .expect("first admission");
        let waiting_context = context(&controller, "b", "waiting");
        let cancellation = waiting_context.cancellation.clone();
        let waiting_controller = controller.clone();
        let waiting =
            tokio::spawn(async move { waiting_controller.admit(10, &waiting_context).await });
        wait_for_queued(&controller, 1).await;
        cancellation.cancel();
        assert!(matches!(
            waiting.await.expect("join"),
            Err(AdmissionError::Cancelled)
        ));
        drop(first);
        assert!(controller.prometheus().expect("metrics").contains(
            "locus_admission_rejections_total{class=\"batch\",tenant=\"b\",reason=\"cancelled\"} 1"
        ));
    }

    #[tokio::test]
    async fn drain_rejects_waiters_and_cancels_after_grace() {
        let controller = TrafficController::new(policy()).expect("controller");
        let active_context = context(&controller, "a", "active");
        let cancellation = active_context.cancellation.clone();
        let _permit = controller
            .admit(100, &active_context)
            .await
            .expect("active admission");
        let report = controller
            .drain(Duration::from_millis(1))
            .await
            .expect("drain");
        assert!(!report.completed);
        assert_eq!(report.forced_cancellations, 1);
        assert!(cancellation.is_cancelled());
        assert!(matches!(
            controller
                .admit(1, &context(&controller, "b", "rejected"))
                .await,
            Err(AdmissionError::Draining)
        ));
    }

    #[tokio::test]
    async fn token_weighted_hierarchy_is_deterministic_across_service_classes() {
        let mut policy = policy();
        policy.classes[0].id = "slow".to_owned();
        policy.tenants[0].service_class = "slow".to_owned();
        policy.tenants[0].weight = 1;
        policy.classes.push(ServiceClassPolicy {
            id: "fast".to_owned(),
            weight: 2,
            max_active_requests: 1,
            max_active_tokens: 100,
            shed_at_global_utilization_bps: None,
        });
        policy.tenants[1].service_class = "fast".to_owned();
        policy.tenants[1].weight = 2;
        let controller = TrafficController::new(policy).expect("controller");
        let first = controller
            .admit(1, &context(&controller, "a", "first"))
            .await
            .expect("first admission");
        let (sender, mut receiver) = mpsc::unbounded_channel();
        for (index, (label, tenant)) in [
            ("a1", "a"),
            ("a2", "a"),
            ("b1", "b"),
            ("b2", "b"),
            ("b3", "b"),
            ("b4", "b"),
        ]
        .into_iter()
        .enumerate()
        {
            let spawned_controller = controller.clone();
            let sender = sender.clone();
            let context = context(&controller, tenant, label);
            tokio::spawn(async move {
                let permit = spawned_controller
                    .admit(20, &context)
                    .await
                    .expect("admission");
                sender.send((label, permit)).expect("admission order");
            });
            wait_for_queued(&controller, index + 1).await;
        }
        drop(sender);
        drop(first);

        let mut order = Vec::new();
        while let Some((label, permit)) = receiver.recv().await {
            order.push(label);
            drop(permit);
        }
        assert_eq!(order, ["b1", "a1", "b2", "b3", "a2", "b4"]);
    }

    #[tokio::test]
    async fn token_weighted_tenants_share_one_class_by_configured_weight() {
        let controller = TrafficController::new(policy()).expect("controller");
        let first = controller
            .admit(1, &context(&controller, "a", "first"))
            .await
            .expect("first admission");
        let (sender, mut receiver) = mpsc::unbounded_channel();
        for (index, (label, tenant)) in [
            ("a1", "a"),
            ("a2", "a"),
            ("b1", "b"),
            ("b2", "b"),
            ("b3", "b"),
            ("b4", "b"),
        ]
        .into_iter()
        .enumerate()
        {
            let spawned_controller = controller.clone();
            let sender = sender.clone();
            let context = context(&controller, tenant, label);
            tokio::spawn(async move {
                let permit = spawned_controller
                    .admit(20, &context)
                    .await
                    .expect("admission");
                sender.send((label, permit)).expect("admission order");
            });
            wait_for_queued(&controller, index + 1).await;
        }
        drop(sender);
        drop(first);

        let mut order = Vec::new();
        while let Some((label, permit)) = receiver.recv().await {
            order.push(label);
            drop(permit);
        }
        assert_eq!(order, ["a1", "b1", "b2", "a2", "b3", "b4"]);
    }

    #[tokio::test]
    async fn overload_shedding_is_class_policy_and_metrics_stay_bounded() {
        let mut policy = policy();
        policy.classes[0].shed_at_global_utilization_bps = Some(5_000);
        let controller = TrafficController::new(policy).expect("controller");
        let active = controller
            .admit(60, &context(&controller, "a", "active-secret-id"))
            .await
            .expect("active admission");
        assert!(matches!(
            controller
                .admit(10, &context(&controller, "b", "rejected-secret-id"))
                .await,
            Err(AdmissionError::OverloadShed)
        ));
        let metrics = controller.prometheus().expect("metrics");
        assert!(metrics.contains("reason=\"overload_shed\""));
        assert!(!metrics.contains("active-secret-id"));
        assert!(!metrics.contains("rejected-secret-id"));
        drop(active);
    }
}
