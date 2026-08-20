use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use thiserror::Error;
use tokio::sync::Notify;

use crate::RequestId;

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            self.state.notify.notify_waiters();
        }
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Resolves when cancellation is observed. The double-check around
    /// `notified` prevents a notification between registration and polling
    /// from being lost.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.state.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Debug)]
pub struct OperationContext {
    pub request_id: RequestId,
    /// Authenticated tenant identity. Callers must only populate this from a
    /// trusted credential-to-policy mapping, never from request JSON or an
    /// unverified tenant header.
    pub tenant_id: Option<String>,
    pub deadline: Option<Instant>,
    pub cancellation: CancellationToken,
}

impl OperationContext {
    #[must_use]
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            tenant_id: None,
            deadline: None,
            cancellation: CancellationToken::default(),
        }
    }

    #[must_use]
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    #[must_use]
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Runs one downstream operation under the request cancellation and
    /// deadline. Dropping the losing future also cancels in-flight transports
    /// such as reqwest requests.
    pub async fn run<F, T>(&self, future: F) -> Result<T, ContextError>
    where
        F: Future<Output = T>,
    {
        self.ensure_active()?;
        match self.deadline {
            Some(deadline) => {
                tokio::select! {
                    biased;
                    () = self.cancellation.cancelled() => Err(ContextError::Cancelled),
                    () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                        Err(ContextError::DeadlineExceeded)
                    }
                    output = future => Ok(output),
                }
            }
            None => {
                tokio::select! {
                    biased;
                    () = self.cancellation.cancelled() => Err(ContextError::Cancelled),
                    output = future => Ok(output),
                }
            }
        }
    }

    pub fn ensure_active(&self) -> Result<(), ContextError> {
        if self.cancellation.is_cancelled() {
            return Err(ContextError::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(ContextError::DeadlineExceeded);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ContextError {
    #[error("operation was cancelled")]
    Cancelled,
    #[error("operation deadline exceeded")]
    DeadlineExceeded,
}
