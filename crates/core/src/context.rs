use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use thiserror::Error;

use crate::RequestId;

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug)]
pub struct OperationContext {
    pub request_id: RequestId,
    pub deadline: Option<Instant>,
    pub cancellation: CancellationToken,
}

impl OperationContext {
    #[must_use]
    pub fn new(request_id: RequestId) -> Self {
        Self {
            request_id,
            deadline: None,
            cancellation: CancellationToken::default(),
        }
    }

    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
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
