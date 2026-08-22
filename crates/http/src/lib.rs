use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

#[derive(Clone)]
pub struct TenantCredential {
    pub tenant_id: String,
    pub bearer_token: String,
}

#[derive(Clone)]
pub struct CredentialIndex {
    buckets: Arc<HashMap<[u8; 32], Vec<IndexedCredential>>>,
}

#[derive(Clone, Default)]
pub struct TransportMetrics {
    inner: Arc<TransportMetricState>,
}

#[derive(Default)]
struct TransportMetricState {
    accepted_connections: AtomicU64,
    active_connections: AtomicU64,
    accept_errors: AtomicU64,
    connection_errors: AtomicU64,
    forced_closes: AtomicU64,
}

impl TransportMetrics {
    #[must_use]
    pub fn connection_opened(&self) -> TransportConnection {
        self.inner
            .accepted_connections
            .fetch_add(1, Ordering::Relaxed);
        self.inner
            .active_connections
            .fetch_add(1, Ordering::Relaxed);
        TransportConnection {
            metrics: self.clone(),
        }
    }

    pub fn record_accept_error(&self) {
        self.inner.accept_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_connection_error(&self) {
        self.inner.connection_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_forced_closes(&self, count: usize) {
        self.inner
            .forced_closes
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    #[must_use]
    pub fn prometheus(&self) -> String {
        format!(
            concat!(
                "# HELP locus_http_connections_accepted_total Accepted HTTP connections.\n",
                "# TYPE locus_http_connections_accepted_total counter\n",
                "locus_http_connections_accepted_total {}\n",
                "# HELP locus_http_connections_active Currently active HTTP connections.\n",
                "# TYPE locus_http_connections_active gauge\n",
                "locus_http_connections_active {}\n",
                "# HELP locus_http_accept_errors_total Listener accept errors.\n",
                "# TYPE locus_http_accept_errors_total counter\n",
                "locus_http_accept_errors_total {}\n",
                "# HELP locus_http_connection_errors_total HTTP connection protocol errors.\n",
                "# TYPE locus_http_connection_errors_total counter\n",
                "locus_http_connection_errors_total {}\n",
                "# HELP locus_http_connections_forced_closed_total Connections aborted after graceful shutdown timeout.\n",
                "# TYPE locus_http_connections_forced_closed_total counter\n",
                "locus_http_connections_forced_closed_total {}\n",
            ),
            self.inner.accepted_connections.load(Ordering::Relaxed),
            self.inner.active_connections.load(Ordering::Relaxed),
            self.inner.accept_errors.load(Ordering::Relaxed),
            self.inner.connection_errors.load(Ordering::Relaxed),
            self.inner.forced_closes.load(Ordering::Relaxed),
        )
    }
}

pub struct TransportConnection {
    metrics: TransportMetrics,
}

impl Drop for TransportConnection {
    fn drop(&mut self) {
        self.metrics
            .inner
            .active_connections
            .fetch_sub(1, Ordering::Relaxed);
    }
}

struct IndexedCredential {
    tenant_id: Arc<str>,
    bearer_token: Box<[u8]>,
}

impl CredentialIndex {
    pub fn new(credentials: Vec<TenantCredential>) -> Result<Self, CredentialIndexError> {
        let mut buckets: HashMap<[u8; 32], Vec<IndexedCredential>> = HashMap::new();
        for credential in credentials {
            if credential.bearer_token.is_empty() {
                return Err(CredentialIndexError::EmptyCredential);
            }
            let digest = token_digest(credential.bearer_token.as_bytes());
            let bucket = buckets.entry(digest).or_default();
            if bucket.iter().any(|candidate| {
                candidate.bearer_token.len() == credential.bearer_token.len()
                    && bool::from(
                        candidate
                            .bearer_token
                            .as_ref()
                            .ct_eq(credential.bearer_token.as_bytes()),
                    )
            }) {
                return Err(CredentialIndexError::DuplicateCredential);
            }
            bucket.push(IndexedCredential {
                tenant_id: Arc::from(credential.tenant_id),
                bearer_token: credential.bearer_token.into_bytes().into_boxed_slice(),
            });
        }
        Ok(Self {
            buckets: Arc::new(buckets),
        })
    }

    /// Resolves a credential with one bounded digest lookup followed by a
    /// constant-time raw-token verification inside the collision bucket.
    #[must_use]
    pub fn authenticate(&self, candidate: &str) -> Option<Arc<str>> {
        let digest = token_digest(candidate.as_bytes());
        self.buckets.get(&digest).and_then(|bucket| {
            let mut tenant = None;
            for credential in bucket {
                let matches = candidate.len() == credential.bearer_token.len()
                    && bool::from(candidate.as_bytes().ct_eq(credential.bearer_token.as_ref()));
                if matches {
                    tenant = Some(Arc::clone(&credential.tenant_id));
                }
            }
            tenant
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }
}

fn token_digest(token: &[u8]) -> [u8; 32] {
    Sha256::digest(token).into()
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CredentialIndexError {
    #[error("credential token must not be empty")]
    EmptyCredential,
    #[error("credential tokens must be unique")]
    DuplicateCredential,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_authentication_handles_large_tenant_sets() {
        let credentials = (0..1_024)
            .map(|index| TenantCredential {
                tenant_id: format!("tenant-{index}"),
                bearer_token: format!("secret-token-{index:04}"),
            })
            .collect();
        let index = CredentialIndex::new(credentials).expect("credential index");
        assert_eq!(
            index.authenticate("secret-token-0999").as_deref(),
            Some("tenant-999")
        );
        assert!(index.authenticate("not-a-token").is_none());
    }

    #[test]
    fn duplicate_and_empty_credentials_fail_closed() {
        assert!(matches!(
            CredentialIndex::new(vec![TenantCredential {
                tenant_id: "tenant".to_owned(),
                bearer_token: String::new(),
            }]),
            Err(CredentialIndexError::EmptyCredential)
        ));
        assert!(matches!(
            CredentialIndex::new(vec![
                TenantCredential {
                    tenant_id: "a".to_owned(),
                    bearer_token: "same".to_owned(),
                },
                TenantCredential {
                    tenant_id: "b".to_owned(),
                    bearer_token: "same".to_owned(),
                },
            ]),
            Err(CredentialIndexError::DuplicateCredential)
        ));
    }

    #[test]
    fn transport_metrics_are_label_free_and_connection_scoped() {
        let metrics = TransportMetrics::default();
        let connection = metrics.connection_opened();
        metrics.record_accept_error();
        metrics.record_connection_error();
        metrics.record_forced_closes(2);
        let active = metrics.prometheus();
        assert!(active.contains("locus_http_connections_active 1\n"));
        assert!(!active.contains('{'));
        drop(connection);
        assert!(
            metrics
                .prometheus()
                .contains("locus_http_connections_active 0\n")
        );
    }
}
