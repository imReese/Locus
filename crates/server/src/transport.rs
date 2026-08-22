use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::service::TowerToHyperService;
use locus_http::TransportMetrics;
use socket2::{SockRef, TcpKeepalive};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinSet;
use tower::ServiceExt;

use crate::HttpSettings;

const ACCEPT_ERROR_BACKOFF: Duration = Duration::from_millis(50);

/// Binds the production listener with an explicit kernel accept backlog.
pub fn bind(address: SocketAddr, settings: &HttpSettings) -> io::Result<TcpListener> {
    settings.validate().map_err(invalid_input)?;
    let socket = if address.is_ipv4() {
        TcpSocket::new_v4()?
    } else {
        TcpSocket::new_v6()?
    };
    socket.set_reuseaddr(true)?;
    socket.bind(address)?;
    socket.listen(settings.listen_backlog)
}

/// Serves HTTP/1.1 and cleartext HTTP/2 on the same listener.
///
/// Accepted sockets and connection tasks are bounded independently from model
/// admission. On shutdown, every connection first receives graceful protocol
/// shutdown; tasks that do not finish within the configured transport grace
/// are aborted so a broken peer cannot hold process termination forever.
pub async fn serve<F>(
    listener: TcpListener,
    app: Router,
    settings: HttpSettings,
    metrics: TransportMetrics,
    shutdown: F,
) -> io::Result<()>
where
    F: Future<Output = ()> + Send,
{
    settings.validate().map_err(invalid_input)?;
    let permits = Arc::new(Semaphore::new(settings.max_connections));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let mut shutdown = pin!(shutdown);

    loop {
        let permit = tokio::select! {
            () = &mut shutdown => break,
            permit = Arc::clone(&permits).acquire_owned() => {
                permit.map_err(|_| io::Error::other("HTTP connection limiter closed"))?
            }
        };
        let accepted = tokio::select! {
            () = &mut shutdown => break,
            accepted = listener.accept() => accepted,
        };
        let (stream, peer) = match accepted {
            Ok(accepted) => accepted,
            Err(error) => {
                drop(permit);
                metrics.record_accept_error();
                tracing::warn!(%error, "HTTP accept failed; applying bounded backoff");
                tokio::time::sleep(ACCEPT_ERROR_BACKOFF).await;
                continue;
            }
        };
        let connection_metrics = metrics.connection_opened();
        if let Err(error) = configure_stream(&stream, &settings) {
            metrics.record_connection_error();
            tracing::debug!(%peer, %error, "HTTP socket configuration failed");
            continue;
        }
        let connection_app = app.clone();
        let connection_settings = settings.clone();
        let connection_shutdown = shutdown_rx.clone();
        let task_metrics = metrics.clone();
        tasks.spawn(async move {
            let _permit = permit;
            let _connection_metrics = connection_metrics;
            if let Err(error) = serve_connection(
                stream,
                connection_app,
                connection_settings,
                connection_shutdown,
            )
            .await
            {
                task_metrics.record_connection_error();
                tracing::debug!(%peer, %error, "HTTP connection closed with protocol error");
            }
        });
    }

    drop(listener);
    let _ = shutdown_tx.send(true);
    drop(shutdown_rx);
    let wait_for_connections = async {
        while let Some(result) = tasks.join_next().await {
            if let Err(error) = result
                && !error.is_cancelled()
            {
                tracing::warn!(%error, "HTTP connection task failed");
            }
        }
    };
    if tokio::time::timeout(settings.connection_shutdown_timeout(), wait_for_connections)
        .await
        .is_err()
    {
        let remaining = tasks.len();
        metrics.record_forced_closes(remaining);
        tracing::warn!(remaining, "forcing transport connection shutdown");
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
    }
    Ok(())
}

fn configure_stream(stream: &TcpStream, settings: &HttpSettings) -> io::Result<()> {
    stream.set_nodelay(settings.tcp_nodelay)?;
    let socket = SockRef::from(stream);
    socket.set_keepalive(true)?;
    socket.set_tcp_keepalive(
        &TcpKeepalive::new().with_time(Duration::from_secs(settings.tcp_keepalive_seconds)),
    )?;
    Ok(())
}

async fn serve_connection(
    stream: TcpStream,
    app: Router,
    settings: HttpSettings,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut builder = Builder::new(TokioExecutor::new());
    builder
        .http1()
        .keep_alive(settings.http1_keep_alive)
        .header_read_timeout(settings.http1_header_read_timeout())
        .max_buf_size(settings.http1_max_buffer_bytes)
        .timer(TokioTimer::new());
    builder
        .http2()
        .adaptive_window(settings.http2_adaptive_window)
        .max_concurrent_streams(settings.http2_max_concurrent_streams)
        .max_header_list_size(settings.http2_max_header_list_bytes)
        .max_send_buf_size(settings.http2_max_send_buffer_bytes)
        .keep_alive_interval(settings.http2_keep_alive_interval())
        .keep_alive_timeout(settings.http2_keep_alive_timeout())
        .timer(TokioTimer::new());

    let service = app.map_request(|request: Request<hyper::body::Incoming>| request.map(Body::new));
    let service = TowerToHyperService::new(service);
    let io = TokioIo::new(stream);
    let mut connection = pin!(builder.serve_connection_with_upgrades(io, service));
    if *shutdown.borrow() {
        connection.as_mut().graceful_shutdown();
    }
    loop {
        tokio::select! {
            result = connection.as_mut() => return result,
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    connection.as_mut().graceful_shutdown();
                }
            }
        }
    }
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use http_body_util::Empty;
    use hyper::client::conn;
    use tokio::sync::oneshot;

    #[test]
    fn default_transport_settings_are_valid() {
        HttpSettings::default().validate().expect("valid defaults");
    }

    #[test]
    fn invalid_transport_limits_fail_closed() {
        let invalid = [
            HttpSettings {
                max_connections: 0,
                ..HttpSettings::default()
            },
            HttpSettings {
                http1_max_buffer_bytes: 8_191,
                ..HttpSettings::default()
            },
            HttpSettings {
                http2_max_concurrent_streams: 0,
                ..HttpSettings::default()
            },
            HttpSettings {
                http2_max_send_buffer_bytes: 0,
                ..HttpSettings::default()
            },
        ];
        for settings in invalid {
            assert!(settings.validate().is_err());
        }
    }

    #[tokio::test]
    async fn production_transport_serves_reused_http1_connection() {
        let settings = HttpSettings::default();
        let listener =
            bind("127.0.0.1:0".parse().expect("address"), &settings).expect("bind listener");
        let address = listener.local_addr().expect("local address");
        let (stop_tx, stop_rx) = oneshot::channel();
        let server = tokio::spawn(serve(
            listener,
            Router::new().route("/healthz", get(|| async { "ok" })),
            settings,
            TransportMetrics::default(),
            async move {
                let _ = stop_rx.await;
            },
        ));

        let stream = TcpStream::connect(address).await.expect("connect");
        let (mut sender, connection) = conn::http1::handshake(TokioIo::new(stream))
            .await
            .expect("HTTP/1 handshake");
        let connection = tokio::spawn(connection);
        for _ in 0..2 {
            let response = sender
                .send_request(
                    Request::builder()
                        .uri("/healthz")
                        .body(Empty::<axum::body::Bytes>::new())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_eq!(response.status(), hyper::StatusCode::OK);
        }
        drop(sender);
        connection.await.expect("connection task").ok();
        stop_tx.send(()).ok();
        server.await.expect("server task").expect("server result");
    }

    #[tokio::test]
    async fn production_transport_negotiates_prior_knowledge_http2() {
        let settings = HttpSettings::default();
        let listener =
            bind("127.0.0.1:0".parse().expect("address"), &settings).expect("bind listener");
        let address = listener.local_addr().expect("local address");
        let (stop_tx, stop_rx) = oneshot::channel();
        let server = tokio::spawn(serve(
            listener,
            Router::new().route("/healthz", get(|| async { "ok" })),
            settings,
            TransportMetrics::default(),
            async move {
                let _ = stop_rx.await;
            },
        ));

        let stream = TcpStream::connect(address).await.expect("connect");
        let (mut sender, connection) = conn::http2::Builder::new(TokioExecutor::new())
            .handshake(TokioIo::new(stream))
            .await
            .expect("HTTP/2 handshake");
        let connection = tokio::spawn(connection);
        let response = sender
            .send_request(
                Request::builder()
                    .uri("http://localhost/healthz")
                    .body(Empty::<axum::body::Bytes>::new())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), hyper::StatusCode::OK);
        drop(sender);
        connection.await.expect("connection task").ok();
        stop_tx.send(()).ok();
        server.await.expect("server task").expect("server result");
    }
}
