use std::env;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use http_body_util::{BodyExt, Empty};
use hyper::Request;
use hyper::client::conn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde_json::json;
use tokio::net::TcpStream;
use tokio::sync::Barrier;

#[derive(Clone, Copy)]
enum Protocol {
    Http1,
    Http2,
}

impl Protocol {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "h1" | "http1" => Ok(Self::Http1),
            "h2" | "http2" => Ok(Self::Http2),
            _ => Err("--protocol must be h1 or h2".to_owned()),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Http1 => "h1",
            Self::Http2 => "h2c",
        }
    }
}

struct Settings {
    address: SocketAddr,
    path: String,
    protocol: Protocol,
    requests: usize,
    connections: usize,
    warmup_per_connection: usize,
    min_requests_per_second: f64,
    max_p99_millis: f64,
}

impl Settings {
    fn parse() -> Result<Self, String> {
        let mut settings = Self {
            address: "127.0.0.1:18081".parse().expect("default address"),
            path: "/bench/noop".to_owned(),
            protocol: Protocol::Http1,
            requests: 100_000,
            connections: 64,
            warmup_per_connection: 16,
            min_requests_per_second: 0.0,
            max_p99_millis: f64::INFINITY,
        };
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value for {argument}"))?;
            match argument.as_str() {
                "--address" => {
                    settings.address = value
                        .parse()
                        .map_err(|_| "--address must be host:port".to_owned())?;
                }
                "--path" => {
                    if !value.starts_with('/') {
                        return Err("--path must begin with /".to_owned());
                    }
                    settings.path = value;
                }
                "--protocol" => settings.protocol = Protocol::parse(&value)?,
                "--requests" => settings.requests = positive_usize(&argument, &value)?,
                "--connections" => settings.connections = positive_usize(&argument, &value)?,
                "--warmup-per-connection" => {
                    settings.warmup_per_connection = value.parse().map_err(|_| {
                        "--warmup-per-connection must be a non-negative integer".to_owned()
                    })?;
                }
                "--min-rps" => {
                    settings.min_requests_per_second = nonnegative_f64(&argument, &value)?;
                }
                "--max-p99-ms" => {
                    settings.max_p99_millis = nonnegative_f64(&argument, &value)?;
                }
                _ => return Err(format!("unknown argument: {argument}")),
            }
        }
        if settings.connections > settings.requests {
            return Err("--connections must not exceed --requests".to_owned());
        }
        Ok(settings)
    }
}

fn positive_usize(name: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{name} must be a positive integer"))
}

fn nonnegative_f64(name: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| format!("{name} must be a non-negative finite number"))
}

#[tokio::main]
async fn main() -> ExitCode {
    let settings = match Settings::parse() {
        Ok(settings) => settings,
        Err(error) => {
            eprintln!("http_load: {error}");
            return ExitCode::from(2);
        }
    };
    match run(settings).await {
        Ok(passed) if passed => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("http_load: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(settings: Settings) -> Result<bool, String> {
    let barrier = Arc::new(Barrier::new(settings.connections + 1));
    let mut tasks = Vec::with_capacity(settings.connections);
    for worker in 0..settings.connections {
        let requests = settings.requests / settings.connections
            + usize::from(worker < settings.requests % settings.connections);
        let barrier = Arc::clone(&barrier);
        let path = settings.path.clone();
        tasks.push(tokio::spawn(worker_run(
            settings.address,
            path,
            settings.protocol,
            requests,
            settings.warmup_per_connection,
            barrier,
        )));
    }
    if tokio::time::timeout(Duration::from_secs(10), barrier.wait())
        .await
        .is_err()
    {
        for task in tasks {
            task.abort();
        }
        return Err("workers did not establish connections within 10 seconds".to_owned());
    }
    let started = Instant::now();
    let mut latencies = Vec::with_capacity(settings.requests);
    let mut errors = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok(worker_latencies)) => latencies.extend(worker_latencies),
            Ok(Err(error)) => errors.push(error),
            Err(error) => errors.push(error.to_string()),
        }
    }
    let elapsed = started.elapsed();
    latencies.sort_unstable();
    let completed = latencies.len();
    let requests_per_second = completed as f64 / elapsed.as_secs_f64();
    let p50_millis = percentile_millis(&latencies, 50);
    let p95_millis = percentile_millis(&latencies, 95);
    let p99_millis = percentile_millis(&latencies, 99);
    let passed = errors.is_empty()
        && completed == settings.requests
        && requests_per_second >= settings.min_requests_per_second
        && p99_millis <= settings.max_p99_millis;
    println!(
        "{}",
        json!({
            "schema": "locus.http-transport-benchmark.v1",
            "status": if passed { "passed" } else { "failed" },
            "claim": "loopback transport ceiling; no model engine or GPU work",
            "protocol": settings.protocol.as_str(),
            "target_path": settings.path,
            "os": env::consts::OS,
            "arch": env::consts::ARCH,
            "requests": settings.requests,
            "completed_requests": completed,
            "connections": settings.connections,
            "warmup_requests": settings.connections * settings.warmup_per_connection,
            "elapsed_seconds": elapsed.as_secs_f64(),
            "requests_per_second": requests_per_second,
            "latency_millis": {"p50": p50_millis, "p95": p95_millis, "p99": p99_millis},
            "errors": errors,
            "gates": {
                "min_requests_per_second": settings.min_requests_per_second,
                "max_p99_millis": if settings.max_p99_millis.is_finite() {
                    Some(settings.max_p99_millis)
                } else {
                    None
                },
            },
        })
    );
    Ok(passed)
}

async fn worker_run(
    address: SocketAddr,
    path: String,
    protocol: Protocol,
    requests: usize,
    warmup: usize,
    barrier: Arc<Barrier>,
) -> Result<Vec<u64>, String> {
    match protocol {
        Protocol::Http1 => http1_worker(address, path, requests, warmup, barrier).await,
        Protocol::Http2 => http2_worker(address, path, requests, warmup, barrier).await,
    }
}

async fn http1_worker(
    address: SocketAddr,
    path: String,
    requests: usize,
    warmup: usize,
    barrier: Arc<Barrier>,
) -> Result<Vec<u64>, String> {
    let stream = TcpStream::connect(address)
        .await
        .map_err(|error| error.to_string())?;
    let (mut sender, connection) = conn::http1::handshake(TokioIo::new(stream))
        .await
        .map_err(|error| error.to_string())?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    for _ in 0..warmup {
        send_http1(&mut sender, &path).await?;
    }
    barrier.wait().await;
    let mut latencies = Vec::with_capacity(requests);
    for _ in 0..requests {
        let started = Instant::now();
        send_http1(&mut sender, &path).await?;
        latencies.push(started.elapsed().as_nanos().min(u64::MAX.into()) as u64);
    }
    Ok(latencies)
}

async fn http2_worker(
    address: SocketAddr,
    path: String,
    requests: usize,
    warmup: usize,
    barrier: Arc<Barrier>,
) -> Result<Vec<u64>, String> {
    let stream = TcpStream::connect(address)
        .await
        .map_err(|error| error.to_string())?;
    let (mut sender, connection) = conn::http2::Builder::new(TokioExecutor::new())
        .handshake(TokioIo::new(stream))
        .await
        .map_err(|error| error.to_string())?;
    tokio::spawn(async move {
        let _ = connection.await;
    });
    for _ in 0..warmup {
        send_http2(&mut sender, &path).await?;
    }
    barrier.wait().await;
    let mut latencies = Vec::with_capacity(requests);
    for _ in 0..requests {
        let started = Instant::now();
        send_http2(&mut sender, &path).await?;
        latencies.push(started.elapsed().as_nanos().min(u64::MAX.into()) as u64);
    }
    Ok(latencies)
}

async fn send_http1(
    sender: &mut conn::http1::SendRequest<Empty<Bytes>>,
    path: &str,
) -> Result<(), String> {
    let response = sender
        .send_request(request(path, false))
        .await
        .map_err(|error| error.to_string())?;
    validate_response(response).await
}

async fn send_http2(
    sender: &mut conn::http2::SendRequest<Empty<Bytes>>,
    path: &str,
) -> Result<(), String> {
    let response = sender
        .send_request(request(path, true))
        .await
        .map_err(|error| error.to_string())?;
    validate_response(response).await
}

fn request(path: &str, absolute: bool) -> Request<Empty<Bytes>> {
    let uri = if absolute {
        format!("http://localhost{path}")
    } else {
        path.to_owned()
    };
    Request::builder()
        .uri(uri)
        .body(Empty::new())
        .expect("valid benchmark request")
}

async fn validate_response(response: hyper::Response<hyper::body::Incoming>) -> Result<(), String> {
    if response.status() != hyper::StatusCode::NO_CONTENT {
        return Err(format!("unexpected HTTP status: {}", response.status()));
    }
    response
        .into_body()
        .collect()
        .await
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn percentile_millis(sorted_nanos: &[u64], percentile: usize) -> f64 {
    if sorted_nanos.is_empty() {
        return f64::INFINITY;
    }
    let index = (sorted_nanos.len() - 1) * percentile / 100;
    sorted_nanos[index] as f64 / 1_000_000.0
}
