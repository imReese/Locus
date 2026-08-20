use std::env;
use std::path::{Path, PathBuf};
use std::time::Duration;

use locus_engine::EngineRegistry;
use locus_runtime::TrafficController;
use locus_server::{ObservabilitySettings, ShutdownSettings, build_server, load_config};
use thiserror::Error;
use tokio::net::TcpListener;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("locus-server: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), MainError> {
    let config_path = config_path()?;
    let config = load_config(&config_path)?;
    let config_directory = config_path.parent().unwrap_or_else(|| Path::new("."));
    let server = build_server(config, config_directory)?;
    init_observability(&server.observability)?;
    let listener = TcpListener::bind(server.listen).await?;
    let local_address = listener.local_addr()?;
    info!(listen = %local_address, "Locus serving started");
    axum::serve(listener, server.app)
        .with_graceful_shutdown(shutdown_signal(
            server.traffic,
            server.engines,
            server.shutdown,
        ))
        .await?;
    info!("Locus serving stopped");
    Ok(())
}

fn config_path() -> Result<PathBuf, MainError> {
    let mut arguments = env::args_os();
    let _binary = arguments.next();
    let explicit = arguments.next();
    if arguments.next().is_some() {
        return Err(MainError::Usage);
    }
    explicit
        .map(PathBuf::from)
        .or_else(|| env::var_os("LOCUS_CONFIG").map(PathBuf::from))
        .ok_or(MainError::Usage)
}

fn init_observability(settings: &ObservabilitySettings) -> Result<(), MainError> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(settings.filter.clone()));
    if settings.json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .try_init()
            .map_err(|error| MainError::Tracing(error.to_string()))?;
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .compact()
            .try_init()
            .map_err(|error| MainError::Tracing(error.to_string()))?;
    }
    Ok(())
}

async fn shutdown_signal(
    traffic: TrafficController,
    engines: EngineRegistry,
    settings: ShutdownSettings,
) {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to install shutdown signal handler");
        return;
    }
    tracing::info!("shutdown signal received; beginning traffic drain");
    let grace = Duration::from_millis(settings.drain_timeout_millis);
    let traffic_report = traffic.drain(grace).await;
    tracing::info!("traffic admission is quiesced; beginning engine drain");
    let engine_report = engines.drain_all(grace).await;
    let forced = traffic_report
        .as_ref()
        .is_ok_and(|report| !report.completed)
        || engine_report.as_ref().is_ok_and(|report| !report.completed);
    match &traffic_report {
        Ok(report) => tracing::info!(
            completed = report.completed,
            forced_cancellations = report.forced_cancellations,
            "traffic drain finished"
        ),
        Err(error) => tracing::error!(%error, "traffic drain failed"),
    }
    match &engine_report {
        Ok(report) => tracing::info!(
            completed = report.completed,
            forced_cancellations = report.forced_cancellations,
            "engine drain finished"
        ),
        Err(error) => tracing::error!(%error, "engine drain failed"),
    }
    if forced {
        tokio::time::sleep(Duration::from_millis(settings.force_cancel_grace_millis)).await;
    }
}

#[derive(Debug, Error)]
enum MainError {
    #[error("usage: locus-server <config.json> (or set LOCUS_CONFIG)")]
    Usage,
    #[error(transparent)]
    Server(#[from] locus_server::ServerError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("failed to initialize tracing: {0}")]
    Tracing(String),
}
