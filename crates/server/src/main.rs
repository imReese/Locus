use std::env;
use std::path::{Path, PathBuf};

use locus_server::{ObservabilitySettings, build_server, load_config};
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
        .with_graceful_shutdown(shutdown_signal())
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

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::error!(%error, "failed to install shutdown signal handler");
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
