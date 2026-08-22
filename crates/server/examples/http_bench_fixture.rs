use std::env;
use std::net::SocketAddr;

use axum::Router;
use axum::http::StatusCode;
use axum::routing::get;
use locus_http::TransportMetrics;
use locus_server::{HttpSettings, transport};

#[tokio::main]
async fn main() {
    let listen = env::var("LOCUS_HTTP_BENCH_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:18081".to_owned())
        .parse::<SocketAddr>()
        .expect("LOCUS_HTTP_BENCH_LISTEN must be a socket address");
    let settings = HttpSettings::default();
    let listener = transport::bind(listen, &settings).expect("bind benchmark fixture");
    println!(
        "LOCUS_HTTP_BENCH_READY={}",
        listener.local_addr().expect("local address")
    );
    let app = Router::new().route("/bench/noop", get(|| async { StatusCode::NO_CONTENT }));
    transport::serve(
        listener,
        app,
        settings,
        TransportMetrics::default(),
        std::future::pending(),
    )
    .await
    .expect("serve benchmark fixture");
}
