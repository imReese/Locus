use std::sync::{Arc, Mutex};

use ahash::AHashMap;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, Response, StatusCode, header};
use axum::routing::{get, post};
use axum::{Json, Router};
use http_body_util::BodyExt;
use locus_server::{ToolParserSettings, build_server, load_config};
use serde_json::{Value, json};
use tempfile::tempdir;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tower::ServiceExt;

#[derive(Clone, Default)]
struct EngineCapture {
    completions: Arc<Mutex<Vec<Value>>>,
}

async fn engine_health() -> &'static str {
    "ok"
}

async fn engine_completion(
    State(capture): State<EngineCapture>,
    Json(body): Json<Value>,
) -> Response<Body> {
    capture
        .completions
        .lock()
        .expect("completion capture")
        .push(body);
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(concat!(
            "data: {\"choices\":[{\"index\":0,\"text\":\"configured\",\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        )))
        .expect("engine SSE response")
}

async fn parser_engine_completion(
    State(capture): State<EngineCapture>,
    Json(body): Json<Value>,
) -> Response<Body> {
    capture
        .completions
        .lock()
        .expect("completion capture")
        .push(body);
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(concat!(
            "data: {\"choices\":[{\"index\":0,\"text\":\"<thi\",\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\"nk>checked</think><tool_\",\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\"call>{\\\"name\\\":\\\"weather\\\",\\\"arguments\\\":{\\\"city\\\":\\\"Beijing\\\"}}</tool_call><tool_call>{\\\"name\\\":\\\"clock\\\",\\\"arguments\\\":{\\\"timezone\\\":\\\"UTC+8\\\"}}\",\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"text\":\"</tool_call>\",\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":12}}\n\n",
            "data: [DONE]\n\n"
        )))
        .expect("parser engine SSE response")
}

async fn malformed_parser_engine_completion() -> Response<Body> {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(concat!(
            "data: {\"choices\":[{\"index\":0,\"text\":\"<think>private chain\",\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )))
        .expect("malformed parser SSE response")
}

async fn spawn_engine() -> (String, EngineCapture) {
    let capture = EngineCapture::default();
    let app = Router::new()
        .route("/health", get(engine_health))
        .route("/v1/completions", post(engine_completion))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock engine");
    let address = listener.local_addr().expect("engine address");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve mock engine");
    });
    (format!("http://{address}"), capture)
}

async fn spawn_parser_engine() -> (String, EngineCapture) {
    let capture = EngineCapture::default();
    let app = Router::new()
        .route("/health", get(engine_health))
        .route("/v1/completions", post(parser_engine_completion))
        .with_state(capture.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind parser engine");
    let address = listener.local_addr().expect("parser engine address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve parser engine");
    });
    (format!("http://{address}"), capture)
}

async fn spawn_malformed_parser_engine() -> String {
    let app = Router::new()
        .route("/health", get(engine_health))
        .route("/v1/completions", post(malformed_parser_engine_completion));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind malformed parser engine");
    let address = listener.local_addr().expect("malformed parser address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve malformed parser engine");
    });
    format!("http://{address}")
}

fn write_config() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempdir().expect("temp directory");
    let tokenizer_path = directory.path().join("tokenizer.json");
    let template_path = directory.path().join("chat_template.jinja");
    let config_path = directory.path().join("locus.json");
    let model = WordLevel::builder()
        .vocab(AHashMap::from_iter([
            ("<unk>".to_owned(), 0),
            ("user".to_owned(), 1),
            ("hello".to_owned(), 2),
            ("assistant".to_owned(), 3),
            ("tool".to_owned(), 4),
            ("weather".to_owned(), 5),
        ]))
        .unk_token("<unk>".to_owned())
        .build()
        .expect("word-level model");
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
        .save(&tokenizer_path, false)
        .expect("save tokenizer");
    std::fs::write(
        &template_path,
        "{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}{% if tools %}tool {{ tools[0].function.name }} {% endif %}assistant",
    )
    .expect("write template");
    std::fs::write(
        &config_path,
        json!({
            "listen": "127.0.0.1:0",
            "models": [{
                "aliases": ["fixture"],
                "model_revision": "fixture-v1",
                "tokenizer_json": "tokenizer.json",
                "tokenizer_revision": "tokenizer-v1",
                "chat_template": "chat_template.jinja",
                "template_revision": "template-v1",
                "reasoning_parser": {
                    "kind": "tagged",
                    "revision": "think-tags-v1",
                    "start_delimiter": "<think>",
                    "end_delimiter": "</think>"
                },
                "tool_parser": {
                    "kind": "tagged_json",
                    "revision": "tool-envelope-v1",
                    "start_delimiter": "<tool_call>",
                    "end_delimiter": "</tool_call>",
                    "max_buffered_bytes": 4096
                }
            }],
            "engines": [{
                "id": "sglang-0",
                "kind": "sglang",
                "base_url": "http://127.0.0.1:9",
                "served_model": "fixture",
                "model": "fixture",
                "runtime_version": "test",
                "target_id": "sglang-0/fixture"
            }]
        })
        .to_string(),
    )
    .expect("write config");
    (directory, config_path)
}

#[tokio::test]
async fn configured_profile_parses_reasoning_and_tools_from_remote_completion_text() {
    let (engine_url, capture) = spawn_parser_engine().await;
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.engines[0].base_url = engine_url;
    let server = build_server(config, directory.path()).expect("build server");
    let response = server
        .app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "fixture",
                        "messages": [{"role": "user", "content": "hello"}],
                        "reasoning_effort": "high",
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "description": "Get weather",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"]
                                    },
                                    "strict": true
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "clock",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"timezone": {"type": "string"}},
                                        "required": ["timezone"]
                                    }
                                }
                            }
                        ],
                        "tool_choice": "required"
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes(),
    )
    .expect("response JSON");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "checked"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "weather"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Beijing\"}"
    );
    assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["index"], 0);
    assert_eq!(body["choices"][0]["message"]["tool_calls"][1]["index"], 1);
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][1]["function"]["name"],
        "clock"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"],
        "{\"timezone\":\"UTC+8\"}"
    );
    let completions = capture.completions.lock().expect("completion capture");
    assert_eq!(completions[0]["prompt"], json!([1, 2, 4, 5, 3]));
}

#[tokio::test]
async fn malformed_parser_output_fails_closed_without_returning_buffered_content() {
    let engine_url = spawn_malformed_parser_engine().await;
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.engines[0].base_url = engine_url;
    let server = build_server(config, directory.path()).expect("build server");
    let response = server
        .app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "fixture",
                        "messages": [{"role": "user", "content": "hello"}],
                        "reasoning_effort": "high"
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes(),
    )
    .expect("error JSON");
    assert_eq!(body["error"]["code"], "internal_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("unterminated reasoning"))
    );
    assert!(!body.to_string().contains("private chain"));
}

#[tokio::test]
async fn relative_profile_paths_build_a_server_with_request_ids() {
    let (directory, config_path) = write_config();
    let config = load_config(&config_path).expect("load config");
    let server = build_server(config, directory.path()).expect("build server");
    let health = server
        .app
        .clone()
        .oneshot(
            Request::get("/healthz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("health response");
    assert_eq!(health.status(), StatusCode::OK);
    assert!(
        health
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("locus-http-"))
    );
    let models = server
        .app
        .clone()
        .oneshot(
            Request::get("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("models response");
    assert_eq!(models.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &models
            .into_body()
            .collect()
            .await
            .expect("models body")
            .to_bytes(),
    )
    .expect("models JSON");
    assert_eq!(body["data"][0]["id"], "fixture");

    let readiness = server
        .app
        .oneshot(
            Request::get("/readyz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("readiness response");
    assert_eq!(readiness.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn configured_huggingface_profile_reaches_remote_engine_as_token_ids() {
    let (engine_url, capture) = spawn_engine().await;
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.engines[0].base_url = engine_url;
    let server = build_server(config, directory.path()).expect("build server");
    let readiness = server
        .app
        .clone()
        .oneshot(
            Request::get("/readyz")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("readiness response");
    assert_eq!(readiness.status(), StatusCode::OK);
    let response = server
        .app
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"model": "fixture", "input": "hello"}).to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes(),
    )
    .expect("response JSON");
    assert_eq!(body["output"][0]["content"][0]["text"], "configured");
    let completions = capture.completions.lock().expect("completion capture");
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0]["prompt"], json!([1, 2, 3]));
    assert_eq!(completions[0]["model"], "fixture");
    assert!(
        completions[0]["rid"]
            .as_str()
            .is_some_and(|request_id| request_id.starts_with("resp_"))
    );
}

#[tokio::test]
async fn configured_raw_completion_bypasses_chat_template_and_forwards_stop() {
    let (engine_url, capture) = spawn_engine().await;
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.engines[0].base_url = engine_url;
    let server = build_server(config, directory.path()).expect("build server");
    let response = server
        .app
        .oneshot(
            Request::post("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "fixture",
                        "prompt": "hello",
                        "stop": ["END"]
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes(),
    )
    .expect("response JSON");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "configured");
    let completions = capture.completions.lock().expect("completion capture");
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0]["prompt"], json!([2]));
    assert_eq!(completions[0]["stop"], json!(["END"]));
    assert!(
        completions[0]["rid"]
            .as_str()
            .is_some_and(|request_id| request_id.starts_with("cmpl_"))
    );
}

#[test]
fn engine_must_reference_a_configured_model_alias() {
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.engines[0].model = "missing".to_owned();
    let error = match build_server(config, directory.path()) {
        Ok(_) => panic!("unknown model alias must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("unknown model alias missing"));
}

#[test]
fn invalid_profile_parser_configuration_fails_startup() {
    let (directory, config_path) = write_config();
    let mut config = load_config(&config_path).expect("load config");
    config.models[0].tool_parser = Some(ToolParserSettings::TaggedJson {
        revision: "tool-envelope-v1".to_owned(),
        start_delimiter: "<tool_call>".to_owned(),
        end_delimiter: "</tool_call>".to_owned(),
        max_buffered_bytes: 0,
    });
    let error = match build_server(config, directory.path()) {
        Ok(_) => panic!("zero parser limit must fail startup"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("must be greater than zero"));
}
