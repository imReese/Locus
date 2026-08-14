# Serving and Configuration

## Status

`locus-server` is the deployable Locus process. It loads one JSON configuration,
constructs production model semantics, registers SGLang or vLLM adapters and an
optional NexusKV provider, applies ingress policy, and serves the OpenAI routes.
The server fails startup on ambiguous or invalid dependency assembly rather than
starting with a silently partial configuration.

## Start the server

Build and run with an explicit configuration path:

```bash
cargo run -p locus-server -- /path/to/locus-server.json
```

The path may instead be supplied through `LOCUS_CONFIG`. The command-line path
wins when both are present. Relative tokenizer and template paths are resolved
against the directory containing the configuration file, not the caller's
working directory.

[`examples/locus-server.json`](../../examples/locus-server.json) shows a single
SGLang target. Its revisions, paths, endpoint, model name, and special tokens
are placeholders and must be replaced with deployment facts.

## Configuration ownership

The top-level configuration contains:

| Field | Purpose |
| --- | --- |
| `listen` | Socket address for the Locus HTTP server |
| `api` | Bearer-secret environment variable and global ingress limits |
| `models` | Public aliases plus immutable model and semantic artifacts |
| `engines` | SGLang/vLLM instances and planner-selectable targets |
| `state` | Disabled or a versioned NexusKV bridge configuration |
| `observability` | Default tracing filter and compact/JSON log format |

Unknown fields are rejected. Empty identities, zero parallel degrees, zero
engine generations, duplicate engine/target IDs, unknown model references,
models without an engine, missing secrets, unreadable artifacts, invalid
tokenizer JSON, and invalid templates all fail startup.

Secrets are named in configuration and read from the environment. The secret
value itself does not belong in the JSON file:

```json
{
  "api": {
    "bearer_token_env": "LOCUS_API_KEY",
    "max_request_bytes": 2097152,
    "max_concurrent_requests": 128
  }
}
```

Omitting `bearer_token_env` disables northbound authentication. This is useful
for a protected development network but should be an explicit deployment
choice. Engine and NexusKV API keys follow the same `*_env` pattern.

## Model profiles

Each model entry pins:

- one or more public aliases;
- immutable model, optional adapter, and execution-profile identities;
- an exact local `tokenizer.json` and declared tokenizer revision;
- an exact local chat-template file and declared template revision;
- bounded template context such as `bos_token` and `eos_token`;
- whether a generation marker is requested; and
- the maximum rendered prompt size;
- an optional tagged reasoning parser with an immutable revision and delimiters;
- an optional tagged-JSON tool parser with a revision, delimiters, and bounded
  per-call buffer.

Locus hashes the tokenizer, template, and canonical parser configurations with
SHA-256 at startup. Those digests become structured semantic-component
fingerprints and feed the umbrella profile fingerprint. A label such as `main`
or `latest` is not accepted as compatibility evidence by itself.

For conversation requests the selected `ModelSemantics` renders the messages
and OpenAI-shaped function definitions once, then tokenizes that rendered prompt
once. For `POST /v1/completions`, raw text is tokenized directly and a raw token
array is preserved without invoking the chat template. The resulting canonical
token IDs cross the engine boundary. The vLLM adapter explicitly sets
`add_special_tokens: false`; neither adapter renders or tokenizes the prompt
again. Canonical stop strings are forwarded to the runtime completion endpoint.

Example parser configuration:

```json
{
  "reasoning_parser": {
    "kind": "tagged",
    "revision": "model-think-tags-v1",
    "start_delimiter": "<think>",
    "end_delimiter": "</think>"
  },
  "tool_parser": {
    "kind": "tagged_json",
    "revision": "model-tool-envelope-v1",
    "start_delimiter": "<tool_call>",
    "end_delimiter": "</tool_call>",
    "max_buffered_bytes": 65536
  }
}
```

The selected delimiters and envelope must match the pinned model/template
dialect. Locus has no heuristic fallback. Stray or incomplete reserved syntax,
unknown functions, non-object arguments, and over-limit calls fail the request.
Profiles without a local parser still require compatible typed runtime events
and are rejected during planning when the adapter cannot provide them.

The current production template context supports text messages, function tool
schemas, tool choice, and deployment values. Multimodal bindings and additional
model-specific parser dialects are not implemented.

## Raw Completions example

The compatibility endpoint accepts one text prompt or one token-ID sequence:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $LOCUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "prompt": "Complete this raw prefix:",
    "max_tokens": 64,
    "stop": ["END"]
  }'
```

Set `"stream": true` for `text_completion` SSE chunks and `[DONE]`. A numeric
array such as `"prompt": [1, 2, 3]` is treated as already-tokenized input under
the configured model tokenizer identity. Batched prompts and multiple choices
are rejected rather than silently flattened.

## Probes

`GET /healthz` is a liveness probe. It answers while the process and router are
running and intentionally does not call a downstream dependency.

`GET /readyz` is a readiness probe. It loads every registered model profile,
discovers configured targets, calls their health snapshots, and returns success
only when every model identity has at least one ready target. Partial engine
failure is tolerated only when another ready target covers the same model.
Failure returns HTTP 503 with concise dependency evidence.

The probes are intentionally outside bearer authentication so Kubernetes or
another local orchestrator can call them. Do not expose them as a substitute for
an authenticated inference API.

## Ingress and observability

Every `/v1/*` route is protected when a bearer secret is configured. Locus uses
a constant-time token comparison after checking the candidate length. The body
limit rejects oversized JSON with HTTP 413; the concurrency limit bounds active
HTTP requests, including streaming requests. These are global safety limits,
not per-tenant rate or fairness admission.

Every request receives or preserves an `x-request-id`, and the response
propagates it. The HTTP stack emits tracing spans without logging prompt bodies,
token IDs, tool arguments, bearer tokens, or provider handles. Set `RUST_LOG` to
override the configured default filter. JSON logs are appropriate for a log
collector; compact logs are convenient locally.

SIGINT initiates graceful shutdown: the listener stops accepting new work and
Axum drains in-flight connections. Deadline policy, per-tenant admission,
telemetry export, and a deployment manifest remain separate follow-on work.

## Validation

Run the repository gate before deployment:

```bash
bash scripts/ci.sh
```

Then qualify the actual engine endpoint with the opt-in live harness described
in [Validation and evidence levels](../validation/serving.md). A green GitHub CI
run proves local protocol and SDK conformance; it does not prove the configured
GPU runtime, model artifacts, topology, performance, or NexusKV transfer path.
