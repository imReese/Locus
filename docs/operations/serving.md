# Serving and Configuration

## Status

`locus-server` is the deployable Locus process. It loads a static model-catalog
source from one JSON configuration, constructs production model I/O profiles,
registers SGLang or vLLM adapters and an optional NexusKV store, applies
ingress policy, and serves the OpenAI routes. Model I/O profiles and live engine
inventory are separate: a catalog profile may exist while no engine currently
serves it. Invalid semantic artifacts and ambiguous identity mappings still fail
startup.

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
| `api` | Request-body limit plus optional legacy single-tenant authentication |
| `traffic` | Trusted tenant credentials, hierarchical admission, token/deadline limits, and overload shedding |
| `models` | Static catalog source: public aliases plus immutable model and semantic artifacts |
| `required_models` | Optional aliases that must be routable for readiness |
| `engines` | SGLang/vLLM instances whose live model inventory is discovered at runtime |
| `store` | Disabled or a versioned NexusKV store configuration |
| `placement` | Shadow/active calibrated placement, durable state, gates, and conservative priors |
| `observability` | Default tracing filter and compact/JSON log format |
| `shutdown` | Engine/traffic drain grace and forced-cancellation grace |

Unknown fields are rejected. Empty identities, zero parallel degrees, zero
engine generations, duplicate engine/target IDs, unknown explicit profile
mappings, missing secrets, unreadable artifacts, invalid tokenizer JSON, and
invalid templates all fail startup. A model profile without a currently loaded
engine target does not fail startup.

Secrets are named in configuration and read from the environment. Production
credentials belong to a tenant policy, so the authenticated bearer token is the
only source of tenant identity. A request body, metadata field, or
`x-tenant-id` header cannot select a different policy:

```json
{
  "traffic": {
    "classes": [{
      "id": "latency",
      "weight": 4,
      "max_active_requests": 48,
      "max_active_tokens": 196608
    }],
    "tenants": [{
      "id": "premium",
      "service_class": "latency",
      "weight": 4,
      "max_active_requests": 32,
      "max_active_tokens": 131072,
      "max_queued_requests": 256,
      "max_tokens_per_request": 32768,
      "default_request_timeout_millis": 60000,
      "max_request_timeout_millis": 120000,
      "bearer_token_env": "LOCUS_PREMIUM_API_KEY"
    }]
  }
}
```

For a custom `traffic` block, anonymous access is disabled unless
`api.anonymous_tenant` explicitly names a configured tenant. The top-level
`api.bearer_token_env` remains a compatibility form for the implicit legacy
`default` tenant and cannot be combined with tenant credentials. Engine and
NexusKV API keys follow the same `*_env` pattern.

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

For conversation requests the selected `ModelIo` renders the messages
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

The `models` array is a semantic catalog, not an engine allowlist. Locus needs a
trusted tokenizer, template, parser configuration, and immutable execution
identity before it can safely normalize a request. A downstream model name that
has no catalog profile is therefore ignored rather than exposed through a
wildcard passthrough.

The current production template context supports text messages, function tool
schemas, tool choice, and deployment values. Multimodal bindings and additional
model-specific parser dialects are not implemented.

## Engine model discovery

Engine entries identify runtime instances, not permanent model deployments. On
target discovery each SGLang/vLLM adapter calls the instance's `GET /v1/models`
endpoint and publishes only the intersection between its live inventory and the
configured model catalog. Loading or unloading a model changes the planner's
target inventory without restarting Locus.

By default every public catalog alias is offered as a candidate downstream model
name to every engine. The minimal engine configuration therefore contains only
instance facts:

```json
{
  "id": "sglang-0",
  "kind": "sglang",
  "base_url": "http://127.0.0.1:30000",
  "runtime_version": "0.5.2",
  "topology": "node-0/gpu-0",
  "hardware": "cuda"
}
```

When the downstream name differs from the public alias, configure an explicit
identity mapping:

```json
{
  "model_mappings": [
    {
      "upstream_model": "org/model-runtime-name",
      "profile": "public-model-name"
    }
  ]
}
```

An optional `target_id` may be added to a mapping; otherwise Locus derives
`<engine-id>/<upstream-model>`. The legacy `served_model`, `model`, and
`target_id` fields remain accepted together as a single-model compatibility
form, but new configurations should use discovery defaults or `model_mappings`.

## Live engine telemetry

Each engine entry has a bounded `telemetry` block. Locus scrapes the runtime's
Prometheus endpoint, filters model-labelled samples to the discovered upstream
model, and understands current SGLang and vLLM names for running/waiting
requests, KV pressure, prompt/output token counters, and SGLang generation
throughput. Counter deltas provide prefill/decode rates after two observations.
Missing, malformed, oversized, non-finite, or expired data remains unavailable;
it is never converted to a zero queue.

```json
{
  "telemetry": {
    "metrics_path": "/metrics",
    "request_timeout_millis": 2000,
    "min_scrape_interval_millis": 500,
    "valid_for_millis": 5000,
    "max_response_bytes": 2097152,
    "max_samples": 20000,
    "require_fresh_metrics": true
  }
}
```

Set `require_fresh_metrics` when an engine must be removed from routing if a
fresh scrape cannot be obtained. When it is false, the target may remain ready,
but the calibrated scorer substitutes configured conservative costs and active
promotion remains blocked for that target.

## Calibrated placement lifecycle

Calibrated placement defaults to `shadow`. Both the legacy plan and calibrated
plan are produced from the same hard model, capability, semantic, state, health,
and policy filters. Only the legacy plan executes in shadow mode. Locus records
decision agreement and repeats calibrated planning to detect nondeterminism.

Outcome learning is keyed by engine ID and generation plus immutable model,
adapter, and execution-profile revisions. Idle TTFT calibrates prefill;
TTFT residual while requests are waiting calibrates queue delay; inter-token
completion time calibrates decode; store estimate versus observed transfer
calibrates materialization; state-import activation overhead calibrates the
topology term. Cancelled, failed, incomplete, or ambiguous observations do not
update the corresponding estimator. Prompts, raw token IDs, output text,
store handles, and credentials are never persisted.

```json
{
  "placement": {
    "mode": "shadow",
    "state_path": "/var/lib/locus/calibration.json",
    "calibration": {
      "min_samples_per_metric": 32,
      "max_mape_bps": 2500,
      "min_shadow_decisions": 128,
      "min_shadow_agreement_bps": 9500,
      "persistence_flush_every_updates": 16,
      "max_records": 4096,
      "max_materialization_paths_per_record": 256,
      "max_state_bytes": 33554432
    }
  }
}
```

The state file is schema-versioned, byte-bounded, and replaced atomically after
a bounded number of updates. Engine/model records and materialization paths
have hard cardinality limits with deterministic eviction. The last incomplete
batch may be lost on an abrupt process exit; this can only remove evidence and
cause demotion. Before any active decision is admitted, pending evidence is
synchronously made durable.

Active mode is fail-closed. It requires a persistent state path and the exact
operator acknowledgement below:

```json
{
  "placement": {
    "mode": "active",
    "state_path": "/var/lib/locus/calibration.json",
    "active_confirmation": "enable-calibrated-placement"
  }
}
```

Even then, each selected path must have fresh telemetry, enough samples for all
cost terms it uses, acceptable EWMA MAPE, sufficient shadow volume and
agreement, zero replay mismatches, and healthy persistence. Any failed gate or
calibration error automatically executes the legacy plan; it does not fail the
inference request or weaken hard constraints.

## Production traffic control

Admission runs after model normalization, not in an HTTP concurrency
middleware. Locus therefore charges the exact normalized prompt-token count
plus the requested output-token reservation (or `default_output_tokens` when
the request omits it). A request must fit the global, service-class, and tenant
active request/token caps. Oversized work fails before engine discovery and
never waits forever for capacity it cannot obtain.

Queued work is selected in two deterministic stages: the service class with the
lowest token-normalized virtual runtime, then the tenant with the lowest
token-normalized virtual runtime inside that class. Each dispatch increments
virtual runtime by `reserved_tokens / configured_weight`. Tenant FIFO order is
preserved. This makes weights meaningful for heterogeneous prompt/output sizes
without allowing a warm-cache hit to bypass fairness.

All queue lengths are bounded globally and per tenant. `QueueFull` and
class-aware overload shedding return HTTP 429. A class may configure
`shed_at_global_utilization_bps`; once active-token utilization reaches that
threshold, new work in that class is rejected before occupying queue space.
This is explicit load shedding: Locus does not silently reduce output tokens or
change sampling semantics.

The authenticated tenant policy supplies a default and maximum request
deadline. A client may request a shorter deadline with
`x-request-timeout-ms`; a longer value is clamped to the tenant maximum. The
same `OperationContext` deadline and cancellation token cover admission,
bounded request-body ingress and the post-parse validation checkpoint,
discovery, store lookup/estimate/materialization, state-import operations,
engine request establishment, calibration persistence, and every streamed
engine event. Deadline expiry returns HTTP 408 with `deadline_exceeded`; client
disconnect cancels the context, drops the downstream transport, and calls the
adapter cancellation hook. State-import cleanup uses a separate bounded
two-second cleanup context so an expired request cannot strand a prepared
import.

`GET /metrics` exports Locus-native Prometheus text. Its only labels are the
configured `class` and `tenant` plus bounded enums such as `outcome` and
`reason`; request IDs, model aliases, engine IDs, prompts, and error strings are
never labels. Startup limits policy cardinality to 64 classes and 1,024 tenants.
The export contains admission/rejection/termination counters,
active and queued request/token gauges, queue-wait sum/count, drain state, and
forced-cancellation count.

The engine registry has explicit `ready -> draining -> stopped` lifecycle.
Draining engines disappear from discovery and reject execution leases while
existing streams retain their leases. `EngineRegistry::drain(engine_id, grace)`
supports bounded maintenance drain of one runtime; it waits only for that
engine's leases, cancels only those contexts at expiry, and leaves other
runtimes routable. On SIGINT, Locus first drains traffic so
requests admitted before the signal may still finish planning and acquire their
engine lease. It rejects queued/new work and waits up to
`shutdown.drain_timeout_millis`, cancelling remaining contexts at expiry. Locus
then drains engine leases with the same bound and gives forced cancellations
`force_cancel_grace_millis` to propagate before Axum finishes shutdown.

## Raw Completions example

The compatibility endpoint accepts one text prompt or one token-ID sequence:

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H "Authorization: Bearer $LOCUS_PREMIUM_API_KEY" \
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

`GET /v1/models` reports the public aliases whose immutable profile has at least
one currently discovered, healthy target. A known profile without a target is
omitted until an engine advertises it.

`GET /readyz` is a readiness probe. It succeeds when at least one catalog model
has a discovered, healthy target. Other catalog profiles may be unavailable
without taking the whole gateway out of service. Add aliases to top-level
`required_models` when a deployment contract requires specific models; each
listed alias must then have a ready target. Failure returns HTTP 503 with concise
dependency evidence. The response reports total profiles, routable models,
required models, observed targets, ready targets, placement mode, calibration
revision, and calibration persistence health.
During drain it returns HTTP 503 with `traffic_draining: true` even if engine
health is otherwise green.

The probes are intentionally outside bearer authentication so Kubernetes or
another local orchestrator can call them. Do not expose them as a substitute for
an authenticated inference API.

## Ingress and observability

Every `/v1/*` route is protected when tenant credentials are configured. Locus
checks every configured credential with a constant-time comparison after the
candidate-length check and injects the matched tenant as a trusted request
extension. The body limit rejects oversized JSON with HTTP 413. Active and
queued work—including streaming bodies—is then owned by the token-weighted
runtime admission permit described above.

Every request receives or preserves an `x-request-id`, and the response
propagates it. The HTTP stack emits tracing spans without logging prompt bodies,
token IDs, tool arguments, bearer tokens, or store handles. Set `RUST_LOG` to
override the configured default filter. JSON logs are appropriate for a log
collector; compact logs are convenient locally.

At info level, placement records the selected target, legacy/shadow/active
source, promotion blockers, freshness/confidence, full selected cost breakdown,
fallback, and redacted observed timing. Enable `locus_runtime=debug` to emit the
bounded legacy and calibrated path audit: deterministic rank, hard-constraint
exclusion reason codes, state kind/store ID, each cost term, and telemetry
revision/TTL. At most 256 path records per decision variant are emitted; a
truncation warning reports larger candidate sets. Neither level includes prompt
content, raw token IDs, generated content, opaque state handles, or credentials.

SIGINT initiates the bounded traffic and engine-drain sequence described above.
While the grace window runs, the process may still answer probes, but new
inference admission returns HTTP 503 and readiness is false.

## Validation

Run the repository gate before deployment:

```bash
bash scripts/ci.sh
```

Then qualify the actual engine endpoint with the opt-in live harness described
in [Validation and evidence levels](../validation/serving.md). A green GitHub CI
run proves local protocol and SDK conformance; it does not prove the configured
GPU runtime, model artifacts, topology, performance, or NexusKV transfer path.
