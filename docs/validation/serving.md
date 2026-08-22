# Serving Validation and Evidence Levels

## Evidence policy

Locus keeps these evidence levels separate:

| Level | What runs | What it establishes |
| --- | --- | --- |
| Static and deterministic | Rust tests, strict Clippy, rustdoc, mock HTTP/SSE | Local contracts, ownership, shape, ordering, and failure policy |
| Real local HTTP client | Official OpenAI and Anthropic Python SDKs against `sdk_fixture` | SDK parsing and transport compatibility through a real socket |
| Local transport ceiling | Release HTTP/1.1 and h2c load against `http_bench_fixture` | Connection-loop throughput and latency without model/GPU work |
| Cross-process state protocol | Locus against a separate NexusKV process with the Rust matcher | Versioned lookup/estimate/materialize compatibility and prepare/commit orchestration |
| Live engine | Opt-in SGLang/vLLM harness | Observed behavior of one configured runtime endpoint and model |
| Live dual-engine traffic control | Locus plus two live runtimes under concurrent multi-tenant load | Both engines execute measured tokens; tenant admission, deadlines, cancellation, metrics cardinality, error ratio, throughput, and latency meet explicit gates |
| Live state/hardware | GPU topology, performance, and physical NexusKV transfer | Production properties that mocks and protocol oracles cannot establish |

GitHub CI runs the static, SDK, and cross-process protocol levels. The transport
ceiling is opt-in because shared-runner timing is not a stable performance gate.
CI has no configured model server, GPU, native engine state-import path, or
physical NexusKV transport, so it cannot promote its result to a live level.

## Repository gate

The single local and CI entry point is:

```bash
bash scripts/ci.sh
```

It runs formatting, workspace-wide strict Clippy, all-feature tests, rustdoc with
warnings denied, and Python syntax checks. The `CI` GitHub workflow then starts a
real local Locus fixture for the pinned official OpenAI and Anthropic Python SDKs. A separate
job checks out NexusKV, starts its bridge process, and runs the shared bridge
conformance fixture through the Locus planner and import handshake.

## Official OpenAI and Anthropic SDK E2E

The automated SDK suite covers:

- Responses non-streaming JSON;
- Responses SSE lifecycle and text deltas;
- Chat Completions non-streaming JSON;
- Chat Completions SSE chunks and terminal completion;
- Completions non-streaming JSON with a raw text prompt and stop sequences;
- Completions caller-supplied token IDs without chat-template rendering;
- Completions SSE text, finish reason, final usage, and `[DONE]` parsed by the
  official SDK;
- strict Completions rejection before inference for unsupported multiple
  choices;
- Responses typed reasoning and function-call items produced by profile parsers;
- Chat SSE reasoning content, indexed tool deltas, arguments, and `tool_calls`
  finish reason produced by the same parser fixture;
- JSON Schema structured-output request and JSON result;
- an unknown-model error parsed as `openai.NotFoundError`;
- bearer authentication on the fixture; and
- closing a Responses stream before completion and observing
  `EngineAdapter.cancel` downstream.

The pinned Anthropic SDK 0.120.2 additionally covers Messages JSON/SSE, ordered
content-block events, client tool definitions and `tool_use` output,
Anthropic-shaped authentication/not-found errors, request IDs, usage/finish
reasons, and stream-close cancellation.

Run it manually without calling OpenAI-hosted models:

```bash
python -m pip install -r scripts/openai-sdk-e2e-requirements.txt
python -m pip install -r scripts/anthropic-sdk-e2e-requirements.txt
cargo run -p locus-server --example sdk_fixture
python scripts/openai_sdk_e2e.py --fixture-counts
python scripts/anthropic_sdk_e2e.py --fixture-counts
```

The first process listens on `127.0.0.1:18080` by default. Override
`LOCUS_E2E_LISTEN` and pass the corresponding `--base-url` when that port is in
use. The default fixture key is intentionally test-only.

## Local HTTP transport benchmark

Build both sides in release mode, start the no-op fixture, and run HTTP/1.1 and
h2c separately:

```bash
cargo build --release -p locus-server \
  --example http_bench_fixture --example http_load
target/release/examples/http_bench_fixture

target/release/examples/http_load \
  --protocol h1 --requests 100000 --connections 64 \
  --warmup-per-connection 32 --min-rps 10000 --max-p99-ms 20
target/release/examples/http_load \
  --protocol h2 --requests 100000 --connections 64 \
  --warmup-per-connection 32 --min-rps 10000 --max-p99-ms 20
```

The JSON result is schema `locus.http-transport-benchmark.v1` and fails its
process exit gate on errors, incomplete work, low throughput, or excessive p99.
It is intentionally labelled a loopback transport ceiling: the no-op route does
not perform authentication, tokenization, planning, upstream engine I/O, model
execution, TLS, or GPU work. Thresholds must be recorded with OS, architecture,
hardware, build profile, commit, and deployment topology; this benchmark cannot
replace live dual-engine acceptance.

## Live SGLang or vLLM qualification

The live harness first reads bounded Prometheus telemetry, sends token IDs
directly to `/v1/completions`, requests SSE usage, checks JSON chunks, finish
reasons and `[DONE]`, and reads telemetry again. It requires runtime scheduler,
KV, and token metrics and verifies that prompt/output counters advance. It then
opens a second request for cancellation. For SGLang it combines client
disconnect with an acknowledged `/abort_request`; for vLLM it records that the
public completion connection was closed but does not invent a server-side abort
acknowledgement.

Example SGLang invocation:

```bash
python scripts/live_engine_conformance.py \
  --runtime sglang \
  --base-url http://127.0.0.1:30000 \
  --model my-served-model \
  --prompt-token-ids 1,2,3,4 \
  --output /tmp/locus-sglang-live.json
```

Example vLLM invocation:

```bash
python scripts/live_engine_conformance.py \
  --runtime vllm \
  --base-url http://127.0.0.1:8000 \
  --model my-served-model \
  --prompt-token-ids 1,2,3,4 \
  --output /tmp/locus-vllm-live.json
```

Use token IDs produced by the exact configured model profile. Supplying
arbitrary IDs proves only transport acceptance and can produce meaningless
model output. Add `--api-key`, `--health-path`, or `--json-schema` when the live
deployment requires them.

Use `--metrics-path` for a non-default or separately hosted endpoint. The
response and sample counts are bounded by default and can be tightened with
`--max-metrics-bytes` and `--max-metric-samples`.

The result uses schema `locus.live-engine-conformance.v2` and records the runtime,
model, observation time, prompt-token count, lifecycle counts, usage, finish
reasons, selected metric names and values, counter deltas, cancellation
evidence, and an explicit claim boundary. It deliberately omits base URLs and
API keys. Do not commit deployment secrets or private endpoint details with a
result.

## Live dual-engine traffic-control acceptance

First qualify each engine independently with `live_engine_conformance.py`.
Then start one Locus process configured with both ready targets and at least two
credential-bound tenants. The dual-engine harness sends deterministic
round-robin tenant traffic through Locus and reads each runtime's own
Prometheus token counters before and after the load. It fails unless both
engines advance prompt and generation counters; two configured targets or a
successful gateway response alone is not accepted as dual-engine evidence.
The two engine arguments must name distinct endpoints. Before sending load, the
harness observes a quiet counter window and fails closed if background token
movement exceeds `--max-background-token-delta`; the second snapshot becomes
the attribution baseline. Both engines must then advance during the normal
multi-tenant phase itself, not only during the later overload probe.

The harness also requires at least two ready targets, bounds Locus metrics
bytes/sample count, rejects undocumented or high-cardinality Locus labels,
checks per-tenant success, enforces configured error-ratio and p95 gates,
closes a live stream and observes the bounded `client_cancelled` counter, and
requires a real HTTP 408/`deadline_exceeded` probe. API keys are read from
named environment variables and are not written to the JSON result. When
`--overload-requests` is non-zero, a separate burst must observe at least one
HTTP 429 with code `overloaded`; successful normal-load requests and shed
overload requests are reported separately.

```bash
export LOCUS_PREMIUM_API_KEY=replace-me
export LOCUS_BATCH_API_KEY=replace-me

python scripts/traffic_control_load.py \
  --base-url http://127.0.0.1:8000 \
  --model my-model \
  --tenant premium=LOCUS_PREMIUM_API_KEY \
  --tenant batch=LOCUS_BATCH_API_KEY \
  --engine sglang,http://127.0.0.1:30000,my-model \
  --engine vllm,http://127.0.0.1:30001,my-model \
  --requests 200 \
  --concurrency 32 \
  --max-tokens 64 \
  --overload-tenant batch \
  --overload-requests 100 \
  --overload-concurrency 64 \
  --overload-max-tokens 1024 \
  --max-error-ratio 0.01 \
  --max-p95-seconds 10 \
  > /tmp/locus-dual-engine-live.json
```

The output labels its evidence level `live_dual_engine` and omits API keys and
endpoint URLs. It is deployment evidence, not a portable benchmark: record the
model revisions, engine versions, hardware/topology, Locus commit,
configuration digest, and command parameters in the surrounding private test
record.

## Current boundary

The official SDK fixture passed locally with `openai` 2.53.0 and is enforced by
GitHub CI, including raw Completions and profile-parsed reasoning/tool calls.
SGLang and vLLM adapter behavior and telemetry aliases are covered by
deterministic mock HTTP/Prometheus/SSE tests; configured
parser behavior uses a mock SGLang endpoint, not a live model. The cross-process
NexusKV protocol path is also automated with protocol-only, zero-byte transfer
evidence. The dual-engine harness is present but intentionally not executed by
provider-free CI. No live GPU engine, native engine state import, physical NexusKV
transfer, or real-model parser conformance was available in this workspace, so
those evidence levels remain unverified rather than failed.
