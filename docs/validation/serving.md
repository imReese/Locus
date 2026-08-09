# Serving Validation and Evidence Levels

## Evidence policy

Locus keeps these evidence levels separate:

| Level | What runs | What it establishes |
| --- | --- | --- |
| Static and deterministic | Rust tests, strict Clippy, rustdoc, mock HTTP/SSE | Local contracts, ownership, shape, ordering, and failure policy |
| Real local HTTP client | Official OpenAI Python SDK against `sdk_fixture` | SDK parsing and transport compatibility through a real socket |
| Cross-process state protocol | Locus against a separate NexusKV process with the Rust matcher | Versioned lookup/estimate/materialize compatibility and prepare/commit orchestration |
| Live engine | Opt-in SGLang/vLLM harness | Observed behavior of one configured runtime endpoint and model |
| Live state/hardware | GPU topology, performance, and physical NexusKV transfer | Production properties that mocks and protocol oracles cannot establish |

GitHub CI runs the first three levels. It does not have a configured model
server, GPU, native engine state-import path, or physical NexusKV transport, so
it cannot promote its result to either live level.

## Repository gate

The single local and CI entry point is:

```bash
bash scripts/ci.sh
```

It runs formatting, workspace-wide strict Clippy, all-feature tests, rustdoc with
warnings denied, and Python syntax checks. The `CI` GitHub workflow then starts a
real local Locus fixture for the pinned official `openai` Python SDK. A separate
job checks out NexusKV, starts its bridge process, and runs the shared bridge
conformance fixture through the Locus planner and import handshake.

## Official OpenAI SDK E2E

The automated SDK suite covers:

- Responses non-streaming JSON;
- Responses SSE lifecycle and text deltas;
- Chat Completions non-streaming JSON;
- Chat Completions SSE chunks and terminal completion;
- Responses typed reasoning and function-call items produced by profile parsers;
- Chat SSE reasoning content, indexed tool deltas, arguments, and `tool_calls`
  finish reason produced by the same parser fixture;
- JSON Schema structured-output request and JSON result;
- an unknown-model error parsed as `openai.NotFoundError`;
- bearer authentication on the fixture; and
- closing a Responses stream before completion and observing
  `EngineAdapter.cancel` downstream.

Run it manually without calling OpenAI-hosted models:

```bash
python -m pip install -r scripts/openai-sdk-e2e-requirements.txt
cargo run -p locus-server --example sdk_fixture
python scripts/openai_sdk_e2e.py --fixture-counts
```

The first process listens on `127.0.0.1:18080` by default. Override
`LOCUS_E2E_LISTEN` and pass the corresponding `--base-url` when that port is in
use. The default fixture key is intentionally test-only.

## Live SGLang or vLLM qualification

The live harness sends token IDs directly to `/v1/completions`, requests SSE
usage, checks JSON chunks, finish reasons and `[DONE]`, and then opens a second
request for cancellation. For SGLang it combines client disconnect with an
acknowledged `/abort_request`; for vLLM it records that the public completion
connection was closed but does not invent a server-side abort acknowledgement.

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

The result uses schema `locus.live-engine-conformance.v1` and records the runtime,
model, observation time, prompt-token count, lifecycle counts, usage, finish
reasons, cancellation evidence, and an explicit claim boundary. Do not commit
deployment secrets or private endpoint details with a result.

## Current boundary

The official SDK fixture passed locally with `openai` 2.53.0 and is enforced by
GitHub CI, including profile-parsed reasoning and tool calls. SGLang and vLLM
adapter behavior is covered by deterministic mock HTTP/SSE tests; configured
parser behavior uses a mock SGLang endpoint, not a live model. The cross-process
NexusKV protocol path is also automated with protocol-only, zero-byte transfer
evidence. No live GPU engine, native engine state import, physical NexusKV
transfer, or real-model parser conformance was available in this workspace, so
those evidence levels remain unverified rather than failed.
