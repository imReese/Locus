<h1 align="center">Locus</h1>

<p align="center">
  <strong>A policy-aware control plane for inference placement across engines.</strong>
</p>

<p align="center">
  Normalize each request once. Apply fleet policy once.<br/>
  Choose compute and reusable state together.
</p>

<p align="center">
  <strong><a href="#start-in-two-minutes">Get started</a></strong>
  &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <strong><a href="docs/design/architecture.md">Architecture</a></strong>
  &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <strong><a href="docs/operations/serving.md">Operate Locus</a></strong>
</p>

<p align="center">
  <sub>English &nbsp;&middot;&nbsp; <a href="README_CN.md">简体中文</a></sub>
</p>

<p align="center">
  <a href="https://github.com/imReese/Locus/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/imReese/Locus/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Rust 1.85+" src="https://img.shields.io/badge/Rust-1.85%2B-orange.svg">
  <img alt="API status: pre-1.0" src="https://img.shields.io/badge/API-pre--1.0-yellow.svg">
</p>

Locus gives applications one OpenAI-compatible entry point across SGLang and
vLLM runtimes. For each request, it resolves exact model semantics, applies
credential-bound tenant and traffic policy, filters targets by capability, and
weighs compute cost together with compatible reusable-state paths.

Use Locus when placement depends on model capabilities, request semantics,
tenant policy, runtime load, and state locality—not only endpoint health or
request counts. Execution engines still own batching, accelerator memory,
kernels, and model execution. Optional state systems such as NexusKV provide
reuse evidence and materialization options; they do not choose the target.

## Start in two minutes

Run a complete HTTP path backed by deterministic fake engines—no GPU, model
download, hosted API key, or external provider required. Rust 1.85 or newer is
required.

```bash
git clone https://github.com/imReese/Locus.git
cd Locus
cargo run -p locus-server --example sdk_fixture
```

In another terminal, send an OpenAI Responses request:

```bash
curl http://127.0.0.1:18080/v1/responses \
  -H 'Authorization: Bearer locus-test-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"locus-test","input":"respond with JSON"}'
```

A successful run returns a completed OpenAI Responses object. The deterministic
fixture response includes:

```json
{
  "model": "locus-test",
  "status": "completed",
  "output": [{"content": [{"text": "{\"answer\":\"ok\"}", "type": "output_text"}]}]
}
```

Or exercise Responses, Chat Completions, raw Completions, JSON/SSE, structured
output, errors, authentication, and cancellation through the pinned official
Python SDK:

```bash
python -m pip install -r scripts/openai-sdk-e2e-requirements.txt
python scripts/openai_sdk_e2e.py --fixture-counts
```

This quickstart proves the local protocol and orchestration path. It does not
claim that a live model ran.

## Follow one request

<p align="center">
  <img src="docs/assets/locus-architecture.svg" alt="Locus architecture showing application protocols, the control plane, reusable-state evidence, and execution targets">
</p>

The planner can reason about facts that disappear at a generic HTTP routing
layer:

- the exact tokenizer, template, parser, and model revision behind an alias;
- whether a target supports the requested output and execution semantics;
- tenant limits, priority, deadlines, cancellation, and drain state;
- queue, prefill, decode, topology, and policy costs; and
- compatible reusable state, its executable coverage, locality, and
  materialization cost.

Planning remains separate from side effects. The planner returns a
`PlacementPlan`; `PlanExecutor` revalidates it, reserves the target, coordinates
optional state import, submits work, applies bounded fallback, and cleans up.

## Know where Locus fits

These components solve different layers of the serving problem:

| Component | Decides | Deliberately leaves elsewhere |
| --- | --- | --- |
| Generic L7 proxy | Which upstream receives an HTTP request | Model semantics, runtime capabilities, reusable state, engine-local execution |
| Inference engine scheduler | How work is batched and executed inside one runtime | Fleet-wide tenant policy and cross-runtime placement |
| State/cache system | What reusable state exists and how it may be materialized | Final request placement and runtime-local consumption |
| **Locus** | Which eligible target and state path should serve a normalized request | Batching, device allocation, kernels, and physical data-plane ownership |

Locus is useful when routing depends on inference facts, not only endpoint
health or request counts. It is not intended to replace the other three layers.

## What ships today

> [!IMPORTANT]
> Locus is pre-1.0. The repository contains a deployable control-plane slice
> with deterministic CI and real local SDK transport. Live GPU execution,
> native engine state import, and physical state transfer require separate
> deployment qualification and are not implied by a green CI badge.

| Surface | Available now | Still requires deployment evidence |
| --- | --- | --- |
| API and model semantics | Responses, Chat Completions, raw text/token Completions, model listing, JSON/SSE, structured output, versioned model profiles, and OpenAI-shaped errors | Additional model dialects and full OpenAI parameter parity |
| Traffic and placement | Credential-bound tenants, weighted admission, deadlines, cancellation, capability filtering, explainable plans, bounded replanning, and shadow calibration | Fairness and placement accuracy under real workloads |
| Engine adapters | SGLang and vLLM completion, SSE, and Prometheus telemetry adapters | Repeated qualification against configured live runtimes and models |
| Reusable state | Generic `StateStore`, state-aware costing, and a versioned NexusKV bridge | Native engine import and physical state transfer |
| Operations | Health/readiness, Prometheus metrics, request IDs, structured tracing, overload shedding, bounded drain, and graceful shutdown | Production soak, topology, and fault-tolerance behavior |

## Run a configured server

[`examples/locus-server.json`](examples/locus-server.json) demonstrates model
profiles, tenant policy, runtime discovery, bounded telemetry, shadow placement,
and graceful drain. Replace every placeholder artifact revision, path,
credential, and endpoint with deployment facts:

```bash
LOCUS_PREMIUM_API_KEY=replace-me \
LOCUS_BATCH_API_KEY=replace-me \
cargo run -p locus-server -- examples/locus-server.json
```

The server exposes `/healthz`, `/readyz`, `/metrics`, `/v1/models`,
`/v1/responses`, `/v1/chat/completions`, and `/v1/completions`. New deployments
should begin with placement calibration in `shadow` mode. Promotion to `active`
requires persistent qualified evidence and exact operator confirmation.

## Evidence and qualification

GitHub CI covers static and deterministic checks, the official OpenAI SDK over
local HTTP, and the versioned cross-process state protocol. Live runtime,
multi-runtime traffic, native state import, physical transfer, and hardware
behavior remain explicit opt-in or deployment-specific gates. See
[Validation and evidence levels](docs/validation/serving.md) for the exact
harnesses, observations, and acceptance criteria.

## Read next

| Goal | Document |
| --- | --- |
| Understand the ownership model | [Architecture](docs/design/architecture.md) |
| Implement an engine integration | [Canonical engine protocol](docs/design/canonical-engine-protocol.md) · [Engine adapter contract](docs/design/engine-adapter-contract.md) |
| Add model semantics or an API dialect | [Model I/O](docs/design/model-io.md) · [OpenAI-compatible API](docs/design/openai-api.md) |
| Follow compute and reusable-state planning | [State-aware scheduling](docs/design/state-aware-scheduling.md) · [NexusKV bridge](docs/design/nexuskv-bridge.md) |
| Configure and operate the server | [Serving and configuration](docs/operations/serving.md) |

The workspace follows the request path: foundation crates (`core`, `model-io`,
`parser`), engine/store ports, planner/runtime control plane, edge adapters, and
the `locus-server` application. See [the crate map](crates/README.md) for package
names and responsibilities.

## Development

Run the same repository gate used by GitHub CI:

```bash
bash scripts/ci.sh
```

It checks formatting, strict workspace Clippy, all-feature tests, rustdoc with
warnings denied, Python syntax, and the traffic-harness unit tests. Rust and
public wire formats remain pre-1.0.

## Contributing

Bug reports and focused contributions are welcome. Read the
[contributing guide](CONTRIBUTING.md) before opening a pull request; substantial
protocol or architecture changes should begin with a GitHub issue.

## License

Locus is licensed under the [Apache License 2.0](LICENSE).
