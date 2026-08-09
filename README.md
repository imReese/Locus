# Locus

**Locus is an engine-neutral inference control plane for compute and model-state
placement.**

Locus sits above inference engines such as SGLang, vLLM, TensorRT-LLM, and
future runtimes. It normalizes model-facing semantics, understands execution
capabilities and reusable-state locality, and makes global placement decisions
across compute and state.

> **Locus decides where inference should happen and where its reusable state
> should live.**

Inference is a placement problem involving both compute and state. Inference
engines should execute model workloads, not repeatedly implement
application-facing semantics or global placement policy.

For intuition, Nginx sits in front of web servers; Locus coordinates workloads
in front of inference engines. Locus is inference-native, not merely another
API gateway or reverse proxy.

> [!IMPORTANT]
> Locus now contains a deployable end-to-end control-plane slice: a configured
> server process, content-addressed Hugging Face tokenizer/chat-template
> profiles, OpenAI Responses and Chat Completions, SGLang/vLLM completion
> adapters, and an optional NexusKV bridge. CI validates the official OpenAI
> Python SDK against a real local HTTP server and runs Locus against a separate
> NexusKV bridge process. Live GPU serving and NexusKV physical transfer remain
> opt-in validations and are not claimed by CI.

## Why Locus?

`Locus` means a location or place. The name centers placement as the core
abstraction without limiting the project to protocol ingress or proxying. The
project category may be described as an inference control plane or inference
orchestrator; **Locus** is the project name.

The previous frontend-centered name and gateway-centered alternatives were
rejected because they overemphasize protocol ingress. No compatibility alias
is required because no implementation artifact was released under that name.

SGLang, vLLM, TensorRT-LLM, and future runtimes should be able to focus on
efficient execution. Applications should not have to adopt a different set of
templates, parsers, request semantics, and traffic policies for each engine.

Locus coordinates those concerns in a common control plane:

- OpenAI-compatible and future northbound protocols
- request validation and normalization
- chat templates, tokenization, and detokenization
- reasoning, tool-call, and multimodal semantics
- sampling and streaming normalization
- routing, load balancing, and admission control
- capability discovery and observability
- cost-based request and state placement

The core architectural contributions are:

1. a stable, engine-neutral semantic normalization layer;
2. a canonical engine protocol;
3. capability-based engine adapters;
4. a generic, optional state-provider interface; and
5. cost-based global placement of requests and reusable state.

State locality is a first-class planner input, not an add-on routing policy.
The design covers KV caches, MLA state, recurrent/KDA checkpoints, multimodal
artifacts, and future model state rather than equating reuse with the longest
matching token prefix.

## Architecture

```text
 Applications and clients
            |
            | OpenAI-compatible and future protocols
            v
+----------------------------------------------------------+
| Locus                                                    |
| protocol | model semantics | admission | global planner  |
| routing  | observability   | compute + state placement   |
+----------------------------+-----------------------------+
             |               |
             | canonical     | generic StateProvider
             | engine        | (optional)
             | protocol      v
             |       NexusKV or another state system
             v
   +---------+---------+------------------+
   |                   |                  |
SGLang adapter     vLLM adapter     future adapters
   |                   |                  |
SGLang engine      vLLM engine      other engines
```

Locus owns global, cross-engine decisions. Execution engines continue to
own continuous batching, engine-local scheduling, GPU memory and KV-page
allocation, kernels, speculative decoding execution, CUDA graphs, distributed
parallelism, and model forward execution.

[NexusKV](https://github.com/imReese/NexusKV) is the intended reference
integration for reusable model state. It is not a dependency of Locus
core. A deployment may use another `StateProvider` implementation or run with
state integration disabled.

## Design principles

- **Engine neutrality:** core request and response types do not expose SGLang,
  vLLM, TensorRT-LLM, or transport-framework types.
- **Capability negotiation:** adapters advertise what an engine can actually
  execute; unsupported features fail explicitly or follow an intentional
  fallback policy.
- **Semantic consistency:** templates, tokenization, parsing, and finish
  semantics remain stable when traffic moves between compatible engines.
- **Extensible inputs:** the canonical `InputBundle` represents token
  sequences, multimodal references, metadata, and future input forms.
- **State as a planning dimension:** reuse boundaries, compatibility,
  placement, and materialization cost participate in every eligible placement
  decision.
- **Clear ownership:** Locus performs global planning; engines retain
  control of their local execution loops.
- **No Python hot-path requirement:** Rust is the primary implementation
  language. Python remains an SDK and compatibility escape hatch.

## Documentation

- [Architecture](docs/design/architecture.md)
- [Canonical engine protocol](docs/design/canonical-engine-protocol.md)
- [Engine adapter contract](docs/design/engine-adapter-contract.md)
- [Model semantics](docs/design/model-semantics.md)
- [State-aware scheduling](docs/design/state-aware-scheduling.md)
- [OpenAI-compatible API](docs/design/openai-api.md)
- [NexusKV bridge](docs/design/nexuskv-bridge.md)
- [Serving and configuration](docs/operations/serving.md)
- [Validation and evidence levels](docs/validation/serving.md)

## Implementation

The Rust 2024 workspace keeps the architecture boundaries small and explicit:

- `locus-core`: canonical requests, execution facts, identities, reusable-state
  contracts, and operation context;
- `locus-semantics`: model registry, normalization, typed semantic events,
  tool-call aggregation, reasoning, and structured-output validation;
- `locus-semantics-hf`: production tokenizer.json loading, bounded MiniJinja
  chat-template rendering, detokenization, and content-derived semantic
  fingerprints;
- `locus-engine`: engine adapter contract, registry, and deterministic fake;
- `locus-state`: state-provider contract, null provider, and deterministic fake;
- `locus-planner`: cost-based path selection and the side-effecting
  `PlanExecutor`;
- `locus-runtime`: `InferenceService`, target discovery, state-candidate
  construction, planning, semantic streaming, and cancellation;
- `locus-openai`: Responses, Chat Completions, model listing, health, SSE, and
  OpenAI-shaped errors;
- `locus-engine-openai`: network adapters for SGLang and vLLM completion
  endpoints;
- `locus-state-nexuskv`: optional, versioned HTTP bridge from NexusKV match and
  materialization results into the generic `StateProvider` handshake;
- `locus-server`: JSON configuration, dependency assembly, SGLang/vLLM and
  optional NexusKV registration, bearer authentication, request limits,
  dependency-aware readiness, request IDs, tracing, and graceful shutdown.

The byte tokenizer and simple template renderer remain deterministic reference
components for unit and SDK fixture tests. Deployments use
`locus-semantics-hf` with pinned tokenizer and template artifacts.

Run the same checks used by GitHub CI with:

```bash
bash scripts/ci.sh
```

Start a configured deployment with:

```bash
LOCUS_API_KEY=replace-me cargo run -p locus-server -- examples/locus-server.json
```

The example contains placeholder artifact revisions and paths; replace them
before use. See [Serving and configuration](docs/operations/serving.md).

## Implementation direction

The implementation uses Rust 2024, Tokio, Axum, Reqwest, Hugging Face
Tokenizers, and MiniJinja. Core contracts do not depend on those frameworks. A
future native southbound protocol may use Tonic and Protobuf. Production model
profiles use exact local tokenizer and chat-template artifacts; model-specific
reasoning/tool parsers are still required before those semantics can be
advertised by the current SGLang/vLLM adapters.

These are implementation choices behind stable Locus interfaces, not types
that define the core architecture. Wire formats and Rust APIs remain pre-1.0.

## Validation boundaries

- OpenAI Responses/Chat and adapter streaming are exercised through in-process
  HTTP/SSE conformance tests.
- Official `openai` Python SDK 2.53.0 E2E covers Responses and Chat JSON/SSE,
  structured output, errors, authentication, and client-disconnect cancellation
  against a real local Locus HTTP fixture in GitHub CI.
- SGLang and vLLM requests send canonical token IDs to `/v1/completions`; tests
  cover request IDs, structured-output mapping, usage, finish events, and the
  SGLang abort path. `scripts/live_engine_conformance.py` can exercise an
  explicitly configured live runtime, but no live runtime or GPU result is
  checked into or implied by CI.
- The NexusKV provider requires a separately deployed
  `locus.nexuskv-bridge.v1` service. NexusKV now ships those endpoints, and CI
  starts that implementation in a separate process with the real Rust matcher.
  Shared fixtures and the complete Locus prepare/materialize/commit handshake
  are enforced; native engine import and physical transfer remain unverified.
- Bearer authentication, global body/concurrency limits, readiness, request IDs,
  and structured tracing are implemented. Per-tenant rate/fairness admission,
  calibrated costs, telemetry export, multimodal normalization, and production
  model-specific reasoning/tool parsers remain future work.

## Scope

Locus is not:

- another model execution runtime;
- a replacement for engine-local schedulers;
- tied to one northbound API, inference engine, cache system, or HTTP stack;
- a guarantee that every engine can emulate every requested feature; or
- a generic reverse proxy that is unaware of inference semantics.

See the design documents for the component boundaries and incremental
implementation path.

## License

Locus is licensed under the [Apache License 2.0](LICENSE).
