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
> Locus is in its initial architecture bootstrap. The repository contains
> engine-neutral domain contracts, deterministic fakes, and a tested
> planner-to-executor vertical slice. It does not yet contain a production
> server or engine adapter, tokenizer pipeline, or state-system integration.

## Why Locus?

`Locus` means a location or place. The name centers placement as the core
abstraction without limiting the project to protocol ingress or proxying. The
project category may be described as an inference control plane or inference
orchestrator; **Locus** is the project name.

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
- [ADR 0001: Engine-neutral boundary](docs/adr/0001-engine-neutral-boundary.md)
- [ADR 0002: State-provider abstraction](docs/adr/0002-state-provider-abstraction.md)
- [ADR 0003: Rust as the primary language](docs/adr/0003-rust-primary-language.md)
- [ADR 0004: Project name Locus](docs/adr/0004-project-name-locus.md)

## Implementation

The initial Rust 2024 workspace keeps the architecture boundaries small and
explicit:

- `locus-core`: canonical requests, execution facts, identities, reusable-state
  contracts, and operation context;
- `locus-semantics`: model-profile and semantic-provider boundaries;
- `locus-engine`: engine adapter contract, registry, and deterministic fake;
- `locus-state`: state-provider contract, null provider, and deterministic fake;
- `locus-planner`: cost-based path selection and the side-effecting
  `PlanExecutor`.

The bootstrap intentionally has no HTTP server and no real SGLang, vLLM,
TensorRT-LLM, or NexusKV integration. Its purpose is to validate ownership,
compatibility filtering, generation fencing, import cleanup, fallback, and
compute-plus-state path selection before framework integration.

Run the local checks with:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
```

## Implementation direction

The intended implementation stack is Rust 2024 with Tokio. Initial protocol
adapters may use Axum/Hyper for HTTP and SSE, while the southbound contract is
expected to use Tonic and Protobuf. Hugging Face Tokenizers and a
Jinja-compatible renderer such as MiniJinja are the initial semantic building
blocks.

These are implementation choices behind stable Locus interfaces, not types
that define the core architecture. Wire formats and Rust APIs remain subject
to validation and change as the bootstrap advances.

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
