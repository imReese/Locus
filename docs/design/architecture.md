# Architecture

## Status

This document defines the target architecture. An initial Rust workspace now
implements the core domain types, deterministic fakes, a cost-based planner,
and a `PlanExecutor` vertical slice. Production integrations and wire schemas
remain conceptual and will be refined through implementation.

## Thesis

Inference is a placement problem involving both compute and state.

Inference engines should execute model workloads, not repeatedly implement
application-facing semantics or global placement policy.

Locus is an engine-neutral inference control plane for compute and model-state
placement. It separates application protocols and model semantics from
runtime-specific execution, understands execution capabilities and reusable
state locality, and makes global placement decisions across compute and state.

## System boundaries

```text
Northbound       API and model semantics
                      |
                      v
Locus            protocol adaptation
                  validation and normalization
                  model semantics
                  admission and policy
                  global request + state planner
                  plan execution and orchestration
                  observability
                      |
                      v
Southbound       canonical engine protocol
                      |
                      v
Backends         capability-based engine adapters
                  SGLang | vLLM | TensorRT-LLM | future engines

State plane      generic StateProvider
                  NexusKV reference integration | other providers | disabled
```

The northbound boundary is application-facing and may support multiple API
protocols. The southbound boundary is an engine-neutral execution contract.
Engine adapters translate the canonical contract into runtime-specific calls.
The state plane is separate because state discovery and movement need not be
implemented by, or share a lifecycle with, an execution engine.

No core type may require an SGLang, vLLM, TensorRT-LLM, NexusKV, Axum, or
Pingora type. Integrations depend inward on Locus contracts.

### Engine-neutral boundary rationale

Using one runtime API as the internal protocol would couple model semantics,
placement, and errors to that engine. Locus therefore owns canonical request
and execution-event types, while capability-based adapters translate at the
edges. This adds translation and conformance-test work, but preserves portable
semantics and keeps engine-local scheduling under engine control.

The following alternatives were rejected:

- use one engine API as the internal protocol;
- route only at the northbound HTTP layer;
- let core call each runtime directly.

## Responsibility split

### Locus owns

- northbound protocol parsing and response shaping;
- request validation and canonicalization;
- chat-template selection and rendering;
- tokenization and detokenization;
- reasoning and tool-call parsing;
- multimodal input normalization;
- sampling-parameter normalization;
- admission control and tenant policy;
- engine capability discovery;
- global routing, load balancing, and placement;
- placement-plan execution and bounded replanning;
- reusable-state discovery and import orchestration;
- cross-engine observability and stable error semantics.

### Execution engines own

- continuous batching and engine-local scheduling;
- GPU memory management and KV-page allocation;
- attention and model kernels;
- speculative decoding execution;
- CUDA graph capture and replay;
- tensor, pipeline, and expert parallelism;
- device-local state representation;
- model forward execution.

Locus chooses an eligible target and supplies normalized work. It does not
dictate how the target batches or executes that work after admission.

### State providers own

- lookup of reusable model state;
- provider-specific state identity and metadata;
- locality and topology reporting;
- compatibility evidence;
- estimated transfer or materialization options;
- provider-specific preload, replication, and transfer operations;
- lifecycle and health of state managed by that provider.

The planner decides whether a provider operation is worthwhile. `PlanExecutor`
coordinates it. The provider does not make the final request-placement
decision or allocate engine-local destination memory.

## Logical components

### Protocol adapters

Protocol adapters translate an external API into an internal semantic request
and translate normalized output events back to that API. An OpenAI-compatible
adapter is an initial target, not the definition of the internal model.

### Model semantics

`ModelSemantics` composes narrowly scoped providers:

- `TokenizerProvider`
- `TemplateRenderer`
- multimodal normalization
- reasoning and tool-call parser factories
- model-specific validation and defaulting

Semantic components are versioned and selected from a deployment-controlled
model profile. The same profile is used when evaluating state compatibility.
See [Model semantics](model-semantics.md).

### Admission controller

Admission applies resource limits, tenant policy, priority, deadlines, and
overload behavior before expensive state movement or engine execution begins.
It produces constraints for the planner rather than selecting an engine by
itself.

### Routing policies

A `RoutingPolicy` contributes hard constraints, preferences, and an explainable
policy-cost term for candidate plans. Policies may express tenant affinity,
residency, fairness, cost, or stability requirements. They do not submit work,
mutate reusable state, or choose an engine independently of the planner.

Policies compose through explicit precedence rules. No policy can make an
engine or state artifact eligible after a capability, compatibility,
authorization, or admission check has rejected it.

### Capability registry

Every engine adapter publishes `EngineInstance` records for runtime processes
and `ExecutionTarget` records for model-bearing execution choices. A target
references an instance generation and adds immutable model revision, optional
adapter/LoRA revision, execution role, parallel layout, residency, and
target-specific capabilities.

The registry separates versioned capabilities from rapidly changing
`EngineSnapshot` load and health signals. Capability predicates decide target
eligibility before candidates are costed. A process restart invalidates its old
generation; model load or unload may change targets without changing the
identity of the process.

### Global planner

The `Planner` evaluates eligible execution targets and state-materialization
options together. Its input includes:

- canonical request requirements;
- policy and admission constraints;
- engine capabilities, health, and load;
- state matches, reusable boundaries, and compatibility;
- estimated queue, prefill, materialization, decode, topology, and policy
  costs.

Its output is a `PlacementPlan` containing an execution target, optional state
actions, fallback conditions, and an explanation suitable for observability.
Planning is pure or mostly side-effect-free. It does not reserve capacity,
mutate state, submit engine work, retry, or manage cleanup.

### Plan executor

`PlanExecutor` is the side-effect boundary between a `PlacementPlan` and an
execution stream:

```text
Planner -> PlacementPlan -> PlanExecutor
                             |-- reserve target
                             |-- prepare state import
                             |-- materialize / transfer state
                             |-- commit or abort state import
                             |-- submit request
                             |-- apply fallback or bounded replan
                             `-- cancel and clean up
```

**Planner decides. `PlanExecutor` performs side effects.** The executor
revalidates the target generation and plan preconditions, coordinates engine
and state-provider transactions, and records actual outcomes. It follows the
fallback encoded in the plan; it does not silently substitute a different
placement decision.

### Engine adapters

`EngineAdapter` instances hide runtime-specific control protocols. They
discover capabilities, report health and load, accept canonical work, emit
canonical output facts, and support cancellation. For reusable state, an
adapter prepares an engine-generation-scoped import destination, then commits
or aborts that import after the provider transfers state. Optional operations
are gated by advertised target capabilities.

See [Engine adapter contract](engine-adapter-contract.md).

### State providers

`StateProvider` supplies typed, evidence-bearing state candidates and explicit
materialization options. It may represent a distributed cache service, an
engine-local index, or another state system. A null implementation makes
state-free operation a normal configuration.

NexusKV is the reference integration, not a core dependency. See
[State-aware scheduling](state-aware-scheduling.md).

### Observability

Observability records the normalized request lifecycle without making a
particular telemetry vendor part of the core API. Important decisions include:

- admission outcome and applied policy;
- capability requirements and exclusion reasons;
- state candidates, compatibility results, and reusable boundaries;
- estimated cost terms and selected plan;
- reservation, materialization, execution, retry, and cancellation timing;
- token, finish-reason, and parser outcomes.

Sensitive prompts, media, tokens, or tool arguments are not included by
default. Deployments choose explicit redaction and sampling policies.

## Stable concepts

The initial implementation should organize around the following concepts,
regardless of concrete Rust module layout:

| Concept | Purpose |
| --- | --- |
| `ModelProfile` | Pin model aliases and versioned semantic components |
| `SemanticIdentity` | Partition input, generation, and output semantics |
| `TokenizerProvider` | Encode, decode, and expose tokenizer identity |
| `TemplateRenderer` | Render typed conversations with a versioned template |
| `ModelSemantics` | Compose model-specific normalization and parser behavior |
| `InputBundle` | Carry ordered token, multimodal, metadata, and future inputs |
| `CanonicalRequest` | Carry normalized engine-executable work |
| `EngineEvent` | Report ordered engine execution facts |
| `EngineInstance` | Identify one runtime process and restart generation |
| `ExecutionTarget` | Identify a model-bearing target on an engine instance |
| `EngineCapabilities` | Describe target support and compatibility limits |
| `EngineSnapshot` | Report dynamic target health and load observations |
| `EngineAdapter` | Operate targets and runtime-specific state destinations |
| `StateRequirement` | Describe reusable state relevant to a request |
| `StateDescriptor` | Report typed, located state with compatibility evidence |
| `ReusableBoundary` | Separate covered input from executable resume point |
| `CompatibilityResult` | Report artifact evidence; unknown fails closed |
| `MaterializationOption` | Estimate one source-to-target state path |
| `StateProvider` | Discover state and transfer it to negotiated destinations |
| `PreparedStateAttachment` | Bind committed state to one target generation |
| `RoutingPolicy` | Add constraints and policy cost without owning execution |
| `Planner` | Select request placement and state actions together |
| `PlacementPlan` | Record target, actions, fallbacks, and rationale |
| `PlanExecutor` | Perform reservations, imports, submission, and cleanup |

Traits should accept core-owned data transfer objects and return structured
errors. Transport clients, generated Protobuf structs, and integration SDK
types remain at the edges.

## Request lifecycle

1. A northbound adapter authenticates, parses, and assigns an internal request
   identity.
2. Validation checks protocol shape, deployment policy, and declared model
   support.
3. The selected `ModelSemantics` profile renders templates, normalizes media,
   tokenizes input, canonicalizes sampling, and declares required output
   semantics.
4. Admission control returns a rejection or planning constraints such as
   priority, deadline, and tenant limits.
5. The capability registry filters execution targets that cannot satisfy the
   request.
6. For eligible requests, the state provider returns reusable-state candidates
   and materialization estimates. With no provider, this is an empty result.
7. The planner chooses an execution target and optional state path without
   causing side effects.
8. `PlanExecutor` reserves the target and asks its engine adapter to prepare a
   generation-scoped state-import destination when reuse is planned.
9. The state provider transfers the selected source to that destination and
   returns a receipt. The engine adapter commits the import to a
   `PreparedStateAttachment`, or aborts partial work on failure.
10. `PlanExecutor` submits the canonical request and optional committed
    attachment through the engine adapter.
11. Engine execution facts flow through detokenization and incremental parsers,
    which derive semantic events for the northbound adapter.
12. Completion, cancellation, or failure releases reservations and records the
    actual cost and outcome for future estimates.

Planning must tolerate changes between observation and execution. A plan may
be invalidated by load, health, or state movement failure. Retries revalidate
capabilities and idempotency; they do not silently duplicate a request.

## Control plane and data plane

The initial design distinguishes logical roles without requiring separate
processes:

- The **control plane** manages model profiles, policies, adapter registration,
  capabilities, and topology.
- The **request data plane** performs normalization, planning, streaming, and
  execution orchestration.
- The **state data plane** performs state lookup and movement through an
  optional provider.

A first implementation may host these roles in one Rust process. Stable traits
and protocol boundaries should allow later separation without forcing it now.

## Primary implementation language

Rust 2024 is the primary implementation language. The initial direction is
Tokio, Axum/Hyper for HTTP and SSE, Tonic and Protobuf for remote contracts,
Hugging Face Tokenizers, and a Jinja-compatible Rust renderer such as
MiniJinja. Library, transport, generated-code, and Python runtime types remain
behind Locus-owned interfaces.

Python is limited to SDKs, tooling, or an explicitly isolated compatibility
worker. It is not required in the normal production hot path.

Rust was selected because a concurrent streaming control plane needs bounded
memory, backpressure, cancellation, and typed extension boundaries. The cost is
that unusual Python-only model behavior requires explicit isolation and parity
testing.

The following alternatives were rejected:

- make Python the primary control-plane runtime;
- embed Python in the Rust process;
- begin with a polyglot core before validating component boundaries.

## Failure model

Errors are classified by ownership and retry safety:

- `InvalidRequest`: client-visible and not retried;
- `Unsupported`: no eligible capability or semantic profile;
- `Rejected`: admission or policy decision;
- `Unavailable`: transient adapter, engine, or provider failure;
- `DeadlineExceeded`: planning, materialization, or execution exceeded budget;
- `ExecutionFailed`: engine accepted the request but failed it;
- `Internal`: invariant or unexpected integration failure.

State lookup failure may degrade to a cold placement only when deployment
policy allows it and the request deadline remains feasible. State
materialization failure triggers the fallback encoded in the plan. Correctness
never depends on treating an unverified state match as compatible.

## Security and isolation

Northbound content, templates, tokenizer assets, tool definitions, media, and
provider metadata cross trust boundaries. Implementations should:

- bound request and streamed-output sizes;
- authenticate engines and state providers;
- validate media schemes and dereference through controlled fetchers;
- isolate tenant state and include tenant scope in provider queries;
- treat model-supplied templates and custom code as untrusted;
- keep credentials and provider-private handles out of logs;
- enforce deadlines and cancellation across downstream operations.

Unusual `trust_remote_code` behavior may eventually run in an isolated Python
semantic worker. Python code is not embedded in the normal Locus hot path.

## Initial implementation sequence

This sequence is directional rather than a commitment that features exist:

1. define core DTOs, semantic identities, and structured errors;
2. validate `Planner`, `PlacementPlan`, `PlanExecutor`, `EngineAdapter`, and
   `StateProvider` boundaries with deterministic fakes;
3. implement one northbound protocol and a conformance test harness;
4. implement the canonical remote protocol and one engine adapter;
5. add capability-aware admission and production calibration;
6. integrate NexusKV as an optional reference provider;
7. add topology, preload, and replication policies.

Each stage should keep the engine-neutral boundary testable with fake adapters
and providers before adding another production integration.
