# Architecture

## Status

This document defines the initial target architecture. It is a design contract,
not a description of an existing implementation. Interfaces and wire schemas
shown here are conceptual and will be refined through implementation.

## Thesis

Inference engines should execute model workloads, not repeatedly implement
application-facing semantics.

InferFront is an engine-neutral frontend and state-aware traffic control plane
for LLM inference. It separates application protocols and model semantics from
runtime-specific execution, then makes global placement decisions using both
engine capacity and reusable-state information.

## System boundaries

```text
Northbound       API and model semantics
                      |
                      v
InferFront        protocol adaptation
                  validation and normalization
                  model semantics
                  admission and policy
                  global request + state planner
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
Pingora type. Integrations depend inward on InferFront contracts.

## Responsibility split

### InferFront owns

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
- reusable-state discovery and materialization orchestration;
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

InferFront chooses an eligible target and supplies normalized work. It does not
dictate how the target batches or executes that work after admission.

### State providers own

- lookup of reusable model state;
- provider-specific state identity and metadata;
- locality and topology reporting;
- compatibility evidence;
- estimated transfer or materialization options;
- provider-specific preload, replication, and transfer operations;
- lifecycle and health of state managed by that provider.

The planner decides whether a provider operation is worthwhile. The provider
does not make the final request-placement decision.

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

Every engine adapter publishes a versioned `EngineCapabilities` document and
dynamic health/load snapshots. The registry separates relatively stable
features from rapidly changing capacity signals. Capability predicates decide
eligibility before candidates are costed.

### Global planner

The `Planner` evaluates eligible execution targets and state-materialization
options together. Its input includes:

- canonical request requirements;
- policy and admission constraints;
- engine capabilities, health, and load;
- state matches, reusable boundaries, and compatibility;
- estimated queue, prefill, materialization, decode, topology, and policy
  costs.

Its output is a `PlacementPlan` containing a target engine, optional state
actions, fallback conditions, and an explanation suitable for observability.
The output remains a plan until the orchestrator successfully reserves
capacity and prepares any required state.

### Engine adapters

`EngineAdapter` instances hide runtime-specific control protocols. They
discover capabilities, report health and load, accept canonical work, emit
canonical output events, and support cancellation. Optional operations such as
state attachment are gated by advertised capabilities.

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
| `TokenizerProvider` | Encode, decode, and expose tokenizer identity |
| `TemplateRenderer` | Render typed conversations with a versioned template |
| `ModelSemantics` | Compose model-specific normalization and parser behavior |
| `InputBundle` | Carry ordered token, multimodal, metadata, and future inputs |
| `EngineCapabilities` | Describe support and compatibility limits |
| `EngineAdapter` | Execute canonical work on one engine integration |
| `StateProvider` | Discover and materialize reusable model state |
| `RoutingPolicy` | Add constraints and policy cost without owning execution |
| `Planner` | Select request placement and state actions together |
| `PlacementPlan` | Record target, actions, fallbacks, and rationale |

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
5. The capability registry filters engines that cannot satisfy the request.
6. For eligible requests, the state provider returns reusable-state candidates
   and materialization estimates. With no provider, this is an empty result.
7. The planner jointly selects an engine and optional state actions.
8. The orchestrator reserves the target, materializes or attaches state when
   planned, and submits a canonical request through the engine adapter.
9. Canonical engine events flow through detokenization and incremental parsers,
   then through the northbound response adapter.
10. Completion, cancellation, or failure releases reservations and records the
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
semantic worker. Python code is not embedded in the normal gateway hot path.

## Initial implementation sequence

This sequence is directional rather than a commitment that features exist:

1. define core DTOs, semantic profiles, and structured errors;
2. implement one northbound protocol and a conformance test harness;
3. implement the canonical protocol and one engine adapter;
4. add capability-aware routing and admission;
5. add the null provider and generic `StateProvider` contract;
6. integrate NexusKV as an optional reference provider;
7. add cost calibration, topology, preload, and replication policies.

Each stage should keep the engine-neutral boundary testable with fake adapters
and providers before adding another production integration.
