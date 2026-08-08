# Engine Adapter Contract

## Status

This is the proposed contract for engine integrations. It defines ownership and
behavior, not a stable Rust API. No SGLang, vLLM, TensorRT-LLM, or other adapter
is implemented yet.

## Purpose

An engine adapter translates between InferFront's canonical engine protocol and
one execution runtime. It prevents runtime-specific protocols, configuration,
and error behavior from leaking into model semantics or global planning.

Adapters are capability-based. Implementing the trait does not imply that an
engine supports every canonical feature. Each adapter must accurately describe
what a particular engine instance and model deployment can execute.

## Conceptual interface

```rust,ignore
#[async_trait]
pub trait EngineAdapter: Send + Sync {
    fn identity(&self) -> &EngineIdentity;

    async fn capabilities(
        &self,
        context: &OperationContext,
    ) -> Result<Versioned<EngineCapabilities>, AdapterError>;

    async fn snapshot(
        &self,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, AdapterError>;

    async fn execute(
        &self,
        request: CanonicalRequest,
        context: OperationContext,
    ) -> Result<EngineEventStream, AdapterError>;

    async fn cancel(
        &self,
        request_id: &RequestId,
        reason: CancellationReason,
        context: &OperationContext,
    ) -> Result<CancelResult, AdapterError>;
}
```

The eventual Rust interface may split discovery, observation, and execution
into separate traits. All methods operate on InferFront-owned domain types.
Generated RPC messages and engine SDK types remain inside the adapter crate.

## Adapter identity

`EngineIdentity` identifies a logical deployment and a concrete instance. It
includes:

- a stable adapter kind, such as an integration namespace;
- deployment and instance identities;
- engine software version and adapter version;
- model artifact revision;
- topology location and execution role;
- a generation number that changes across incompatible restarts.

Core code does not branch on known adapter-kind strings. Selection is based on
capabilities, policy, and cost.

## Capability discovery

Capabilities are obtained from the engine where possible and supplemented by
adapter knowledge where necessary. They are evidence-bearing and have a
validity interval. An adapter must invalidate them after configuration or model
changes.

Capability categories include:

- model, architecture, dtype, quantization, and parallel layout;
- canonical protocol versions;
- input item and relation kinds;
- context and output limits;
- sampling, grammar, log-probability, and candidate support;
- streaming granularity and usage accounting;
- cancellation and idempotency guarantees;
- prepared-state kinds, attachment namespaces, and compatibility limits;
- prefill, decode, or combined execution roles;
- adapter-specific required extensions.

A claim answers whether an operation is supported under declared constraints.
It is not a health signal and not a best-effort guess. Unknown support is
reported as unknown and treated as ineligible for required features.

## Dynamic engine snapshots

`EngineSnapshot` reports rapidly changing observations used by the planner:

- health and readiness;
- queue depth and estimated queue delay;
- active request and token load;
- memory pressure and available capacity where reliably observable;
- recent prefill and decode throughput estimates;
- topology or role availability;
- observation time, source, and confidence.

Missing metrics remain missing; adapters do not substitute zero. The planner
uses freshness limits and conservative defaults. Snapshots are advisory because
state may change before execution.

## Request execution

Before calling an adapter, InferFront has already:

1. normalized northbound semantics;
2. constructed a canonical request;
3. checked declared capabilities;
4. selected the engine in a placement plan;
5. prepared any selected reusable-state attachment.

The adapter then:

1. revalidates protocol version and required capabilities;
2. translates canonical input and parameters without changing their meaning;
3. binds a prepared state attachment when present;
4. submits exactly one logical execution;
5. maps runtime output to ordered canonical events;
6. maps runtime completion or failure to one terminal event;
7. propagates cancellation and deadlines.

Adapters may split or coalesce transport frames but must preserve semantic event
ordering. They may not silently drop sampling parameters, tool-related tokens,
multimodal inputs, or finish information.

## Prepared state

The state provider discovers and materializes reusable state; the adapter owns
the final runtime-specific binding needed to execute with it.

An adapter advertises a set of supported attachment namespaces and state kinds.
A prepared attachment is scoped to a target engine generation and expires. The
adapter validates:

- target instance and generation;
- model and semantic compatibility fingerprints;
- state kind and runtime layout;
- reusable input boundary;
- tenant scope and attachment lifetime.

Binding failure is explicit. InferFront follows the fallback encoded in the
placement plan: cold execution on the same engine, replanning elsewhere, or
request failure. The adapter does not decide to use a different state artifact
on its own.

This division allows NexusKV or another provider to manage discovery and
movement while the runtime integration retains control of device-local state
installation. Engines that cannot accept external state simply omit the
capability.

## Runtime-specific features

A runtime-specific feature may be exposed through a namespaced extension when:

- its schema and owner are explicit;
- capability discovery declares support;
- the request identifies the extension as optional or required;
- core planning remains correct without interpreting its opaque payload;
- it does not redefine a canonical field with different semantics.

Once multiple adapters need the same behavior, the feature should be evaluated
for promotion into the canonical contract. Core types must not accumulate an
open-ended set of SGLang or vLLM fields.

## Error mapping

Adapters preserve enough information for InferFront to distinguish:

- invalid canonical requests or violated preconditions;
- capability drift between planning and submission;
- engine overload or unavailability;
- deadline and cancellation outcomes;
- execution failure after acceptance;
- adapter protocol violations;
- authentication or authorization failure;
- internal translation bugs.

An `AdapterError` includes a stable class, retry hint, engine acceptance state,
safe message, and optional private diagnostic metadata. Raw engine messages may
be recorded under redaction policy but are not stable client-facing errors.

## Concurrency, deadlines, and backpressure

Adapters must be safe for concurrent calls. Each operation carries a deadline
and cancellation context. Implementations must not detach work that survives
the request without an explicitly managed lifecycle.

Output uses a bounded asynchronous stream. If the consumer disappears, the
adapter initiates cancellation. Transport and engine buffer sizes are
observable and bounded to prevent one slow stream from exhausting the process.

## Registration and lifecycle

An adapter instance moves through explicit states:

```text
discovered -> probing -> ready -> draining -> stopped
                    \-> unhealthy -/
```

- `probing` instances do not receive normal traffic;
- `ready` requires valid capabilities and a fresh health observation;
- `unhealthy` instances may continue probes but are ineligible for placement;
- `draining` accepts no new work but allows active requests to finish or reach
  their deadline;
- restart with a new generation invalidates reservations and prepared state
  attachments for the old generation.

Static configuration can register an adapter initially. A later control plane
may support dynamic discovery without changing the execution contract.

## Prefill/decode disaggregation

Disaggregated prefill and decode are modeled as capabilities and topology, not
as hard-coded SGLang concepts. Engine instances advertise roles and compatible
handoff mechanisms. The planner may produce a multi-stage placement plan when:

- the request permits the additional latency and failure surface;
- prefill and decode roles share a compatible state handoff;
- the state provider or adapter can estimate and execute that handoff;
- both stages meet model, semantic, and topology constraints.

Each execution engine still owns local scheduling within its stage. The global
planner owns selection of stages and the cost of crossing between them.

## Conformance requirements

Every adapter is tested against the same fake-core harness. Required tests
cover:

- truthfulness and invalidation of capability claims;
- canonical request translation for every advertised input kind;
- exact sampling/default semantics;
- stream ordering, terminal events, and backpressure;
- error, finish-reason, usage, and cancellation mapping;
- stale generation and incompatible state attachment rejection;
- duplicate IDs and retry safety;
- draining, restart, and health transitions;
- absence of adapter-specific types in the core interface.

Integration-specific tests supplement rather than replace contract tests.
