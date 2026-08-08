# Engine Adapter Contract

## Status

This is the proposed contract for engine integrations. It defines ownership and
behavior, not a stable Rust API. No SGLang, vLLM, TensorRT-LLM, or other adapter
is implemented yet.

## Purpose

An engine adapter translates between Locus's canonical engine protocol and
one execution runtime. It prevents runtime-specific protocols, configuration,
and error behavior from leaking into model semantics or global planning.

Adapters are capability-based. Implementing the trait does not imply that an
engine supports every canonical feature. Each adapter must accurately describe
what a particular engine instance and model deployment can execute.

## Conceptual interface

```rust,ignore
#[async_trait]
pub trait EngineAdapter: Send + Sync {
    fn instance(&self) -> &EngineInstance;

    async fn execution_targets(
        &self,
        context: &OperationContext,
    ) -> Result<Vec<ExecutionTarget>, AdapterError>;

    async fn capabilities(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Versioned<EngineCapabilities>, AdapterError>;

    async fn snapshot(
        &self,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<EngineSnapshot, AdapterError>;

    async fn prepare_state_import(
        &self,
        target: &ExecutionTarget,
        spec: &StateImportSpec,
        context: &OperationContext,
    ) -> Result<StateImportTarget, AdapterError>;

    async fn commit_state_import(
        &self,
        import: &StateImportTarget,
        receipt: &TransferReceipt,
        context: &OperationContext,
    ) -> Result<PreparedStateAttachment, AdapterError>;

    async fn abort_state_import(
        &self,
        import: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<(), AdapterError>;

    async fn execute(
        &self,
        target: &ExecutionTarget,
        request: CanonicalRequest,
        state: Option<PreparedStateAttachment>,
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
into separate traits. All methods operate on Locus-owned domain types.
Generated RPC messages and engine SDK types remain inside the adapter crate.

## Engine instances and execution targets

`EngineInstance` identifies a runtime process or service instance. It includes:

- a stable adapter kind and deployment/instance identity;
- engine and adapter versions;
- a generation that changes across incompatible restarts;
- topology, hardware, and health-endpoint information.

It does not contain one permanent model revision. A multi-model process may
load or unload several targets during the same instance generation.

`ExecutionTarget` is the planner-selectable unit. It references an engine
instance and generation, then identifies:

- immutable model and optional adapter/LoRA revisions;
- execution role such as prefill, decode, or combined;
- parallel layout and execution profile;
- current residency and target-specific capability identity.

Core code does not branch on adapter-kind strings. It filters and selects
execution targets by requirements, compatibility, policy, and cost.

## Capability discovery

Capabilities are target-specific. They are obtained from the engine where
possible and supplemented by adapter knowledge where necessary. They are
evidence-bearing and have a validity interval. An adapter invalidates them
after runtime configuration, model residency, adapter, or execution-profile
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

`EngineSnapshot` reports rapidly changing target observations used by the
planner:

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

Before calling an adapter, Locus has already:

1. normalized northbound semantics;
2. constructed a canonical request;
3. checked declared capabilities;
4. selected an execution target in a placement plan.

For a cold request, `PlanExecutor` calls the adapter directly. For a reuse plan,
it first completes the import handshake described below. The adapter then:

1. revalidates protocol version and required capabilities;
2. translates canonical input and parameters without changing their meaning;
3. validates and binds a committed state attachment when present;
4. submits exactly one logical execution;
5. maps runtime output to ordered canonical events;
6. maps runtime completion or failure to one terminal event;
7. propagates cancellation and deadlines.

Adapters may split or coalesce transport frames but must preserve event
ordering. They report execution facts and may not silently drop sampling
parameters, output tokens, multimodal inputs, or finish information. Tool-call,
reasoning, and application finish semantics remain in `ModelSemantics` unless a
separate optional engine capability is explicitly selected.

## State import transaction

The provider owns state discovery, source identity, and movement. The adapter
owns the runtime destination, device-local allocation, install, and bind.
`PlanExecutor` coordinates the transaction:

1. `prepare_state_import` validates the target generation and allocates or
   reserves a destination. It returns an expiring `StateImportTarget` with an
   opaque, negotiated sink handle.
2. The state provider transfers the planned source to that target and returns
   a `TransferReceipt`.
3. `commit_state_import` validates the receipt, completeness, compatibility,
   and current generation, then returns a `PreparedStateAttachment`.
4. `execute` accepts only an attachment scoped to the selected target and
   current generation.

`abort_state_import` is idempotent and releases partial allocation after
transfer failure, timeout, cancellation, stale generation, commit failure, or
executor fallback. Import targets and attachments expire. A restart fails
closed and invalidates both.

Provider-private source objects and engine-private allocation/layout objects do
not cross core APIs. Opaque handles are allowed only with an explicit namespace
and scope understood by the participating adapter and provider.

Binding failure is explicit. `PlanExecutor` follows the fallback encoded in the
placement plan: cold execution on the same target, bounded replanning, or
request failure. Neither adapter nor provider silently chooses another target
or state artifact.

This division allows NexusKV or another provider to manage movement while the
runtime retains device-local allocation and installation. Engines that cannot
accept external state omit the capability.

## Engine and semantic finish reasons

Adapters normalize runtime termination into `EngineFinishReason`: execution
facts such as stop, length, cancellation, error, or a namespaced
runtime-specific value. They do not infer tool calls, reasoning completion,
content filtering, or another application outcome by default.

`ModelSemantics` consumes the ordered engine output and derives
`SemanticFinishReason`. If a runtime provides semantic events itself, the
adapter advertises that optional capability and Locus selects it knowingly.

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

Adapters preserve enough information for Locus to distinguish:

- invalid canonical requests or violated preconditions;
- capability drift between planning and submission;
- engine overload or unavailability;
- stale target generation or expired state-import handle;
- state-import prepare, commit, or abort failure;
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
- restart with a new generation invalidates targets, reservations, import
  handles, and prepared state attachments for the old generation;
- model load or unload changes the available `ExecutionTarget` set without
  conflating model identity with process identity.

Static configuration can register an adapter initially. A later control plane
may support dynamic discovery without changing the execution contract.

## Prefill/decode disaggregation

Disaggregated prefill and decode are modeled as target roles and topology, not
as hard-coded SGLang concepts. Execution targets advertise roles and compatible
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
- error, engine-finish, usage, and cancellation mapping;
- absence of application-semantic finish reasons in default events;
- prepare/materialize/commit/abort lifecycle and cleanup;
- stale generation and incompatible import/attachment rejection;
- duplicate IDs and retry safety;
- draining, restart, and health transitions;
- absence of adapter-specific types in the core interface.

Integration-specific tests supplement rather than replace contract tests.
