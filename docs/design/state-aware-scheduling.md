# State-Aware Scheduling

## Status

This document defines the planner and state-provider abstractions. The Rust
workspace contains a deterministic cost-based planner, fake providers, the
state-import execution handshake, and an optional `locus-state-nexuskv` HTTP
bridge. The bridge maps versioned NexusKV match data into generic descriptors
and exercises lookup, estimate, prepare, materialize, commit, and execution in
tests. It is not a calibrated production scheduler, and physical NexusKV/GPU
transfer has not been validated.

## Principle

State locality and materialization cost are first-class placement inputs.

Locus must not first choose an engine using a conventional load balancer
and then opportunistically ask whether a cache happens to be present. For each
eligible target, the planner evaluates the execution path and reusable-state
options together.

A conceptual objective is:

```text
cost = queue_cost
     + unmatched_prefill_cost
     + state_materialization_cost
     + decode_cost
     + topology_cost
     + policy_cost
```

The terms are estimates in a comparable unit, initially expected latency or a
policy-weighted score. They are not raw values that can be added without
calibration.

## State is more than a token prefix

Longest-token-prefix matching is insufficient because reusable state may be:

- paged KV cache for a transformer;
- compressed or latent MLA state;
- recurrent or KDA checkpoints whose valid restore points are discrete;
- multimodal preprocessing results or model embeddings;
- encoder output or cross-attention state;
- future model-specific state with non-token coordinates.

A match therefore reports a typed artifact, structured input coverage, model
and semantic compatibility, placement, and the work required to use it. Token
prefix length may be one input to the estimate for one state kind, never the
generic interface.

## Core model

### State requirement

`StateRequirement` is derived from the canonical request and target
capabilities. It describes what could be reused without assuming that it
exists:

```text
StateRequirement
  model_identity
  relevant_semantic_requirements
  input_fingerprint
  canonical_token_ids (when authorized and required by a prefix index)
  input_structure
  accepted_state_kinds
  compatibility_constraints
  tenant_scope
  deadline
```

The input fingerprint may contain token-segment digests, media digests,
preprocessing identities, or other typed component identities. A provider sees
only the information authorized by tenant and privacy policy.

### State descriptor

A `StateDescriptor` returned by a provider includes:

- provider-scoped artifact identity and state kind;
- producer model identity and artifact-relevant semantic identity;
- execution compatibility attributes such as dtype, quantization, layout,
  adapter revision, and parallelism;
- a `ReusableBoundary` describing valid input coverage;
- present location or locations and topology metadata;
- size and materialization attributes;
- freshness, lifetime, ownership, and tenant scope;
- evidence source and confidence;
- a provider-private reference kept opaque to core logic.

State kinds use an extensible namespace, not a closed enum whose last value is
`KvCache`.

### Reusable boundary

`ReusableBoundary` describes exactly which portion of an `InputBundle` has
valid state and where execution may resume. It may use:

- ordered token positions within a named token segment;
- completed item or segment identities;
- a recurrent checkpoint number or logical step;
- a media item plus preprocessing stage;
- a provider-defined typed coordinate negotiated by capability.

```text
ReusableBoundary
  covered_components: repeated ComponentCoverage
  resume_coordinate: typed ResumeCoordinate
  completeness: complete | checkpointed | partial
  validation_digest
```

Coverage and resume coordinates are separate. A recurrent artifact can cover a
long input but allow resumption only from its latest valid checkpoint. A paged
cache match can identify shared pages while lacking the terminal page required
for a valid continuation. The provider must report the executable boundary, not
just the amount of data that appears to match.

### Compatibility

Compatibility is an explicit result:

```text
CompatibilityResult
  verdict: compatible | incompatible | unknown
  checked_constraints
  evidence
  required_conversion
```

Relevant constraints vary by state kind and can include:

- immutable model weights and adapter/LoRA revision;
- tokenizer and chat-template behavior;
- input and media-preprocessing identity;
- state representation version;
- dtype, quantization, attention layout, and parallel decomposition;
- positional encoding and sequence-coordinate rules;
- engine or kernel ABI;
- tenant and security scope.

Compatibility is artifact-specific. KV state typically depends on model
execution identity, relevant input semantics, state layout, positional
semantics, and runtime compatibility; it does not normally depend on a
reasoning parser. A prepared vision artifact depends on model identity, media
digest, and preprocessing identity. Tool-parser compatibility belongs to an
output semantic profile rather than a prefill-state check.

Unknown required evidence is incompatible for planning purposes. A conversion
is represented as a materialization option with cost and target compatibility,
not as a claim that the original artifact is directly compatible.

## `StateProvider` abstraction

The conceptual asynchronous interface is:

```rust,ignore
#[async_trait]
pub trait StateProvider: Send + Sync {
    fn identity(&self) -> &StateProviderIdentity;
    fn capabilities(&self) -> &StateProviderCapabilities;

    async fn lookup(
        &self,
        requirement: &StateRequirement,
        context: &OperationContext,
    ) -> Result<Vec<StateDescriptor>, StateProviderError>;

    async fn estimate(
        &self,
        state: &StateDescriptor,
        target: &ExecutionTarget,
        context: &OperationContext,
    ) -> Result<Vec<MaterializationOption>, StateProviderError>;

    async fn materialize(
        &self,
        option: &MaterializationOption,
        target: &StateImportTarget,
        context: &OperationContext,
    ) -> Result<TransferReceipt, StateProviderError>;

    async fn preload(
        &self,
        request: PreloadRequest,
        context: &OperationContext,
    ) -> Result<StateOperation, StateProviderError>;

    async fn replicate(
        &self,
        request: ReplicationRequest,
        context: &OperationContext,
    ) -> Result<StateOperation, StateProviderError>;
}
```

The eventual API may use streaming results or split read and mutation traits.
Important contract properties are:

- lookup does not authorize placement or state movement;
- estimates are target-specific, timestamped, and uncertainty-aware;
- mutations require explicit `PlanExecutor` action encoded by a plan;
- materialization transfers to a negotiated target and returns a receipt; it
  does not allocate engine-local memory or produce the final attachment;
- cancellation and deadlines apply to every operation;
- no provider is required.

A `NullStateProvider` returns no matches and rejects mutation operations as
unsupported. This makes cold, state-unaware operation part of normal contract
testing.

## State import handshake

Real runtime state often needs destination pages, device memory, layout, or an
import handle before transfer begins. Locus therefore uses a coordinated
transaction rather than `StateProvider.materialize -> PreparedStateAttachment`:

```text
PlanExecutor
  |
  |-- EngineAdapter.prepare_state_import(StateImportSpec)
  |     -> StateImportTarget
  |
  |-- StateProvider.materialize(MaterializationOption, StateImportTarget)
  |     -> TransferReceipt
  |
  |-- EngineAdapter.commit_state_import(StateImportTarget, TransferReceipt)
  |     -> PreparedStateAttachment
  |
  `-- EngineAdapter.execute(..., PreparedStateAttachment)
```

The provider owns discovery, source identity, and movement. The engine adapter
owns destination allocation, runtime layout, install, and bind. `PlanExecutor`
owns ordering, deadlines, cancellation, abort, cleanup, fallback, and bounded
replanning.

`StateImportTarget` is opaque, namespaced, target-generation scoped, and
expiring. `TransferReceipt` proves what the provider transferred without
exposing provider-private objects. `PreparedStateAttachment` is created only
after the adapter validates and commits the import. A partial import is aborted
idempotently; stale generations fail closed.

Provider and adapter may negotiate a transport namespace, but engine-private
allocation objects never enter the provider API and provider-private source
objects never enter the adapter API. NexusKV remains one optional provider.

## NexusKV integration

NexusKV is the reference `StateProvider`. The optional
`locus-state-nexuskv` crate uses a versioned `locus.nexuskv-bridge.v1` HTTP
contract. Lookup requests carry canonical token IDs plus structured model and
input-semantic identities. A hit is accepted only when the bridge echoes the
identities it validated and returns `nexuskv.contract.v1` data; missing or
mismatched evidence fails closed.

The bridge returns target-specific estimates and performs materialization only
after `EngineAdapter.prepare_state_import` supplies a generation-scoped sink.
It returns a receipt that the adapter must commit before execution. The Planner
still sees only immutable candidates and remains side-effect free.

The current NexusKV repository has the underlying JSON contract, radix lookup,
and execution/transfer abstractions, but it does not expose the bridge HTTP
operations. A deployment must supply that bridge service. Locus's tests use a
protocol double, so they prove contract mapping and ownership, not live
NexusKV transport or device attachment.

The integration lives outside Locus core and depends on Locus's
provider API. Locus core does not import the NexusKV SDK, use NexusKV
identifiers as domain types, or assume NexusKV deployment. Another provider can
implement the same contract, and a deployment can run without any provider.

NexusKV-specific capabilities may use namespaced extensions. Portable
planner behavior depends only on the generic descriptor, compatibility,
boundary, location, and cost contracts.

## State-provider decision rationale

Reusable state can be KV cache, MLA state, a recurrent/KDA checkpoint, a
multimodal artifact, encoder output, or a future representation. Its valid
resume point may not be a token offset, and a matching hash or token prefix does
not prove that execution can continue. A generic provider contract is therefore
necessary for typed boundaries, compatibility evidence, locations, and
target-specific materialization options.

The abstraction adds schema and lifecycle complexity. Provider estimates can
be stale, state installation crosses provider and engine ownership, and
extensions must avoid becoming an unstructured escape channel. These costs are
handled through evidence-bearing descriptors, expiring attachments,
calibration, reservations, and explicit fallbacks.

The following alternatives were rejected:

- **Depend directly on NexusKV:** this would make one reference integration a
  core deployment requirement and prevent independent providers.
- **Add cache affinity after engine routing:** this cannot compare queue,
  transfer, recompute, decode, topology, and policy costs as complete paths.
- **Standardize longest-token-prefix lookup:** this cannot represent recurrent
  checkpoints, multimodal state, incomplete pages, or non-token boundaries.
- **Let the provider allocate runtime destination memory:** this leaks
  engine-private page and layout ownership into the provider contract.
- **Let the provider choose the engine:** the provider does not own admission,
  engine load, decode cost, tenant fairness, or global topology policy.

## Planner and executor ownership

The planner compares complete feasible paths and returns a `PlacementPlan`.
It does not call provider mutation methods, allocate destination memory, reserve
an engine, submit work, retry, or clean up partial imports.

`PlanExecutor` performs those side effects according to the chosen plan. It may
apply the encoded fallback or request bounded replanning when observations have
changed. It cannot silently turn a failed path into an unplanned placement.

## Planning inputs

For each admitted request, planning begins with:

- immutable normalized request and state requirement;
- capability-eligible execution targets;
- fresh engine load snapshots;
- state descriptors and materialization options;
- topology graph and transfer observations;
- request deadline, priority, tenant, and policy constraints;
- calibrated cost-model parameters and confidence bounds.

Capability and correctness constraints filter candidates before cost
comparison. A low estimated cost cannot make an incompatible state or engine
eligible.

## Candidate plans

A candidate plan is a complete feasible path, for example:

- cold combined prefill/decode on engine A;
- use state already local to engine B, then continue there;
- transfer a checkpoint from host C to engine A, then continue;
- recompute unmatched prefill on engine D instead of transferring state;
- prefill on P, hand off compatible state, and decode on D.

Each candidate records:

- target stage or stages and reservations required;
- selected state and reusable boundary, if any;
- materialization or handoff actions;
- expected cost terms and uncertainty;
- deadline feasibility;
- fallback on reservation or state-action failure;
- an explanation of constraints and trade-offs.

The planner compares complete paths. It does not rank state matches independently
of the engines on which they can be used.

## Cost terms

### Queue cost

Estimated time until execution can begin, derived from fresh engine snapshots,
request shape, scheduling class, and recent service rates. Queue depth alone is
not comparable across engines or request sizes.

### Unmatched prefill cost

Expected compute for the input not covered by the executable reusable boundary.
It accounts for input structure, model architecture, target throughput, and any
required reconstruction. It is not simply `unmatched_tokens * constant`.

### State materialization cost

Expected time and resource pressure for lookup completion, transfer,
conversion, decompression, device installation, or attachment. An already-local
artifact can still have a nonzero binding cost.

### Decode cost

Expected generation cost using requested output limits, sampling mode, target
decode throughput, and execution role. It is uncertain because generated
length is unknown; policy selects an estimator or percentile.

### Topology cost

Penalties for network links, failure domains, data residency, prefill/decode
handoff, accelerators, and contention not already represented by transfer time.
Hard topology constraints filter rather than penalize candidates.

### Policy cost

Explicit, auditable preferences such as tenant affinity, fairness debt, energy,
monetary cost, provider quotas, or stability. Policy cost cannot override a
correctness or authorization constraint.

## Comparing transfer and recompute

For compatible state and a target, the planner compares at least:

```text
reuse_path = queue_with_reuse
           + materialization
           + unmatched_prefill
           + decode
           + topology_and_policy

cold_path  = queue_cold
           + full_prefill
           + decode
           + topology_and_policy
```

The comparison includes uncertainty and deadline risk. Transfer is not chosen
merely because a match exists. Recompute can win for small reusable regions,
slow or congested links, format conversion, short deadlines, or an otherwise
idle target.

The planner records the chosen and rejected alternatives so observed outcomes
can recalibrate estimates.

## Cache-aware routing

Cache-aware routing is the special case where reusable state is already located
on, or cheaply attachable to, a candidate engine. The same general algorithm
handles it:

1. query compatible state for the normalized request;
2. obtain executable reuse boundaries, not just hash matches;
3. estimate paths for each eligible engine and state option;
4. compare against cold and transfer/recompute alternatives;
5. reserve and revalidate the selected path.

Affinity is therefore conditional on total expected cost and feasibility, not
a rule to always route to the longest match.

## Preload, warming, and replication

Preload and replication are planned background actions with budgets. They are
not implicit side effects of lookup.

Policies may use expected demand, artifact size, topology, eviction pressure,
and cost avoided to propose actions. Every action has:

- target state and location;
- tenant and authorization scope;
- resource budget and priority below foreground work;
- expiry or invalidation condition;
- estimated benefit and cost;
- observable outcome.

Speculative warming must not reserve unbounded memory, amplify a hot artifact
across every engine, or move tenant state across prohibited boundaries.

## Topology and PD-aware placement

Topology is expressed as data: nodes, execution roles, state locations, links,
failure domains, and observed transfer properties. Core planning does not rely
on one deployment's host naming or one engine's disaggregation protocol.

For prefill/decode disaggregation, a candidate plan includes:

- prefill target and its queue/compute cost;
- state produced or reused at the prefill boundary;
- compatible handoff method and transfer/materialization cost;
- decode target and queue/decode cost;
- failure and fallback behavior across both stages.

A reusable recurrent checkpoint or MLA state can participate if its provider
and adapters describe a compatible handoff. The planner does not assume a
conventional KV representation.

## Reservations and races

Lookup and snapshots are observations, not reservations. Between planning and
execution, an engine can fill, an artifact can expire, or link cost can change.

`PlanExecutor` uses this sequence:

1. select a plan from bounded-fresh inputs;
2. reserve target capacity where supported;
3. revalidate state, target identity, and engine generation;
4. ask the engine adapter to prepare an expiring import target;
5. ask the state provider to materialize into that target;
6. ask the engine adapter to commit the receipt or abort partial state;
7. submit execution with the committed attachment;
8. apply the encoded fallback or bounded replan on failure.

Plans carry a short validity horizon. Replanning is bounded to avoid livelock.
State attachments are scoped and expiring so stale handles fail closed.

## Admission control interaction

Admission precedes expensive state movement. It constrains planning with
priority, tenant budgets, maximum queue/materialization delay, and overload
policy. A warm state hit does not bypass admission or fairness.

Admission may reserve separate budgets for foreground execution and background
preload/replication. Provider degradation can trigger a cold-only mode without
changing tenant authorization.

## Provider failure behavior

State is an optimization unless a request explicitly requires a prepared
artifact. Failure policy is explicit:

- lookup timeout: consider cold candidates if allowed;
- estimate unavailable: do not treat the option as zero-cost;
- compatibility unknown: reject that state candidate;
- materialization failure: apply the plan's cold/replan/fail fallback;
- stale attachment: never submit it as compatible;
- provider-wide outage: open a circuit and plan cold while policy permits.

Fallback decisions still respect the original deadline, residency, capability,
and admission constraints.

## Observability and calibration

For each decision, record redacted structured data:

- candidate targets and exclusion reasons;
- state kind, location, compatibility verdict, and reusable boundary;
- every estimated cost term, confidence, and data freshness;
- selected actions and fallback;
- actual queue, materialization, prefill, decode, and transfer timing;
- whether state was accepted and the boundary actually used;
- cancellation or failure stage.

Comparing estimates with outcomes supports per-engine, per-model, per-state-kind,
and per-topology calibration. Planner versions and parameter revisions are part
of the trace so decisions remain explainable.

Prompts, raw token sequences, media, provider-private handles, and tenant data
are excluded unless an explicit privacy policy authorizes them.

## Initial planner strategy

The first implementation should favor a transparent, bounded candidate scorer
over a complex optimizer:

1. filter by hard capability, policy, and compatibility constraints;
2. generate cold plus a bounded number of best state options per target;
3. compute calibrated cost and conservative deadline feasibility;
4. choose the lowest-cost feasible plan with deterministic tie-breaking;
5. expose a complete decision explanation;
6. learn calibration parameters from observations without allowing an online
   learner to bypass hard constraints.

More advanced optimization can replace the scorer behind `Planner` after trace
data demonstrates a need.

## Validation

The state and planner contract is tested with fake providers and engines, and
the NexusKV bridge path is tested with a protocol double. The broader
conformance matrix includes:

- equal token prefix with incompatible model or semantic identity;
- matched pages without a valid terminal checkpoint;
- recurrent checkpoint with a non-token resume coordinate;
- multimodal state with changed preprocessing or media digest;
- local short match versus remote long match;
- transfer slower than recompute;
- PD handoff with incompatible layout;
- missing, stale, or low-confidence cost observations;
- provider timeout and materialization failure fallbacks;
- prepare, transfer, commit, abort, and partial-import cleanup;
- tenant isolation and topology restrictions;
- engine restart invalidating import targets and prepared attachments;
- planner purity and `PlanExecutor` side-effect ownership;
- deterministic choice and explanation for equal costs.

Passing these tests validates decision semantics, not production cost accuracy.
