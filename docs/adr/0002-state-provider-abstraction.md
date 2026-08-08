# ADR 0002: Make Reusable State a Provider Abstraction

- Status: Accepted
- Date: 2026-08-08
- Scope: Design; implementation has not started

## Context

Global placement should account for reusable model state. That state may be a
KV cache, MLA representation, recurrent/KDA checkpoint, multimodal artifact,
encoder output, or a future form. Its valid resume boundary may not be a token
offset, and a hash or longest-prefix match does not prove that execution can
continue from it.

[NexusKV](https://github.com/imReese/NexusKV) is a natural reference system for
state discovery and movement, but making it a core dependency would tie
InferFront's architecture and deployment lifecycle to one implementation.

## Decision

InferFront will define a generic, optional `StateProvider` abstraction.

A provider reports typed state descriptors, structured reusable boundaries,
compatibility evidence, locations, and target-specific materialization options.
It performs lookup, materialization, preload, and replication only through
explicit operations. The global planner compares complete request-and-state
placement paths and makes the final decision.

NexusKV will be developed as the reference integration outside core. Other
providers may implement the same contract. A null provider is a supported
configuration and participates in contract tests.

Provider-private handles and SDK types do not enter core domain types. A
successful materialization produces a short-lived, engine-scoped prepared
attachment that a capable engine adapter can validate and bind.

## Consequences

### Positive

- State locality and materialization cost are first-class planner inputs.
- The abstraction covers non-token and checkpointed state.
- InferFront core can run without NexusKV or any cache service.
- Providers and engine adapters can evolve independently behind versioned
  contracts.
- Compatibility and resume boundaries fail closed when evidence is missing.

### Costs and risks

- A portable descriptor and cost contract is more complex than prefix lookup.
- Provider estimates are stale and uncertain, so planning requires calibration,
  reservations, and fallback behavior.
- State installation crosses provider and engine ownership and needs a strict
  attachment lifecycle.
- The abstraction must avoid either a lowest-common-denominator design or an
  unstructured provider-extension channel.

## Rejected alternatives

### Depend directly on NexusKV in core

This would prevent independent providers and make the state-free deployment
path artificial. NexusKV remains the reference implementation instead.

### Add cache affinity as a later routing policy

Choosing an engine before considering state cannot compare transfer, recompute,
queue, decode, and topology costs as complete alternatives.

### Standardize only longest-token-prefix lookup

This cannot represent recurrent checkpoints, multimodal artifacts, incomplete
paged matches, or non-token resume coordinates.

### Let the state provider choose the engine

The provider does not own admission, engine load, decode cost, tenant fairness,
or all topology policy. It supplies evidence and operations to the planner.

## Follow-up

[State-aware scheduling](../design/state-aware-scheduling.md) defines the
conceptual provider and planner contracts. Implementation must validate them
with fake providers and non-token test cases before treating a NexusKV adapter
as evidence that the abstraction is generic.
