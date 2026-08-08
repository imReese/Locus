# ADR 0002: Use an Optional State-Provider Abstraction

- Status: Accepted
- Date: 2026-08-08

## Context

Reusable state includes paged KV, MLA, recurrent/KDA checkpoints, multimodal
artifacts, and future non-token forms. Discovery and movement belong to state
systems, while device-local allocation and installation belong to runtimes.

## Decision

`StateProvider` reports typed state, compatibility evidence, reusable
boundaries, and materialization options. NexusKV is the reference integration,
not a core dependency.

State import is coordinated by `PlanExecutor`: `EngineAdapter` prepares an
expiring target-generation-scoped destination, `StateProvider` transfers into
it, and `EngineAdapter` commits or aborts the import. Provider-private and
engine-private objects remain behind opaque negotiated handles.

## Consequences

The contract is more complex than prefix lookup and requires transaction
cleanup, generation fencing, calibrated estimates, and explicit fallback.
Deployments may use another provider or the null provider.

## Rejected alternatives

- depend directly on NexusKV in core;
- add cache affinity only after engine routing;
- standardize longest-token-prefix matching;
- let the provider allocate runtime memory or choose the engine.
