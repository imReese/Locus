# NexusKV State Provider Bridge

## Status

`locus-state-nexuskv` is an optional network `StateProvider`. It implements the
Locus side of a versioned bridge and is covered by an end-to-end protocol test
through planning and the prepare/materialize/commit handshake. The current
NexusKV repository does not ship these HTTP endpoints, and no physical cache
transfer or GPU attachment has been validated.

## Contract

The configurable bridge exposes:

- `POST /locus/v1/lookup`
- `POST /locus/v1/estimate`
- `POST /locus/v1/materialize`

Every envelope uses `locus.nexuskv-bridge.v1`; lookup payloads and results name
the upstream `nexuskv.contract.v1` schema. Authentication is an optional bearer
token configured on the provider.

Lookup sends the tenant, namespace, immutable model revision, engine family,
state semantic type, canonical token IDs, request input fingerprint, and
structured Locus model/input-semantic identities. The bridge must evaluate the
NexusKV index and echo the exact identities it validated. Locus
rejects missing or mismatched evidence, schema drift, model mismatch, zero
coverage, provider incompatibility, and unsupported state kinds.

Estimate returns one target- and engine-generation-specific materialization
option with locality, topology, cost, and an opaque bridge handle.
Materialization receives that handle only after the engine adapter creates a
short-lived import sink. Its receipt is committed by the engine adapter before
execution.

## Ownership

The bridge owns NexusKV lookup, source locators, estimates, and transfer. The
engine adapter owns destination allocation and attachment. `PlanExecutor` owns
operation ordering, abort, cleanup, and fallback. `Planner` receives immutable
generic candidates and never performs HTTP calls or mutations.

The HTTP bridge is deliberately outside `locus-state`, so core contracts do
not depend on NexusKV or Reqwest types. Another provider can implement the same
generic `StateProvider`, and `NullStateProvider` remains a normal deployment.
