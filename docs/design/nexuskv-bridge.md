# NexusKV State Store Bridge

## Status

`locus-store-nexuskv` is an optional network `StateStore`. It implements the
Locus side of a versioned bridge and is covered by an end-to-end protocol test
through planning and the prepare/materialize/commit handshake. NexusKV now
ships the corresponding HTTP service, and cross-repository E2E starts it as a
separate process backed by the real Rust matcher. No physical cache transfer,
native engine state import, or GPU attachment has been validated.

## Contract

The configurable bridge exposes:

- `POST /locus/v1/lookup`
- `POST /locus/v1/estimate`
- `POST /locus/v1/materialize`

Every envelope uses `locus.nexuskv-bridge.v1`; lookup payloads and results name
the upstream `nexuskv.contract.v1` schema. Authentication is an optional bearer
token configured on the store. The JSON Schema source of truth lives in
NexusKV at `schema/locus_nexuskv_bridge.json`.

Lookup sends the tenant, namespace, immutable model revision, engine family,
state semantic type, canonical token IDs, request input fingerprint, and
structured Locus model/input-semantic identities. The bridge must evaluate the
NexusKV index and echo the exact identities it validated. Locus
rejects missing or mismatched evidence, schema drift, model mismatch, zero
coverage, store incompatibility, and unsupported state kinds. A successful
lookup includes an opaque source capability; Locus must return it during
estimate, so knowledge of a state ID is insufficient.

Estimate returns one target- and engine-generation-specific materialization
option with locality, topology, cost, and an opaque bridge handle.
Materialization receives that handle only after the engine adapter creates a
short-lived import sink. Its receipt is committed by the engine adapter before
execution. Protocol-only receipts must report zero transferred bytes,
`physical_transfer_verified=false`, and the
`nexuskv.protocol-transfer-receipt.v1` namespace. Locus rejects internally
inconsistent evidence.

## Ownership

The bridge owns NexusKV lookup, source locators, estimates, and transfer. The
engine adapter owns destination allocation and attachment. `PlanExecutor` owns
operation ordering, abort, cleanup, and fallback. `Planner` receives immutable
generic candidates and never performs HTTP calls or mutations.

The HTTP bridge is deliberately outside `locus-store`, so core contracts do
not depend on NexusKV or Reqwest types. Another store can implement the same
generic `StateStore`, and `NullStateStore` remains a normal deployment.

The conformance fixture is vendored byte-identically in both repositories. Run
the real cross-process gate from a Locus checkout with a sibling NexusKV
checkout:

```bash
python scripts/nexuskv_bridge_e2e.py --nexuskv-root ../NexusKV
```

This gate proves the process boundary, Rust matcher, Locus placement path, and
prepare/materialize/commit ordering. It deliberately does not claim GPU import,
physical movement, or production performance.
