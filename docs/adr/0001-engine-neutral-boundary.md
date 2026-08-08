# ADR 0001: Establish an Engine-Neutral Boundary

- Status: Accepted
- Date: 2026-08-08

## Context

Inference runtimes expose different request types, stream events, sampling
fields, and errors. Using one runtime API inside core would couple model
semantics and global planning to that engine.

## Decision

Locus owns a canonical request and execution-event model. Northbound protocols
normalize into Locus types; capability-based `EngineAdapter` integrations
translate those types to runtimes. `InputBundle` remains extensible beyond
token IDs.

The planner chooses an `ExecutionTarget`, not an engine-specific object.
Framework, generated transport, and runtime SDK types remain at integration
edges.

## Consequences

Adapters require translation and conformance tests. Namespaced extensions may
expose runtime-specific capabilities without redefining canonical semantics.
Engine-local scheduling and execution remain under engine control.

## Rejected alternatives

- use one engine API as the internal protocol;
- route only at the northbound HTTP layer;
- let core call each runtime directly.
