# ADR 0001: Establish an Engine-Neutral Boundary

- Status: Accepted
- Date: 2026-08-08
- Scope: Design; implementation has not started

## Context

Inference runtimes expose different HTTP APIs, sampling fields, streaming
events, error behavior, and model-specific helpers. Allowing those differences
into the core frontend would couple application semantics and global routing to
the first supported engine. Adding an engine would then require changes across
protocol, planning, and model-semantic code.

InferFront also needs richer inputs than a flat token list. Multimodal
references, prepared inputs, metadata, and future model forms must cross the
boundary without being encoded as runtime-specific escape fields.

## Decision

InferFront will define a canonical, engine-neutral southbound protocol and core
domain model.

Northbound adapters normalize application requests into InferFront-owned types.
Capability-based `EngineAdapter` integrations translate canonical requests and
events to a runtime. Core code selects adapters by declared capabilities,
health, policy, and cost; it does not branch on SGLang, vLLM,
TensorRT-LLM, or another engine name.

The canonical `InputBundle` is extensible and represents ordered token
sequences, multimodal references, typed metadata, prepared inputs, and future
input kinds. Token-delta streaming is preferred, with explicit capability-
gated fallbacks.

Execution engines retain continuous batching, engine-local scheduling, GPU
memory allocation, kernels, parallel execution, and model forward execution.
InferFront owns semantic normalization and global placement.

## Consequences

### Positive

- Applications receive stable semantics across compatible engines.
- Engine integrations are isolated and contract-testable.
- Planning can use capabilities without importing runtime types.
- Future protocols and input forms can evolve independently of an engine API.
- Engine-local optimization remains under the engine's control.

### Costs and risks

- The canonical model requires careful versioning and conformance tests.
- Some runtime-specific features need namespaced extensions before they can be
  generalized.
- The least-common-denominator temptation could hide useful engine features;
  capability negotiation and typed extensions must prevent that.
- Adapter translation adds code and may add transport overhead.

## Rejected alternatives

### Use one engine API as the internal protocol

This would make its request types and semantics the de facto architecture and
force other engines to emulate them.

### Route only at the northbound HTTP layer

A generic reverse proxy cannot reliably normalize templates, tokens, streamed
parser state, capabilities, or reusable model state.

### Let core code call every engine directly

This scatters version and behavior checks through the planner and makes adding
an engine a cross-cutting change.

## Follow-up

The conceptual contract is specified in
[Canonical engine protocol](../design/canonical-engine-protocol.md) and
[Engine adapter contract](../design/engine-adapter-contract.md). Implementation
must add a conformance harness before claiming adapter compatibility.
