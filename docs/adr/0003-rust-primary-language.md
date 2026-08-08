# ADR 0003: Use Rust as the Primary Implementation Language

- Status: Accepted
- Date: 2026-08-08
- Scope: Design; implementation has not started

## Context

Locus will be a concurrent, streaming service on the production inference
path. It must maintain bounded memory, propagate cancellation and backpressure,
compose multiple protocol clients, and expose stable extension boundaries.

The model ecosystem contains Python-only custom behavior, but embedding Python
in the Locus process would make the interpreter and its global runtime behavior
part of the normal hot path.

## Decision

Rust is the primary implementation language, targeting Rust 2024 edition.

The initial direction is:

- Tokio for asynchronous execution;
- Axum/Hyper initially for HTTP and SSE;
- Tonic and Protobuf for remote canonical engine contracts;
- Hugging Face Tokenizers' Rust implementation for compatible tokenizers;
- MiniJinja or an equivalent Rust renderer for Jinja-compatible templates.

These libraries remain behind Locus-owned interfaces. Core architecture is
not coupled to Axum, Tonic, or one renderer.

Python is supported for SDKs, tooling, and an explicit compatibility escape
hatch. Unusual custom model semantics may eventually run in an isolated Python
worker with a narrow RPC interface. Python is not required in the normal
production request path and is not embedded into the Locus process.

## Consequences

### Positive

- Strong ownership and type checking support safe concurrent streaming code.
- Rust provides predictable runtime and memory behavior without a garbage-
  collected or Python interpreter hot path.
- Traits and enums can express versioned provider and adapter contracts.
- A single native service can host the initial HTTP, planning, and RPC roles.

### Costs and risks

- Some tokenizer, template, or custom-model behavior will lag Python ecosystem
  support.
- Async trait and streaming API design requires discipline around cancellation,
  lifetimes, and object safety.
- Contributors may need both Rust and Python knowledge for compatibility tools.
- Isolated semantic workers add deployment and parity-testing complexity when
  they are introduced.

## Rejected alternatives

### Implement the control plane primarily in Python

This offers ecosystem reach but makes Python a production hot-path requirement
and weakens the intended isolation of custom remote code.

### Embed Python in the Rust control plane

Embedding keeps one process but couples safety, resource limits, packaging, and
failure behavior to the interpreter. An isolated worker provides a clearer
trust and operational boundary.

### Use separate languages for every component initially

A polyglot first version would increase build and deployment complexity before
the logical boundaries have been validated. Remote contracts still permit
future components in other languages.

## Follow-up

The first code bootstrap should pin a Rust toolchain and edition, establish
workspace layering, and test cancellation and bounded-stream primitives. A
Python worker should be added only with an explicit profile, isolation model,
and Rust/Python semantic parity tests.
