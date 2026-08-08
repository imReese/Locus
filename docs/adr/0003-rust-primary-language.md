# ADR 0003: Use Rust as the Primary Language

- Status: Accepted
- Date: 2026-08-08

## Context

Locus is a concurrent streaming control plane requiring bounded memory,
backpressure, cancellation, and stable typed extension boundaries.

## Decision

Use Rust 2024 with Tokio. Initial edge implementations may use Axum/Hyper and
Tonic/Protobuf; model semantics may use Hugging Face Tokenizers and MiniJinja
or equivalents. These library types stay behind Locus-owned interfaces.

Python is limited to SDKs, tooling, or an isolated compatibility worker and is
not required in the normal production hot path.

## Consequences

Rust provides predictable native execution and strong ownership boundaries,
but unusual Python-only model behavior requires explicit isolation and parity
testing.

## Rejected alternatives

- make Python the primary control-plane runtime;
- embed Python in the Rust process;
- begin with a polyglot core before validating component boundaries.
