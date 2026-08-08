# ADR 0004: Name the Project Locus

- Status: Accepted
- Date: 2026-08-08

## Context

Names centered on a frontend or gateway overemphasize protocol ingress and
proxying. The system coordinates compute placement, state placement,
Prefill/Decode roles, model residency, and heterogeneous runtimes.

## Decision

Use **Locus**, meaning a location or place. Placement is the central
abstraction. The category may be described as an inference control plane or
inference orchestrator; Locus is the project name.

Future package and protocol namespaces use `locus`. No compatibility aliases
are needed because no implementation artifact was released under the previous
name.

## Consequences

The positioning statement should accompany the name when its category is not
obvious. The name remains valid as placement expands across compute, state,
models, and topology.

## Rejected alternatives

- retain the frontend-centered previous name;
- adopt a gateway-centered project name.
