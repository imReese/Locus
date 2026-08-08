# ADR 0004: Name the Project Locus

- Status: Accepted
- Date: 2026-08-08
- Scope: Project naming; implemented by the repository rename

## Context

The original name, InferFront, described the project's initial position in
front of inference engines. As the architecture developed, `Front`
overemphasized API handling and frontend preprocessing. Calling the system a
gateway would similarly overemphasize proxying and request routing.

The project's central responsibility is broader: it coordinates both compute
placement and reusable model-state placement. This includes engine selection,
cache locality, preload and replication, transfer-versus-recompute decisions,
Prefill/Decode placement, model placement, heterogeneous runtimes, and
topology-aware execution.

The architectural thesis is that inference is a placement problem involving
both compute and state.

## Decision

Rename the project from InferFront to **Locus**.

`Locus` means a location or place. It reflects placement as the central
abstraction without tying the project to one protocol, engine, cache system, or
deployment topology. The name remains appropriate as the project expands from
request placement into state, model, Prefill/Decode, and heterogeneous-runtime
placement.

The project category may be described as an inference control plane or
inference orchestrator, while **Locus** is the project name.

The canonical repository is
[`imReese/Locus`](https://github.com/imReese/Locus). Future package, module, and
protocol namespaces will use `locus`. Because no public implementation or
artifact has been released under the old name, no compatibility aliases are
introduced.

## Consequences

### Positive

- The name centers compute and state placement rather than protocol ingress.
- It remains valid for cache placement, Prefill/Decode scheduling, model
  placement, and topology-aware orchestration.
- It avoids defining the project as merely a gateway or reverse proxy.
- Package and protocol namespaces can adopt the permanent name before release.

### Costs and risks

- Existing links and local remotes must follow the GitHub repository rename.
- Early design discussions using the former name need historical context.
- `Locus` alone does not describe the product category, so the positioning line
  should accompany it in user-facing material.

## Architecture impact

The rename does not change component ownership or contracts. Locus remains an
engine-neutral control plane with a canonical engine protocol, capability-based
engine adapters, a generic optional `StateProvider`, and a first-class
cost-based planner for compute and state.

NexusKV remains the reference state integration rather than a core dependency.
Execution engines continue to own batching, local scheduling, device memory,
kernels, parallelism, and model execution.

## Rejected alternatives

### Keep InferFront

The old name preserves early recognition but frames the architecture around its
position in the request path rather than its placement responsibility.

### Use a name centered on gateway

Gateway terminology is useful for one deployment intuition, but it is too
narrow for state placement, transfer, replication, and multi-stage execution.
