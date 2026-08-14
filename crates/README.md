# Workspace crates

Directory names are intentionally short because they already live under the
Locus workspace. Published Cargo package names keep the `locus-` prefix.

| Layer | Directory | Cargo package | Responsibility |
| --- | --- | --- | --- |
| Foundation | `core` | `locus-core` | Canonical requests, identities, execution facts, and state contracts |
| Ports | `model-io` | `locus-model-io` | Model request/event contracts, normalization, and Hugging Face rendering |
| Ports | `parser` | `locus-parser` | Bounded reasoning and tool-call output parsers |
| Ports | `engine` | `locus-engine` | Engine registry and adapter contract |
| Ports | `store` | `locus-store` | Reusable-state store contract |
| Control plane | `planner` | `locus-planner` | Placement selection and plan execution |
| Control plane | `runtime` | `locus-runtime` | End-to-end inference orchestration |
| Northbound adapter | `openai` | `locus-openai` | OpenAI-compatible HTTP API |
| Engine adapter | `engine-openai` | `locus-engine-openai` | SGLang and vLLM HTTP adapters |
| Store adapter | `store/nexuskv` | `locus-store-nexuskv` | NexusKV bridge |
| Application | `server` | `locus-server` | Configuration and dependency assembly |

The dependency direction is application and adapters -> control plane -> ports
and foundation. Core crates must not depend on deployment-specific adapters.
