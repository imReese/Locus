# OpenAI-Compatible API

## Status

`locus-openai` implements an Axum router over the protocol-neutral
`InferenceService`. The implementation is covered by in-process HTTP and SSE
conformance tests and is assembled by the deployable `locus-server` binary.
GitHub CI runs official OpenAI Python SDK 2.53.0 against a real local HTTP
fixture, covering Responses, Chat Completions, and raw-prompt Completions
JSON/SSE, structured output, OpenAI-shaped errors, authentication, and
stream-close cancellation.

## Routes

- `POST /v1/responses`: non-streaming JSON and SSE streaming
- `POST /v1/chat/completions`: non-streaming JSON and SSE streaming
- `POST /v1/completions`: single raw text or token-ID prompt, JSON and SSE
- `GET /v1/models`: public aliases with at least one live, healthy execution target
- `GET /healthz`: process liveness; intentionally does not probe dependencies
- `GET /readyz`: gateway routability, explicit required-model gates, and target health
- `GET /metrics`: low-cardinality Locus admission, lifecycle, and drain metrics

Responses is the primary interface. Chat Completions and Completions are
compatibility layers: all three translate into `ModelRequest`, call the same
`InferenceService`, and shape the same `ModelEvent` stream. Completions uses
an explicit raw-prompt input mode, so text is tokenized directly and token IDs
are preserved; neither form passes through the chat template. No HTTP handler
constructs planner candidates or invokes an engine adapter directly.

## Supported semantics

The current subset supports text conversations, raw text and token-ID prompts,
sampling limits, temperature, top-p, seed and stop sequences, function
definitions and tool choice, typed function-call deltas, reasoning effort and
deltas, JSON object/JSON Schema output contracts, usage, finish reasons, and
streaming cancellation on client disconnect. Structured output is passed to
capable runtime adapters and checked for valid JSON object shape at semantic
completion; full local JSON Schema validation is not yet implemented.

Completions supports exactly one prompt and one choice. `n > 1`, batched text or
token prompts, `best_of`, `echo=true`, `logprobs`, and `suffix` fail explicitly
before inference. Its response uses `text_completion`, standard choice fields,
completion-shaped usage, and a terminal `[DONE]` for SSE. `stream_options` may
request a final usage chunk.

Unknown JSON fields and explicitly unsupported options are rejected with an
OpenAI-shaped error envelope rather than ignored. An unknown semantic profile
returns `model_not_found` with HTTP 404; a known profile without an eligible live
target returns `no_available_target` with HTTP 503. Invalid requests,
cancellation, deadlines, and internal failures map to stable error categories.

`locus-server` maps each configured bearer credential to exactly one tenant
policy while leaving probes unauthenticated for an orchestrator. Credentials
use a SHA-256 lookup index followed by a length check and constant-time raw-token
verification. Request JSON and arbitrary
headers cannot override the authenticated tenant. The tenant-clamped deadline
is created before bounded request-body ingestion, then reused by normalization,
admission, planning, engine execution, and streaming. After exact model
normalization, the runtime charges prompt plus reserved output tokens and queues
by service-class and tenant virtual runtime. Global/class/tenant request and
token caps, bounded queues, class-aware overload shedding, and tenant-owned
deadline clamps apply to JSON and streaming responses.

## Streaming ownership

Responses SSE emits lifecycle events including `response.created`, output item
and content-part creation, text/reasoning/tool deltas, done events, and
`response.completed`. Chat streaming emits `chat.completion.chunk` data.
Completions streaming emits `text_completion` chunks, a finish chunk, and an
optional final usage chunk. Both compatibility streams terminate with `[DONE]`.
Dropping any streaming HTTP body drops the semantic stream, which propagates
cancellation to the selected `EngineAdapter`.

## Not implemented

Distributed admission-state sharing across multiple Locus replicas, request
persistence and `previous_response_id`, image or audio inputs, hosted tools,
logprobs, multiple choices, and full OpenAI API surface parity are outside the
current subset. One process is authoritative for its configured admission
budget. The serde DTOs use strict field checking so API gaps fail explicitly.

Function calling and reasoning require either compatible typed engine events or
an explicitly configured model-profile parser. With a profile parser, planning
requires incremental text rather than claiming the SGLang/vLLM completion
adapter emits native typed events. Without either path, the API returns
no-available-target; plain text is never silently treated as typed semantics.
