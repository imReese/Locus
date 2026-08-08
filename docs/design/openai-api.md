# OpenAI-Compatible API

## Status

`locus-openai` implements an Axum router over the protocol-neutral
`InferenceService`. The implementation is covered by in-process HTTP and SSE
conformance tests. It is a library router, not yet a deployable server binary,
and has not been tested against a live OpenAI SDK release in this repository.

## Routes

- `POST /v1/responses`: non-streaming JSON and SSE streaming
- `POST /v1/chat/completions`: non-streaming JSON and SSE streaming
- `GET /v1/models`: registered public model aliases
- `GET /healthz`: process-level readiness response

Responses is the primary interface. Chat Completions is a compatibility layer:
both translate into `SemanticRequest`, call the same `InferenceService`, and
shape the same `SemanticEvent` stream. Neither HTTP handler constructs planner
candidates or invokes an engine adapter directly.

## Supported semantics

The current subset supports text conversations, sampling limits, temperature,
top-p, function definitions and tool choice, typed function-call deltas,
reasoning effort and deltas, JSON object/JSON Schema output contracts, usage,
finish reasons, and streaming cancellation on client disconnect. Structured
output is passed to capable runtime adapters and checked for valid JSON object
shape at semantic completion; full local JSON Schema validation is not yet
implemented.

Unknown JSON fields and explicitly unsupported options are rejected with an
OpenAI-shaped error envelope rather than ignored. Model lookup failures,
invalid requests, unavailable placement, cancellation, deadlines, and internal
failures map to stable error categories.

## Streaming ownership

Responses SSE emits lifecycle events including `response.created`, output item
and content-part creation, text/reasoning/tool deltas, done events, and
`response.completed`. Chat streaming emits `chat.completion.chunk` data and a
terminal `[DONE]`. Dropping either HTTP body drops the semantic stream, which
propagates cancellation to the selected `EngineAdapter`.

## Not implemented

Authentication, rate limits, persistence and `previous_response_id`, image or
audio inputs, hosted tools, logprobs, multiple choices, and full OpenAI API
surface parity are outside the current subset. The serde DTOs use strict field
checking so these gaps fail explicitly.

Function calling and reasoning require an engine adapter that advertises typed
tool and reasoning events. The current SGLang/vLLM pretokenized completion
adapters do not advertise those capabilities; the API returns no-available-
target rather than silently treating plain text as typed semantics.
