# Anthropic-Compatible API

## Status

`locus-anthropic` implements the text-inference subset of Anthropic Messages
over the same protocol-neutral `InferenceService` used by the OpenAI adapter.
Deterministic JSON/SSE tests and the official Anthropic Python SDK 0.120.2 cover
authentication, non-streaming messages, streaming text, tool use, protocol
errors, usage, finish reasons, and stream-close cancellation through a real
local socket.

## Route and authentication

- `POST /v1/messages`: non-streaming JSON or named-event SSE

The stable `anthropic-version: 2023-06-01` header is required. Locus accepts the
SDK-native `x-api-key` header and the Bearer form, maps either credential to one
trusted tenant, and returns Anthropic-shaped errors plus a `request-id` header.
The same tenant deadline starts before bounded body ingestion and remains in
force through normalization, admission, planning, engine execution, streaming,
and cleanup.

## Supported semantics

The request adapter supports text `system` content, user/assistant text turns,
text tool results, `max_tokens`, temperature, top-p, stop sequences, metadata
`user_id`, client tool schemas, and `auto`/`any`/named/`none` tool choice. It
normalizes these once into `ModelRequest`; no handler selects an engine or
bypasses the traffic controller.

Non-streaming output uses Anthropic `message`, text, and `tool_use` objects.
Streaming follows the documented `message_start`, content-block start/delta/
stop, `message_delta`, and `message_stop` order. Tool arguments use
`input_json_delta`; the completed input must be a JSON object. Mid-stream
failures use an Anthropic `error` event. Dropping the SDK stream drops the
semantic stream and invokes downstream engine cancellation.

## Explicit boundary

This is an Anthropic-compatible Messages subset, not the complete Claude
Platform. Image/document inputs, assistant-history `tool_use` blocks, extended
thinking/signatures, `top_k`, token counting, Models, Batches, Files, hosted
server tools, prompt caching, and beta headers are rejected or not routed.
Those features require additional semantic contracts; they are not silently
flattened into text. Unknown fields also fail closed.
