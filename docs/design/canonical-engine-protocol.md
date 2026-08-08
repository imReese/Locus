# Canonical Engine Protocol

## Status

This document proposes the semantic southbound contract between Locus and
engine adapters. No wire-compatible protocol has been released. Names and
field shapes are illustrative; behavior and invariants are the design contract.

## Goals

The canonical engine protocol provides:

- one engine-neutral representation of executable model work;
- typed, extensible inputs that are not limited to token IDs;
- explicit request requirements and capability negotiation;
- normalized streaming, usage, finish, error, and cancellation semantics;
- a clean boundary for runtime-specific engine adapters;
- optional attachment of prepared reusable state without exposing a particular
  state provider.

It does not standardize engine internals, memory layouts, batching algorithms,
or provider-specific state-transfer protocols.

## Placement in the request path

The canonical protocol begins after northbound validation and model-semantic
normalization. At this point Locus has selected a semantic profile,
constructed an `InputBundle`, normalized sampling parameters, and determined
the required engine capabilities.

The protocol ends at an `EngineAdapter`. An adapter may use an in-process API,
Tonic, another RPC system, or translate into an engine's existing HTTP API.
Tonic and Protobuf are the intended initial remote transport, but generated
wire types must not become the core domain model.

## Identity and versioning

Every request carries:

- a globally unique `request_id` used for idempotency and cancellation;
- a `model` identity and immutable model-artifact revision;
- a `semantics_fingerprint` identifying tokenizer, template, normalizer, and
  relevant parser configuration;
- a protocol major/minor version;
- an optional tenant scope and trace context.

Protocol major versions change only for incompatible behavior. Minor versions
add optional fields, events, or capabilities. Peers negotiate a supported
range before accepting traffic. Unknown required features are rejected rather
than ignored.

## Canonical request

The conceptual request is:

```text
CanonicalRequest
  identity: RequestIdentity
  model: ModelIdentity
  input: InputBundle
  sampling: SamplingParameters
  output: OutputContract
  requirements: CapabilityRequirements
  execution: ExecutionConstraints
  prepared_state: optional PreparedStateAttachment
  extensions: repeated typed Extension
```

### `InputBundle`

`InputBundle` is an ordered graph of input items and their relationships. A
simple text-only request normally contains one token-sequence item. A
multimodal request may interleave token sequences with image, audio, video, or
other model-input references. Metadata can describe relationships such as the
placeholder position corresponding to a media item.

```text
InputBundle
  items: repeated InputItem
  edges: repeated InputRelation
  annotations: typed metadata

InputItem
  item_id: stable within the request
  value: one of
    TokenSequence
    MediaReference
    TensorReference
    PreparedInputReference
    TypedMetadata
    ExtensionItem
```

Initial item forms have these semantics:

- `TokenSequence` contains token IDs, positions or segment information when
  required, and the tokenizer fingerprint used to produce them.
- `MediaReference` identifies immutable media plus its content type, digest,
  shape hints, and an access reference. It does not require embedding media
  bytes in every request.
- `TensorReference` describes an explicitly supported, typed tensor input with
  dtype, shape, layout, and an access reference.
- `PreparedInputReference` identifies a previously materialized, engine-
  compatible input artifact such as vision embeddings.
- `TypedMetadata` carries namespaced, schema-versioned metadata. It is not an
  unbounded map for silently changing execution behavior.
- `ExtensionItem` permits future input kinds under a negotiated type URL.

Relations express ordering, placeholder binding, cross-attention association,
or another declared relation. The graph must be acyclic where a relation type
requires it. Engines advertise the item and relation types they support.

This structure avoids three incorrect assumptions: that all input is text,
that every reusable boundary is a token offset, and that a multimodal artifact
has the same compatibility rules as a KV cache.

### Sampling parameters

`SamplingParameters` contains canonical, explicitly optional fields such as:

- maximum generated tokens;
- temperature, top-p, top-k, and minimum-p;
- repetition, frequency, and presence penalties;
- stop token sequences and stop text sequences;
- seed and requested determinism level;
- number of candidates;
- log-probability requests;
- constrained-output or grammar references.

Locus applies defaults before submission or marks a field as intentionally
engine-defaulted. Adapters do not guess how to translate an unsupported
parameter. Capability evaluation chooses one of three declared behaviors:

1. exact support;
2. an explicit, policy-approved degradation reported to the caller; or
3. rejection before execution.

Runtime-specific sampling knobs belong in a namespaced extension and make the
request ineligible for adapters that do not understand that extension.

### Output contract

`OutputContract` describes what Locus needs from the engine, including:

- preferred token-delta streaming or an allowed text-delta fallback;
- log probabilities and their alignment;
- usage granularity;
- prompt-logprob or hidden-output requests;
- normalized finish information;
- whether multiple candidates are required.

Token deltas are preferred because Locus normally owns detokenization and
incremental semantic parsing. A text-only engine can participate only when it
advertises text-delta behavior sufficient for the selected semantic profile.
The capability decision is explicit; core semantics do not silently depend on
engine-specific detokenization.

### Execution constraints

Constraints describe request-level limits rather than engine scheduling
instructions:

- deadline and cancellation token;
- priority class and tenant fairness context;
- maximum queue delay;
- allowed model or adapter revisions;
- determinism requirements;
- data-residency or topology restrictions;
- retry and idempotency policy.

They do not expose batch size, CUDA graph selection, page allocation, or other
engine-local decisions.

### Prepared state attachment

When the planner selects reusable state, the orchestrator resolves it to a
short-lived `PreparedStateAttachment`:

```text
PreparedStateAttachment
  attachment_id: opaque capability-scoped handle
  provider_kind: opaque namespace
  target_engine: engine instance identity
  compatibility_proof: verified compatibility fingerprint
  reusable_boundary: structured input coverage
  expires_at: timestamp
```

The attachment never embeds a NexusKV or engine-specific object in the core
request. Its handle is meaningful only to an adapter that advertised support
for the attachment namespace. Absence of an attachment is a normal cold
request.

## Canonical output events

Execution produces an ordered stream with monotonically increasing sequence
numbers:

```text
EngineEvent
  request_id
  sequence_number
  emitted_at
  payload: one of
    Accepted
    TokenDelta
    TextDelta
    CandidateDelta
    UsageUpdate
    Finish
    EngineError
```

Required behavior:

- `Accepted` confirms engine ownership and is emitted at most once.
- `TokenDelta` identifies the candidate, token IDs, and optional aligned
  log-probability data.
- `TextDelta` is used only under a negotiated text-streaming capability.
- `CandidateDelta` carries other negotiated typed outputs.
- `UsageUpdate` is monotonic and identifies whether values are estimated or
  final.
- `Finish` is terminal and contains a normalized reason and final usage.
- `EngineError` is terminal and contains a structured error classification.

Normalized finish reasons initially include `stop`, `length`, `tool_boundary`,
`content_filter`, `cancelled`, and `error`, with an extensible unknown value.
The northbound adapter may translate these further. Engine adapters preserve
the original runtime reason in debug metadata, not in the portable contract.

An engine stream must contain exactly one terminal event. Events received
after a terminal event are a protocol violation. A transport failure without a
terminal event becomes a structured `Unavailable` or `ExecutionFailed` result
depending on whether the engine had accepted the request.

## Control operations

The initial protocol surface should remain small:

```text
Negotiate(protocol_range) -> NegotiatedProtocol
DescribeEngine() -> EngineCapabilities
ObserveEngine() -> EngineSnapshot
Execute(CanonicalRequest) -> stream EngineEvent
Cancel(request_id, reason) -> CancelResult
Probe() -> HealthStatus
```

Remote transports may add a reservation operation if measurements show that
planning-to-submit races require it. The domain model treats reservation as an
orchestration concern, not a guarantee that an engine exposes a particular RPC.

State preload, replication, and transfer are `StateProvider` operations.
An adapter may expose a capability-gated operation to bind a materialized
attachment to an engine, but the canonical protocol does not become a general
cache-management API.

## Capabilities

Capabilities are versioned, machine-readable claims. Relevant categories
include:

- model architectures and immutable revisions;
- quantization, dtype, and parallel-layout constraints;
- supported `InputItem` and relation types;
- context and output limits;
- sampling and constrained-output features;
- token- or text-delta streaming behavior;
- log-probability and usage support;
- prepared-state attachment namespaces and state kinds;
- cancellation and idempotency behavior;
- speculative or disaggregated-prefill/decode roles.

Capabilities describe support, not current availability. Dynamic queue depth,
memory pressure, and health belong in `EngineSnapshot`.

## Error semantics

Errors carry a stable class, a safe message, retryability, the responsible
component, and optional typed details. An adapter maps runtime errors into the
common classes defined by the architecture. It must not return arbitrary HTTP
status codes or engine exception strings as the domain API.

Retryability is contextual. An error marked transient is not permission to
replay a request after output has been observed. The orchestrator combines the
error with acceptance and output progress to enforce the request's idempotency
policy.

## Backpressure and cancellation

The event stream is bounded. A slow northbound consumer propagates
backpressure until a configured buffer limit, after which policy chooses
cancellation or bounded spooling; unbounded buffering is invalid.

Cancellation is deadline-aware and idempotent. Locus stops semantic
parsing, sends `Cancel`, drains or closes the transport according to adapter
behavior, and releases reservations and state attachments. An adapter must
declare when cancellation is best-effort rather than confirmed.

## Compatibility tests

Every adapter should pass a shared conformance suite covering:

- capability claims versus actual rejection behavior;
- simple and multimodal `InputBundle` handling;
- parameter absence, defaults, and unsupported parameters;
- ordered streaming and exactly one terminal event;
- token/text delta fragmentation across parser boundaries;
- cancellation before acceptance and during generation;
- usage monotonicity and finish-reason mapping;
- prepared-state compatibility rejection;
- duplicate request IDs and retry behavior;
- preservation of unknown optional fields and rejection of unknown required
  features.

Conformance establishes contract behavior, not performance equivalence between
engines.
