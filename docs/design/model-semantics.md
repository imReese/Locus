# Model Semantics

## Status

This document defines the model-semantic layer. The Rust implementation now
contains `ModelRegistry`, `SemanticRequest`, `ModelSemantics`, an output-pipeline
factory, typed `SemanticEvent`s, function-call aggregation, reasoning events,
and JSON structured-output validation. `locus-semantics-hf` now loads an exact
Hugging Face `tokenizer.json`, renders a pinned chat template with bounded
MiniJinja, decodes through the same tokenizer, and derives tokenizer/template
identities from SHA-256 content digests. Production profiles can bind strict
tagged-reasoning and tagged-JSON tool parsers; parser revisions, delimiters, and
limits participate in output identities and the umbrella fingerprint.
`SemanticInput` also carries raw text or caller-tokenized prompts for the legacy
Completions API. Those inputs bypass the chat template and use a distinct
`locus-raw-prompt` input-semantic identity.
`ByteTokenizer`, `ByteDecoder`, and `SimpleTemplateRenderer` remain deterministic
reference components. Multimodal normalization and additional model output
dialects remain future work.

## Purpose

Model semantics turn an application request into engine-executable input and
turn engine output into application-visible meaning. Centralizing these rules
keeps behavior consistent when a request is placed on a different compatible
engine.

The layer includes:

- structured request validation and defaulting;
- explicit conversation versus raw-prompt input semantics;
- chat-template selection and rendering;
- tokenization and detokenization;
- multimodal input normalization;
- sampling-parameter normalization;
- reasoning and tool-call parsing;
- stop and finish semantics.

It does not execute model kernels, schedule batches, or decide request
placement.

## Model profiles

A deployment-controlled `ModelProfile` selects a versioned set of semantic
components for an immutable model revision. A profile includes:

- public model aliases and immutable artifact identity;
- tokenizer identity, revision, and configuration;
- template identity, source, revision, and declared inputs;
- multimodal normalizer identities and limits;
- reasoning and tool-call parser configuration;
- supported northbound features and defaults;
- engine requirements derived from those semantics;
- a structured `SemanticIdentity` plus an optional umbrella fingerprint.

Aliases are mutable operational configuration; semantic component identities
are not. Reusable-state compatibility uses the immutable model identity and
only the semantic subset relevant to that artifact, never a user-facing alias
or one umbrella hash alone.

Profiles are loaded, validated, and activated atomically. In-flight requests
retain the profile revision selected at admission.

## Semantic identity

`SemanticIdentity` separates three dependency groups:

```text
SemanticIdentity
  input_semantics
    tokenizer
    template
    multimodal_preprocessing

  generation_semantics
    sampling_normalization
    stop_behavior
    constrained_generation

  output_semantics
    detokenization
    reasoning_parser
    tool_parser
```

The exact public structs may evolve, but compatibility is always
artifact-specific. An umbrella fingerprint remains useful for tracing, profile
equality, and cache fast paths. It never replaces structured evidence about the
components on which an artifact actually depends.

## Conceptual interfaces

The interfaces below illustrate separation of responsibility, not a committed
Rust API:

```rust,ignore
pub trait TokenizerProvider: Send + Sync {
    fn identity(&self) -> &TokenizerIdentity;
    fn encode(&self, input: &str, options: EncodeOptions)
        -> Result<TokenSequence, SemanticError>;
    fn decoder(&self, options: DecodeOptions)
        -> Result<Box<dyn IncrementalDecoder>, SemanticError>;
}

pub trait TemplateRenderer: Send + Sync {
    fn identity(&self) -> &TemplateIdentity;
    fn render(&self, conversation: &Conversation, context: &TemplateContext)
        -> Result<RenderedPrompt, SemanticError>;
}

pub trait ModelSemantics: Send + Sync {
    fn identity(&self) -> &SemanticIdentity;
    fn normalize(&self, request: SemanticRequest)
        -> Result<NormalizedRequest, SemanticError>;
    fn output_pipeline(&self, contract: &OutputContract)
        -> Result<OutputPipeline, SemanticError>;
}
```

Reasoning and tool parsers are factories that create request-scoped,
incremental parser state. Multimodal normalizers follow the same narrow-provider
pattern. A composed `ModelSemantics` service coordinates them but does not hide
their identities from compatibility checks.

## Input normalization pipeline

The semantic pipeline follows a deterministic order:

1. **Protocol-independent validation.** Check the input mode, conversation
   roles, tool schemas, media counts, generation limits, and mutually exclusive
   options.
2. **Profile resolution.** Pin the immutable model and semantic profile.
3. **Template selection.** Render structured conversation data, tool
   definitions, and declared variables, or select the explicit raw-prompt
   bypass identity.
4. **Multimodal normalization.** Validate and canonicalize media references,
   bind them to template placeholders, and produce typed input items.
5. **Tokenization.** Encode rendered conversation or raw prompt text with the
   pinned tokenizer. Preserve caller-supplied token IDs under that same
   tokenizer identity without rendering or re-tokenizing them.
6. **Sampling normalization.** Resolve aliases and defaults, validate ranges,
   and express required engine capabilities.
7. **Input assembly.** Construct an ordered canonical `InputBundle` containing
   token sequences, media or prepared-input references, relations, and typed
   metadata.
8. **Output contract construction.** Select detokenizer and incremental parsers
   and declare the engine output required to drive them.

Errors point to the stage and offending public field without exposing template
source or private deployment data.

## Raw prompts

`SemanticInput::Prompt` represents either text or one token-ID sequence. Text is
encoded directly with the pinned tokenizer. Token IDs cross the semantic layer
unchanged and are labeled with that tokenizer fingerprint. Neither form invokes
`TemplateRenderer`, adds chat roles, or emits a generation marker.

The canonical input replaces the profile's chat-template component with the
fixed `locus-raw-prompt` revision and does not reuse the chat profile's umbrella
fingerprint. This prevents reusable-state evidence from claiming that a chat
template participated in a raw prompt. Raw prompts currently reject tools,
reasoning controls, and structured-output contracts; those features remain on
Responses and Chat Completions.

## Chat templates

Templates consume a typed context rather than an unrestricted process object.
The production Hugging Face profile uses MiniJinja behind `TemplateRenderer`.
It supplies conversation messages, an explicit `add_generation_prompt`, an
OpenAI-shaped function tool list and tool choice, and deployment-configured
special-token values. Configuration cannot replace reserved fields. File,
network, environment, clock, and arbitrary code access are unavailable during
rendering, and rendered output is bounded by `max_rendered_bytes`.

A template declares:

- required conversation and tool fields;
- supported roles and multimodal placeholders;
- special-token handling;
- whether it emits generation markers;
- renderer and language compatibility version;
- a content digest used in the semantic fingerprint.

Rendering is deterministic for a pinned context. The profile is validated at
startup, and the template source digest participates in the input-semantic and
umbrella fingerprints. Templates that depend on runtime-native tool context or
unsupported helpers fail rather than silently changing prompt semantics.

Template output is not always one flat string. The intermediate
`RenderedPrompt` preserves segments and placeholder bindings so that
multimodal inputs and reusable state boundaries remain meaningful.

## Tokenization and detokenization

The normal production path uses a Rust tokenizer implementation, initially
Hugging Face Tokenizers where compatible. Tokenizer identity includes all
behavior that can change token IDs: vocabulary and merges, normalization,
pre-tokenization, special tokens, added tokens, and relevant options.

Incremental decoding must handle:

- byte sequences split across token events;
- replacement or cleanup behavior at token boundaries;
- stop sequences that cross engine event boundaries;
- multiple candidates with isolated decoder state;
- engines that coalesce or fragment token deltas.

Token-delta output is preferred. A text-delta fallback is allowed only when the
engine advertises sufficient behavior and the selected parsers can operate
correctly without raw token boundaries. The fallback is a capability decision,
not a silent change in ownership.

## Multimodal normalization

Multimodal semantics are not reduced to textual placeholder tokens. A
normalizer produces typed `InputItem`s and relations within `InputBundle`.

For each media input it records, as applicable:

- media kind, content type, and immutable digest;
- source or access reference after security policy;
- dimensions, duration, frame selection, or other shape metadata;
- preprocessing profile and revision;
- binding to rendered prompt segments;
- whether a prepared model input may be substituted;
- capability and context-cost requirements.

Fetching and decoding media occurs through bounded, policy-controlled services.
Remote URLs are not passed blindly to engines. Reusable vision embeddings or
other prepared artifacts have their own compatibility and reuse boundaries.

## Sampling normalization

Northbound parameters are converted into canonical sampling semantics before
placement. The normalizer distinguishes:

- an omitted value;
- a profile default;
- an explicitly requested value;
- an engine-defaulted value allowed by policy.

The implemented remote completion adapters forward canonical `max_tokens`,
temperature, top-p, seed, and non-empty stop-sequence lists. A supplied stop is
not silently discarded or enforced only in the northbound response shaper.

It also derives capability requirements. An adapter cannot reinterpret a field
because its engine uses a similar name. Unsupported exact semantics result in
policy-approved degradation or pre-execution rejection.

## Incremental output pipeline

Canonical engine events flow through request-scoped stages:

```text
token events       -> incremental decoder -> reasoning/tool parser state
engine finish fact -----------------------> semantic finish normalization
                                          -> northbound protocol adapter
```

Engines report execution facts. `ModelSemantics` derives application meaning.
An `EngineFinishReason` contains facts such as stop, length, cancellation,
error, or a namespaced runtime-specific reason. A separate
`SemanticFinishReason` may identify a tool call, content filtering, reasoning
outcome, or another northbound meaning after the output pipeline has observed
the complete semantic context.

An engine may optionally provide structured semantic events through an
explicit capability. Locus consumes them only when the model profile and
adapter declare compatible semantics; adapters do not infer application finish
reasons by default.

Parsers consume an ordered stream and may buffer only bounded partial syntax.
They emit typed events rather than directly constructing OpenAI response
objects. This keeps the same semantics usable by future northbound protocols.

The pipeline preserves three distinct views where needed:

- raw canonical engine events for diagnostics and accounting;
- decoded model text for parsing;
- application-visible content after semantic classification.

Reasoning content must not leak into answer text or tool arguments because of
chunk boundaries. Likewise, partial JSON in a tool call is not declared valid
until parser rules permit it.

## Reasoning parsers

A reasoning parser recognizes model-specific delimiters or structures and
emits typed reasoning and answer deltas. Its configuration declares:

- recognized syntax and version;
- whether raw token access is required;
- bounded buffering behavior;
- behavior for malformed or unterminated reasoning;
- interaction with stops, tools, and finish reasons.

Malformed output follows an explicit profile policy: preserve it as ordinary
content, return a parse error, or report a degraded parse. Silent loss is not
valid.

The implemented `tagged` parser selects distinct start and end delimiters from
the model profile. It holds delimiter prefixes across transport chunks, emits
reasoning incrementally, rejects stray, repeated, partial, or unterminated
reserved syntax, and never reclassifies reasoning bytes as answer text. The
current implementation uses the fail-closed parse-error policy.

## Tool-call parsers

A tool parser transforms model output into typed call identifiers, function
names, and arguments. It is responsible for streaming fragmentation and model
syntax, but not for authorizing or executing tools.

The parser distinguishes:

- tentative syntax from committed tool-call structure;
- partial argument bytes from valid completed arguments;
- one call from parallel or sequential calls;
- parser completion from engine finish reason.

Tool definitions are validated before template rendering. Model-emitted tool
names and arguments remain untrusted application data.

The implemented `tagged_json` parser accepts one or more sequential envelopes:

```text
<tool_call>{"name":"weather","arguments":{"city":"Beijing"}}</tool_call>
```

Delimiters and a maximum buffered call size are profile configuration. A call is
transactional: Locus buffers tentative syntax, requires an envelope containing
exactly a non-empty `name` and object-valued `arguments`, then emits start,
arguments, and completion events together. Unknown tools, a mismatch with an
explicit function choice, malformed JSON, incomplete calls, duplicate call IDs,
and inconsistent runtime finish reasons fail closed. Sequential calls receive
stable request-scoped indices in observation order.

## Python compatibility boundary

Python is not required in the normal production hot path. A Python SDK may
construct requests or manage deployment configuration without changing this
rule.

Some models may require unusual custom tokenizer, template, or
`trust_remote_code` behavior that cannot be safely or promptly represented in
Rust. A future compatibility path may run those semantics in an isolated Python
worker with:

- a narrow, versioned RPC boundary;
- no in-process Python interpreter in the Locus process;
- explicit model-profile opt-in;
- process, filesystem, network, time, and memory isolation;
- deterministic inputs and bounded outputs;
- separate health, admission, and observability.

Using this worker is a declared capability and a performance/security trade-off,
not a transparent fallback.

## Caching and compatibility

Semantic artifacts may be cached, but each has a typed key:

- rendered prompts depend on template identity and complete typed inputs;
- token sequences depend on tokenizer identity and encoding options;
- prepared multimodal inputs depend on preprocessing and model identities;
- reusable model state additionally depends on execution layout and state kind.

Examples of artifact-specific compatibility include:

- KV state depends on model execution identity, relevant input semantics,
  positional behavior, state layout, and runtime compatibility;
- a reasoning parser does not normally affect prefill-state compatibility;
- a prepared vision artifact depends on model identity, media digest, and
  multimodal preprocessing identity;
- a tool parser depends on the output semantic profile, not the reusable input
  state.

The umbrella semantic fingerprint summarizes the full profile for quick
equality. It does not replace these structured checks.

## Validation strategy

Semantic conformance tests should include:

- golden template and tokenization vectors pinned to upstream revisions;
- special-token, Unicode, and incremental-decoding edge cases;
- multimodal placeholder and ordering cases;
- cross-event reasoning and tool syntax fragmentation;
- malformed and unterminated parser inputs;
- sampling default and unsupported-feature matrices;
- deterministic profile fingerprinting;
- parity tests for any Python compatibility profile;
- execution of the same normalized request through fake engine adapters.

Golden tests validate declared semantics. They do not establish equivalence for
uncontrolled remote model code or different tokenizer revisions.
