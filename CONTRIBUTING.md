# Contributing to Locus

Thank you for helping improve Locus. Focused bug fixes, tests, documentation,
and narrowly scoped integrations are welcome.

## Before you start

- Search existing [issues](https://github.com/imReese/Locus/issues) before
  opening a new one.
- For bugs, include the smallest reproducible configuration, expected and
  observed behavior, and the validation level involved. Remove credentials,
  private endpoints, prompts, and deployment-specific identifiers.
- Open an issue before substantial protocol, public API, architecture, or
  ownership changes so the scope and compatibility impact can be agreed first.

## Development

The repository pins its Rust toolchain in `rust-toolchain.toml`. Run the same
gate used by GitHub CI before submitting a change:

```bash
bash scripts/ci.sh
```

For the local OpenAI SDK path, follow the
[two-minute quickstart](README.md#start-in-two-minutes). Live runtime and
hardware validation are separate opt-in gates documented in
[Serving Validation and Evidence Levels](docs/validation/serving.md).

## Change expectations

- Keep runtime-specific types inside their adapters and preserve the canonical
  request and event contracts.
- Reject unsupported capabilities explicitly; do not silently guess or promote
  missing compatibility evidence.
- Add focused tests for behavior changes and failure paths.
- Document public Rust APIs and versioned wire-format changes.
- Report deterministic, SDK, live-runtime, and hardware evidence separately.

## Pull requests

Keep each pull request reviewable and limited to one coherent change. Describe:

- the problem and intended behavior;
- the files and contracts affected;
- the checks that ran and what they establish; and
- any live-runtime, state-transfer, GPU, soak, or fault testing that was not
  performed.

All contributions are submitted under the repository's
[Apache License 2.0](LICENSE).
