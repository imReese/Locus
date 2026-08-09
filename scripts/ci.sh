#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
PYTHONPYCACHEPREFIX=/tmp/locus-pycache python3 -m py_compile \
  scripts/nexuskv_bridge_e2e.py \
  scripts/openai_sdk_e2e.py \
  scripts/live_engine_conformance.py
