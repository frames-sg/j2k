# J2K-M5 — Hardening

Status: approved implementation spec derived from the umbrella design.

## Goal

Harden `slidecodec-j2k` with in-process differential checks, stress-oriented
regressions, and CI wiring.

## Scope

In scope:

- in-process OpenJPEG differential tests for classic J2K full/region/scaled
  decode
- in-process Grok differential tests for the same supported surfaces where
  available
- explicit out-of-bounds ROI regression coverage
- CI updates for J2K bench-build and fuzz-target compilation

Out of scope:

- benchmark validation
- long-running fuzz jobs in CI
- memory-usage profiling infrastructure

## Verification

M5 is complete when:

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo bench -p slidecodec-j2k --bench compare --no-run`

all pass, and the in-process OpenJPEG/Grok parity tests pass when the relevant
libraries are available locally.
