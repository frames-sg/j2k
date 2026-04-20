# J2K-M5 — Hardening

Status: approved implementation spec derived from the umbrella design.

## Goal

Harden `slidecodec-j2k` with external differential checks, stress-oriented
regressions, and CI wiring.

## Scope

In scope:

- OpenJPEG differential tests for classic J2K full/region/scaled decode
- explicit out-of-bounds ROI regression coverage
- CI updates for J2K bench-build and fuzz-target compilation

Out of scope:

- long-running fuzz jobs in CI
- memory-usage profiling infrastructure

## Verification

M5 is complete when:

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo bench -p slidecodec-j2k --bench compare --no-run`

all pass, and local OpenJPEG parity tests pass when `opj_compress` and
`opj_decompress` are available.
