# J2K-M4 — SIMD / Benchmark Signoff

Status: approved implementation spec derived from the umbrella design.

## Goal

Make the J2K milestone measurable on real hardware by adding a dedicated compare
bench for `slidecodec-j2k`, including an OpenJPEG reference path.

## Scope

In scope:

- Criterion compare bench for inspect, full decode, region decode, scaled decode,
  and repeated tile-batch decode
- synthetic always-available J2K and HTJ2K bench inputs generated at runtime
- OpenJPEG comparator integration through the local `opj_decompress` CLI
- manual signoff documentation for both `aarch64` and `x86_64`

Out of scope:

- replacing the backend dependency's SIMD implementation
- in-process OpenJPEG FFI bindings

## Architecture

`slidecodec-j2k` already depends on a backend that enables SIMD by default.
M4 therefore focuses on benchmarking and signoff rather than decoder rewrites.

The compare bench:

- generates deterministic grayscale/RGB J2K and HTJ2K codestreams
- benchmarks `slidecodec-j2k` directly through its public API
- benchmarks OpenJPEG by invoking `opj_decompress` with matching full/region/
  reduce-factor settings

This is an end-to-end tool comparison, not an in-process library microbenchmark,
and is documented as such in `docs/bench.md`.

## Verification

M4 is complete when:

- `cargo bench -p slidecodec-j2k --bench compare --no-run`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

all pass, and the compare bench can be executed locally when OpenJPEG is
installed.
