# J2K-M4 — SIMD

Status: approved implementation spec derived from the umbrella design.

## Goal

Implement the SIMD J2K hot path in `slidecodec-j2k`:

- NEON and AVX2 acceleration for the DWT paths
- SIMD-friendly Tier-1 and color-transform kernels
- parity-preserving behavior against the scalar decoder

## Scope

In scope:

- SIMD implementations for the hot decode kernels
- architecture dispatch for `aarch64` and `x86_64`
- scalar fallback coverage for unsupported CPUs
- bench updates that verify the SIMD path, without treating a comparator run as
  the milestone exit criterion

Out of scope:

- benchmark validation
- in-process comparator ownership, which belongs to M5 hardening

## Architecture

`slidecodec-j2k` already has the scalar decoder path. M4 layers SIMD kernels on
top of the existing hot loop structure:

- dispatch chooses the best available implementation at decoder construction
- DWT, Tier-1, and color transforms each get architecture-specific kernels
- scalar code remains the reference behavior for tests and unsupported CPUs

The milestone uses Criterion as a regression and sanity check, not as a release
gate. The bench confirms that the SIMD path executes and stays parity-preserving
against the scalar path.

## Verification

M4 is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo bench -p slidecodec-j2k --bench compare --no-run`

all pass, and the bench exercises the SIMD kernels on the supported hosts.
