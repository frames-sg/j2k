# CUDA-Oxide Kernel Path

## Goal

Continue evaluating a Rust-authored CUDA kernel path using cuda-oxide for the
J2K CUDA runtime. The existing CUDA C and checked-in PTX path remains the
compatibility baseline until the Rust-authored path proves build stability,
parity, and benchmark coverage across more than one kernel.

## Current State

- `crates/j2k-cuda-runtime/build.rs` invokes `nvcc` for `.cu` sources and
  falls back to checked-in PTX for selected kernels.
- The `cuda-oxide-copy-u8` feature adds an opt-in Rust-authored `CopyU8` PTX
  path without changing the default CUDA C/PTX runtime.
- Unsupported host builds emit placeholder PTX by default so all-features
  checks keep working. The documented require-build environment flag makes the
  build fail loudly when cuda-oxide generation is required but unavailable.
- Runtime guards prevent placeholder PTX from being loaded accidentally.
- `crates/j2k-cuda-runtime/src/kernels.rs` exposes the shared kernel registry
  used by the J2K, HTJ2K, JPEG, and transcode CUDA adapters.
- `crates/j2k-cuda`, `crates/j2k-jpeg-cuda`, and
  `crates/j2k-transcode-cuda` already route through `j2k-cuda-runtime`.

## Landed

- Added the first feature-gated cuda-oxide build path in `j2k-cuda-runtime`.
- Ported `CopyU8` as the initial low-risk Rust-authored kernel.
- Loaded generated PTX through the existing runtime module boundary.
- Added parity and metadata tests that stay host-safe when cuda-oxide is not
  available.
- Documented the unsafe kernel source in `docs/unsafe-audit.md`.

## Remaining Work

1. Measure build-time and runtime differences for the CopyU8 path on the CUDA
   validation runner.
2. Decide whether the next cuda-oxide port should target a simple
   deinterleave-style kernel, JPEG decode support code, J2K encode stages,
   HTJ2K decode/encode, or transcode kernels.
3. Add the next kernel behind a separate feature gate if it has different build
   or runtime constraints.
4. Keep expanding parity tests against the existing CUDA C/PTX path and CPU
   oracle before considering broader migration.

## Acceptance Criteria For The Next PR

- The default build remains unchanged for users without cuda-oxide.
- The cuda-oxide path is opt-in and fails loudly when explicitly requested but
  unavailable.
- The next kernel is generated from Rust, loaded by the existing runtime, and
  covered by CPU/CUDA parity tests.
- CI or GPU validation documents build requirements, runtime behavior, and
  performance evidence.
- The PR records whether cuda-oxide is ready for broader migration or should
  remain limited to selected kernels.

## Notes

cuda-oxide is currently an experimental Rust-to-CUDA compiler. Treat this as a
controlled migration path, not an immediate removal of CUDA C.
