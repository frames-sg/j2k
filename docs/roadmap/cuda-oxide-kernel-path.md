# CUDA-Oxide Kernel Path

## Goal

Continue migrating practical J2K CUDA runtime kernels to Rust-authored
cuda-oxide equivalents. The existing CUDA C and checked-in PTX path remains the
default compatibility baseline; cuda-oxide routes are opt-in until build
stability, parity, and benchmark coverage are broad enough to promote them.

## Current State

- `crates/j2k-cuda-runtime/build.rs` invokes `nvcc` for `.cu` sources and
  falls back to checked-in PTX for selected kernels.
- The `cuda-oxide-copy-u8` feature adds an opt-in Rust-authored `CopyU8` PTX
  path without changing the default CUDA C/PTX runtime.
- The `cuda-oxide-j2k-encode`, `cuda-oxide-j2k-decode-store`,
  `cuda-oxide-j2k-dequantize`, `cuda-oxide-j2k-idwt`, and
  `cuda-oxide-transcode` features cover selected J2K CUDA paths where the GPU
  work maps cleanly to cuda-oxide kernels.
- Unsupported host builds emit placeholder PTX by default so all-features
  checks keep working. The documented require-build environment flag makes the
  build fail loudly when cuda-oxide generation is required but unavailable.
- Runtime guards prevent placeholder PTX from being loaded accidentally.
- `crates/j2k-cuda-runtime/src/kernels.rs` exposes the shared kernel registry
  used by the J2K, HTJ2K, JPEG, and transcode CUDA adapters.
- `crates/j2k-cuda`, `crates/j2k-jpeg-cuda`, and
  `crates/j2k-transcode-cuda` already route through `j2k-cuda-runtime`.

## Landed

- Added feature-gated cuda-oxide build paths in `j2k-cuda-runtime`.
- Ported `CopyU8` as the initial low-risk Rust-authored kernel.
- Ported supported J2K encode-stage kernels: deinterleave, RCT/ICT, forward
  DWT 5/3 and 9/7, quantization, HTJ2K encoded-byte compaction, and HTJ2K
  packetization.
- Ported supported J2K decode-store, HTJ2K dequantize, generic IDWT, and
  JPEG-to-J2K transcode kernels where the current CUDA path is data-parallel.
- Loaded generated PTX through the existing runtime module boundary.
- Added parity and metadata tests that stay host-safe when cuda-oxide is not
  available.
- Documented the unsafe kernel source in `docs/unsafe-audit.md`.

## Remaining Work

1. Keep the remaining HTJ2K entropy encode/decode kernels on CUDA C unless a
   measured cuda-oxide port shows a practical win.
2. Add benchmarks for the landed cuda-oxide routes on self-hosted CUDA
   validation hardware before changing default routing.
3. Keep expanding parity tests against the existing CUDA C/PTX path and CPU
   oracle before considering broader migration.
4. Treat JPEG decode kernels as a separate follow-up because the current goal
   is J2K/HTJ2K coverage first.

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
