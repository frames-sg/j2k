# CUDA-Oxide Kernel Path

## Goal

Evaluate and, if viable, add a Rust-authored CUDA kernel path using
cuda-oxide for the J2K CUDA runtime. The existing CUDA C and checked-in PTX
path remains the compatibility baseline until the Rust-authored path proves
build stability, parity, and benchmark coverage.

## Current State

- `crates/j2k-cuda-runtime/build.rs` invokes `nvcc` for `.cu` sources and
  falls back to checked-in PTX for selected kernels.
- `crates/j2k-cuda-runtime/src/kernels.rs` exposes the shared kernel registry
  used by the J2K, HTJ2K, JPEG, and transcode CUDA adapters.
- `crates/j2k-cuda`, `crates/j2k-jpeg-cuda`, and
  `crates/j2k-transcode-cuda` already route through `j2k-cuda-runtime`.

## Proposed Work

1. Add a feature-gated cuda-oxide build path in `j2k-cuda-runtime` without
   changing the default CUDA C path.
2. Port one low-risk kernel first, such as `CopyU8` or a simple deinterleave
   kernel, and load its generated PTX through the existing runtime module
   boundary.
3. Add parity tests that compare the cuda-oxide kernel output with the
   existing CUDA C/PTX path and CPU oracle.
4. Measure build-time and runtime differences on the CUDA validation runner.
5. Decide whether the next ports should target JPEG decode, J2K encode stages,
   HTJ2K decode/encode, or transcode kernels.

## Acceptance Criteria

- The default build remains unchanged for users without cuda-oxide.
- The cuda-oxide path is opt-in and fails loudly when explicitly requested but
  unavailable.
- At least one kernel is generated from Rust, loaded by the existing runtime,
  and covered by CPU/CUDA parity tests.
- CI or GPU validation documents how to enable the cuda-oxide path.
- The PR records whether cuda-oxide is ready for broader migration or should
  remain a spike.

## Notes

cuda-oxide is currently an experimental Rust-to-CUDA compiler. Treat this as a
controlled migration path, not an immediate removal of CUDA C.
