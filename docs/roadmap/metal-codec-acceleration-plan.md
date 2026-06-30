# CUDA Oxide GPU Acceleration Plan

This is the single working roadmap for GPU codec acceleration in the J2K
workspace. The filename is retained for compatibility with existing links, but
the plan supersedes the previous Metal-only roadmap.

## Goal

Make CUDA Oxide the only product CUDA device-kernel path while preserving the
existing Rust CUDA Driver API runtime for context, memory, stream, event, and
module orchestration. Metal remains the Apple GPU backend, and CUDA resident
buffers are the CUDA-side equivalent for Metal buffer/texture residency until a
backend-neutral texture abstraction exists.

The target state is:

- product CUDA device kernels are generated from CUDA Oxide projects
- strict CUDA requests run CUDA Oxide or return structured unsupported/build
  errors
- CUDA C/PTX source files and checked-in product PTX artifacts are removed
- the NVIDIA JPEG and JPEG 2000 comparator has one final recorded comparison and
  no active harness remains
- public docs describe CUDA Oxide and Metal as selective, benchmark-gated GPU
  acceleration paths

## Current State

All product CUDA device-kernel families now use CUDA Oxide:

- CopyU8 test kernel
- J2K encode stages: deinterleave, forward RCT/ICT, forward 5/3 and 9/7 DWT,
  quantization, HTJ2K compaction, and HTJ2K cleanup packetization
- J2K decode store / inverse MCT
- HTJ2K dequantize
- J2K IDWT
- HTJ2K cleanup/refinement decode
- HTJ2K encode codeblock kernels
- JPEG baseline decode kernels
- JPEG baseline encode kernels
- JPEG-to-J2K/HTJ2K transcode stages

CUDA C/PTX source artifacts and checked-in product PTX files have been retired
from `crates/j2k-cuda-runtime/src`. The NVIDIA baseline harness was removed
after the final strict CUDA Oxide comparison was recorded in
`docs/benchmark-evidence.md`.

## Kernel And Pathway Matrix

| Area | Current product path | CUDA Oxide status | Metal equivalent | Required action |
| --- | --- | --- | --- | --- |
| CopyU8 device copy | CUDA Oxide | Covered | N/A | Keep as CUDA Oxide runtime helper |
| J2K encode stage deinterleave/RCT/ICT/DWT/quantize | CUDA Oxide | Covered | Metal encode stages | Maintain strict parity and benchmarks |
| HTJ2K compaction/packetize cleanup | CUDA Oxide | Covered | Metal HT packetization paths | Maintain strict parity and benchmarks |
| J2K decode store and inverse MCT | CUDA Oxide | Covered | Metal store/MCT kernels | Maintain strict parity and benchmarks |
| HTJ2K dequantize | CUDA Oxide | Covered | Metal decode preparation | Maintain strict parity and benchmarks |
| J2K IDWT | CUDA Oxide | Covered | Metal IDWT kernels | Maintain strict parity and benchmarks |
| HTJ2K cleanup/refinement decode | CUDA Oxide | Covered | Metal HT cleanup kernels | Maintain strict parity and benchmarks |
| HTJ2K encode codeblocks | CUDA Oxide | Covered | Metal HT codeblock encode | Maintain strict parity and benchmarks |
| JPEG baseline decode | CUDA Oxide | Covered | Metal JPEG decode | Widen CUDA JPEG APIs where Metal supports additional shapes |
| JPEG baseline encode | CUDA Oxide | Covered | Metal JPEG encode | Maintain strict parity and benchmarks |
| JPEG-to-J2K/HTJ2K transcode | CUDA Oxide | Covered | Metal transcode kernels | Maintain strict parity and benchmarks |
| Resident J2K/HTJ2K codestream output | CUDA resident codestream buffer with host metadata | Covered; codestream assembly is host-staged then uploaded to CUDA | Metal resident codestream buffer | Replace host-staged assembly only if a future GPU packet assembly path lands |
| JPEG viewport/batch resident output | Full-tile RGB8 surfaces and caller-owned CUDA RGB8 output buffers, including batch APIs | Partial; region, scaled, region+scaled, Gray8, and Rgba8 strict CUDA decode return structured unsupported errors | Metal buffer/texture viewport paths | Add native CUDA viewport/format kernels only if required for v1 parity |

## Implementation Rules

- Use `trash <path>` for local deletions.
- Preserve strict backend semantics: `BackendRequest::Cuda` must never silently
  fall back to CPU.
- Keep the public feature `cuda-runtime`; existing `cuda-oxide-*` feature names
  may stay temporarily as compatibility aliases while the cutover lands.
- Any missing CUDA Oxide PTX at runtime must produce a typed, non-sensitive
  error naming the missing build gate.
- Repo lints must reject product `.cu` files, checked-in product `.ptx` files,
  and active NVIDIA comparator workflows.

## Workstream A: CUDA Oxide Build And Dispatch

1. Use the shared strict build gate `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`.
2. Keep dispatch collapsed to one CUDA Oxide module family per `CudaKernel`.
3. Keep per-family CUDA Oxide selection environment variables removed.
4. Keep product `.cu` files and checked-in product `.ptx` files out of
   `crates/j2k-cuda-runtime/src`.
5. Keep repo lint coverage rejecting product `.cu`, checked-in product `.ptx`,
   and active NVIDIA comparator workflows.

## Workstream B: Missing CUDA Oxide Kernels

Implemented during the migration:

- `cuda_oxide_htj2k_decode`
- `cuda_oxide_htj2k_encode`
- `cuda_oxide_jpeg_decode`
- `cuda_oxide_jpeg_encode`

Each project must include:

- a host crate and `simt` device crate matching existing CUDA Oxide layout
- entrypoint names compatible with current host dispatch where practical
- ABI metadata tests that inspect generated PTX
- scalar CPU parity fixtures for deterministic small cases
- strict runtime tests on NVIDIA/Linux with `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`

## Workstream C: CUDA Equivalents For Metal Surfaces

Add CUDA resident-buffer equivalents for Metal-only pathways:

- J2K/HTJ2K resident codestream output: implemented as
  `CudaResidentCodestreamBuffer`, `CudaEncodedJ2k`, and
  `CudaLosslessBufferEncodeOutcome`; final codestream assembly is host-staged and
  uploaded to a CUDA buffer.
- JPEG full-tile RGB8 decode, including 4:2:0, 4:2:2, and 4:4:4 inputs
- JPEG batch decode into caller-owned CUDA RGB8 buffers: implemented for full
  tiles through `CudaJpegDecodeOutputTile` and
  `Codec::decode_tiles_rgb8_into_cuda_buffers_with_session`
- JPEG region, scaled, region+scaled, Gray8, and Rgba8 strict CUDA decode:
  documented and tested as structured unsupported shapes for this phase
- JPEG viewport composition into CUDA buffers: structured unsupported for this
  phase because no backend-neutral CUDA texture/viewport abstraction exists yet
- JPEG baseline encode from CUDA buffers: implemented for single and same-buffer
  batch resident `Gray8`/`Rgb8` inputs

No CUDA texture API is required for this phase.

## Workstream D: NVIDIA Baseline Retirement

Before deleting the harness, run the final comparison on a CUDA/NVIDIA host:

```bash
export J2K_REQUIRE_CUDA_RUNTIME=1
export J2K_REQUIRE_CUDA_OXIDE_BUILD=1
export J2K_CUDA_OXIDE_ARCH=<host-sm>
```

Required comparisons:

- `decode_compare` for J2K/HTJ2K decode
- `jpeg_decode_compare` for JPEG decode
- `transcode_compare` for JPEG-to-HTJ2K transcode

Record host, GPU, CUDA version, corpus, command lines, JSON/CSV artifacts,
correctness deltas, wall time, GPU event time, and publication blockers in
`docs/benchmark-evidence.md`.

After evidence is recorded:

- delete the temporary NVIDIA comparator harness with `trash`
- remove NVIDIA JPEG/JPEG 2000 comparator features, dependencies, and workflow
  references
- update docs so no active workflow asks users to build or run the NVIDIA
  comparator

## Validation Gates

Non-GPU gates:

```bash
cargo fmt --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p xtask --all-targets
```

CUDA/NVIDIA gates:

```bash
J2K_REQUIRE_CUDA_RUNTIME=1 \
J2K_REQUIRE_CUDA_OXIDE_BUILD=1 \
J2K_CUDA_OXIDE_ARCH=<host-sm> \
cargo test -p j2k-cuda-runtime --all-targets --features cuda-runtime

J2K_REQUIRE_CUDA_RUNTIME=1 \
J2K_REQUIRE_CUDA_OXIDE_BUILD=1 \
J2K_CUDA_OXIDE_ARCH=<host-sm> \
cargo test -p j2k-cuda --all-targets --features cuda-runtime

J2K_REQUIRE_CUDA_RUNTIME=1 \
J2K_REQUIRE_CUDA_OXIDE_BUILD=1 \
J2K_CUDA_OXIDE_ARCH=<host-sm> \
cargo test -p j2k-jpeg-cuda --all-targets --features cuda-runtime

J2K_REQUIRE_CUDA_RUNTIME=1 \
J2K_REQUIRE_CUDA_OXIDE_BUILD=1 \
J2K_CUDA_OXIDE_ARCH=<host-sm> \
cargo test -p j2k-transcode-cuda --all-targets --features cuda-runtime
```

macOS Metal parity gates:

```bash
cargo test -p j2k-metal --all-targets
cargo test -p j2k-jpeg-metal --all-targets
cargo test -p j2k-transcode-metal --all-targets
```

## Completion Criteria

The CUDA Oxide kernel cutover is complete when:

- no product `.cu` files remain under `crates/j2k-cuda-runtime/src`
- no checked-in product `.ptx` files remain under `crates/j2k-cuda-runtime/src`
- per-family CUDA Oxide selection environment variables are gone
- strict CUDA launches cannot select CUDA C/PTX
- final NVIDIA comparison evidence is recorded
- the temporary NVIDIA comparator harness is removed
- docs and lints prevent reintroducing active NVIDIA comparator workflows

The broader Metal-equivalent CUDA resident API work is complete for this phase:
implemented CUDA paths cover full-tile resident JPEG/J2K output and batch
caller-owned RGB8 JPEG decode, while remaining JPEG viewport, scaled, region,
Gray8, and Rgba8 strict CUDA requests are documented and tested as structured
unsupported shapes.
