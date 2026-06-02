# Unsafe Audit

This inventory lists Rust sources that currently contain `unsafe` blocks or
unsafe operations. `cargo xtask unsafe-audit` checks that every matching source
under `crates/` appears here, so update this file whenever unsafe code is added,
moved, or removed.

## Inventory

| Path | Scope |
| --- | --- |
| `crates/signinum-core/src/backend.rs` | CPU feature detection and backend probing. |
| `crates/signinum-cuda-runtime/src/lib.rs` | CUDA driver/runtime FFI, module loading, kernel launch, and device memory handling. |
| `crates/signinum-cuda-runtime/src/nvjpeg.rs` | nvJPEG FFI handle, stream, and device buffer integration. |
| `crates/signinum-j2k-compare/src/grok.rs` | Grok FFI comparison harness. |
| `crates/signinum-j2k-compare/src/openjpeg.rs` | OpenJPEG FFI comparison harness. |
| `crates/signinum-j2k-metal/src/compute.rs` | Metal buffer pointer access, shader setup, and command submission. |
| `crates/signinum-j2k-metal/src/encode.rs` | Metal-backed codestream buffer views and encode validation helpers. |
| `crates/signinum-j2k-metal/src/lib.rs` | Metal surface byte views and host/device transfer helpers. |
| `crates/signinum-j2k-metal/src/mct.rs` | SIMD-assisted color transform helpers. |
| `crates/signinum-jpeg-metal/src/compute.rs` | Metal buffer pointer access, shader setup, and command submission. |
| `crates/signinum-jpeg-metal/src/lib.rs` | Metal surface byte views and host/device transfer helpers. |
| `crates/signinum-jpeg/benches/encode_cpu.rs` | Benchmark allocation-counting global allocator wrapper. |
| `crates/signinum-jpeg/benches/common/libjpeg_turbo.rs` | libjpeg-turbo FFI benchmark harness. |
| `crates/signinum-jpeg/src/backend/mod.rs` | Runtime backend dispatch and SIMD entry points. |
| `crates/signinum-jpeg/src/backend/neon.rs` | AArch64 NEON SIMD kernels. |
| `crates/signinum-jpeg/src/backend/x86.rs` | x86 SIMD kernels and CPU feature-gated dispatch. |
| `crates/signinum-jpeg/src/bench_support.rs` | Benchmark-only buffer and decoder helpers. |
| `crates/signinum-jpeg/src/decoder.rs` | Decoder buffer slicing and performance-critical pixel paths. |
| `crates/signinum-jpeg/src/entropy/sequential.rs` | Entropy decoder bitstream fast paths. |
| `crates/signinum-jpeg/src/idct/avx2.rs` | AVX2 IDCT implementation. |
| `crates/signinum-jpeg/src/idct/neon.rs` | NEON IDCT implementation. |
| `crates/signinum-transcode-cuda/src/cuda.rs` | CUDA accelerator and context integration for transcode stages. |
| `crates/signinum-transcode-metal/src/metal.rs` | Metal accelerator integration for transcode stages. |

## Review Expectations

Unsafe code must stay isolated behind safe APIs, validate pointer lengths before
forming slices, keep FFI ownership explicit, and prefer feature-gated dispatch
over unchecked CPU or GPU capability assumptions.
