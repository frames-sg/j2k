# Unsafe Audit

This inventory lists Rust sources that currently contain `unsafe` blocks or
unsafe operations. `cargo xtask unsafe-audit` checks that every matching source
under `crates/` appears here, so update this file whenever unsafe code is added,
moved, or removed.

## Inventory

| Path | Scope | Invariants | Regression guards |
| --- | --- | --- | --- |
| `crates/signinum-core/src/accelerator.rs` | Shared GPU ABI marker trait and byte-view helpers for host/device struct transfers. | Implementers are plain-data GPU ABI values with stable layout and valid byte views. | Core API tests plus backend layout/parity tests for GPU ABI structs. |
| `crates/signinum-core/src/backend.rs` | CPU feature detection and backend probing. | CPU intrinsics are called only behind matching architecture/configuration checks. | Cross-architecture CI matrix and backend feature tests. |
| `crates/signinum-cuda-runtime/src/bytes.rs` | GPU ABI marker impls and typed host/device struct byte views. | Typed values are copied as initialized plain-data bytes with compatible CUDA layout. | CUDA runtime unit tests and ABI compile checks. |
| `crates/signinum-cuda-runtime/src/context.rs` | CUDA context creation, module loading, pinned staging views, and context teardown. | CUDA handles outlive copied function/module uses and pinned host memory is uniquely borrowed. | CUDA runtime tests and strict GPU validation workflow. |
| `crates/signinum-cuda-runtime/src/driver.rs` | CUDA/NVTX driver FFI symbol loading and error-name conversion. | Loaded symbols match CUDA Driver API signatures and the driver library outlives function pointers. | CUDA runtime tests with real and fake driver paths. |
| `crates/signinum-cuda-runtime/src/execution.rs` | CUDA kernel parameter ABI, kernel launch, streams, events, and timing. | Kernel parameter pointers remain valid through launch and owned event/stream handles are destroyed once. | CUDA launch/parity tests and strict GPU validation workflow. |
| `crates/signinum-cuda-runtime/src/jpeg.rs` | CUDA JPEG kernel parameter ABI structs and entropy diagnostic launch metadata. | Kernel parameter structs are repr(C) plain-data values matching CUDA kernel signatures and validated launch ranges. | CUDA runtime JPEG kernel metadata tests and JPEG CUDA adapter tests. |
| `crates/signinum-cuda-runtime/src/memory.rs` | CUDA device/pinned memory allocation, copies, downloads, and buffer views. | Device ranges are bounds-checked and uninitialized host capacity is marked initialized only after successful copies. | CUDA memory tests and surface download tests. |
| `crates/signinum-cuda-runtime/src/tests.rs` | Test-only fake CUDA driver table construction. | Fake FFI signatures match production driver types. | CUDA runtime unit tests compile and execute fake-driver paths. |
| `crates/signinum-cuda-runtime/src/transcode.rs` | CUDA transcode host staging byte views for DCT-grid uploads. | Staging byte views match typed host slices and CUDA kernels receive bounded ranges. | CUDA transcode parity tests. |
| `crates/signinum-j2k-compare/src/grok.rs` | Grok FFI comparison harness. | External decoder pointers are checked for null and output buffers are sized before copy. | Optional Grok/OpenJPEG parity tests. |
| `crates/signinum-j2k-compare/src/openjpeg.rs` | OpenJPEG FFI comparison harness. | OpenJPEG stream/image lifetimes are paired with cleanup and component buffers are bounds-checked. | Optional Grok/OpenJPEG parity tests. |
| `crates/signinum-j2k-cuda/src/surface.rs` | CUDA surface batch downloads into preallocated host output buffers. | Batch range math is checked and host Vec length is set only after successful device copy. | CUDA surface and batch download tests. |
| `crates/signinum-j2k-metal/src/compute.rs` | Metal buffer pointer access, shader setup, and command submission. | Metal buffer contents are accessed only after size/alignment validation and command buffers are completed before host reads. | Metal runtime tests and GPU validation workflow. |
| `crates/signinum-j2k-metal/src/encode.rs` | Metal-backed codestream buffer views and encode validation helpers. | Encoded byte ranges are kernel-produced, bounds-checked, and copied before exposure. | Metal encode tests and parity benches. |
| `crates/signinum-j2k-metal/src/lib.rs` | Metal surface byte views and host/device transfer helpers. | Surface buffers carry valid residency/format metadata and checked host download lengths. | Metal surface tests. |
| `crates/signinum-j2k-metal/src/mct.rs` | SIMD-assisted color transform helpers. | SIMD loads/stores remain within row slices and preserve channel layout. | MCT unit tests and cross-backend parity. |
| `crates/signinum-jpeg-metal/src/compute.rs` | Metal buffer pointer access, shader setup, and command submission. | Buffer sizes, strides, and texture dimensions are validated before kernel dispatch/readback. | JPEG Metal viewport and compare tests. |
| `crates/signinum-jpeg-metal/src/lib.rs` | Metal surface byte views and host/device transfer helpers. | Metal surfaces expose only initialized bytes for validated dimensions/format. | JPEG Metal host surface and viewport tests. |
| `crates/signinum-jpeg-metal/tests/viewport.rs` | Test-only Metal texture byte access for viewport validation. | Test texture reads use known dimensions and synchronized command completion. | Viewport test itself. |
| `crates/signinum-jpeg/benches/encode_cpu.rs` | Benchmark allocation-counting global allocator wrapper. | Allocator wrapper forwards all layout requests unchanged and records counts atomically. | Benchmark build gate. |
| `crates/signinum-jpeg/benches/common/libjpeg_turbo.rs` | libjpeg-turbo FFI benchmark harness. | libjpeg-turbo handles are checked and destroyed and output buffers are sized by library-reported geometry. | Optional libjpeg-turbo comparison benches. |
| `crates/signinum-jpeg/src/backend/mod.rs` | Runtime backend dispatch and SIMD entry points. | SIMD entry points are called only when CPU features match the implementation. | Backend dispatch tests on x86_64/aarch64 CI. |
| `crates/signinum-jpeg/src/backend/neon.rs` | AArch64 NEON SIMD kernels. | NEON loads/stores stay within blocks/rows and match scalar reference math. | NEON hot-path and backend parity tests. |
| `crates/signinum-jpeg/src/backend/x86.rs` | x86 SIMD kernels and CPU feature-gated dispatch. | AVX/SSE paths require detected CPU features and match scalar reference math. | x86 backend parity tests. |
| `crates/signinum-jpeg/src/bench_support.rs` | Benchmark-only buffer and decoder helpers. | Benchmark buffers mirror production size checks before unsafe fast paths. | Benchmark build gate. |
| `crates/signinum-jpeg/src/decoder.rs` | Decoder buffer slicing and performance-critical pixel paths. | Entropy/pixel buffers are sized before unsafe slicing and decode caps are enforced before allocation. | JPEG regression, fuzz, and decode parity tests. |
| `crates/signinum-jpeg/src/entropy/sequential.rs` | Entropy decoder bitstream fast paths. | Bitstream reads never advance past scan data and block writes stay inside MCU buffers. | JPEG decode fuzz and regression tests. |
| `crates/signinum-jpeg/src/idct/avx2.rs` | AVX2 IDCT implementation. | AVX2 loads/stores use fixed block sizes and match scalar IDCT output. | IDCT parity tests. |
| `crates/signinum-jpeg/src/idct/neon.rs` | NEON IDCT implementation. | NEON loads/stores use fixed block sizes and match scalar IDCT output. | IDCT parity tests. |
| `crates/signinum-metal-support/src/lib.rs` | Metal command queue creation through Objective-C runtime calls. | Objective-C returned pointers are null-checked and buffer content helpers validate bounds/alignment. | Metal support and runtime tests. |
| `crates/signinum-transcode-cuda/src/cuda.rs` | CUDA accelerator and context integration for transcode stages. | CUDA surfaces are submitted only with matching residency and validated ranges. | CUDA transcode parity tests. |
| `crates/signinum-transcode-metal/src/metal.rs` | Metal accelerator integration for transcode stages. | Metal surfaces are submitted only with matching residency and validated ranges. | Metal transcode tests and validation workflow. |

## Review Expectations

Unsafe code must stay isolated behind safe APIs, validate pointer lengths before
forming slices, keep FFI ownership explicit, and prefer feature-gated dispatch
over unchecked CPU or GPU capability assumptions.
