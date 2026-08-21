# Architecture

This document records current workspace boundaries. It is not a roadmap.

The public crate release centers on `j2k`. Runtime backend selection defaults to `Auto`:
CPU remains the portable baseline, and explicit CUDA or Metal
requests are strict. Decode settings are strict by default. Explicit lenient
settings are retained per image, never on shared `J2kContext`, and are limited
to the JP2/JPH metadata recoveries documented by `DecodeSettings::lenient`.
Codestream, bounds, overflow, allocation, and resource-limit validation remain
strict in both modes. Decode outcomes surface
`J2kDecodeWarning::LenientMetadataRecovery` only when a recovery actually
occurred. The living support boundary is maintained in
[`docs/public-support.md`](public-support.md).

The codec support boundary is JPEG 2000 Part 1 codestreams, JP2 still-image
files, HTJ2K Part 15 codestreams, and JPH still-image files. JPX / JPEG 2000
Part 2 extensions are out of scope unless required for standard JP2/JPH
still-image correctness. Keep row-level status synchronized with
[`docs/public-support.md`](public-support.md).

## Crate classes

| Crate | Class | Role |
| --- | --- | --- |
| `j2k` | public codec | Primary user-facing JPEG 2000 / HTJ2K API, including owned preparation and CPU batch decode. |
| `j2k-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface contracts. |
| `j2k-types` | core | Shared encode-stage contracts and semver-visible value types used by the facade, native engine, and adapters. |
| `j2k-codec-math` | support | No-std shared constants and pure math tables for CPU, CUDA-Oxide, and Metal parity. |
| `j2k-jpeg`, `j2k-tilecodec` | codec | CPU/native codec implementations and stable codec APIs. |
| `j2k-native` | engine | Native JPEG 2000 / HTJ2K engine used by J2K APIs and adapter validation. |
| `j2k-profile`, `j2k-metal-support` | support | Runtime/profile helpers used by adapters and codec crates. |
| `j2k-cuda-runtime` | CUDA runtime | Codec-neutral CUDA Driver API integration, checked generic module/kernel launch, context/stream/event lifecycle, memory pools, pinned staging, diagnostics, completion, and guarded external-allocation validation shared by CUDA engines. |
| `j2k-cuda-build-support` | build support | Internal shared CUDA-Oxide project staging, toolchain invocation, placeholder policy, and PTX packaging for codec engine build scripts. |
| `j2k-cuda-j2k-engine` | CUDA engine | Internal borrowed J2K/HTJ2K/ML operation boundary over the low-level CUDA context; owns transform, Tier-1, dequantization, final-store, encode, packetization, ABI, validation, orchestration, tests, and CUDA-Oxide packaging. |
| `j2k-cuda-jpeg-engine` | CUDA engine | Internal borrowed JPEG operation boundary over the low-level CUDA context; owns JPEG plans, validation, host allocation, ABI byte views, CUDA-Oxide projects, and launch orchestration. |
| `j2k-cuda-transcode-engine` | CUDA engine | Internal borrowed coefficient-domain transcode boundary; owns reversible/irreversible transform and quantization models, validation, launch geometry, stage timings, tests, and CUDA-Oxide packaging. |
| `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` | CUDA adapter | Codec-facing CUDA APIs, persistent batch sessions, route policy, resident output, and validated caller-owned destinations for supported paths. |
| `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` | Metal adapter | macOS Metal adapters over `j2k-metal-support`; J2K transform, Tier-1, packetization, store, and resident encode/decode live behind the private `j2k-metal::engine` boundary, while transcode owns its coefficient-domain kernels without depending on the public J2K adapter. |
| `j2k-ml` | framework integration | Thin Burn allocation and codec-interop adapter for owned integer batch output. |
| `j2k-transcode` | transcode | JPEG-to-HTJ2K coefficient-domain transcode algorithms and shared contracts. |
| `j2k-cli` | CLI | Command-line inspection and JPEG-to-HTJ2K smoke transcode entry point. |
| `j2k-test-support`, `j2k-transcode-test-support` | dev helper | Shared fixture, benchmark input, and transcode oracle helpers for tests, benches, and examples. |
| `j2k-alloc-probe` | dev helper | Serial process-wide measurement of successful allocation calls and gross requested bytes at real codec boundaries. |
| `j2k-compare` | tooling | Comparator tooling. |
| `j2k-t803` | conformance tooling | Unpublished T.803 corpus, comparison, report, and adapter-IUT runner support. |
| `xtask` | workspace tool | Repository automation under `xtask/`. |

## Dependency rules

- The public `j2k` crate owns the JPEG 2000 / HTJ2K API surface.
- `j2k`, `j2k-native`, `j2k-cuda`, and `j2k-metal` own codec parsing,
  preparation, grouping, decoding, scratch reuse, and device execution.
- `j2k-ml` may allocate or materialize Burn tensors and establish safe
  framework/codec ordering. It must not duplicate entropy decode, transforms,
  grouping policy, normalization, or training behavior.
- Codec crates may depend on `j2k-core` and support crates.
- Adapter crates may depend inward on codec/core/support crates.
- Support crates must not depend on adapters.
- Test support and comparator crates must not become runtime dependencies of
  stable public crates.
- CUDA paths must use J2K-owned CUDA kernels for codec stages they claim to
  support.

## Crate dependency graph

```text
j2k-codec-math -> j2k-types
j2k -> j2k-core, j2k-native, j2k-types
j2k-native -> j2k-codec-math, j2k-types, j2k-profile
j2k-test-support -> j2k-core, j2k-native
j2k-transcode-test-support -> j2k-transcode, j2k-types
j2k-cuda -> j2k-core, j2k-cuda-j2k-engine, j2k-cuda-runtime, j2k, j2k-native, j2k-profile
j2k-metal -> j2k-codec-math, j2k-core, j2k, j2k-native, j2k-metal-support, j2k-profile, j2k-types
j2k-jpeg -> j2k-codec-math, j2k-core, j2k-profile
j2k-jpeg-cuda -> j2k-core, j2k-cuda-jpeg-engine, j2k-cuda-runtime, j2k-jpeg, j2k-profile
j2k-jpeg-metal -> j2k-core, j2k-jpeg, j2k-metal-support, j2k-profile
j2k-tilecodec -> j2k-core
j2k-compare -> j2k-core, j2k, j2k-native, j2k-test-support
j2k-t803 -> j2k, j2k-codec-math, j2k-compare, j2k-core, j2k-cuda, j2k-cuda-runtime, j2k-metal, j2k-native
j2k-transcode -> j2k-codec-math, j2k-core, j2k, j2k-native, j2k-jpeg, j2k-profile
j2k-metal-support -> j2k-core
j2k-cuda-runtime -> j2k-core
j2k-cuda-j2k-engine -> j2k-codec-math, j2k-core, j2k-cuda-runtime, j2k-types
j2k-cuda-jpeg-engine -> j2k-codec-math, j2k-core, j2k-cuda-runtime
j2k-cuda-transcode-engine -> j2k-core, j2k-cuda-runtime
j2k-ml -> j2k, j2k-cuda, j2k-metal, j2k-metal-support
j2k-transcode-metal -> j2k-codec-math, j2k-core, j2k-metal-support, j2k-transcode, j2k-types
j2k-transcode-cuda -> j2k-core, j2k-cuda-j2k-engine, j2k-cuda-runtime, j2k-cuda-transcode-engine, j2k-native, j2k-transcode
j2k-cli -> j2k, j2k-jpeg, j2k-transcode
xtask -> j2k, j2k-codec-math, j2k-compare, j2k-native, j2k-profile, j2k-test-support
```

## Backend policy

CPU is the correctness baseline. The owned fast-batch surface returns
homogeneous Gray/RGB/RGBA groups as native `U8`, `U16`, or `I16` samples in
NCHW or NHWC order and preserves source indices. Straight and premultiplied
alpha are distinct grouping keys. Preparation retains the caller-owned
codestream bytes and reusable decode plans without duplicating the codestream.
Broader component layouts remain on the component-plane APIs.

### CPU JPEG SIMD boundary

`j2k-jpeg` selects its CPU backend once while constructing a decoder. The
internal backend value carries the capability needed to execute accelerated
code: `Scalar`, `Avx2(ExactAvx2)`, or `Neon(fearless_simd::Neon)`. A diagnostic
backend kind is not executable authority, and tests requesting a specialization
must obtain the same runtime token as production. The `scalar-only` feature
always selects `Scalar`.

AArch64 entry kernels use the safe `fearless_simd 0.7` kernel boundary. x86-64
uses a project-private equivalent that enables exactly AVX2. This distinction
is intentional: the `fearless_simd::Avx2` token in 0.7 represents the broader
x86-64-v3 feature set, including FMA, BMI, and other features. Requiring that
token would silently remove acceleration from CPUs that satisfy the decoder's
existing AVX2-plus-operating-system-state contract but not all of v3.

Dispatch, benchmark adapters, and arithmetic helpers are safe Rust. Raw vector
memory operations are confined to private fixed-size array leaves and one x86
row cursor carrying the AVX2 capability and source-slice lifetimes. The cursor
constructor fixes its readable extent to the shortest complete eight-byte
chunk count, and private state advances all three rows together. These leaves
use unaligned-capable operations and preserve Rust's reference aliasing and
initialization rules. The optimized IDCT and color paths retain their existing
integer arithmetic, chunk sizes, edge repair, crop rules, and scalar tails;
this refactor does not substitute a new portable-SIMD algorithm.

The unsafe-audit task parses every Rust source under the JPEG backend, IDCT,
and SIMD directories. It rejects `unsafe fn`, rejects unsafe outside the
private feature/memory modules, requires a five-part safety proof for each
remaining block, and caps the explicit production SIMD boundary at 24 blocks.
The refactored boundary currently contains 10 blocks and no `unsafe fn`.
SIMD output remains differentially tested against scalar output. Performance
acceptance uses same-host Criterion comparisons at 95% confidence, 50 samples,
a three-second warm-up, and a ten-second measurement; a confidence-bound
slowdown above 2% for a microbenchmark or 1% for end-to-end decode is repeated
with twice the measurement time before accepting a narrow unsafe memory leaf.

Device adapters can add resident outputs and validated caller-owned
destinations, but explicit requests must return unsupported errors instead of
falling back to CPU staging. A direct external destination is the final output
allocation: decoded pixels must not cross a GPU-to-CPU-to-GPU path or a second
device output merely for framework integration.

CUDA adapters use `j2k-cuda-runtime` for the shared CUDA Driver API runtime,
generic module loading, checked launch geometry, memory, and completion.
`j2k-jpeg-cuda` enters codec operations through the internal
`j2k-cuda-jpeg-engine` boundary, which owns JPEG plans, validation, CUDA-Oxide
packaging, and launch orchestration without changing adapter APIs. `j2k-cuda`
likewise binds through `j2k-cuda-j2k-engine`; that engine already owns J2K-ML,
classic Tier-1 decode, HTJ2K decode, and J2K dequantization, including queued
completion and PTX packaging, while the remaining transform, store, encode,
and packetization slices are migrated. Product CUDA codec kernels are
generated from CUDA Oxide projects while Rust host code retains Driver API
orchestration. `cuda-runtime` support is an implementation dependency, not
proof of NVIDIA performance.
The Burn CUDA upload adapter waits for codec-owned resident output, copies its
dense decoded pixels to host staging, and constructs the Burn tensor through
the framework's ordinary public upload API.

Metal adapters use `j2k-metal-support` for device, queue, shader-library,
pipeline loading, checked buffer access, and route-label helpers. It is the
codec-side raw Objective-C resource-construction boundary: nil is checked
before any codec resource handle is formed, and autoreleased command resources
are retained into owned Rust handles before return. Codec-specific kernels stay
in codec adapter crates. The `j2k-ml` Metal upload adapter reads the validated
dense range from completed codec-owned resident storage into host staging and
constructs the Burn tensor through the framework's ordinary public upload API.

HTJ2K is the optimized batch priority; classic JPEG 2000 shares the public
grouping, destination, and completion contracts and remains regression-covered.
Supported fast-batch inputs prepare one of two immutable, facade-owned plan
views. `PreparedHtj2kPlan` retains per-tile HT cleanup/refinement geometry and
byte ranges; `PreparedClassicPlan` retains per-tile classic packet/code-block
geometry plus ordered fragment ranges. Both reference compressed payloads by
offset from the original `Arc<[u8]>` and are reusable across sessions without
reparsing or duplicating the codestream. Inputs outside those retained-plan
boundaries keep metadata only and use the general CPU decoder when that broader
codec path supports them.

`CpuBatchDecoder` uses a bounded scheduler with retained worker workspaces. It
allocates one typed buffer per homogeneous group and lets workers decode into
disjoint image regions, avoiding per-image output owners and a final batch
assembly copy. `CudaBatchDecoder` and `MetalBatchDecoder` likewise retain their
device context, streams or queues, modules or pipelines, lookup tables, events,
staging owners, and scratch pools across submissions.

The exact experimental framework-adapter boundary and its focused correctness
evidence are maintained in [`docs/j2k-ml.md`](j2k-ml.md). Architecture does not
duplicate the hardware validation matrix.

HT entropy work is flattened across images, bucketed by cleanup-only,
SigProp, and MagRef work, and split into bounded pass-homogeneous chunks. Chunk
status retains the original source identity where the device reports a failing
job, while the final native store still writes one dense destination per
homogeneous group. Resident and external-destination routes share the codec
pipeline; an external destination receives the final samples without a decoded
host transfer or an intermediate final device allocation.

GPU prepared decode remains fail-closed for nonzero ROI maxshift from codestream
RGN markers and for shapes outside a backend's retained-plan boundary.
Subsampled components, mixed precision or signedness, arbitrary component
counts, and precision above 16 bits remain on the CPU component-plane APIs or
return a structured fast-batch representability error. Backend selection stays
explicit until the requested shape has appropriate evidence. Dated machines,
measurements, and publication qualifications are owned by
[`docs/benchmark-evidence.md`](benchmark-evidence.md).
