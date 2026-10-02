# Architecture

How the workspace is organized today.

`j2k` is the main crate. The default backend is `Auto`: the CPU path is always
available, and explicit CUDA or Metal requests return an error rather than
falling back. Decoding is strict by default. Lenient settings apply to one
image at a time (never to a shared `J2kContext`) and only allow the JP2/JPH
metadata recoveries listed under `DecodeSettings::lenient`; codestream, bounds,
overflow, allocation, and resource-limit checks are the same in both modes.
`J2kDecodeWarning::LenientMetadataRecovery` is reported only when a recovery
actually happened.

Supported inputs are JPEG 2000 Part 1 codestreams, JP2 files, HTJ2K Part 15
codestreams, and JPH files. JPX (Part 2) extensions are not supported except
where JP2/JPH decoding needs them. The detailed list is in
[`docs/public-support.md`](public-support.md).

## Crate classes

| Crate | Class | Role |
| --- | --- | --- |
| `j2k` | public codec | Primary user-facing JPEG 2000 / HTJ2K API, including owned preparation and CPU batch decode. |
| `j2k-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface traits. |
| `j2k-types` | core | Shared encode-stage types used by `j2k`, the native engine, and the adapters. |
| `j2k-codec-math` | support | `no_std` constants and math tables shared by the CPU, CUDA-Oxide, and Metal code so they produce identical results. |
| `j2k-jpeg`, `j2k-tilecodec` | codec | CPU/native codec implementations and stable codec APIs. |
| `j2k-native` | engine | Native JPEG 2000 / HTJ2K engine used by J2K APIs and adapter validation. |
| `j2k-profile`, `j2k-metal-support` | support | Runtime/profile helpers used by adapters and codec crates. |
| `j2k-cuda-runtime` | CUDA runtime | Codec-neutral CUDA Driver API integration, checked generic module/kernel launch, context/stream/event lifecycle, memory pools, pinned staging, diagnostics, completion, and guarded external-allocation validation shared by CUDA engines. |
| `j2k-cuda-build-support` | build support | Internal shared CUDA-Oxide project staging, toolchain invocation, placeholder policy, and PTX packaging for codec engine build scripts. |
| `j2k-cuda-j2k-engine` | CUDA engine | Internal J2K/HTJ2K/ML operations on a borrowed low-level CUDA context: transforms, Tier-1, dequantization, final store, encode, packetization, kernel ABI, validation, launch orchestration, and CUDA-Oxide packaging. |
| `j2k-cuda-jpeg-engine` | CUDA engine | Internal JPEG operations on a borrowed low-level CUDA context: JPEG plans, validation, host allocation, kernel ABI byte views, CUDA-Oxide projects, and launch orchestration. |
| `j2k-cuda-transcode-engine` | CUDA engine | Internal coefficient-domain transcode operations: reversible/irreversible transforms and quantization, validation, launch geometry, stage timings, and CUDA-Oxide packaging. |
| `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` | CUDA adapter | Codec-facing CUDA APIs, persistent batch sessions, route policy, resident output, and validated caller-owned destinations for supported paths. |
| `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` | Metal adapter | macOS Metal adapters built on `j2k-metal-support`. J2K transforms, Tier-1, packetization, store, and resident encode/decode are in the private `j2k-metal::engine` module; the transcode adapter has its own coefficient-domain kernels and does not depend on `j2k-metal`. |
| `j2k-ml` | framework integration | Thin Burn allocation and codec-interop adapter for owned integer batch output. |
| `j2k-mpsgraph-support` | support | Codec-independent graph submission, callback/error ownership, completion, and retained input lifetime shared with JPEG XR. |
| `j2k-mpsgraph` | framework integration | Experimental Apple Silicon direct bridge from Metal-resident native integer batches to static rank-four MPSGraph programs. |
| `j2k-transcode` | transcode | JPEG-to-HTJ2K coefficient-domain transcode algorithms and shared types. |
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
- `j2k-mpsgraph` may retain MPSGraph objects and allocate validated external
  Metal destinations. It reuses `j2k-metal` grouping, kernels, and queue
  ordering and must not add decoded-pixel readback/upload staging.
- Codec crates may depend on `j2k-core` and support crates.
- Adapter crates may depend inward on codec/core/support crates.
- Support crates must not depend on adapters.
- Test support and comparator crates must not become runtime dependencies of
  stable public crates.
- CUDA codec stages run on J2K's own CUDA kernels.

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
j2k-mpsgraph -> j2k, j2k-core, j2k-metal, j2k-metal-support, j2k-mpsgraph-support
j2k-transcode-metal -> j2k-codec-math, j2k-core, j2k-metal-support, j2k-transcode, j2k-types
j2k-transcode-cuda -> j2k-core, j2k-cuda-j2k-engine, j2k-cuda-runtime, j2k-cuda-transcode-engine, j2k-native, j2k-transcode
j2k-cli -> j2k, j2k-jpeg, j2k-transcode
xtask -> j2k, j2k-codec-math, j2k-compare, j2k-native, j2k-profile, j2k-test-support
```

## Backend policy

The CPU path is the reference implementation. The owned fast-batch API returns
homogeneous Gray/RGB/RGBA groups as native `U8`, `U16`, or `I16` samples in
NCHW or NHWC order and preserves source indices. Straight and premultiplied
alpha are distinct grouping keys. Preparation retains the caller-owned
codestream bytes and reusable decode plans without duplicating the codestream.
Broader component layouts remain on the component-plane APIs.

### CPU JPEG SIMD boundary

`j2k-jpeg` selects its CPU backend once while constructing a decoder. The
internal backend value holds the CPU-feature token needed to run SIMD code:
`Scalar`, `Avx2(ExactAvx2)`, or `Neon(fearless_simd::Neon)`. The diagnostic
backend kind cannot be used to run SIMD code, and tests that want a specific
backend must get the token the same way production does. The `scalar-only`
feature always selects `Scalar`.

AArch64 entry kernels use the safe `fearless_simd 1.0` kernel boundary. x86-64
uses a private equivalent that requires only AVX2. `fearless_simd::Avx2`
requires all of x86-64-v3 (FMA, BMI, and more), so using it would disable SIMD
on CPUs that have AVX2 and OS support for it but not the rest of v3.

Dispatch, benchmark adapters, and arithmetic helpers are safe Rust. Raw vector
memory operations are confined to private fixed-size array leaves and one x86
row cursor carrying the AVX2 capability and source-slice lifetimes. The cursor
constructor fixes its readable extent to the shortest complete eight-byte
chunk count, and private state advances all three rows together. These leaves
use unaligned-capable operations and preserve Rust's reference aliasing and
initialization rules. The IDCT and color paths use the same integer
arithmetic, chunk sizes, edge handling, crop rules, and scalar tails as the
scalar code.

The unsafe audit parses every Rust file under the JPEG backend, IDCT, and SIMD
directories. It rejects `unsafe fn` and any `unsafe` outside the private
feature/memory modules, requires a five-part SAFETY comment on each block, and
allows at most 24 blocks; there are currently 10 and no `unsafe fn`. SIMD
output is tested against the scalar output. Performance changes are compared
on the same host with Criterion (95% confidence, 50 samples, three-second
warm-up, ten-second measurement); a slowdown above 2% for a microbenchmark or
1% for end-to-end decode is rerun with twice the measurement time before a new
unsafe memory helper is accepted.

GPU adapters can decode into GPU memory, including caller-owned buffers, but
explicit GPU requests return an unsupported error instead of falling back to
the CPU. When decoding into a caller's buffer, that buffer is the final output:
pixels never go GPU-to-CPU-to-GPU or through a second GPU buffer.

CUDA adapters use `j2k-cuda-runtime` for the shared CUDA Driver API runtime,
generic module loading, checked launch geometry, memory, and completion.
`j2k-jpeg-cuda` calls into `j2k-cuda-jpeg-engine` for JPEG plans, validation,
CUDA-Oxide packaging, and launch orchestration. `j2k-cuda` calls into
`j2k-cuda-j2k-engine` for J2K-ML, transforms, classic Tier-1 decode, HTJ2K
decode, dequantization, the final store, encode, and packetization, including
queued completion and PTX packaging. The CUDA kernels are compiled from CUDA
Oxide projects; the Rust host code drives them through the Driver API.
The Burn CUDA upload adapter waits for the decoded GPU output, copies the
pixels to host memory, and creates the Burn tensor with Burn's public upload
API.

Metal adapters use `j2k-metal-support` for device, queue, shader-library,
pipeline loading, checked buffer access, and route-label helpers. All raw
Objective-C resource construction goes through it: it checks for nil before
creating a handle, and retains autoreleased command resources into owned Rust
handles before returning. Codec kernels live in the adapter crates. The
`j2k-ml` Metal upload adapter copies the decoded pixels from GPU memory to the
host and creates the Burn tensor with Burn's public upload API.

Batch decoding is optimized mainly for HTJ2K; classic JPEG 2000 uses the same
grouping, destination, and completion behavior and is covered by regression
tests. Supported fast-batch inputs prepare one of two immutable plans owned by
`j2k`. `PreparedHtj2kPlan` retains per-tile HT cleanup/refinement geometry and
byte ranges; `PreparedClassicPlan` retains per-tile classic packet/code-block
geometry plus ordered fragment ranges. Both reference compressed payloads by
offset from the original `Arc<[u8]>` and are reusable across sessions without
parsing again or copying the codestream. Other inputs keep only metadata and
use the general CPU decoder if it supports them.

`CpuBatchDecoder` uses a bounded scheduler with retained worker workspaces. It
allocates one typed buffer per homogeneous group and lets workers decode into
disjoint image regions, avoiding per-image output owners and a final batch
assembly copy. `CudaBatchDecoder` and `MetalBatchDecoder` likewise retain their
device context, streams or queues, modules or pipelines, lookup tables, events,
staging owners, and scratch pools across submissions.

The Burn adapter's scope and tests are described in [`docs/j2k-ml.md`](j2k-ml.md).

HT entropy work is flattened across images, bucketed by cleanup-only,
SigProp, and MagRef work, and split into bounded pass-homogeneous chunks. Chunk
status retains the original source identity where the device reports a failing
job, while the final native store still writes one dense destination per
homogeneous group. Resident and external-destination routes share the codec
pipeline; an external destination receives the final samples without a decoded
host transfer or an intermediate final device allocation.

GPU prepared decode returns an error for nonzero ROI maxshift (RGN markers)
and for shapes a backend's plans cannot describe. Subsampled components, mixed
precision or signedness, arbitrary component counts, and precision above 16
bits use the CPU component-plane APIs or return a fast-batch
representability error. A shape stays off the GPU until it has been measured.
Machines, dates, and measurements are in
[`docs/benchmark-evidence.md`](benchmark-evidence.md).

## Graph submission lifetime

The J2K and JPEG XR adapters share `j2k-mpsgraph-support`. It retains the graph,
placeholder, feed/target/result dictionaries, descriptor and completion block as
one in-flight owner. A concrete input guard retains either a standalone buffer or
a resident batch lease until graph completion. Codec submission, tensor validation,
metadata and output allocation remain in each adapter. Drop waits without invoking
result extraction, so cleanup has no output-vector allocation or metadata-consumption
precondition. Graph errors are copied out of NSError during the callback.

The shared crate adds no codec dependency or new Objective-C version. Its input
owner type covers exactly the two storage lifetimes above. Both adapters kept
their public APIs, and their device tests cover identity outputs, early drop,
and repeated submissions.

## Metal kernel initialization

`MetalRuntime` owns the device, queue, scratch pools, prepared-plan cache, and four
lazy kernel groups: decode, encode, profiling and small buffer operations. Each
OnceLock caches either a fully initialized immutable group or its typed failure.
Pipeline handles and HT lookup buffers belong to the stage that uses them. Requesting
decode does not compile or allocate encode/profile resources, and an optional group's
failure leaves decode and buffer validation usable. Leaf IDWT, store and Tier-1
dispatchers receive initialized kernel references rather than initializing inside
transform loops.

Shader compilation follows the same boundaries. Forward and inverse MCT code have
separate sources with one ABI owner; production classic encoding excludes the profile
entrypoints and their token-planning implementation. Device tests exercise actual
pipeline creation and resource queries, cached reuse, failure isolation, and decode/
encode parity. This change affects startup and resource ownership, not throughput.

## Metal sampled-component decode

Full unsigned, origin-zero RGB codestreams without MCT can use native component
grids when the existing full-resolution direct plan rejects subsampling. The
`Image::build_component_grid_color_plan_with_context` returns the image
dimensions, compact component plans, and sampling factors. The existing direct
plan is unchanged. Multi-tile, offset,
signed, alpha, MCT, and reduced/region geometry retain their existing routes.

The legacy Metal full-image and tile-batch APIs prepare these plans before
submission. Matching component graphs are stacked across images, including
classic entropy jobs, IDWT, and component stores. Geometry that cannot be stacked
uses independent resident graphs in the same command buffer. A small GPU pass
replicates samples onto the image grid, including incomplete blocks at odd image
edges, before the existing RGB/RGBA packer. It performs no ICC or YCbCr transform;
those remain the caller's job. The owned dense batch API still rejects
subsampling.

One command buffer retains all intermediate resources until completion. Existing
checked allocation and status-retirement paths own scratch buffers and propagate
errors. Stacking increases simultaneous device working memory compared with
per-block readback, and expansion allocates full-sized float component planes.
The CPU source decoder remains available. The focused sampled tests compare
every output byte, exercise odd dimensions and several sampling factors, and
require one submission for both distinct and repeated inputs.

The opt-in `local_sampled_color_batch_characterization` test accepts eight
extracted DICOM codestreams per level through `J2K_SAMPLED_CORPUS`. It compares
every RGB byte to CPU output outside the timer and retains the existing two-code
value limit for irreversible images. Initial measurements showed that batching
removed the large synchronous fallback cost, while classic GPU entropy remained
slower than CPU decoding on 256-pixel tiles. The general entropy kernel was
slower than the existing plain kernel and is not used.
