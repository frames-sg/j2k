# Architecture

How the workspace is organized.

`j2k` is the main crate. The default backend is `Auto`: the CPU path is always
available, and an explicit CUDA or Metal request returns an error rather than
falling back. Decoding is strict by default. Lenient settings apply to one
image at a time, never to a shared `J2kContext`, and allow only the JP2/JPH
metadata recoveries listed under `DecodeSettings::lenient`; codestream, bounds,
overflow, allocation, and resource-limit checks are the same in both modes.

Supported inputs are JPEG 2000 Part 1 codestreams, JP2 files, HTJ2K Part 15
codestreams, and JPH files. JPX (Part 2) extensions are not supported except
where JP2/JPH decoding needs them. The detailed list is in
[`docs/public-support.md`](public-support.md).

## Crate classes

| Crate | Class | Role |
| --- | --- | --- |
| `j2k` | public codec | Primary JPEG 2000 / HTJ2K API: inspect, decode, encode, recode, owned preparation, and CPU batch decode. |
| `j2k-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface traits. |
| `j2k-types` | core | Types shared by `j2k`, the native engine, and the adapters: encode-stage jobs, decode and prepared plans, wavelet results, limits, and encode geometry. |
| `j2k-codec-math` | support | `no_std` constants and math shared by the CPU, CUDA-Oxide, and Metal code so they produce identical results. |
| `j2k-jpeg`, `j2k-tilecodec` | codec | CPU JPEG codec and tile decompression (Deflate, Zstd, LZW). |
| `j2k-native` | engine | Native JPEG 2000 / HTJ2K engine behind `j2k` and the adapters. |
| `j2k-profile`, `j2k-metal-support` | support | Profiling rows and Metal runtime helpers used by adapters and codec crates. |
| `j2k-cuda-runtime` | CUDA runtime | Codec-neutral CUDA Driver API runtime: module and kernel launch, contexts, streams, events, memory pools, pinned staging, and validation of external allocations. |
| `j2k-cuda-build-support` | build support | Stages CUDA-Oxide projects and packages their PTX for the engine build scripts. |
| `j2k-cuda-j2k-engine` | CUDA engine | JPEG 2000 / HTJ2K kernels and launch orchestration: transforms, Tier-1, dequantization, final store, encode, packetization. |
| `j2k-cuda-jpeg-engine` | CUDA engine | JPEG kernels and launch orchestration: baseline decode and encode, lossless encode. |
| `j2k-cuda-transcode-engine` | CUDA engine | Coefficient-domain transcode kernels: transforms and quantization. |
| `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda` | CUDA adapter | Codec-facing CUDA APIs: batch sessions, route policy, resident output, and caller-owned destinations. |
| `j2k-jpeg-metal`, `j2k-metal`, `j2k-transcode-metal` | Metal adapter | macOS Metal adapters built on `j2k-metal-support`. The JPEG 2000 kernels are in the private `j2k-metal::engine` module. The transcode adapter has its own kernels and does not depend on `j2k-metal`. |
| `j2k-ml` | framework integration | Burn tensors from JPEG, JPEG 2000, and HTJ2K batches. |
| `j2k-python` | framework integration | PyO3 extension (the `j2k` wheel): decode to DLPack tensors, sRGB and region planning, encode, wavelet bands, and optional CUDA or MPS output. Unpublished; no Python dependency in ordinary Rust builds. |
| `j2k-c` | C/C++ integration | CPU C ABI, C++17 wrapper, and native packages: decode, row streaming, sRGB regions, and encode. Unpublished. |
| `j2k-mpsgraph-support` | support | Codec-independent MPSGraph submission and lifetime handling. |
| `j2k-mpsgraph` | framework integration | Experimental Apple Silicon bridge from Metal-resident batches to MPSGraph. |
| `j2k-transcode` | transcode | JPEG-to-HTJ2K coefficient-domain transcode. |
| `j2k-cli` | CLI | The `j2k` command: inspect, JPEG-to-HTJ2K transcode, and an `opj_decompress`-compatible decoder. |
| `j2k-test-support`, `j2k-transcode-test-support` | dev helper | Fixtures, benchmark inputs, and transcode oracles for tests, benches, and examples. |
| `j2k-alloc-probe` | dev helper | Counts allocation calls and requested bytes at codec boundaries. |
| `j2k-compare` | tooling | Comparison harnesses against other codecs. |
| `j2k-t803` | conformance tooling | T.803 corpus, comparison, and report runner. Unpublished. |
| `xtask` | workspace tool | Repository automation. |

## Dependency rules

- `j2k` owns the public JPEG 2000 / HTJ2K API.
- `j2k`, `j2k-native`, `j2k-jpeg`, `j2k-cuda`, and `j2k-metal` own parsing,
  preparation, grouping, decoding, scratch reuse, and device execution.
- `j2k-ml` allocates Burn tensors and orders framework and codec work. It does
  not decode, transform, group, or normalize.
- `j2k-mpsgraph` retains MPSGraph objects and allocates validated Metal
  destinations. It reuses `j2k-metal` grouping, kernels, and queue ordering,
  and does not read decoded pixels back to the host.
- `j2k-types` owns the wavelet selection and result types, `j2k-native`
  extracts the bands, and `j2k` owns the public method.
- `j2k-python` owns interpreter conversion, DLPack lifetime transfer, and the
  PyTorch allocation boundary on CUDA and Metal.
- `j2k-c` owns raw C pointers, handle ownership, ABI versioning, and native
  packaging. It calls `j2k` for all codec work and implements no codec
  algorithm or sample transformation.
- Codec crates may depend on `j2k-core` and support crates.
- Adapter crates may depend inward on codec, core, and support crates.
- Support crates must not depend on adapters.
- Test-support and comparator crates must not be runtime dependencies of
  published crates.
- CUDA codec stages run on J2K's own kernels.

## Crate dependency graph

```text
j2k-codec-math -> j2k-types
j2k -> j2k-core, j2k-native, j2k-types
j2k-native -> j2k-codec-math, j2k-types, j2k-profile
j2k-test-support -> j2k-codec-math, j2k-core, j2k-native
j2k-transcode-test-support -> j2k-codec-math, j2k-jpeg, j2k-test-support, j2k-transcode, j2k-types
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
j2k-ml -> j2k, j2k-core, j2k-cuda, j2k-cuda-runtime, j2k-jpeg, j2k-metal, j2k-metal-support
j2k-python -> j2k, j2k-cuda, j2k-cuda-runtime, j2k-jpeg, j2k-metal, j2k-metal-support, j2k-native
j2k-c -> j2k
j2k-mpsgraph -> j2k, j2k-core, j2k-metal, j2k-metal-support, j2k-mpsgraph-support
j2k-transcode-metal -> j2k-codec-math, j2k-core, j2k-metal-support, j2k-transcode, j2k-types
j2k-transcode-cuda -> j2k-codec-math, j2k-core, j2k-cuda-j2k-engine, j2k-cuda-runtime, j2k-cuda-transcode-engine, j2k-native, j2k-transcode
j2k-cli -> j2k, j2k-jpeg, j2k-transcode
xtask -> j2k, j2k-codec-math, j2k-compare, j2k-native, j2k-profile, j2k-test-support
```

## CPU decode

The CPU path is the reference implementation.

**Owned batches.** The batch API returns homogeneous Gray, RGB, or RGBA groups
as native `U8`, `U16`, or `I16` samples in NCHW or NHWC order and preserves
source indices. Straight and premultiplied alpha are separate groups. Other
component layouts use the component-plane APIs.
`j2k_jpeg::JpegBatchDecoder` decodes complete 8-bit JPEGs into dense RGB8
groups in the same layouts. Both use the `BatchLayout` defined in `j2k-core`.

**Prepared plans.** Supported inputs prepare one of two immutable plans owned
by `j2k`: `PreparedHtj2kPlan` (per-tile HT cleanup and refinement geometry and
byte ranges) or `PreparedClassicPlan` (per-tile classic packet and code-block
geometry and fragment ranges). Both refer to the original `Arc<[u8]>` by
offset, so they can be reused across sessions without parsing or copying the
codestream. Other inputs keep only metadata and use the general CPU decoder.

**Scheduling.** `CpuBatchDecoder` uses a bounded scheduler with retained
worker workspaces, and workers decode into disjoint regions of one buffer per
group. It counts each decode at the native decoder's 512 MiB per-operation
bound and admits as many as fit in half of the memory available to the process
(physical memory, or a lower Linux cgroup limit), and at least four. When the
caller sets no worker count and the host has more threads than admitted
workers, each worker also decodes its image in parallel on the shared Rayon
pool.

**Within one image**, under `CpuDecodeParallelism::Auto`:

- A multi-tile full-image decode whose components are all sampled 1x1 decodes
  its tiles concurrently. On any failure the serial tile loop decodes the
  image again and reports its error.
- An HT-only tile decodes its code blocks in one parallel pass, one row of
  code blocks per task.
- The inverse wavelet transform reconstructs each level in row strips, top to
  bottom, keeping only a few rows live.
- The HT cleanup pass decodes magnitude and sign bits for two quads at a time
  in 4-lane vectors when a block's samples code at most 25 bits, which covers
  16-bit medical images. Other blocks, and builds without `simd`, use the
  scalar step.

All of these produce output bit-identical to the serial path.

## CPU JPEG SIMD boundary

`j2k-jpeg` selects its CPU backend once, when a decoder is constructed. The
backend value holds the CPU-feature token needed to run SIMD code: `Scalar`,
`Avx2(ExactAvx2)`, or `Neon(fearless_simd::Neon)`. Tests that want a specific
backend get the token the way production does. The `scalar-only` feature
always selects `Scalar`.

AArch64 kernels use the safe `fearless_simd 1.0` kernel boundary. x86-64 uses
a private equivalent that needs only AVX2, because `fearless_simd::Avx2`
requires all of x86-64-v3 and would disable SIMD on CPUs that have AVX2
without the rest.

Dispatch and arithmetic are safe Rust. Raw vector memory operations are
confined to private fixed-size array helpers and one x86 row cursor. The
unsafe audit (`cargo xtask unsafe-audit`) parses every Rust file under the
JPEG backend, IDCT, and SIMD directories: it rejects `unsafe fn` and any
`unsafe` outside the private feature and memory modules, requires a five-part
SAFETY comment on each block, and caps the block count at 24. SIMD output is
tested against the scalar output.

A new unsafe memory helper is accepted only with Criterion evidence from the
same host (95% confidence, 50 samples, three-second warm-up, ten-second
measurement). A slowdown above 2% for a microbenchmark or 1% for end-to-end
decode is rerun with twice the measurement time first.

## GPU adapters

GPU adapters can decode into GPU memory, including a caller-owned buffer. A
caller's buffer is the final output: pixels never go GPU to CPU to GPU or
through a second GPU buffer. In `j2k-metal`, `J2kDecoder::decode_into_metal`
does this for one full image.

**CUDA.** The adapters use `j2k-cuda-runtime` for the Driver API runtime,
module loading, checked launch geometry, memory, and completion.
`j2k-jpeg-cuda` calls `j2k-cuda-jpeg-engine`, and `j2k-cuda` calls
`j2k-cuda-j2k-engine`. The kernels are compiled from CUDA-Oxide projects, and
Rust host code drives them through the Driver API.

**Metal.** The adapters use `j2k-metal-support` for the device, queue, shader
library, pipeline loading, checked buffer access, and route labels. All raw
Objective-C resource construction goes through it: it checks for nil before
creating a handle and retains autoreleased command resources into owned Rust
handles. Codec kernels live in the adapter crates.

`j2k-metal` has four lazily initialized kernel groups: decode, encode,
profiling, and small buffer operations. A `MetalRuntime` owns its device,
queue, and scratch pools, and shares the kernel groups with other runtimes on
the same device through a process-wide per-device cache. Each group caches
either a fully initialized set of pipelines or its typed failure, so asking
for decode does not compile encode or profiling pipelines, and a failure in
one group leaves the others usable. The direct prepared-plan caches belong to
`MetalBackendSession`.

**Batches.** `CudaBatchDecoder` and `MetalBatchDecoder` keep their device
context, streams or queues, pipelines, lookup tables, staging, and scratch
pools across submissions. HT entropy work is flattened across images, grouped
by pass kind (cleanup only, SigProp, MagRef), and split into bounded chunks.
A device error on one job keeps its source image's identity. The final store
writes one dense destination per group, whether codec-owned or supplied by
the caller.

**What a GPU decodes.**

- Sample conversion for signed grayscale and color sources matches the CPU.
  Color components of mixed signedness are rejected.
- ROI maxshift (RGN markers) is supported when the coded bitplane count
  including the shift is at most 31. Classic blocks must fit in 64x64; HT
  geometry and pass limits depend on the backend. Tested profiles are in
  [`j2k-ml.md`](j2k-ml.md#gpu-adapter-support).
- Mixed precision or signedness, other component counts, and precision above
  16 bits use the CPU component-plane APIs.
- Images whose components differ in sampling batch on the CPU for every
  request kind. Metal batches them for full-image RGB (below); other such
  requests are a capability error on a GPU.
- A shape stays off the GPU under `Auto` until it has been measured. Machines,
  dates, and measurements are in
  [`docs/benchmark-evidence.md`](benchmark-evidence.md).

Batch decoding is optimized mainly for HTJ2K. Classic JPEG 2000 uses the same
grouping, destinations, and completion behavior.

### Metal sampled-component decode

Full-image, unsigned, origin-zero RGB codestreams without a multi-component
transform, whose components are subsampled, decode on Metal through
per-component plans (`Image::build_component_grid_color_plan_with_context`).
Matching component graphs are stacked across the images of a group. A GPU
pass replicates samples onto the image grid, including incomplete blocks at
odd edges, before the RGB/RGBA packer. No ICC or YCbCr conversion is applied.

- The legacy full-image and tile-batch APIs and the owned batch decoder both
  use this route.
- Multi-tile, offset, signed, alpha, and MCT images keep their other routes.
  A region or reduced request for a subsampled input is a capability error
  from the Metal batch decoder; use a single-image decode for it.
- One command buffer holds all intermediate resources until completion, so
  stacking raises device working memory, and replication allocates
  full-size float component planes.

## Framework adapters

The Burn adapter is described in [`docs/j2k-ml.md`](j2k-ml.md), and the
MPSGraph adapter in [`docs/j2k-mpsgraph.md`](j2k-mpsgraph.md).

`j2k-mpsgraph-support` owns an in-flight MPSGraph submission: the graph,
placeholder, feed, target and result dictionaries, descriptor, and completion
block. An input guard keeps either a standalone buffer or a resident batch
alive until the graph completes. Dropping a submission waits for completion
without extracting results. Graph errors are copied out of `NSError` during
the callback. Codec submission, tensor validation, and output allocation stay
in the adapter.

## Geometry and provenance

Multi-tile encode keeps absolute tile-component coordinates through the
forward transform, subband partitioning, and packet progression, following the
reference-grid mappings in ITU-T T.800 (11/2015) B.5-B.7 and B.12 and the
symmetric extension and lifting of F.4. The CPU, Metal, and CUDA stages carry
the same origin and phase. Decomposition depth follows the smallest edge
tile.

Image parsing has no 60,000-pixel dimension cap. Grid arithmetic handles
clipped precinct and code-block endpoints at the `u32` boundary, and decode
allocations are bounded by structural and retained-memory budgets.

The codec began as an import (see [`NOTICES.md`](../NOTICES.md)). Inherited
parsing, entropy coding, packet handling, inverse transforms, and shared
tables remain subject to that attribution. Later work was done with exposure
to that source, and the project makes no clean-room claim.
