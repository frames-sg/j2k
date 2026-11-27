# Architecture

`j2k` is the main crate. Its default backend is `Auto`, which can always fall
back to the CPU; an explicit CUDA or Metal request returns an error instead of
falling back. Decoding is strict by default. `DecodeSettings::lenient` applies
to one decode, never to a shared `J2kContext`, and only allows the JP2/JPH
metadata recoveries listed in its rustdoc. Codestream, bounds, overflow,
allocation, and resource-limit checks are the same in both modes.

J2K reads JPEG 2000 Part 1 and HTJ2K Part 15 codestreams and the JP2 and JPH
file formats. It doesn't support JPX (Part 2) beyond what JP2/JPH decoding
needs. The detailed list is in [Format support](../book/format-support.md).

## Crates

| Crate | Kind | What it does |
| --- | --- | --- |
| `j2k` | codec | The public JPEG 2000 / HTJ2K API: inspect, decode, encode, recode, prepared plans, and CPU batch decode. |
| `j2k-core` | core | Shared traits, errors, geometry, pixel formats, backend requests, and device-surface traits. |
| `j2k-types` | core | Types shared by `j2k`, the engine, and the adapters: encode-stage jobs, decode and prepared plans, wavelet results, limits, encode geometry. |
| `j2k-codec-math` | support | `no_std` constants and math shared by the CPU, CUDA, and Metal code so they produce identical results. |
| `j2k-native` | engine | The CPU JPEG 2000 / HTJ2K engine behind `j2k` and the adapters. |
| `j2k-jpeg` | codec | CPU JPEG codec. |
| `j2k-jpegls` | codec | JPEG-LS codec, as DICOM uses it. |
| `j2k-tilecodec` | codec | Tile decompression (Deflate, Zstd, LZW, PackBits) and the DICOM RLE frame codec. |
| `j2k-transcode` | transcode | JPEG to HTJ2K in the coefficient domain. |
| `j2k-profile` | support | Profiling rows used by the codec and adapter crates. |
| `j2k-cuda-runtime` | CUDA runtime | Codec-neutral CUDA Driver API runtime: modules and launches, contexts, streams, events, memory pools, pinned staging, and checks on external allocations. |
| `j2k-cuda-build-support` | build support | Stages the CUDA-Oxide kernel projects and packages their PTX for the engine build scripts. |
| `j2k-cuda-j2k-engine` | CUDA engine | JPEG 2000 / HTJ2K kernels and launches: transforms, Tier-1, dequantization, final store, encode, packetization. |
| `j2k-cuda-jpeg-engine` | CUDA engine | JPEG kernels and launches: baseline decode and encode, lossless encode. |
| `j2k-cuda-transcode-engine` | CUDA engine | Transcode kernels: transforms and quantization. |
| `j2k-cuda`, `j2k-jpeg-cuda`, `j2k-transcode-cuda` | CUDA adapter | The CUDA APIs users call: batch sessions, routing, resident output, caller-owned destinations. |
| `j2k-metal-support` | support | Metal device, queue, shader library, pipelines, checked buffers, and route labels. |
| `j2k-metal`, `j2k-jpeg-metal`, `j2k-transcode-metal` | Metal adapter | The Metal APIs users call. JPEG 2000 kernels live in the private `j2k-metal::engine` module. The transcode adapter has its own kernels and doesn't depend on `j2k-metal`. |
| `j2k-mpsgraph-support` | support | Codec-independent MPSGraph submission and lifetime handling. |
| `j2k-mpsgraph` | framework | Experimental bridge from Metal-resident batches to MPSGraph. |
| `j2k-ml` | framework | Burn tensors from JPEG, JPEG 2000, and HTJ2K batches. |
| `j2k-python` | binding | The `j2k` Python wheel (PyO3): decode to DLPack tensors, sRGB and region planning, encode, wavelet bands, optional CUDA or MPS output. Not on crates.io. |
| `j2k-c` | binding | C ABI, C++17 wrapper, and native packages: decode, row streaming, sRGB regions, encode. Not on crates.io. |
| `j2k-cli` | CLI | The `j2k` command: inspect, JPEG to HTJ2K transcode, and an `opj_decompress`-compatible decoder. |
| `j2k-test-support`, `j2k-transcode-test-support` | dev | Fixtures, benchmark inputs, and transcode oracles for tests, benches, and examples. |
| `j2k-alloc-probe` | dev | Counts allocations and requested bytes at codec boundaries. |
| `j2k-compare` | tooling | Comparison harnesses against other codecs. |
| `j2k-t803` | tooling | T.803 corpus, comparison, and report runner. Not on crates.io. |
| `xtask` | tooling | Repository automation (`cargo xtask`). |

## Dependency rules

Who owns what:

- `j2k` owns the public JPEG 2000 / HTJ2K API.
- `j2k`, `j2k-native`, `j2k-jpeg`, `j2k-cuda`, and `j2k-metal` do the parsing,
  preparation, grouping, decoding, scratch reuse, and device work.
- `j2k-ml` allocates Burn tensors and orders framework and codec work. It
  doesn't decode, transform, group, or normalize.
- `j2k-mpsgraph` keeps MPSGraph objects alive and allocates checked Metal
  destinations. It reuses `j2k-metal`'s grouping, kernels, and queue ordering,
  and never reads decoded pixels back to the host.
- `j2k-types` owns the wavelet band selection and result types, `j2k-native`
  extracts the bands, and `j2k` owns the public method.
- `j2k-python` owns interpreter conversion, DLPack lifetime transfer, and
  PyTorch allocation on CUDA and Metal.
- `j2k-c` owns raw C pointers, handles, ABI versioning, and packaging. It calls
  `j2k` for all codec work and implements no codec algorithm or sample
  conversion. Its module layout is in
  [`crates/j2k-c/docs/architecture.md`](../crates/j2k-c/docs/architecture.md).

Which way dependencies point:

- Codec crates may depend on `j2k-core` and support crates.
- Adapter crates may depend inward on codec, core, and support crates.
- Support and runtime crates don't depend on adapters, and transcode adapters
  don't depend on the full `j2k-cuda` or `j2k-metal` adapters.
- Published crates don't depend on test-support or comparator crates at
  runtime.
- CUDA codec stages run on J2K's own kernels.

The repo-lint test
`forbidden_architecture_edges_match_reviewed_migration_inventory` in
`xtask/tests/repo_lint_support/architecture_policy.rs` checks the direction
rules against `cargo metadata`.

### Third-party types in public APIs

A dependency whose types appear in a published crate's public API is a public
dependency. An application has to use a compatible version of it, so changing
its version range can break that application.

- `objc2`, `objc2-foundation`, `objc2-metal`, the Metal Performance Shaders
  crates, and `block2` are public dependencies of the Metal and MPSGraph
  crates, whose expert APIs take and return objc2 protocol objects. Keep each
  one a caret requirement inside one minor line, never an exact pin, so an
  application can share one copy with J2K. Moving to a new minor is a breaking
  change for those crates and needs a **Breaking:** changelog entry. The
  repo-lint test
  `metal_crates_use_the_objc2_stack_without_legacy_metal_or_block` checks the
  ranges.
- `moxcms` is private to `j2k` and pinned exactly. The sRGB decode charges the
  memory moxcms holds for an ICC conversion against the caller's limit, and
  that figure is measured for one moxcms version. Before changing the version,
  re-run `crates/j2k/tests/icc_memory.rs` and update
  `icc_conversion_working_bytes` in `crates/j2k/src/decode/srgb8.rs`.

## Crate dependency graph

Normal dependencies between workspace crates. The repo-lint test
`architecture_dependency_graph_matches_cargo_metadata` compares this block with
`cargo metadata`, so update it in the same pull request as a `Cargo.toml`
change.

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

The CPU path is the reference implementation. Every GPU route is tested
against it.

### Owned batches

The batch API returns homogeneous Gray, RGB, or RGBA groups of native `U8`,
`U16`, or `I16` samples in NCHW or NHWC order, and keeps each image's source
index. Straight and premultiplied alpha go in separate groups. Other component
layouts use the component-plane APIs. `j2k_jpeg::JpegBatchDecoder` decodes
complete 8-bit JPEGs into dense RGB8 groups the same way. Both use the
`BatchLayout` from `j2k-core`.

### Prepared plans

Supported inputs prepare one of two immutable plans owned by `j2k`:

- `PreparedHtj2kPlan`: per-tile HT cleanup and refinement geometry and byte
  ranges.
- `PreparedClassicPlan`: per-tile classic packet and code-block geometry and
  fragment ranges.

Both refer to the original `Arc<[u8]>` by offset, so sessions can reuse them
without parsing or copying the codestream again. Other inputs keep only
metadata and use the general CPU decoder.

### Scheduling

`CpuBatchDecoder` runs a bounded scheduler with retained worker workspaces.
Workers decode into disjoint regions of one buffer per group. Each decode is
counted at the native decoder's 512 MiB per-operation limit, and the scheduler
admits as many as fit in half of the memory available to the process
(physical memory, or a lower Linux cgroup limit), but never fewer than four.
When the caller sets no worker count and the host has more threads than
admitted workers, each worker also decodes its image in parallel on the shared
Rayon pool.

Within one image, `CpuDecodeParallelism::Auto` does the following:

- A multi-tile full-image decode whose components are all sampled 1x1 decodes
  its tiles concurrently. If any tile fails, the serial tile loop decodes the
  image again and reports its error.
- An HT-only tile decodes its code blocks in one parallel pass, one row of
  code blocks per task.
- The inverse wavelet transform reconstructs each level in row strips, top to
  bottom, keeping only a few rows live.
- The HT cleanup pass decodes magnitude and sign bits for two quads at a time
  in 4-lane vectors when a block's samples fit in 25 bits, which covers 16-bit
  medical images. Other blocks, and builds without the `simd` feature, use the
  scalar code.

All of these produce output identical to the serial path, bit for bit.

## CPU JPEG SIMD

`j2k-jpeg` picks its CPU backend once, when a decoder is built. The backend
value carries the CPU-feature token needed to run SIMD code: `Scalar`,
`Avx2(ExactAvx2)`, or `Neon(fearless_simd::Neon)`. Tests that want a specific
backend get the token the same way production code does. The `scalar-only`
feature always selects `Scalar`.

AArch64 kernels use `fearless_simd`'s safe kernel boundary. x86-64 uses a
private equivalent that needs only AVX2, because `fearless_simd::Avx2` requires
all of x86-64-v3 and would turn SIMD off on CPUs that have AVX2 without the
rest.

Dispatch and arithmetic are safe Rust. Raw vector loads and stores are confined
to private fixed-size array helpers and one x86 row cursor, in the modules
listed in `JPEG_SIMD_UNSAFE_BOUNDARIES` in `xtask/src/quality_commands.rs`.
`cargo xtask unsafe-audit` parses every file under the JPEG backend, IDCT, and
SIMD directories and fails if:

- any `unsafe fn` exists, or `unsafe` appears outside those modules;
- an `unsafe` block lacks a `SAFETY:` comment covering feature availability,
  bounds, alignment, aliasing, and initialization;
- the number of `unsafe` blocks goes above 24.

SIMD output is tested against the scalar output.

To add an unsafe memory helper, bring Criterion results from one host showing
it pays for itself: 95% confidence, 50 samples, a three-second warm-up, and a
ten-second measurement. If a run shows a slowdown above 2% on a
microbenchmark or 1% on end-to-end decode, rerun it with twice the
measurement time before drawing a conclusion.

## GPU adapters

GPU adapters can decode into GPU memory, including a buffer the caller owns.
A caller's buffer is the final output: pixels never go GPU to CPU to GPU, or
through a second GPU buffer. In `j2k-metal`, `J2kDecoder::decode_into_metal`
does this for one full image.

### CUDA

The adapters use `j2k-cuda-runtime` for the Driver API, module loading,
checked launch geometry, memory, and completion. `j2k-jpeg-cuda` calls
`j2k-cuda-jpeg-engine` and `j2k-cuda` calls `j2k-cuda-j2k-engine`. The kernels
are written in Rust as CUDA-Oxide projects, and Rust host code launches them
through the Driver API.

### Metal

The adapters use `j2k-metal-support` for the device, queue, shader library,
pipeline loading, checked buffer access, and route labels. All raw
Objective-C resource creation goes through it: it checks for nil before
creating a handle and retains autoreleased command resources in owned Rust
handles. The codec kernels live in the adapter crates.

`j2k-metal` has four lazily built kernel groups: decode, encode, profiling,
and small buffer operations. A `MetalRuntime` owns its device, queue, and
scratch pools, and shares the kernel groups with other runtimes on the same
device through a process-wide cache. Each group caches either its full set of
pipelines or the error it hit, so asking for decode doesn't compile the encode
or profiling pipelines, and a failure in one group leaves the others usable.
The prepared-plan caches belong to `MetalBackendSession`.

The Metal JPEG decoder has its own chapter:
[Metal JPEG decoder](metal-jpeg-decoder.md).

### Batches

`CudaBatchDecoder` and `MetalBatchDecoder` keep their device context, streams
or queues, pipelines, lookup tables, staging, and scratch pools across
submissions. HT entropy work is flattened across images, grouped by pass kind
(cleanup only, SigProp, MagRef), and split into bounded chunks. A device error
on one job reports which source image it came from. The final store writes one
dense destination per group, whether J2K allocated it or the caller supplied
it.

Batch decoding is tuned mainly for HTJ2K. Classic JPEG 2000 uses the same
grouping, destinations, and completion handling.

### What runs on a GPU

- Sample conversion for signed grayscale and color matches the CPU. Color
  components of mixed signedness are rejected.
- ROI maxshift (RGN markers) works when the coded bitplane count including the
  shift is at most 31. Classic blocks must fit in 64x64; HT geometry and pass
  limits depend on the backend. The tested profiles are under
  [GPU adapter support](../book/machine-learning.md#gpu-adapter-support), and
  what each backend runs is in [GPU decoding and encoding](../book/gpu.md).
- Mixed precision or signedness, other component counts, and precision above
  16 bits go through the CPU component-plane APIs.
- The CPU batch decoder handles images whose components are sampled
  differently, for every kind of request. On a GPU only Metal does, and only
  for full-image RGB (below); other such requests are a capability error.
- Under `Auto`, a shape stays on the CPU until it has been measured. See
  [Benchmarks](benchmarks.md#auto-routing-tables) for how routing tables are
  built and [Performance](../book/performance.md) for the results.

### Metal sampled-component decode

Full-image, unsigned, origin-zero RGB codestreams with subsampled components
and no multi-component transform decode on Metal through per-component plans
(`Image::build_component_grid_color_plan_with_context`). Matching component
graphs are stacked across the images of a group. A GPU pass replicates samples
onto the image grid, including partial blocks at odd edges, before the
RGB/RGBA packer. No ICC or YCbCr conversion is applied.

- Both the older full-image and tile-batch APIs and the owned batch decoder
  use this route.
- Multi-tile, offset, signed, alpha, and MCT images keep their other routes.
  A region or reduced-resolution request for a subsampled input is a
  capability error from the Metal batch decoder; decode it as a single image.
- One command buffer holds every intermediate resource until it completes, so
  stacking raises device working memory, and replication allocates full-size
  float component planes.

## Framework adapters

The user guide describes the
[Burn adapter](../book/machine-learning.md#burn) and the
[MPSGraph adapter](../book/machine-learning.md#mpsgraph).

`j2k-mpsgraph-support` owns an in-flight MPSGraph submission: the graph,
placeholder, feeds, targets and results, descriptor, and completion block. An
input guard keeps a standalone buffer or a resident batch alive until the
graph completes. Dropping a submission waits for completion without reading
results. Graph errors are copied out of `NSError` inside the callback. Codec
submission, tensor validation, and output allocation stay in the adapter.

## Geometry

Multi-tile encode keeps absolute tile-component coordinates through the
forward transform, subband partitioning, and packet progression, following the
reference-grid mappings in ITU-T T.800 (11/2015) B.5-B.7 and B.12 and the
symmetric extension and lifting in F.4. The CPU, Metal, and CUDA stages carry
the same origin and phase. The number of decomposition levels follows the
smallest edge tile.

Image parsing has no 60,000-pixel dimension cap. Grid arithmetic handles
clipped precinct and code-block endpoints at the `u32` boundary, and decode
allocations are bounded by structural and retained-memory budgets instead.

## Provenance

The codec started as an import of existing code (see
[NOTICES.md](../NOTICES.md)). Inherited parsing, entropy coding, packet
handling, inverse transforms, and shared tables remain under that attribution.
Later work was done with that source in view, so we make no clean-room claim.
