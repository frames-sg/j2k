# Burn batch decoding with `j2k-ml`

`j2k-ml` is a Burn 0.21 integration maintained by the J2K project. It is not
an official Tracel or Burn crate and follows the workspace semver policy. The
GPU adapters use only released CubeCL, wgpu, and Burn APIs and are built as a
packaged crate outside the workspace before each release.

## What `j2k-ml` does and does not do

`j2k-ml` is a thin adapter, not a second codec. Parsing, JPEG 2000 and HTJ2K
decoding, preparation, grouping, memory reuse, and GPU execution are done by
`j2k`, `j2k-native`, `j2k-cuda`, and `j2k-metal`. `j2k-ml` turns CPU groups
into Burn tensors directly, or copies GPU output through host memory and
uploads it as an ordinary Burn tensor.

Dataset assembly, DICOM parsing, labels, sampling, resizing, padding,
prefetching, augmentation, float conversion, and normalization remain outside
the codec and this adapter. Applications cast and normalize the returned
integer tensors with ordinary Burn tensor operations.

Burn's `DataLoader` selects application items and calls the application's
`Batcher`. The codec then partitions those selected items into homogeneous
decode groups. `j2k-ml` does not choose samples or labels, and one codec group
is not necessarily a whole training batch.

## Runnable examples

The generic training example owns a persistent `CpuBurnDecoder` inside a
mutex-protected Burn `Batcher`, handles decode failures as batch results,
realigns labels through `source_indices`, and performs float normalization
after decode:

```bash
cargo run -p j2k-ml --example training_batcher --features cpu
```

The GPU examples decode on the GPU, copy the pixels to the host, and return
ordinary Burn integer tensors:

```bash
cargo run -p j2k-ml --example cuda_upload --features cuda
cargo run -p j2k-ml --example metal_upload --features metal
```

The CPU decoder works with any Burn backend through `TensorData`; on a GPU
backend this includes a host-to-device upload. With the `cuda` and `metal`
features, decoding runs on that GPU, and the decoded pixels are copied back
and uploaded the same way.

## Batch behavior

Inputs are owned `EncodedImage` values: an `Arc<[u8]>` containing JP2, JPH,
raw J2K, or raw HTJ2K bytes plus a full, ROI, reduced-resolution, or
ROI-and-reduced decode request. A persistent decoder can prepare and decode a
batch in one call, retain the codec's `PreparedBatch` and call
`decode_prepared` repeatedly, or regroup caller-supplied `PreparedImage`
values with `prepare_prepared_images`/`decode_prepared_images`. Regrouping does
not reparse or copy codestreams. Its source indices are positions in the new
submission; `PreparedImage::source_index` is still the original index.
Strict/lenient settings must match, while layout and worker settings may
change. Preparation keeps the original bytes and codec plans without copying
the codestream. Supported direct-plan inputs
report `PreparationDepth::Htj2kOffsetPlan` or
`PreparationDepth::ClassicOffsetPlan`; other valid inputs may remain
`MetadataOnly` for the general CPU path.

The codec groups representable images by compatible decoded dimensions,
channel count, exact native sample type, requested layout, transform, and
backend execution shape, and never pads different shapes to match. The adapter
returns one rank-4 Burn integer tensor per group in NCHW or NHWC layout:

- unsigned samples with precision at most 8 bits use `U8`;
- unsigned samples with precision from 9 through 16 bits use `U16`;
- signed samples with precision at most 16 bits use `I16`.

`BurnBatchGroup` preserves the codec metadata, original source indices in
tensor order, actual decoded rectangles, and warnings. `BurnBatchDecode`
returns the successful groups along with per-input preparation failures and a
`BurnBatchGroupError` for each group that failed during submission or
completion. One failed group does not affect the others, and all submitted
work finishes before the result is returned. Batches only support uniform
Gray, RGB, or RGBA output; other component layouts are available through the
codec's component-plane APIs.

JP2 channel definitions are part of the grouping key. In particular, straight
and premultiplied alpha are never combined into the same RGBA group, and the
alpha interpretation remains available in `BatchGroupInfo`.

Batch sessions decode strictly by default. Lenient decoding is opt-in and
reports a warning only when one of its JP2/JPH metadata recoveries was used.
An explicit CUDA or Metal route never falls back to CPU decoding, but its
decoded pixels are copied through host memory. Unsupported inputs and transfer
failures return errors.

## Persistent routes

`CpuBurnDecoder<B>` retains a `j2k::CpuBatchDecoder` and its codec workspaces.
The codec decodes directly into one contiguous native Rust allocation per
group. The adapter constructs one `TensorData` from that allocation, preserving
the exact `U8`, `U16`, or `I16` dtype. The retained worker workspaces reuse
component, Tier-1, and IDWT owners. Supported inputs retain either a per-tile
HTJ2K cleanup/refinement offset plan or a per-tile classic packet/code-block
plan, and CPU prepared decode consumes single- and multi-tile forms without
parsing again. Other inputs keep only metadata and use the general CPU
decoder.

`CudaUploadBurnDecoder` retains the CUDA codec session and a Burn CUDA device.
The codec decodes the batch into CUDA memory. When it completes and its status
checks pass, the adapter copies each group's pixels to the host in one copy and
creates the Burn tensor with `Tensor::from_data`, which uploads it to the
device.

`MetalUploadBurnDecoder::system_default` retains a Metal codec session and
uses Burn's default wgpu device. When a Metal group completes, the adapter
copies its pixels to the host and creates the Burn tensor the same way.

Both adapters keep the codec's submission guards, and sessions can be reused
safely after a drop. `submit` only overlaps the decoding; `wait` covers
completion, the copy to the host, and the Burn upload. Pixels are not handed to
Burn in GPU memory.

## GPU adapter support

The GPU adapters support less than the CPU decoder. In the table, `all
requests` means `Full`, `Region`, `Reduced`, and `RegionReduced`, and both NCHW
and NHWC are supported. Not every combination has its own hardware test with a
batch larger than one. Unsupported inputs return a group error; they never fall
back to the CPU.

| Output | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Gray `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases and exact multi-tile subset below |
| RGB `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases and exact multi-tile subset below |
| RGBA `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases |

The canonical `cargo xtask release-cuda` lane passes on the RTX 4070 runner.
Its codec targets validate classic and HT Gray/RGB/RGBA `U8`/`U16`/`I16`, all
four requests, both layouts, resident and external destinations, multi-tile
regressions, asynchronous drop and session reuse, and a 2,000-operation soak.
Its Burn targets test the same dtypes, requests, and layouts through staged
tensor uploads, plus prepared regrouping, group isolation, drop-safe reuse,
and a 1,000-batch soak. Reversible output is bit-exact; irreversible 9/7 output
is within one integer LSB of the CPU output. These tests check correctness,
not speed. An explicit CUDA request fails, rather than decoding on the CPU, if
the runtime or plan checks fail.

Focused Metal hardware cases collectively exercise classic and HT inputs,
`U8`/`U16`/`I16`, Gray/RGB/RGBA, all four requests, both layouts, resident
output, and external destinations, but not every dtype, request, and layout
combination with a batch larger than one. Independent multi-tile HT Gray12 and
RGB8 external output is
bit-exact across full, region, reduced, and region-plus-reduced requests;
generated multi-tile classic RGB8 is covered for full external output.

CUDA and Metal external destinations support both NCHW and NHWC. A resident
image-surface view is exposed only for NHWC because `Surface` denotes
interleaved pixels; CUDA also exposes an explicit dense resident owner for
NCHW, while Metal NCHW callers use the dense external-destination API. Both
prepared offset plans include every tile involved. Nonzero ROI maxshift (from
an RGN marker) is not supported on either GPU.

The GPU sessions keep successful groups when another group fails. Classic and
HT jobs keep track of their source image through batching, and HT jobs also
through pass bucketing and chunking, so a decode error on a job names its
source. A Metal command-buffer failure cannot be attributed to one image, so
the whole group is discarded and its error lists every source index in it.
Partially written tensors are never returned.

Aggregate HT descriptor and compressed arenas are split into bounded,
pass-homogeneous chunks without changing the output grouping. A single job
that exceeds the configured limits returns an error; it does not fall back to
the host.

Reversible 5/3 output must match the CPU output bit for bit. Irreversible 9/7
output may differ from it by at most one integer LSB.

## Validation and benchmarks

Portable tests run with `cargo test -p j2k-ml --features cpu`. Metal hardware
validation is part of `cargo xtask release-metal`; CUDA hardware validation is
part of `cargo xtask release-cuda`. Both make missing hardware or skipped GPU
tests a failure.

Linux AArch64 test and benchmark builds use Burn's NdArray backend. In this
repository's Burn 0.21 all-feature build, Flex selects `gemm-f16` 0.19 and its
debug AArch64 assembly is compiled without the FP16 target-feature gating that
the selected runner requires ([issue
#31](https://github.com/sarah-quinones/gemm/issues/31)). Upstream has a gating
fix under review ([pull request
#43](https://github.com/sarah-quinones/gemm/pull/43)). This only affects tests;
the `j2k-ml` library and its release dependencies do not pick a CPU backend. The CPU benchmark group IDs are
`j2k_owned_batch_codec_cpu/input_{distinct|repeated}` and
`j2k_owned_batch_burn_cpu/input_{distinct|repeated}`. Record which host backend
was used; results from different backends are not comparable.

Run `cargo bench -p j2k-ml --bench batch_decode --features cpu` for the
portable owned-batch benchmark. It reports codec-resident CPU output and
Burn-materialized output separately, includes one-shot and prepared reuse, and
covers HT-dominant unsigned and signed Gray12/Gray16 plus RGB8/RGB16 and
RGBA8/RGBA16 workloads at batch sizes 1/8/32/64 with full, ROI, reduced, and
ROI-and-reduced requests. The CUDA and Metal harnesses cover the same request
and output matrix. Their GPU rows measure GPU decoding, the copy back to the
host, and the Burn upload. The
`staged_cpu_upload_pixels` row measures CPU codec decode followed by Burn
upload on the same backend.
Run the hardware matrices with `cargo xtask j2k-ml-bench-cuda` and
`cargo xtask j2k-ml-bench-metal`. Decode rows report decoded spatial pixels per second;
divide by the decoded pixels per image for images per second or by 1,000,000
for megapixels per second. `prepare_images` rows report images per second. The
default `J2K_ML_BATCH_INPUT_MODE=distinct`
generates 64 deterministic, content-distinct codestreams per workload outside
the timed region and keeps only one workload in memory at a time.
`J2K_ML_BATCH_INPUT_MODE=repeated` reuses one `Arc` for every image, as a
diagnostic; one process never mixes the two modes. The input mode is part of
every Criterion group ID.

Criterion runs default to `J2K_ML_BATCH_PROCESS_MODE=criterion` and do not
collect the one-shot telemetry probes. Metal Criterion runs reject enabled
stage timing, signposts, split-command profiling, or Xcode capture. Run a
separate low-batch diagnostic process with
`J2K_ML_BATCH_PROCESS_MODE=profile`; it covers only batches 1 and 8 and emits
telemetry without running Criterion. Codec-resident iterations wait for codec
completion. CUDA and Metal staged-adapter measurements include decoded-pixel
readback, Burn upload, and the final consumer synchronization required by the
harness.

Results, machines, and dates are in
[`docs/benchmark-evidence.md`](benchmark-evidence.md); results on generated
fixtures do not replace a run on a pinned external corpus. The CUDA harness emits
codec-runtime H2D/D2H, kernel,
runtime-owned allocation, live/high-water memory, pool, event, and host-wait
counters in `cuda_telemetry_v2` rows for one completed probe per resident or
Burn-upload case. Probe throughput and counter deltas describe the same
completed decode and include the input mode. Throughput comes from the
separate Criterion samples. The counters do not include Burn/CubeCL allocations
or consumer kernels. The high-water values are session-cumulative rather than per-case peak
memory. `consumer_host_syncs` records the explicit Burn synchronization, not a
CUDA driver wait. The Metal harness emits `metal_telemetry_v2` codec submission
and retained-pool counters plus input mode. Its decoded-transfer,
final-destination, group-wait, and consumer-synchronization columns are
prefixed `asserted_`: they report behavior checked by tests, not measured
hardware counters. The codec submission delta is prefixed
`measured_` and comes from session diagnostics.

The pre-timing Metal telemetry snapshot performs runtime and pipeline
initialization before the timed interval. No decode is used as an unrecorded
warmup. The first one-shot decode remains cold for prepared-plan and
execution-arena caches, and the first prepared decode includes its
immutable-arena upload. Neither GPU adapter should be described as faster than
the CPU path until each backend has new results for the adapter recorded.
