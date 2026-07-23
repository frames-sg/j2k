# Burn batch decoding with `j2k-ml`

`j2k-ml` is an independently maintained integration for Burn 0.21. It is not
an official Tracel or Burn crate. It follows the workspace's reviewed semver
policy. The staged accelerator adapters use only released CubeCL, wgpu, and
Burn APIs and remain subject to clean-consumer validation before publication.

## Ownership boundary

`j2k-ml` is a thin framework adapter, not a second codec. The `j2k`,
`j2k-native`, `j2k-cuda`, and `j2k-metal` crates own parsing, JPEG 2000 and
HTJ2K decoding, preparation, homogeneous grouping, memory reuse, and device
execution. `j2k-ml` materializes CPU groups directly or stages completed
accelerator output through host memory before an ordinary Burn tensor upload.

Dataset assembly, DICOM parsing, labels, sampling, resizing, padding,
prefetching, augmentation, float conversion, and normalization remain outside
the codec and this adapter. Applications cast and normalize the returned
integer tensors with ordinary Burn tensor operations.

Burn's `DataLoader` selects application items and calls the application's
`Batcher`. The codec then partitions those selected items into homogeneous
decode groups. These are distinct batching responsibilities: `j2k-ml` does not
choose samples or labels, and a codec group is not automatically a complete
training batch.

## Runnable examples

The generic training example owns a persistent `CpuBurnDecoder` inside a
mutex-protected Burn `Batcher`, handles decode failures as batch results,
realigns labels through `source_indices`, and performs float normalization
after decode:

```bash
cargo run -p j2k-ml --example training_batcher --features cpu
```

The accelerator examples return ordinary Burn integer tensors after an
explicit decoded-pixel readback and upload:

```bash
cargo run -p j2k-ml --example cuda_upload --features cuda
cargo run -p j2k-ml --example metal_upload --features metal
```

The CPU codec can target any compatible Burn backend through `TensorData`; a
GPU target on that route includes a host-to-device upload. With the explicit
`cuda` and `metal` features, codec execution occurs on that accelerator before
the decoded pixels are read back and uploaded through the same ordinary Burn
boundary.

## Batch contract

Inputs are owned `EncodedImage` values: an `Arc<[u8]>` containing JP2, JPH,
raw J2K, or raw HTJ2K bytes plus a full, ROI, reduced-resolution, or
ROI-and-reduced decode request. A persistent decoder can prepare and decode a
batch in one call, retain the codec's `PreparedBatch` and call
`decode_prepared` repeatedly, or regroup caller-supplied `PreparedImage`
values with `prepare_prepared_images`/`decode_prepared_images`. Regrouping does
not reparse or copy codestreams. Its source indices are positions in the new
submission; `PreparedImage::source_index` remains the original preparation
index. Strict/lenient settings must match, while layout and worker policy may
change. Preparation retains the original byte owner and codec plans; it does
not duplicate the codestream. Supported direct-plan inputs
report `PreparationDepth::Htj2kOffsetPlan` or
`PreparationDepth::ClassicOffsetPlan`; other valid inputs may remain
`MetadataOnly` for the general CPU path.

The codec groups representable images by compatible decoded dimensions,
channel count, exact native sample type, requested layout, transform, and
backend execution shape. It never pads unlike shapes implicitly. The adapter
returns one ordinary rank-4 Burn integer tensor per homogeneous group in NCHW
or NHWC layout:

- unsigned samples with precision at most 8 bits use `U8`;
- unsigned samples with precision from 9 through 16 bits use `U16`;
- signed samples with precision at most 16 bits use `I16`.

`BurnBatchGroup` preserves the codec metadata, original source indices in
tensor order, actual decoded rectangles, and warnings. `BurnBatchDecode`
returns successful groups alongside indexed preparation failures and
`BurnBatchGroupError` values for homogeneous groups discarded during
submission or completion. A nonfatal group failure does not suppress other
groups; every submitted pending owner is retired before the result is exposed.
The fast batch representation is intentionally limited to uniform Gray, RGB,
or RGBA output; broader component layouts remain available through the codec's
component-plane APIs.

JP2 channel definitions are part of the grouping key. In particular, straight
and premultiplied alpha are never combined into the same RGBA group, and the
alpha interpretation remains available in `BatchGroupInfo`.

New batch sessions use strict decoding by default. Lenient decoding is
opt-in and reports warnings. An explicit CUDA or Metal route never substitutes
CPU codec decoding, but its completed decoded pixels are deliberately staged
through host memory. Unsupported inputs and transfer failures are structured
errors.

## Persistent routes

`CpuBurnDecoder<B>` retains a `j2k::CpuBatchDecoder` and its codec workspaces.
The codec decodes directly into one contiguous native Rust allocation per
group. The adapter constructs one `TensorData` from that allocation, preserving
the exact `U8`, `U16`, or `I16` dtype. The retained worker workspaces reuse
component, Tier-1, and IDWT owners. Supported inputs retain either a per-tile
HTJ2K cleanup/refinement offset plan or a per-tile classic packet/code-block
plan, and CPU prepared decode consumes single- and multi-tile forms without
reparsing. Inputs outside those retained-plan boundaries can remain
metadata-only and continue through the broader CPU decoder.

`CudaUploadBurnDecoder` retains the CUDA codec session and a Burn CUDA device.
The codec first produces its ordinary codec-owned CUDA resident batch. After
completion and status validation, the adapter performs one dense
device-to-host pixel copy per homogeneous group and constructs the Burn tensor
with `Tensor::from_data`, which performs the normal host-to-device upload.

`MetalUploadBurnDecoder::system_default` retains a Metal codec session and
targets Burn's default wgpu device. After the codec's resident Metal group
completes, the adapter copies its validated dense byte range into host staging
and constructs the Burn tensor through the same ordinary public API.

Both adapters preserve codec submission guards and drop-safe session reuse.
The `submit` methods overlap only codec work; `wait` includes completion,
decoded-pixel readback, and Burn upload. These APIs make no direct-destination,
zero-copy, or asynchronous cross-runtime handoff claim.

## Current exact accelerator boundary

The accelerator adapters are deliberately narrower than the portable codec
contract. The table describes the supported adapter contract: `all requests`
means `Full`, `Region`, `Reduced`, and `RegionReduced`, and both NCHW and NHWC
Burn tensors are supported. It is not a claim that every Cartesian cell has a
separate batch-greater-than-one hardware test. Unsupported inputs return a
structured group error; they never fall back through CPU pixels.

| Output | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Gray `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases and exact multi-tile subset below |
| RGB `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases and exact multi-tile subset below |
| RGBA `U8`/`U16`/`I16` | Classic and HT, all requests | Release-validated classic/HT, all requests | Supported single-tile classic/HT, all requests; focused hardware cases |

The canonical `cargo xtask release-cuda` lane passes on the RTX 4070 runner.
Its codec targets validate classic and HT Gray/RGB/RGBA `U8`/`U16`/`I16`, all
four requests, both layouts, resident and external destinations, multi-tile
regressions, asynchronous drop and session reuse, and a 2,000-operation soak.
Its Burn targets validate the same dtype/request/layout boundary through staged
tensor uploads, plus prepared regrouping, group isolation, drop-safe reuse,
and a 1,000-batch soak. Reversible output is bit-exact; irreversible 9/7 output
agrees with the CPU oracle within one integer LSB. This establishes the support
boundary, not a throughput advantage. An explicit CUDA request still fails
rather than substituting CPU codec execution if runtime or retained-plan
validation does not succeed.

Focused Metal hardware cases collectively exercise classic and HT inputs,
`U8`/`U16`/`I16`, Gray/RGB/RGBA, all four requests, both layouts, resident
output, and external destinations. This is not a Cartesian
batch-greater-than-one validation of every dtype, request, and layout
combination. Independent multi-tile HT Gray12 and RGB8 external output is
bit-exact across full, region, reduced, and region-plus-reduced requests;
generated multi-tile classic RGB8 is covered for full external output.

CUDA and Metal external destinations support both NCHW and NHWC. A resident
image-surface view is exposed only for NHWC because `Surface` denotes
interleaved pixels; CUDA also exposes an explicit dense resident owner for
NCHW, while Metal NCHW callers use the dense external-destination API. Both
prepared offset-plan forms retain every participating tile. Nonzero ROI
maxshift declared by a codestream RGN marker remains unsupported on both GPU
prepared routes.

The GPU sessions retain successful homogeneous groups when a different group
fails. Device classic and HT codec jobs retain their original source identity
through batch flattening; HT jobs additionally retain it through pass bucketing
and chunk splitting. A codec-status record that identifies a failed job can
name that original source. A Metal command-buffer completion failure cannot be
safely assigned to one image, so it remains a group-level failure: the dense
group is discarded and its group error preserves every affected source index.
Partially written tensors are never returned.

Aggregate HT descriptor and compressed arenas are split into bounded,
pass-homogeneous chunks without changing the public output grouping. A single
job that cannot fit the configured hard limits returns a structured error; it
is not permission to stage decoded pixels through the host.

Reversible 5/3 output is required to match the CPU integer oracle bit for bit.
Irreversible 9/7 reconstruction may differ from that oracle by at most one
integer LSB.

## Validation and benchmarks

Portable tests run with `cargo test -p j2k-ml --features cpu`. Metal hardware
validation is part of `cargo xtask release-metal`; CUDA hardware validation is
part of `cargo xtask release-cuda`. Both release lanes set the repository's
fail-closed runtime gates, so missing hardware or skipped accelerator work is a
failure.

Linux AArch64 test and benchmark builds use Burn's NdArray backend. In this
repository's Burn 0.21 all-feature build, Flex selects `gemm-f16` 0.19 and its
debug AArch64 assembly is compiled without the FP16 target-feature gating that
the selected runner requires ([issue
#31](https://github.com/sarah-quinones/gemm/issues/31)). Upstream has a gating
fix under review ([pull request
#43](https://github.com/sarah-quinones/gemm/pull/43)). This substitution is
test-only; the generic `j2k-ml` library API and its release dependencies do not
select either CPU backend. The CPU benchmark group IDs are
`j2k_owned_batch_codec_cpu/input_{distinct|repeated}` and
`j2k_owned_batch_burn_cpu/input_{distinct|repeated}`. Record the selected host
backend separately in benchmark provenance; results are not cross-backend
comparisons.

Run `cargo bench -p j2k-ml --bench batch_decode --features cpu` for the
portable owned-batch benchmark. It reports codec-resident CPU output and
Burn-materialized output separately, includes one-shot and prepared reuse, and
covers HT-dominant unsigned and signed Gray12/Gray16 plus RGB8/RGB16 and
RGBA8/RGBA16 workloads at batch sizes 1/8/32/64 with full, ROI, reduced, and
ROI-and-reduced requests. The CUDA and Metal harnesses cover the same request
and native-output matrix. Their accelerator rows measure accelerator codec
decode followed by explicit decoded-pixel readback and Burn upload. The
`staged_cpu_upload_pixels` row measures CPU codec decode followed by Burn
upload on the same backend.
Run the hardware matrices with `cargo xtask j2k-ml-bench-cuda` and
`cargo xtask j2k-ml-bench-metal`. Decode rows report decoded spatial pixels per second;
divide by the decoded pixels per image for images per second or by 1,000,000
for megapixels per second. `prepare_images` rows report images per second. The
default `J2K_ML_BATCH_INPUT_MODE=distinct`
generates 64 deterministic, content-distinct codestreams per workload outside
the measured region and retains only one materialized workload at a time.
`J2K_ML_BATCH_INPUT_MODE=repeated` is an explicitly labeled broadcast/reuse
diagnostic that clones one `Arc`; a process and its retained sessions never mix
the two modes. The input mode is part of every Criterion group ID.

Criterion runs default to `J2K_ML_BATCH_PROCESS_MODE=criterion` and do not
collect the one-shot telemetry probes. Metal Criterion runs reject enabled
stage timing, signposts, split-command profiling, or Xcode capture. Run a
separate low-batch diagnostic process with
`J2K_ML_BATCH_PROCESS_MODE=profile`; it covers only batches 1 and 8 and emits
telemetry without running Criterion. Codec-resident iterations wait for codec
completion. CUDA and Metal staged-adapter measurements include decoded-pixel
readback, Burn upload, and the final consumer synchronization required by the
harness.

Publication status, dated machines, and local results are maintained in
[`docs/benchmark-evidence.md`](benchmark-evidence.md); generated local fixtures
do not replace a pinned external adoption run. The CUDA harness emits
codec-runtime H2D/D2H, kernel,
runtime-owned allocation, live/high-water memory, pool, event, and host-wait
counters in `cuda_telemetry_v2` rows for one completed probe per resident or
Burn-upload case. Probe throughput and counter deltas describe the same
completed decode and include the input mode. Criterion remains the separately
sampled throughput result. Those counters explicitly exclude Burn/CubeCL allocation and consumer
kernels. The high-water values are session-cumulative rather than per-case peak
memory. `consumer_host_syncs` records the explicit Burn synchronization, not a
CUDA driver wait. The Metal harness emits `metal_telemetry_v2` codec submission
and retained-pool counters plus input mode. Its decoded-transfer,
final-destination, group-wait, and consumer-synchronization columns are
prefixed `asserted_`: they disclose route contracts checked by focused tests,
not sampled hardware counters. The codec submission delta is prefixed
`measured_` and comes from session diagnostics.

The pre-timing Metal telemetry snapshot performs runtime and pipeline
initialization before the timed interval. No decode is used as an unrecorded
warmup. The first one-shot decode remains cold for prepared-plan and
execution-arena caches, and the first prepared decode includes its
immutable-arena upload. Both backends must record new device-specific
staged-adapter results before either accelerator adapter is described as
faster than the portable path.
