# Burn batch decoding with `j2k-ml`

`j2k-ml` is an experimental, independently maintained integration for Burn
0.21. It is not an official Tracel or Burn crate and remains `publish = false`
through the 0.7 release cycle.

## Ownership boundary

`j2k-ml` is a thin framework adapter, not a second codec. The `j2k`,
`j2k-native`, `j2k-cuda`, and `j2k-metal` crates own parsing, JPEG 2000 and
HTJ2K decoding, preparation, homogeneous grouping, memory reuse, and device
execution. `j2k-ml` only materializes a CPU codec group once or lends unique
Burn-owned CUDA/Metal storage to the codec's external-destination API.

Dataset assembly, DICOM parsing, labels, sampling, resizing, padding,
prefetching, augmentation, float conversion, and normalization remain outside
the codec and this adapter. Applications cast and normalize the returned
integer tensors with ordinary Burn tensor operations.

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
opt-in and reports warnings. An explicit CUDA or Metal route never stages
through CPU memory or silently falls back; unsupported inputs and interop
failures are structured errors.

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

`CudaBurnDecoder` retains the codec CUDA session for one Burn CUDA device. It
allocates the final Burn tensor first, validates the allocation's device,
bounds, alignment, and unique mutable ownership, and submits the codec's final
store directly into that allocation. CUDA events order CubeCL's stream and the
codec stream without a normal-path CPU wait. The decoder retains in-flight
codec resources until completion. Decoded pixels have no device-to-host
transfer, intermediate dense decoded-pixel allocation, or final
device-to-device copy.

`MetalBurnDecoder::system_default` constructs the codec session and paired Burn
`WgpuDevice` from the same underlying Metal device. The adapter retains the
Burn-owned `MTLBuffer`, validates its suballocation layout and device identity,
and passes it to `MetalImageDestination`. The codec writes the final native
integer layout directly into that buffer before the tensor is registered with
Burn. Decoded pixels are neither read back nor uploaded again.

Both GPU adapters retain scratch owners and destination access until one
group-level completion boundary. Dropping pending work safely retires those
resources and leaves the persistent session reusable; the normal path does not
introduce a per-image or per-kernel device synchronization.

The direct-write guarantee is specifically zero decoded-pixel host round trips,
not a claim that compressed input or internal coefficient/scratch storage needs
no transfer or allocation. Those internal codec resources remain allowed.

## Current exact accelerator boundary

The accelerator adapters are deliberately narrower than the portable codec
contract. For every supported row below, `all requests` means `Full`, `Region`,
`Reduced`, and `RegionReduced`, and both NCHW and NHWC Burn tensors are
supported. Unsupported inputs return a structured group error; they never fall
back through CPU pixels.

| Output | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Gray `U8`/`U16`/`I16` | Classic and HT, all requests | Hardware-validated classic/HT, all requests | Hardware-validated single-tile classic/HT, all requests; exact multi-tile subset below |
| RGB `U8`/`U16`/`I16` | Classic and HT, all requests | Hardware-validated classic/HT, all requests | Hardware-validated single-tile classic/HT, all requests; exact multi-tile subset below |
| RGBA `U8`/`U16`/`I16` | Classic and HT, all requests | Hardware-validated classic/HT, all requests | Hardware-validated single-tile classic/HT, all requests |

The canonical `cargo xtask release-cuda` lane passes on the RTX 4070 runner.
Its codec targets validate classic and HT Gray/RGB/RGBA `U8`/`U16`/`I16`, all
four requests, both layouts, resident and external destinations, multi-tile
regressions, asynchronous drop and session reuse, and a 2,000-operation soak.
Its Burn targets validate the same dtype/request/layout boundary through direct
tensor destinations, plus prepared regrouping, group isolation, drop-safe
reuse, and a 1,000-batch soak. Reversible output is bit-exact; irreversible 9/7
output agrees with the CPU oracle within one integer LSB. This establishes the
support boundary, not a throughput advantage. An explicit CUDA request still
fails rather than staging through CPU if runtime validation, device identity,
or retained-plan validation does not succeed.

Metal hardware tests are bit-exact for independent multi-tile HT Gray12 and
RGB8 external output across full, region, reduced, and region-plus-reduced
requests, and for generated multi-tile classic RGB8 full external output. This
supplements the hardware-validated single-tile matrix; it does not imply that
every multi-tile dtype/request/layout combination has been run.

CUDA and Metal external destinations support both NCHW and NHWC. A resident
image-surface view is exposed only for NHWC because `Surface` denotes
interleaved pixels; CUDA also exposes an explicit dense resident owner for
NCHW, while Metal NCHW callers use the dense external-destination API. Both
prepared offset-plan forms retain every participating tile. Nonzero ROI
maxshift declared by a codestream RGN marker remains unsupported on both GPU
prepared routes.

The GPU sessions retain successful homogeneous groups when a different group
fails. Device HT jobs keep their original source identity through pass
bucketing and chunk splitting, so a status record that identifies a failing job
can name its source. When a lower-level failure cannot identify one image, the
entire dense group is discarded and its group error preserves every affected
source index. Partially written tensors are never returned.

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
select either CPU backend. Criterion group names include `flex` or
`ndarray_arm_linux`; results from those groups are not cross-backend
comparisons.

Run `cargo bench -p j2k-ml --bench batch_decode --features cpu` for the
portable owned-batch benchmark. It reports codec-resident CPU output and
Burn-materialized output separately, includes one-shot and prepared reuse, and
covers HT-dominant unsigned and signed Gray12/Gray16 plus RGB8/RGB16 and
RGBA8/RGBA16 workloads at batch sizes 1/8/32/64 with full, ROI, reduced, and
ROI-and-reduced requests. The CUDA and Metal harnesses cover the same request
and native-output matrix. Each accelerator harness also retains a
`staged_cpu_upload_pixels` row so direct device decode can be compared with the
supported persistent CPU-decode-and-upload path on the same Burn backend.
Run the hardware matrices with `cargo xtask j2k-ml-bench-cuda` and
`cargo xtask j2k-ml-bench-metal`. Criterion reports decoded pixels per second;
divide by the decoded pixels per image for images per second or by 1,000,000
for megapixels per second. The default `J2K_ML_BATCH_INPUT_MODE=distinct`
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
completion. CUDA Burn-direct measurements retain their final consumer sync.
Synchronous Metal Burn-direct decode has already completed and validated the
codec work before returning, so it does not add a redundant `Wgpu::sync()`;
the staged CPU-upload row retains that synchronization.

No publication throughput run has yet been recorded for the new batch
architecture. A July 19, 2026 local M4 Pro diagnostic run exercised the full
Metal matrix and is recorded in `docs/benchmark-evidence.md`; it does not
replace a pinned external adoption run. The CUDA harness emits codec-runtime H2D/D2H, kernel,
runtime-owned allocation, live/high-water memory, pool, event, and host-wait
counters in `cuda_telemetry_v2` rows for one completed probe per resident or
Burn-direct case. Probe throughput and counter deltas describe the same
completed decode and include the input mode. Criterion remains the separately
sampled throughput result. Those counters explicitly exclude Burn/CubeCL allocation and consumer
kernels. The high-water values are session-cumulative rather than per-case peak
memory. `consumer_host_syncs` records the explicit Burn synchronization, not a
CUDA driver wait. The Metal harness emits `metal_telemetry_v2` codec submission
and retained-pool counters plus input mode. Its decoded-transfer,
final-destination, group-wait, and consumer-synchronization columns are
prefixed `asserted_`: they disclose route contracts checked by focused tests,
not sampled hardware counters. The codec submission delta is prefixed
`measured_` and comes from session diagnostics. Neither profile
harness performs an unrecorded warmup: the first one-shot decode is cold, and
the first prepared decode includes its immutable-arena upload. Both backends must record
device-specific direct-versus-staged results before either route is described
as faster than the portable or staged path. Metal remains explicit pending a
corrected content-distinct run; the historical July 19 batch-greater-than-one
rows are qualified in `docs/benchmark-evidence.md`.
