# Burn batch decoding with `j2k-ml`

`j2k-ml` turns JPEG, JPEG 2000, and HTJ2K batches into Burn 0.21 integer
tensors. It is maintained by the J2K project, is not an official Tracel or
Burn crate, and follows the workspace semver policy. For PyTorch and NumPy,
see [`j2k-python`](../crates/j2k-python/README.md).

## Scope

`j2k-ml` is an adapter, not a second codec. `j2k`, `j2k-native`, `j2k-jpeg`,
`j2k-cuda`, and `j2k-metal` parse, prepare, group, and decode. `j2k-ml` puts
the result in a Burn tensor.

Dataset assembly, DICOM parsing, labels, sampling, resizing, padding,
augmentation, float conversion, and normalization stay in your application.
Cast and normalize the returned integer tensors with ordinary Burn tensor
operations.

A codec group is not a training batch. Burn's `DataLoader` picks items and
calls your `Batcher`; the codec then splits those items into groups of the
same shape and sample type, so one batch can come back as several tensors.

## Decoders

| Decoder | Feature | Codec runs on | How pixels reach Burn |
| --- | --- | --- | --- |
| `CpuBurnDecoder<B>` | `cpu` | CPU | One `TensorData` per group on any Burn backend. A GPU backend uploads it. |
| `CudaDirectBurnDecoder` | `cuda` | CUDA | Decoded into Burn-owned CUDA storage. No host copy. Synchronous. |
| `CudaUploadBurnDecoder` | `cuda` | CUDA | Copied to host memory, then uploaded with `Tensor::from_data`. |
| `MetalUploadBurnDecoder` | `metal` | Metal | Copied to host memory, then uploaded to Burn's default wgpu device. |

No feature is on by default.

```bash
cargo run -p j2k-ml --example training_batcher --features cpu
cargo run -p j2k-ml --example cuda_direct --features cuda
cargo run -p j2k-ml --example cuda_upload --features cuda
cargo run -p j2k-ml --example metal_upload --features metal
```

`training_batcher` keeps a `CpuBurnDecoder` behind a mutex inside a Burn
`Batcher`, treats decode failures as batch results, realigns labels through
`source_indices`, and normalizes after decode.

- **`CpuBurnDecoder`** keeps a `j2k::CpuBatchDecoder` and its workspaces. The
  codec decodes each group into one contiguous allocation, which becomes the
  tensor's `TensorData`.
- **`CudaDirectBurnDecoder`** allocates each destination through CubeCL,
  checks that the pointer belongs to the device's retained primary context,
  lets the codec write into it, and registers the same allocation as a fused
  `burn_cuda::Cuda` tensor once the codec's status checks pass. Burn owns the
  allocation throughout. `decode` and `decode_prepared` return only when the
  tensors are safe to use; there is no submitted handle, because Burn 0.21
  does not expose the CUDA stream needed to order asynchronous work safely.
- **The upload decoders** have `submit`, which overlaps only the decoding, and
  `wait`, which covers completion, the host copy, and the Burn upload. On
  CUDA, RGB and RGBA groups larger than 32 images decode in 32-image chunks,
  and `submit` waits for every chunk except the last. A session can be reused
  after a submitted batch is dropped.

## Inputs and preparation

Inputs are owned `EncodedImage` values: an `Arc<[u8]>` holding JP2, JPH, or a
raw codestream, plus a full, region, reduced, or region-and-reduced request.
A decoder can:

- prepare and decode in one call;
- keep the codec's `PreparedBatch` and call `decode_prepared` repeatedly;
- regroup `PreparedImage` values with `prepare_prepared_images` or
  `decode_prepared_images`, without parsing or copying the codestreams again.
  Source indices are then positions in the new submission;
  `PreparedImage::source_index` is still the original.

Strict or lenient settings must match across regrouped images; layout and
worker settings may change. Supported inputs report
`PreparationDepth::Htj2kOffsetPlan` or `ClassicOffsetPlan`. Other valid inputs
stay `MetadataOnly` and use the general CPU decoder.

## Mixed JPEG and JPEG 2000 input

`CpuBurnDecoder::decode_batch(&[ImageInput<'_>])` takes both codec families in
one call. `ImageInput::Jpeg` carries a `j2k_jpeg::JpegBatchInput`, and
`ImageInput::Jpeg2000` borrows an `EncodedImage`.

- JPEG decodes on the CPU, full image only. Its output is RGB `U8`, with
  grayscale expanded to three channels. Grayscale JPEG 2000 stays one channel.
- The session's strictness applies to both codecs, and JPEG color hints are
  kept.
- Errors and warnings are tagged with their codec (`ImageDecodeError`,
  `ImageDecodeWarning`). `j2k_info()` gives the JPEG 2000 metadata of a group.

## Output

Each group is one rank-4 Burn integer tensor in NCHW or NHWC:

| Source samples | Tensor type |
| --- | --- |
| Unsigned, up to 8 bits | `U8` |
| Unsigned, 9 to 16 bits | `U16` |
| Signed, up to 16 bits | `I16` |

- Images share a group only if they match in decoded size, channel count,
  sample type, layout, transform, backend execution shape, and alpha kind.
  Straight and premultiplied alpha never mix. Nothing is padded.
- Batches support Gray, RGB, and RGBA. Other component layouts need the
  codec's component-plane APIs.
- `BurnBatchGroup` has the codec metadata, the source indices in tensor order,
  the decoded rectangles, and warnings. Group order is unspecified; use
  `source_indices`.
- `BurnBatchDecode` has the groups, per-input preparation errors, and one
  `BurnBatchGroupError` per group that failed. A failed group does not affect
  the others, and all submitted work finishes before the result is returned.
  Read it with `groups()`, `errors()`, `group_errors()`, or consume it with
  `into_groups()` and `into_tensor()`.
- Decoding is strict by default. `recovered()` is set only for JPEG rows. A
  lenient JPEG 2000 metadata recovery is reported as a warning on the group;
  the lenient JPEG 2000 path does not salvage entropy data.
- An explicit CUDA or Metal decoder never falls back to the CPU. Unsupported
  inputs and transfer failures are errors.

## GPU adapter support

The GPU decoders support less than the CPU decoder. "All requests" means
full, region, reduced, and region-and-reduced, in NCHW and NHWC.

| Output | CPU | CUDA | Metal |
| --- | --- | --- | --- |
| Gray, RGB, RGBA as `U8`, `U16`, `I16` | Classic and HT, all requests | Classic and HT, all requests | Classic and HT, all requests, for single-tile images. Multi-tile: the cases below |

- **CUDA.** `cargo xtask release-cuda` on an RTX 4070 validates classic and HT
  Gray/RGB/RGBA `U8`/`U16`/`I16`, all four requests, both layouts, resident
  and external destinations, and multi-tile inputs. A direct-destination test
  confirms that the only device-to-host traffic is status validation.
- **Metal.** Hardware tests cover the same sample types, colors, requests, and
  layouts, but not every combination at batch sizes above one. Multi-tile
  coverage is HT Gray12 and RGB8 external output for all four requests, and
  classic RGB8 for full external output.
- **Bit-exact.** Reversible 5/3 and irreversible 9/7 output must match the CPU
  bit for bit. For 9/7, the GPU rounds each centered sample ties-to-even
  before the level shift, as the CPU does.
- **Not on a GPU.** Components wider than 16 bits.
- **ROI.** Both backends decode ROI maxshift (RGN markers) when the coded
  bitplane count including the shift is at most 31. Classic blocks must fit in
  64x64; HT geometry and pass limits depend on the backend. Tested: CUDA
  reversible classic Gray8 and RGB8 with shift 7 (both layouts, full and
  region-and-reduced); Metal classic signed Gray4 with shift 7 and HT RGB8
  with component shifts 3/0/0.
- **Destinations.** External destinations support NCHW and NHWC. Each group
  also has a dense codec-owned resident buffer in its own layout
  (`MetalBatchGroup::resident_batch`, `CudaResidentBatchBuffer`). The
  `Surface` views exist only for interleaved (NHWC or grayscale) groups.
- **Errors.** A successful group is kept when another fails. A decode error on
  one job names its source image. A Metal command-buffer failure cannot be
  attributed, so the whole group fails and its error lists every source index.
  A single job larger than the configured HT chunk limits is an error. Partly
  written tensors are never returned.

## Tests and benchmarks

```bash
cargo test -p j2k-ml --features cpu          # portable tests
cargo bench -p j2k-ml --bench batch_decode --features cpu
cargo xtask j2k-ml-bench-cuda                # hardware matrices
cargo xtask j2k-ml-bench-metal
```

Hardware validation is part of `cargo xtask release-cuda` and
`cargo xtask release-metal`; both fail on missing hardware or a skipped GPU
test.

The benchmark covers unsigned and signed Gray12/Gray16, RGB8/RGB16, and
RGBA8/RGBA16 at batch sizes 1, 8, 32, and 64, for all four requests. It
reports codec output and Burn tensors separately, one-shot and prepared.

- Decode rows report decoded pixels per second; `prepare_images` rows report
  images per second.
- Staged GPU rows include the GPU decode, the copy to the host, and the Burn
  upload. `staged_cpu_upload_pixels` is CPU decode plus Burn upload on the
  same backend.
- CUDA `one_shot_normalized_pixels` rows (groups
  `j2k_owned_batch_burn_staged_cuda` and `j2k_owned_batch_burn_direct_cuda`)
  also include preparation, integer-to-float normalization, and the final
  Burn synchronization.
- `J2K_ML_BATCH_INPUT_MODE` selects 64 distinct generated codestreams per
  workload (`distinct`, the default) or one repeated input (`repeated`). The
  mode is part of every Criterion group ID.
- `J2K_ML_BATCH_PROCESS_MODE=profile` runs a separate process that emits
  telemetry for batches 1 and 8 instead of Criterion samples. Metal Criterion
  runs reject stage timing, signposts, split-command profiling, and Xcode
  capture.
- Telemetry rows are `cuda_telemetry_v2` and `metal_telemetry_v2`. CUDA
  counters cover the codec runtime only, not Burn or CubeCL, and its
  high-water values are cumulative for the session. Metal columns prefixed
  `asserted_` state behavior that tests check; `measured_` columns come from
  session diagnostics.
- The first one-shot decode is cold for the prepared-plan and execution-arena
  caches, and the first prepared decode includes its arena upload.

Linux AArch64 test and benchmark builds use Burn's NdArray backend, because
Flex pulls in `gemm-f16` 0.19, whose debug AArch64 assembly needs FP16 target
features the runner lacks ([gemm issue
#31](https://github.com/sarah-quinones/gemm/issues/31)). This affects tests
only. Results from different host backends are not comparable.

Results, machines, and dates are in
[`docs/benchmark-evidence.md`](benchmark-evidence.md). Do not describe a GPU
adapter as faster than the CPU path without results recorded there for that
backend.
