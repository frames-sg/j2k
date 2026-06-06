# Signinum 1.0 Stable API Inventory

This document is the 1.0 boundary for the stable Signinum crates. It is written
to prevent accidental stabilization: if a public item is not listed here, either
make it private, move it to a support crate, or add it here with tests and docs
before the 1.0 release.

The inventory covers `signinum`, `signinum-core`, `signinum-jpeg`,
`signinum-j2k`, `signinum-tilecodec`, `signinum-cli`, `signinum-jpeg-metal`,
`signinum-j2k-metal`, `signinum-jpeg-cuda`, `signinum-j2k-cuda`,
`signinum-transcode`, `signinum-transcode-metal`, `signinum-j2k-native`,
`signinum-cuda-runtime`, and `signinum-profile`.

The item-level companion inventory is
`docs/stable-api-1.0.public-api.txt`. It is generated from `cargo public-api`
and records every stable public module, type, trait, function, method,
constant, variant, and field by crate. Review that snapshot together with this
human tier inventory before 1.0.

Generated public API output is still the mechanical source for semver checks and
inventory drift:

```sh
cargo install cargo-public-api --version 0.52.0 --locked
cargo xtask stable-api
cargo xtask semver
```

The release gate for this document is:

- `missing_docs` for every 1.0-stable library crate.
- `cargo xtask stable-api` to prove the checked-in item-level public API
  snapshot matches the current crates.
- `cargo xtask semver` for every stable library crate in the CI semver matrix.
- Behavior-focused CPU/GPU parity and fallback tests for the contracts below.
- No new `#[doc(hidden)] pub` item unless this document explicitly lists it as
  Adapter Integration API and explains why it is externally needed.

## Stability Tiers

Application API is the default user surface. It includes the facade crate,
prelude-style imports, high-level encode/decode entry points, the CLI command
contract, caller-owned output buffers, structured errors, ROI/scaled decode,
tile batching, row streaming, passthrough decisions, and lossless J2K/HTJ2K
encode behavior.

Direct Codec API is the crate-specific CPU-first surface in `signinum-core`,
`signinum-jpeg`, `signinum-j2k`, and `signinum-tilecodec`. It is stable for
portable CPU behavior and must not require Metal or CUDA.

Adapter API is the user-facing GPU surface: session types, device-backed surface
types, typed errors, encode/decode entry points, and observable dispatch reports.
It is intentionally narrow.

Adapter Integration API is public only because companion Signinum adapter crates
need shared planning data. This tier includes current kernel packet builders,
checkpoint structs, raw device plans, helper table builders, encode-stage jobs,
and benchmark/DICOM extraction hooks that are already public. These items are
semver-visible while public, but they are not the preferred end-user API.

## Trait Implementation Policy

The stable public traits in this inventory are intentional extension points
unless a future section explicitly marks them sealed. `ImageCodec`,
`ImageDecode`, `ImageDecodeRows`, `ImageDecodeDevice`, `ImageDecodeSubmit`,
`TileBatchDecode`, `TileBatchDecodeDevice`, `TileBatchDecodeManyDevice`,
`TileBatchDecodeSubmit`, `TileDecompress`, `DeviceSurface`,
`DeviceSubmission`, `RowSink`, `ScratchPool`, `Sample`, `CodecContext`,
`J2kEncodeStageAccelerator`, and `DctToWaveletStageAccelerator` are
implementable because Signinum's direct codec and adapter crates use the same
contracts downstream users see.

New traits that callers should only invoke, not implement, must be sealed before
they become stable. Existing traits must not gain required methods after 1.0
unless the change is semver-compatible for third-party implementations.

## GPU Contract

`BackendRequest::Cpu` / `BackendRequest::CPU_ONLY` never uses GPU.

`BackendRequest::Auto` / `BackendRequest::ACCELERATED` is adaptive
accelerated routing. GPU-shaped stages may use Metal/CUDA only when
workload-specific stage and end-to-end gates prove the split is faster than
optimized CPU. CPU fallback is allowed and observable through surface
residency, backend kind, dispatch reports, route reports, or typed outcome
metadata.

`BackendRequest::Metal` / `BackendRequest::STRICT_METAL` and
`BackendRequest::Cuda` / `BackendRequest::STRICT_CUDA` are strict requests.
They must return device-backed output or a typed unsupported/unavailable error
with no silent CPU fallback.

Hybrid encode/decode paths must report which backend and stages ran when that
affects user expectations. Dispatch-sensitive APIs expose dispatch reports.

GPU acceleration must preserve CPU/GPU parity for decoded pixels and public
output semantics. Explicitly documented lossy/hybrid tolerances are allowed only
for coefficient-domain transcode paths and must be tested against CPU metrics.

## Crate Inventory

### `signinum`

Tier: Application API.

Stable modules: `core`, `jpeg`, `jpeg::cuda`, `jpeg::metal`, `j2k`,
`j2k::cuda`, `j2k::metal`, `tilecodec`, and `prelude`.

Stable re-exported types: backend selection (`BackendRequest`, `BackendKind`,
`BackendCapabilities`), buffer/errors (`BufferError`, `CodecError`,
`JpegError`, `J2kError`, `TileCodecError`), core image contracts
(`PixelFormat`, `ColorSpace`, `Rect`, `Downscale`, `DecodeOutcome`,
`ImageCodec`, `ImageDecode`, `ImageDecodeRows`, `ImageDecodeDevice`,
`TileBatchDecode`, `TileBatchDecodeManyDevice`, `TileDecompress`,
`DeviceSurface`, `RowSink`), passthrough types (`CompressedTransferSyntax`,
`CompressedPayloadKind`, `PassthroughCandidate`, `PassthroughRequirements`,
`PassthroughDecision`, `PassthroughRejectReason`), JPEG types (`JpegCodec`,
`JpegDecoder`, `JpegView`, `DecodeOptions`, `ColorTransform`), J2K types
(`J2kCodec`, `J2kContext`, `J2kDecoder`, `J2kView`, `J2kScratchPool`,
`J2kLosslessSamples`, `J2kLosslessEncodeOptions`, `EncodedJ2k`,
`EncodeBackendPreference`, `J2kEncodeValidation`, `J2kEncodeDispatchReport`,
`J2kEncodeStageAccelerator`, `J2kProgressionOrder`, `J2kBlockCodingMode`,
`ReversibleTransform`, `J2kToHtj2kOptions`, `J2kToHtj2kMode`,
`J2kToHtj2kReport`, `ReencodedHtj2k`), and tile codecs (`UncompressedCodec`,
`DeflateCodec`, `LzwCodec`, `ZstdCodec`).

Stable functions: `encode_j2k_lossless`,
`j2k::encode_j2k_lossless`, `j2k::encode_j2k_lossless_cpu`,
`j2k::encode_j2k_lossless_with_accelerator`,
`j2k::j2k_lossless_decomposition_levels`, and
`j2k::recode_j2k_to_htj2k_lossless`.

### `signinum-core`

Tier: Direct Codec API.

Stable modules: `backend`, `batch`, `context`, `error`, `passthrough`,
`pixel`, `row_sink`, `sample`, `scale`, `scratch`, `traits`, and `types`.

Stable types and traits: `BackendRequest`, `BackendKind`,
`BackendCapabilities`, `CpuFeatures`, `TileBatchOptions`,
`IndexedBatchResult`, `CacheStats`, `CodecContext`, `DecoderContext`,
`BufferError`, `CodecError`, `InputError`, `NotImplemented`, `Unsupported`,
`CompressedTransferSyntax`, `CompressedPayloadKind`, `PassthroughCandidate`,
`PassthroughRequirements`, `PassthroughDecision`,
`PassthroughRejectReason`, `PixelFormat`, `PixelLayout`, `Sample`,
`SampleType`, `Downscale`, `ScratchPool`, `DecodeRowsError`,
`DeviceSubmission`, `DeviceSurface`, `ImageCodec`, `ImageDecode`,
`ImageDecodeSubmit`, `ImageDecodeDevice`, `ImageDecodeRows`,
`ReadySubmission`, `TileBatchDecode`, `TileBatchDecodeDevice`,
`TileBatchDecodeManyDevice`, `TileBatchDecodeSubmit`, `TileDecompress`,
`CodedUnitLayout`, `Colorspace`, `DecodeOutcome`, `Info`, `Rect`,
`TileLayout`, and `WarningKind`.

Stable functions: `strided_output_len`, `validate_strided_output_buffer`,
`copy_tight_pixels_to_strided_output`, `collect_indexed_batch_results`, and
`tile_batch_worker_count`.

### `signinum-jpeg`

Tier: Direct Codec API plus Adapter Integration API for `adapter`.

Stable modules: `info`, `context`, `adapter`, `error`, `encoder`,
`transcode`, `decoder`, and `bench_support`.

Stable direct-codec types: `SofKind`, `ColorSpace`, `SamplingFactors`,
`McuGeometry`, `RestartIndex`, `RestartSegment`, `Rect`, `ColorTransform`,
`DecodeOptions`, `Info`, `DecoderContext`, `BuilderConflictReason`,
`HuffmanFailure`, `JpegError`, `MarkerKind`, `TableKind`,
`UnsupportedReason`, `Warning`, `EncodedJpeg`, `JpegBackend`,
`JpegEncodeError`, `JpegEncodeOptions`, `JpegSamples`, `JpegSubsampling`,
`DctExtractOptions`, `JpegDctImage`, `JpegDctCodingMode`,
`JpegDctComponent`, `ComponentRowWriter`, `DecodeOutcome`, `Decoder`,
`JpegView`, `TileBatchError`, `TileBatchOptions`, `TileDecodeJob`,
`TileScaledDecodeJob`, `TileRegionScaledDecodeJob`, `ScratchPool`,
`JpegDecodeOp`, `JpegCapabilityRequest`, `JpegBackendEligibility`,
`JpegCapabilityReport`, and `JpegCodec`.

Stable direct-codec functions and methods: `Decoder::inspect`,
`Decoder::new`, `Decoder::from_view`, `Decoder::info`, `Decoder::bytes`,
`Decoder::passthrough_candidate`, `Decoder::decode_into`,
`Decoder::decode_region_into`, `Decoder::decode_scaled_into`,
`Decoder::decode_region_scaled_into`, `Decoder::decode_rows`,
`decode_tile_into`, `decode_tile_into_in_context`,
`decode_tile_into_in_context_with_options`,
`decode_tile_region_into_in_context`,
`decode_tile_region_into_in_context_with_options`,
`decode_tile_scaled_into_in_context`,
`decode_tile_scaled_into_in_context_with_options`,
`decode_tile_region_scaled_into_in_context`,
`decode_tile_region_scaled_into_in_context_with_options`,
`decode_tiles_into`, `decode_tiles_into_with_options`,
`decode_tiles_scaled_into`, `decode_tiles_scaled_into_with_options`,
`decode_tiles_region_scaled_into`,
`decode_tiles_region_scaled_into_with_options`, `encode_jpeg_baseline`,
`extract_dct_blocks`, `encode_baseline_dct_image`, `idct_islow_block`,
`JpegCapabilityReport::inspect`, `JpegCapabilityReport::for_decoder`, and
`JpegCapabilityReport::metal_resident_rgb8_batch_output`.

Stable Adapter Integration API: `adapter::DeviceComponentPlan`,
`adapter::DeviceDecodePlan`, `adapter::DeviceBatchSummary`,
`adapter::build_device_plan`, `adapter::summarize_device_batch`,
`adapter::decoder_bytes`, `adapter::DeviceCheckpoint`,
`adapter::JpegBaselineSampling`, `adapter::JpegBaselineHuffmanTable`,
`adapter::JpegBaselineEncodeTables`,
`adapter::build_baseline_encode_tables`,
`adapter::JpegMetalFast420PacketV1`,
`adapter::JpegMetalFast422PacketV1`,
`adapter::JpegMetalFast444PacketV1`,
`adapter::build_metal_fast420_packet`,
`adapter::build_metal_fast420_packet_for_decoder`,
`adapter::build_metal_fast422_packet`,
`adapter::build_metal_fast422_packet_for_decoder`,
`adapter::build_metal_fast444_packet`, and
`adapter::build_metal_fast444_packet_for_decoder`.

`bench_support` remains public only for benchmarks and regression tests until a
dedicated dev-support crate replaces it. It must not be used by downstream
applications.

### `signinum-j2k`

Tier: Direct Codec API plus Adapter Integration API for `adapter` and encode
stage acceleration.

Stable modules: `context`, `error`, `scratch`, `adapter`, `view`, and the
crate-level encode/recode/batch exports.

Stable direct-codec types: `J2kContext`, `J2kError`, `J2kScratchPool`,
`J2kCodec`, `J2kDecoder`, `J2kView`, `TileBatchError`, `TileBatchOptions`,
`TileDecodeJob`, `TileRegionScaledDecodeJob`, `CpuDecodeParallelism`,
`EncodeBackendPreference`, `EncodedJ2k`, `J2kBlockCodingMode`,
`J2kEncodeValidation`, `J2kLosslessEncodeOptions`, `J2kLosslessSamples`,
`J2kProgressionOrder`, `ReversibleTransform`, `J2kToHtj2kMode`,
`J2kToHtj2kOptions`, `J2kToHtj2kReport`, and `ReencodedHtj2k`.

Stable direct-codec functions and methods: `J2kDecoder::inspect`,
`J2kDecoder::new`, `J2kDecoder::from_view`, `J2kDecoder::info`,
`J2kDecoder::cpu_decode_parallelism`,
`J2kDecoder::set_cpu_decode_parallelism`, `J2kDecoder::bytes`,
`J2kDecoder::passthrough_candidate`, `J2kDecoder::decode_into`,
`J2kDecoder::decode_into_with_scratch`, `J2kDecoder::decode_region_into`,
`J2kDecoder::decode_scaled_into`,
`J2kDecoder::decode_region_scaled_into`,
`decode_tile_into_in_context`, `decode_tile_region_scaled_into_in_context`,
`decode_tiles_into`, `decode_tiles_region_scaled_into`,
`encode_j2k_lossless`, `encode_j2k_lossless_with_accelerator`,
`j2k_lossless_decomposition_levels`,
`j2k_lossless_decomposition_levels_for_options`,
`j2k_lossless_decomposition_levels_for_progression`, and
`recode_j2k_to_htj2k_lossless`.

Stable Adapter Integration API: `adapter::device_plan::DeviceDecodeRequest`,
`adapter::device_plan::DeviceDecodePlan`,
`adapter::device_plan::build_device_decode_plan`,
`EncodedHtJ2kCodeBlock`, `EncodedJ2kCodeBlock`,
`J2kEncodeDispatchReport`, `J2kEncodeStageAccelerator`,
`J2kForwardDwt53Job`, `J2kForwardDwt53Level`,
`J2kForwardDwt53Output`, `J2kForwardRctJob`,
`J2kHtCodeBlockEncodeJob`, `J2kPacketizationBlockCodingMode`,
`J2kPacketizationCodeBlock`, `J2kPacketizationEncodeJob`,
`J2kPacketizationProgressionOrder`, `J2kPacketizationResolution`,
`J2kPacketizationSubband`, and `J2kTier1CodeBlockEncodeJob`.

### `signinum-tilecodec`

Tier: Direct Codec API.

Stable types: `UncompressedCodec`, `DeflateCodec`, `LzwCodec`, `ZstdCodec`,
`TileCodecError`, `NoPool`, `DeflatePool`, `LzwPool`, `ZstdPool`, and the
re-exported `TileDecompress` trait.

Stable behavior: caller-owned buffers, explicit decompressed byte counts,
structured errors, and no hidden allocation requirement beyond each codec pool.

### `signinum-cli`

Tier: Application API.

Stable command surface: `signinum --help`, `signinum help`, and
`signinum inspect <file>`.

Stable stdout/stderr contract: successful `inspect` writes one summary line to
stdout and exits 0. File I/O and parse failures write an error to stderr and
exit 1. Usage or unknown-subcommand errors write to stderr and exit 2. `inspect`
auto-detects JPEG, raw JPEG 2000 codestreams, and JP2 containers.

### `signinum-jpeg-metal`

Tier: Adapter API.

Stable modules and types: `viewport`, `Error`, `SurfaceResidency`, `Surface`,
`Codec`, `Decoder`, `MetalBackendSession`, `MetalSession`,
`MetalSubmission`, `MetalBatchOutputBuffer`, `MetalBatchTextureOutput`,
`JpegBaselineMetalEncodeTile`, `JpegMetalResidentBatchReport`, `Info`,
`JpegRectPublic`, and `ViewportResidentOutputStrategy`.

Stable functions and methods: `Surface::pitch_bytes`, `Surface::residency`,
`Surface::as_bytes`, `Surface::download_into`, `Surface::metal_buffer`,
`Decoder::new`, `JpegMetalResidentBatchReport::required_tile_capacity`,
`Codec` trait implementations, `MetalSession` constructors and decode/submit
methods, `TileBatchDecodeManyDevice::decode_tiles_to_device` for full-tile JPEG
Metal batches,
`Codec::inspect_rgb8_decoder_batch_metal_output`,
`Codec::decode_rgb8_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_batch_into_resizable_metal_textures_with_session`,
`Codec::decode_rgb8_scaled_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_scaled_batch_into_resizable_metal_textures_with_session`,
`Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_batch_into_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_batch_into_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_scaled_batch_into_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_scaled_batch_into_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_region_scaled_batch_into_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_region_scaled_batch_into_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_scaled_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_scaled_batch_into_resizable_metal_textures_with_session`,
`Codec::decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_buffer_with_session`,
`Codec::decode_rgb8_decoder_region_scaled_batch_into_resizable_metal_textures_with_session`,
`MetalBatchOutputBuffer::ensure_rgb8_tiles`,
`MetalBatchOutputBuffer::ensure_rgb8_scaled_tiles`,
`MetalBatchOutputBuffer::ensure_rgb8_region_scaled_tiles`,
`MetalBatchTextureOutput::ensure_rgba8_tiles`,
`MetalBatchTextureOutput::ensure_rgba8_scaled_tiles`,
`MetalBatchTextureOutput::ensure_rgba8_region_scaled_tiles`,
`encode_jpeg_baseline_from_metal_buffer`,
`encode_jpeg_baseline_batch_from_metal_buffers`, and viewport helpers
`viewport_source_bounds`, `is_contiguous_viewport_workload`,
`choose_viewport_surface_strategy`,
`choose_resizable_metal_viewport_strategy`, `suggest_viewport_workload`,
`compose_viewport_cpu`, `decode_viewport_region_cpu`,
`decode_viewport_to_surface`, `decode_viewport_region_cpu_to_surface`,
`compose_viewport_cpu_to_surface`, `compose_viewport_hybrid`,
`compose_viewport_to_resizable_metal_buffer_with_session`,
`compose_viewport_to_resizable_metal_textures_with_session`,
`decode_viewport_region_hybrid`,
`decode_viewport_region_to_resizable_metal_buffer_with_session`,
`decode_viewport_region_to_resizable_metal_textures_with_session`,
`decode_viewport_to_resizable_metal_buffer_with_session`,
`decode_viewport_to_resizable_metal_textures_with_session`,
`decode_viewport_to_resizable_metal_buffer_with_decoder_session`, and
`decode_viewport_to_resizable_metal_textures_with_decoder_session`.

Stable behavior: `BackendRequest::Cpu` is host-backed, `BackendRequest::Auto`
may choose Metal and must expose residency, and `BackendRequest::Metal` returns
Metal-resident output or `Error::MetalUnavailable`,
`Error::UnsupportedBackend`, or `Error::UnsupportedMetalRequest`.

### `signinum-j2k-metal`

Tier: Adapter API plus Adapter Integration API for benchmark and DICOM helpers.

Stable types: `Error`, `SurfaceResidency`, `Surface`, `Codec`,
`MetalBackendSession`, `MetalSession`, `MetalSubmission`,
`MetalEncodeStageAccelerator`, `MetalEncodedJ2k`,
`MetalLosslessEncodeOutcome`, `MetalLosslessBufferEncodeOutcome`,
`MetalLosslessBufferEncodeBatchOutcome`, `MetalLosslessEncodeBatchStats`,
`MetalLosslessEncodeConfig`, `MetalLosslessEncodeResidency`,
`MetalLosslessEncodeStageStats`, `MetalLosslessEncodeTile`,
`SubmittedJ2kLosslessMetalEncode`,
`SubmittedJ2kLosslessMetalEncodeBatch`,
`SubmittedJ2kLosslessMetalBufferEncodeBatch`, `BenchmarkGroupedRequests`,
and `DicomFrameExtractError`.

Stable functions and methods: `Surface::residency`, `Surface::pitch_bytes`,
`Surface::as_bytes`, `Surface::download_into`, `Surface::metal_buffer`,
the `Codec` decode/submit trait implementations, all exported
`encode_lossless_from_*_metal_buffer*` and
`submit_lossless_from_*_metal_buffer*` functions,
`validate_lossless_roundtrip_on_metal`,
`validate_lossless_roundtrip_on_metal_with_session`,
`benchmark_group_region_scaled_requests`,
`benchmark_region_scaled_direct_plan_prepare`,
`extract_dicom_encapsulated_frames`, and
`extract_dicom_encapsulated_frames_with_limit`.

Stable behavior: explicit Metal decode cannot return CPU-only output; CPU-staged
uploads are observable through `SurfaceResidency::CpuStagedMetalUpload` and only
accepted by APIs that document CPU staging.

### `signinum-jpeg-cuda`

Tier: Adapter API.

Stable types: `Codec`, `Decoder`, `Error`, `CudaSession`, `Surface`,
`CudaSurface`, `CudaSurfaceStats`, `DecoderContext`, and `ScratchPool`.

Stable functions and methods: `Decoder::new`, `Surface::pitch_bytes`,
`Surface::as_host_bytes`, `Surface::download_into`,
`Surface::cuda_surface`, `CudaSurface::device_ptr`, `CudaSurface::stats`,
and `CudaSurfaceStats::{kernel_dispatches,copy_kernel_dispatches,decode_kernel_dispatches,used_hardware_decode}`.

Stable behavior: explicit CUDA returns a CUDA-backed surface or a typed
unavailable/unsupported error. Auto mode returns observable host-backed output
for JPEG CUDA requests unless a documented owned CUDA route is explicitly
selected.

### `signinum-j2k-cuda`

Tier: Adapter API.

Stable types: `Codec`, `J2kDecoder`, `CudaEncodeStageAccelerator`, `Error`,
`CudaSession`, `Surface`, `SurfaceResidency`, `CudaSurface`,
`CudaSurfaceStats`, `CudaHtj2kDecodePlan`, `CudaHtj2kProfileReport`,
`J2kContext`, and `J2kScratchPool`.

Stable functions and methods: `J2kDecoder::new`, `Surface::pitch_bytes`,
`Surface::as_host_bytes`, `Surface::download_into`,
`Surface::cuda_surface`, `CudaSurface::device_ptr`, `CudaSurface::stats`,
`Surface::residency`, `CudaSurfaceStats` dispatch-counter accessors,
`J2kDecoder` strict-device, host-surface, CPU-staged CUDA, and profiled HTJ2K
plan methods, and `CudaEncodeStageAccelerator` dispatch-counter accessors.

Stable behavior: explicit CUDA decode is reserved for strict CUDA-resident
HTJ2K codestream work and returns a typed unavailable/unsupported error when
that path cannot run. CPU-decode-then-CUDA-upload is available only through
explicit CPU-staged APIs and is visible as
`SurfaceResidency::CpuStagedCudaUpload`.

### `signinum-transcode`

Tier: Direct Codec API for coefficient-domain transcode and Adapter
Integration API for acceleration traits.

Stable modules: `accelerator`, `corpus_validation`, `dct53_1d`, `dct53_2d`,
`dct53_multilevel`, `dct97_2d`, `htj2k_wavelet`, and `metrics`.

Stable types and functions: `EncodeProgressionOrder`, `jpeg_to_htj2k`,
`jpeg_to_htj2k_batch`, `BatchTranscodeReport`, `EncodedTranscode`,
`EncodedTranscodeBatch`, `JpegTileBatchInput`,
`JpegToHtj2kCoefficientPath`, `JpegToHtj2kError`,
`JpegToHtj2kOptions`, `JpegToHtj2kTranscoder`,
`TranscodeComponentReport`, `TranscodeReport`,
`TranscodeTimingReport`, `TranscodeValidationClassification`,
`TranscodeValidationMetrics`, and
`JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE`.

Stable Adapter Integration API: `DctToWaveletStageAccelerator` and the
accelerator job/report types re-exported from `accelerator` for DCT to DWT and
HTJ2K prequantized-codeblock stages.

Stable behavior: lossless paths must round-trip through J2K/HTJ2K decode.
Lossy 9/7 coefficient transcode must report validation metrics and stay within
documented lossy/hybrid tolerances.

### `signinum-transcode-metal`

Tier: Adapter API plus Adapter Integration API for Metal transcode weights.

Stable types and constants: `METAL_UNAVAILABLE`, `MetalTranscodeError`, and
`MetalDctToWaveletStageAccelerator`.

Stable functions and methods: `MetalTranscodeError::as_static_str`,
`MetalDctToWaveletStageAccelerator::new_explicit`,
`MetalDctToWaveletStageAccelerator::for_auto`,
the threshold override methods, dispatch counter methods, and trait
implementations for `DctToWaveletStageAccelerator`.

Stable Adapter Integration API: `weights` exposes the Metal/CPU coefficient
weights used to verify parity.

### `signinum-j2k-native`

Tier: Direct Codec API for the pure-Rust engine plus Adapter Integration API
for low-level J2K/HTJ2K stage jobs.

Stable high-level types: `Image`, `DecodeSettings`, `Bitmap`, `RawBitmap`,
`DecodedComponents`, `ComponentPlane`, `ColorSpace`, `DecodeError`,
`DecodingError`, `FormatError`, `MarkerError`, `TileError`,
`ValidationError`, `ColorError`, `CpuDecodeParallelism`, `DecoderContext`,
`Reversible53CoefficientImage`, `EncodeOptions`, and
`EncodeProgressionOrder`.

Stable high-level functions: `Image::new`, `Image::decode`,
`Image::decode_components`, `Image::decode_region`,
`Image::decode_region_scaled`, `Image::decode_scaled`, `encode`,
`encode_with_accelerator`, `encode_htj2k`, `encode_precomputed_htj2k_53`,
`encode_precomputed_htj2k_53_with_accelerator`,
`encode_precomputed_htj2k_53_with_mct`,
`encode_precomputed_htj2k_53_with_mct_and_accelerator`,
`encode_precomputed_htj2k_97`,
`encode_precomputed_htj2k_97_with_accelerator`,
`encode_prequantized_htj2k_97`,
`encode_prequantized_htj2k_97_with_accelerator`, and
`idwt_band_index`.

Stable Adapter Integration API: direct-plan types, HT/J2K code-block decode
jobs, scalar decode helpers, encode-stage jobs, `J2kEncodeDispatchReport`,
`J2kEncodeStageAccelerator`, `CpuOnlyJ2kEncodeStageAccelerator`,
`HtUvlcTableEntry`, `ht_uvlc_encode_table`, and precomputed/prequantized
HTJ2K image/component/resolution/subband/codeblock types.

These native implementation types must not be re-exported by `signinum`,
`signinum-j2k`, `signinum-transcode`, or adapter public APIs. Public adapter
contracts that need J2K encode-stage types live under `signinum-j2k::adapter`
as owned wrapper types, with private conversion at the native boundary.

### `signinum-cuda-runtime`

Tier: Adapter Integration API.

Stable types: `CudaError`, `CudaContext`, `CudaDeviceBuffer`,
`CudaKernelOutput`, `CudaDwt53Output`, `CudaDwt53LevelShape`, and
`CudaExecutionStats`.

Stable functions and methods: `CudaContext::system_default`,
`CudaContext::upload`, `CudaContext::allocate`,
`CudaContext::copy_with_kernel`,
`CudaContext::copy_device_to_device_with_kernel`,
`CudaContext::j2k_forward_rct`, `CudaContext::j2k_forward_dwt53`,
`CudaDeviceBuffer::device_ptr`, `CudaDeviceBuffer::byte_len`,
`CudaDeviceBuffer::copy_to_host`, `CudaKernelOutput::into_parts`,
`CudaDwt53Output::{transformed,levels,ll_dimensions,execution}`, and
`CudaExecutionStats::{kernel_dispatches,copy_kernel_dispatches,decode_kernel_dispatches,used_hardware_decode}`.

### `signinum-profile`

Tier: Adapter Integration API.

Stable types: `ProfileStageMode`, `SummaryLabel`, and `ProfileSummary`.

Stable functions and methods: `SummaryLabel::new`, `SummaryLabel::same`,
`same_summary_labels`, `env_flag_from_value`,
`profile_stage_mode_from_value`, `profile_stage_mode_from_env`,
`gpu_route_profile_stage_mode_from_value`, `gpu_route_profile_mode_enabled`,
`gpu_route_profile_stage_mode`, `gpu_route_profile_enabled`,
`gpu_route_summary_labels`, `gpu_route_profile_summary`,
`emit_gpu_route_profile`, `duration_us_string`, `format_profile_row`,
`format_profile_row_u128`, `ProfileSummary::new`,
`ProfileSummary::counts_only`, `ProfileSummary::record_str`,
`ProfileSummary::record_u128`, `ProfileSummary::format_rows`,
`record_timing_summary_str`, `emit_profile_row`, `emit_profile_row_u128`,
and `emit_profile_row_with_timing_summary`.

## Pre-1.0 Cleanup Rules

Before 1.0, every `#[doc(hidden)] pub` item must either move to a private
module, move to a separate support crate, or be promoted in this file as
Adapter Integration API. Hidden public API is still public API for semver.

Struct fields in Application API and Direct Codec API must be public only when
callers need direct construction or matching. Otherwise use constructors,
builders, getters, and `#[non_exhaustive]` where future variants or fields are
expected.

Traits that downstream users should not implement must be sealed before 1.0.
Traits that downstream users may implement need explicit object-safety,
blanket-impl, and semver guarantees.

All strict backend requests must return precise unavailable/unsupported errors.
No silent failures, no ambiguous fallback, and no logging-only status.
