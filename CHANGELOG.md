# Changelog

Release notes for every published version, newest first.

## [Unreleased]

- A final tile part with `Psot = 0` now ends at the EOC marker, as T.800
  specifies. Before, the decoder read EOC and any bytes after it, such as the
  padding at the end of a DICOM frame, as packet data.
- **Output change:** lossless (reversible 5/3) decodes on the CPU, CUDA and
  Metal return whole numbers where a resolution is one sample wide or tall at
  an odd position, which can happen in images with a nonzero origin. That
  sample was halved to a fraction such as 169.5, which each decode route
  rounded its own way, so pixels could be one code value off.
  ([#332](https://github.com/frames-sg/j2k/pull/332),
  [#333](https://github.com/frames-sg/j2k/issues/333))
- Python wheels and command-line builds use libz-sys 1.1.33, whose bundled
  zlib fixes RUSTSEC-2026-0544. j2k calls only zlib's `inflate` and
  `deflate`, not the affected `gzprintf` and `gzwrite`.
- Lowers the minimum supported Rust version from 1.99.0 to 1.97. CI checks
  the whole workspace with Rust 1.97; every other job still uses 1.99.0.
  `j2k-types` and `j2k-codec-math` still build with Rust 1.96.
- `CudaLosslessEncoder::encode` under `EncodeBackendPreference::Auto` uses
  CUDA only for a shape whose CUDA route was measured faster than the CPU, as
  decode routing does. No encode shape is listed yet, so `Auto` now encodes on
  the CPU; `fallback_reason()` is `None` for it, because the CPU is the chosen
  route. It used to run every CUDA stage it had, and the routing runs
  measured CUDA-assisted encode slower than the CPU in every cell (528%
  slower for HTJ2K lossless). `RequireDevice` still forces CUDA. The public
  API is unchanged.
- Python `Decoder.decode` emits `RecoveredImageWarning` when a lenient JPEG
  2000 decode used a metadata recovery, as `decode_srgb8` does. It dropped
  the warning before.
- Licensing: parts of the JPEG codec are derived from libjpeg-turbo, which
  `NOTICES.md` denied. `j2k-jpeg`, `j2k-codec-math`, `j2k-jpeg-metal`, and
  the Python wheel now carry libjpeg-turbo's license texts
  (`LICENSE.libjpeg-turbo`, `README.ijg`), and the derived files name their
  upstream sources.
- Codestreams that are cut off, such as the bytes a progressive viewer has
  fetched so far, can be decoded with
  `DecodeSettings::with_partial_codestream(true)`.
  `J2kDecoder::codestream_completeness` reports the smallest reduction whose
  data is all present, and a decode at that reduction or deeper returns the
  same samples as the complete file. A decode that needs missing data fails
  with the new `J2kError::IncompleteCodestream`. Raw codestreams and JP2/JPH
  files cut inside their codestream box are accepted; decoding runs on the CPU.
- **Breaking:** `j2k_native::DecodeSettings` has a new public field,
  `partial_codestream`. Code that builds the struct with a literal must set
  it, or start from `DecodeSettings::strict()` or `DecodeSettings::lenient()`.

## [0.14.0] - 2026-11-12

### Upgrading from 0.13

- **Rust code mostly builds unchanged.** The public API grows. Outside
  eight CUDA error fields now spelled `Box<Self>`, which callers build and
  match as before, the reviewed API diff removes nothing. Three changes can
  need an edit:
  - Code that builds `JpegToHtj2kOptions` with a struct literal must add
    the new `precision` field or use a constructor.
  - Decodes that exceed a memory or image-size limit fail with
    `J2kError::ResourceLimit` instead of `J2kError::NativeDecode`.
  - Metal Tier-1 device failures are `Error::Decode` and
    `J2kError::Backend` (displayed as `backend failed (decode):`) instead
    of `Error::MetalKernel`.
- **Python wheels require CPython 3.11 or later** (abi3-py311).
- **Security:** a JP2 with a crafted ICC profile could make decoding
  allocate memory quadratic in the profile's size, outside every decode
  limit (a 29 KB profile held about 200 MB). Upgrade if you decode
  untrusted files.
- **Output that changes bytes:**
  - `Gray16`/`Rgb16`/`Rgba16` output of signed sources of at most 8 bits
    is 16-bit two's complement on the CPU, Metal and CUDA; 0.13 turned
    negative samples into 65535.
  - 16-bit output of images deeper than 16 bits holds each sample clamped
    to 16 bits, and 8-bit output of such images is quantized over the
    component's range; 0.13 wrote bytes of the wider samples as samples.
  - The lossy 9/7 JPEG-to-HTJ2K transcode computes in canonical binary32
    on every backend: 19 of 60 test outputs changed bytes, none changed
    length, and the worst PSNR change was -0.0019 dB.
    `TranscodePrecision::F64` reproduces the 0.13 CPU bytes.
  - Baseline JPEG encode writes the same bytes on the CPU, Metal and CUDA
    and on every platform. Coefficients within 2^-20 of a rounding tie can
    round differently from 0.13, and Metal and CUDA pad partial edge
    blocks by repeating the edge, as the CPU does, instead of with zeros.
  - Lossy encodes to a byte or bits-per-pixel target can choose a
    different quantization within the size allowance.
  - CUDA JPEG decode, and more GPU JPEG 2000 decode and encode paths, now
    match the CPU byte for byte.
  - Reduced-resolution decodes of images whose reference-grid origin is
    not a multiple of the scale return the samples the standard places
    there, as OpenJPEG does, which can be one sample narrower. Images at
    origin zero are unchanged.
  - Lossless coding of samples deeper than 20 bits is exact. It ran in
    `f32`, which holds the reversible transforms' values only through 20
    bits: a lossless encode of 21- to 24-bit samples could decode to other
    samples in any decoder, and a lossless file from another encoder could
    decode a few code values off. These encodes and decodes now use
    integers, as 25- to 38-bit ones already did. They take 2 to 7 times as
    long and about 2.6 times the memory, and a full-image decode past about
    15 megapixels of one such component reaches the decode limit; region
    and bounded-row decodes do not.
  - Fixed wrong output: the last row of a strided caller buffer, region
    decodes of tiles with empty coarse resolutions, ROI encodes whose
    shift overflowed 32-bit coefficients, reduced decodes that skipped
    packets by a too-short PLT length, irreversible ROI decodes whose
    background reached half the maxshift threshold, 33- to 38-bit unsigned
    colour-transformed images, 8-bit direct-plan output of samples deeper
    than 16 bits, and CUDA reversible JPEG-to-HTJ2K transcodes of blocks
    with very large DC terms.
- **Stricter validation.** Files that 0.13 accepted can now fail:
  - SOP segments need `Lsop` 4 and the right `Nsop`;
  - a PLT length too short for a packet that a region or reduced decode
    skips is a packet error;
  - ICC profiles with repeated tags or more than 256 tags are rejected,
    and lookup-table tags are refused before allocation;
  - JP2 palette columns deeper than 38 bits are an invalid box;
  - a lossless image of 21- to 24-bit samples whose tiles or components mix
    the reversible and irreversible wavelets, or whose tiles differ in the
    colour transform, is unsupported, as deeper ones already were;
  - GPU and batch plans reject a one-component codestream that asks for a
    colour transform, as single decodes already did.
- **More files decode:** tile-parts with data after the last packet; tile
  COD markers that turn packet markers off; JP2 files with extra
  unspecified components (such as Kakadu RGBA files written without
  `-jp2_alpha`); ICC profiles equivalent to sRGB into `Rgb8`/`Rgba8`; and
  images whose component planes would exceed the decode budget, which now
  stream their rows instead of failing.
- **JPEG files with a reserved Adobe APP14 transform byte:** strict
  decoding, the default, still returns `JpegError::AmbiguousColorTransform`.
  `DecodeOptions::with_strict(false)` (Python `strict=False`) now reads
  them as libjpeg-turbo does, as YCbCr for three components and YCCK for
  four, with a warning.
- **JPEG region decodes reject empty regions.** A region with zero width
  or height returns `JpegError::RectOutOfBounds`, as a region outside the
  image and `lossless_crop` already do.
- **C and C++:** the C API now lives in this repository as `crates/j2k-c`;
  ABI 1 layouts are unchanged. See `crates/j2k-c/CHANGELOG.md`.
- New dependency: `bytemuck`.

### Highlights

New:

- **Exact regions and resolutions.** `J2kDecoder::region_decode_plan`
  plans a region at an exact JPEG 2000 reduction before decoding (output
  rectangle, buffer size and route). `PreparedImageOperation` with
  `decode_operations` decodes such regions of a prepared image on the CPU,
  Metal and CUDA without parsing it again, and `decode_region_srgb8`
  decodes sRGB regions from the region's wavelet window instead of
  cropping a full decode.
- **Decode controls.** `J2kDecodeControl` gives one operation a working
  memory limit, an output limit and a `J2kCancellationToken`, from parsing
  through planning, batch preparation and decoding.
- **Streaming instead of failing.** 8-bit and 16-bit full-image and region
  decodes whose component planes would exceed the decode budget stream
  their rows: a 5661 x 8287 RGB image decodes with about 100 MiB peak
  heap, where the planes route needs about 980 MiB. `full_decode_plan` and
  `rows_decode_plan` report the planned route and why rows cannot stream.
- **C and C++ API** in the repository: the C ABI, a header-only C++17
  wrapper and relocatable `libj2k` packages for CMake and pkg-config. It
  adds encoding (rate, size and PSNR targets, quality layers, Qfactor,
  tiles, markers, ICC profiles), colour reporting, exact region operations
  and cancellation.
- **Python:** `j2k.encode`; `j2k.inspect` reports colour, ICC, resolution
  levels, tiling, quality layers and progression; `Decoder.plan_region`,
  `decode_srgb8`, `DecodeControl` and `CancellationToken`; any bytes-like
  input, read in place where possible; and CUDA and MPS decoders run exact
  operations.
- **IIIF and Cantaloupe.** `inspect_coding_style` and `J2kReferenceGrid`
  report what an image server needs, and `docs/iiif.md` maps the IIIF
  Image API 3.0 onto the Rust, C, Python and CLI interfaces. `j2k decode`,
  also run as `opj_decompress`, writes BMP, PNG, TIFF and PNM with
  opj_decompress options, so Cantaloupe's OpenJpegProcessor can use it.
- **JPEG:** lossless MCU-aligned crops of baseline and progressive files,
  and coefficient extraction that keeps only the quantized blocks it
  needs.
- **Same bytes on every backend.** Baseline JPEG encode, CUDA JPEG decode,
  the 9/7 transcode and more GPU JPEG 2000 paths now match the CPU byte for
  byte. Their constants have one owner, `j2k_codec_math`, and the Metal
  tables are generated from it.

Faster. Measured end to end against the pre-release `main` (f1a5441e), in
ten interleaved rounds, on an Apple M5 Max, an Apple M4, and a Ryzen 7
5800X3D with an RTX 4070 SUPER:

- **Encoding to a size or bits-per-pixel target takes 77-86% less time**
  (4-7x as fast), because the rate search runs about 3 encodes instead of
  about 15. A 4096 x 2048 RGB8 classic encode at 2 bpp went from 316 to
  54 ms on the M5 Max.
- **Gray and high-bit-depth HTJ2K lossless encodes take 47-66% less
  time:** 4096 x 2048 Gray8 from 14.7 to 5.7 ms, and a 12-bit
  2394 x 3062 mammogram from 13.8 to 4.7 ms, on the M5 Max. Classic Gray
  encodes at 2 bpp take 55-59% less, 12-bit CT HTJ2K encodes 17-32% less
  and 512 x 512 RGB8 HTJ2K encodes 14-21% less.
- **Metal decode:** classic lossless 4096 x 2048 RGB takes 48-70% less
  time (57.5 to 29.9 ms on the M5 Max, 209 to 63 ms on the M4), and HTJ2K
  1024 x 1024 regions 58-69% less (11.1 to 3.4 ms on the M5 Max).
- **GPU JPEG encode:** a CUDA batch of sixteen 512 x 512 tiles takes 52%
  less time, and a Metal batch 14-19% less.
- **CPU JPEG decode takes 4-31% less time in every baseline case
  measured:** a batch of 64 slide tiles 17-31%, full 4096 x 2048 images
  8-25% and 1024 x 1024 regions 4-14%. Metal JPEG decode takes 10-15%
  less.
- **CPU JPEG 2000 decode on x86-64:** on the Ryzen, most full decodes of
  gray, 12-bit and 16-bit images take 16-35% less time, and RGB decodes
  take about the same time, as do JPEG 2000 decodes on the Apple CPUs.

The end-to-end figures leave out changes that were already on `main`
before f1a5441e. The full list gives each change's own measurements,
for example: bounded row decodes 10-14x as fast (12.9 to 1.34 s for a
5661 x 8287 RGB image on one thread), overviews of an 18000 x 12000 image
47-53% faster at 18 threads, warm JPEG device DCT windows 3.3-180x as
fast, and repeated deep seeks into restart-coded JPEGs 25-128x as fast.

### All changes

The full list of 134 changes, with their measurements and root causes, is in
the [changelog as tagged for this release](https://github.com/frames-sg/j2k/blob/v0.14.0/CHANGELOG.md#all-changes).

## [0.13.1] - 2026-10-29

- `CpuBatchDecoder` region and region-plus-reduced requests no longer
  allocate coefficient and IDWT buffers for whole resolution levels
  ([#299](https://github.com/frames-sg/j2k/issues/299)). Preparation keeps
  the shared whole-level plan that Metal and CUDA execute, and adds a CPU copy
  whose sub-bands and IDWT levels hold only what the region's stores read,
  with the windows native region decode uses. On an 18000 x 12000 RGB image,
  1024 px tiles at full and half resolution failed with `ImageTooLarge`; they
  now decode with output identical to single-image region decode. Quarter
  resolution, which already fit, takes 24 ms instead of 48 ms for four HTJ2K
  tiles and 310 to 317 ms instead of 343 to 346 ms for classic. The CPU uses
  the copy only when it at least halves the plan's coefficients or the
  whole-level plan holds 16 Mi coefficients or more, because the windowed
  IDWT is slower per sample than the banded whole-level one; full and reduced
  requests, and regions covering most of a small image, are unchanged. See
  `docs/benchmark-evidence.md`.
- Owned-batch preparation no longer fails with "failed to decode code-block"
  on reduced-resolution regions that end where a tile begins on the reduced
  grid ([#298](https://github.com/frames-sg/j2k/issues/298)). Tile selection
  rounds tile bounds outward, so it can keep a boundary tile that contributes
  no samples; the referenced plan builders rejected such a tile and now leave
  it out. This affected the CPU and Metal batches alike. On the 19 x 13
  OpenJPH fixture in 11 x 7 tiles, 4,598 of the 17,290 half-scale regions
  failed to prepare; all now match single-image decode.
- All-stage Metal lossy encodes seed code-blocks without coding passes with zero
  in the zero-bitplane tag tree, as the CPU encoder does, so their packet headers
  match the CPU's byte for byte. Earlier codestreams were valid but could differ
  from the CPU's by a few header bytes.
- CPU classic (Part 1) code-block decoding reads several MQ code bytes ahead
  per refill instead of one per renormalization, visits each column's
  candidate rows in the significance propagation, refinement and cleanup
  passes as a bit set instead of testing every row, passes each row's window
  shift instead of its row to the context helpers (the classic encoder shares
  them), picks the next MQ probability state with one shift, and indexes its
  contexts by label without a bounds check. On a 4096 x 2048 RGB image,
  one-thread decodes are 11% faster lossless and 9% faster at 2 bpp, and
  12-thread lossless decodes 11% faster; eight concurrent 1024 x 1024 region
  decodes of an 18000 x 12000 image are 9% faster. Decoded samples and
  encoded codestreams are unchanged. See `docs/benchmark-evidence.md`.
- `j2k-tilecodec` LZW decode no longer reports valid TIFF strips and tiles
  as truncated. weezl reads input ahead of decoding, so the call that
  consumes the last byte can still hold codes up to the end code, often in
  short streams; decoding stopped there with a truncation error. It now
  finishes the buffered codes, and reports truncation only when no input is
  left and decoding makes no progress. As in libtiff, a stream with no end
  code decodes when it fills the output and is reported as truncated when it
  stops short; a cut that loses data is always truncated. An empty or nearly
  empty stream is reported as truncated rather than malformed.
- Fixed CPU decodes of codestreams whose tile-part headers set their own
  coding style ([#295](https://github.com/frames-sg/j2k/issues/295)). The
  inverse DWT and the rounding of integer output used the main header's
  wavelet, so a tile-part COD or COC that changed the wavelet gave the wrong
  samples in full, region, row and reduced decodes. Each tile now uses its
  own coding style. Images whose tiles differ in their multi-component
  transform, or in its wavelet, apply it per tile instead of applying the
  first tile's to every tile. A tile-part COC or QCC now takes precedence over
  the tile-part COD or QCD whichever comes first in the header, as T.800
  specifies. Tiles of 25 to 38-bit images that differ in needing exact
  integer decode are rejected with `UnsupportedFeature` rather than decoded
  wrong. Output of codestreams without tile-part coding overrides is
  unchanged.
- `MetalBatchDecoder` region and region-plus-reduced requests no longer
  allocate coefficient and IDWT buffers for the whole decoded level. Their
  prepared plans already store only the region; preparation now narrows each
  sub-band and IDWT window to what that store reads, as the single-image Metal
  region route does. On an 18000 x 12000 RGB image, a batch of 1024 px tiles
  at full or half resolution requested a 2.6 GB or 648 MB buffer, failed the
  512 MiB cap, and fell back to the CPU; those tiles now decode on Metal with
  output identical to the CPU. Quarter resolution, which already fit, takes
  9 ms instead of 36 ms for four HTJ2K tiles and 137 to 146 ms instead of
  about 165 ms for classic. Cropping a prepared HTJ2K plan whose payloads stay
  in the input no longer fails when it drops code-blocks. See
  [`docs/benchmark-evidence.md`](docs/benchmark-evidence.md).
- Owned batch preparation accepts images whose components use different
  sampling factors, such as YCbCr 4:2:2, instead of reporting
  `NonRepresentableReason::ComponentSubsampling`. Decode replicates the
  subsampled components onto the image grid, as single-image decode does; the
  CPU batch decoder matches it for full, region, reduced, and
  region-plus-reduced requests. These images retain metadata only, so they
  never share an execution group with unit-sampled images. Images whose
  components all share one non-unit sampling still report
  `ComponentSubsampling`, because native decode then produces the smaller
  component grid.
  `MetalBatchDecoder` decodes full-image subsampled RGB without MCT, such as
  Aperio 33003 YCbCr 4:2:2 tiles, in the group's one command buffer and
  replicates chroma on the GPU before the final store. 32 tiles of
  `JP2K-33003-1.svs` take 11 to 30 ms instead of 150 to 390 ms through one
  Metal decode per tile; see
  [`docs/benchmark-evidence.md`](docs/benchmark-evidence.md). Region and
  reduced subsampled requests on Metal, and subsampled groups on CUDA, now
  fail as a group with a capability error (`UnsupportedMetalRequest` or
  `UnsupportedCudaRequest`) instead of an indexed preparation error; callers
  that fell back to single-image decode on `NonRepresentableBatchOutput`
  should do the same for these group errors.
- Serial CPU region decodes on the row-streamed IDWT decode each sub-band's
  code-blocks one code-block row at a time, as the IDWT reaches it, and hold
  only that row instead of the sub-band's whole window. A 1024 x 1024 region
  of an 18000 x 12000 RGB image holds under 1 MiB of coefficients instead of
  5 MiB, and peaks at 14 to 16 MiB of heap instead of 18 to 20 MiB. Every
  code-block the whole-window path decodes is still decoded, and a failing
  decode reruns Tier-1 over whole windows to report the same error. Parallel
  decodes keep whole windows. Output is unchanged.

- Serial 8-bit region decodes (`decode_region_into`,
  `decode_region_scaled_into` and `decode_region_scaled_pow2_into` with
  `Rgb8`, `Rgba8` or `Gray8`) decode all components together and write each
  row into the caller's buffer as the streamed IDWT finishes it, instead of
  filling whole component planes first. A 1024 x 1024 RGB region of an
  18000 x 12000 image peaks at about 3.5 to 6 MiB of heap instead of 14 to
  16 MiB, not counting the caller's buffer. Output is unchanged. A decode that
  fails may leave part of the region written. A decode that streams gives up
  the decoder context's retained scratch and component planes before the first
  row, so a reused context fits exactly what a fresh one does without retrying
  and writing rows twice. Parallel decodes, and images with a palette, channel
  definitions, sYCC or CIELab, still fill planes first. The native
  `Image::decode_region_rows_for_integer_output_with_context` and
  `ComponentRowSink` expose this for other callers.

- Security: the lockfiles move to libz-sys 1.1.31, which fixes
  RUSTSEC-2026-0418. Its build script printed the build environment into
  `target/` when built with cc 1.6 or later on Unix. J2K's locked builds use
  cc 1.2 and were not affected. `cargo install --locked j2k-cli` now builds
  the fixed version.

## [0.13.0] - 2026-10-26

Summary. The full notes, with measurements and implementation detail for each
change, are in the
[changelog as tagged at v0.14.0](https://github.com/frames-sg/j2k/blob/v0.14.0/CHANGELOG.md#0130---2026-10-26).

### Upgrading from 0.12

- Native `EncodeOptions` struct literals must add
  `irreversible_quantization_profile: None`, or use `Default` with struct
  update syntax.
- `J2kDirectGrayscalePlan` has a required `signed` field. Set it from the
  source component's signedness, including in the component plans of colour
  and RGBA plans.
- `BatchDecodeOptions` is `#[non_exhaustive]`: start from
  `BatchDecodeOptions::default()` and assign fields. `BatchLayout` is defined
  in `j2k-core` and still re-exported by `j2k`.
- Exhaustive matches need new arms: `JpegEncodeError` gains
  `RestartIntervalTooLarge`, `UnsupportedPixelFormat`, `UnsupportedPrecision`,
  `InvalidPredictor`, `InvalidPointTransform`, and `SampleOutOfRange`;
  `JpegResolvedDecodePath` gains `MetalHybrid`. `j2k_jpeg::ColorTransform`,
  `BurnBatchTensor`, and `BurnDecodeError` are `#[non_exhaustive]`.
- `j2k_jpeg::DecodeOutcome` and `DecodedTile` add `recovered`.
  `BurnBatchDecode` and `BurnBatchGroup` fields are private; use their
  accessors, `into_groups`, `into_tensor`, and `j2k_info`.
- JPEG decode is strict by default and returns `JpegError::RecoveryRequired`
  for damaged input. `DecodeOptions::with_strict(false)` accepts the supported
  recoveries.
- The hidden `irreversible_quantization_step_for_subband` helper takes the
  sub-band level and the decomposition depth as its last two arguments.
- Removed: the `cuda-oxide-j2k-ml` feature of `j2k-cuda-j2k-engine`, the
  hidden `CudaJ2kMl*` types, and `J2kCudaEngine::j2k_ml_convert_into_external`.
- Security: `j2k-tilecodec` requires zstd 0.14.3, which fixes
  RUSTSEC-2026-0391, RUSTSEC-2026-0398, RUSTSEC-2026-0403, and
  RUSTSEC-2026-0410. Only the last affects `ZstdCodec`, on wasm32 builds that
  decode untrusted Zstandard tiles.
- Release evidence: published without the CUDA full release validation job
  and the CUDA T.803 report, under
  [`cuda-waiver-0.13.0.md`](xtask/release-evidence/cuda-waiver-0.13.0.md).
- The full compatibility review is the
  [0.13.0 API review](xtask/release-evidence/public-api/public-api-review-0.13.0.yml).

### Output and behavior that changed

- Default irreversible 9/7 quantization steps follow each band's synthesis
  norm. Default-scale lossy codestreams are larger and more accurate, and
  rate-targeted encodes gain fidelity at equal bytes.
- An explicit lossy `max_decomposition_levels` selects that many levels,
  capped by geometry. It used to cap only the default, so LRCP and RLCP
  encodes used one level. Lossy encodes that set it produce different
  codestreams.
- A reduced decode deeper than the codestream's resolution ladder fails with
  `requested reduction exceeds the codestream resolution ladder` instead of
  being clamped.
- Reversible coefficients whose low bitplanes were not decoded are
  reconstructed at the midpoint (T.800 Equation E-8, r = 1/2). Lossless files
  from encoders that stop coding passes early, such as GDCM/OpenJPEG 2.0,
  decode exactly instead of off by up to 2, on the CPU, CUDA, and Metal.
- JPEG decode matches libjpeg-turbo in three more places: YCCK is converted
  through complemented CMY before K is applied (every Adobe YCCK JPEG decoded
  to wrong colors before); subsampled lossless components are replicated, not
  smoothed; and color spaces follow libjpeg-turbo's defaults, so unmarked
  three-component lossless frames are RGB.
- Metal: an explicit `Gray16` decode of a signed grayscale image returns the
  CPU's bytes. Negative samples were clamped to zero.
- Metal and CUDA decodes follow the CPU's sample conversion for signed
  grayscale and colour sources, and CUDA's requested-format precision
  thresholds and synthesized 16-bit alpha match the CPU.
- Irreversible quantization steps outside the QCD/QCC range saturate at the
  coarsest or finest stored step instead of panicking or wrapping.
- Reduced decodes of images whose origin is not a multiple of the reduction
  factor no longer panic on edge tiles without samples, and match the
  standard's output size.
- `Auto` full-image RGB8 decodes keep tiled images on the CPU. Explicit Metal
  requests are unchanged.
- A `MetalBatchDecoder` grayscale group whose stacked coefficients exceed one
  512 MiB allocation no longer fails; it decodes in runs that fit.
- Some precinct layouts that exceeded the structural decode cap now decode.
- JPEG device planning accepts 0xFF fill bytes before markers, as CPU decode
  already did.
- Metal and CUDA packetization accept empty edge sub-bands in a non-empty
  precinct.
- Rate search resolves both bounds of a quantization profile's scale and keeps
  a feasible minimum-PSNR candidate after an initial quality dip.

### Added

- `J2kDecoder::decode_wavelets`: selected wavelet bands on the CPU without
  the inverse transform, as exact `I64` (5/3) or dequantized `F32` (9/7), with
  the `WaveletSelection`, `WaveletImage`, and related types.
- The unpublished `j2k-python` crate, built as the `j2k` wheel: inspection,
  native-dtype pixel batches, and wavelet bands for PyTorch and NumPy through
  DLPack. Its `cuda` feature decodes into PyTorch-owned CUDA storage and its
  `metal` feature returns `mps` tensors. `decode` and `decode_batch` take
  8-bit JPEG alongside JPEG 2000, and `ImageRequest` takes JPEG color hints.
  It also has the research helpers `encode_rgb`, `decode_rgb`,
  `forward_rgb97`, and `inverse_rgb97`.
- `j2k_jpeg::JpegBatchDecoder`: dense RGB8 batches of complete JPEGs on the
  CPU, in NCHW or NHWC.
- `CpuBurnDecoder::decode_batch` takes mixed JPEG and JPEG 2000 input.
- `j2k-jpeg` decodes the remaining JPEG classes libjpeg-turbo decodes:
  sequential frames split across several scans, progressive CMYK and YCCK,
  12-bit CMYK and YCCK, and lossless SOF3 at 2 to 16 bits. Output matches
  `djpeg` and DCMTK's `dcmdjpeg` byte for byte.
- `j2k-jpeg` encoders for the forms DICOM writes: `encode_jpeg_lossless`,
  `encode_jpeg_extended12`, and `encode_jpeg_baseline_optimized`.
- GPU lossless JPEG encode, byte-identical to the CPU:
  `encode_jpeg_lossless_from_metal_buffer` (6.2 ms against 39.7 ms on the CPU
  for a 2048x2500 16-bit image on an M4 Pro) and
  `encode_jpeg_lossless_from_cuda_buffer`.
- JPEG Metal decode covers every JPEG the CPU decoder supports, through hybrid
  decode (CPU entropy stage, Metal for the rest), in single decodes, batches,
  viewports, and sessions. `j2k-jpeg` adds `metal_hybrid_eligibility`,
  `JpegCapabilityReport::metal_hybrid`, and
  `JpegResolvedDecodePath::MetalHybrid`.
- Per-component quantization profiles for irreversible 9/7 encodes:
  `J2kLossyEncodeOptions::with_quantization_profile` and
  `EncodeOptions::irreversible_quantization_profile`.
- Image opening and header inspection accept dimensions beyond 60,000 pixels.
- Multi-tile encoding carries absolute tile-component origins, so tile grids
  that were rejected for wavelet, block, or precinct misalignment now encode.
  Encode accelerators can implement `encode_forward_dwt53_at_origin` and
  `encode_forward_dwt97_at_origin`.
- `J2kDecoder::decode_into_metal` and `decode_into_metal_with_session`
  (`j2k-metal`): decode one full image into a caller-owned
  `MetalImageDestination`.
- The tile-batch functions, `prepare_batch`, and `CpuBatchDecoder` scale their
  concurrent-decode limit with host memory (up to half of it, at 512 MiB per
  decode) instead of a fixed four.
- Default CPU JPEG-to-HTJ2K batches process whole tiles in bounded parallel
  waves.
- `j2k-native` has a `profile` feature, on by default, for stage profiling.
  Without it a wasm32 decoder is 16.8 KB smaller gzipped.
- `J2kEncodeStageAccelerator::declines_every_stage`, and
  `HtCodeBlockDecoder` classic hooks with an explicit
  `J2kClassicReconstruction` contract.
- `j2k-metal-support` adds `checked_shared_heap` and
  `checked_heap_shared_buffer`.
- The `jp2k_dicom_bench` comparison binary and the `dicom_decode_bench`
  example.

### Faster and smaller, with unchanged output

- **CPU HTJ2K decode** is faster on every path: vectorized native-bit-depth
  conversion, one parallel pass over a tile's code blocks, a row-streamed
  inverse wavelet transform, and vectorized cleanup decoding.
- **CPU classic decode** decodes a component's code blocks tile-wide in
  parallel, and one-thread decodes write straight into coefficients: a
  4096x2048 RGB image needs about 25 MB less heap.
- **CPU colour decode** fuses the last wavelet level, the inverse colour
  transform, and RGB8 packing, and decodes the tiles of a multi-tile image
  concurrently under `CpuDecodeParallelism::Auto`.
- **Region decode** keeps less state: a 1024x1024 region of an 18000x12000
  RGB image peaks at about 20 MiB of heap instead of up to 32, and
  `decode_region_into` releases its planes when it returns.
- **CPU encode** runs more stages on the pool and frees coefficients earlier:
  a lossless classic encode of a 4096x2048 RGB image peaks at about 135 MB of
  heap instead of 219 MB. Rate and PSNR searches run the colour transform and
  wavelet once per search.
- **CUDA** HTJ2K tile encodes use bounded dispatches and pooled device
  buffers, and colour region decodes plan only the code blocks, packets, and
  wavelet levels the region needs.
- **Metal** decode runs each inverse wavelet level as one tiled kernel, takes
  buffers from runtime-owned heaps, and fuses the last level with the final
  store. A 3064x4774 `Gray16` frame decodes in 3.35 ms instead of 7.66 ms
  through a new decoder, and a 16-frame batch in 11.7 ms instead of 22 ms.
- **Metal** encode reuses pooled buffers and runs the lossless single-tile
  transform in one dispatch per level.
- JPEG Metal `Decoder::new` no longer hashes its input twice: about 2.7 ms
  less per 5 MB input.

## [0.12.0] - 2026-10-06

- Breaking (experimental `j2k-cuda-j2k-engine`):
  `J2kCudaEngine::upload_j2k_decode_payload_with_pool` and
  `upload_htj2k_decode_resources_with_tables_and_pool` take the payload as
  parts (`&[&[u8]]`) instead of one slice; pass `&[payload]` for a single
  buffer. `CudaQueuedHtj2kCleanupGroup::new` takes two more `usize`
  arguments. This requires a minor release.
- Raises the minimum supported Rust version to 1.99.0. `j2k-types` and
  `j2k-codec-math`, which the CUDA kernels share, still build with Rust 1.96.
- CUDA color and grayscale batches decoded from encoded bytes plan images in
  parallel and no longer concatenate payloads into one shared buffer.
  Single-tile RGB images whose blocks are cleanup-only upload straight from
  the codestream into pinned memory; other inputs are copied into a per-image
  buffer first. Batches built from prepared native color plans still use one
  shared buffer.
- Color batches of more than 4 images and 16 MiB are decoded in groups of 4,
  planning the next group while the current one runs. Images of different
  shapes and bit depths share one batched wavelet and store sequence.
- Decode planning no longer allocates host coefficient buffers, which lowers
  CUDA host memory use, and code blocks with no data are skipped.
- CUDA HTJ2K batches whose blocks are all cleanup-only decode with two
  kernels; batches with refinement passes keep the single kernel. Full-image
  RGB8 and RGBA8 batches finish the last wavelet step together with color
  conversion and reuse pooled output buffers.
- CUDA queues every pooled pinned upload on a separate transfer stream, and
  fills pinned upload memory from up to four threads for uploads over 4 MiB.
- CUDA integer output from irreversible (9/7) images is rounded before the
  level shift, as the CPU decoder does. Before, a few samples could differ by
  one from the CPU output.
- A CUDA session keeps up to 2,048 idle batch buffers (2 GiB) for reuse, and
  the pinned staging cache keeps up to 32 buffers instead of 8.
- Dropping a pooled RGB8 or RGBA8 CUDA surface returns its memory to the
  session pool without synchronizing. Keep the surface alive until other
  streams are done with `device_ptr`.
- Classic (non-HT) CUDA block decoding keeps its lookup tables in on-chip
  memory and skips stripe columns that are already fully decoded.
- A failing GPU job in a CUDA color batch is reported as `CudaTier1JobFailed`
  with its source image and job index, whatever the batch size.
- Starting a pinned-upload transaction in `j2k-cuda-runtime` on a thread that
  already holds one returns an error instead of deadlocking.
- Metal stacked batches accept empty high-frequency sub-bands in small and
  reduced-resolution images while checking each nonempty band's full range.
- Metal classic RGB batches run the Tier-1 work of every prepared group in
  one GPU dispatch, with code blocks packed into SIMD groups by estimated
  cost. The 24-image Kodak batch decodes GPU-only in about 64 ms on an M4 Pro,
  down from 103 ms. `MetalBatchDecoder::submit_prepared_groups_into` submits
  several prepared groups through that shared dispatch.
- `MetalBatchDecoder::decode_prepared_cooperative` decodes the Tier-1 work of
  part of a large classic RGB8 group on a bounded number of CPU workers while
  Metal decodes the rest. `MetalDecodeDispatchReport` gains
  `cpu_tier1_images`.
- Metal plan preparation no longer copies classic code-block data into GPU
  buffers that the batch routes never read. Routes that need them build them
  on first use.
- Calling `MetalBatchDecoder::submit_prepared_group_into` once per group is
  slower than before for multi-group batches (about 140 ms vs 103 ms on the
  Kodak batch). Use `submit_prepared_groups_into` for those.

## [0.11.3] - 2026-09-28

- Bounds native JPEG 2000 tag-tree construction to populated branches. Tall,
  narrow code-block grids no longer expand empty quadrants, preventing the
  truncated raw-codestream timeout found by the wsi-rs release fuzz campaign.

- 12-bit extended and progressive JPEG decodes 4:4:0, 4:1:1, 4:1:0 and 1x4
  color layouts, and any other layout whose component factors divide the
  maximum factors, to `Rgb16`/`Rgba16` at full size and at every DCT scale.
  These layouts render through the 12-bit component-plane pipeline and match
  the libjpeg-turbo 3.1.4.1 reference matrix bit for bit. CPU capability
  reports mark them eligible. 12-bit CMYK/YCCK still supports 4:4:4, 4:2:2
  and 4:2:0 only.

## [0.11.2] - 2026-09-27

- Fixes subsampled JPEG 2000 decoding with nonzero image origins at reduced
  resolutions and for cropped regions, including the wsi-rs fuzz reproducer.
- Improves JPEG entropy decoding, extended and lossless sample output,
  progressive scan handling, and AArch64 NEON conversion and IDCT.
- Improves native bitplane scanning and full-resolution subsampled placement.
- Improves grouped JPEG Metal decode, pooled output reuse, HT cleanup,
  and inverse wavelet transform dispatch with regression coverage.

- JPEG DCT-scaled decodes (1/2, 1/4, 1/8) now match libjpeg-turbo, and
  therefore OpenSlide's NDPI/VMS levels, bit for bit. Checked against a
  committed libjpeg-turbo 3.1.4.1 reference matrix covering every chroma
  layout, restart intervals, progressive scans and 12-bit precision:
  - Reduced IDCTs round like libjpeg's `DESCALE` instead of truncating.
  - Each component gets libjpeg-turbo's IDCT size: 4:2:0 and 4:1:0 chroma is
    decoded with a larger reduced IDCT, so 4:2:0 needs no upsampling below
    full size.
  - Chroma is replicated instead of smoothed at 1/8 scale, and at any scale
    (including full size) when 2:1 chroma is at most two samples wide.
  - Progressive images apply the reduced IDCT to their coefficients instead
    of decimating a full-size decode.
  - 12-bit images use 12-bit reduced IDCTs instead of decimating a full-size
    decode. 12-bit 4:4:0, 4:1:1, 4:1:0 and 1x4 layouts still return
    `NotImplemented`.
- Fixes a panic in 12-bit region decodes when the region excluded image
  blocks on its left or right.
- Includes output rows in the 12-bit scratch budget so tall, narrow JPEGs
  decode without false memory-cap failures.
- `j2k-jpeg-cuda` CPU-backed scaled and region-scaled surfaces now have the
  scaled dimensions; they were sized from the source rectangle.
- JPEG Metal scaled decodes follow the same rules, and 4:2:0/4:2:2 region
  decodes keep the neighbouring chroma their smoothing reads when a region
  ends inside the image. JPEG images at most four pixels wide with 4:2:0 or
  4:2:2 chroma are no longer Metal or CUDA fast shapes: explicit Metal
  requests reject them and automatic routing decodes them on the CPU.

- Upgrades `fearless_simd` to 1.0.0 for the `j2k-jpeg` AArch64 NEON kernels
  and the optional `j2k-native` `simd` feature. The dependency is private, so
  public APIs are unchanged. `j2k-jpeg` keeps its private exact-AVX2 token on
  x86-64 because `fearless_simd::Avx2` still represents the broader x86-64-v3
  feature set.

## [0.11.1] - 2026-09-12

- Adds full-image and batched Metal reconstruction for sampled RGB components without MCT, including odd dimensions and 4:2:2/4:2:0 sampling.
- Adds native component-grid planning while retaining the existing full-resolution direct-plan contract.
- Isolates synchronous cache-build test counters from unrelated test threads.

## [0.11.0] - 2026-09-06

- Batches baseline JPEG 4:2:0 and 4:2:2 Metal texture decode through reusable component planes, with bounded scratch leases held through status consumption.
- Keeps lossy HT Metal encode payloads resident through packetization, reuses allocation capacity, and shares immutable pipelines per device.
- Adds the separately published `j2k-mpsgraph-support` ownership boundary for codec-independent graph submission.
- Adds lossy HT quality-factor options, reusable HT block-encoding workspaces, prepared region decoding, and candidate-set accelerator integration.
- Corrects the Windows OpenHTJ2K reference runtime configuration used by release validation.

- Corrects TIFF `JPEGTables` assembly for abbreviated JPEG tiles, including
  SOI/EOI-bearing tiles with external DQT/DHT definitions, missing tile or
  length-delimited `JPEGTables` EOI markers, and TIFF-defined DRI/DAC handling.
- Adds progressive, general lossless, and DICOM lossless-SV1 JPEG passthrough
  syntax classification plus ordered APP2 ICC profile extraction, insertion,
  and replacement helpers that reject malformed input.
- Hardens `j2k-mpsgraph` completed-buffer handoff with same-device validation,
  adds automated F32 graph parity against a higher-precision CPU oracle, and
  makes the direct-handoff benchmark paths use the same prepared input, warm
  up, interleave, and check correctness before timing. Its example now builds
  its own graph and passes it to `MpsGraphProgram::new`. Breaking: removes the repository-specific
  identity, RGB8 reference-graph, and CPU-oracle convenience APIs from the
  experimental production surface; equivalent fixtures remain dev-only.

## [0.10.0] - 2026-08-25

- Adds the experimental `j2k-mpsgraph` crate for Apple Silicon macOS 11+.
  Completed resident batches and pipelined direct decode feed static rank-four
  MPSGraph programs without decoded-pixel readback/re-upload; blocking,
  nonblocking, reference-oracle, benchmark, package, and release validation
  paths are included. It is not zero-copy and no speedup has been measured.

- Breaking: narrows `j2k-cuda-runtime` to codec-neutral CUDA allocation,
  launch, completion, and diagnostics. CUDA JPEG 2000, JPEG, and transcode
  ownership now lives in the published implementation crates
  `j2k-cuda-j2k-engine`, `j2k-cuda-jpeg-engine`, and
  `j2k-cuda-transcode-engine`; callers of
  `j2k_cuda_runtime::transcode_kernels_built` must use
  `j2k_cuda_transcode_engine::transcode_kernels_built`.
- Breaking: generalizes
  `j2k_transcode_metal::resident_codestream_buffer_from_metal_encoded_j2k`
  from `&j2k_metal::MetalEncodedJ2k` to any `&impl DeviceCodestream`. Existing
  `MetalEncodedJ2k` calls continue to type-check, while custom device
  codestreams implement `j2k_core::DeviceCodestream`; this removes the
  transcode adapter's dependency on the full public Metal adapter.
- Moves stable codec contracts, prepared-plan geometry, capability rejection,
  host allocation phases, packing, and encode geometry into single typed
  owners. Native produces typed plans consumed by CPU, CUDA, and Metal; the
  adapters no longer downcast erased plans or duplicate resource policy.
- Decomposes the native color, Metal JPEG/JPEG 2000, CUDA, and transcode
  orchestration roots by pipeline stage, with repository checks for dependency
  direction, unsafe inventory, clone limits, benchmark registration, and
  generated routing tables.
- Adds bounded HTJ2K lossy candidate generation with two consecutive HT sets,
  exact cleanup/SigProp/MagRef byte boundaries, per-pass distortion scoring,
  and dependency-aware slope allocation. CUDA and Metal can produce the exact
  candidate sets on device; CUDA entropy writing remains serial within each
  code block. Each selected HT set is emitted atomically in the layer of its
  final selected pass so external decoders retain an unambiguous cleanup
  boundary.
- Makes bounded classic JPEG 2000 and HTJ2K row decode through 24-bit component
  precision parse the tile graph once per row operation and reuse it across
  stripes while retaining the parsed metadata in the aggregate allocation
  baseline. Higher-precision exact-integer rows keep their compatibility path.
- Promotes measured staged CUDA JPEG encoding and adaptive CUDA JPEG checkpoint
  launch geometry. The staged encoder improves the representative 512 x 512
  batch-8 cells by about 95%, while the checkpoint policy uses one-thread
  launches below 128 checkpoints and 128-thread blocks at or above that
  threshold.
- Records and removes rejected Metal and CUDA candidates whose product
  intervals did not qualify, including terminal IDWT/store specialization,
  wide-axis and FDWT staging, RGB input fusion, packetization, terminal
  column/quantization fusion, and JPEG coefficient/IDCT defusion. No rejected
  selector, fallback, or benchmark-only production branch remains.
- Repairs ties-to-even irreversible color rounding in Metal and CUDA, the JPEG
  restart-marker bit-reader boundary, CUDA restart-coded capability routing,
  clean packaged CUDA-Oxide sibling resolution, and lane-specific changed-line
  coverage orchestration.

- Refactors the CPU JPEG AVX2 and NEON paths around unforgeable capability
  tokens and safe kernel entry points. SIMD dispatch, IDCT/color arithmetic,
  tests, and benchmark wrappers no longer require unsafe calls; the remaining
  raw loads/stores and exact-AVX2 bridge are isolated behind fixed-size safe
  interfaces. Ten production SIMD unsafe blocks remain, with no `unsafe fn`,
  under an AST-enforced 24-block cap.
- Uses `fearless_simd 0.7.0` for AArch64 NEON kernels while preserving the
  existing AVX2-only x86 acceleration envelope with a private exact-AVX2 token;
  `fearless_simd::Avx2` is intentionally not used because its 0.7 contract is
  the broader x86-64-v3 feature set.
- Adds deterministic CPU JPEG decode and expanded SIMD microbenchmarks covering
  grayscale, 4:4:4, 4:2:2, full/odd/cropped 4:2:0, row streaming, specialized
  IDCT, tail widths, and unaligned source rows.

## [0.9.0] - 2026-08-10

- Breaking: all expert Metal APIs in `j2k-metal-support`, `j2k-metal`,
  `j2k-jpeg-metal`, and `j2k-transcode-metal` now use objc2 ownership directly:
  owned objects are `Retained<ProtocolObject<dyn MTL...>>`, borrowed objects are
  `&ProtocolObject<dyn MTL...>`, and Metal value/descriptor types come from
  `objc2-metal`. Callers must replace `metal::Device`, `*Ref`, `MTLSize`,
  texture, queue, command-buffer, and buffer spellings with their objc2
  equivalents; there is no `metal-rs` compatibility feature or adapter.
- Replaces the workspace's deprecated `metal-rs 0.33` dependency with the
  pinned `objc2 0.6.4`, `objc2-foundation 0.3.2`, and `objc2-metal 0.3.2`
  stack. This removes the transitive `block 0.1.6` dependency, its local patch,
  and all associated lint, coverage, packaging, and release exceptions while
  preserving the existing Metal kernels and codec behavior.
- Centralizes Metal object creation, retained command submission, checked
  buffer access, shared-event creation, pipeline loading, and resident-image
  ownership in `j2k-metal-support`. Typed objc2 APIs now cover queue/resource
  identity, GPU timing, capture, textures, and event ordering; the remaining
  pointer and immediate-byte bindings have explicit safety proofs.
- Shader compilation failures retain their typed Metal diagnostics. The objc2
  binding does not expose warning-only `NSError` values returned alongside a
  successful library, so successful compilations no longer log those warnings.
- Removes `checked_texture_descriptor` and the unreachable raw-message-send
  `MetalSupportError` variants. Expert callers construct
  `MTLTextureDescriptor` directly and handle the remaining typed allocation
  and availability errors.

- Extends the shared ISO/IEC 15444-4:2024 / ITU-T T.803 v3 framework with
  HTJ2K Part 15 development results. The CPU decoder passes all 160
  selected Part 1 and Part 15 cases with zero skips on macOS arm64, Linux
  x86-64, and native Windows x86-64/MSVC, including DS1-HM Cclass-1h at MMAGB
  15, Cclass-1HFh at MMAGB 20, and Annex G JPH at MMAGB 15.
- Extends the CUDA and Metal reports to the same Part 15 points. Of the 160
  Part 1/Part 15 cases, 81 use the GPU for some stages and 79 run on the CPU;
  none runs entirely on the GPU. Dispatch counters record where each stage ran.
  Part 15 results for a clean release commit were still to come.
- Fixes HTJ2K decoder defects in irreversible midpoint reconstruction, ROI,
  packet/header parsing, HT cleanup/refinement, progression, tile parts, and
  JPH validation without vector-specific exceptions or relaxed tolerances.
- Versions the Annex D/F encoder ICS matrix for classic and HT modes. CPU passes
  55/56 cases through the pinned T.804 OpenJPEG decoder; OpenJPEG 2.5.3 rejects
  HT+RGN, so the mandatory 56th capability case passes through the independently
  pinned OpenHTJ2K decoder instead, as an interoperability check.
  The CUDA and Metal matrices pass 35/35 through OpenJPEG. Encoder results are
  informative, not formal decoder conformance.
- Adds HTJ2K/JPH routing workloads and routes a cell to the GPU only when it
  gives identical output, at least 10% median improvement, and non-overlapping
  Criterion 95% confidence intervals. The production Metal HTJ2K host-output route now uses
  Metal coefficient preparation and HT Tier-1 with CPU packetization for the
  verified RGB8 1,024 x 1,024 and Gray8/RGB8 2,048 x 2,048 cells; measured
  smaller cells stay on CPU. Explicit CUDA and Metal requests still never fall
  back to the CPU.

## [0.8.1] - 2026-08-06

- Adds ISO/IEC 15444-4:2024 / ITU-T T.803 v3 decoder conformance results for
  this release. The CPU decoder is Profile-1 Cclass-1 compliant, Profile-1 Cclass-1HF compliant,
  and Annex G JP2 reader compliant across all 90 selected cases with zero skips
  on macOS arm64, Linux x86-64, and Windows x86-64.
- Publishes separate CUDA and Metal results for the same codestream classes. Of
  the 90 cases, 48 use the GPU for some stages and 42 run on the CPU; none runs
  entirely on the GPU. The reports show where parsing, Tier-1, dequantization,
  IDWT, MCT, color/output, and transfers ran for each case.
- Adds exact codestream-resolution decoding through
  `decode_native_components_at_reduction` and explicit Annex G normalization
  through `decode_srgb8`, including Gray/RGB/RGBA layouts, component mapping,
  palettes, CDEF ordering, subsampling, enumerated color spaces, and restricted
  ICC conversion.
- Fixes Part 1 decoder conformance defects in irreversible midpoint
  reconstruction, ROI handling, packet/header parsing, progression and
  tile-part handling, component transforms, and JP2 validation without
  vector-specific exceptions or relaxed tolerances.
- Adds deterministic Annex D/F encoder ICS matrices. The CPU matrix passes
  28/28 cases and the CUDA and Metal matrices pass 25/25 through the pinned
  T.804 OpenJPEG decoder. Encoder results are informative, not formal decoder
  conformance.
- Independently verifies `p0_13.j2k` before harness normalization: the
  production decoder and OpenJPEG match all 257 native components exactly.
- `Auto` routes a cell to the GPU only when benchmarks show identical output, at
  least 10% median improvement, and non-overlapping Criterion 95% confidence
  intervals. Explicit CUDA and Metal requests still never fall back to the CPU.
- Makes tag publication verify all three CPU reports plus the CUDA and Metal
  reports for the exact release SHA, and rotates the public API compatibility
  baseline to the published `v0.8.0` release.

## [0.8.0] - 2026-07-29

- Breaking: decoding is strict by default in `j2k` and `j2k-native`. Callers
  that intentionally accept the documented JP2/JPH optional-metadata
  recoveries must select `DecodeSettings::lenient()` per job; raw codestream,
  entropy, bounds, overflow, allocation, and resource-safety checks remain
  strict in both modes.
- Breaking: lenient decoding now warns only when a permitted recovery was
  actually used. `J2kDecodeWarning::LenientDecodeMode` is renamed to
  `J2kDecodeWarning::LenientMetadataRecovery`. New
  `J2kView::parse_with_settings` and `J2kDecoder::new_with_settings`
  constructors keep this policy local to one input.
- Adds `CudaLosslessEncoder`, a reusable, exclusively borrowed lossless-encode
  façade with explicit `Auto`, CPU-only, and strict-CUDA routing. Its opaque
  result reports requested and actual execution plus any unavailable-device or
  incomplete-route fallback; execution errors are returned without a hidden
  full CPU retry, and the encoder can be reused after failure.
- Routes 25–38-bit `Auto` lossless encode requests to the supported CPU path
  while preserving `RequireDevice` failure semantics.
- Defines stable, experimental, implementation, and binary API tiers in the
  ordered release manifest, and makes the `0.7.5`→`0.8.0` semver check fail
  unless every source and behavior break is listed. A later `0.8.x`
  candidate must use a published `v0.8.0` baseline.
- Fixes the published `j2k-ml 0.7.5` `cuda` and `metal` clean-consumer
  failures. The adapters now use accelerator codec decode, explicit
  decoded-pixel readback, and ordinary Burn tensor upload through released
  dependency APIs.
- Adds `CudaUploadBurnDecoder` and `MetalUploadBurnDecoder` as explicit aliases
  for staged host-memory upload behavior while preserving `CudaBurnDecoder`,
  `MetalBurnDecoder`, and their inherent method paths unchanged.
- Removes the private CubeCL and wgpu patch sources from the release graph.
  This is not a direct-destination or zero-copy accelerator contract.

## [0.7.5] - 2026-07-22

- Known release defect: the published `j2k-ml 0.7.5` `cuda` and `metal`
  features depend on accelerator interop methods that are not present in the
  registry CubeCL and wgpu releases selected by clean consumers. The `cpu`
  feature is unaffected. Version `0.8.0` replaces those routes with explicitly
  staged accelerator decode, decoded-pixel readback, and ordinary Burn tensor
  upload using released public APIs.
- Compatibility exception: this release is an explicitly source-incompatible
  `0.7.x` patch against the published `0.7.3` API. The API diff and migration
  notes cover every removed item.
- Adds persistent, high-throughput owned batch decoding for JPEG 2000 and
  HTJ2K, including prepared-input reuse, homogeneous exact-integer output
  groups, CPU decode, and resident/external-destination CUDA and Metal paths.
- Publishes `j2k-ml`, the thin Burn 0.21 adapter for ordinary rank-4 `U8`,
  `U16`, and `I16` tensors. The codec remains responsible for parsing,
  grouping, and decoding; the adapter owns only tensor allocation and guarded
  CPU/CUDA/Metal interop.
- Fixes Metal stacked multi-tile batch output and removes redundant same-queue
  synchronization while preserving strict completion and status validation.
- Removes the pass-through `j2k_core::DecoderContext<C>` type and its
  `j2k::DecoderContext` re-export. Construct and pass the concrete context
  directly: `J2kContext::new()` (or `Default`) for `j2k`, and
  `j2k_jpeg::DecoderContext::new()` (or `Default`) for JPEG. Batch trait
  implementations now receive `&mut Self::Context`; call `CodecContext::clear`
  and `CodecContext::cache_stats` on that concrete value.
- Removes `try_collect_ordered_batch_results`. Migrate to
  `try_collect_ordered_batch_results_with_limits`, passing the previous
  retained-byte and maximum-retained-byte pair for both its aggregate and
  collection limit pairs.
- Removes the five forwarding accessors on
  `J2kResidentHtj2kTileEncodeJob`. Read the validated input directly through
  `job.input.width()`, `height()`, `num_components()`, `bit_depth()`, and
  `signed()`.
- Internal: removes pass-through decode, encode, allocation, crop, recode,
  coverage-normalization, fixture, cache-key, progression, and GPU-routing
  layers while retaining codec-specific scratch-pool types.

## [0.7.3] - 2026-07-15

- Fixes invalid TLM `Stlm` descriptors in CPU and Metal codestream writers so
  HTJ2K RPCL output is accepted by conforming external decoders.

## [0.7.2] - 2026-07-15

- Fixes strict CUDA Oxide builds from crates.io by resolving the packaged
  `j2k-codec-math` source through Cargo build metadata instead of assuming the
  unpublished workspace directory layout.

## [0.7.1] - 2026-07-14

- Adds CUDA classic JPEG 2000 Tier-1 code-block decoding with shared host/device
  MQ and context tables, batched resident submission, bounded host accounting,
  and explicit CUDA ABI and unsafe-boundary inventories.
- Adds an opaque, zero-copy Metal resident-image contract with checked layout
  and device identity, submission-owned input and output lifetimes, safe
  decoder-to-encoder handoff, and explicit unsafe raw-buffer interop.
- Improves Metal decode residency with dynamic pool ceilings and enforced
  inflight limits, fixes irreversible inverse-MCT addends, fuses native inverse
  MCT output handling, and uses a staged irreversible 9/7 IDWT implementation.
- Adds hardened CUDA and Metal profiling/capture controls and documents their
  benchmark-only environment variables.
- Adds the unpublished experimental `j2k-ml` workspace crate with CPU and Metal
  validation paths and an optional CUDA-Oxide kernel route, including checked
  adoption and stream-order completion for CubeCL memory-pool allocations.

## [0.7.0] - 2026-07-12

Summary. The full notes, with the reason and the replacement for every
change, are in the
[changelog as tagged at v0.14.0](https://github.com/frames-sg/j2k/blob/v0.14.0/CHANGELOG.md#070---2026-07-12).

Version `0.7.0` removes parts of the `0.6.2` API and is not source compatible
with `0.6.x`.

### Breaking API Changes

Errors:

- Fallible `J2kEncodeStageAccelerator` methods return
  `J2kEncodeStageResult<T>`, and `j2k_native::EncodeError::Accelerator`
  carries `source: J2kEncodeStageError`. Return a typed error after accepting
  work; `Ok(None)` or `Ok(false)` only declines it. `EncodeError`,
  `ResidentHtj2kEncodeError`, and `J2kError` are no longer `Clone` or `Copy`.
- Native encode entry points return `EncodeResult<_>` instead of
  `Result<_, &'static str>`, including the scalar code-block helpers and
  `collect_ht_cleanup_encode_distribution`.
  `packet_math::ht_segment_lengths` returns `HtSegmentLengthError`.
- `j2k_native::DecodeError` adds `AllocationTooLarge` and
  `HostAllocationFailed`. `J2kError` adds `NativeEncode`, `NativeValidation`,
  and `NativeResidentEncode`.
- `j2k::TileBatchError` is `j2k_core::BatchDecodeError<J2kError>`, with
  `Tile { index, source }` and `Infrastructure(BatchInfrastructureError)`.
- `From<String>` and `From<&str>` are removed for `j2k::BackendError`; use
  `BackendError::new(kind, message)`.
- `TranscodeStageError::Backend` is `Backend { backend, operation, source }`,
  with new `DeviceMemoryCapExceeded` and `DeviceAllocationFailed` variants,
  and no `Clone` or equality. `DctToWaveletStageAccelerator` methods return
  it instead of `&'static str`, and `From<&'static str>` is removed.
- `j2k-transcode` DCT transform functions return `DctTransformError`.
  `JpegToHtj2kError::Grid` and `Grid97` become `Dct53` and `Dct97`; `Encode`
  carries `Htj2kEncodeError`; `Metrics` carries `MetricsError`; and
  `InternalInvariant` is new. `validate_htj2k97_codeblock_options` returns
  `Htj2k97CodeBlockOptionsError`.
- `CudaTranscodeError` is no longer `Copy` and adds host-cap,
  host-allocation, and runtime variants.
- `j2k_tilecodec::TileCodecError` is non-exhaustive: `Backend(String)` is
  removed, `Io { context, source }` is added, and `Malformed` carries a
  `std::io::Error`.
- `JpegEncodeError` adds `InvalidDctImage`, `MemoryCapExceeded`, and
  `HostAllocationFailed`, and drops `Internal(String)` for
  `InternalInvariant { reason }`. `JpegError` adds `HostAllocationFailed`.
- `j2k_metal::Error` adds `MetalSupport`, `PreparedPlanCacheAllocation`,
  `PreparedPlanCacheInvariant`, `BatchInfrastructure`, and
  `MetalStateInvariant`. `j2k_jpeg_metal::Error` adds `MetalSupport`.
- `Error::CudaRuntime` in `j2k-cuda` and `j2k-jpeg-cuda` carries
  `source: CudaError` instead of a message string.
- `J2kResidentEncodeInput::new` returns `J2kResidentEncodeInputError`.

Ownership:

- `Clone` is removed from large owners that can approach the allocation cap.
  Share them with `Arc`. This covers `PreparedJpeg` (which has `try_clone`),
  `EncodedJpeg`, `RestartIndex`, `JpegDctImage`, `JpegDctComponent`, the
  decoded-output, JP2 metadata, recode, coefficient, encoded code-block, and
  precomputed-coefficient types of `j2k`, `j2k-native`, `j2k-types`, and
  `j2k-transcode`, and `ProfileSummary`.

Metal:

- `Surface::as_bytes` in `j2k-metal` and `j2k-jpeg-metal` returns
  `Result<Cow<'_, [u8]>, Error>`.
- `MetalEncodedJ2k::codestream_bytes` returns `Result<Vec<u8>, _>`. Its
  resource fields are private, and raw construction and buffer handoff are
  `unsafe`.
- Borrowed Metal `Buffer` and `Texture` accessors and the external input-tile
  constructors (`MetalLosslessEncodeTile::from_buffer`,
  `JpegBaselineMetalEncodeTile::new`) are `unsafe`.
- The 26 `j2k-metal` lossless encode and submit functions become
  `submit_lossless_batch` and `submit_lossless_batch_to_metal`, which take a
  `MetalLosslessEncodeBatchRequest`. `SubmittedJ2kLosslessMetalEncode` is
  removed.
- The 24 `j2k-jpeg-metal` `Codec::decode_rgb8_*_with_session` batch functions
  become `decode_rgb8_batch_into_buffer_with_session` and
  `decode_rgb8_batch_into_textures_with_session`, which take an
  `Rgb8MetalBatchRequest`. Three convenience wrappers remain.
- `j2k-metal-support` drops `MetalDeviceSession` and the borrowed
  `buffer_contents_slice` and `buffer_contents_slice_mut` wrappers.

CUDA:

- The `cuda-oxide-*` passthrough features of `j2k-cuda` and
  `j2k-transcode-cuda` are removed; enable `cuda-runtime`.
- `j2k-cuda-runtime` root exports are narrowed, and its hidden asynchronous
  enqueue methods are `unsafe`.
- `CudaJpegHuffmanTable` is opaque; construct it with
  `from_jpeg_bits_values`.
- `CudaLosslessBufferEncodeOutcome::host_outcome` is flattened into the
  outcome, and `CudaEncodedJ2k::encoded` is replaced by `metadata()`.
- GPU diagnostic profile structs are `#[non_exhaustive]`; use their
  constructors.

Removed and renamed:

- The new `j2k-types` crate defines the 49 encode-stage types that `j2k` and
  `j2k-native` share. Both re-export them at their old paths, and the
  `*_from_native` and `*_to_native` converters are removed.
- `j2k-core` drops `DecodeRequest` and its four `*_request` trait defaults,
  `BackendFailureKind`, `WarningKind`, and the hidden
  `collect_indexed_batch_results` (use `try_collect_indexed_batch_results`).
- The `j2k` facade drops its experimental adaptive-routing items, internal
  view and context items, and legacy preference aliases. Glob re-exports are
  now explicit lists, and `TileBatchDecodeDevice` and `TileBatchDecodeSubmit`
  are exported at the root.
- Implementation-detail parser, entropy, tile-helper, kernel, planning, and
  resident-handoff items are removed from `j2k-core`, `j2k-jpeg`,
  `j2k-tilecodec`, `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode-cuda`,
  `j2k-transcode-metal`, `j2k-native`, and `j2k-cuda-runtime`.
- `j2k_native::deinterleave_reference` is removed; use
  `try_deinterleave_reference`.
- `j2k-profile` constructors, summaries, parsing, and formatting return
  `ProfileResult`. `parse_profile_line` returns `Result<Option<_>, _>`.
  `MetricUnit`, `emit_gpu_route_profile`, `flush_profile_summary_to`,
  `ProfileSummary::flush_to`, `ProfileField::raw`, `duration_us_string`, and
  several hidden helpers are removed; the full notes list each replacement.
- The JPEG fast-packet adapter types are renamed from Metal-specific names to
  `JpegFast*`, and backend capability detection is renamed to compile-time
  defaults.
- JPEG sampling-factor construction is fallible, and the JPEG output-buffer
  constructors apply the host allocation cap.

### Additive API Changes

- `j2k_metal::MetalBufferPoolDiagnostics`, `MetalBufferPoolsDiagnostics`, and
  `MetalBackendSession::buffer_pool_diagnostics`.
- `j2k-codec-math::dwt::max_decomposition_levels`, shared by the native,
  facade, and CUDA encoders, and `linearized_dwt53_row`.
- `j2k-native::DecodeErrorClass` and `DecodeError::classify`.
- `j2k-cuda-runtime::CudaError::CompletionFailed` and
  `ResourceReleaseFailed`, and `j2k-cuda::Error::CudaCleanupFailed`.
- A resident-input HTJ2K encode contract shared by `j2k-types`, `j2k-native`,
  and `j2k`. CUDA-resident encode resources are bound to a `CudaSession`.
- `Reversible53CoefficientImage::encode_htj2k`.
- `j2k_jpeg::transcode::JpegDctImageError`: coefficient-domain baseline JPEG
  re-emission validates its input before encoding.

### Maturity And Fixes

- Errors keep their typed source across the encode-stage, transcode,
  tile-codec, Metal, and CUDA boundaries instead of being flattened to
  strings.
- Host and device allocations in native encode and decode, JPEG, transcode,
  and batch paths are reserved fallibly under shared caps and report typed
  errors instead of aborting.
- Multi-tile PPM decoding advances packed headers per packet instead of
  assigning one header per tile-part.
- Paletted JP2/JPH pixel fallback materializes palette mappings as component
  planes before dropping the palette boxes.
- `JpegTilePrepareOptions::duplicate_table_policy` is enforced across every
  DQT and DHT definition in `JPEGTables`.
- The public AVX2 benchmark IDCT wrapper detects AVX2 at run time and falls
  back to scalar.
- CUDA: pooled allocations are not recycled or destroyed while asynchronous
  work can still reference them, and store, IDWT, and JPEG encode and decode
  plans are validated before driver work.
- Sparse or unrepresentable packet-state indices are rejected before
  packetization allocates.
- `j2k-metal-support` owns every raw Metal buffer, texture, shader, pipeline,
  queue, command-buffer, and encoder constructor, and rejects nil results.
- J2K Metal prepared-plan caches are bounded LRUs (64 MiB host, 256 MiB
  device). JPEG Metal builds and caches only the matching fast-packet family.
- The `nvcc`/C++ kernel probe and the checked-in PTX fallback are removed:
  strict GPU validation requires every enabled CUDA-Oxide project to build.
- Test-only modules are moved out of the shipped `j2k-transcode` library.
- Supported `J2K_*` environment variables are documented, and repository
  policy checks moved into `cargo xtask repo-lint`.
- Tag publication requires a dated changelog heading, private vulnerability
  reporting, an annotated tag on the exact candidate, and a clean worktree.

## [0.6.2] - 2026-06-28

- Defines the supported formats as JPEG 2000 Part 1 codestreams, JP2
  still-image wrapping, HTJ2K Part 15 support, and JPH wrapping, with
  JPX / JPEG 2000 Part 2 extensions explicitly out of scope.
- Adds the public support matrix and `cargo xtask public-support` gate covering
  signed/high-bit samples, wide component counts, progression/tile-part cases,
  packet-marker combinations, ROI maxshift, JP2/JPH metadata, HT refinement
  decode/encode, J2K-to-HTJ2K recode, and publication gates.
- Expands the public `j2k` facade with component-plane decode/encode APIs for
  arbitrary component counts, signedness, sampling, mixed bit depths, and
  native high-bit sample access while keeping Gray/RGB/RGBA convenience paths.
- Adds JP2/JPH wrapper helpers and validation for still-image metadata,
  codestream branding, color boxes, palettes/component mapping, channel
  definitions, BPCC, and ICC preservation paths.
- Completes repo-local HTJ2K cleanup/refinement, multi-layer/rate, JPH, and
  recode tests; OpenJPH/Kakadu comparisons remain required before publishing
  results.
- Removes the cuda-oxide transcode IDCT per-thread local-memory table
  materialization, reducing the self-hosted RTX 4070 SUPER cuda-oxide
  JPEG-to-HTJ2K transcode profile from `40.813 MP/s` before the fix to
  `380.411 MP/s` in the `v0.6.2` validation run.
- Keeps cuda-oxide transcode opt-in and records the CUDA C vs cuda-oxide
  benchmark results in the benchmark documentation.
- Adds `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53` to the
  CLI as the first conservative JPEG-to-HTJ2K smoke-test command.
- Refreshes adoption-facing docs with a shorter quickstart, support matrix, and
  current `0.6.x` security/environment-variable line.

## [0.6.1] - 2026-06-22

- Refreshes public package metadata, docs.rs landing text, and README search
  signals for the `j2k` rename, Rust programming language discovery, and
  CUDA/Metal GPU JPEG 2000 / HTJ2K queries.

## [0.6.0] - 2026-06-20

- Stages the `j2k` facade release.
- Keeps CPU decode as the portable correctness baseline.
- Treats Runtime backend selection defaults to `Auto` as the public backend
  policy.
- Adds resident Metal and CUDA device memory surfaces for supported adapter
  paths through cuda-runtime integration.
- Uses J2K-owned CUDA kernels for supported CUDA codec stages.
- NVIDIA performance numbers must come from recorded benchmark runs.
- Consolidates shared J2K encode-stage, CUDA submit, Metal runtime, tilecodec,
  JPEG output, and test-support helpers.
