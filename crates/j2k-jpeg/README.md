# j2k-jpeg

Pure-Rust JPEG inspect, decode, and encode crate for pathology, medical
imaging, transcode, and general codec pipelines in the j2k workspace.

Only `x86_64` and `aarch64` targets are supported; other architectures fail at
compile time.

The CPU decoder is the reference implementation. Unsupported JPEG types return
an error. The CPU encoders are the reference for the Metal and CUDA adapter
crates' GPU baseline and lossless encode.

When testing, avoid `--all-features`: it enables `scalar-only`, which forces
the portable backend. `cargo xtask test` runs the SIMD and `scalar-only`
configurations separately, and `cargo xtask j2k-bench-signoff` runs the
exhaustive scaled-region sweep.

Pathology-oriented integrations can use `prepare_tiff_jpeg_tile` to normalize
TIFF `JPEGTables` plus abbreviated strip/tile payloads, including zero-SOF
dimension repair from container metadata. `extract_icc_profile`,
`insert_icc_profile`, and `set_icc_profile` provide ordered APP2 ICC assembly
and replacement; color management is left to the caller.

Use this crate directly for JPEG input; use `j2k` for JPEG 2000 / HTJ2K and
`j2k-transcode` for JPEG-to-HTJ2K coefficient-domain transcode paths.

## Lossless JPEG crops

The `transcode` module is experimental and hidden from the rendered docs.

`transcode::lossless_crop(bytes, rect, LosslessCropOptions::default())` copies
quantized DCT coefficients into a standalone baseline JPEG, preserving the
source's JFIF/Adobe color signalling, component IDs, sampling, and quantization
tables. The output begins at the MCU corner above-left of `rect` and ends at
its requested far edges, clamped to the source. The returned `region` locates
the requested pixels inside that output.

For several windows from one source, use `transcode::extract_dct_windows` with
`DctExtractOptions::quantized_only()`. Baseline extraction traverses the scan
once and retains only the requested blocks, using restart markers to skip to
the first window. `estimate_dct_windows` estimates the extraction working set
before entropy decoding; `LosslessCropOptions::max_working_bytes` limits crop
admission. Progressive and multi-scan inputs accumulate whole-image
coefficients, which the estimate includes.

## Dense owned batches

`JpegBatchDecoder::new(JpegBatchDecodeOptions::default())` decodes complete
8-bit JPEG inputs into owned dense RGB8 groups; set the options' `layout` (a
`j2k_core::BatchLayout`) and `workers` to change the defaults. Submit borrowed
`JpegBatchInput::new(bytes).with_options(options)` values and keep compressed
bytes alive until decode returns. Input fields are private; read them through
`bytes()` and `options()`. Outputs own their storage independently, and errors
use `j2k_core::TileBatchError<JpegError>`.

Group order is unspecified. NCHW produces `[N, 3, H, W]`; NHWC produces
`[N, H, W, 3]`. Source indices, warnings and recovery flags follow successful
rows. Unsupported, oversized and implausible images fail by index before output
allocation. Valid groups are split to fit the shared output and worker budget.
Allocation and scheduling failures fail the whole call.

`JpegBatchSession` is the lower-level interface for caller-owned tile buffers.
Its jobs use the same central DecodeOptions integrity policy. Read dimensions
and layout before consuming a dense group with into_parts, which returns
indices, warnings, recovery flags and samples.

## Coverage

The decoder targets libjpeg-turbo's decoded output, byte for byte:

- Baseline, extended, and progressive DCT JPEG at 8 and 12 bits, including
  sequential frames whose components are split across several scans.
- Grayscale, YCbCr, RGB, and Adobe CMYK/YCCK at any sampling whose factors
  divide the maximum factors.
- Lossless (SOF3) JPEG at every precision from 2 to 16 bits: predictors 1–7,
  point transforms, restart intervals, and split scans. Subsampled components
  are replicated, as libjpeg-turbo's lossless upsampler does. Samples keep
  their native range: 8-bit output up to 8-bit precision, 16-bit above.
- Color spaces chosen as libjpeg-turbo chooses them: a JFIF marker means
  YCbCr, then the Adobe transform decides, then component IDs. Unmarked
  three-component lossless frames are RGB, as DICOM encoders write them.

The committed conformance matrices pin these against libjpeg-turbo 3.1.4.1
(`djpeg`) and, for DICOM codestreams, against DCMTK 3.7 (`dcmdjpeg`).
Lossless YCbCr has no libjpeg-turbo reference, because libjpeg-turbo does not
color-convert in lossless mode. j2k converts it with chroma centered on
2^(P-1).

Unsupported: arithmetic coding, hierarchical JPEG, four-component lossless
JPEG, and sampling whose factors do not divide the maximum factors.

## Encoding

- `encode_jpeg_baseline`: 8-bit baseline (SOF0) grayscale or YCbCr 4:4:4,
  4:2:2, or 4:2:0 from `Gray8`/`Rgb8`, with the standard Huffman tables.
- `encode_jpeg_baseline_optimized`: the same frame with Huffman tables built
  from the image, as `cjpeg -optimize` writes it. Pixels are identical to
  `encode_jpeg_baseline`; the tables and entropy-coded data match
  `jpegtran -optimize` byte for byte.
- `encode_jpeg_extended12`: 12-bit extended sequential (SOF1), DICOM's 12-bit
  lossy transfer syntax, from 12-bit grayscale or RGB samples, with optimized
  tables (standard tables cannot code 12-bit coefficients).
- `encode_jpeg_lossless`: lossless (SOF3) at 2 to 16 bits from `Gray8`,
  `Gray16`, `Rgb8`, or `Rgb16`, with predictors 1–7 (1 is DICOM's SV1), point
  transforms, and restart intervals in rows. Output matches libjpeg-turbo's
  `cjpeg -lossless` byte for byte, and DCMTK's `dcmdjpeg` decodes it exactly.

Every encoder writes the JPEG codestream only; wrapping it in a DICOM file
(encapsulated Pixel Data) is left to the caller.

## Scaled decoding

Sequential and progressive DCT JPEGs use reduced IDCTs at 1/2, 1/4, and 1/8
scale. Component-specific IDCT sizes and chroma interpolation follow
libjpeg-turbo, including subsampled color, small chroma planes, and cropped
regions. The committed libjpeg-turbo 3.1.4.1 reference matrix covers these paths,
restart intervals, progressive and multi-scan sequential coding, 12-bit
inputs, and CMYK/YCCK. Lossless JPEG has no DCT scaling; reduced lossless
output point-samples the full-size image.

Unsupported sampling layouts return `NotImplemented` rather than being
converted.

## Strict pixel integrity

`DecodeOptions` is strict by default: input that needs recovery, or whose
completion cannot be verified, returns `JpegError::RecoveryRequired`.
`with_strict(false)` or `set_strict(false)` permits the supported recoveries.

- A strict success has all the required image data, with no damage
  concealment or substituted data.
- Fatal entropy, restart, and marker errors are errors under either policy.
  Optional metadata that does not affect samples is a warning.
- A missing EOI is accepted only when completeness is verified: a progressive
  file without one needs every coefficient band of every component at its
  final approximation. A file that ends with EOI may stop its progressive
  script early.
- Lenient decoding never reduces precision.
- Coefficient extraction and transcode use the same policy.

A reserved Adobe APP14 transform byte leaves the color interpretation of a
three- or four-component frame unresolved. Strict decode refuses it with
`JpegError::AmbiguousColorTransform`. Lenient decode reads the samples as
libjpeg-turbo does, YCbCr for three components and YCCK for four, keeps the
`AdobeApp14Ambiguous` warning and reports the image as recovered. For three
components, `ColorTransform::ForceRgb` or `ColorTransform::ForceYCbCr` resolves
the interpretation under either policy, without recovery. Four-component
frames have no hint, and grayscale frames are unaffected.

`DecodeOutcome::recovered`, `DecodedTile::recovered`, and
`JpegBatchGroup::recovered()` identify lenient recovery. Cosmetic warnings leave
them false. A recovered image is returned once, without an indexed error.

## Third-party code

Parts of this crate are derived from libjpeg-turbo and are covered by its
licenses: see `LICENSE.libjpeg-turbo` and `README.ijg` in this crate, and
[`NOTICES.md`](https://github.com/frames-sg/j2k/blob/main/NOTICES.md). This
software is based in part on the work of the Independent JPEG Group.

## Links

- Project website: <https://frames-sg.com/rust-jpeg-codec/>
- API docs: <https://docs.rs/j2k-jpeg>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
