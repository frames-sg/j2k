# j2k-jpeg

Pure-Rust JPEG inspect, decode, and encode crate for pathology, medical
imaging, transcode, and general codec pipelines in the j2k workspace.

Only `x86_64` and `aarch64` targets are supported; other architectures fail at
compile time.

The CPU decoder is the reference implementation. Unsupported JPEG types return
an error. The CPU encoders are the reference for the Metal and CUDA adapter
crates' GPU baseline and lossless encode.

Pathology-oriented integrations can use `prepare_tiff_jpeg_tile` to normalize
TIFF `JPEGTables` plus abbreviated strip/tile payloads, including zero-SOF
dimension repair from container metadata. `extract_icc_profile`,
`insert_icc_profile`, and `set_icc_profile` provide ordered APP2 ICC assembly
and replacement; color management is left to the caller.

Use this crate directly for JPEG input; use `j2k` for JPEG 2000 / HTJ2K and
`j2k-transcode` for JPEG-to-HTJ2K coefficient-domain transcode paths.

## Dense owned batches

`JpegBatchDecoder::new(JpegBatchDecodeOptions::default())` decodes complete
8-bit JPEG inputs into owned dense RGB8 groups. Submit borrowed
`JpegBatchInput::new(bytes).with_options(options)` values and keep compressed
bytes alive until decode returns. Outputs own their storage independently.

Group order is unspecified. NCHW produces `[N, 3, H, W]`; NHWC produces
`[N, H, W, 3]`. Source indices, warnings and recovery flags follow successful
rows. Unsupported, oversized and implausible images fail by index before output
allocation. Valid groups are split to fit the shared output and worker budget.
Allocation or scheduling failures remain batch infrastructure errors.

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

## Links

- Project website: <https://frames-sg.com/rust-jpeg-codec/>
- API docs: <https://docs.rs/j2k-jpeg>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>

## Strict pixel integrity and dense batches

`DecodeOptions` defaults to strict. Use `with_strict(false)` or `set_strict(false)`
only to permit supported recovery or unverified completion. Strict success
requires complete required image data under the requested interpretation, with
no damage concealment or missing-data substitution. Existing fatal entropy,
restart and marker errors remain errors under either policy. Optional metadata
not affecting samples remains a warning. Missing EOI is accepted only after
completeness is verified; progressive EOF requires every coefficient band at
final approximation for every component. An actual EOI can terminate a legal
partial progressive script. Unresolved Adobe APP14 interpretation requires an
explicit color hint. Leniency never authorizes precision loss.

`DecodeOutcome::recovered`, `DecodedTile::recovered`, and
`JpegBatchGroup::recovered()` identify lenient recovery. Cosmetic warnings leave
them false. A recovered image is returned once, without an indexed error.
This default tightens acceptance of damaged progressive JPEGs, including callers
using coefficient extraction and transcode; accepted intact pixels are unchanged.

Construct `JpegBatchDecoder::new(JpegBatchDecodeOptions::default())` and submit
`JpegBatchInput::new(bytes).with_options(options)`. Input fields are private;
read them through `bytes()` and `options()`. Errors use
`j2k_core::TileBatchError<JpegError>`. Dense groups share the codec memory budget
with worker storage and may split; order is unspecified and source indices are
authoritative.

JPEG tensor output remains RGB uint8 by default, with grayscale expanded to three identical channels. Grayscale JPEG 2000 and HTJ2K remain one channel. Output metadata describes the returned tensor. A later explicit `channels="native"` option may preserve grayscale JPEG as one channel without changing existing defaults. Source component counts, when exposed, are per input row and remain aligned with source_indices.
