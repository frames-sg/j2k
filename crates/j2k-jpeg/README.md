# j2k-jpeg

Pure-Rust JPEG inspect, decode, and encode crate for pathology, medical
imaging, transcode, and general codec pipelines in the j2k workspace.

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

- API docs: <https://docs.rs/j2k-jpeg>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
