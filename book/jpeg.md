# JPEG

`j2k-jpeg` reads and writes regular JPEG. It is written in Rust, has no C
dependencies, and its decoder matches libjpeg-turbo's output byte for byte.
It is a separate crate from `j2k`:

```bash
cargo add j2k-jpeg
```

`j2k-jpeg` builds only for `x86_64` and `aarch64`; other targets fail at
compile time.

## Decode

```rust
use j2k_jpeg::{Decoder, PixelFormat};

let decoder = Decoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let stride = width as usize * 3;
let mut rgb = vec![0_u8; stride * height as usize];
let outcome = decoder.decode_into(&mut rgb, stride, PixelFormat::Rgb8)?;
```

`Decoder::inspect(&bytes)` reads the size, color space, sampling, frame type
(`SofKind`), bit depth, and restart interval without decoding. The output
formats depend on the source: `Gray8`, `Rgb8`, or `Rgba8` for 8-bit JPEG;
native-range (0-4095) `Gray16`, `Rgb16`, or `Rgba16` for 12-bit JPEG; and for
lossless JPEG, the 8-bit formats up to 8-bit precision and the 16-bit formats
above.

Other ways to decode:

- `decode_scaled_into` decodes at 1/2, 1/4, or 1/8 size (`Downscale`) using
  reduced IDCTs, the way libjpeg-turbo does, including its chroma
  interpolation for subsampled color. Lossless JPEG has no DCT to scale, so
  reduced lossless output point-samples the full-size image.
- `decode_region_into` and `decode_region_scaled_into` decode a rectangle.
  For 8-bit sequential images they entropy-decode only the MCU rows the
  region needs, starting at the nearest restart marker.
- `decode_rows` hands rows to a `RowSink` without a whole-image buffer.
- `decode_tiles_into`, `decode_tiles_scaled_into`, and
  `decode_tiles_region_scaled_into` decode many independent JPEG tiles in
  parallel into your buffers.

### What it decodes

- Baseline, extended, and progressive DCT JPEG at 8 and 12 bits, including
  sequential frames whose components are split across several scans.
- Grayscale, YCbCr, RGB, and Adobe CMYK and YCCK, at any sampling whose
  factors divide the largest factor.
- Lossless JPEG (SOF3) at every precision from 2 to 16 bits: predictors 1-7,
  point transforms, restart intervals, and split scans. Subsampled components
  are replicated, as libjpeg-turbo does, and samples keep their native range.

Color spaces are chosen the way libjpeg-turbo chooses them: a JFIF marker
means YCbCr, then the Adobe marker decides, then the component IDs. A
three-component lossless frame without markers is RGB, as DICOM encoders
write it. Lossless YCbCr has no libjpeg-turbo reference, because
libjpeg-turbo does not convert color in lossless mode; we convert it with
chroma centered on 2^(P-1).

Arithmetic coding, hierarchical JPEG, four-component lossless JPEG, and
sampling whose factors don't divide the largest factor return an error.

## Strict and lenient decoding

`DecodeOptions` is strict by default: input that needs recovery, or whose
completeness can't be verified, returns `JpegError::RecoveryRequired`. A
strict success has all the image data, with nothing concealed or
substituted. To allow the supported recoveries:

```rust
use j2k_jpeg::{DecodeOptions, Decoder, JpegView};

let options = DecodeOptions::default().with_strict(false);
let decoder = Decoder::from_view(JpegView::parse_with_options(&bytes, options)?)?;
```

- Fatal entropy, restart, and marker errors are errors either way. Optional
  metadata that doesn't affect samples is a warning.
- A missing EOI marker is accepted only when the image is verifiably
  complete: a progressive file without one needs every coefficient band of
  every component at its final approximation. A file that does end with EOI
  may stop its progressive script early.
- Lenient decoding never reduces precision.
- `DecodeOutcome::recovered`, `DecodedTile::recovered`, and
  `JpegBatchGroup::recovered()` say whether a recovery was used. Cosmetic
  warnings leave them false.
- Coefficient extraction and transcoding use the same policy.

A reserved Adobe APP14 transform value leaves the color interpretation of a
three- or four-component image undefined. Strict decoding returns
`JpegError::AmbiguousColorTransform`. Lenient decoding reads it as
libjpeg-turbo does (YCbCr for three components, YCCK for four), adds the
`AdobeApp14Ambiguous` warning, and marks the image recovered. For three
components, `with_color_transform(ColorTransform::ForceRgb)` or `ForceYCbCr`
settles it under either policy, without a recovery. Use that when the
container knows the photometric interpretation.

## Batch decoding into tensors

`JpegBatchDecoder` decodes complete 8-bit JPEGs into dense RGB8 groups, one
per output size:

```rust
use j2k_jpeg::{BatchLayout, JpegBatchDecodeOptions, JpegBatchDecoder, JpegBatchInput};

let mut options = JpegBatchDecodeOptions::default();
options.layout = BatchLayout::Nhwc; // [N, H, W, 3]; the default is NCHW, [N, 3, H, W]
let mut decoder = JpegBatchDecoder::new(options);

let inputs: Vec<JpegBatchInput<'_>> = tiles.iter().map(|t| JpegBatchInput::new(t)).collect();
let result = decoder.decode(&inputs)?;
for error in result.errors() {
    eprintln!("{error}");
}
for group in result.groups() {
    let (width, height) = group.dimensions();
    let rgb = group.samples();
    let inputs_in_group = group.source_indices();
}
```

`JpegBatchInput::new(bytes).with_options(...)` sets decode options per
input; the bytes must stay alive until `decode` returns. Group order is
unspecified, and `source_indices`, `warnings`, and `recovered` follow the
rows of each group. Unsupported, oversized, and implausible images fail by
index before any output is allocated, and large groups are split to fit the
output and worker budget. Allocation and scheduling failures fail the whole
call. `JpegBatchSession` is the lower-level interface for decoding into your
own tile buffers.

## Encode

```rust
use j2k_jpeg::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples, JpegSubsampling};

let options = JpegEncodeOptions {
    quality: 85,
    subsampling: JpegSubsampling::Ybr420,
    ..JpegEncodeOptions::default() // quality 90, 4:2:2
};
let jpeg = encode_jpeg_baseline(JpegSamples::Rgb8 { data: &rgb, width, height }, options)?;
std::fs::write("image.jpg", &jpeg.data)?;
```

- `encode_jpeg_baseline` writes 8-bit baseline (SOF0) grayscale or YCbCr
  4:4:4, 4:2:2, or 4:2:0 with the standard Huffman tables.
- `encode_jpeg_baseline_optimized` writes the same frame with Huffman tables
  built from the image, as `cjpeg -optimize` does. Pixels are identical to
  `encode_jpeg_baseline`, and the tables and entropy-coded data match
  `jpegtran -optimize` byte for byte.
- `encode_jpeg_extended12` writes 12-bit extended sequential (SOF1), DICOM's
  12-bit lossy transfer syntax, from 12-bit gray or RGB samples
  (`JpegSamples12`). It always uses optimized tables, since the standard
  tables can't code 12-bit coefficients.
- `encode_jpeg_lossless` writes lossless (SOF3) at 2 to 16 bits from `Gray8`,
  `Gray16`, `Rgb8`, or `Rgb16` samples, with predictors 1-7 (1 is DICOM's
  SV1), point transforms, and restart intervals in rows. The output matches
  libjpeg-turbo's `cjpeg -lossless` byte for byte, and DCMTK's `dcmdjpeg`
  decodes it exactly.

```rust
use j2k_jpeg::{encode_jpeg_lossless, JpegLosslessEncodeOptions, JpegLosslessSamples};

let jpeg = encode_jpeg_lossless(
    JpegLosslessSamples::Gray16 { data: &ct, width, height },
    JpegLosslessEncodeOptions::new(12), // 12-bit precision, predictor 1
)?;
```

The encoders write the JPEG codestream only. Wrapping it in a DICOM file
(encapsulated pixel data) is up to you.

## Lossless crops and coefficients

The `j2k_jpeg::transcode` module works on quantized DCT coefficients. It is
experimental and hidden from the API docs.

`lossless_crop` cuts a rectangle out of a JPEG without decoding and
re-encoding it, so there is no generation loss:

```rust
use j2k_jpeg::transcode::{lossless_crop, LosslessCropOptions};
use j2k_jpeg::Rect;

let area = Rect { x: 1000, y: 700, w: 512, h: 512 };
let crop = lossless_crop(&bytes, area, LosslessCropOptions::default())?;
std::fs::write("crop.jpg", &crop.data)?;
```

The output is a standalone baseline JPEG that keeps the source's JFIF or
Adobe signalling, component IDs, sampling, and quantization tables. Because
it can only cut on MCU boundaries, it starts at the MCU corner above and to
the left of the requested rectangle and ends at the rectangle's far edges,
clamped to the image. `crop.region` locates the requested pixels inside it.
`LosslessCropOptions::max_working_bytes` limits the memory a crop may use.

For several windows from one image, `extract_dct_windows` with
`DctExtractOptions::quantized_only()` reads a baseline scan once, keeps only
the requested blocks, and uses restart markers to skip to the first window.
`estimate_dct_windows` estimates the memory before entropy decoding.
Progressive and multi-scan inputs hold the whole image's coefficients, and
the estimate includes them. `extract_dct_blocks` returns every block.

## TIFF and ICC helpers

`prepare_tiff_jpeg_tile` turns a TIFF tile or strip, stored as abbreviated
JPEG plus the file's `JPEGTables`, into a complete JPEG. It can also repair
a frame header whose dimensions are zero, using the size from the container.
`extract_icc_profile`, `insert_icc_profile`, and `set_icc_profile` read and
write ICC profiles split across APP2 segments. Color management is left to
you.

## GPU JPEG

`j2k-jpeg-cuda` and `j2k-jpeg-metal` decode JPEG into GPU memory and encode
JPEG from GPU buffers. Both use `BackendRequest` as described in [GPU
decoding and encoding](gpu.md): an explicit GPU request returns an error for
anything it can't do, and never falls back to the CPU.

### CUDA

`j2k-jpeg-cuda` needs the `cuda-runtime` feature to run anything; without
it, CUDA requests return `CudaUnavailable`. See
[GPU decoding and encoding](gpu.md#cuda) for the build requirements.

- Decode: an explicit CUDA request decodes the full image to `Rgb8` for
  single-scan baseline or extended 8-bit YCbCr JPEG at 4:2:0, 4:2:2, or
  4:4:4. Region and scaled requests are errors. `Auto` never picks CUDA for
  JPEG decode; it decodes on the CPU.
- Baseline encode: `encode_jpeg_baseline_from_cuda_buffer` and its batch
  form take `Gray8` or `Rgb8` CUDA buffers and return a host `EncodedJpeg`.
- Lossless encode: `encode_jpeg_lossless_from_cuda_buffer` takes `Gray8`,
  `Gray16`, `Rgb8`, or `Rgb16` CUDA buffers and produces the same bytes as
  `encode_jpeg_lossless`.

### Metal

`j2k-jpeg-metal` runs on macOS. An explicit `BackendRequest::Metal` decode
accepts every request the CPU decoder accepts, including regions and 1/2,
1/4, and 1/8 scaling, and returns the same bytes in Metal memory. Requests
the CPU decoder rejects return `UnsupportedMetalRequest` before any GPU work.

| JPEG | How Metal decodes it |
| --- | --- |
| 8-bit baseline or extended YCbCr 4:2:0 or 4:2:2 wider than 4 pixels; YCbCr or RGB 4:4:4 | Entirely on the GPU |
| Grayscale; other whole-ratio sampling; narrow 4:2:0 or 4:2:2; subsampled Adobe RGB; CMYK and YCCK | Entropy decoding on the CPU, the rest on the GPU |
| Progressive 8-bit; extended and progressive 12-bit; multi-scan sequential | Entropy decoding on the CPU, the rest on the GPU |
| Lossless at 2 to 16 bits | Difference decoding on the CPU, prediction and the rest on the GPU |

The split routes run at about CPU speed for single DCT images, because CPU
entropy decoding dominates, and up to about twice as slow as the CPU for
lossless JPEG. Use them when the pixels are needed on the GPU, or to keep the
CPU free of IDCT and color conversion. The [Metal JPEG
decoder](../contributor-book/metal-jpeg-decoder.md) chapter of the
contributor book explains the design.

`Auto` keeps single images on the CPU. It uses Metal for full-image RGB
batches of at least 16 tiles of 4:2:0 or 4:2:2 JPEG without restart markers,
each at least 256 x 256 and sharing dimensions, tables, and checkpoint count.
Everything else under `Auto` stays on the CPU. Explicit Metal batches can
mix JPEG types; tiles that can't share the GPU-only kernels are decoded
through the split route into their slots.

`encode_jpeg_baseline_from_metal_buffer` and its batch form encode `Gray8`
or `Rgb8` Metal buffers as baseline JPEG entirely on the GPU.
`encode_jpeg_lossless_from_metal_buffer` encodes `Gray8`, `Gray16`, `Rgb8`, or
`Rgb16` Metal buffers as lossless JPEG with the options of
`encode_jpeg_lossless`; the GPU predicts and packs the bits, the CPU builds
the Huffman table, and the output is byte-identical to the CPU encoder.

Batch baseline encode, on either GPU, submits consecutive tiles of one buffer
together. Each submission is limited to 512 MiB of worst-case output, so a
large batch is encoded in chunks. A `MemoryCapExceeded` error means one tile
alone needs more than that, or the frames already encoded leave no room for
the next tile.
