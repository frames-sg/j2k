# j2k-jpeg-metal

Metal adapter for J2K JPEG decode, baseline encode, and lossless encode on
macOS.

Explicit Metal requests decode every JPEG the CPU decoder supports into Metal
memory and return an error for anything else.

Since version 0.9, the session, buffer, and texture APIs take retained or
borrowed `objc2-metal` protocol objects. The older `metal-rs` types are not
accepted.

## JPEG Decode Scope

Explicit `BackendRequest::Metal` decode accepts every request the portable
CPU decoder accepts and returns the same bytes in Metal-resident storage
(`SurfaceResidency::MetalResidentDecode`). Requests CPU decode rejects return
`UnsupportedMetalRequest` before any Metal work; there is no silent CPU
fallback.

| JPEG class | Metal route |
|---|---|
| Baseline/extended 8-bit YCbCr 4:2:0 or 4:2:2 more than 4 px wide, and YCbCr or RGB 4:4:4 | Fast-packet kernels: entropy decode, IDCT, and color on Metal |
| Grayscale; 4:4:0, 4:1:1, 4:1:0, 1x4, and other whole-ratio sampling; 4:2:0/4:2:2 at most 4 px wide; subsampled APP14 RGB; CMYK and YCCK | Hybrid: CPU entropy decode, Metal IDCT, upsampling, color, and packing |
| Progressive 8-bit (CMYK and YCCK included); extended and progressive 12-bit, including four-component layouts; multi-scan sequential (components split across scans) at 8 and 12 bits | Hybrid |
| Lossless (SOF3) at any precision from 2 to 16 bits: grayscale, RGB and YCbCr at any whole-ratio sampling and size; predictors 1–7; point transforms; restart intervals; components split across scans with their own predictors | Hybrid: CPU difference decode, Metal predictor reconstruction, replication, color, and packing |

Output formats follow CPU decode: `Gray8`, `Rgb8`, or `Rgba8` for 8-bit JPEG;
native-range (0–4095) `Rgb16` or `Rgba16`, plus `Gray16` for grayscale, for
12-bit JPEG; and for lossless JPEG the 8-bit formats up to 8-bit precision and
the 16-bit formats above it, holding native-range samples. Color spaces follow
libjpeg-turbo: JFIF means YCbCr, then the Adobe transform decides, and
unmarked lossless color is RGB, as DICOM writes it.
Full, region, scaled (1/2, 1/4, 1/8), and region-scaled requests use the
CPU decoder's component-specific IDCT sizes, chroma interpolation, and
lossless point sampling, so every route matches CPU output byte for byte.
Regions of 8-bit sequential images entropy-decode only the MCU rows they
need, entering the scan at the nearest restart marker as CPU region decode
does.

Hybrid decode runs at roughly CPU speed for single DCT images, because the
serial CPU entropy stage dominates, and up to 2.2× slower for lossless JPEG; see
[`docs/decode-architecture.md`](docs/decode-architecture.md). Use it when the
consumer is on the GPU or when the CPU should not spend time on IDCT and color
conversion. Neither backend decodes arithmetic-coded or hierarchical JPEG,
four-component lossless JPEG, or sampling whose factors do not divide the
maximum factors.

`BackendRequest::Auto` keeps single-image decode on the CPU. Full RGB batches
can use Metal for at least 16 compatible non-restart 4:2:0 or 4:2:2 tiles of at
least 256×256 pixels, including distinct inputs. The tiles must share dimensions,
tables, and checkpoint count. Smaller, mixed-table, restart-coded, and scaled
batches, and every JPEG class outside the fast-packet kernels, stay on the CPU.
The threshold was measured on an M4 Pro; widen it only with results from the
benchmark groups in [`docs/routing-benchmarks.md`](docs/routing-benchmarks.md).

Explicit Metal batches may mix JPEG classes. The reusable caller-owned RGB8
buffer and RGBA8 texture batch outputs need only one output shape: tiles the
batched fast-packet kernels cannot share are hybrid-decoded into their slots.

## JPEG Encode

`encode_jpeg_baseline_from_metal_buffer` (and its batch form) encodes
Metal-resident `Gray8`/`Rgb8` images as baseline JPEG entirely on the GPU.

`encode_jpeg_lossless_from_metal_buffer` encodes a Metal-resident `Gray8`,
`Gray16`, `Rgb8`, or `Rgb16` image as lossless (SOF3) JPEG with the options of
`j2k_jpeg::encode_jpeg_lossless`. Metal predicts and categorizes every sample,
sums code lengths, and packs the bit stream; the CPU builds the Huffman table
from the GPU histogram and copies the packed stream out with byte stuffing.
Output is byte-identical to CPU lossless encode. On an M4 Pro a 2048×2500
16-bit SV1 image encodes in 6.2 ms against 39.7 ms on the CPU (the
`jpeg_lossless_encode` group of `benches/encode_baseline.rs`).

## Links

- API docs: <https://docs.rs/j2k-jpeg-metal>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
