# Serving IIIF images with j2k

This guide maps [IIIF Image API 3.0](https://iiif.io/api/image/3.0/) requests
onto j2k's Rust facade, C ABI, Python package and `opj_decompress`-compatible
CLI. It covers the image request, `info.json`, and running j2k behind
Cantaloupe.

j2k decodes JPEG 2000 and HTJ2K. It does not resize, rotate or convert
quality; an image server does those steps on the decoded pixels.

## The image request

An IIIF image request is
`{scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}`,
applied in the order region, size, rotation, quality, format (Image API 3.0
§2.1 and §4.6). With j2k:

| Parameter | Where it happens | How |
| --- | --- | --- |
| region | j2k | `full`, `square`, `x,y,w,h` and `pct:x,y,w,h` become one rectangle in image pixels, `(0, 0)` at the top-left pixel, clipped to the image (§4.1). That is the facade's source region (`Rect { x, y, w, h }`, the C ABI's `x, y, width, height`, Python's `region=(x, y, w, h)`). |
| size | j2k, then the server | Compute the target width and height from the size form (§4.2). Decode the region at the deepest reduction whose output still covers the target (below), then resize the decoded pixels to the target on the server. |
| rotation | server | Rotate (and mirror for `!n`) the resized pixels. |
| quality | server | `color` and `default` keep the decoded sRGB samples; `gray` and `bitonal` convert them. |
| format | server, or j2k-jpeg | Encode the result. `jpg` can use `j2k_jpeg::encode_jpeg_baseline` (Rust) or `encode_rgb(..., codec="jpeg")` (Python, RGB only); the C ABI has no JPEG encoder. |

The specification says a region of zero size or wholly outside the image
should get a 400 (Bad Request) response (§4.1), as should a size that needs
upscaling without `^` (§4.2).

### Choosing the reduction

Discarding `r` resolution levels halves each axis `r` times. Use the largest
`r` from 0 to the image's maximum reduction whose decoded region is still at
least the target size on both axes; resize from there.

- The maximum reduction is `J2kDecoder::max_reduction_levels` (Rust),
  `j2k_region_plan.max_reduction` (C) or `inspect(data)["max_reduction"]`
  (Python). It is the smallest decomposition count of every tile and
  component, so tile-part COD/COC markers that lower it are honoured; the main
  header's level count can be larger.
- The decoded size at `r` comes from the plan, before any pixels are decoded:
  `J2kDecoder::srgb8_region_decode_plan(region, r).output_region` (Rust),
  `j2k_decoder_plan_region` with `J2K_INTERPRETATION_SRGB8` (C, `output.width`
  and `output.height`), or `Decoder.plan_region(data, region=..., reduction=r,
  interpretation="srgb8")` (Python, `"width"` and `"height"`).
- A plan fails when the region has no samples at `r`; stop there.

The plan's output is every reduced sample whose footprint overlaps the region,
so it can include part of a sample beyond each edge. The exact window of ITU-T
T.800 B.5, `[ceil(x0 / 2^r), ceil(x1 / 2^r))` on each axis, is
`J2kReferenceGrid::reduced_area` in Rust. When the image origin is `(0, 0)`
and the request is an `info.json` tile at one of its scale factors, the two
agree and the decoded tile is exactly the IIIF tile size: the region's left
edge `n * tw * s` is a multiple of `s = 2^r`, so the decoded width is
`ceil(wr / s)`, the size the Image API's tile calculation gives.

Decode with the explicit sRGB output, which also applies palettes, ICC
profiles, sYCC and channel definitions: `decode_region_srgb8_at_reduction`
(Rust), `j2k_decoder_decode_region_into` with `J2K_INTERPRETATION_SRGB8` (C),
or `Decoder.decode_srgb8` (Python). It returns 8-bit gray, RGB or RGBA with
straight alpha.

### Rust

```rust
use j2k::{J2kDecoder, J2kSrgb8Layout, Rect};
use j2k_jpeg::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples, JpegSubsampling};

/// The reduction and 8-bit sRGB pixels for an IIIF region (image pixels,
/// already clipped) that will be resized to `target`.
fn decode_for_size(
    bytes: &[u8],
    region: Rect,
    target: (u32, u32),
) -> Result<(u8, j2k::J2kSrgb8Image), j2k::J2kError> {
    let mut decoder = J2kDecoder::new(bytes)?;
    let mut reduction = 0;
    for levels in 1..=decoder.max_reduction_levels()? {
        match decoder.srgb8_region_decode_plan(region, levels) {
            Ok(plan) if plan.output_region.w >= target.0 && plan.output_region.h >= target.1 => {
                reduction = levels;
            }
            _ => break,
        }
    }
    Ok((reduction, decoder.decode_region_srgb8_at_reduction(region, reduction)?))
}

/// Encode resized 8-bit gray or RGB pixels as a baseline JPEG.
fn jpeg(layout: J2kSrgb8Layout, data: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let (samples, subsampling) = match layout {
        J2kSrgb8Layout::Gray => (JpegSamples::Gray8 { data, width, height }, JpegSubsampling::Gray),
        J2kSrgb8Layout::Rgb => (JpegSamples::Rgb8 { data, width, height }, JpegSubsampling::Ybr420),
        _ => return None, // flatten RGBA onto a background first
    };
    let options = JpegEncodeOptions { quality: 80, subsampling, ..JpegEncodeOptions::default() };
    encode_jpeg_baseline(samples, options).ok().map(|encoded| encoded.data)
}
```

### C

```c
j2k_decoder *decoder = NULL;
j2k_operation_options request = J2K_OPERATION_OPTIONS_INIT;
j2k_region_plan plan = J2K_REGION_PLAN_INIT;
request.x = x; request.y = y; request.width = w; request.height = h;
request.interpretation = J2K_INTERPRETATION_SRGB8;
if (j2k_decoder_open(bytes, len, NULL, &decoder, &error) != J2K_OK) { /* ... */ }
if (j2k_decoder_plan_region(decoder, &request, &plan, &error) != J2K_OK) { /* ... */ }
uint32_t max_reduction = plan.max_reduction, reduction = 0;
for (uint32_t r = 1; r <= max_reduction; ++r) {
    j2k_region_plan reduced = J2K_REGION_PLAN_INIT;
    request.reduction = r;
    if (j2k_decoder_plan_region(decoder, &request, &reduced, &error) != J2K_OK ||
        reduced.output.width < target_w || reduced.output.height < target_h) break;
    reduction = r;
    plan = reduced;
}
request.reduction = reduction;
/* allocate plan.output.buffer_bytes, then: */
j2k_decoder_decode_region_into(decoder, &request, pixels, plan.output.buffer_bytes,
                               plan.output.row_bytes, NULL, &error);
```

`plan.output.channels` is 1, 3 or 4 interleaved 8-bit samples.

### Python

```python
import j2k

decoder = j2k.Decoder(layout="nhwc")
info = j2k.inspect(data)

def decode_for_size(data, region, target):
    reduction = 0
    for levels in range(1, info["max_reduction"] + 1):
        try:
            plan = decoder.plan_region(data, region=region, reduction=levels,
                                       interpretation="srgb8")
        except ValueError:  # no samples left at this reduction
            break
        if plan["width"] < target[0] or plan["height"] < target[1]:
            break
        reduction = levels
    pixels = decoder.decode_srgb8(data, region=region, reduction=reduction)
    return pixels.to_numpy()  # height x width x channels, uint8
```

`data` may be `bytes` or a read-only `mmap` of the file, which is read in
place; see the package README.

### CLI

`opj_decompress -i image.jp2 -d x0,y0,x1,y1 -r <reduction> -o region.bmp` (or
`j2k decode` with the same options) decodes a region at a reduction to BMP,
PNG, TIFF or PNM. `-d` is on the reference grid: for an image whose origin is
`(0, 0)`, that is `x, y, x + w, y + h` of the IIIF region. At reduction `r`
the CLI writes the T.800 window above. See
[`crates/j2k-cli/README.md`](../crates/j2k-cli/README.md).

## info.json

The required properties (§5.2) are `@context`
(`http://iiif.io/api/image/3/context.json`), `id`, `type` (`ImageService3`),
`protocol` (`http://iiif.io/api/image`), `profile` (`level0`, `level1` or
`level2`), `width` and `height`. j2k supplies the image facts without decoding
pixels:

| info.json | Rust | C | Python `inspect` |
| --- | --- | --- | --- |
| `width`, `height` | `J2kDecoder::inspect_reference_grid(bytes)?.image_size` | `j2k_image_info.width`, `.height` | `width`, `height` |
| `tiles[].width`, `.height` | `J2kReferenceGrid::tile_size` | `tile_width`, `tile_height` | `tile_width`, `tile_height` |
| `tiles[].scaleFactors` | `2^r` for `r` in `0..=max_reduction_levels()` | `2^r` up to `j2k_region_plan.max_reduction` | `2**r` up to `max_reduction` |
| `sizes` | the full image's plan at each `r` | the full image's plan at each `r` | `plan_region(data, reduction=r)` |

Tiles in `info.json` tell clients which regions to ask for; j2k decodes any
region, so an untiled image (one tile as large as the image) can advertise a
tile size the server chooses. `inspect` also reports the image and tile-grid
origins, the tile grid (`tiles_x`, `tiles_y`), quality layers, progression
order and code-block size; `J2kDecoder::inspect_coding_style` returns the
main-header coding style in Rust. A Python example that builds `info.json` is
in [`crates/j2k-python/README.md`](../crates/j2k-python/README.md).

## Cantaloupe

Cantaloupe's `OpenJpegProcessor` runs `opj_decompress` and reads its output,
so it can run j2k through an `opj_decompress` symlink. This section follows
the Cantaloupe source at commit `f1845927c9d642f4d522b5167bc9b7c219f6a1b6`
(`OpenJpegProcessor.java`, `CommandLocator.java`, `Key.java` and
`cantaloupe.properties.sample`).

1. Install `j2k-cli` and link it under OpenJPEG's name in a directory of its
   own:

   ```bash
   mkdir -p /opt/j2k/bin
   ln -s "$(command -v j2k)" /opt/j2k/bin/opj_decompress
   ```

2. Select the processor for JPEG 2000 sources and point it at that directory
   in `cantaloupe.properties`:

   ```properties
   processor.selection_strategy = ManualSelectionStrategy
   processor.ManualSelectionStrategy.jp2 = OpenJpegProcessor
   OpenJpegProcessor.path_to_binaries = /opt/j2k/bin
   ```

   `OpenJpegProcessor.path_to_binaries` is a directory that contains
   `opj_decompress`; a full path ending in `opj_decompress` also works. Left
   empty, Cantaloupe runs `opj_decompress` from the `PATH` (`CommandLocator`).

What Cantaloupe does with it, and what j2k does in return:

- It runs `opj_decompress -h` once and looks for a `vX.Y.Z` version; when it
  finds one below 2.2 it stops passing `-quiet`. j2k's help names no such
  version, so Cantaloupe keeps passing `-quiet`.
- Each request runs
  `opj_decompress -quiet -i <source> [-d x0,y0,x1,y1] [-r n] -o <temp>/OpenJpegProcessor-<uuid>.bmp`,
  where the output name is a symlink to `/dev/stdout`, and reads a BMP from
  the process's stdout. j2k writes only the image there; notes and errors go
  to stderr.
- `-d` comes from the IIIF region in image pixels, and `-r` from the scale,
  limited by the decomposition count Cantaloupe's own metadata reader finds.
  Cantaloupe post-processes the BMP with Java 2D (resize, rotation and the
  rest) and encodes the response itself.
- `-i` is the source path; files in Cantaloupe's source cache have no
  extension. j2k recognises its input by signature, not by name.
- When stderr contains `Unknown input file format`, Cantaloupe reports a
  source-format error; j2k prints that message for input that is not JPEG
  2000.
- Cantaloupe passes no `-threads`; j2k, like `opj_decompress`, then reads
  `OPJ_NUM_THREADS` from the environment Cantaloupe runs in, and decodes on one
  thread when it is unset.

A limit of this route, shared with OpenJPEG: `-d` is on the reference grid,
but Cantaloupe computes it in image pixels. For an image whose origin is not
`(0, 0)`, both tools read the values as reference-grid coordinates, so the
decoded area is offset from the requested one by the origin and clamped to the
image.

`GrokProcessor` runs `grk_decompress` with `-i`, `-d`, `-r` and `-o` and no
`-quiet`. j2k does not answer to that name.

## Parity with opj_decompress

The CLI was compared with OpenJPEG 2.5.4 `opj_decompress` on 80 files: 17
generated (8- to 16-bit gray and RGB, signed and unsigned, tiled and untiled,
classic and HTJ2K, 5/3 and 9/7, quality layers, ICC, palette, JP2 and JPH) and
63 in-repo fixtures, including images with nonzero origins. Every file was
decoded at every reduction from 0 to its maximum, whole and in regions inside
one tile, across a tile corner, at odd offsets and starting before the origin,
to BMP and PNM, and at full resolution also to PNG and TIFF, with identical
arguments in Cantaloupe's order.

| Outcome | Cases |
| --- | --- |
| Identical samples | 2006 |
| HTJ2K regions: OpenJPEG's region differs from its own whole-image decode; j2k's equals the window of both tools' identical whole-image decodes | 662 |
| ICC profiles: Little CMS vs j2k colour management, at most one 8-bit step | 84 |
| Both tools fail (no samples left at a reduction, unsupported precision or components) | 60 |
| One tool fails | 36 |

The 36 are differences listed in the CLI README: samples deeper than 16 bits
(j2k refuses BMP, OpenJPEG writes it), components with different sampling
(j2k refuses; OpenJPEG writes one PNM per component or the first component
to BMP), a palette image at a reduction (j2k fails, OpenJPEG writes an image),
and a 6x6 image whose coarse levels are empty, which OpenJPEG fails to decode
at reduction 1.
