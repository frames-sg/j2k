# Serving IIIF images

This page shows how to answer [IIIF Image API
3.0](https://iiif.io/api/image/3.0/) requests with J2K, from Rust, C, Python,
or the command line, and how to run J2K behind the Cantaloupe image server.

J2K decodes JPEG 2000 and HTJ2K. It doesn't resize, rotate, or change
quality; your image server does those steps on the decoded pixels.

## The image request

An IIIF image request is
`{scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}`,
applied in the order region, size, rotation, quality, format (Image API 3.0
§2.1 and §4.6). With J2K:

| Parameter | Done by | How |
| --- | --- | --- |
| region | J2K | `full`, `square`, `x,y,w,h` and `pct:x,y,w,h` all become one rectangle in image pixels, with `(0, 0)` at the top-left pixel, clipped to the image (§4.1). Pass it as the source region: `Rect { x, y, w, h }` in Rust, `x, y, width, height` in C, `region=(x, y, w, h)` in Python. |
| size | J2K, then the server | Work out the target width and height from the size form (§4.2). Decode the region at the deepest reduction whose output still covers the target ([below](#choosing-the-reduction)), then resize to the target on the server. |
| rotation | Server | Rotate the resized pixels, and mirror them for `!n`. |
| quality | Server | `color` and `default` keep the decoded sRGB samples; `gray` and `bitonal` convert them. |
| format | Server, or `j2k-jpeg` | Encode the result. For `jpg`, use `j2k_jpeg::encode_jpeg_baseline` in Rust or `j2k.encode_rgb(..., codec="jpeg")` in Python (RGB only). The C library has no JPEG encoder. |

The specification says to answer 400 (Bad Request) for a region of zero size
or wholly outside the image (§4.1), and for a size that needs upscaling
without `^` (§4.2).

### Choosing the reduction

Discarding `r` resolution levels halves each axis `r` times. Use the largest
`r`, from 0 up to the image's maximum, whose decoded region is still at least
the target size on both axes, and resize from there.

- The maximum reduction is `J2kDecoder::max_reduction_levels` in Rust,
  `j2k_region_plan.max_reduction` in C, and `inspect(data)["max_reduction"]`
  in Python. It is the smallest decomposition count of any tile and
  component, so it accounts for tile-part COD and COC markers that lower it.
  The main header's level count can be larger.
- The decoded size at `r` comes from a plan, before any pixels are decoded:
  `J2kDecoder::srgb8_region_decode_plan(region, r).output_region` in Rust,
  `j2k_decoder_plan_region` with `J2K_INTERPRETATION_SRGB8` in C
  (`output.width` and `output.height`), or `Decoder.plan_region(data,
  region=..., reduction=r, interpretation="srgb8")` in Python (`"width"` and
  `"height"`).
- Planning fails when the region has no samples left at `r`. Stop there.

A plan's output covers every reduced sample whose footprint overlaps the
region, so it can include part of a sample beyond each edge. The exact window
of ITU-T T.800 B.5, `[ceil(x0 / 2^r), ceil(x1 / 2^r))` on each axis, is
`J2kReferenceGrid::reduced_area` in Rust. For an image whose origin is
`(0, 0)` and a tile request at one of the scale factors in `info.json`, the
two agree, and the decoded tile is exactly the IIIF tile size.

Decode with the 8-bit sRGB output, which also applies palettes, ICC profiles,
sYCC, and channel definitions: `decode_region_srgb8_at_reduction` in Rust,
`j2k_decoder_decode_region_into` with `J2K_INTERPRETATION_SRGB8` in C, or
`Decoder.decode_srgb8` in Python. It returns 8-bit gray, RGB, or RGBA with
straight alpha.

### Rust

```rust
use j2k::{J2kDecoder, J2kSrgb8Layout, Rect};
use j2k_jpeg::{encode_jpeg_baseline, JpegEncodeOptions, JpegSamples, JpegSubsampling};

/// The reduction and 8-bit sRGB pixels for an IIIF region (in image pixels,
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
j2k_error error;
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

`plan.output.channels` is 1, 3, or 4 interleaved 8-bit samples.

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

`data` can be `bytes` or a read-only `mmap` of the file, which is read in
place.

### Command line

`opj_decompress -i image.jp2 -d x0,y0,x1,y1 -r <reduction> -o region.bmp`,
or `j2k decode` with the same options, decodes a region at a reduction to BMP,
PNG, TIFF, or PNM. `-d` is in reference-grid coordinates: for an image whose
origin is `(0, 0)`, that is `x, y, x + w, y + h` of the IIIF region. At
reduction `r` the command line writes the T.800 window described above. See
[Python, C, and the command line](bindings.md) for the other options.

## info.json

The required properties (§5.2) are `@context`
(`http://iiif.io/api/image/3/context.json`), `id`, `type` (`ImageService3`),
`protocol` (`http://iiif.io/api/image`), `profile` (`level0`, `level1` or
`level2`), `width`, and `height`. J2K supplies the image facts without
decoding pixels:

| info.json | Rust | C | Python `inspect` |
| --- | --- | --- | --- |
| `width`, `height` | `J2kDecoder::inspect_reference_grid(bytes)?.image_size` | `j2k_image_info.width`, `.height` | `width`, `height` |
| `tiles[].width`, `.height` | `J2kReferenceGrid::tile_size` | `tile_width`, `tile_height` | `tile_width`, `tile_height` |
| `tiles[].scaleFactors` | `2^r` for `r` in `0..=max_reduction_levels()` | `2^r` up to `j2k_region_plan.max_reduction` | `2**r` up to `max_reduction` |
| `sizes` | The full image's plan at each `r` | The full image's plan at each `r` | `plan_region(data, reduction=r)` |

In Python:

```python
info = j2k.inspect(data)
decoder = j2k.Decoder()
reductions = range(info["max_reduction"] + 1)
sizes = []
for reduction in reversed(reductions):
    plan = decoder.plan_region(data, reduction=reduction)
    sizes.append({"width": plan["width"], "height": plan["height"]})
info_json = {
    "@context": "http://iiif.io/api/image/3/context.json",
    "id": identifier,
    "type": "ImageService3",
    "protocol": "http://iiif.io/api/image",
    "profile": "level1",
    "width": info["width"],
    "height": info["height"],
    "tiles": [{"width": info["tile_width"], "height": info["tile_height"],
               "scaleFactors": [2**reduction for reduction in reductions]}],
    "sizes": sizes,
}
```

Take sizes from `plan_region` rather than dividing the full size yourself: it
gives the exact output of a reduced decode, including for images whose origin
shifts the reduced grid.

The tiles in `info.json` tell clients which regions to ask for. J2K decodes
any region, so an untiled image (one tile as large as the image) can advertise
whatever tile size the server picks. Python's `inspect` also reports the
image and tile-grid origins, the tile grid (`tiles_x`, `tiles_y`), quality
layers, progression order, and code-block size. In Rust,
`J2kDecoder::inspect_coding_style` returns the main header's coding style.

## Cantaloupe

Cantaloupe's `OpenJpegProcessor` runs `opj_decompress` and reads its output,
so it can run J2K through an `opj_decompress` symlink. What follows was
checked against the Cantaloupe source at commit `f1845927c9d6`
(`OpenJpegProcessor.java`, `CommandLocator.java`, `Key.java`, and
`cantaloupe.properties.sample`).

1. Install `j2k-cli` and link it under OpenJPEG's name in a directory of its
   own:

   ```bash
   mkdir -p /opt/j2k/bin
   ln -s "$(command -v j2k)" /opt/j2k/bin/opj_decompress
   ```

2. In `cantaloupe.properties`, select the processor for JPEG 2000 sources and
   point it at that directory:

   ```properties
   processor.selection_strategy = ManualSelectionStrategy
   processor.ManualSelectionStrategy.jp2 = OpenJpegProcessor
   OpenJpegProcessor.path_to_binaries = /opt/j2k/bin
   ```

   `OpenJpegProcessor.path_to_binaries` is a directory that contains
   `opj_decompress`; a full path ending in `opj_decompress` also works. Left
   empty, Cantaloupe runs `opj_decompress` from the `PATH`.

How Cantaloupe drives it, and how J2K responds:

- Cantaloupe runs `opj_decompress -h` once and looks for a `vX.Y.Z` version.
  If it finds one below 2.2, it stops passing `-quiet`. J2K's help names no
  such version, so Cantaloupe keeps passing `-quiet`.
- Each request runs
  `opj_decompress -quiet -i <source> [-d x0,y0,x1,y1] [-r n] -o <temp>/OpenJpegProcessor-<uuid>.bmp`,
  where the output name is a symlink to `/dev/stdout`, and reads a BMP from
  the process's stdout. J2K writes only the image there; notes and errors go
  to stderr.
- `-d` comes from the IIIF region in image pixels, and `-r` from the scale,
  limited by the decomposition count Cantaloupe's own metadata reader finds.
  Cantaloupe then resizes, rotates, and encodes the response itself.
- `-i` is the source path. Files in Cantaloupe's source cache have no
  extension; J2K recognizes its input by signature, not by name.
- When stderr contains `Unknown input file format`, Cantaloupe reports a
  source-format error. J2K prints that message for input that isn't JPEG
  2000.
- Cantaloupe passes no `-threads`, so J2K, like `opj_decompress`, reads
  `OPJ_NUM_THREADS` from Cantaloupe's environment, and decodes on one thread
  when it is unset.

One limit, shared with OpenJPEG: `-d` is in reference-grid coordinates, but
Cantaloupe computes it in image pixels. For an image whose origin isn't
`(0, 0)`, both tools read the values as reference-grid coordinates, so the
decoded area is offset from the requested one by the origin and clamped to the
image.

Cantaloupe's `GrokProcessor` runs `grk_decompress`, with `-i`, `-d`, `-r`,
and `-o` and no `-quiet`. J2K doesn't answer to that name.

## Comparison with opj_decompress

We compared the command line with OpenJPEG 2.5.4's `opj_decompress` on 80
files: 17 generated (8- to 16-bit gray and RGB, signed and unsigned, tiled and
untiled, classic and HTJ2K, 5/3 and 9/7, quality layers, ICC, palette, JP2 and
JPH) and 63 from the repository's test fixtures, including images with
nonzero origins. Each file was decoded at every reduction, whole and in
regions (inside one tile, across a tile corner, at odd offsets, and starting
before the origin), to BMP and PNM, and at full resolution also to PNG and
TIFF, with the same arguments in Cantaloupe's order.

| Outcome | Cases |
| --- | --- |
| Identical samples | 2,006 |
| HTJ2K regions where OpenJPEG's region differs from its own whole-image decode, and J2K's matches the corresponding window of both tools' identical whole-image decodes | 662 |
| ICC profiles: Little CMS and J2K color management differ by at most one 8-bit step | 84 |
| Both tools fail (no samples left at a reduction, unsupported precision or components) | 60 |
| One tool fails | 36 |

The one-tool failures have four causes. Three are known differences of the
J2K command line: it refuses samples deeper than 16 bits, where OpenJPEG
writes BMP anyway; it refuses components with different sampling, where
OpenJPEG writes one PNM per component or the first component to BMP; and it
fails on a palette image at a reduction, where OpenJPEG writes an image. The
fourth is a 6 x 6 image whose coarse levels are empty, which OpenJPEG fails
to decode at reduction 1.
