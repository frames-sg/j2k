# Decoding

`J2kDecoder` decodes raw JPEG 2000 and HTJ2K codestreams (`.j2k`, `.j2c`)
and JP2 and JPH files. It borrows the compressed bytes and keeps the parsed
headers between calls, so decode several regions from one decoder rather
than building a new one each time. [Format support](format-support.md) lists
which parts of the standard are supported.

## Full images

Pick the call by what you need:

- `decode_srgb8()` returns 8-bit sRGB gray, RGB, or RGBA with straight alpha.
  It applies palettes, component sampling, channel definitions, sYCC
  conversion, and the supported ICC profiles, so it is the call for display.
- `decode_into(&mut out, stride, format)` writes into your buffer in the
  `PixelFormat` you choose, without ICC conversion.
- `decode_native_components()` returns each component at its own bit depth,
  signedness, and sampling, for images with more than four components or
  with components that differ from each other.

```rust
use j2k::{J2kDecoder, PixelFormat};

let mut decoder = J2kDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let stride = width as usize * 3 * 2;
let mut pixels = vec![0_u8; stride * height as usize];
let outcome = decoder.decode_into(&mut pixels, stride, PixelFormat::Rgb16)?;
```

A decode that fails part-way can leave the first rows of your buffer written.
`J2kDecoder::inspect(&bytes)` reads the image's size, components, bit depth,
color space, and tiling without decoding.

## Output sample conversion

`PixelFormat` has gray, RGB, and RGBA layouts with `*8`, `*16`, and signed
`*I16` samples. Sixteen-bit samples are in native byte order. For 1- to
16-bit sources, the CPU, CUDA, and Metal decoders all convert samples this
way:

- `Gray8`, `Rgb8`, and `Rgba8` scale the unsigned sample range to 0-255.
  Negative samples clamp to zero.
- `Gray16`, `Rgb16`, and `Rgba16` widen unsigned sources of 8 bits or less to
  0-65535 and keep the values of 9- to 16-bit sources. Signed sources keep
  their values as 16-bit two's complement.
- Native batch output (`U8`, `U16`, `I16`; see [Batch
  decoding](#batch-decoding)) keeps sample values and signedness.
- Alpha the decoder adds is the output's maximum value (above 8 bits, the
  native maximum). Alpha from the file is converted like any other component.
- Irreversible 9/7 output rounds ties to even before the level shift.

## Regions and reduced resolutions

Decode a 1024 x 1024 area at quarter resolution without decoding the rest of
the image:

```rust
use j2k::{J2kDecoder, J2kScratchPool, PixelFormat, Rect};

let mut decoder = J2kDecoder::new(&bytes)?;
let mut scratch = J2kScratchPool::new();

let area = Rect { x: 4096, y: 8192, w: 1024, h: 1024 }; // full-resolution pixels
let plan = decoder.region_decode_plan(PixelFormat::Rgb8, area, 2)?;
let mut pixels = vec![0_u8; plan.buffer_bytes];
decoder.decode_region_scaled_pow2_into(
    &mut scratch, &mut pixels, plan.row_bytes, PixelFormat::Rgb8, area, plan.reduction_levels,
)?;
```

The region is in full-resolution image coordinates, and the reduction is the
number of resolution levels to drop: 1 is half size, 2 a quarter.
`region_decode_plan` reads only headers and allocates no pixels. It returns
the output rectangle, the reduced image size, the buffer size
(`buffer_bytes`, or `buffer_len(stride)` for a padded stride), the sample
interpretation, and the route the decode will take. The decode can still fail
on bad entropy data or a memory limit. It reconstructs only the part of the
wavelet transform the region needs (for some subsampled images, the tiles it
touches); it never decodes the whole image and crops it.

- `max_reduction_levels()` is the deepest reduction every tile and component
  supports, including tile-level overrides. A deeper request is an error;
  reductions are never clamped.
- An invalid region, a region with no samples at the requested reduction, an
  unsupported output, and a resource limit are separate errors.
- `srgb8_region_decode_plan` and `decode_region_srgb8_at_reduction` give the
  `decode_srgb8` output for a region. The plain 8-bit formats accept ICC
  profiles equivalent to sRGB; other profiles need the sRGB calls, and
  unsupported profiles are errors.
- sRGB output un-premultiplies premultiplied color at source precision, after
  the palette and before color conversion. Premultiplied CIE Lab is rejected,
  and premultiplied input with subsampled or signed components is accepted
  only as 8-bit samples that need no color conversion. Other outputs keep the
  file's alpha association.
- `decode_scaled_into` and `decode_region_scaled_into` take a `Downscale`
  (`Half`, `Quarter`, `Eighth`) instead of a level count.

## Tiles

Slide and imaging containers (TIFF, DICOM) often store each tile as its own
codestream. `decode_tiles_into` decodes many of them in parallel into your
buffers and returns one outcome per tile, in input order:

```rust
use j2k::{decode_tiles_into, PixelFormat, TileBatchOptions, TileDecodeJob};

let stride = tile_width as usize * 3;
let mut outputs = vec![vec![0_u8; stride * tile_height as usize]; tiles.len()];
let mut jobs: Vec<TileDecodeJob<'_, '_>> = tiles.iter().zip(&mut outputs)
    .map(|(input, out)| TileDecodeJob { input, out, stride })
    .collect();
let outcomes = decode_tiles_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())?;
```

`decode_tiles_region_into`, `decode_tiles_scaled_into`, and
`decode_tiles_region_scaled_into` do the same for regions and reductions.
Concurrency is capped so the decodes together fit in about half the host's
memory. Worker buffers are kept between calls while they total 256 MiB or
less; `release_tile_batch_workspaces()` frees them. For a tile inside one
multi-tile codestream, decode the tile's rectangle as a region;
`reference_grid()` and `info().tile_layout` give the tile grid.

For images too large for one buffer, `decode_rows_u8_bounded` and
`decode_rows_u16_bounded` hand rows to a `RowSink` a stripe at a time.
`rows_decode_plan()` says whether the rows will stream or the image's coding
forces a whole-image decode first. Above 24 bits a row decode is always a
whole-image decode.

## Batch decoding

`CpuBatchDecoder` decodes many owned images in parallel and groups the
results by shape into dense tensors:

```rust
use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, DecodeRequest,
    Downscale, EncodedImage,
};

let mut options = BatchDecodeOptions::default();
options.layout = BatchLayout::Nhwc;
let mut decoder = CpuBatchDecoder::new(options);

let inputs = files // Vec<Arc<[u8]>>
    .into_iter()
    .map(|bytes| EncodedImage::new(bytes, DecodeRequest::Reduced { scale: Downscale::Half }))
    .collect();
let result = decoder.decode(inputs)?; // result.errors() lists failed inputs by index
for group in result.groups() {
    let info = group.info(); // dimensions, channels, precision, layout
    if let CpuBatchSamples::U8(samples) = group.samples() {
        // image i of the group came from input group.source_indices()[i]
    }
}
```

`BatchDecodeOptions` also sets `settings` (strict or lenient), `workers`, and
`max_working_memory_bytes`. Requests can be full, region, reduced, or region
and reduced. Output is gray, RGB, or RGBA as `U8`, `U16`, or `I16` samples,
in NCHW or NHWC, without padding.

Components with different sampling factors are replicated onto the image
grid. Images whose components all share one subsampling, other component
counts, mixed precision or signedness, and precision above 16 bits can't be
batched; use the component-plane calls for those. `prepare` parses a batch
once, and `decode_prepared` decodes it as often as you like, for example once
per training epoch.

To decode several regions of one image without parsing it again, prepare it
with `EncodedImage::operation_source` and pass `PreparedImageOperation`s,
each with its own region and reduction, to `decode_operations`.
`plan_operations` returns their output shapes first. The GPU batch decoders
take the same operations:

| | `CpuBatchDecoder` | `MetalBatchDecoder` | `CudaBatchDecoder` |
| --- | --- | --- | --- |
| `decode_operations` | Yes | Yes, and `submit_operations` | Yes, and `submit_operations` |
| Output limit and cancellation | Yes | Yes | Yes |
| Working-memory limit | Yes | No (`Unsupported`) | No (`Unsupported`) |
| Components with different sampling | Yes | Full-image RGB only | No |

Geometry, sample values, and grouping are the same on all three. On a GPU,
cancellation is checked during preparation and before each group is
submitted; submitted work runs to completion. An explicit GPU request fails
for inputs it can't handle instead of falling back to the CPU. The Python
`Decoder`, the C operation API, and the Burn adapters in [Machine
learning](machine-learning.md) expose the same operations.

## Limits and cancellation

Every decode is limited to 512 MiB of working memory, so a hostile file
returns an error instead of exhausting memory. A `J2kDecodeControl` sets
tighter limits for one operation and can cancel it:

```rust
use std::num::NonZeroUsize;
use j2k::{DecodeSettings, J2kCancellationToken, J2kDecodeControl, J2kDecoder};

let token = J2kCancellationToken::new(); // call token.cancel() from another thread
let mut control = J2kDecodeControl::default();
control.max_working_memory_bytes = NonZeroUsize::new(256 << 20);
control.max_output_bytes = Some(64 << 20);
control.cancellation = Some(token.clone());

let mut decoder = J2kDecoder::new_with_control(&bytes, DecodeSettings::default(), &control)?;
let image = decoder.decode_srgb8_with_control(&control)?;
```

A control applies only to the call you pass it to. The region calls have
`*_with_control` forms, and `J2kView::parse_with_control` and
`CpuBatchDecoder::prepare_with_control` take one too.

- Working memory counts parsed metadata, plans, scratch, and buffers the
  codec allocates, but not your compressed input, an existing prepared image,
  or your output buffer. It is not a process memory limit, and it can lower
  the 512 MiB ceiling but not raise it.
- The output limit covers the packed output span, row padding included.
- Under a working-memory limit, ICC conversion needs a matrix/TRC or gray-TRC
  profile with an XYZ connection space; other profiles are `Unsupported`.
- Cancellation is checked between markers, packets, code blocks, transform
  steps, and rows, so a cancellable decode runs its blocks and rows serially.
  It can't interrupt a submitted GPU command. A cancelled decode returns
  `J2kError::Cancelled`, can leave the start of your buffer written, and
  doesn't replay rows already delivered. A token stays cancelled; reuse the
  decoder with a new one.
- A decoder keeps one reduction's parsed image between region decodes;
  `set_preparation_cache_limit_bytes` bounds it and 0 turns it off.
  `CpuBatchDecoder::set_operation_cache_limit_bytes` does the same for
  operation plans.

## Strict and lenient parsing

`J2kDecoder::new` and `J2kView::parse` are strict: malformed metadata is an
error. To accept some damaged JP2 and JPH files, pass
`DecodeSettings::lenient()` to `J2kDecoder::new_with_settings` or
`J2kView::parse_with_settings`.

Lenient mode relaxes only metadata checks. It can ignore a malformed
top-level box after the header and codestream, a malformed child box after
`ihdr` and `colr`, or malformed optional `cdef` or `pclr` boxes (keeping an
earlier complete one), and it can infer one undeclared alpha component.
Codestream and entropy validation, bounds checks, and memory limits stay as
strict. A decode that used one of these recoveries reports
`J2kDecodeWarning::LenientMetadataRecovery`.

## Partial codestreams

A viewer that fetches a file a piece at a time can show a low-resolution
image before the rest arrives. With
`DecodeSettings::with_partial_codestream(true)`, the decoder accepts input
that stops anywhere after the codestream's main header:

```rust
use j2k::{DecodeSettings, J2kDecoder, J2kScratchPool, PixelFormat, Rect};

let settings = DecodeSettings::strict().with_partial_codestream(true);
let mut decoder = J2kDecoder::new_with_settings(&bytes_so_far, settings)?;
if let Some(reduction) = decoder.codestream_completeness()?.min_exact_reduction() {
    let (width, height) = decoder.info().dimensions;
    let area = Rect { x: 0, y: 0, w: width, h: height };
    let plan = decoder.region_decode_plan(PixelFormat::Gray16, area, reduction)?;
    let mut pixels = vec![0_u8; plan.buffer_bytes];
    decoder.decode_region_scaled_pow2_into(
        &mut J2kScratchPool::new(), &mut pixels, plan.row_bytes, PixelFormat::Gray16,
        area, reduction,
    )?;
}
```

`codestream_completeness` reads the packet headers without decoding samples
and reports:

- `min_exact_reduction`: the fewest resolution levels to drop for a decode
  of the whole image that only reads data that is present. A decode at that
  reduction or deeper returns the same samples as the complete file decoded
  at the same reduction. It is `None` until the lowest resolution of every
  tile has arrived, and `Some(0)` once every packet is present.
- `tile_min_exact_reduction`: the same for each tile, so a region over tiles
  that have arrived can be decoded while others are still missing.
- `is_complete`: whether every packet is present. The final EOC marker may
  still be missing.

A decode that needs data that hasn't arrived fails with
`J2kError::IncompleteCodestream`, which counts as truncated input
(`is_truncated()`). Decoding never fills in missing data.

How quickly resolutions become usable depends on the progression order.
With a resolution-first order (RPCL or RLCP), each resolution level is
complete as soon as its bytes arrive, which makes RPCL the order to choose
for progressive delivery. With a layer-first order (LRCP), a resolution is
complete only once every quality layer of it has arrived.

The input can be a raw codestream or a JP2/JPH file that stops inside its
codestream box; the boxes before it must be complete. The JP2/JPH container
is otherwise read as strictly as without the setting, and so is every tile
part that is present in full: a malformed one is still an error. Partial
input is never offered for byte copying, and it decodes on the CPU only.

## Wavelet coefficients

`decode_wavelets` returns the wavelet coefficients of the bands you select,
without running the inverse transform:

```rust
use j2k::{J2kDecoder, WaveletSelection};

let mut decoder = J2kDecoder::new(&bytes)?;
let selection = WaveletSelection {
    resolutions: vec![0], // the LL band only; empty lists select everything
    ..WaveletSelection::default()
};
let wavelets = decoder.decode_wavelets(&selection)?;
for band in wavelets.tiles.iter().flat_map(|t| &t.components).flat_map(|c| &c.bands) {
    // band.kind, band.rect, band.quantization_step, band.samples
}
```

- Reversible 5/3 bands are exact `WaveletSamples::I64`, which needs a
  codestream coded without quantization; irreversible 9/7 bands are
  dequantized `WaveletSamples::F32`. Classic and HTJ2K both work.
- The output keeps tile and component geometry, sampling, precision,
  signedness, the transform, whether a color transform applies, and each
  band's quantization step.
- Selecting tiles, components, resolutions, or orientations skips entropy
  decoding of everything else. Packets are still parsed, and coefficient
  storage is still allocated for the active tile.
- No inverse transform, color conversion, JP2 component mapping, or ICC is
  applied. This runs on the CPU only, with no incremental refinement.

## Copying compressed bytes

When moving JPEG 2000 between containers (DICOM to TIFF, say) whose
destination takes the same format, copy the compressed bytes instead:

```rust
use j2k::{CompressedPayloadKind, CompressedTransferSyntax, J2kView};
use j2k::{PassthroughDecision, PassthroughRequirements};

let view = J2kView::parse(&bytes)?;
let wanted = PassthroughRequirements::new(
    CompressedTransferSyntax::HtJpeg2000Lossless,
    CompressedPayloadKind::Jpeg2000Codestream,
)
.with_dimensions((1024, 1024));
match view.passthrough_candidate().map(|c| c.evaluate(&wanted)) {
    Some(PassthroughDecision::Copy { bytes }) => { /* write `bytes` unchanged */ }
    _ => { /* decode and re-encode */ }
}
```

The candidate describes the format, container, size, components, bit depth,
and tile layout. A file read with a lenient recovery is never a candidate.
