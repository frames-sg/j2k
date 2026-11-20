# Encoding

The encode functions in `j2k` take interleaved samples and return a raw
codestream. Put it in a `.jp2` or `.jph` file with
[`wrap_j2k_codestream`](#containers). Encoding runs on the CPU unless you pass
a GPU accelerator; see [GPU decoding and encoding](gpu.md). To convert JPEG
files, see [Transcoding](transcoding.md).

## Lossless

```rust
use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

let samples = J2kLosslessSamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
    .with_max_decomposition_levels(Some(5));
let encoded = encode_j2k_lossless(samples, &options)?;
std::fs::write("image.j2c", &encoded.codestream)?;
```

`J2kLosslessSamples::new` takes the sample bytes, width, height, component
count, bit depth, and signedness. Samples deeper than 8 bits take as many
little-endian bytes as the bit depth needs (two for 9 to 16 bits). Images
with three or four components get the reversible color transform on the
first three, unless you choose
`with_reversible_transform(ReversibleTransform::None53)`; lossy encode uses
the irreversible one. Other component counts are coded independently.

For components that don't share one grid or one sample type, use
`encode_j2k_lossless_components` (components with their own sampling) or
`encode_j2k_lossless_typed_components` (components with different precision
or signedness).

### Decomposition levels

Check the number of wavelet levels if you plan to decode at reduced
resolution, for example to serve thumbnails or deep-zoom tiles. Without
`with_max_decomposition_levels`, the encoder uses one level for the LRCP and
RLCP progressions (LRCP is the default), and for RPCL, PCRL, and CPRL enough
levels to bring the smaller side to 64 or less.

Images whose smaller side is under 64 get no levels, even if you ask for
some, and a request is capped by the smallest tile, counting partial edge
tiles and component sampling. `j2k_lossless_decomposition_levels_for_options`
tells you what an encode will use. Lossy encode follows the same rules.

## Classic or HTJ2K

`J2kBlockCodingMode::Classic`, the default, writes JPEG 2000 Part 1 code
blocks, which every JPEG 2000 decoder reads. `HighThroughput` writes HTJ2K
(Part 15) code blocks, which are much faster to encode and decode but need a
decoder with HTJ2K support. Everything else (tiles, progression orders,
markers, containers) works the same with both. The two have different
bit-depth limits; see [Format support](format-support.md).

## Lossy

Lossy encode uses the irreversible 9/7 wavelet. Give it a target, or leave the
target out to use the default quantization:

```rust
use j2k::{
    encode_j2k_lossy, J2kBlockCodingMode, J2kLossyEncodeOptions, J2kLossySamples,
    J2kRateTarget,
};

let samples = J2kLossySamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLossyEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
    .with_rate_target(Some(J2kRateTarget::PsnrDb(42.0)));
let encoded = encode_j2k_lossy(samples, &options)?;
println!(
    "{} bytes, {:.2} bpp, PSNR {:?} dB",
    encoded.report.actual_bytes, encoded.report.actual_bits_per_pixel, encoded.report.psnr_db,
);
```

The targets are:

- `J2kRateTarget::Bytes(n)`: total codestream size.
- `J2kRateTarget::BitsPerPixel(r)`: total codestream bits per image pixel.
- `J2kRateTarget::PsnrDb(db)`: a minimum decoded PSNR.
- `with_qfactor(Some(1..=100))`: a visual quality factor, HTJ2K only, that
  writes the same quantization OpenHTJ2K writes for that setting.

For progressive quality, pass cumulative targets as
`with_quality_layers(vec![J2kQualityLayer::new(...), ...])`, up to 32
layers. Byte and bits-per-pixel layer targets must not decrease.

### How the target search works

Every byte, bits-per-pixel, or PSNR target is a search over one quantization
scale. Each probe is a full encode, though probes share the color transform
and wavelet transform.

- Byte and bits-per-pixel targets return the first result within 1% of the
  target, or the finest admissible scale if that fits. After
  `psnr_iteration_budget` bracketed probes (8 by default), the search returns
  the candidate closest to the target within the size allowance.
- A PSNR target is a minimum within `psnr_tolerance_db` (0.25 dB by default).
  The search doubles the scale, then bisects, and checks the codestream it
  returns. It does not try every scale, so if quality is not monotonic in
  the scale for your image, it can miss a better one.
- The Qfactor can't be combined with a rate target, quality layers, a region
  of interest, or a quantization profile. It works for grayscale and
  three-component RGB without subsampling, and rejects more than 16
  decomposition levels.

### Quantization profiles

`with_quantization_profile` scales the quantization step of each band, per
component. Each component's row has `1 + 3 * levels` weights for the levels
the encode actually uses: the final LL band, then HL, LH, and HH from the
coarsest level to the finest.

Weights multiply the nominal step before it is rounded to a step the
codestream can store, and a weighted step that can't be stored is an error. A
profile of all ones is the same as no profile; without a profile, steps can
saturate instead. For example, with 30-bit input and two guard bits the
nominal step 0.25 clips to the finest storable step, 0.5, so weights near one
are rejected and a weight of two is accepted.

## Layout options

Both option types take the same layout settings:

- `with_tile_size(Some((w, h)))` writes a multi-tile codestream. Tiles don't
  need to align with the wavelet, code-block, or precinct grids.
- `with_progression` picks one of the five progression orders
  (`J2kProgressionOrder`).
- `with_marker_segments` adds TLM, PLT, PLM, PPM, PPT, SOP, or EPH markers
  (`J2kMarkerSegment`). TLM is checked against the tile-part lengths.
- `with_tile_part_packet_limit` splits tiles into tile-parts.

`encode_j2k_lossless_with_roi_regions` and `encode_j2k_lossy_with_roi_regions`
encode rectangular regions of interest with the maxshift method. Each
`J2kRoiRegion` names a component, a rectangle in image pixels, and a shift;
all regions of one component must use the same shift.

## Checking the output

By default (`J2kEncodeValidation::CpuRoundTrip`) the encoder decodes its own
output before returning it. A lossless encode compares every sample with the
input; a lossy encode checks the geometry and measures the PSNR it reports in
`report.psnr_db`. A mismatch is an error.

The check costs one full decode. If you validate the output yourself, for
example by decoding it on the GPU, turn it off with
`with_validation(J2kEncodeValidation::External)`. Lossy reports then have no
PSNR.

## Containers

`wrap_j2k_codestream` puts a codestream in a JP2 or JPH file:

```rust
use j2k::{wrap_j2k_codestream, J2kFileColorSpec, J2kFileWrapOptions};

let jph = wrap_j2k_codestream(&encoded.codestream, J2kFileWrapOptions::jph())?;
let jp2 = wrap_j2k_codestream(
    &classic_codestream,
    J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(&icc)),
)?;
```

HTJ2K goes in JPH files and classic JPEG 2000 in JP2 files; the other
combinations are errors. The color specification defaults to
`J2kFileColorSpec::Infer`, which picks an enumerated color space from the
codestream. `with_color_specs` writes several COLR boxes, and
`with_metadata` writes palette, component-mapping, and channel-definition
boxes. To rewrap an existing file, build both from its metadata with
`J2kFileColorSpec::from_file_metadata` and
`J2kFileBoxMetadata::from_file_metadata`.

## Converting JPEG 2000 to HTJ2K

`recode_j2k_to_htj2k_lossless` turns a JPEG 2000 file or codestream into
lossless HTJ2K with identical pixels:

```rust
use j2k::{
    recode_j2k_to_htj2k_lossless, CompressedPayloadKind, J2kEncodeValidation,
    J2kProgressionOrder, J2kToHtj2kOptions,
};

let options = J2kToHtj2kOptions::new(
    CompressedPayloadKind::JphFile, // the default is a raw codestream
    J2kProgressionOrder::Rpcl,
    J2kEncodeValidation::CpuRoundTrip,
);
let recoded = recode_j2k_to_htj2k_lossless(&bytes, options)?;
println!("{:?}: {} bytes", recoded.report.mode, recoded.bytes.len());
```

`report.mode` says how it was done:

- `Passthrough`: the input already matched, so the bytes are copied.
- `CodestreamPreserving`: the HTJ2K codestream is copied into a new
  container.
- `CoefficientPreserving`: the code blocks are re-coded without decoding to
  pixels. This is used for reversible 5/3 gray and RGB images with unsigned 8-
  or 16-bit samples, subsampled ones included, and it keeps COLR, ICC, and
  CDEF metadata.
- `PixelPreserving`: anything else is decoded and re-encoded losslessly. This
  path drops palette, component-mapping, and channel-definition boxes,
  because it works on resolved pixels.

`J2kToHtj2kOptions::default().with_coefficient_only()` returns an error
instead of taking the pixel path.
