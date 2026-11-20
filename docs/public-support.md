# JPEG 2000 and HTJ2K support

Which JPEG 2000 Part 1 and HTJ2K Part 15 features are implemented, and their
limits. This is not a conformance statement; ISO/IEC 15444-4 results are in
[`docs/t803-conformance.md`](t803-conformance.md).

## Support matrix

| ID | Status | Supported | Remaining limitation |
| --- | --- | --- | --- |
| part1-signed-samples | Done | Signed 8- and 16-bit components decode and encode, with signedness reported per component. Component-plane output returns signed samples. Typed component-plane encode can mix signed and unsigned components. | None known. |
| part1-high-bit-depths | Done | Inspect and decode up to 38 bits per component, the Part 1 maximum. Lossless coding is exact at every depth. Lossless encode: classic up to 35 bits with a wavelet transform and 37 without; HTJ2K up to 24 bits with a wavelet transform and 31 without. Lossy encode: classic up to 38 bits, HTJ2K up to 24. | See [High bit depths](#high-bit-depths). |
| part1-component-counts | Done | Up to 16,384 components in inspect, decode, and lossless and lossy encode. Component-plane decode and encode keep each component's sampling. | Interleaved raw-pixel encode rejects subsampled components; use the component-plane encoder. Very large counts are bounded by resource limits. |
| part1-progression-tileparts | Done | All five progression orders. POC in the main header and in tile-part headers. Multi-tile and multi-tile-part encode. TLM is written and checked against the tile-part lengths. | At most 32 quality layers, in decode and encode. |
| part1-packet-markers | Done | PLT, PLM, PPM, PPT, SOP, and EPH in decode and in lossless and lossy encode, for single-tile, multi-tile, and multi-tile-part codestreams. Oversized PPM and PPT payloads are split across markers. | T.803 has no case with PLM (`plm_iso_vector_absent`), so PLM is covered by this repository's tests only. |
| part1-roi-maxshift | Done | Maxshift ROI decode on the CPU, CUDA, and Metal, from RGN in the main header or a tile-part header. Whole-component and rectangular ROI encode, lossless and lossy, including multi-tile. | RGN with any other style is rejected. Coded bitplanes including the shift: classic up to 55, HTJ2K up to 31; more is `Unsupported`. One request cannot mix whole-component and rectangular ROI. |
| jp2-color-component-metadata | Done | Reads and writes IHDR, one or more COLR boxes (enumerated or ICC), BPCC, PCLR, CMAP, and CDEF, including signed palette columns. Decode resolves palettes and component mapping and checks IHDR and BPCC against the codestream. Rewrapping JP2 or JPH keeps the metadata. | ICC round trips are tested against this repository's fixtures only (`icc_roundtrip_deferred`). |
| part15-ht-refinement-decode | Done | HT cleanup, SigProp, and MagRef decode. The selected T.803 Part 15 cases pass. | None known. |
| part15-ht-refinement-encode | Done | Lossy HTJ2K to a byte or bits-per-pixel target with SigProp and MagRef refinement passes and quality layers, on the CPU, CUDA, and Metal. Lossless HTJ2K with quality layers. | A block's selected passes all go into one layer. On CUDA, entropy writing is serial within a code block. |
| part15-openhtj2k-qfactor | Done | Opt-in HTJ2K Qfactor 1-100 for grayscale and three-component 4:4:4 RGB, writing the QCD and QCC values OpenHTJ2K writes. | Not combined with a byte, bits-per-pixel, or PSNR target, ROI encode, or a quantization profile. More than 16 decomposition levels is rejected. |
| jph-wrapper | Done | Reads and writes JPH. The file type must match the codestream: HTJ2K in a JP2 file, or classic JPEG 2000 in a JPH file, is rejected. An HTJ2K codestream can be wrapped as JPH unchanged. | None known. |
| j2k-to-htj2k-recode | Done | Lossless recode to HTJ2K with identical pixels, as a raw codestream or JPH. Reversible 5/3 gray and RGB sources with unsigned 8- or 16-bit samples, subsampled ones included, are recoded in the coefficient domain and keep COLR, ICC, and CDEF metadata. Other sources are decoded and re-encoded losslessly. | The pixel path drops palette, component-mapping, and channel-definition boxes, because it works on resolved pixels. |
| benchmark-publication-gates | Done | `adoption-report` accepts a benchmark bundle for publication only with the required external corpora and comparator results (`publication_eligible`). | None. |
| external-speed-comparisons | Done | Comparison harnesses against OpenJPEG and Grok, with optional OpenJPH and Kakadu rows. | OpenJPH and Kakadu rows are command-line runs and are reported separately. |
| bounded-row-decode | Done | `decode_rows_u8_bounded` and `decode_rows_u16_bounded` stream the rows of classic and HTJ2K images, parsing the tile graph once. | Above 24 bits a row decode is a whole-image decode. |
| jpx_part2_deferred | Out of scope | | JPX and other Part 2 extensions are Out of scope, except what JP2/JPH still-image decoding needs. |

Also:

- A tile-part coding style with fewer decomposition levels than the requested
  reduction is rejected; `max_reduction_levels` reports what every tile
  supports.
- Tiles that differ in the multi-component transform need components sampled
  1x1.
- Header inspection accepts dimensions beyond 60,000 pixels; decode resource
  limits still apply. Multi-tile encode supports absolute tile origins. See
  [geometry](architecture.md#geometry-and-provenance).

### High bit depths

Reversible coding runs in `f32` through 20 bits and in integers from 21 to 38
bits, on encode and decode.

| | Classic JPEG 2000 | HTJ2K |
| --- | --- | --- |
| Decode | Up to 38 bits | Up to 31 coded bitplanes |
| Lossless encode with a wavelet transform | Up to 35 bits | Up to 24 bits |
| Lossless encode without one | Up to 37 bits | Up to 31 bits |
| Lossy encode | Up to 38 bits | Up to 24 bits |

- Requests beyond these are `Unsupported`: Part 1 cannot signal more than 37
  coded bitplanes without quantization, and an HT code block carries at most
  31.
- Lossless encode above 20 bits and lossy encode above 24 bits run on the CPU
  only. Nothing above 16 bits decodes on a GPU.
- The `f32` component-plane APIs (`decode_components` and its region forms)
  stop at 24 bits. `decode_native_components` and `decode_native` return
  deeper samples.
- A tile with a reversible component deeper than 20 bits needs the 5/3
  wavelet and no quantization in every component. Tiles that mix this with
  irreversible coding, or that differ in the color transform, are rejected.
- The integer path decodes code blocks on one thread and holds 64-bit
  coefficients, so a full-image decode of one 21- to 38-bit component past
  about 15 megapixels reaches the decode memory limit. Region decodes do not.
  Bounded-row decodes do not for 21 to 24 bits.

## Region planning

`J2kDecoder::region_decode_plan` describes a region decode without running it:
the source and reduced rectangles, the reference-grid origin, the packed
layout, the sample and alpha interpretation, the buffer span (tight or
padded), and the decode route the headers imply. It parses bounded metadata
and allocates no pixels. The decode can still fail on bad entropy data or a
working-memory limit.

- `max_reduction_levels` is the deepest reduction every tile and component
  supports, counting COD and COC overrides. A reduction is never clamped.
- An invalid source region, a region with no samples at the reduction, an
  unsupported output interpretation, and a resource failure are distinct
  errors.
- A region decode reconstructs only the wavelet window it needs or, for some
  subsampled images, only the tiles it touches. It never decodes and crops
  the full image.

`srgb8_region_decode_plan` and `decode_region_srgb8_at_reduction` return 8-bit
sRGB with straight alpha for a region, applying palettes, component sampling,
and the supported ICC profiles.

- The ordinary packed 8-bit routes also accept ICC profiles recognized as
  equivalent to sRGB. Other profiles need the sRGB calls; unsupported
  profiles are errors.
- Premultiplied color is un-premultiplied at the source precision, after
  palette mapping and before sYCC conversion or quantization. Premultiplied
  CIE Lab is rejected. Premultiplied input with subsampled or signed
  components is accepted only as 8-bit mapped samples that need no color
  conversion.
- Native output keeps its alpha association and precision.

## Decode controls

A `J2kDecodeControl` gives one operation a working-memory limit, an output
limit, and an optional `J2kCancellationToken`.

- **Working memory** counts newly parsed metadata, derived plans, scratch, and
  the codec's own conversion and output buffers. It does not count the
  caller's compressed input, an existing source preparation, or the caller's
  output buffer. It is not a process memory limit and cannot raise the native
  ceiling.
- **Output** is the packed span, stride gaps included.
- **ICC under a working limit** needs a matrix/TRC or gray-TRC profile with an
  XYZ connection space; other profiles are `Unsupported`.
- **Cancellation** is permanent for a token. It is checked before preparation
  and between marker, packet, code-block, transform, and row work. A
  cancellable native operation runs its blocks and row bands serially. It
  cannot interrupt a submitted GPU command or a library call. A cancelled
  decode can leave the start of the caller's buffer written, rows already
  delivered are not replayed, and the decoder can be reused with a new token.
- `J2kView::parse_with_control` and `J2kDecoder::new_with_control` parse under
  a control. `CpuBatchDecoder::prepare_with_control` prepares one input at a
  time; an input that fails a limit or sees a cancellation becomes an indexed
  preparation error.
- A borrowed decoder keeps at most one reduction's parsed image and packet
  information between regions; `set_preparation_cache_limit_bytes` bounds it,
  and 0 turns it off. A CPU session bounds its most recent derived operation
  plan with `set_operation_cache_limit_bytes`.

## Owned batch decoding

The batch decoders take full, region, reduced, and region-and-reduced
requests and group Gray, RGB, or RGBA output by shape without padding, as
`U8`, `U16`, or `I16` samples in NCHW or NHWC.

- Components with different sampling factors are replicated onto the image
  grid.
- Images whose components all share one subsampling, other component counts,
  mixed precision or signedness, and precision above 16 bits are not batched;
  use the component-plane APIs.
- On the CPU, a region request decodes a copy of the plan sized to the region
  when that copy holds at most half the coefficients, or when the whole-level
  plan holds 16 Mi or more. The GPU backends always keep whole levels.
- Preparation failures are reported per input. An explicit GPU request is an
  error for inputs its plans cannot describe; it does not fall back to the
  CPU. A GPU error tied to one job names its source image; a command-buffer
  failure fails the whole group.

`PreparedImageOperation` adds an exact reduction and source region to a
prepared image:

| | `CpuBatchDecoder` | `MetalBatchDecoder` | `CudaBatchDecoder` |
| --- | --- | --- | --- |
| `decode_operations` | Yes | Yes, and `submit_operations` | Yes, and `submit_operations` |
| Geometry, samples, and grouping | Reference | Same as the CPU | Same as the CPU, for 5/3 and 9/7 |
| Output limit and cancellation | Yes | Yes | Yes |
| Working-memory limit | Yes | `Unsupported` | `Unsupported` |
| Components with different sampling | Yes | Full-image RGB only; other requests fail per group | Fail per group |

On a GPU, a token is checked during preparation and before each group is
submitted; submitted work runs to completion. Operations share a group only
when they carry the same token. An operation with a control is placed in its
own output groups on the CPU.

Other interfaces to the same operations:

- **C**: `j2k_decoder_plan_region` and `j2k_decoder_decode_region_into` with
  `J2K_OPERATION_OPTIONS_INIT`, and the optional
  `j2k_decoder_prepare_operations`. See the
  [C API](../crates/j2k-c/docs/api.md).
- **Python**: `ImageRequest`, `Decoder.plan_region`, `Decoder.decode_srgb8`,
  and `DecodeControl`. See the [Python README](../crates/j2k-python/README.md).
- **Burn**: [`docs/j2k-ml.md`](j2k-ml.md). **MPSGraph**:
  [`docs/j2k-mpsgraph.md`](j2k-mpsgraph.md).

The Metal full-image and tile-batch display APIs also decode subsampled RGB;
see [Metal sampled-component decode](architecture.md#metal-sampled-component-decode).
`sampled_color_batch_keeps_one_resident_submission` tests it.

## Native wavelet output

`J2kDecoder::decode_wavelets` (CPU) returns reversible 5/3 bands as exact
`I64` and irreversible 9/7 bands as dequantized `F32`, for classic and HTJ2K.

- It keeps tile and component geometry, sampling, precision, signedness, the
  transform, whether a color transform applies, and each band's quantization
  step.
- Selecting tiles, components, resolutions, or orientations skips the entropy
  decoding of everything else. Packets are still parsed and coefficient
  storage is still allocated for the active tile.
- It runs no inverse DWT, color conversion, JP2 component mapping, or ICC.
- Reversible output needs a codestream coded without quantization.
- There is no incremental refinement and no GPU coefficient output.

Tests: `classic_and_ht_bands_reconstruct_original_odd_sized_image`,
`irreversible_coefficients_reconstruct_within_one_lsb_of_pixel_decode`,
`independent_signed_multitile_coefficients_preserve_origins_and_selection`,
`high_precision_coefficients_never_round_through_float_storage`, and
`selected_ll_bands_skip_entropy_work_and_never_run_inverse_dwt`.

## Checking this page

```bash
cargo xtask public-support          # this page, the inventory, and the test list
cargo xtask public-support --final  # also requires the core rows to be Done
cargo test -p j2k-native
cargo test -p j2k
```

`cargo xtask public-support` checks that this page has every required row and
comparator and names every row of
`corpus/j2k-conformance/support-inventory.tsv`, and that every test backing
the matrix (listed in `xtask/src/public_support.rs`) still exists in the
source tree. It does not run the tests. Full validation runs `--final`.
Benchmark publication is checked separately:

```bash
cargo run -p xtask --features adoption -- adoption-report
```

Run it that way, not as `cargo xtask adoption-report`, which is built without
the feature.

## Support inventory rows

Rows of `corpus/j2k-conformance/support-inventory.tsv`:

- `part1_core_lossless_53`: Part 1 reversible 5/3.
- `part1_core_lossy_97_layers_precincts`: Part 1 irreversible 9/7, quality
  layers, precincts, and progression orders.
- `part1_poc_tlm_sop`: POC, TLM, SOP, and progression orders.
- `part1_plt_sop_eph`: PLT, SOP, EPH, and POC.
- `openhtj2k_ds0_ht_12_b11`, `openhtj2k_ds0_ht_09_b11`: HTJ2K refinement-pass
  fixtures.
- `plm_iso_vector_absent`: no external vector has PLM; this repository's
  tests cover it.
- `icc_roundtrip_deferred`: ICC write, rewrap, and recode are covered locally;
  external ICC fixtures are not.
- `encode_gt16_deferred`: encode above 16 bits is covered locally at the
  limits in [High bit depths](#high-bit-depths); external encoder parity is
  not.
- `jpx_part2_deferred`: Part 2 extensions are out of scope.
