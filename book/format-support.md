# Format support

This page lists which parts of JPEG 2000 Part 1 and HTJ2K (Part 15) J2K
implements, and their limits. It is a feature list, not a conformance
statement: results from the ISO/IEC 15444-4 test suite are in
[Conformance](conformance.md).

## Support matrix

| ID | Status | Supported | Remaining limitation |
| --- | --- | --- | --- |
| part1-signed-samples | Done | Signed 8- and 16-bit components decode and encode, with signedness reported per component. Component-plane output returns signed samples. Typed component-plane encode can mix signed and unsigned components. | None known. |
| part1-high-bit-depths | Done | Inspect and decode up to 38 bits per component, the Part 1 maximum. Lossless coding is exact at every depth. Lossless encode: classic up to 35 bits with a wavelet transform and 37 without; HTJ2K up to 24 bits with a wavelet transform and 31 without. Lossy encode: classic up to 38 bits, HTJ2K up to 24. | See [High bit depths](#high-bit-depths). |
| part1-component-counts | Done | Up to 16,384 components in inspect, decode, and lossless and lossy encode. Component-plane decode and encode keep each component's sampling. | Interleaved raw-pixel encode rejects subsampled components; use the component-plane encoder. Very large counts are bounded by resource limits. |
| part1-progression-tileparts | Done | All five progression orders. POC in the main header and in tile-part headers. Multi-tile and multi-tile-part encode. TLM is written and checked against the tile-part lengths. | At most 32 quality layers, in decode and encode. |
| part1-packet-markers | Done | PLT, PLM, PPM, PPT, SOP, and EPH in decode and in lossless and lossy encode, for single-tile, multi-tile, and multi-tile-part codestreams. Oversized PPM and PPT payloads are split across markers. | No T.803 case contains PLM, so PLM is tested only by our own tests. |
| part1-roi-maxshift | Done | Maxshift ROI decode on the CPU, CUDA, and Metal, from RGN in the main header or a tile-part header. Whole-component and rectangular ROI encode, lossless and lossy, including multi-tile. | RGN with any other style is rejected. Coded bitplanes including the shift: classic up to 55, HTJ2K up to 31; more returns `Unsupported`. One request cannot mix whole-component and rectangular ROI. |
| jp2-color-component-metadata | Done | Reads and writes IHDR, one or more COLR boxes (enumerated or ICC), BPCC, PCLR, CMAP, and CDEF, including signed palette columns. Decode resolves palettes and component mapping and checks IHDR and BPCC against the codestream. Rewrapping JP2 or JPH keeps the metadata. | ICC round trips are tested against our own fixtures only. |
| part15-ht-refinement-decode | Done | HT cleanup, SigProp, and MagRef decode. The selected T.803 Part 15 cases pass. | None known. |
| part15-ht-refinement-encode | Done | Lossy HTJ2K to a byte or bits-per-pixel target with SigProp and MagRef refinement passes and quality layers, on the CPU, CUDA, and Metal. Lossless HTJ2K with quality layers. | A code block's selected passes all go into one layer. On CUDA, entropy coding is serial within a code block. |
| part15-openhtj2k-qfactor | Done | Opt-in HTJ2K Qfactor 1-100 for grayscale and three-component 4:4:4 RGB, writing the same QCD and QCC values as OpenHTJ2K. | Can't be combined with a byte, bits-per-pixel, or PSNR target, ROI encode, or a quantization profile. More than 16 decomposition levels is rejected. |
| jph-wrapper | Done | Reads and writes JPH. The file type must match the codestream: HTJ2K in a JP2 file, or classic JPEG 2000 in a JPH file, is rejected. An HTJ2K codestream can be wrapped as JPH without re-encoding. | None known. |
| j2k-to-htj2k-recode | Done | Lossless recode to HTJ2K with identical pixels, as a raw codestream or JPH. Reversible 5/3 gray and RGB sources with unsigned 8- or 16-bit samples, subsampled ones included, are recoded in the coefficient domain and keep COLR, ICC, and CDEF metadata. Other sources are decoded and re-encoded losslessly. | The decode-and-re-encode path drops palette, component-mapping, and channel-definition boxes, because it works on resolved pixels. |
| benchmark-publication-gates | Done | `cargo run -p xtask --features adoption -- adoption-report` marks a benchmark bundle `publication_eligible` only when it includes the required external corpora and comparator results. | None known. |
| external-speed-comparisons | Done | Decode comparisons against OpenJPEG, Grok, OpenJPH, and OpenHTJ2K (linked in process), libjpeg-turbo for JPEG, and Kakadu. Results are in [Performance](performance.md). | Kakadu is compared only as a command-line tool. |
| bounded-row-decode | Done | `decode_rows_u8_bounded` and `decode_rows_u16_bounded` stream the rows of classic and HTJ2K images, parsing the tile structure once. | Above 24 bits, a row decode is a whole-image decode. |
| jpx_part2_deferred | Out of scope | | JPX and other Part 2 extensions are out of scope, except what reading JP2 and JPH files needs. |

Other limits:

- A tile-part coding style with fewer decomposition levels than the requested
  reduction is rejected. `J2kDecoder::max_reduction_levels` returns the
  deepest reduction every tile supports.
- Tiles that differ in the multi-component transform need components sampled
  1x1.
- Header inspection accepts images wider or taller than 60,000 pixels; decode
  resource limits still apply.
- Multi-tile encode supports absolute tile origins.

### High bit depths

Reversible coding runs in `f32` up to 20 bits and in integers from 21 to 38
bits, in both encode and decode.

| | Classic JPEG 2000 | HTJ2K |
| --- | --- | --- |
| Decode | Up to 38 bits | Up to 31 coded bitplanes |
| Lossless encode with a wavelet transform | Up to 35 bits | Up to 24 bits |
| Lossless encode without one | Up to 37 bits | Up to 31 bits |
| Lossy encode | Up to 38 bits | Up to 24 bits |

- Requests beyond these limits return `Unsupported`. Part 1 cannot signal more
  than 37 coded bitplanes without quantization, and an HT code block holds at
  most 31.
- Lossless encode above 20 bits and lossy encode above 24 bits run on the CPU
  only. Nothing above 16 bits decodes on a GPU.
- `J2kDecoder::decode_components` and its region forms return `f32` planes
  and stop at 24 bits. `decode_native_components` returns deeper samples, as
  does `decode_native` in `j2k-native`.
- A tile with a reversible component deeper than 20 bits needs the 5/3
  wavelet and no quantization in every component. Tiles that mix this with
  irreversible coding, or that differ in the color transform, are rejected.
- The integer path decodes code blocks on one thread and holds 64-bit
  coefficients. A full-image decode of one 21- to 38-bit component larger than
  about 15 megapixels therefore reaches the decode memory limit. Region
  decodes don't, and neither do bounded-row decodes of 21 to 24 bits.

## Test coverage

Each feature above is tested by J2K's own tests. Where an external test suite
or reference decoder covers it too, the support inventory in
`corpus/j2k-conformance/support-inventory.tsv` records that. Its rows:

| Inventory row | Features | Covered by |
| --- | --- | --- |
| `part1_core_lossless_53` | Part 1 reversible 5/3 | Our tests and the T.803 suite |
| `part1_core_lossy_97_layers_precincts` | Part 1 irreversible 9/7, quality layers, precincts, progression orders | Our tests and the T.803 suite |
| `part1_poc_tlm_sop` | POC, TLM, SOP, progression orders | Our tests and selected T.803 cases |
| `part1_plt_sop_eph` | PLT, separated packet headers, SOP, EPH, POC | Our tests and selected T.803 cases |
| `openhtj2k_ds0_ht_12_b11`, `openhtj2k_ds0_ht_09_b11` | HTJ2K refinement-pass decode | Fixtures derived from OpenHTJ2K |
| `plm_iso_vector_absent` | PLM | Our tests only: no T.803 case contains PLM |
| `icc_roundtrip_deferred` | ICC write, rewrap, and recode | Our tests only: no external ICC fixtures yet |
| `encode_gt16_deferred` | Encode above 16 bits | Our tests at the limits in [High bit depths](#high-bit-depths); not yet compared with other encoders |
| `jpx_part2_deferred` | JPX and Part 2 | Out of scope |

## Decoding features

Region and reduced-resolution planning, 8-bit sRGB output, and their limits
are described in [Regions and reduced
resolutions](decoding.md#regions-and-reduced-resolutions). Working-memory
limits and cancellation are in [Limits and
cancellation](decoding.md#limits-and-cancellation), owned CPU, CUDA, and Metal
batches in [Batch decoding](decoding.md#batch-decoding), wavelet band
output in [Wavelet coefficients](decoding.md#wavelet-coefficients), and
decoding a codestream that has only partly arrived in [Partial
codestreams](decoding.md#partial-codestreams).

`cargo xtask public-support` checks this page against the support inventory;
see [Testing](../contributor-book/testing.md).
