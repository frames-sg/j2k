# JPEG DCT To HTJ2K Notes

This document tracks the experimental coefficient-domain path in
`signinum-transcode`. The current implementation keeps codec coupling outside
the JPEG and J2K crates:

```text
JPEG bytes
  -> signinum-jpeg quantized/dequantized DCT extraction
  -> signinum-transcode DCT-domain 5/3 or 9/7 coefficient mapping
  -> signinum-j2k-native precomputed-band HTJ2K encode
```

## Current Validation

- `cargo test -p signinum-transcode --test corpus_validation` runs committed
  grayscale, 4:4:4, 4:2:2, and 4:2:0 JPEG fixtures through `jpeg_to_htj2k`.
- `signinum-jpeg::transcode::extract_dct_blocks` now exposes both quantized and
  dequantized natural-order DCT blocks at the JPEG boundary. The production
  HTJ2K path still consumes dequantized blocks for the current reversible 5/3
  mapping, while quantized blocks remain available for later pure
  coefficient-domain experiments.
- The default production path is `IntegerDirect53`: the first 5/3 level is
  computed from JPEG DCT blocks without materializing a full spatial image
  plane, then later levels recurse over LL.
- `FloatDirectLinear97` is an opt-in irreversible path: the first 9/7 level is
  computed directly from JPEG DCT blocks using cached linearized lifting
  weights, later levels recurse over LL, and the result encodes through
  `encode_precomputed_htj2k_97`.
- Progressive JPEG inputs use the existing progressive scan accumulator to
  expose final quantized/dequantized DCT blocks to the same transcode path as
  baseline JPEG. No progressive IDCT, RGB conversion, or chroma upsample is
  performed before HTJ2K wavelet generation.
- The corpus report aggregates integer-reference coefficient metrics:
  sample count, exact-match count, maximum absolute error, and absolute-error
  histogram buckets. Aggregate and per-fixture reports now also carry the same
  `Exact` / `OneLsbBounded` / `OutsideThreshold` classification used by
  individual transcodes.
- `TranscodeReport` now carries the typed coefficient path plus optional
  validation classifications. Enabled validation metrics are classified as
  `Exact`, `OneLsbBounded` using the 99.9% exact-match / max-1-LSB threshold,
  or `OutsideThreshold`.
- `signinum-j2k-native::encode_precomputed_htj2k_53` validates precomputed
  5/3 band geometry against the component dimensions implied by SIZ
  `XRsiz`/`YRsiz` before the accelerated DWT hook reaches packetization.
- `signinum-transcode::htj2k_wavelet::WaveletImage53<i32>` now converts to the
  native precomputed HTJ2K representation after descriptor validation and
  reference-grid/SIZ sampling checks. This gives the standalone wavelet-band
  descriptor a direct, tested route into the encoder while keeping the adapter
  outside both codec crates.
- The native encoder has an irreversible 9/7 precomputed-band entry point, and
  `signinum-transcode::htj2k_wavelet::WaveletImage97<f32>` converts into that
  representation with the same geometry and SIZ sampling validation.
- `cargo test -p signinum-transcode --test jpeg_to_htj2k` verifies native
  decoder acceptance, SIZ component sampling, multilevel output, optional
  integer-reference metrics, and external decoder acceptance when OpenJPEG or
  Grok is installed.

## Optional Local WSI Corpus

Normal CI is deterministic and uses committed fixtures only. Local signoff runs
can add extracted WSI JPEG tiles with:

```bash
SIGNINUM_TRANSCODE_WSI_ROOT=/path/to/extracted/jpeg_tiles \
cargo test -p signinum-transcode --test corpus_validation -- --nocapture
```

Environment variables:

- `SIGNINUM_TRANSCODE_WSI_ROOT`: one or more local files/directories separated
  by the platform path separator.
- `SIGNINUM_REQUIRE_TRANSCODE_WSI_ROOT`: fail if no configured external JPEGs
  are found.
- `SIGNINUM_TRANSCODE_WSI_TILE_LIMIT`: maximum number of external tiles; `0`
  means no limit. Default: `8`.
- `SIGNINUM_TRANSCODE_WSI_MAX_PAYLOAD_BYTES`: skip external JPEGs above this
  byte size. Default: `67108864`.

## Scalar Layout Baseline

The first optimization-track benchmark records layout conversion cost before any
SIMD backend is chosen:

```bash
cargo bench --profile release-bench -p signinum-transcode --bench dct53 dct53_layout_candidates
```

Run on 2026-05-23 against 64 synthetic natural-order DCT blocks:

- `row_window_packed_f64`: 803.21-810.43 ns
- `aos_8x8_f64`: 998.67 ns-1.0092 us
- `soa_coefficient_major_f64`: 1.3608-1.3680 us

These numbers only measure scalar packing cost. They are not a final SIMD layout
decision; row-window packing is currently the cheapest scalar conversion
candidate, while SoA remains a candidate for vectorized coefficient-lane work.

## Reusable Scratch

`signinum_transcode::JpegToHtj2kTranscoder` is the stateful API for repeated tile
work. For the float-linear path it reuses the DCT block conversion buffer and
direct 2D projection weight-row scratch. For the integer-direct path it reuses a
block-local ISLOW sample cache and row scratch. The benchmark suite includes
`grayscale_8x8_stateful_reuse` under the `jpeg_to_htj2k` group so future
allocation/layout changes can be measured against the stateless path.

Initial 2026-05-23 tiny-fixture timing is the same order of magnitude: stateless
`grayscale_8x8` measured 92.394-93.799 us and stateful
`grayscale_8x8_stateful_reuse` measured 90.359-91.116 us. This is not a broad
performance claim; it verifies the benchmark surface and shows that scratch
reuse needs larger tile/corpus measurement before promotion.

The direct 2D-grid projection benchmark now has a scratch-reuse comparison for
cached 5/3 weight rows. On the 13x11 synthetic grid, stateless
`direct_linear_13x11` measured 99.164-99.506 us, while
`direct_linear_13x11_scratch_reuse` measured 96.724-97.816 us. This is a small
scalar allocation/layout win, not a SIMD result.

## Integer-Direct Default Benchmark

After switching the default production path to `IntegerDirect53` and adding the
block-local ISLOW sample cache, then exposing true quantized blocks at the JPEG
extraction boundary, the latest `release-bench` verification run measured:

- `grayscale_8x8`: 35.459-35.769 us
- `grayscale_8x8_stateful_reuse`: 35.455-35.806 us
- `grayscale_13x11`: 43.879-44.429 us
- `ycbcr_444_8x8`: 52.989-53.536 us
- `ycbcr_422_16x8`: 55.017-55.837 us
- `ycbcr_420_16x16`: 55.887-56.554 us

These are tiny conformance fixtures, not WSI-scale throughput claims. The
integer-direct path is faster than the previous float-linear default here
because it avoids the expensive scalar matrix projection while producing exact
integer 5/3 coefficients relative to the signinum ISLOW oracle.

The same run measured JPEG DCT extraction with quantized+dequantized block
capture enabled:

- `jpeg_dct_extract/baseline_420_16x16`: 1.3235-1.3358 us
- `jpeg_dct_extract/baseline_420_restart_32x16`: 1.5814-1.5933 us

Criterion reported small extraction-only fixture regressions against the
previous run. This reporting slice did not touch extraction code; end-to-end
tiny-fixture transcode timings stayed within noise except for the 4:4:4 and
4:2:2 fixtures, which were also within Criterion's noise threshold.

## Float-Direct 9/7 Benchmark Baseline

The first irreversible 9/7 scalar benchmark run after adding
`FloatDirectLinear97` measured:

- `dct97_2d_grid_scalar/direct_linear_13x11_scratch_reuse`: 272.90-274.15 us
- `dct97_2d_grid_scalar/idct_then_dwt_reference_13x11`: 8.7117-8.8085 us
- `jpeg_to_htj2k/grayscale_8x8_float_direct_97`: 137.47-138.80 us
- `jpeg_to_htj2k/ycbcr_420_16x16_float_direct_97`: 878.51-882.35 us

This is a correctness-first scalar baseline, not an optimization result. The
direct matrix projection is intentionally expensive before SIMD/GPU work because
it expands cached lifting weights over every DCT basis contribution.

## Open Issues

- The integer-direct production path still uses scalar, on-demand ISLOW block
  sample evaluation for exactness; block-local caching removes repeated block
  decode work, but the path is still scalar and correct-first.
- JPEG extraction now retains both quantized and dequantized DCT blocks. That is
  the correct boundary for later pure coefficient-domain experiments, but it
  adds extraction-only work until an option or downstream consumer can avoid
  one representation.
- No SIMD optimization claims are made yet. The scalar Criterion groups are the
  baseline for later work.
- RGB conversion and chroma upsample remain out of scope for this experimental
  path.
