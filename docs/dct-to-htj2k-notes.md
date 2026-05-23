# JPEG DCT To HTJ2K Notes

This document tracks the experimental coefficient-domain path in
`signinum-transcode`. The current implementation keeps codec coupling outside
the JPEG and J2K crates:

```text
JPEG bytes
  -> signinum-jpeg DCT extraction
  -> signinum-transcode DCT-domain 5/3 coefficient mapping
  -> signinum-j2k-native precomputed-band HTJ2K encode
```

## Current Validation

- `cargo test -p signinum-transcode --test corpus_validation` runs committed
  grayscale, 4:4:4, 4:2:2, and 4:2:0 JPEG fixtures through `jpeg_to_htj2k`.
- The corpus report aggregates rounded float-reference coefficient metrics:
  sample count, exact-match count, maximum absolute error, and absolute-error
  histogram buckets.
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

- `row_window_packed_f64`: 801.45-804.71 ns
- `aos_8x8_f64`: 988.16-991.98 ns
- `soa_coefficient_major_f64`: 1.3766-1.3817 us

These numbers only measure scalar packing cost. They are not a final SIMD layout
decision; row-window packing is currently the cheapest scalar conversion
candidate, while SoA remains a candidate for vectorized coefficient-lane work.

## Reusable Scratch

`signinum_transcode::JpegToHtj2kTranscoder` is the stateful API for repeated tile
work. It currently reuses the DCT block conversion buffer and direct 2D
projection weight-row scratch across calls while preserving the same output path
as the stateless `jpeg_to_htj2k` convenience function. The benchmark suite
includes `grayscale_8x8_stateful_reuse` under the `jpeg_to_htj2k` group so
future allocation/layout changes can be measured against the stateless path.

Initial 2026-05-23 tiny-fixture timing is the same order of magnitude: stateless
`grayscale_8x8` measured 92.394-93.799 us and stateful
`grayscale_8x8_stateful_reuse` measured 90.359-91.116 us. This is not a broad
performance claim; it verifies the benchmark surface and shows that scratch
reuse needs larger tile/corpus measurement before promotion.

The direct 2D-grid projection benchmark now has a scratch-reuse comparison for
cached 5/3 weight rows. On the 13x11 synthetic grid, stateless
`direct_linear_13x11` measured 97.579-98.022 us, while
`direct_linear_13x11_scratch_reuse` measured 95.646-96.028 us. This is a small
scalar allocation/layout win, not a SIMD result.

## Open Issues

- The production path still emits rounded float-direct coefficients. The
  integer-reference path is a validation oracle, not yet a replacement
  production transform.
- No SIMD optimization claims are made yet. The scalar Criterion groups are the
  baseline for later work.
- Progressive JPEG, 9/7 lossy, RGB conversion, and chroma upsample remain out
  of scope for this experimental path.
