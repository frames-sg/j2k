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

## Open Issues

- The production path still emits rounded float-direct coefficients. The
  integer-reference path is a validation oracle, not yet a replacement
  production transform.
- No SIMD/layout optimization claims are made yet. The scalar Criterion groups
  are the baseline for later work.
- Progressive JPEG, 9/7 lossy, RGB conversion, and chroma upsample remain out
  of scope for this experimental path.
