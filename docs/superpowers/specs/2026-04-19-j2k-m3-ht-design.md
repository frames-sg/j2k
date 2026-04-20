# J2K-M3 — HT (Part 15)

Status: approved implementation spec derived from the umbrella design.

## Goal

Add explicit HTJ2K support to `slidecodec-j2k` and verify that the committed
borrowed, row-streaming, region, scale, and tile-batch APIs all work on HT
codestreams and JP2 containers.

## Scope

In scope:

- `inspect()` support for HT codestreams and JP2 files
- decode coverage for HTJ2K in the in-tree decoder path
- dedicated regression tests using `encode_htj2k`
- M2 API parity on HT inputs

Out of scope:

- any backend-fallback HT decode path
- HT-specific context caching
- OpenHTJ2K parity fixtures outside locally generated roundtrip cases

## Architecture

`slidecodec-j2k` extends the in-tree decoder so HT codestream markers and block
structures are handled directly:

- `J2kDecoder::inspect` must mirror `J2kView::parse` for HT marker detection.
- All decode surfaces continue to route through the crate's own decoder path.
- HT correctness is locked by dedicated tests over raw codestream and JP2-wrapped
  inputs, across 8-bit and native-depth paths.

## Tests

Required tests:

- HT codestream `inspect()` returns sane core `Info`
- HT JP2 `inspect()` returns sane core `Info`
- HT `decode_into` roundtrips grayscale and RGB samples
- HT `decode_scaled_into` returns the expected lower-resolution HT output
- HT `decode_region_into` matches the requested ROI directly
- HT `ImageDecodeRows<'a, u8>` matches full decode
- HT `TileBatchDecode` matches borrowed decoder output

## Verification

J2K-M3 is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`

all pass with the new HT-focused tests in place.
