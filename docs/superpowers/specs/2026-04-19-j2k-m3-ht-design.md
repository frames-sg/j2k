# J2K-M3 — HT (Part 15)

Status: approved implementation spec derived from the umbrella design.

## Goal

Add explicit HTJ2K support to `slidecodec-j2k` and verify that the committed
borrowed, row-streaming, region, scale, and tile-batch APIs all work on HT
codestreams and JP2 containers.

## Scope

In scope:

- `inspect()` support for HT codestreams and JP2 files
- decode coverage for HTJ2K via the existing backend dependency
- dedicated regression tests using `encode_htj2k`
- M2 API parity on HT inputs
- graceful scaled-decode fallback when backend HT `target_resolution` is unavailable

Out of scope:

- native in-tree HT block decoder
- HT-specific context caching
- OpenHTJ2K parity fixtures outside locally generated roundtrip cases

## Architecture

`dicom-toolkit-jpeg2000` already enables `openjph-htj2k` by default and
auto-detects HT block coding at parse time. `slidecodec-j2k` therefore needs
only a thin correctness layer:

- `J2kDecoder::inspect` must mirror `J2kView::parse` and fall back to backend
  inspection when the local scalar parser rejects HT markers such as `CAP`.
- All decode surfaces continue to route through the existing backend adapter.
- `decode_scaled_into` keeps the public API available on HT inputs by falling
  back to full decode + power-of-two decimation when the backend rejects
  scaled HT decode through OpenJPH.
- HT correctness is locked by dedicated tests over raw codestream and JP2-wrapped
  inputs, across 8-bit and native-depth paths.

## Tests

Required tests:

- HT codestream `inspect()` returns sane core `Info`
- HT JP2 `inspect()` returns sane core `Info`
- HT `decode_into` roundtrips grayscale and RGB samples
- HT `decode_scaled_into` matches backend target-resolution decode
- HT `decode_region_into` matches cropping the full decode
- HT `ImageDecodeRows<'a, u8>` matches full decode
- HT `TileBatchDecode` matches borrowed decoder output

## Verification

J2K-M3 is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`

all pass with the new HT-focused tests in place.
