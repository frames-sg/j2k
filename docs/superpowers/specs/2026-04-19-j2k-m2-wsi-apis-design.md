# J2K-M2 — WSI APIs

Status: approved implementation spec derived from the umbrella design.

## Goal

Add the WSI-facing API surface to `slidecodec-j2k`:

- `decode_scaled_into`
- `decode_region_into`
- `ImageDecodeRows<'a, u8>`
- `ImageDecodeRows<'a, u16>`
- `TileBatchDecode` via a unit `J2kCodec`
- caller-owned `J2kContext` and expanded `J2kScratchPool`

## Scope

In scope:

- decode-time resolution reduction through backend target-resolution support
- functional region decode by cropping decoded output into the requested ROI
- row streaming over decoded 8-bit and 16-bit output
- tile-batch convenience entry points through `TileBatchDecode`

Out of scope:

- codestream-native ROI skipping
- tile/header cache reuse inside `J2kContext`
- performance claims against OpenJPEG

## Architecture

M2 still builds on the committed J2K-M1 backend adapter.

### Scale

`decode_scaled_into` uses backend `DecodeSettings::target_resolution` to request
a lower-resolution decode directly from the JPEG 2000 engine.

### Region

`decode_region_into` decodes the full requested resolution, then crops the
requested ROI into the caller buffer. This is functionally correct and keeps the
public API stable; codestream-native ROI skipping remains a later optimization.

### Row decode

Row decode uses `J2kScratchPool`-owned reusable buffers:

- `Vec<u8>` for 8-bit packed output
- `Vec<u8>` for native-depth byte output
- `Vec<u16>` as a row staging buffer for `RowSink<u16>`

### Tile-batch

`J2kCodec` implements `TileBatchDecode` and forwards to the borrowed decoder.
`J2kContext` exists now as the per-worker hook point even though M2 does not yet
cache tile state.

## Public Types

Add:

- `pub struct J2kScratchPool`
- `pub struct J2kContext`
- `pub struct J2kCodec`

`J2kScratchPool` now tracks reusable internal buffers and reports their total
reserved bytes through `ScratchPool::bytes_allocated()`.

`J2kContext` implements `CodecContext` with empty cache stats in M2.

## Tests

Required tests:

- scaled decode matches backend target-resolution decode
- region decode matches cropping the corresponding full decode
- `ImageDecodeRows<'a, u8>` matches `decode_into(..., Rgb8/Gray8)`
- `ImageDecodeRows<'a, u16>` matches `decode_into(..., Rgb16/Gray16)`
- `TileBatchDecode::decode_tile` matches borrowed decoder decode
- `TileBatchDecode::decode_tile_region` matches region decode
- `TileBatchDecode::decode_tile_scaled` matches scaled decode

## Verification

J2K-M2 is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`

all pass with the new row/tile/region/scaled tests in place.
