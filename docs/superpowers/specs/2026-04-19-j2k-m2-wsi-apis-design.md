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

- native decode-time resolution reduction through codestream resolution descent
- native region decode that constrains the decode window instead of cropping a
  full-frame output buffer after the fact
- row streaming over decoded 8-bit and 16-bit output
- tile-batch convenience entry points through `TileBatchDecode`

Out of scope:

- benchmark comparator acceptance gate
- tile/header cache reuse inside `J2kContext`
- any decode-then-crop or decode-then-decimate acceptance path for native ROI
  or scaled decode

## Architecture

M2 builds on the committed J2K-M1 in-tree decoder path.

### Scale

`decode_scaled_into` uses codestream resolution descent so the decoder produces
the requested lower-resolution output directly.

### Region

`decode_region_into` constrains the codestream traversal to the requested ROI
and writes only the requested pixels into the caller buffer. The milestone does
not accept a full-frame decode followed by an in-memory crop as the native ROI
implementation.

### Row decode

Row decode uses `J2kScratchPool`-owned reusable buffers:

- `Vec<u8>` for 8-bit packed output
- `Vec<u8>` for native-depth byte output
- `Vec<u16>` as a row staging buffer for `RowSink<u16>`

### Tile-batch

`J2kCodec` implements `TileBatchDecode` and forwards to the borrowed decoder.
`J2kContext` exists as the per-worker hook point for tile-state reuse.

## Public Types

Add:

- `pub struct J2kScratchPool`
- `pub struct J2kContext`
- `pub struct J2kCodec`

`J2kScratchPool` tracks reusable internal buffers and reports their total
reserved bytes through `ScratchPool::bytes_allocated()`.

`J2kContext` implements `CodecContext` with empty cache stats in M2.

## Tests

Required tests:

- scaled decode returns the same pixels as a reference codestream decoded at
  the lower resolution directly
- region decode matches the requested ROI from the same source without
  accepting a crop-after-decode implementation as the contract
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
