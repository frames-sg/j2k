# TIFF JPEG Tile Preparation API Design

- **Date:** 2026-06-07
- **Status:** Design approved for first implementation slice.
- **Scope:** `signinum-jpeg` public JPEG marker utilities, TIFF/WSI JPEG tile
  preparation, and a prepared-tile RGB8 batch facade.

## 1. Boundary decision

Statumen owns TIFF/WSI container interpretation. It reads IFD metadata, tile
offsets and byte counts, `JPEGTables`, expected tile dimensions, pyramid state,
and NDPI-specific container facts.

Signinum owns JPEG interchange preparation. It parses JPEG markers, combines
`JPEGTables` with tile scan bytes, normalizes SOI/EOI framing, handles duplicate
JPEG tables, preserves DRI/restart semantics, repairs NDPI zero-SOF dimensions
when the caller supplies expected dimensions, and returns decode-ready JPEG
payloads.

This split keeps Statumen from carrying a second JPEG scanner. Statumen supplies
container-derived facts; Signinum performs codec-specific validation and byte
rewriting.

## 2. Goals

- Let Statumen delete duplicated JPEG scanners in TIFF, NDPI, and decode glue.
- Keep the implementation in `signinum-jpeg`, close to the existing parser and
  decoder invariants.
- Preserve zero-copy behavior when a tile is already a complete valid JPEG.
- Allocate only when assembly, marker normalization, table deduplication, or SOF
  repair changes bytes.
- Surface malformed or ambiguous JPEG structures as typed `JpegError`s rather
  than silent fallback.

## 3. Non-goals

- Statumen does not gain new JPEG parsing responsibilities.
- The API does not parse TIFF tags or read tile bytes from storage.
- This slice does not include JP2K codestream diagnostics, JP2K raster shaping,
  or unified device fallback decisions.
- This slice does not introduce a new runtime queue or hidden output allocator.

## 4. Public JPEG marker API

Add a public marker module in `signinum-jpeg`, re-exported by the facade
`signinum::jpeg`.

```rust
pub fn iter_segments(input: &[u8]) -> JpegSegmentIter<'_>;
pub fn is_sof_marker(marker: u8) -> bool;
pub fn parse_sof_info(marker: u8, payload: &[u8]) -> Result<JpegSofInfo, JpegError>;
pub fn parse_dri(payload: &[u8]) -> Result<Option<u16>, JpegError>;
pub fn find_scan_ranges(input: &[u8]) -> Result<JpegScanRanges, JpegError>;
pub fn rewrite_sof_dimensions(
    input: &[u8],
    dimensions: (u16, u16),
) -> Result<Vec<u8>, JpegError>;
```

The segment iterator yields borrowed marker records with offsets and payload
ranges. It must distinguish header segments from entropy data so callers do not
mistake stuffed `0xff00` bytes or restart markers for standalone headers.

```rust
pub struct JpegSegment<'a> {
    pub marker: u8,
    pub marker_offset: usize,
    pub payload_offset: usize,
    pub payload: &'a [u8],
}

pub struct JpegSofInfo {
    pub marker: u8,
    pub sof_kind: SofKind,
    pub bit_depth: u8,
    pub dimensions: (u16, u16),
    pub component_ids: Vec<u8>,
    pub sampling: SamplingFactors,
    pub quant_table_ids: Vec<u8>,
}

pub struct JpegScanRanges {
    pub sos_marker_offset: usize,
    pub sos_payload_range: core::ops::Range<usize>,
    pub entropy_range: core::ops::Range<usize>,
    pub eoi_marker_offset: Option<usize>,
}
```

These APIs are deliberately marker-level, not decoder-planner-level. They expose
facts needed by WSI container code while preserving the existing `Decoder` and
`JpegView` decode APIs.

## 5. TIFF tile preparation API

Add a preparation module to `signinum-jpeg`:

```rust
pub fn prepare_tiff_jpeg_tile<'a>(
    tile: &'a [u8],
    tables: Option<&'a [u8]>,
    opts: JpegTilePrepareOptions,
) -> Result<PreparedJpeg<'a>, JpegError>;

pub struct JpegTilePrepareOptions {
    pub expected_dimensions: Option<(u16, u16)>,
    pub duplicate_table_policy: DuplicateTablePolicy,
    pub repair_zero_sof_dimensions: bool,
    pub validate_restart_markers: bool,
}

pub enum DuplicateTablePolicy {
    AllowIdentical,
    RejectConflicting,
}

pub enum PreparedJpeg<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}
```

`PreparedJpeg` exposes `as_bytes()` and implements `AsRef<[u8]>`. If the input
tile is already a complete JPEG interchange stream and no repair is needed,
preparation returns `PreparedJpeg::Borrowed(tile)`.

If the tile is abbreviated and `JPEGTables` are provided, preparation assembles:

```text
SOI + normalized table segments + tile scan/header segments + EOI
```

The assembler strips SOI/EOI from `JPEGTables`, avoids duplicating table
segments already present in the tile, preserves DRI where semantically valid,
and rejects conflicting duplicate DQT/DHT definitions by default.

## 6. NDPI zero-SOF repair

NDPI can carry JPEG SOF dimensions as zero while TIFF metadata contains the real
tile dimensions. Signinum should repair this only when both conditions are true:

- `repair_zero_sof_dimensions` is true.
- `expected_dimensions` is present and both dimensions are non-zero.

Without expected dimensions, zero SOF dimensions remain a `JpegError`. If a SOF
declares non-zero dimensions that conflict with `expected_dimensions`, the
preparer rejects the tile unless a later API explicitly adds a permissive
override mode. The first slice should stay strict.

## 7. Restart and DRI handling

Preparation preserves DRI segments and restart markers. With
`validate_restart_markers`, Signinum verifies that restart markers found in the
entropy range are legal `RST0..RST7` markers and appear in sequence when DRI is
non-zero. Validation should be lightweight and structural; entropy decoding
remains the decoder's job.

If both tables and tile bytes contain DRI, preparation accepts identical DRI
values and rejects conflicting values. A zero DRI is normalized to `None`, in
line with the existing parser.

## 8. Prepared JPEG RGB8 batch facade

After preparation is in place, add an ordered per-job batch decode API:

```rust
pub struct PreparedJpegTileJob<'i, 'o> {
    pub input: &'i PreparedJpeg<'i>,
    pub out: &'o mut [u8],
    pub stride: usize,
    pub options: DecodeOptions,
}

pub struct DecodedTile {
    pub dimensions: (u32, u32),
    pub decoded: Rect,
}

pub fn decode_prepared_jpeg_tiles_rgb8(
    jobs: &mut [PreparedJpegTileJob<'_, '_>],
) -> Vec<Result<DecodedTile, JpegError>>;
```

Unlike existing batch helpers that return the first failing tile as
`TileBatchError`, this API preserves input order and returns one result per job.
That matches WSI viewport behavior where one damaged tile should not erase the
diagnostics for every other tile in the batch.

The facade should use existing `DecoderContext`, `ScratchPool`, and batch worker
machinery internally. It should not introduce a hidden global queue.

## 9. Error handling

Add typed `JpegError` variants only where existing variants cannot express the
failure clearly. Expected additions:

- conflicting duplicate JPEG table definition
- expected dimensions missing for zero-SOF repair
- expected dimensions conflicting with non-zero SOF dimensions
- invalid TIFF JPEG assembly state, such as a scan without SOF after table
  assembly

Errors must include marker offsets when available.

## 10. Testing

Use behavior-focused integration tests in `crates/signinum-jpeg/tests`.

Required tests:

- A full JPEG tile returns `PreparedJpeg::Borrowed`.
- An abbreviated tile plus `JPEGTables` assembles into a decode-ready JPEG.
- `JPEGTables` with SOI/EOI are normalized.
- Identical duplicate DQT/DHT tables are accepted under `AllowIdentical`.
- Conflicting duplicate DQT/DHT tables are rejected.
- DRI survives table assembly.
- Conflicting DRI values are rejected.
- Zero-SOF dimensions are rejected without expected dimensions.
- Zero-SOF dimensions are repaired when expected dimensions are supplied.
- Non-zero SOF dimensions conflicting with expected dimensions are rejected.
- Prepared batch decode returns ordered per-tile `Result`s and keeps successful
  tiles successful when another tile fails.

## 11. Migration path for Statumen

Statumen should replace TIFF/NDPI JPEG scanner code with:

1. Read tile bytes and optional `JPEGTables`.
2. Build `JpegTilePrepareOptions` from TIFF metadata.
3. Call `prepare_tiff_jpeg_tile`.
4. Pass `PreparedJpeg::as_bytes()` to existing decode paths, or use
   `decode_prepared_jpeg_tiles_rgb8` for viewport batches.

After this lands, Statumen should retain only container parsing and public
Signinum API adaptation.
