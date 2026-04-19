# J2K-M1 — Scalar Baseline Decode

Status: approved implementation spec derived from the umbrella design.

## Goal

Add full-frame JPEG 2000 decode to `slidecodec-j2k` behind the
`slidecodec-core` API surface:

- `ImageDecode<'a>` for `J2kDecoder<'a>`
- `decode_into` for `Rgb8`, `Rgba8`, `Gray8`, `Rgb16`, `Gray16`
- JP2 and raw J2K input support
- irreversible 9-7 + ICT support

M1 is correctness-first. ROI, tile-batch, row-streaming, and scaled decode land
in later milestones.

## Implementation Substrate

M1 uses the pure-Rust `dicom-toolkit-jpeg2000` crate as the execution engine.
This is an implementation choice, not a public API commitment.

Why this is the right M1 choice:

- it is pure Rust (`MIT OR Apache-2.0`)
- it already supports Part 1 decode with native bit-depth output
- it can later support HTJ2K through the pure-Rust `openjph-core` crate
- it gives `slidecodec-j2k` a working decode surface now, so later milestones
  can focus on WSI-native APIs and performance rather than basic correctness

`slidecodec-j2k` keeps its own inspect parser and `J2kView<'a>` shape. Decode is
performed by constructing the backend image from the original bytes and mapping
its output into `slidecodec-core` types.

## Scope

In scope:

- JP2 and raw codestream decode
- full-frame decode only
- 8-bit and 16-bit output formats
- alpha-preserving output for `Rgba8`
- typed `J2kError` composition with core buffer/input errors
- parser hardening discovered during the J2K-M0 review

Out of scope:

- ROI decode
- row streaming
- tile-batch reuse
- decode-time resolution reduction
- HTJ2K public decode support
- performance claims against OpenJPEG
- any custom in-tree MQ/EBCOT/DWT implementation

## Public API

`slidecodec-j2k` exposes:

- `J2kDecoder::inspect(&[u8]) -> Result<Info, J2kError>`
- `J2kView::parse(&[u8]) -> Result<J2kView<'_>, J2kError>`
- `J2kDecoder::new(&[u8]) -> Result<J2kDecoder<'_>, J2kError>`
- `J2kDecoder::from_view(J2kView<'_>) -> Result<J2kDecoder<'_>, J2kError>`
- `impl<'a> ImageCodec for J2kDecoder<'a>`
- `impl<'a> ImageDecode<'a> for J2kDecoder<'a>`

M1 does not implement `ImageDecodeRows` or `TileBatchDecode`.
`ImageDecode<'a>::decode_region_into` and `decode_scaled_into` are implemented
as explicit `NotImplemented` stubs so the core trait remains satisfied without
claiming M2 functionality early.

## Decoder Model

`J2kView<'a>` remains a lightweight borrowed parsed view:

- `bytes: &'a [u8]`
- `info: slidecodec_core::Info`
- typed J2K extras needed for later milestones

`J2kDecoder<'a>` remains borrowed and decode-ready:

- `bytes: &'a [u8]`
- `info: slidecodec_core::Info`

No long-lived backend decoder state is stored in M1. Decode calls construct a
backend image from `self.bytes` each time.

## Output Behavior

### Supported pixel formats

- `PixelFormat::Rgb8`
- `PixelFormat::Rgba8`
- `PixelFormat::Gray8`
- `PixelFormat::Rgb16`
- `PixelFormat::Gray16`

### Unsupported in M1

- `PixelFormat::Rgba16`

`Rgba16` returns `J2kError::Unsupported`.

### Color handling

The backend decode result is treated as authoritative for component transforms
and JP2 colorspace application.

Mapping rules:

- backend grayscale -> `Gray8` / `Gray16` direct
- backend RGB -> `Rgb8` / `Rgb16` direct
- backend RGB + alpha -> `Rgba8` direct
- backend RGB without alpha + requested `Rgba8` -> append opaque alpha `255`
- any other backend colorspace requested as `Gray*` or `Rgb*` without a clear
  mapping -> `J2kError::Unsupported`

### 16-bit output

`Rgb16` and `Gray16` are built from the backend native decode path:

- if source precision `<= 8`, scale to the full 16-bit range
  (`sample * 65535 / ((1 << bit_depth) - 1)`, which is `sample * 257` for
  8-bit input)
- if source precision `> 8`, preserve the backend sample value and write it as
  little-endian `u16`

## Buffer Validation

Before decode:

- reject unsupported `PixelFormat`
- compute required row bytes from `Info.dimensions` and `PixelFormat`
- validate `stride`
- validate output length

These errors must surface as `J2kError::Buffer(BufferError::...)`.

## Error Model

Extend `J2kError` with:

- `Buffer(BufferError)`
- `Backend(String)` for backend decode failures that do not map cleanly to an
  existing structured J2K error

Existing `Input`, `Unsupported`, and parser-specific variants remain.

`CodecError` classification:

- truncated parser failures -> `is_truncated() == true`
- unsupported pixel formats / unsupported backend color mappings ->
  `is_unsupported() == true`
- buffer validation failures -> `is_buffer_error() == true`

## Parser Hardening

M1 also fixes the known permissiveness in the M0 inspector:

- raw codestream inspect requires `COD`
- raw codestream inspect must terminate on `SOT`, `SOD`, or `EOC`; plain EOF is
  not accepted as a valid complete codestream header
- JP2 inspect enforces sane ordering for required boxes: `jP  `, `ftyp`,
  `jp2h`, `jp2c`

These parser fixes land before decode work so the decode surface is not built on
accepting malformed headers as valid.

## Tests

Required tests:

- parser regressions for:
  - missing `COD`
  - EOF after main header
  - out-of-order required JP2 boxes
- full-frame decode integration tests using synthetic codestreams/containers
  generated from the backend encoder:
  - 8-bit RGB irreversible sample -> `Rgb8`
  - 8-bit RGB irreversible sample -> `Rgba8`
  - 8-bit grayscale irreversible sample -> `Gray8`
  - 16-bit grayscale reversible sample -> `Gray16`
  - 16-bit RGB reversible sample -> `Rgb16`
- output-buffer validation tests:
  - stride too small
  - output too small
  - unsupported format (`Rgba16`)
- trait-surface tests:
  - `J2kDecoder::parse/from_view/info`
  - `ImageDecode<'a>::inspect/parse/from_view/decode_into`

Fixtures may be generated inline in tests using the backend encoder. M1 does not
require committed binary J2K fixtures yet.

## Fuzz

Extend the `slidecodec-j2k` fuzz crate with `decode_fuzz` that:

- attempts `J2kDecoder::new`
- if construction succeeds, attempts a bounded `decode_into` to a validated
  output buffer for one supported format

Milestone completion still inherits the umbrella hardening gate: once the decode
surface is in place, the crate must be able to survive the longer fuzz/proptest
signoff runs recorded for J2K in the umbrella plan.

## Verification

J2K-M1 is complete when:

- `cargo fmt --all --check`
- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo deny check`

all pass, and `slidecodec-j2k` can decode both JP2 and raw J2K into the
supported M1 pixel formats.
