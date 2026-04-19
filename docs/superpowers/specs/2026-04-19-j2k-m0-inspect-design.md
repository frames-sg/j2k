# J2K-M0 — `slidecodec-j2k` Skeleton + Inspect

Status: approved implementation spec derived from the umbrella design.

## Goal

Add a new `slidecodec-j2k` crate that can inspect JPEG 2000 inputs and report
 core `slidecodec_core::Info` without decoding pixels yet.

M0 scope is strictly:

- raw codestream (`.j2k` / `.j2c`) inspect
- JP2 container (`.jp2`) inspect
- CLI magic-byte dispatch for `slidecodec inspect`
- parser robustness tests plus a parse-fuzz scaffold

Out of scope for M0:

- pixel decode
- `ImageDecode<'a>` trait impls
- ROI / scaled decode
- HTJ2K
- OpenJPEG parity or ISO decode conformance

## Supported Inputs

Two entry forms are supported:

1. Raw codestream starting with `SOC` (`0xFF4F`)
2. JP2 container starting with the standard signature box

Anything else returns a typed `J2kError`.

## Public API

`slidecodec-j2k` exposes:

- `J2kDecoder::inspect(&[u8]) -> Result<slidecodec_core::Info, J2kError>`
- `J2kView::parse(&[u8]) -> Result<J2kView<'_>, J2kError>`
- `J2kView::info(&self) -> &slidecodec_core::Info`
- `J2kDecoder::from_view(J2kView<'_>) -> Result<J2kDecoder<'_>, J2kError>`
- `J2kDecoder::info(&self) -> &slidecodec_core::Info`

`J2kDecoder` is a future decode-ready shell, but in M0 it only carries parsed
 inspect metadata.

## Parsing Model

### Raw Codestream

Parse codestream markers from `SOC` through the main header until the first
 tile-part boundary (`SOT`) or `EOC`.

Required marker support in M0:

- `SOC`
- `SIZ`
- `COD`
- length-prefixed skip for other main-header markers (`COC`, `QCD`, `QCC`,
  `COM`, `RGN`, `POC`, `TLM`, `PLM`, `PPM`, `CRG`, unknown reserved markers)
- `SOT` and `EOC` as termination markers for inspect

Extract:

- image dimensions from `SIZ`
- tile layout from `SIZ`
- component count and per-component precision from `SIZ`
- resolution levels from `COD` (`decomposition_levels + 1`)
- color transform hint from `COD` MCT flag when no container colorspace exists

### JP2 Container

Parse top-level boxes sequentially:

- signature box (`jP  `) required
- file type box (`ftyp`) required
- JP2 header superbox (`jp2h`) required
- contiguous codestream box (`jp2c`) required

Inside `jp2h`, parse:

- image header box (`ihdr`) required
- colour specification box (`colr`) optional but used when present

The codestream inside `jp2c` is then inspected using the raw codestream parser.

## `Info` Mapping

Populate `slidecodec_core::Info` as follows:

- `dimensions`: `SIZ.(Xsiz - XOsiz, Ysiz - YOsiz)`
- `components`: `Csiz`
- `bit_depth`: max component precision from `SIZ`
- `tile_layout`: `Some(TileLayout)` when `XTsiz`/`YTsiz` are valid
- `resolution_levels`: `COD.decomposition_levels + 1`, default `1` if `COD`
  absent
- `colorspace`:
  - `colr enum 16` -> `Colorspace::SRgb`
  - `colr enum 17` -> `Colorspace::SGray`
  - `colr ICC` -> `Colorspace::IccTagged`
  - raw codestream or missing `colr`, `components == 1` -> `Colorspace::SGray`
  - raw codestream or missing `colr`, `components == 3` and MCT off ->
    `Colorspace::Rgb`
  - raw codestream or missing `colr`, `components == 3` and MCT on ->
    `Colorspace::Ict`
  - all other cases -> `Colorspace::IccTagged`

If component precisions differ, `bit_depth` is the maximum precision observed.

## Error Model

`J2kError` composes `slidecodec_core` sub-errors where appropriate and adds
 format-specific variants:

- `Input(InputError)`
- `Unsupported(Unsupported)`
- `InvalidBox`
- `InvalidMarker`
- `MissingRequiredBox`
- `MissingRequiredMarker`
- `DimensionOverflow`
- `InvalidSiz`
- `InvalidCod`

Inspect must never panic on malformed input.

## CLI Integration

`slidecodec inspect <file>` should:

- detect JP2 by signature box bytes
- detect raw codestream by leading `SOC`
- otherwise fall back to JPEG inspect

For J2K, print the core `Info` fields that exist today:

- dimensions
- colorspace
- bit depth
- components
- resolution levels
- tile layout

This is a distinct J2K output shape, not the existing JPEG-only inspect line.

## Tests

Required J2K-M0 tests:

- unit tests for box parsing and codestream marker parsing
- integration tests covering:
  - minimal raw codestream inspect
  - minimal JP2 inspect
  - bad signature / missing boxes / missing `SIZ`
  - CLI magic-byte dispatch to J2K
- proptest parser robustness with arbitrary byte slices

Fixtures may be inline synthetic byte streams in tests; no external corpus is
 required for M0.

## Fuzz

Add `crates/slidecodec-j2k/fuzz` with a `parse_fuzz` target that only calls
 `J2kDecoder::inspect`.

## Verification

J2K-M0 is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo bench -p slidecodec-jpeg --bench compare --no-run`
- `cargo deny check`

all pass, and the CLI can inspect both JPEG and J2K inputs without ambiguity.
