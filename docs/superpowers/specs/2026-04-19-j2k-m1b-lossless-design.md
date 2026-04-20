# J2K-M1b — Lossless (5-3 Reversible, RCT)

Status: approved implementation spec derived from the umbrella design.

## Goal

Make lossless JPEG 2000 an explicit, tested part of `slidecodec-j2k`:

- reversible 5-3 DWT decode
- reversible component transform (RCT) handling
- exact native-depth output for grayscale and RGB lossless codestreams

M1b builds directly on the committed J2K-M1 in-tree decode path.

## Scope

In scope:

- parser recognition of reversible transform state from `COD`
- correct `Colorspace` inference for raw 3-component codestreams using MCT:
  - irreversible + MCT -> `Colorspace::Ict`
  - reversible + MCT -> `Colorspace::Rct`
- exact-output tests for reversible grayscale and RGB decode

Out of scope:

- ROI
- scaled decode
- tile-batch context reuse
- HTJ2K public decode support

## Implementation

The decode path remains the same as M1. M1b does not add a new engine. It makes
the reversible path explicit and guarded.

Parser changes:

- extend `ParsedCod` with a `reversible` flag from the wavelet-transform field
- use that flag when inferring `Info.colorspace` for raw codestreams

Decode surface:

- no public API additions
- reversible grayscale and RGB codestreams must continue to decode through the
  existing `decode_into` surface

## Tests

Required tests:

- inspect test for a reversible raw codestream with MCT -> `Colorspace::Rct`
- exact reversible grayscale native-depth decode test
- exact reversible RGB native-depth decode test

Existing M1 decode tests already cover exact reversible output for grayscale
and RGB native-depth paths; M1b adds the missing inspect-level assertion.

## Verification

J2K-M1b is complete when:

- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

all pass with the reversible inspect/decode coverage in place.
