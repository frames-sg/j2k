# Parity Strategy

`signinum` keeps parity checks close to the codec surface instead of relying
on a single visual smoke test.

## JPEG

- Primary conformance fixtures live in `corpus/conformance/manifest.json` and
  compare decoded bytes against libjpeg-turbo-generated raw outputs.
- WSI-shaped fixtures and policy checks live in the `signinum-jpeg` test and
  bench suites.
- Tolerance is bit-exact for the committed baseline fixtures. Any future lossy
  tolerance must be recorded per fixture in the manifest.
- New JPEG support classes follow the CPU-first phase plan in
  [`docs/jpeg-support-phases`](jpeg-support-phases/README.md). Initial
  sequential CMYK/YCCK `Rgb8` full/ROI/scaled/region-scaled plus `Rgba8`
  full/ROI CPU coverage, RGB row streaming, and progressive 8-bit
  ROI/scaled/region-scaled CPU coverage have landed. Initial
  full-image/ROI/scaled/region-scaled 12-bit extended sequential grayscale
  `Gray16`/`Rgb16` coverage, including restart-coded grayscale streams, initial
  12-bit progressive grayscale `Gray16`/`Rgb16` coverage, and initial 12-bit
  APP14 RGB 4:4:4 and YCbCr 4:4:4/4:2:2/4:2:0 `Rgb16` coverage, including
  restart-coded color streams, have landed. Initial lossless SOF3 8-bit
  grayscale/RGB/YCbCr row streaming, 16-bit grayscale row streaming, 16-bit
  APP14 RGB `Rgb16` output/row streaming, and 8-bit plus 16-bit YCbCr 4:4:4
  `Rgb8`/`Rgb16` output/row streaming have also landed, but expanded
  four-component subsampled/malformed fixtures, other 12-bit subsampled color
  output, stronger non-constant 12-bit oracle fixtures, and remaining lossless
  SOF3 16-bit color layouts must land CPU parity fixtures and reference
  outputs before any Metal route is promoted.
- A/B/C fixture entries must record the oracle source and version, output
  pixel format, and accepted tolerance. If libjpeg-turbo does not support a
  class, the alternative oracle must be recorded with the exact command used.
- JPEG Metal and CUDA adapter parity must compare resident outputs against the
  CPU oracle for the same JPEG class. CPU-staged upload paths do not count as
  resident decode parity.

## JPEG 2000 / HTJ2K

- CPU parity tests compare generated codestreams against the in-repo native
  engine and, where available, OpenJPEG/Grok comparator paths.
- ROI, scaled, combined ROI+scaled, row, and tile-batch surfaces are tested as
  API behavior, not only as full-frame decode.
- J2K Metal and CUDA-named adapter crates must preserve CPU parity for fallback
  host surfaces. Metal crates must preserve decoded bytes for explicit
  Metal-backed ROI+scaled surfaces.

## Maintenance Rules

- Every committed conformance input must be listed in the matching manifest.
- Fixture generation scripts are maintainer tools; CI reads committed fixtures
  and does not regenerate them.
- New codec behavior needs at least one focused parity or regression test
  before benchmark numbers are updated.
