# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Documented that J2K Metal resident encode diagnostic report structs are
  crate-constructed API surfaces that may grow diagnostic fields in adapter
  releases, and clarified that resident RCA split buckets measure host-side
  Metal command encoding rather than GPU kernel elapsed time.

## [0.4.4] - 2026-05-26

### Added

- Added adoption-facing support matrix docs, facade-first examples, and a
  transcode example for new users.
- Added stable library semver CI and docs.rs metadata for publishable crates.
- Added J2K benchmark-publication signoff gates for OpenJPEG and Grok
  comparator runs.

### Changed

- Lowered the workspace MSRV to Rust 1.88 after auditing Rust 1.85, 1.88,
  1.90, 1.92, 1.93, and 1.94.
- Tightened J2K compare benchmarks to print comparator availability, version,
  and path; align batch worker counts; and include explicit signinum serial
  rows.

## [0.4.3] - 2026-05-25

### Fixed

- Refreshed release-gate metadata and package-integrity checks for the current
  facade release staging version.
- Kept transcode test and benchmark fixtures inside their crate packages so
  packaged tests and benches remain runnable outside the workspace checkout.

## [0.4.2] - 2026-05-15

### Changed

- Refactored shared core, profiling, JPEG adapter, and J2K adapter internals
  toward focused crate-owned contracts without intended behavior changes.
- Split CUDA adapter implementations into focused modules while preserving
  their public exports.
- Moved repeated benchmark and reference fixture generators into the dev-only
  `signinum-test-support` crate.

## [0.4.1] - 2026-05-12

### Fixed

- Removed the unsupported Intel macOS CI gate from the public release matrix.
- Prevented CI tests from executing benchmark binaries as tests.
- Stabilized hosted macOS Metal CI by keeping J2K Metal runtime validation on
  self-hosted GPU runners.

## [0.4.0] - 2026-05-12

### Changed

- Refreshed crates.io package metadata for the transfer to the
  `frames-sg/signinum` repository.

## [1.0.3] - 2026-05-06

### Changed

- Updated the `signinum` facade Metal feature to depend on
  `signinum-jpeg-metal` 0.2.2 for resident fast 4:4:4 JPEG Metal decode
  outputs.

## [0.2.2] - 2026-05-06

### Fixed

- Marked fast 4:4:4 Metal JPEG decode outputs as Metal-resident instead of
  CPU-staged Metal uploads, allowing strict device-decode consumers to use the
  resident buffer path end to end.

## [1.0.0] - 2026-05-01

CPU-first 1.0 release posture.

### Changed

- Promoted `signinum-core`, `signinum-jpeg`, `signinum-j2k`, `signinum-tilecodec`,
  and `signinum-cli` to the stable CPU-first 1.0 release set.
- Kept `signinum-j2k-native` as a published pre-1.0 implementation dependency
  for `signinum-j2k`.
- Excluded Metal, CUDA, and comparator crates from the 1.0 publish workflow.
- Clarified that CUDA crates can use `cuda-runtime` to return CUDA device memory
  surfaces, with `signinum-jpeg-cuda` using nvJPEG for full-frame RGB8 JPEG
  decode when the CUDA driver and `libnvjpeg` are available. NVIDIA performance
  claims require recorded self-hosted GPU benchmark evidence.

## [0.1.0] - 2026-04-25

Initial public-source checkpoint. The workspace remains pre-1.0 while the
JPEG 2000 / HTJ2K ROI and GPU adapter APIs settle.

### Added

- `signinum-core` shared trait/type crate:
  - `ImageDecode`, `ImageDecodeRows`, `TileBatchDecode`, `TileDecompress`
  - `PixelFormat`, `Downscale`, `Info`, `Rect`, `DecodeOutcome`
  - `ScratchPool` and `DecoderContext` contracts
- `signinum-jpeg` as the WSI-oriented JPEG implementation with:
  - borrowed parse/decode surfaces
  - row-streaming decode
  - region and scaled decode
  - tile-batch/context/scratch reuse
  - external-corpus and parity coverage
- `signinum-j2k` with:
  - JP2 / raw codestream inspect
  - full-frame, region, scaled, row-streaming, and tile-batch decode
  - HTJ2K coverage
  - OpenJPEG differential tests and compare bench
- `signinum-tilecodec` with:
  - `DeflateCodec`
  - `ZstdCodec`
  - `LzwCodec`
  - `UncompressedCodec`
  - typed scratch pools and compare bench coverage
- `signinum-cli` inspect dispatch for JPEG and JPEG 2000 inputs
- workspace-level CI coverage for tests, clippy, bench build, fuzz-target
  build, and `cargo deny`

### Changed

- Workspace version promoted to `0.1.0`
- Benchmark documentation now covers JPEG, JPEG 2000, and tile decompression
- Top-level README now documents the full pathology codec stack instead of the
  original JPEG-only scope
