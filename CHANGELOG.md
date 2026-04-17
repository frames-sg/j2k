# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Workspace, CI (fmt, clippy, test × stable/beta/MSRV × linux/macos, wasm32,
  cargo-deny), licensing, and module skeleton (M0).
- Public API surface for header parsing: `Decoder::inspect(bytes) -> Info`
  (M1a). Supports SOF0 baseline, SOF1 extended 8/12-bit, SOF2 progressive
  (headers only), SOF3 lossless (headers only). Rejects arithmetic-coded
  and hierarchical variants with `JpegError::UnsupportedSof`.
- Typed error and warning enums: `JpegError`, `Warning`, `MarkerKind`,
  `UnsupportedReason`, `HuffmanFailure`, `BuilderConflictReason`, `TableKind`.
- Property-based test suite (`proptest`, 4096 cases) and `cargo-fuzz`
  `parse_fuzz` target covering `Decoder::inspect`.
- `slidecodec inspect <file>` CLI subcommand.
- `inspect` example in `examples/`.

### Notes

- No decode APIs yet — those land in M1b.
- No SIMD yet — the only IDCT/color paths that exist are stubs for M1b.
