# Changelog

Changes for the next release. Notes for published releases are on
[GitHub Releases](https://github.com/frames-sg/j2k/releases).

To add an entry, see [the changelog guide](contributor-book/documentation.md#changelog).

## Unreleased

### Decoding

- A final tile part with `Psot = 0` now ends at the EOC marker, as T.800
  specifies. Before, the decoder read EOC and any bytes after it, such as the
  padding at the end of a DICOM frame, as packet data.
- **Output change:** lossless (reversible 5/3) decodes on the CPU, CUDA and
  Metal return whole numbers where a resolution is one sample wide or tall at
  an odd position, which can happen in images with a nonzero origin. That
  sample was halved to a fraction such as 169.5, which each decode route
  rounded its own way, so pixels could be one code value off.
  ([#332](https://github.com/frames-sg/j2k/pull/332),
  [#333](https://github.com/frames-sg/j2k/issues/333))
- Codestreams that are cut off, such as the bytes a progressive viewer has
  fetched so far, can be decoded with
  `DecodeSettings::with_partial_codestream(true)`.
  `J2kDecoder::codestream_completeness` reports the smallest reduction whose
  data is all present, and a decode at that reduction or deeper returns the
  same samples as the complete file. A decode that needs missing data fails
  with the new `J2kError::IncompleteCodestream`. Raw codestreams and JP2/JPH
  files cut inside their codestream box are accepted; decoding runs on the CPU.
- **Breaking:** `j2k_native::DecodeSettings` has a new public field,
  `partial_codestream`. Code that builds the struct with a literal must set
  it, or start from `DecodeSettings::strict()` or `DecodeSettings::lenient()`.

### JPEG-LS

- New crate `j2k-jpegls` decodes and encodes JPEG-LS (ISO/IEC 14495-1)
  codestreams, lossless and near-lossless, as used by DICOM: 2 to 16 bits,
  all interleave modes, restart intervals and preset coding parameters. Its
  output matches CharLS for the same parameters; mapping tables, subsampling
  and HP color transforms return an error.

### Tile codecs

- `j2k-tilecodec` decompresses TIFF PackBits tiles (`PackBitsCodec`), and
  decodes and encodes DICOM RLE Lossless frames (`DicomRleCodec`) at 8, 16 and
  32 bits allocated. Decoding is strict by default; `decode_frame_lenient`
  also reads frames with extra data after a segment or nonzero unused header
  offsets, as some encoders write them, and reports what it accepted.

### GPU

- **Behavior change:** `CudaLosslessEncoder::encode` with
  `EncodeBackendPreference::Auto` now uses CUDA only for image shapes where
  CUDA was measured to be faster than the CPU, the same way decode routing
  works. No shape qualifies yet, so `Auto` encodes on the CPU: CUDA-assisted
  lossless encode was slower than the CPU in every case we measured. Use
  `EncodeBackendPreference::RequireDevice` to force CUDA.

### Python

- `Decoder.decode` emits `RecoveredImageWarning` when a lenient JPEG 2000
  decode recovers from malformed metadata, as `decode_srgb8` already did.

### C and C++

- Build `libj2k_c.a` beside the shared library for consumers that link the C
  ABI statically. Binary packages ship the static archive as `lib/libj2k.a`
  alongside the versioned shared library, and the CMake package exports
  `J2K::j2k_static` with `J2K_STATIC` and the archive's private link
  libraries.
- Keep a decoder across codestreams with `j2k_decoder_feed`, which keeps its
  threads, settings, and limits and replaces the input only after the new
  headers pass inspection. Report the compressed transfer syntax with
  `j2k_decoder_transfer_syntax` (`J2K_SYNTAX_*`: `J2K_SYNTAX_UNKNOWN` = 0,
  `J2K_SYNTAX_JPEG2000_LOSSLESS` = 1, `J2K_SYNTAX_JPEG2000_LOSSY` = 2,
  `J2K_SYNTAX_HT_JPEG2000_LOSSLESS` = 3, `J2K_SYNTAX_HT_JPEG2000_LOSSY` = 4,
  or `J2K_SYNTAX_OTHER` = 5); callers must check the status before interpreting
  a `J2K_SYNTAX_*` output. C++ adds `decoder::feed` and
  `decoder::transfer_syntax`.
- Decode a full image through the core whole-image decoder with
  `j2k_decoder_output_info_native` and `j2k_decoder_decode_full_into`, which
  skip the batch engine's per-image planning for raw one- or
  three-component unsigned 8- to 16-bit images without uniform component
  subsampling (mixed sampling stays supported). The output samples and stride
  contract match `j2k_decoder_decode_into`. C++ adds
  `decoder::output_info_native` and `decoder::decode_full_into`.

### Build and licensing

- The minimum supported Rust version is 1.97, down from 1.99. `j2k-types` and
  `j2k-codec-math` still build with Rust 1.96.
- Python wheels and command-line builds use libz-sys 1.1.33, whose bundled
  zlib fixes RUSTSEC-2026-0544. j2k calls only zlib's `inflate` and
  `deflate`, not the affected `gzprintf` and `gzwrite`.

- The Metal and MPSGraph crates no longer require one exact version of
  `objc2`, `objc2-foundation`, `objc2-metal`, the Metal Performance Shaders
  crates, and `block2`. Any patch release in the same minor line works, so an
  application can share them with J2K.

### Documentation

- The guides moved out of `docs/` into the [J2K Book](book/README.md), and
  contributor material moved into the
  [contributor book](contributor-book/README.md).
