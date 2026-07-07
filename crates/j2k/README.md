# j2k

JPEG 2000 Part 1 and HTJ2K Part 15 public decoder/encoder crate for J2K.

This crate exposes inspect, decode, encode, recode, device-surface, and
encode-stage adapter contracts backed by the native J2K engine and optional
device adapters.

The primary adoption claim is a memory-safe Rust public API with CPU as the
portable correctness baseline. GPU-backed routes are optional and selected only
for supported, benchmark-backed shapes; single-frame HTJ2K host-output encode
stays CPU by default.

The public support boundary is raw J2K/J2C codestreams, JP2 still-image files,
raw HTJ2K codestreams, and JPH still-image files. JPX / JPEG 2000 Part 2
extensions are not part of this crate's support claim unless required for
standard JP2/JPH still-image correctness.

The encode-stage adapter module is a backend SPI for CUDA, Metal, and transcode
integration. It is not the primary end-user encode API.

For JPEG 2000 / HTJ2K application code, including CPU and supported GPU-backed
paths, use this crate directly.

## Decode strictness

`j2k_native::DecodeSettings::default()` remains lenient for compatibility.
Lenient mode may tolerate recoverable optional container metadata problems that
`DecodeSettings::strict()` rejects. Public `j2k` decode outcomes report
`J2kDecodeWarning::LenientDecodeMode` when the retained lenient default is used;
callers that need fail-closed validation should construct native images with
`DecodeSettings::strict()` or treat that warning as nonpublishable input.

## Links

- API docs: <https://docs.rs/j2k>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
