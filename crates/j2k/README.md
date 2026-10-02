# j2k — Pure-Rust JPEG 2000 and HTJ2K Codec

The main J2K crate: JPEG 2000 Part 1 and HTJ2K Part 15 on the CPU.

It provides inspection, decode, encode, lossless J2K-to-HTJ2K recode, JP2/JPH
container wrapping, and GPU-decode planning, all backed by the native J2K
engine.

The public API is safe Rust, and the CPU path is the reference implementation.
GPU paths are optional and are used only for shapes where they were measured to
be faster. Single-image HTJ2K encode to host memory runs on the CPU by default.

Supported inputs are raw J2K/J2C codestreams, JP2 files, raw HTJ2K
codestreams, and JPH files. JPX (JPEG 2000 Part 2) extensions are not
supported except where JP2/JPH decoding needs them.

The encode-stage types shared with the CUDA, Metal, and transcode adapters are
re-exported at the crate root but hidden from the rendered docs; they are not
the encode API for applications. GPU surfaces and runtimes live in the
`j2k-cuda` and `j2k-metal` crates.

For CPU JPEG 2000 / HTJ2K code, depend on this crate. For GPU decoding, also
depend on the adapter crate for your backend.

## Decode strictness

`j2k::DecodeSettings::default()` and `j2k_native::DecodeSettings::default()`
are strict. `J2kView::parse` and `J2kDecoder::new` use the default.

To accept some malformed files, pass `DecodeSettings::lenient()` to
`J2kView::parse_with_settings` or `J2kDecoder::new_with_settings`. Lenient mode
only relaxes a few metadata checks: it may ignore a malformed trailing
top-level box after the JP2/JPH header and codestream, ignore a malformed
trailing child box after `ihdr` and `colr`, ignore malformed optional `cdef` or
`pclr` metadata (keeping an earlier complete value), or infer one undeclared
alpha component. Codestream and entropy validation, bounds and overflow
checks, and allocation limits are the same as in strict mode.

A successful lenient decode reports `J2kDecodeWarning::LenientMetadataRecovery`
only when one of those recoveries was actually used.

## Links

- API docs: <https://docs.rs/j2k>
- [Pure-Rust JPEG 2000 codec documentation](https://frames-sg.github.io/j2k/rust-jpeg2000-codec/)
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
