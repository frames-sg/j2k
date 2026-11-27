# J2K Types

> Encode-stage and decode-plan types shared by the [J2K](https://github.com/frames-sg/j2k) crates

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-types.svg)](https://crates.io/crates/j2k-types)
[![Documentation](https://docs.rs/j2k-types/badge.svg)](https://docs.rs/j2k-types)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares types between `j2k`, `j2k-native`, and the GPU crates; use
[`j2k`](https://crates.io/crates/j2k) instead. `j2k` re-exports the ones
applications need, such as the wavelet types returned by
`J2kDecoder::decode_wavelets`. The rest are the jobs and dispatch reports of
the encode-stage accelerator trait, and the decode plans the GPU crates run.
It is `no_std`, has no dependencies, and contains no unsafe code.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
