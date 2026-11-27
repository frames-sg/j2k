# J2K CUDA Transcode Engine

> CUDA JPEG-to-HTJ2K transcode kernels for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda-transcode-engine.svg)](https://crates.io/crates/j2k-cuda-transcode-engine)
[![Documentation](https://docs.rs/j2k-cuda-transcode-engine/badge.svg)](https://docs.rs/j2k-cuda-transcode-engine)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate holds the CUDA kernels that turn JPEG DCT coefficients into
quantized wavelet coefficients for the other J2K crates; use
[`j2k-transcode-cuda`](https://crates.io/crates/j2k-transcode-cuda) instead.

## Feature flags

The features are internal. `j2k-transcode-cuda` enables `cuda-oxide-transcode`
with its `cuda-runtime` feature.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
