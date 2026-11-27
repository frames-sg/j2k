# J2K CUDA JPEG Engine

> CUDA JPEG kernels for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda-jpeg-engine.svg)](https://crates.io/crates/j2k-cuda-jpeg-engine)
[![Documentation](https://docs.rs/j2k-cuda-jpeg-engine/badge.svg)](https://docs.rs/j2k-cuda-jpeg-engine)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate holds the CUDA JPEG kernels (baseline decode and encode, lossless
encode) and their host-side validation for the other J2K crates; use
[`j2k-jpeg-cuda`](https://crates.io/crates/j2k-jpeg-cuda) instead.

## Feature flags

The features are internal. Each `cuda-oxide-*` feature builds one kernel
project; `j2k-jpeg-cuda` enables them with its `cuda-runtime` feature.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
