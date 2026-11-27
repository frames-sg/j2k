# J2K CUDA JPEG 2000 Engine

> CUDA JPEG 2000 and HTJ2K kernels for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda-j2k-engine.svg)](https://crates.io/crates/j2k-cuda-j2k-engine)
[![Documentation](https://docs.rs/j2k-cuda-j2k-engine/badge.svg)](https://docs.rs/j2k-cuda-j2k-engine)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate holds the CUDA JPEG 2000 and HTJ2K kernels and their host-side
validation for the other J2K crates; use
[`j2k-cuda`](https://crates.io/crates/j2k-cuda) instead.

## Feature flags

The features are internal. Each `cuda-oxide-*` feature builds one kernel
project; `j2k-cuda` enables them with its `cuda-runtime` feature.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
