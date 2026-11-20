# J2K CUDA Runtime

> CUDA Driver API runtime for [J2K](https://github.com/frames-sg/j2k)'s CUDA crates

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda-runtime.svg)](https://crates.io/crates/j2k-cuda-runtime)
[![Documentation](https://docs.rs/j2k-cuda-runtime/badge.svg)](https://docs.rs/j2k-cuda-runtime)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares CUDA runtime code between J2K's CUDA crates; use
[`j2k-cuda`](https://crates.io/crates/j2k-cuda),
[`j2k-jpeg-cuda`](https://crates.io/crates/j2k-jpeg-cuda), or
[`j2k-transcode-cuda`](https://crates.io/crates/j2k-transcode-cuda) instead.
It loads the NVIDIA driver at run time and manages device memory, streams,
events, PTX modules, and kernel launches. Launch geometry is checked before it
reaches the driver, and a pooled buffer is not reused until the GPU work that
used it has finished. The codec kernels are in the engine crates.

## Feature flags

The features are internal. The J2K CUDA crates enable the ones they need.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
