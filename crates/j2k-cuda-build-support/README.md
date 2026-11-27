# J2K CUDA Build Support

> Build-script support for [J2K](https://github.com/frames-sg/j2k)'s CUDA kernels

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-cuda-build-support.svg)](https://crates.io/crates/j2k-cuda-build-support)
[![Documentation](https://docs.rs/j2k-cuda-build-support/badge.svg)](https://docs.rs/j2k-cuda-build-support)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares build-script code between J2K's CUDA engine crates; use
[`j2k-cuda`](https://crates.io/crates/j2k-cuda) instead. It stages each
CUDA-Oxide kernel project and, on Linux, compiles it to PTX with
`cargo oxide build`. On other hosts, or when the CUDA-Oxide toolchain is
missing, it writes placeholder PTX, and calling those kernels returns an
error. Set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` to make that a build error instead.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
