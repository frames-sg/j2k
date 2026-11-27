# J2K Codec Math

> Codec constants and helpers shared by [J2K](https://github.com/frames-sg/j2k)'s CPU, CUDA, and Metal code

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-codec-math.svg)](https://crates.io/crates/j2k-codec-math)
[![Documentation](https://docs.rs/j2k-codec-math/badge.svg)](https://docs.rs/j2k-codec-math)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares codec constants, tables, and small helper algorithms between
the other J2K crates; use [`j2k`](https://crates.io/crates/j2k) or
[`j2k-jpeg`](https://crates.io/crates/j2k-jpeg) instead. The CPU code, the
CUDA kernels, and the Metal shaders take their values from here, which keeps
the backends' output identical. It is `no_std`, does not allocate, and
contains no unsafe code.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
