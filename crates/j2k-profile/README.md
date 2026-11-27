# J2K Profile

> Profiling helpers shared by the [J2K](https://github.com/frames-sg/j2k) crates

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-profile.svg)](https://crates.io/crates/j2k-profile)
[![Documentation](https://docs.rs/j2k-profile/badge.svg)](https://docs.rs/j2k-profile)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares profiling code between the other J2K crates; use
[`j2k`](https://crates.io/crates/j2k) instead. It reads the `J2K_*` profiling
environment variables and formats the timing and GPU-route rows the codecs
print to stderr, and its API is hidden from the documentation. The variables
are listed in [Environment variables](https://github.com/frames-sg/j2k/blob/main/book/environment-variables.md).

## Feature flags

- `std` (default): print rows and read environment variables. Without it, the
  crate is `no_std` and does no I/O.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
