# J2K Metal Support

> Metal runtime helpers shared by [J2K](https://github.com/frames-sg/j2k)'s Metal crates

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-metal-support.svg)](https://crates.io/crates/j2k-metal-support)
[![Documentation](https://docs.rs/j2k-metal-support/badge.svg)](https://docs.rs/j2k-metal-support)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares Metal setup code between J2K's Metal crates; use
[`j2k-metal`](https://crates.io/crates/j2k-metal) or
[`j2k-jpeg-metal`](https://crates.io/crates/j2k-jpeg-metal) instead. It looks
up the system device, creates buffers, textures, and command objects with
checks for failed allocations, compiles shader libraries, and loads pipelines.
It uses [`objc2-metal`](https://crates.io/crates/objc2-metal) types, not
`metal-rs`. The Metal parts build only on macOS.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
