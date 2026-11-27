# J2K Transcode

> JPEG to HTJ2K transcoding for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-transcode.svg)](https://crates.io/crates/j2k-transcode)
[![Documentation](https://docs.rs/j2k-transcode/badge.svg)](https://docs.rs/j2k-transcode)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Convert JPEG to HTJ2K from the JPEG's DCT coefficients, without decoding to
pixels first. `JpegToHtj2kOptions::lossless_53()`, the default, writes a
reversible 5/3 codestream, and `lossy_97()` an irreversible 9/7 one. Inputs
are 8-bit baseline and progressive JPEGs in grayscale, YCbCr, or RGB; other
JPEGs return an error. The components keep the JPEG's color space and
subsampling, and the output is a raw HTJ2K codestream.

This crate runs every stage on the CPU.
[`j2k-transcode-cuda`](https://crates.io/crates/j2k-transcode-cuda) and
[`j2k-transcode-metal`](https://crates.io/crates/j2k-transcode-metal) move the
transform stages to a GPU, and with the default precision all three produce
the same bytes. The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_transcode::{jpeg_to_htj2k, JpegToHtj2kOptions};

let jpeg = std::fs::read("tile.jpg")?;
let encoded = jpeg_to_htj2k(&jpeg, &JpegToHtj2kOptions::lossless_53())?;
std::fs::write("tile.j2c", &encoded.codestream)?;
```

`jpeg_to_htj2k_batch` converts many tiles in parallel. See the
[transcoding chapter](https://github.com/frames-sg/j2k/blob/main/book/transcoding.md)
of the J2K Book for batches, 9/7 precision, and GPU acceleration.

## Feature flags

- `dev-support`: internal, for J2K's own tests and benchmarks.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
