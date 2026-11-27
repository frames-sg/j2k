# J2K JPEG

> Pure-Rust JPEG decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-jpeg.svg)](https://crates.io/crates/j2k-jpeg)
[![Documentation](https://docs.rs/j2k-jpeg/badge.svg)](https://docs.rs/j2k-jpeg)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Inspect, decode, and encode JPEG. The decoder's output matches
libjpeg-turbo's byte for byte. It handles baseline, extended, and progressive
JPEG at 8 and 12 bits and lossless (SOF3) JPEG at 2 to 16 bits, in grayscale,
YCbCr, RGB, CMYK, and YCCK. It decodes whole images, regions, and 1/2, 1/4,
or 1/8 scale, and `JpegBatchDecoder` decodes batches of 8-bit JPEGs into dense
RGB8 arrays. The encoders write 8-bit baseline JPEG with standard or optimized
Huffman tables, 12-bit extended JPEG, and lossless JPEG. There are also
helpers for TIFF tiles that share a `JPEGTables` header
(`prepare_tiff_jpeg_tile`) and for ICC profiles in APP2 segments.

Arithmetic coding, hierarchical JPEG, four-component lossless JPEG, and
sampling factors that don't divide the largest factor are not supported.
Decoding is strict by default: a file that needs recovery is an error unless
you turn strictness off with `DecodeOptions::with_strict(false)`. The crate
builds only for `x86_64` and `aarch64` targets.

For JPEG 2000, use [`j2k`](https://crates.io/crates/j2k). To convert JPEG to
HTJ2K without decoding to pixels, use
[`j2k-transcode`](https://crates.io/crates/j2k-transcode).
[`j2k-jpeg-cuda`](https://crates.io/crates/j2k-jpeg-cuda) and
[`j2k-jpeg-metal`](https://crates.io/crates/j2k-jpeg-metal) run JPEG on the
GPU.

## Usage

```rust
use j2k_jpeg::{DecodeRequest, Decoder, PixelFormat};

let bytes = std::fs::read("photo.jpg")?;
let decoder = Decoder::new(&bytes)?;
let (rgb, outcome) = decoder.decode_request(DecodeRequest::full(PixelFormat::Rgb8))?;
println!("{}x{}", outcome.decoded.w, outcome.decoded.h);
```

The [JPEG chapter](https://github.com/frames-sg/j2k/blob/main/book/jpeg.md) of
the J2K Book covers regions, scaled and batch decoding, strictness, color
spaces, and the encoders.

## Feature flags

- `scalar-only`: use the portable scalar code instead of SIMD, for fuzzing
  and differential testing.
- `bench-libjpeg-turbo` and `bench-internals` are for this crate's
  benchmarks.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
