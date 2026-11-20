# J2K

> Pure-Rust JPEG 2000 and HTJ2K codec

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k.svg)](https://crates.io/crates/j2k)
[![Documentation](https://docs.rs/j2k/badge.svg)](https://docs.rs/j2k)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

J2K reads and writes JPEG 2000 (Part 1) and HTJ2K (Part 15) images on the CPU.
It decodes whole images, regions, reduced resolutions, tiles, and batches,
returns wavelet coefficients for selected bands, encodes lossless and lossy
codestreams, recodes JPEG 2000 to lossless HTJ2K, and wraps codestreams in JP2
or JPH files. It reads raw J2K/J2C and HTJ2K codestreams and JP2 and JPH files.
JPEG 2000 Part 2 (JPX) extensions are not supported; the full list is in
[Format support](https://github.com/frames-sg/j2k/blob/main/book/format-support.md).
The API is safe Rust and is built to take untrusted files.

This is the crate most code needs. For GPU decoding, add
[`j2k-cuda`](https://crates.io/crates/j2k-cuda) or
[`j2k-metal`](https://crates.io/crates/j2k-metal). For regular JPEG, use
[`j2k-jpeg`](https://crates.io/crates/j2k-jpeg), and to convert JPEG to HTJ2K,
[`j2k-transcode`](https://crates.io/crates/j2k-transcode).

## Usage

Decode a file to 8-bit gray, RGB, or RGBA, depending on the image:

```rust
use j2k::J2kDecoder;

let bytes = std::fs::read("image.jp2")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let image = decoder.decode_srgb8()?;
println!("{width}x{height}, {:?}, {} bytes", image.layout(), image.data().len());
```

Encode 8-bit RGB pixels as lossless HTJ2K:

```rust
use j2k::{encode_j2k_lossless, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLosslessSamples};

let samples = J2kLosslessSamples::new(&rgb, width, height, 3, 8, false)?;
let options = J2kLosslessEncodeOptions::default()
    .with_block_coding_mode(J2kBlockCodingMode::HighThroughput);
let encoded = encode_j2k_lossless(samples, &options)?;
std::fs::write("image.j2c", &encoded.codestream)?;
```

Decoding is strict by default; `DecodeSettings::lenient()` accepts a few kinds
of malformed metadata. The [decoding](https://github.com/frames-sg/j2k/blob/main/book/decoding.md)
and [encoding](https://github.com/frames-sg/j2k/blob/main/book/encoding.md)
chapters of the J2K Book cover regions, batches, output sample conversion,
lossy rate and quality targets, and lenient decoding.

## Feature flags

`j2k` has no optional features. GPU support is in the `j2k-cuda` and
`j2k-metal` crates.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
