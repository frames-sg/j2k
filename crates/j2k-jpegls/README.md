# J2K JPEG-LS

> Pure-Rust JPEG-LS decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-jpegls.svg)](https://crates.io/crates/j2k-jpegls)
[![Documentation](https://docs.rs/j2k-jpegls/badge.svg)](https://docs.rs/j2k-jpegls)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Inspect, decode, and encode JPEG-LS (ISO/IEC 14495-1, ITU-T T.87)
codestreams, the format of DICOM's JPEG-LS lossless (1.2.840.10008.1.2.4.80)
and near-lossless (1.2.840.10008.1.2.4.81) transfer syntaxes. It handles 2 to
16 bits per sample, any number of components, all three interleave modes,
restart intervals, and preset coding parameters. Decoding returns samples
interleaved by pixel whatever the file's interleave mode. The encoder writes
lossless and near-lossless codestreams and by default checks its output by
decoding it again. For the same parameters, its output matches CharLS byte
for byte.

The crate reads and writes the codestream only; pass it the pixel data of a
DICOM file, not the file itself. Mapping tables, subsampled components,
point transforms, HP color transforms, and images larger than 65535 pixels on
a side are not supported and return an error. Decoding is strict: truncated
or malformed data is an error, never a partial image.

For JPEG, use [`j2k-jpeg`](https://crates.io/crates/j2k-jpeg); for JPEG 2000,
[`j2k`](https://crates.io/crates/j2k).

## Usage

```rust
use j2k_jpegls::{encode_jpegls, Decoder, JpegLsEncodeOptions, JpegLsSamples};

let bytes = std::fs::read("image.jls")?;
let decoder = Decoder::new(&bytes)?;
let (width, height) = decoder.info().dimensions;
let image = decoder.decode()?;

let samples = JpegLsSamples::U16 { data: &pixels, width, height, components: 1 };
let encoded = encode_jpegls(samples, &JpegLsEncodeOptions::new(12).with_near(2))?;
std::fs::write("near-lossless.jls", &encoded.data)?;
```

## Feature flags

`j2k-jpegls` has no optional features.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
