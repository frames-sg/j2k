# J2K Core

> Shared traits and types for [J2K](https://github.com/frames-sg/j2k)'s codecs

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-core.svg)](https://crates.io/crates/j2k-core)
[![Documentation](https://docs.rs/j2k-core/badge.svg)](https://docs.rs/j2k-core)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

The traits and types that J2K's codec crates share: `PixelFormat`, `Rect`,
`Downscale`, `BackendRequest`, `BatchLayout`, `DecodeOutcome`, and the decode
traits (`ImageDecode`, `ImageDecodeDevice`, `TileBatchDecode`,
`TileDecompress`). It implements no codec and is `no_std` with `alloc`.

Applications rarely need it directly: [`j2k`](https://crates.io/crates/j2k)
and [`j2k-jpeg`](https://crates.io/crates/j2k-jpeg) re-export the common
types. Depend on it to write code that works with any J2K decoder, or to call
the device-decode traits of the GPU crates.

## Usage

Decode a JPEG or a JPEG 2000 image with the same function:

```rust
use j2k_core::{ImageDecode, PixelFormat};

fn decode_rgb8<'a, D: ImageDecode<'a>>(bytes: &'a [u8]) -> Result<Vec<u8>, D::Error> {
    let (width, height) = D::inspect(bytes)?.dimensions;
    let stride = width as usize * 3;
    let mut pixels = vec![0; stride * height as usize];
    let mut decoder = D::from_view(D::parse(bytes)?)?;
    decoder.decode_into(&mut pixels, stride, PixelFormat::Rgb8)?;
    Ok(pixels)
}

let jpeg = decode_rgb8::<j2k_jpeg::Decoder>(&jpeg_bytes)?;
let jp2 = decode_rgb8::<j2k::J2kDecoder>(&jp2_bytes)?;
```

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
