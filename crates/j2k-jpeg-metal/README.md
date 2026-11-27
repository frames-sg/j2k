# J2K JPEG Metal

> Metal JPEG decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-jpeg-metal.svg)](https://crates.io/crates/j2k-jpeg-metal)
[![Documentation](https://docs.rs/j2k-jpeg-metal/badge.svg)](https://docs.rs/j2k-jpeg-metal)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decode JPEG into Metal memory on an Apple GPU, and encode baseline and
lossless JPEG from Metal buffers. An explicit `BackendRequest::Metal` decode
accepts every JPEG and request the CPU decoder in
[`j2k-jpeg`](https://crates.io/crates/j2k-jpeg) accepts (whole images, regions,
and 1/2, 1/4, or 1/8 scale) and returns the same bytes. Anything else is an
error before any Metal work; there is no CPU fallback. Baseline encode runs
entirely on the GPU. Lossless encode writes the same bytes as
`j2k_jpeg::encode_jpeg_lossless`.

Sequential 8-bit JPEGs in YCbCr 4:2:0, 4:2:2, or 4:4:4, or RGB 4:4:4, decode
entirely on the GPU. Other JPEGs are entropy-decoded on the CPU and finished
on the GPU, which runs at about CPU speed for a single image and slower for
lossless JPEG; that pays off when the pixels are used on the GPU. `Auto` keeps
single images on the CPU and uses Metal only for large batches of same-shape
4:2:0 or 4:2:2 tiles.

Metal needs macOS. On other platforms the crate builds, and Metal requests
return `Error::MetalUnavailable`. The buffer and texture APIs take
[`objc2-metal`](https://crates.io/crates/objc2-metal) types, not `metal-rs`.
The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_jpeg_metal::Decoder;

let bytes = std::fs::read("photo.jpg")?;
let mut decoder = Decoder::new(&bytes)?;
let surface = decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)?;
println!("{:?} {:?}", surface.residency(), surface.dimensions());
```

The [GPU JPEG](https://github.com/frames-sg/j2k/blob/main/book/jpeg.md#gpu-jpeg)
section of the J2K Book lists which JPEG types run where, and covers batches
and encode.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
