# J2K Metal

> Metal decode and encode for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-metal.svg)](https://crates.io/crates/j2k-metal)
[![Documentation](https://docs.rs/j2k-metal/badge.svg)](https://docs.rs/j2k-metal)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decode JPEG 2000 and HTJ2K on an Apple GPU, and encode lossless HTJ2K there.
`J2kDecoder` decodes whole images, regions, and reduced resolutions into Metal
buffers; `MetalBatchDecoder` decodes batches into NCHW or NHWC buffers that
can stay on the GPU for the next stage. Lossless encode can take its input
from Metal buffers and leave the codestream in one.

You choose the backend per call with `BackendRequest`. `Auto` uses Metal only
for the image types and sizes where it was measured to be faster than the CPU
and to give identical output, and runs everything else on the CPU. `Metal`
always uses Metal and returns an error for an image it can't handle; it never
falls back to the CPU.

Metal needs macOS. On other platforms the crate builds, and explicit Metal
requests return `Error::MetalUnavailable`. The expert APIs take
[`objc2-metal`](https://crates.io/crates/objc2-metal) types, not `metal-rs`.
The API is experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

```rust
use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_metal::J2kDecoder;

let bytes = std::fs::read("image.jph")?;
let mut decoder = J2kDecoder::new(&bytes)?;
let surface = decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)?;
println!("{:?} {:?}", surface.residency(), surface.dimensions());
```

The [GPU chapter](https://github.com/frames-sg/j2k/blob/main/book/gpu.md) of
the J2K Book covers batches, routing, encoding, and decoding into your own
Metal buffers.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
