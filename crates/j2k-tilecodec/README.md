# J2K Tile Codec

> TIFF tile decompression (Deflate, Zstd, LZW, PackBits) and DICOM RLE Lossless for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-tilecodec.svg)](https://crates.io/crates/j2k-tilecodec)
[![Documentation](https://docs.rs/j2k-tilecodec/badge.svg)](https://docs.rs/j2k-tilecodec)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decompress the tiles of TIFF-style image containers: `DeflateCodec` (zlib or
raw Deflate), `ZstdCodec`, `LzwCodec` (TIFF LZW), `PackBitsCodec` (TIFF
PackBits), and `UncompressedCodec`. Each one implements
`j2k_core::TileDecompress` and writes into a buffer you provide, so a
malformed or oversized tile returns an error instead of allocating without
limit. Scratch memory lives in a reusable pool per codec.

`DicomRleCodec` decodes and encodes DICOM RLE Lossless frames (transfer
syntax 1.2.840.10008.1.2.5) with 8, 16, or 32 bits allocated and any number
of samples per pixel up to the format's 15 segments. You describe the frame
with a `DicomRleLayout` from the data set's attributes and choose
color-by-pixel or color-by-plane samples. Decoding is strict by default;
`decode_frame_lenient` also reads frames with extra segment data or nonzero
unused header offsets, as some encoders write them, and reports what it
accepted.

Use [`j2k-jpeg`](https://crates.io/crates/j2k-jpeg) or
[`j2k`](https://crates.io/crates/j2k) for JPEG and JPEG 2000 tiles.

## Usage

```rust
use j2k_tilecodec::{DeflateCodec, DeflatePool, TileDecompress};

let mut pool = DeflatePool::new();
let mut tile = vec![0_u8; 256 * 256 * 3];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut tile)?;
```

```rust
use j2k_tilecodec::{DicomRleCodec, DicomRleLayout, DicomRlePlanarConfiguration};

// Rows, Columns, Samples per Pixel, Bits Allocated.
let layout = DicomRleLayout::new(512, 512, 1, 16, DicomRlePlanarConfiguration::Interleaved)?;
let mut samples = vec![0_u8; layout.frame_len()];
DicomRleCodec::decode_frame(&rle_frame, layout, &mut samples)?;
```

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
