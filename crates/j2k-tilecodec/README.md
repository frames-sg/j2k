# J2K Tile Codec

> Deflate, Zstd, and LZW tile decompression for [J2K](https://github.com/frames-sg/j2k)

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-tilecodec.svg)](https://crates.io/crates/j2k-tilecodec)
[![Documentation](https://docs.rs/j2k-tilecodec/badge.svg)](https://docs.rs/j2k-tilecodec)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Decompress the tiles of TIFF-style image containers: `DeflateCodec` (zlib or
raw Deflate), `ZstdCodec`, `LzwCodec` (TIFF LZW), and `UncompressedCodec`.
Each one implements `j2k_core::TileDecompress` and writes into a buffer you
provide, so a malformed or oversized tile returns an error instead of
allocating without limit. Scratch memory lives in a reusable pool per codec.
Use [`j2k-jpeg`](https://crates.io/crates/j2k-jpeg) or
[`j2k`](https://crates.io/crates/j2k) for JPEG and JPEG 2000 tiles.

## Usage

```rust
use j2k_tilecodec::{DeflateCodec, DeflatePool, TileDecompress};

let mut pool = DeflatePool::new();
let mut tile = vec![0_u8; 256 * 256 * 3];
let written = DeflateCodec::decompress_into(&mut pool, &compressed, &mut tile)?;
```

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
