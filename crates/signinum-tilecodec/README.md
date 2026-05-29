# signinum-tilecodec

Tile decompression primitives for pathology image containers.

Install:

```sh
cargo add signinum-tilecodec
```

The stable `0.4.x` API provides `TileDecompress` implementations for Deflate,
Zstd, LZW, and Uncompressed payloads, with caller-owned scratch pools where a
codec benefits from reuse.
