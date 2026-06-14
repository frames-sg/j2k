# signinum-tilecodec

Tile decompression helpers for Signinum.

Supported codecs include Deflate, Zstd, LZW, and uncompressed copy paths.
Shared bounded-read and scratch-pool helpers keep errors explicit and avoid
unbounded temporary allocation.
