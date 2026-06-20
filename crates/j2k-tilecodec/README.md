# j2k-tilecodec

Tile decompression helpers for J2K.

Supported codecs include Deflate, Zstd, LZW, and uncompressed copy paths.
Shared bounded-read and scratch-pool helpers keep errors explicit and avoid
unbounded temporary allocation.
