# j2k-tilecodec

Tile decompression helpers for J2K.

Supports Deflate, Zstd, LZW, and uncompressed tiles. Reads are bounded and
scratch buffers are pooled, so a malformed tile returns an error instead of
allocating without limit.

## Links

- API docs: <https://docs.rs/j2k-tilecodec>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
