# j2k-metal-support

Shared Metal runtime setup helpers for J2K Metal adapters.

The crate centralizes system device lookup, nil-checked buffer/texture and
command-resource construction, checked buffer access, shader-library
compilation, named pipeline loading, and stable route labels. Autoreleased
command buffers and encoders are retained into owned Rust handles before they
leave the constructor boundary. Codec-specific kernels stay in the codec
adapter crates.

## Links

- API docs: <https://docs.rs/j2k-metal-support>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
