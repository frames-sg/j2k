# j2k-metal-support

Shared Metal runtime setup helpers for J2K Metal adapters.

The crate centralizes system device lookup, nil-checked buffer/texture and
command-resource construction, checked buffer access, shader-library
compilation, named pipeline loading, and stable route labels. Autoreleased
command buffers and encoders are retained into owned Rust handles before they
leave the constructor boundary. Codec-specific kernels stay in the codec
adapter crates.

Version 0.9 exposes this ownership model directly. Owned expert values are
`objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTL...>>`; borrowed
values are `&ProtocolObject<dyn MTL...>`. Submission guards retain command
buffers, resources, and events until `wait` or blocking `Drop`, and exact
resource/queue matching uses pointer identity. There is no `metal-rs`
compatibility feature.

## Links

- API docs: <https://docs.rs/j2k-metal-support>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
