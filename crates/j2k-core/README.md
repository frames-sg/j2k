# j2k-core

Shared traits and types for the J2K codec crates.

This crate defines backend requests, pixel formats, decode outcomes, device
surfaces, scratch-pool traits, row output traits, tile-batch decode traits, and
the `BatchLayout` (NCHW or NHWC) shared by the JPEG and JPEG 2000 batch
decoders.
It does not implement a codec.

Use it when implementing or integrating J2K codec adapters.

## Links

- API docs: <https://docs.rs/j2k-core>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
