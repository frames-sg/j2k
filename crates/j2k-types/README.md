# j2k-types

Encode-stage types and helpers shared by the J2K crates.

The `j2k` crate, the `j2k-native` engine, and the GPU adapters all use these
types. The crate defines encode-stage jobs, outputs, and dispatch reports,
progression-order encoding, and packet descriptor sorting, plus the
encode-stage accelerator trait and its default CPU-only implementation.

## Links

- API docs: <https://docs.rs/j2k-types>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
