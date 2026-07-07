# j2k-types

Shared JPEG 2000 encode-stage contract types for the j2k workspace.

This crate is the neutral public contract between the `j2k` adapter
surface and the `j2k-native` codec engine: encode-stage job, output,
and report types are defined once here so neither crate mirrors the other's
definitions. It contains plain data types only — codec behavior lives in
`j2k-native`, and the encode-stage accelerator traits stay in their
owning crates.

## Links

- API docs: <https://docs.rs/j2k-types>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
