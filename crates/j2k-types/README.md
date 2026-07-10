# j2k-types

Shared JPEG 2000 and HTJ2K encode-stage contracts and helpers for the j2k
workspace.

This crate is the neutral public contract between the `j2k` facade, the
`j2k-native` codec engine, and device adapters. It defines encode-stage jobs,
outputs, and dispatch reports, plus progression-order encoding and packet
descriptor sorting. It also owns the shared encode-stage accelerator trait and
its default CPU-only implementation, so participating crates do not mirror
those contracts.

## Links

- API docs: <https://docs.rs/j2k-types>
- Repository: <https://github.com/frames-sg/j2k>
- Support policy: <https://github.com/frames-sg/j2k/blob/main/docs/public-support.md>
