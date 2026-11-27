# J2K MPSGraph

> Run MPSGraph on [J2K](https://github.com/frames-sg/j2k)'s Metal-decoded batches

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-mpsgraph.svg)](https://crates.io/crates/j2k-mpsgraph)
[![Documentation](https://docs.rs/j2k-mpsgraph/badge.svg)](https://docs.rs/j2k-mpsgraph)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

Hand batches of JPEG 2000 and HTJ2K images decoded by
[`j2k-metal`](https://crates.io/crates/j2k-metal) to Apple's MPSGraph without
copying the pixels back to the CPU. `MpsGraphBatchDecoder` decodes a batch
into Metal memory and runs your graph on the same command queue, and
`MpsGraphProgram` runs a graph on a batch that is already decoded. Each group
becomes one rank-4 tensor: Gray, RGB, or RGBA, as `U8`, `U16`, or `I16`, in
NCHW or NHWC order. You build the `MPSGraph` and its one image placeholder;
the crate provides no models.

It needs Apple Silicon and macOS 11 or newer; elsewhere every call returns
`Error::UnsupportedPlatform`. For Burn, use
[`j2k-ml`](https://crates.io/crates/j2k-ml) instead. The crate is
experimental, so a minor release can change it (see
[API stability](https://github.com/frames-sg/j2k/blob/main/book/stability.md)).

## Usage

The repository's `resident_reference_graph` example builds a small graph and
runs it on a decoded batch in each execution mode:

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

The [MPSGraph](https://github.com/frames-sg/j2k/blob/main/book/machine-learning.md#mpsgraph)
section of the J2K Book covers tensor shapes, execution modes, and safety.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
