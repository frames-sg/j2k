# J2K MPSGraph Support

> Asynchronous MPSGraph submission for [J2K](https://github.com/frames-sg/j2k) and other Metal codec adapters

[![Current Crates.io Version](https://img.shields.io/crates/v/j2k-mpsgraph-support.svg)](https://crates.io/crates/j2k-mpsgraph-support)
[![Documentation](https://docs.rs/j2k-mpsgraph-support/badge.svg)](https://docs.rs/j2k-mpsgraph-support)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/frames-sg/j2k#license)

This crate shares MPSGraph submission code between
[`j2k-mpsgraph`](https://crates.io/crates/j2k-mpsgraph) and a JPEG XR adapter
outside this repository. To run MPSGraph on decoded JPEG 2000 batches, use
`j2k-mpsgraph` instead.

`MpsGraphSubmission` runs an `MPSGraph` with one placeholder asynchronously on
a caller's command queue. It owns everything the run needs: the graph, feeds,
targets, execution descriptor, completion callback, and a caller-supplied
owner that keeps the input storage alive. Submitting is `unsafe`, because the
caller must guarantee that the tensor data matches the placeholder and stays
valid. `wait` returns the graph's Foundation error as a `GraphExecutionError`,
and dropping an unfinished submission waits for it. The crate builds on every
platform but is empty except on Apple Silicon macOS.

<!-- j2k-crate-footer -->

---

Part of [J2K](https://github.com/frames-sg/j2k), a JPEG 2000 and HTJ2K codec
for Rust. See the [J2K Book](https://github.com/frames-sg/j2k/tree/main/book)
and the [API documentation](https://docs.rs/j2k).
