# j2k-mpsgraph

Experimental direct MPSGraph integration for JPEG 2000 and HTJ2K batches on
Apple Silicon macOS 11 or newer.

The crate wraps completed `j2k-metal` resident batches as rank-four
`MPSGraphTensorData`, or queues direct decode and graph work on one Metal
command queue. Neither path reads decoded pixels to the CPU or uploads them
again. The public contract covers Gray, RGB, and RGBA `U8`, `U16`, and `I16`
groups in NCHW and NHWC layout.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

The example constructs a caller-owned graph and exercises completed-buffer
handoff, pipelined blocking execution, and nonblocking submission against a
CPU oracle. See
[`../../docs/j2k-mpsgraph.md`](../../docs/j2k-mpsgraph.md) for API, safety,
validation, and benchmark details.
