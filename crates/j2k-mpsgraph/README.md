# j2k-mpsgraph

Experimental direct MPSGraph integration for JPEG 2000 and HTJ2K batches on
Apple Silicon macOS 11 or newer.

The crate wraps completed `j2k-metal` resident batches as rank-four
`MPSGraphTensorData`, or queues direct decode and graph work on one Metal
command queue. Neither path reads decoded pixels to the CPU or uploads them
again. It supports Gray, RGB, and RGBA `U8`, `U16`, and `I16` groups in NCHW
and NHWC layout.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

The example builds a graph and runs it three ways (on completed buffers,
pipelined and blocking, and nonblocking), checking each against the CPU
decoder. See
[`../../docs/j2k-mpsgraph.md`](../../docs/j2k-mpsgraph.md) for the API, safety
notes, tests, and benchmarks.
