# Direct MPSGraph batch integration

`j2k-mpsgraph` is an experimental adapter for Apple Silicon on macOS 11 or
newer. Unlike `j2k-ml`, the portable Burn adapter that copies Metal output
through host memory, it hands decoded batches directly to Apple's MPSGraph.

## Tensor shapes

Every successful homogeneous codec group maps to one static rank-four tensor:

| Codec group | NCHW | NHWC | MPS dtype |
| --- | --- | --- | --- |
| Gray `U8` | `[N,1,H,W]` | `[N,H,W,1]` | `UInt8` |
| RGB `U8` | `[N,3,H,W]` | `[N,H,W,3]` | `UInt8` |
| RGBA `U8` | `[N,4,H,W]` | `[N,H,W,4]` | `UInt8` |
| Gray/RGB/RGBA `U16` | corresponding shape | corresponding shape | `UInt16` |
| Gray/RGB/RGBA `I16` | corresponding shape | corresponding shape | `Int16` |

`MpsGraphBatchDecode` keeps source indices, decoded rectangles, warnings,
per-input preparation failures, and per-group failures. Building a tensor from a
completed batch aliases the codec's `MTLBuffer` without copying pixels.

`MpsGraphProgram` accepts one static rank-four image placeholder whose shape
and dtype exactly match its `MpsGraphTensorSpec`, plus one or more targets.
Other runtime inputs are not supported in this version; weights and other model
inputs must be graph constants.

## Execution modes

- `MpsGraphProgram::submit_completed` takes an `MpsGraphInputGroup` and runs the
  graph on an already-decoded batch. The command queue must be on the same
  Metal device; a queue from another device is rejected before submission.
- `MpsGraphBatchDecoder::run_prepared_group` allocates a checked private Metal
  destination, queues decode and graph execution on one command queue, then
  waits. There is no CPU wait between decode and inference.
- `MpsGraphBatchDecoder::submit_prepared_group` returns immediately with a
  `SubmittedMpsGraphRun`. `is_complete` does not block; `wait` consumes the run
  and checks the decode status and graph completion before returning outputs.

`SubmittedMpsGraphRun` is neither `Send` nor `Sync`. It owns the
destination allocation, codec submission, graph, feeds, result dictionary,
execution descriptor, completion block, and completion state. Dropping an
in-flight guard waits before releasing the input because MPSGraph does not
promise to retain an `MTLBuffer` used by `MPSGraphTensorData`.

On other operating systems and Intel macOS, decoder methods return
`Error::UnsupportedPlatform`.

## Caller-owned graph workflow

The crate does not provide models or graph builders. Callers build an
`MPSGraph`, its static image placeholder, and one or more target tensors, and
pass them to `MpsGraphProgram::new`. The example builds a simple F32
normalize-and-average graph this way and checks it against a higher-precision
CPU calculation in the dev-only test support.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

## Benchmarks

The benchmark reports staged decode/readback/upload/MPSGraph, completed
resident handoff, pipelined direct execution, and nonblocking submission
latency for repository-generated reversible HTJ2K and classic J2K
pathology-sized RGB tiles at 512×512 and 1024×1024 with batches 1, 8, and 32.
The submission-latency row excludes the subsequent wait; every other row is
end-to-end through completed result readback:

```bash
cargo bench -p j2k-mpsgraph --bench direct_handoff
```

Set `J2K_MPSGRAPH_BENCH_ITERATIONS` to control repeated samples; release builds
default to 30. Before timing, every path is warmed and every per-image F32 score
must match the higher-precision CPU calculation within `1e-5`. The order of the
timed paths is rotated between samples, and every result is checked again.
Rows report a mean and a two-sided 95% Student-t interval (runs above 30
samples use the 30-sample critical value; runs below 30 report no interval).
Unsupported platform cells are printed as such. The benchmark prints
`speed_claim_qualified=true` only when there are at least 30 samples, pipelined
direct execution is at least 10% faster than staged MPSGraph, and the intervals
do not overlap. No MPSGraph speedup has been published yet, and the adapter is
not described as zero-copy because MPSGraph may copy internally.

## Validation

Portable tests cover every shape/dtype mapping and overflow. Apple Silicon
tests cover every color/dtype/layout combination, all request geometries,
irreversible output within one LSB, F32 graph output against a
higher-precision CPU calculation,
completed/pipelined/nonblocking execution, immediate-drop safety, same-device
checks, and session reuse. The full Metal release run also runs the ignored
1,000-run session soak.
