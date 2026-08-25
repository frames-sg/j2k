# Direct MPSGraph batch integration

`j2k-mpsgraph` is an experimental Apple Silicon macOS 11+ adapter. It is
separate from `j2k-ml`: the latter remains the portable Burn adapter and stages
Metal output through host memory, while this crate speaks directly to Apple's
MPSGraph framework.

## Data contract

Every successful homogeneous codec group maps to one static rank-four tensor:

| Codec group | NCHW | NHWC | MPS dtype |
| --- | --- | --- | --- |
| Gray `U8` | `[N,1,H,W]` | `[N,H,W,1]` | `UInt8` |
| RGB `U8` | `[N,3,H,W]` | `[N,H,W,3]` | `UInt8` |
| RGBA `U8` | `[N,4,H,W]` | `[N,H,W,4]` | `UInt8` |
| Gray/RGB/RGBA `U16` | corresponding shape | corresponding shape | `UInt16` |
| Gray/RGB/RGBA `I16` | corresponding shape | corresponding shape | `Int16` |

`MpsGraphBatchDecode` preserves source indices, decoded rectangles, warnings,
indexed preparation failures, and homogeneous group failures. Completed-buffer
construction aliases the codec-owned `MTLBuffer`; it does not copy decoded
pixels.

`MpsGraphProgram` accepts one static rank-four image placeholder whose shape
and dtype exactly match its `MpsGraphTensorSpec`, plus one or more targets.
Additional runtime feeds are intentionally unsupported in v1; weights and
other model inputs must be constants in the graph.

## Execution modes

- `MpsGraphProgram::submit_completed` consumes an `MpsGraphInputGroup` and
  submits graph work over a completed codec-owned allocation.
- `MpsGraphBatchDecoder::run_prepared_group` allocates a checked private Metal
  destination, queues decode and graph execution on one command queue, then
  waits. There is no CPU wait between decode and inference.
- `MpsGraphBatchDecoder::submit_prepared_group` returns immediately with a
  `SubmittedMpsGraphRun`. `is_complete` is nonblocking and consuming `wait`
  validates codec status and graph completion before exposing outputs.

`SubmittedMpsGraphRun` is deliberately neither `Send` nor `Sync`. It owns the
destination allocation, codec submission, graph, feeds, result dictionary,
execution descriptor, completion block, and completion state. Dropping an
in-flight guard waits before releasing the input because MPSGraph does not
promise to retain an `MTLBuffer` used by `MPSGraphTensorData`.

On other operating systems and Intel macOS, portable constructors and decoder
methods return `Error::UnsupportedPlatform`.

## Reference workflow

`MpsGraphProgram::rgb8_nhwc_reference` casts RGB8/NHWC input to F32, normalizes
it to `[0,1]`, applies fixed `[0.2126, 0.7152, 0.0722]` channel weights, and
returns one spatially reduced score per image. The matching
`rgb8_nhwc_reference_cpu` function is the correctness oracle.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

## Performance evidence

The benchmark reports staged decode/readback/upload/MPSGraph, completed
resident handoff, pipelined direct execution, and nonblocking direct execution
for repository-generated reversible HTJ2K and classic J2K pathology-sized RGB tiles at
512×512 and 1024×1024 with batches 1, 8, and 32:

```bash
cargo bench -p j2k-mpsgraph --bench direct_handoff
```

Set `J2K_MPSGRAPH_BENCH_ITERATIONS` to control repeated samples. Every row
reports a mean and normal-approximation 95% confidence interval; unsupported
platform cells are printed explicitly. The benchmark emits
`speed_claim_qualified=true` only when pipelined direct execution is at least
10% faster than staged MPSGraph and the intervals do not overlap. No project
documentation currently makes an MPSGraph speed claim, and “zero-copy” is not
claimed without Metal tracing that excludes framework-internal copies.

## Validation

Portable tests cover every shape/dtype mapping, overflow, CPU-oracle behavior,
and a source policy forbidding decoded-pixel readback/upload calls in production
adapter code. Apple Silicon tests cover the complete native color/dtype/layout
identity matrix, all request geometries, irreversible one-LSB parity,
completed/pipelined/nonblocking execution, immediate-drop safety, and session
reuse. Full Metal release validation additionally runs the ignored 1,000-run
session soak.
