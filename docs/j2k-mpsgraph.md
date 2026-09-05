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
  submits graph work over a completed codec-owned allocation. The command queue
  must belong to the same Metal device and is rejected before submission when
  its registry identity differs.
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

On other operating systems and Intel macOS, decoder methods return
`Error::UnsupportedPlatform`.

## Caller-owned graph workflow

The production crate deliberately supplies no model or reference-graph
builder. Callers construct an `MPSGraph`, its static image placeholder, and
one or more target tensors, then adopt them through `MpsGraphProgram::new`.
The runnable example builds a simple F32 normalization-and-average graph this
way and checks it against a higher-precision CPU oracle kept in dev-only
support.

```bash
cargo run -p j2k-mpsgraph --example resident_reference_graph
```

## Performance evidence

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
must match the higher-precision CPU oracle within `1e-5`. Timed paths are
rotated between samples to reduce fixed-order bias, and every measured result
is checked again. Rows report a mean and conservative two-sided 95% Student-t
interval; custom runs above 30 samples retain the 30-sample critical value.
Runs below 30 samples report no interval and cannot qualify. Unsupported
platform cells are printed explicitly. The benchmark emits
`speed_claim_qualified=true` only with at least 30 samples, when pipelined
direct execution is at least 10% faster than staged MPSGraph, and when the
intervals do not overlap. No project documentation currently makes an
MPSGraph speed claim, and “zero-copy” is not claimed without Metal tracing that
excludes framework-internal copies.

## Validation

Portable tests cover every shape/dtype mapping, overflow, and a source policy
forbidding decoded-pixel readback/upload calls in production adapter code.
Apple Silicon tests cover the complete native color/dtype/layout identity
matrix, all request geometries, irreversible one-LSB parity, caller-built F32
graph parity against a higher-precision CPU oracle,
completed/pipelined/nonblocking execution, immediate-drop safety, same-device
handoff validation, and session reuse. Full Metal release validation
additionally runs the ignored 1,000-run session soak.
