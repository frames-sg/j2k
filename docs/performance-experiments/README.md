# GPU performance experiment records

Every GPU optimization candidate uses a versioned JSON record and the same
sequence: preflight, baseline, instrumentation, implementation, A/B,
correctness, performance, promote or reject, cleanup, and documentation.
Validate a record before citing it:

```bash
cargo xtask gpu-experiment validate docs/performance-experiments/P1.json
```

The environment object records the full commit, branch and dirty state, CPU,
GPU, RAM, OS, driver/runtime, Rust and LLVM versions, Metal or CUDA toolchain,
build profile, feature flags, environment variables, corpus SHA-256, sample
count, warm-up, and measurement duration. A missing tool metric is represented
as `null`, with the limitation explained in the decision rationale; it must not
be replaced by an estimate.

Each applicable experiment selects representative cells from this matrix:

| Dimension | Required values |
|---|---|
| Transform | reversible 5/3; irreversible 9/7 |
| Entropy | HT; Classic |
| Code block | 32×32; 64×64 |
| Image | 512×512; 640×480; 1024×1024; 2592×1944 |
| Batch | 1; 16 |
| Components | 1; 3; 4 |
| Output | native; RGB8; RGBA8 |
| Operation | full; ROI; half-scale |
| Axis | below and above 512 and 1024 |
| JPEG sampling | 4:4:4; 4:2:2; 4:2:0 |
| JPEG restart | none; present |

Every measured workload has exactly one `baseline` and one `treatment` row.
Rows record wall-time confidence intervals and, when tools expose them,
GPU/stage time, dispatches,
transfer and device traffic, registers, private/shared memory, occupancy,
spills, and cache observations. Output SHA-256, exact parity, and conformance
status are mandatory. The validator rejects a measured decision without both
variants or exact output parity.

Schema versions 1 and 2 remain valid. Schema version 3 adds optional launch and
workspace evidence without requiring it from earlier or unrelated records:

- `launch_geometries` maps a measured stage name to `grid` and `block` arrays in
  CUDA `(x, y, z)` order. Every recorded dimension is nonzero.
- `checkpoint_count` is the number of entropy checkpoints assigned to the
  measured decode launch, not a kernel-dispatch count.
- `component_workspace_bytes` is the resident component-plane workspace
  footprint for the measured workload. It is allocation size, not device-memory
  traffic.
- `coefficient_scratch_bytes` is the resident intermediate coefficient-scratch
  footprint. It is likewise allocation size, not read/write traffic.
- `coefficient_scratch_traffic_bytes` is logical scratch traffic derived from
  the measured route's explicit accesses. For P19 defusion it is exactly three
  times the scratch footprint: one clear write, one entropy-coefficient write,
  and one IDCT read. It is not a hardware-counter estimate.

When any optional measurement field is present, its availability must match in
the baseline and treatment rows. A record should use the dedicated
`device_read_bytes` and `device_write_bytes` fields for measured traffic rather
than deriving traffic from either workspace-footprint field.

The P19 CUDA packed-checkpoint record uses the `checkpoint_decode` launch name
and the exact stage keys `resource_upload`, `fused_decode_kernel`, `conversion`,
`status_readback`, and `profiled_product_wall`; all stage values are nanoseconds.
For every paired workload, input and output hashes, checkpoint count, and
component workspace must agree. Both arms retain zero coefficient scratch. The
baseline uses one block per checkpoint with one thread per block. The adaptive
treatment keeps that geometry below 128 checkpoints, then uses 128 threads per
block and `ceil(checkpoint_count / 128)` blocks. Every P19 workload whose launch
geometry changes must have a treatment point estimate no slower than its
baseline; below-threshold rows are unchanged-route controls.

The rejected P19 CUDA decode-defusion record retains the same exact workload
matrix and adaptive `checkpoint_decode` geometry. Its stage keys additionally
separate `coefficient_scratch_clear`, `entropy_coefficients`, and
`idct_deposit`. Baseline rows use the fused route with zero coefficient scratch
and zero split-stage time. Treatment 4:2:0 rows record exact nonzero i32
coefficient scratch, its three logical accesses, and the per-block
`idct_deposit` launch. Treatment 4:2:2 and 4:4:4 rows are unchanged fused-route
controls with zero scratch and zero split-stage time.

Valid statuses are `measured`, `promoted`, `rejected`, and `blocked`.
Promotion additionally requires confidence-interval support, no material
representative regression, and a recorded judgment that complexity is
proportional to the end-to-end benefit. A stage-local or dispatch-count win is
not sufficient. `split-command` profiling may attribute costs, but its altered
command-buffer and cache behavior makes it ineligible as the final throughput
comparison.
