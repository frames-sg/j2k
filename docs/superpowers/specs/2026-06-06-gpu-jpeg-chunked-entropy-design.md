# GPU JPEG Chunked Entropy Feasibility Design

## Context

This spike evaluates the technique from Weißenberger and Schmidt,
“Accelerating JPEG Decompression on GPUs” (arXiv:2111.09219, submitted
2021-11-17), inside Signinum’s CUDA JPEG runtime.

The paper divides entropy bitstreams into fixed-size subsequences, starts
decoders at arbitrary bit positions, uses JPEG Huffman self-synchronization to
recover `(bit position, component, zig-zag index)` state, computes a prefix sum
over decoded symbol counts, then writes coefficients in parallel. It then
performs DC prefix sums, inverse zig-zag, dequantization, and IDCT on the GPU.

Signinum already has:

- CPU JPEG header parsing, Huffman table preparation, and entropy extraction.
- Session-cached fast packets for 4:2:0, 4:2:2, and 4:4:4 JPEG shapes.
- Signinum-owned CUDA kernels that decode full-frame RGB8 output from those
  fast packets.
- A current CUDA entropy kernel that still consumes CPU-generated restart or
  synthetic checkpoints.

The first feasibility question is whether the paper’s arbitrary-position
entropy synchronization can replace or beat Signinum’s CPU-side checkpoint
planning for generated 4:2:0 JPEGs.

## Scope

First slice:

- Generated baseline sequential YCbCr 4:2:0 JPEGs from Signinum’s encoder.
- Full-tile RGB8 output only.
- No restart markers.
- No 4:2:2, 4:4:4, grayscale, progressive, CMYK, ROI, or scaled decode.
- No production routing changes.
- No claim of user-visible performance until remote CUDA benchmarks pass.

Second slice, only if the first slice is promising:

- Run the same machinery on large WSI-shaped generated tiles.
- Expand validation to 4:2:2 and 4:4:4.
- Compare against current owned CUDA kernels and the nvJPEG comparator.

## Design

Add an experimental CUDA entropy synchronization path behind an internal feature
or explicit benchmark/test entrypoint. The path should not replace
`decode_jpeg_rgb8_owned` yet.

The first implementation should produce synchronization and coefficient-layout
diagnostics, not a production-quality decoder. That keeps the spike focused on
the paper’s core claim: arbitrary bitstream positions can be synchronized
quickly enough on GPU to create useful independent decode chunks.

### Components

1. **Host Plan Builder**

   Reuse `JpegMetalFast420PacketV1` for entropy bytes, Huffman tables,
   quantization tables, dimensions, and MCU shape. Add an experimental
   chunked-entropy plan that chooses fixed bit subsequences, for example
   `s * 32` bits as described in the paper.

2. **CUDA Sync Kernel**

   Add a kernel that assigns one thread to each subsequence. Each thread starts
   decoding at its subsequence bit offset with assumed state `(Y, z = 0)` and
   decodes until the end of the subsequence. It records the last plausible
   state:

   - bit position
   - decoded symbol count
   - component
   - zig-zag index
   - local status

   This mirrors the paper’s `s_info` structure without yet implementing the
   full overflow protocol.

3. **Overflow Synchronization Kernel**

   Add a second kernel that compares adjacent subsequence states by decoding
   forward from the previous subsequence into the next one. It records overflow
   distance and whether synchronization was found. The first spike can use one
   block per sequence with a fixed small sequence size.

4. **Validation Harness**

   Add host tests that run on generated 4:2:0 JPEGs and compare GPU-discovered
   synchronized bit positions against Signinum’s existing CPU planner
   checkpoints. The validation should report:

   - percentage of subsequences that synchronize
   - maximum overflow distance
   - mean overflow distance
   - count of invalid Huffman/status failures

5. **Bench Harness**

   Extend the CUDA JPEG bench or add a narrow bench case that times:

   - existing CPU packet/checkpoint planning
   - GPU sync kernel
   - GPU overflow kernel
   - device-to-host diagnostic readback

## Success Criteria

The first slice is worth extending if all are true on the RTX 4070 SUPER remote
host:

- Generated 4:2:0 JPEGs at `1024x1024`, `2048x2048`, and `4096x4096` complete
  synchronization without kernel status errors.
- At least 99% of subsequences synchronize within a bounded overflow window for
  smooth generated inputs.
- The sync plus overflow kernels are faster than current CPU checkpoint
  planning for at least `2048x2048` and `4096x4096`.
- Diagnostic output gives enough state to implement coefficient writes without
  guessing about component order or block boundaries.

The spike should be stopped or redesigned if synchronization frequently fails
on generated 4:2:0, if overflow dominates runtime, or if diagnostic readback
cost is already larger than current CPU planning.

## Testing

Local tests:

- Compile-only tests for the new runtime plan and kernel metadata.
- Unit tests for host-side subsequence layout and bounds checks.
- CPU-only parity tests that compare the experimental plan’s expected chunk
  count and bit ranges against entropy length.

Remote CUDA tests:

- Runtime-gated sync tests for generated 4:2:0 JPEGs.
- Bench output captured for `1024`, `2048`, and `4096` dimensions.
- Current owned CUDA full decode tests must continue to pass.

## Non-Goals

- Do not replace current owned CUDA decode routing in this spike.
- Do not delete the current CPU checkpoint planner.
- Do not implement ROI or scaled chunked decode.
- Do not add external JPEG dependencies.
- Do not reintroduce nvJPEG into production crates.

## References

- Weißenberger, André, and Bertil Schmidt. “Accelerating JPEG Decompression on
  GPUs.” arXiv:2111.09219, submitted 2021-11-17.
- Signinum current branch: `codex/jpeg-gpu-chunked-decode`.
