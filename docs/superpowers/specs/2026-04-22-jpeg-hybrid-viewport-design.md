# JPEG Hybrid Viewport Design

## Goal

Make `CPU+Metal` beat `CPU-only` on an end-to-end JPEG WSI viewer task by benchmarking and optimizing the full `decode -> resize -> composite` pipeline instead of raw decode-only microbenches.

## Constraints

- No public API compatibility constraints yet.
- Apple Silicon is the primary target.
- `CPU-only` remains first-class on hosts without Metal.
- The hybrid path should only move the stages that are actually favorable for Metal.

## Task Shape

The benchmark task is a viewer-style viewport assembly:

- N JPEG tile requests
- each request is `region + scaled`
- each request targets a viewport destination rect
- final output is one composited RGB viewport

This is more representative than isolated decode benchmarks because it includes the real downstream work where Metal can win.

## Architecture

### CPU-only path

- Use `slidecodec-jpeg` region+scaled decode into RGB scratch/output buffers.
- Composite decoded RGB tiles into the viewport buffer on CPU.

This is the best non-GPU baseline for the targeted viewer task.

### Hybrid CPU+Metal path

- CPU performs JPEG parse, entropy decode, IDCT/downscale, and region-local component row production.
- CPU writes scaled component rows directly into Metal shared buffers using the existing `ComponentRowWriter` seam.
- Metal performs Y/Cb/Cr or RGB plane packing and viewport compositing in a single command buffer.
- Only the final composited viewport is downloaded to host if the caller needs bytes.

This keeps the serial JPEG work on CPU and moves the parallel color/pack/composite work to Metal.

## Execution Units

Add a viewport-oriented internal layer in `slidecodec-jpeg-metal`:

- `ViewportTileRequest`
  - source JPEG bytes
  - source ROI
  - downscale
  - destination origin/rect in the viewport
- `ViewportComposer`
  - owns the output Metal buffer
  - owns one command buffer per composed viewport
  - accepts multiple per-tile plane stages
- `HybridPlanner`
  - groups tile requests that share output format and scale behavior
  - drives CPU decode into plane stages, then a single Metal composite pass

## Kernel Strategy

The first kernel slice does not attempt full GPU JPEG entropy decode.

Instead it adds one new Metal kernel:

- `jpeg_pack_into_viewport`
  - reads one tile's plane buffers
  - converts grayscale / YCbCr / RGB to RGB(A)
  - writes directly into the destination viewport rect

The kernel is launched once per tile but encoded into one command buffer for the viewport.

## Scheduler Policy

Default policy:

- `CPU-only`
  - no Metal available
  - tiny/single-tile work where hybrid overhead dominates
- `Hybrid CPU+Metal`
  - multi-tile viewport composition
  - repeated region+scaled viewer work
  - any request where the final product is a composited viewport

This policy is heuristic-driven and can be tightened after measurement.

## Benchmark Success Criteria

Primary benchmark:

- `viewer_region_scaled_composite_rgb`
  - CPU-only vs Hybrid CPU+Metal
  - measured on real local JPEG corpora

Completion criterion:

- Hybrid must beat CPU-only on at least one real viewer-style benchmark configuration on Apple Silicon.

## Non-Goals For This Slice

- Full GPU JPEG entropy decode
- New stable public traits in `slidecodec-core`
- Container-aware WSI policy
- CUDA

## Risks

- CPU region+scaled decode may still dominate too much for Metal compositing to matter.
- Per-tile Metal dispatch overhead may still be high if the viewport contains too few tiles.
- Large source levels may require careful ROI selection so the benchmark reflects viewer work instead of whole-level decode.

## Next Steps

1. Add the viewport-composite kernel and CPU/hybrid viewport helpers.
2. Add correctness tests for viewport byte parity.
3. Add a real benchmark group for viewer region+scaled composite.
4. Measure on restart-coded and non-restart local JPEG corpora.
5. Iterate until hybrid beats CPU-only on at least one real configuration.
