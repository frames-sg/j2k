# Metal JPEG Path Coverage

## Goal

Expand Metal acceleration coverage in `j2k-jpeg-metal` for supported JPEG
decode and baseline encode workloads while keeping strict errors for unsupported
formats and packet shapes.

## Current State

- Explicit Metal JPEG decode requires fast baseline 4:2:0, 4:2:2, or 4:4:4
  packets.
- Explicit Metal JPEG decode currently supports `Gray8`, `Rgb8`, and `Rgba8`
  output formats.
- Routing and benchmark evidence is documented in
  `crates/j2k-jpeg-metal/docs/routing-benchmarks.md`.
- Single-request `BackendRequest::Auto` remains conservative unless benchmark
  evidence shows Metal should be selected for a supported workload.
- Viewport paths use hybrid strategies for selected contiguous or resident
  workloads, but unsupported shapes fall back or return structured errors.
- Metal JPEG encode is baseline-only and accepts `Gray8` or `Rgb8` input
  buffers.

## Landed

- Added benchmark harness coverage for JPEG Metal routing evidence.
- Documented where Metal should and should not be selected automatically.
- Preserved strict Metal errors for unsupported packet shapes.

## Remaining Work

1. Add a support matrix test suite for baseline sampling modes, restart
   intervals, output formats, and viewport shapes.
2. Widen explicit Metal decode only where fast packet extraction and parity are
   proven.
3. Add benchmark-backed Auto routing for full-frame or viewport workloads where
   Metal consistently wins.
4. Extend baseline Metal encode coverage only after parity and output-size
   bounds are stable.
5. Document unsupported JPEG features explicitly rather than silently falling
   back for strict Metal requests.

## Acceptance Criteria For The Next PR

- Strict Metal requests reject unsupported JPEG shapes before launching kernels.
- Any Auto routing changes are backed by benchmark evidence and do not
  initialize Metal for workloads that remain CPU-preferred.
- Decode and encode tests compare Metal output against the CPU JPEG oracle.
- Viewport tests cover contiguous, sparse, scaled, and resident-output paths.
- Public docs distinguish JPEG Metal acceleration from full JPEG feature
  coverage.
