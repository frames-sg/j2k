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
- Single-request `BackendRequest::Auto` routes to CPU even when Metal
  capabilities match.
- Viewport paths use hybrid strategies for selected contiguous or resident
  workloads, but unsupported shapes fall back or return structured errors.
- Metal JPEG encode is baseline-only and accepts `Gray8` or `Rgb8` input
  buffers.

## Proposed Work

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

## Acceptance Criteria

- Strict Metal requests reject unsupported JPEG shapes before launching kernels.
- Auto routing changes are backed by benchmark evidence and do not initialize
  Metal for workloads that remain CPU-preferred.
- Decode and encode tests compare Metal output against the CPU JPEG oracle.
- Viewport tests cover contiguous, sparse, scaled, and resident-output paths.
- Public docs distinguish JPEG Metal acceleration from full JPEG feature
  coverage.
