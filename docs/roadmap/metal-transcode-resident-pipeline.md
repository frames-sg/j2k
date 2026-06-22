# Metal Transcode Resident Pipeline

## Goal

Move JPEG-to-J2K/HTJ2K transcode toward a more resident Metal pipeline on
macOS, starting from the existing coefficient-domain transform accelerator and
adding adjacent stages only where correctness and benchmarks justify it.
JPEG-to-J2K/HTJ2K transcoding itself is implemented; this document tracks
optional Metal residency and performance work, not feature completion.

## Current State

- `j2k-transcode-metal` accelerates coefficient-domain DCT-grid to one-level
  5/3 and 9/7 wavelet projections used by HTJ2K paths.
- The Metal accelerator includes explicit and Auto modes, with Auto declining
  small or unsupported jobs so the caller can use the scalar fallback.
- `j2k-transcode` exposes a pipeline residency map that reports CPU fallback
  points and host/device transfer boundaries from timing reports.
- CPU scalar code remains the oracle and fallback.
- This is not yet a complete Metal JPEG entropy decode to final J2K/HTJ2K
  codestream pipeline.

## Landed

- Added a transcode pipeline map that identifies stage residency and fallback
  points from current timing data.
- Added tests and bench harness coverage for the pipeline map.
- Kept public wording at Metal-accelerated transcode stages rather than
  claiming a complete resident pipeline.

## Optional Metal Residency Work

1. Add resident Metal handoff types for transform outputs that can flow into
   downstream HTJ2K encode or packetization stages.
2. Extend Metal coverage for batches that currently decline in Auto mode only
   because thresholds or shape gates are conservative.
3. Evaluate whether JPEG entropy decode, coefficient preparation, packetization,
   or codestream assembly should be moved next, based on measured CPU time and
   transfer overhead. Leave those stages on CPU when the workload is irregular,
   small, or transfer-bound.
4. Add benchmark artifacts for JPEG-to-J2K and JPEG-to-HTJ2K workloads that are
   not limited to WSI examples.

## Acceptance Criteria For The Next PR

- The pipeline map identifies every CPU fallback and host/device transfer.
- New resident stages preserve CPU oracle parity for 5/3 and 9/7 paths.
- Explicit Metal requests fail clearly when a pipeline segment is unsupported.
- Auto mode dispatch changes are benchmark-backed and keep scalar fallback
  behavior explicit.
- Public docs describe this as Metal-accelerated transcode stages unless a
  complete resident pipeline is actually implemented.
