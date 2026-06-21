# Metal Transcode Resident Pipeline

## Goal

Move JPEG-to-J2K/HTJ2K transcode toward a more resident Metal pipeline on
macOS, starting from the existing coefficient-domain transform accelerator and
adding adjacent stages only where correctness and benchmarks justify it.

## Current State

- `j2k-transcode-metal` accelerates coefficient-domain DCT-grid to one-level
  5/3 and 9/7 wavelet projections used by HTJ2K paths.
- The Metal accelerator includes explicit and Auto modes, with Auto declining
  small or unsupported jobs so the caller can use the scalar fallback.
- CPU scalar code remains the oracle and fallback.
- This is not yet a complete Metal JPEG entropy decode to final J2K/HTJ2K
  codestream pipeline.

## Proposed Work

1. Map the current transcode pipeline stage by stage, including host/device
   transfers and CPU fallback points.
2. Add resident Metal handoff types for transform outputs that can flow into
   downstream HTJ2K encode or packetization stages.
3. Extend Metal coverage for batches that currently decline in Auto mode only
   because thresholds or shape gates are conservative.
4. Evaluate whether JPEG entropy decode, coefficient preparation, packetization,
   or codestream assembly should be moved next, based on measured CPU time and
   transfer overhead.
5. Add benchmark artifacts for JPEG-to-J2K and JPEG-to-HTJ2K workloads that are
   not limited to WSI examples.

## Acceptance Criteria

- The pipeline map identifies every CPU fallback and host/device transfer.
- New resident stages preserve CPU oracle parity for 5/3 and 9/7 paths.
- Explicit Metal requests fail clearly when a pipeline segment is unsupported.
- Auto mode dispatch changes are benchmark-backed and keep scalar fallback
  behavior explicit.
- Public docs describe this as Metal-accelerated transcode stages unless a
  complete resident pipeline is actually implemented.
