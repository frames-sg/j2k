# Metal Encode-Stage Coverage

## Goal

Fill the missing Metal encode-stage acceleration gaps for JPEG 2000 and HTJ2K
encoding on macOS while keeping CPU fallback behavior explicit and testable.

## Current State

`crates/j2k-metal/src/encode/stage_accelerator.rs` currently reports Metal
dispatches for:

- forward RCT
- forward 5/3 DWT
- classic Tier-1 code-block encode
- HT code-block encode
- packetization

The dispatch report still returns zero for:

- deinterleave
- forward ICT
- forward 9/7 DWT
- subband quantization

The automatic host-output path is conservative and does not try every available
Metal stage.

## Proposed Work

1. Add Metal deinterleave kernels for the lossless/lossy encode input layouts
   already accepted by the public encode API.
2. Add forward ICT and irreversible 9/7 DWT kernels for lossy J2K/HTJ2K encode
   paths.
3. Add quantization kernels that preserve the current CPU oracle output and
   error handling.
4. Extend dispatch reporting and tests so each new stage has clear attempt and
   dispatch counters.
5. Revisit Auto routing after benchmark evidence exists for each new stage.

## Recommended First PR

Start with Metal encode deinterleave. It is the smallest missing encode-stage
surface, has a direct CPU oracle, and should not require changing packetization,
Tier-1 coding, or Auto routing policy. Keep this PR limited to kernel plumbing,
dispatch accounting, CPU parity tests, and explicit Metal errors for unsupported
input shapes.

## Acceptance Criteria

- Explicit Metal encode requests either dispatch supported stages or return a
  structured unsupported request error.
- Auto mode remains conservative unless benchmarks justify widening dispatch.
- CPU parity tests cover each new Metal stage.
- GPU validation records stage-level dispatch counts and performance artifacts.
- Public docs describe the supported Metal encode-stage surface without
  implying full end-to-end Metal coverage for unsupported shapes.
