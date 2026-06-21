# Metal Encode-Stage Coverage

## Goal

Fill the missing Metal encode-stage acceleration gaps for JPEG 2000 and HTJ2K
encoding on macOS while keeping CPU fallback behavior explicit and testable.

## Current State

`crates/j2k-metal/src/encode/stage_accelerator.rs` currently reports Metal
dispatches for:

- deinterleave for public 1-4 component, 1-16 bit host encode sample layouts
- forward RCT
- forward ICT
- forward 5/3 DWT
- forward 9/7 DWT
- classic Tier-1 code-block encode
- HT code-block encode
- packetization

The dispatch report still returns zero for:

- subband quantization

The automatic host-output path is conservative and does not try every available
Metal stage.

## Proposed Work

1. Add quantization kernels that preserve the current CPU oracle output and
   error handling.
2. Extend dispatch reporting and tests so each new stage has clear attempt and
   dispatch counters.
3. Revisit Auto routing after benchmark evidence exists for each new stage.

## Remaining Gap

Subband quantization is the remaining unimplemented Metal encode stage. Keep the
next implementation limited to kernel plumbing, dispatch accounting, CPU parity
tests, and explicit Metal errors for unsupported input shapes. This is one
encode-stage implementation, not full end-to-end Metal encode coverage for every
public encode route.

## Acceptance Criteria

- Explicit Metal encode requests either dispatch supported stages or return a
  structured unsupported request error.
- Auto mode remains conservative unless benchmarks justify widening dispatch.
- CPU parity tests cover each new Metal stage.
- GPU validation records stage-level dispatch counts and performance artifacts.
- Public docs describe the supported Metal encode-stage surface without
  implying full end-to-end Metal coverage for unsupported shapes.
