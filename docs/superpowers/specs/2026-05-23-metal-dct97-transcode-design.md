# Metal DCT 9/7 Transcode Design

## Goal

Add a Metal-first hybrid acceleration path for JPEG DCT to HTJ2K 9/7
transcoding. The first accelerated stage is direct DCT-grid to one-level 9/7
wavelet projection. JPEG parsing, entropy decoding, DCT dequantization,
component scheduling, HTJ2K codestream assembly, and small-workload fallback
remain CPU responsibilities.

## Architecture

The implementation lives in a new `signinum-transcode-metal` crate. The CPU
`signinum-transcode` crate stays portable and remains the scalar oracle. The
Metal crate implements `signinum_transcode::accelerator::DctToWaveletStageAccelerator`
as `MetalDctToWaveletStageAccelerator`.

Only the 9/7 hook is implemented first:

```text
JPEG DCT extraction on CPU
  -> signinum-transcode float-direct 9/7 path
  -> MetalDctToWaveletStageAccelerator::dct_grid_to_dwt97
  -> Metal kernel writes LL/HL/LH/HH bands
  -> signinum-j2k-native precomputed 9/7 HTJ2K encode
```

The 5/3 hook remains part of the trait shape but returns `None` in the Metal
accelerator until a later task proves it can match the 5/3 oracle.

## Hybrid Routing

The accelerator exposes two constructors:

- `new_explicit()`: try Metal for supported 9/7 jobs and return a clear error
  if Metal is unavailable or the job is unsupported.
- `for_auto()`: use Metal only above a conservative component-size threshold
  and otherwise return `Ok(None)` so `signinum-transcode` uses scalar fallback.

Non-macOS builds expose the same API. Explicit mode returns
`MetalUnavailable`; auto mode returns scalar fallback.

## Metal Kernel

The first kernel is correctness-first. One thread computes one coefficient in
one output band. Inputs are:

- natural-order dequantized 8x8 DCT blocks as `f32`
- component geometry and block grid
- cached row projection weights for x and y
- fixed 8-point IDCT basis constants

The CPU host builds 9/7 projection weights using the same lifting equations as
the scalar `dct97_2d` module, uploads them as contiguous `f32`, dispatches one
kernel for each band, then downloads the four bands into
`Dwt97TwoDimensional<f64>`. The f32-to-f64 return keeps the existing trait
stable while documenting that Metal comparison uses tolerance, not bit identity.

## Validation

Validation is coefficient-first:

- Direct Metal 9/7 projection must match scalar `dct97_2d` within tolerance for
  8x8, cropped 13x11, and 16x16 synthetic grids.
- JPEG 4:2:0 through Metal 9/7 must produce a natively decodable HTJ2K
  codestream and preserve SIZ `XRsiz/YRsiz`.
- Non-macOS tests must prove explicit Metal reports unavailable and auto mode
  falls back.
- Criterion must include scalar vs Metal 9/7 groups. Performance claims require
  benchmark output from `--profile release-bench`.

## Scope Limits

This design does not accelerate JPEG entropy decode or HTJ2K packet assembly.
It does not make 5/3 Metal the default. It does not claim a speedup until the
benchmark command reports one on this branch.
