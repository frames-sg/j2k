# signinum-transcode-cuda

CUDA acceleration for `signinum-transcode`'s coefficient-domain JPEG→HTJ2K paths.

It implements `signinum_transcode::accelerator::DctToWaveletStageAccelerator` for
direct DCT-grid → one-level **5/3 (reversible)** and **9/7 (irreversible)** wavelet
projections, plus the fused 9/7 → prequantized HTJ2K code-block path — mirroring
`signinum-transcode-metal`. This lets JPEG be transcoded to HTJ2K without an
IDCT→pixels→DWT spatial round-trip.

CPU JPEG parsing, entropy decode, dequantization, and HTJ2K packet/codestream
assembly stay outside this crate (shared `signinum-transcode` / `signinum-j2k-native`
code); this crate only accelerates the transform stage. The CPU scalar code remains
the oracle and fallback and is never reimplemented here.

## Features

- `cuda-runtime` — compile and dispatch the CUDA kernels (which live in
  `signinum-cuda-runtime`). Without it, the accelerator behaves like Metal's
  non-macOS path: `new_explicit()` returns a typed error, `for_auto()` returns
  `Ok(None)` so the caller uses the scalar oracle.
- `cuda-profiling` — enable `signinum-cuda-runtime` profiling.

## Dispatch modes

- `CudaDctToWaveletStageAccelerator::new_explicit()` — unavailable/unsupported CUDA
  dispatch is a typed error (no silent scalar fallback).
- `CudaDctToWaveletStageAccelerator::for_auto()` / `default()` — small or unsupported
  jobs fall back to the scalar oracle via `Ok(None)`.

## Validation

Parity against the scalar oracle is enforced on the self-hosted CUDA runner
(`gpu-validation.yml`): reversible 5/3 is asserted bit-exact; 9/7 within the same
tolerances as the Metal backend (band max-abs-diff ≤ 2.0e-2; prequantized
coefficients within ±1 LSB).
