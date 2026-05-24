# signinum-transcode-metal

Metal acceleration experiments for `signinum-transcode`.

The crate is intentionally optional. CPU JPEG parsing, entropy decode,
dequantization, and HTJ2K assembly stay outside this crate; this crate only
implements transform-stage acceleration hooks.

`MetalDctToWaveletStageAccelerator::for_auto()` is the normal hybrid entry
point. It may return `None` for small or unsupported jobs so
`signinum-transcode` can use its scalar/Rayon fallback. `new_explicit()` is
strict and returns an error when Metal is unavailable or the requested job shape
is unsupported.

Current accelerated stages:

- direct DCT-grid to irreversible 9/7 first-level projection
- direct DCT-grid to floating-point 5/3 first-level projection
- exact reversible integer 5/3 first-level projection
- same-geometry batches of exact reversible integer 5/3 projections

The reversible integer 5/3 path remains bit-identical to the scalar
`signinum-transcode` oracle in the test coverage. JPEG entropy decode,
dequantization, tile grouping, and HTJ2K packet/codestream writing remain CPU
work.
