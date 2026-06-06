# CUDA Direct JPEG→HTJ2K Transcode (`signinum-transcode-cuda`) — Design

- **Date:** 2026-05-30
- **Status:** Building (goal-directed). Spec captures scope/architecture for traceability.
- **Goal:** A CUDA backend implementing `signinum_transcode::accelerator::DctToWaveletStageAccelerator`
  for **direct, DCT-coefficient-domain** JPEG→HTJ2K transcode — both **reversible 5/3**
  and **irreversible 9/7** — mirroring the existing `signinum-transcode-metal` backend.
  This avoids the IDCT→pixels→DWT spatial round-trip (encode/decode) by projecting
  dequantized DCT blocks straight into wavelet bands / HTJ2K code-block coefficients.

## 1. Why

CUDA today has only the two *separate* endpoints — owned full-frame JPEG CUDA decode
(`signinum-jpeg-cuda`) and HTJ2K *pixel* encode (`signinum-j2k-cuda`) — which force a
pixel round-trip. Metal already has the coefficient-domain shortcut
(`signinum-transcode-metal`). This brings CUDA to parity.

## 2. Architecture (follows the repo's CUDA split)

- **Facade crate `crates/signinum-transcode-cuda`** — mirrors `signinum-transcode-metal`:
  a `CudaDctToWaveletStageAccelerator` with `new_explicit()` / `for_auto()` modes implementing
  the trait. The `cuda-runtime` cargo feature (`["dep:signinum-cuda-runtime"]`) gates the GPU
  path; the default build compiles with the same mode semantics as Metal's non-macOS path
  (explicit → typed `Err`, auto → `Ok(None)` scalar fallback).
- **Kernels in `signinum-cuda-runtime`** — repo convention keeps all `.cu` + `build.rs` PTX
  there (as `signinum-j2k-cuda` does). New `src/transcode_kernels.cu`; new Rust job types +
  dispatch in `src/lib.rs`. A new **optional-PTX** build path sets cfg
  `signinum_cuda_transcode_ptx_built` on `nvcc` success (no checked-in PTX fallback, so non-nvcc
  hosts skip the kernel cleanly; the existing strict env requires it on the runner). `--fmad=false`
  is applied (same FP-ordering reason as the encode kernels).

### Reuse (no reinvention)
- Trait + job/output types (`DctGridToReversibleDwt53Job`, `DctGridToDwt97Job`,
  `DctGridToHtj2k97CodeBlockJob`, `ReversibleDwt53FirstLevel`, `Dwt53TwoDimensional<f64>`,
  `Dwt97TwoDimensional<f64>`, `PrequantizedHtj2k97Component`, `Htj2k97CodeBlockOptions`) — from
  `signinum-transcode`.
- **Scalar oracle (parity reference, reused verbatim, never reimplemented):**
  `reversible_dwt53_first_level_from_block_samples`, `idct_blocks_to_signed_samples_rayon`,
  `dct8x8_blocks_then_dwt97_float` (`signinum-transcode::accelerator` / `dct97_2d`).
- Device buffers / kernel launch infra + `build.rs` PTX pattern — from `signinum-cuda-runtime`.
- The GPU kernel **math is ported faithfully from the Metal kernels** (`dct97.metal`), not
  re-derived.

## 3. Scope (mirror Metal exactly)

In scope (the `DctToWaveletStageAccelerator` methods Metal implements):
- `dct_grid_to_reversible_dwt53` (+ `_batch`) — reversible 5/3.
- `dct_grid_to_dwt53` — float 5/3.
- `dct_grid_to_dwt97` (+ `_batch`) — irreversible 9/7.
- `dct_grid_to_htj2k97_codeblock_batch` — fused 9/7 → prequantized HTJ2K code blocks.
- `supports_dwt97_batch` / `supports_htj2k97_codeblock_batch` → `true`.

Chroma subsampling (grayscale / 4:4:4 / 4:2:2 / 4:2:0) is handled per-component by the existing
`signinum-transcode` pipeline; each component is one job. **Out of scope** (matches the transcode
crate): RGB conversion, chroma upsample, progressive-specific handling beyond what the shared
pipeline already does.

## 4. Parity contract (match the Metal tests)

| Path | Oracle | Assertion |
| --- | --- | --- |
| reversible 5/3 | `reversible_dwt53_first_level_from_block_samples` (i32, exact) | **bit-exact** `assert_eq!` on i32 bands |
| float 5/3 | scalar `dct8x8_blocks_to_dwt53_float` (f64) | max abs diff ≤ `2.0e-2` |
| 9/7 | `dct8x8_blocks_then_dwt97_float` (f64) | max abs diff ≤ `2.0e-2` |
| 9/7 prequantized code blocks | scalar prequantized component | layout-equal + quantized coeff within **±1 LSB** |

Numerical model: reversible is exact integer (islow IDCT + euclidean-floor 5/3); 9/7 is f32 on
device vs f64 scalar (hence the tolerance, and `--fmad=false`).

## 5. No-silent-fallback contract

This is an **accelerator** (the trait's `Ok(None)` *is* the defined scalar-fallback path — not a
correctness violation, unlike the strict lossless *encoder*). Modes mirror Metal:
- **Explicit** (`new_explicit`): CUDA-unavailable / unsupported job → typed `Err`.
- **Auto** (`for_auto`, `Default`): small/unsupported jobs → `Ok(None)` (caller uses scalar).

## 6. Validation — fail-closed, runner-gated

- Gated tests on the self-hosted CUDA runner mirror `signinum-transcode-metal`'s
  `dct53.rs` / `dct97.rs` / `jpeg_to_htj2k.rs`: CUDA output vs the scalar oracle at the contract
  tolerances, plus an end-to-end `transcode_with_accelerator` JPEG→HTJ2K test.
- `gpu-validation.yml`: add a `signinum-transcode-cuda --features cuda-runtime` test + clippy step
  with an executed-count floor (a skipped gated test cannot false-green).
- **Local (no GPU/nvcc):** the default-feature build and crate structure compile and are verified;
  the kernels + GPU parity are validated only on the runner.

## 7. Phasing

1. Scaffold crate + trait impl skeleton (compiles default build). *(this commit)*
2. Reversible 5/3: CUDA kernel + runtime job + dispatch + bit-exact gated test.
3. 9/7: IDCT row-lift / column-lift / band projection + codeblock quantize kernels + dispatch +
   tolerance tests; `--fmad=false`.
4. CI wiring + runner validation (parity green).
5. Auto-mode size thresholds + batch paths + end-to-end transcoder integration test.

## 8. Non-goals

- Rewriting the scalar oracle, trait, or job types (reused).
- Porting Metal's tree/packet machinery (HTJ2K assembly stays the shared CPU path).
- The strict lossless *encoder* parity work (separate; Phase 2 spec, deferred).
