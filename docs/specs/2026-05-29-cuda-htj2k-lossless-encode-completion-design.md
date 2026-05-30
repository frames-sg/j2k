# CUDA HTJ2K Lossless Encode — Completion to Native Parity (Approach C)

- **Date:** 2026-05-29
- **Status:** Design approved; pending spec review → implementation plan.
- **Approach:** C — Coverage-maximal (chosen over A/B).
- **Implementation base branch:** `codex/cuda-htj2k-runner` (the only branch that
  carries the CUDA encode pipeline; the current `codex/maturity-hardening` does not).
  Exact merge/rebase logistics are deferred to the implementation plan.

## 1. Problem & goal

The strict CUDA HTJ2K **lossless** encode engine already exists and works: the
lower-level path (`encode_with_accelerator` + `CudaEncodeStageAccelerator`) runs
deinterleave → RCT → forward DWT 5/3 → quantize → HT cleanup codeblock encode →
GPU packetization entirely on device and **round-trips byte-exact** in tests such
as `cuda_encode_uses_resident_dwt_tile_body_when_runtime_required`
(`decoded.data == pixels`).

Two gaps keep it from being "complete":

1. The **public strict facade** `encode_j2k_lossless_with_cuda` has **no passing
   end-to-end test** — its only three tests `expect_err()` (they assert rejection
   of Classic/tier-1 mode). The success path and the coverage edges are unproven.
2. Codestream marker assembly (`SOC…EOC`) runs on host. This is *not* a
   no-fallback violation (the backend is tagged `Cuda` from accelerator stage
   dispatch counts; host marker-writing does not un-tag it), but it leaves the
   pipeline not-fully-resident relative to Metal.

**Goal:** complete the CUDA lossless HTJ2K encoder to **byte-exact parity with the
native reference** across native's full producible lossless set, fully
device-resident, with no silent CPU fallback, all validated on the CUDA runner.

## 2. Definition of done (acceptance criteria)

- `encode_j2k_lossless_with_cuda` returns a codestream **byte-identical** to
  `signinum_j2k_native::encode_htj2k` for every in-scope input (§3).
- Every in-scope input also **round-trips**: `decode(bytes) == input_pixels`.
- The pipeline is **fully device-resident**, including codestream assembly (C4;
  this criterion is satisfied at Phase 2 — see §8).
- **No silent CPU fallback**: out-of-scope inputs return typed errors (§6).
- All of the above run under CI on the self-hosted CUDA runner, gated by
  `SIGNINUM_REQUIRE_CUDA_RUNTIME`.

## 3. Scope

Parity is defined against native, so the in-scope set is exactly **native's
producible lossless set**.

### In scope
- Reversible 5/3 DWT (lossless), HTJ2K **cleanup-pass-only** codeblocks.
- **Single** tile / layer / precinct.
- 1–4 components (MCT/RCT applies at exactly 3, and to the first three planes of a
  4-component image; 1–2 components carry no MCT).
- All bit depths 8–16, signed and unsigned.
- Multi-level DWT (0..N resolutions), multi-codeblock, multi-subband.

### Out of scope (with rationale)
- **Classic/tier-1 EBCOT** — not HTJ2K; `encode_tier1_code_block` is permanently
  unsupported on GPU and stays that way.
- **Lossy 9/7** — never byte-exact, so native parity is impossible by definition.
- **Multiple quality layers** — native hard-errors on `num_layers != 1`
  (`encode.rs:1658`); no reference.
- **SigProp/MagRef passes (target_coding_passes 2–3)** — already built but beyond
  native and validatable only by round-trip; **frozen as a documented
  experimental extra**, not part of "complete lossless."
- **Multi-tile** — native is hardcoded single-tile (`codestream_write.rs:94`,
  XTsiz/YTsiz = image), so no parity reference; *and* it is architecturally
  redundant: the codec already tiles at the caller level (independent
  single-tile codestreams per tile, `TileBatchDecode`, Metal's
  `SubmittedJ2kLosslessMetalEncodeBatch → Vec<EncodedJ2k>`), which already
  delivers random access, parallelism, and bounded memory.

### Conditional (spike-gated)
- **Component subsampling ≠ (1,1)** — native carries SIZ `XRsiz/YRsiz` plumbing
  (`codestream_write.rs:153`, `params.component_sampling`) but its full
  DWT/codeblock pipeline support for non-(1,1) is unverified, and CUDA hard-rejects
  `sampling != (1,1)` (`encode_htj2k_tile`). **A spike must first confirm native
  round-trips subsampled lossless.** If yes → lift the CUDA `(1,1)` rejection and
  add parity coverage. If no → document subsampling as out (it would require native
  work first). Do not commit blind.

## 4. Architecture

The resident GPU **engine is unchanged** — no edits to the per-pixel/per-codeblock
kernels in Approach C's core. Work sits above the engine plus **one** new device
kernel (C4). Six components:

- **C1 — Facade contract.** Make `encode_j2k_lossless_with_cuda` the single strict
  entry: route HTJ2K-lossless to the resident accelerator, add the **success path**
  (today only `expect_err`), and map kernel/stage detail codes to precise typed
  errors for out-of-scope configs.
- **C2 — Coverage closure.** Close the real in-scope gaps in `encode_htj2k_tile`:
  the `use_mct && num_components != 3` rejection (handle 1- and 2-component
  no-MCT inputs, and 4-component: RCT over the first three planes, 4th plane
  passthrough, matching native), and the 16-bit / signed deinterleave paths. Each fix is driven by a
  failing parity-matrix cell first.
- **C3 — Fixed parity matrix.** Table-driven test over the in-scope set; per cell,
  encode via CUDA facade and via native, assert byte-identical, then round-trip.
- **C4 — GPU codestream-assembly kernel.** Port Metal's
  `j2k_assemble_lossless_codestream_batched` (`encode_bitstream.metal:2481`) to a
  CUDA kernel so `SOC…EOC` framing runs on-device; flip the strict path to fully
  resident. Tested byte-exact vs (i) the current host assembler and (ii) native.
- **C5 — Exhaustive coverage.** Extend C3 to every bit depth 8–16 and the full
  component set, signed+unsigned, representative codeblock sizes and DWT levels.
- **C6 — Property/fuzz parity harness.** Seeded, deterministic randomized inputs
  over the in-scope set → assert `bytes_cuda == bytes_native` and round-trip. Any
  skipped/dropped config is logged (no silent truncation).

## 5. Data flow (per parity-matrix / fuzz cell)

```
pixels ─┬─> encode_j2k_lossless_with_cuda   ─> bytes_cuda
        └─> native::encode_htj2k (reference) ─> bytes_native
assert bytes_cuda == bytes_native            # primary: byte parity
assert decode(bytes_cuda) == pixels          # secondary: round-trip
```

Byte-parity is the primary oracle (strongest). Round-trip guards against a shared
bug on both sides. Lossless integer math (5/3 integer lifting, integer quantize =
round, integer RCT, level-shift) makes byte-exactness attainable; the existing
per-stage parity tests (codeblock and packetization vs native scalar) already
demonstrate it at stage granularity.

## 6. Error handling / rejection taxonomy

The strict path never silently falls back; it returns typed, non-sensitive errors:

| Input | Result |
| --- | --- |
| Classic/tier-1 block coding | `UnsupportedCudaRequest` — "HTJ2K-only encoder" |
| Lossy 9/7 | `UnsupportedCudaRequest` — "lossless-only" |
| `num_layers != 1` | `UnsupportedCudaRequest` — "single-layer only" |
| Unsupported component count / subsampling (pre-spike) | `UnsupportedCudaRequest` with the specific reason |
| CUDA runtime unavailable | existing unavailable error |

Existing kernel detail-code rejections stay; the facade translates each into one
clear message. Every rejection gets a negative test.

## 7. Testing & rollout

- **Core:** the C3/C5 parity matrix and the C6 fuzz harness (gated by
  `SIGNINUM_REQUIRE_CUDA_RUNTIME`).
- Keep existing per-stage parity tests; **add the missing facade success tests**;
  add negative/rejection tests for every out-of-scope config.
- **C4:** byte-exact tests of the GPU assembler vs the host assembler and vs native.
- **CI:** `.github/workflows/gpu-validation.yml` (self-hosted CUDA job) runs the
  matrix + fuzz under the env gate; fmt/clippy/bench-compile gates as today.
- **Docs:** update `docs/architecture.md` and `docs/wsi-decode-api.md` to state
  CUDA lossless encode is at native parity, list the supported matrix, and record
  the explicit non-goals (tier-1, lossy, layers, multi-tile, SigProp/MagRef).

## 8. Phasing (all in scope; lands incrementally)

- **Phase 1 — Validated lossless parity (host assembly).** C1 + C2 + C3 with the
  1–4-component and 8/16-bit cells; 4-component coverage fix. Ships defensible
  byte-exact native parity for lossless before any new kernel (the full-residency
  criterion lands in Phase 2).
- **Phase 2 — Fully resident.** C4 GPU codestream-assembly kernel + its parity tests.
- **Phase 3 — Maximal + hardening.** C5 full bit-depth/component sweep, C6 fuzz
  harness, and the subsampling spike (then include or document-out per result).

Phases 2–3 carry the new risk; Phase 1 still delivers value if they slip.

## 9. Risks

- **C4 assembly kernel** is the only new device code — port carefully from Metal;
  validate byte-exact against the host assembler first.
- **Subsampling spike** may come back negative (native lacks full support) → keep
  subsampling out and document.
- **Byte-exactness** relies on lossless integer math; any float path (must be none
  for 5/3 lossless) would break parity — assert integer pipeline in tests.

## 10. Open items to resolve in the implementation plan

- Native's exact 4-component lossless behavior (RCT-on-first-three vs none) — match
  it bit-for-bit.
- Native's 16-bit/signed deinterleave level-shift vs the CUDA deinterleave path —
  confirm bit-identical.
- Base-branch/merge logistics: the encode pipeline lives on
  `codex/cuda-htj2k-runner`; decide how this work and the in-flight test fixes on
  `codex/maturity-hardening` are reconciled.

## 11. Key code references (evidence)

- Native single-tile assembly: `crates/signinum-j2k-native/src/j2c/codestream_write.rs:94`
  (SOT), `:115` (SIZ), `:153` (per-component XRsiz/YRsiz).
- Native single-layer hard-error: `crates/signinum-j2k-native/src/j2c/encode.rs:1658`.
- Native HTJ2K cleanup-only codeblock: `crates/signinum-j2k-native/src/j2c/ht_block_encode.rs`
  (module doc "cleanup-only"; `num_coding_passes: 1` for non-empty, `0` for all-zero).
- CUDA facade (strict, no-fallback): `crates/signinum-j2k-cuda/src/encode.rs:32`
  (`encode_j2k_lossless_with_cuda`), `strict_cuda_encode_options`,
  `reject_non_cuda_encode_backend`.
- CUDA tile coverage rejections: `encode_htj2k_tile` (`use_mct && num_components != 3`,
  `component_sampling != (1,1)`).
- CUDA encode kernels: `crates/signinum-cuda-runtime/src/htj2k_encode_kernels.cu`
  (cleanup + frozen SigProp/MagRef; pass ceiling `> 3` → UNSUPPORTED detail 5 at ~:1135).
- Metal reference for C4: `crates/.../encode_bitstream.metal:2481`
  (`j2k_assemble_lossless_codestream_batched`).
- Caller-level tiling model: `docs/wsi-decode-api.md` (`TileBatchDecode`,
  `decode_region_into`); Metal `SubmittedJ2kLosslessMetalEncodeBatch → Vec<EncodedJ2k>`.
