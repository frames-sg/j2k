# CUDA HTJ2K Lossless Encode — Completion to Native Parity (Approach C)

- **Date:** 2026-05-29 (revised after adversarial review)
- **Status:** Design approved with review fixes applied; → implementation plan.
- **Approach:** C — Coverage-maximal (chosen over A/B).
- **Implementation base branch:** `codex/cuda-htj2k-runner` (the only branch that
  carries the CUDA encode pipeline; current `codex/maturity-hardening` does not).
  Merge/rebase logistics deferred to the implementation plan.
- **Review:** A 5-lens adversarial review (workflow `w0n5v1e6m`) verified six
  blockers in code; their fixes are folded into this revision. See §12 for the
  audit trail.

## 1. Problem & goal

The strict CUDA HTJ2K **lossless** encode engine already exists and works: the
lower-level path (`encode_with_accelerator` + `CudaEncodeStageAccelerator`) runs
deinterleave → RCT → forward DWT 5/3 → quantize → HT cleanup codeblock encode →
GPU packetization on device and **round-trips** in gated tests such as
`cuda_encode_uses_resident_dwt_tile_body_when_runtime_required` (`decoded.data ==
pixels`, `encode.rs:4079`).

Two gaps keep it from "complete":

1. The public facade `encode_j2k_lossless_with_cuda` has **gated round-trip
   success tests (8-bit unsigned, 1-/3-component) and `expect_err` negative
   tests, but no test asserting `bytes_cuda == bytes_native`.** Byte-parity — the
   actual acceptance criterion — is **entirely unproven**, and the coverage edges
   (2-/4-component, 16-bit, signed, large subbands) are untested.
2. Codestream marker assembly (`SOC…EOC`) runs on host via native's
   `write_codestream`. This is **not** a no-fallback violation (the backend is
   tagged `Cuda` from accelerator stage-dispatch counts), and it is in fact why
   header bytes are byte-identical to native *by construction* in Phase 1.

**Goal:** complete the CUDA lossless HTJ2K encoder to **byte-exact parity with the
native reference** across native's full producible lossless set, with no silent
CPU fallback, validated by tests that genuinely run (not skip) on the CUDA runner.

## 2. Definition of done (acceptance criteria)

- `encode_j2k_lossless_with_cuda` returns a codestream **byte-identical** to
  `signinum_j2k_native::encode_htj2k` for every in-scope input (§3) — asserted
  directly as `assert_eq!(bytes_cuda, bytes_native)`, not merely round-trip.
- Every in-scope input also **round-trips** (`decode(bytes) == pixels`).
- **No silent CPU fallback**: out-of-scope inputs return typed errors; the
  accelerator never returns `Ok(None)` for an in-scope input (§6).
- The parity tests **actually execute** on the runner and **fail closed** if the
  CUDA runtime is required-but-absent or the `cuda-runtime` feature is missing
  (§7) — a skipped test cannot masquerade as green.
- Codestream framing stays byte-identical to native (guaranteed in Phases 1–2 by
  reusing native `write_codestream`; preserved if the optional C4 kernel lands).

## 3. Scope

Parity is defined against native, so the in-scope set is exactly **native's
producible lossless set** — and only inputs native is *proven* to round-trip.

### In scope
- Reversible 5/3 DWT (lossless), HTJ2K **cleanup-pass-only** codeblocks.
- **Single** tile / layer / precinct.
- 1–4 components (MCT/RCT applies at exactly 3, and to the first three planes of a
  4-component image with the 4th passed through; 1–2 components carry no MCT).
  **Precondition:** 2- and 4-component cells become parity targets only after a
  native-only round-trip test proves native itself round-trips them (§7) — native
  has no existing 2-/4-component round-trip coverage, so we must not codify a
  native bug as the oracle.
- All bit depths 8–16, signed and unsigned.
- Multi-level DWT (0..N resolutions), multi-codeblock, multi-subband — **bounded
  by GPU tag-tree capacity** (next bullet).

### Bounded dimension — tag-tree capacity
The CUDA packetizer uses fixed tag-tree buffers (≤2048 nodes, ≤16 levels) and
**hard-errors** on subbands exceeding them; native is unbounded
(`tag_tree_encode.rs`). At the default 64×64 code block a component wider/taller
than ~5120 px produces an over-capacity subband that native encodes but CUDA
aborts. **Default plan: grow/segment the GPU tag-tree buffers to remove the
ceiling** (true coverage-maximal). If growing proves infeasible within budget,
the documented fallback is a **typed `UnsupportedCudaRequest`** with the capacity
reason (§6) plus a logged bound — never a silent abort. This is characterized by a
dedicated boundary test (§7).

### Out of scope (with rationale)
- **Classic/tier-1 EBCOT** — not HTJ2K; `encode_tier1_code_block` stays
  unsupported and is rejected with a typed error.
- **Lossy 9/7** — never byte-exact; native parity impossible by definition.
- **Multiple quality layers** — native hard-errors on `num_layers != 1`
  (`encode.rs:1658`).
- **SigProp/MagRef passes (target_coding_passes 2–3)** — beyond native; round-trip
  only. **Frozen** as a documented experimental extra.
- **Multi-tile** — native is hardcoded single-tile (`codestream_write.rs:94`); no
  parity reference, and architecturally redundant with the codec's existing
  per-tile-codestream batching (`TileBatchDecode`; Metal
  `SubmittedJ2kLosslessMetalEncodeBatch → Vec<EncodedJ2k>`).
- **Component subsampling ≠ (1,1)** — **definitively OUT** (was "spike-gated";
  the spike is statically resolved as negative). In the pixel-input `encode_htj2k`
  path, `component_sampling` reaches only the SIZ writer
  (`codestream_write.rs:153-161`); `num_pixels` is the full reference grid and the
  forward DWT/deinterleave use full width/height for every component
  (`encode.rs:1183-1185,1314-1326,2363-2409`), while the decoder sizes each
  component grid by `div_ceil(resolution)` (`codestream.rs:646-662`,
  `tile.rs:333-352`). The result is an internally inconsistent codestream that
  cannot reconstruct the input — i.e. native does **not** round-trip subsampled
  lossless. The CUDA `(1,1)` rejection stays (as a typed error, not `Ok(None)`).
  Any future subsampling support belongs to the `encode_precomputed_htj2k_53`
  path, not `encode_htj2k`, and is a separate project.

## 4. Architecture

The resident GPU **per-pixel/per-codeblock kernels are unchanged** except for two
determinism fixes (build flag + possible tag-tree buffer growth). Work sits above
the engine plus one *optional* new device kernel (C4). Components:

- **C1 — Facade contract.** Make `encode_j2k_lossless_with_cuda` the single strict
  entry: route HTJ2K-lossless to the resident accelerator, add the byte-parity
  success path, and map stage/kernel rejections to precise typed errors.
  **Invariant: the accelerator never returns `Ok(None)` for an in-scope input** —
  it returns `Some`/`Ok` (handled on GPU) or a typed `Err` (explicit, no silent
  CPU fallback).
- **C2 — Coverage closure.** The one genuine **missing code path** is 4-component
  resident MCT: the resident RCT wrapper currently requires exactly 3 planes
  (`lib.rs:2480`) and `encode_htj2k_tile` returns `Ok(None)` for `use_mct &&
  num_components != 3` (`encode.rs:2113`). Implement: RCT planes 0–2, pass plane 3
  through, consume native's already-computed `guard_bits` (`.max(2)` under MCT,
  `encode.rs:1205`) and quant params (no recomputation). **16-bit, signed, and
  2-component are already supported** by the resident deinterleave
  (`lib.rs:2414-2423`; signed cast matches native) — for those the gap is **parity
  coverage, not code**.
- **C3 — Byte-parity matrix (Phase-1 gate).** Table-driven test: per cell, encode
  via the CUDA facade and via native, **`assert_eq!(bytes_cuda, bytes_native)`**,
  then round-trip. Plus **stage-level parity tests** for forward 5/3 DWT, forward
  RCT, reversible quantize, and deinterleave vs their native f32 counterparts
  (requires new native scalar-reference exports analogous to the existing
  `encode_ht_code_block_scalar` / `encode_j2k_packetization_scalar`).
- **C5 — Exhaustive coverage.** Extend C3 to every bit depth 8–16, the full
  component set, signed+unsigned, representative codeblock sizes/DWT levels, and
  the tag-tree boundary cell.
- **C6 — Property/fuzz parity harness.** Seeded, deterministic randomized inputs
  over the in-scope set → `bytes_cuda == bytes_native` + round-trip. Any skipped
  config is logged (no silent truncation).
- **C7 — Build determinism.** Add `--fmad=false` to the encode-kernel `nvcc`
  invocation (`build.rs:59-60`) so `a*b+c`-shaped DWT/RCT updates are **not**
  contracted into single-rounding FMA (Rust does not auto-contract; nvcc does),
  guaranteeing bit-identical f32 results vs native at high magnitude / deep
  transforms. Locked in by a 16-bit-signed multi-level parity test.
- **C4 — GPU codestream-assembly kernel (optional, last).** *Residency-cosmetic
  only:* host assembly already produces byte-identical-to-native framing by
  construction (shared `write_codestream`), and ~126 header bytes are trivial vs
  the tile body. **Demoted below C5/C6.** Build only if a measured residency need
  (e.g. avoiding a body readback) materializes; if so, implement a **thin "device
  memcpy of body + small fixed header-writer"** (≈10 data-dependent scalar fields)
  — **do not** port Metal's tree/packet machinery — with `bytes-vs-host` **and**
  `bytes-vs-native` tests. Otherwise drop it and document host assembly as the
  intentional final step.

## 5. Data flow & the parity rationale (corrected)

```
pixels ─┬─> encode_j2k_lossless_with_cuda   ─> bytes_cuda
        └─> native::encode_htj2k (reference) ─> bytes_native
assert bytes_cuda == bytes_native            # primary: byte parity (the gate)
assert decode(bytes_cuda) == pixels          # secondary: round-trip
```

**Why byte-exactness holds (corrected — the v1 "integer math" claim was false):**
the lossless 5/3 DWT and RCT are computed in **f32 on both sides**
(`fdwt.rs:51` operates on `&[f32]`; CUDA mirror in `j2k_encode_kernels.cu` is
`float*`). Parity therefore rests on:
1. **Shared framing** — Phases 1–2 reuse native's `write_codestream`, so every
   `SOC/SIZ/CAP/COD/QCD/SOT/SOD/EOC` byte is produced by identical code; only
   packet bytes are GPU-sourced.
2. **Bit-identical f32 operation ordering** between the CUDA kernels and native's
   lifting, **with FMA contraction disabled** (C7).

Consequence for testing: the failure modes to probe are **FP-ordering edge cases**
— boundary mirroring, even/odd lengths, deep multi-level magnitude growth, high
bit depth — **not** "is it integer." The byte-parity oracle's determinism under
native's `rayon` parallelism (`encode.rs:2205,2237`) must itself be asserted
(a CPU-only `encode_htj2k(x) == encode_htj2k(x)` test under varied
`RAYON_NUM_THREADS`).

## 6. Error handling / rejection taxonomy

The strict path never silently falls back and never returns `Ok(None)` for an
in-scope input; it returns typed, non-sensitive errors:

| Input | Result |
| --- | --- |
| Classic/tier-1 block coding | `UnsupportedCudaRequest` — "HTJ2K-only encoder" |
| Lossy 9/7 | `UnsupportedCudaRequest` — "lossless-only" |
| `num_layers != 1` | `UnsupportedCudaRequest` — "single-layer only" |
| Subsampling ≠ (1,1) | `UnsupportedCudaRequest` — "subsampling unsupported" |
| Subband exceeds GPU tag-tree capacity (if ceiling retained) | `UnsupportedCudaRequest` — capacity reason + logged bound |
| 4-component MCT | **handled on GPU (`Some`)**, not rejected |
| CUDA runtime unavailable | existing unavailable error |

Existing kernel detail-code rejections stay; the facade translates each into one
clear message, and the current whole-encode-aborting `Err` from the tag-tree cap
is wired into the typed facade. Every rejection gets a negative test that asserts
the **specific** error (not a generic one — today's non-gated negative tests pass
for the wrong reason off-GPU, §12).

## 7. Testing & rollout — fail-closed

- **Authoritative gate:** the `cuda-x86_64-compatibility` job
  (`.github/workflows/gpu-validation.yml`), which sets
  `SIGNINUM_REQUIRE_CUDA_RUNTIME` and runs `cargo test -p signinum-j2k-cuda
  --features cuda-runtime`. **Add a `push`/`pull_request` (or merge-queue) trigger**
  so parity is enforced on every merge, not only `workflow_dispatch`.
- **Fail-closed CI steps** on that job:
  (a) assert `SIGNINUM_REQUIRE_CUDA_RUNTIME` is set before the test step;
  (b) assert an **executed-test-count floor** for the parity module (nextest/JSON
  output) so a silent early-return cannot pass as green;
  (c) a single ungated `#[test]` **tripwire** that panics when
  `SIGNINUM_REQUIRE_CUDA_RUNTIME` is set but `cfg!(feature = "cuda-runtime")` is
  false.
- Either add the env gate to the `linux-ci` (`cargo xtask ci`) path too, or
  **document that `cargo xtask ci` is explicitly NOT the CUDA parity gate**.
- **CPU-validatable subset runs on every host (ungated):** native scalar-reference
  parity helpers, the comparison-harness structure, native determinism under
  `rayon`, fmt/clippy/compile. Only the CUDA-side byte-parity needs the runner.
- Keep existing per-stage parity tests; add the missing **facade byte-parity**
  tests, the **stage-level** DWT/RCT/quantize/deinterleave parity tests, and
  negative tests for every out-of-scope config.
- **Docs:** update `docs/architecture.md` + `docs/wsi-decode-api.md` to state CUDA
  lossless encode is at native parity, list the supported matrix and the tag-tree
  bound, and record the non-goals (tier-1, lossy, layers, multi-tile, subsampling,
  SigProp/MagRef).

## 8. Phasing (reordered — correctness before residency cosmetics)

- **Phase 1 — Validated byte-exact lossless parity.** C7 (`--fmad=false`) + C1
  (facade contract, no `Ok(None)` for in-scope) + C2 (4-component MCT code) + C3
  (byte-parity matrix as the gate + stage-level parity tests + native scalar-ref
  exports) + the fail-closed CI gates (§7) + the native 2-/4-component round-trip
  precondition tests. **This satisfies the full Definition of Done** (host
  assembly retained — framing is already byte-identical to native).
- **Phase 2 — Maximal coverage + hardening.** C5 (full bit-depth/component sweep)
  + C6 (fuzz harness) + the tag-tree boundary: grow/segment the GPU buffers to
  remove the ceiling (preferred), else land the typed rejection + boundary test.
- **Phase 3 — Optional residency purity.** C4 GPU assembly kernel **only if** a
  measured residency requirement appears; otherwise drop and document host
  assembly as intentional.

## 9. Risks

- **FP operation-ordering divergence** (not integer math): the real parity risk.
  Mitigated by C7 (`--fmad=false`) + stage-level + high-bit-depth + fuzz parity
  tests. (The v1 "integer math" assumption was wrong and is removed.)
- **Tag-tree capacity ceiling**: growing GPU buffers is real device work; if
  deferred, large in-scope subbands are rejected (documented), narrowing
  "coverage-maximal."
- **Native oracle untested for 2-/4-component**: guarded by the §7 native
  round-trip precondition — do not build parity on an unproven native path.
- **Native determinism under `rayon`**: asserted before trusting the byte oracle.
- **C4 (if built)** introduces a new parity surface with no native byte-vs-native
  assembler precedent (Metal tests only round-trip) — hence demoted/optional.

## 10. Open items to resolve in the implementation plan

- Tag-tree ceiling: **grow GPU buffers vs typed-reject** — pick during Phase 2 by
  feasibility/budget; default is grow.
- Confirm native round-trips 2- and 4-component HTJ2K lossless (precondition).
- Native 4-component behavior is now characterized (RCT planes 0–2 + passthrough
  3, `guard_bits.max(2)`); lock it with a native fixture before the CUDA path.
- Base-branch/merge logistics: encode pipeline on `codex/cuda-htj2k-runner`;
  reconcile with the in-flight test fixes on `codex/maturity-hardening`.

## 11. Key code references (evidence)

- f32 DWT (both sides): native `crates/signinum-j2k-native/src/j2c/fdwt.rs:51,219,235`;
  CUDA `codex/cuda-htj2k-runner:crates/signinum-cuda-runtime/src/j2k_encode_kernels.cu:142,180`.
- nvcc flags (no `--fmad=false`): `…/signinum-cuda-runtime/build.rs:59-60`.
- Facade (strict, reuses native framing): `…/signinum-j2k-cuda/src/encode.rs:32-46`;
  native marker writer `…/signinum-j2k-native/src/j2c/encode.rs:1266` →
  `codestream_write::write_codestream`.
- `Ok(None)` silent fallbacks: `…/signinum-j2k-cuda/src/encode.rs:2107-2115`;
  resident RCT requires 3 planes `…/signinum-cuda-runtime/src/lib.rs:2480`.
- Tag-tree cap (hard `Err`): `…/signinum-j2k-cuda/src/encode.rs:594-595,619,628`;
  native unbounded `…/signinum-j2k-native/src/j2c/tag_tree_encode.rs`,
  `packet_encode.rs:112`.
- Subsampling SIZ-only: `…/codestream_write.rs:153-161`; decoder grid sizing
  `…/codestream.rs:646-662`, `…/tile.rs:333-352`.
- Native single-tile/single-layer/cleanup-only: `codestream_write.rs:94`,
  `encode.rs:1658`, `ht_block_encode.rs`.
- 4-component MCT semantics: `forward_mct.rs:19-44`, `encode.rs:1199,1205-1206`.
- CI gate: `.github/workflows/gpu-validation.yml:16-20,30-43,83-89,111`.
- C4 reference (Metal, round-trip-only): `…/encode_bitstream.metal` assembly fn.

## 12. Review audit trail (workflow `w0n5v1e6m`)

Five lenses + adversarial verification + synthesis. Verdict: **go-with-changes**;
six blockers confirmed in code (none refuted), all folded above:
1. Wrong parity rationale (f32, not integer) — §5 corrected.
2. FMA contraction risk (no `--fmad=false`) — C7 added.
3. Byte-parity unproven (only round-trip tests today) — C3 made the Phase-1 gate.
4. Tag-tree cap hard-rejects in-scope subbands — §3 bound + §6 taxonomy + Phase 2.
5. Subsampling not round-trippable via `encode_htj2k` — §3 moved to definitive OUT.
6. CI false-green (silent skip, dispatch-only job) — §7 fail-closed gates.
Plus should-fixes: no `Ok(None)` for in-scope (§4 C1); native 2-/4-component
round-trip precondition (§3/§7); C4 demoted to optional (§4/§8).
