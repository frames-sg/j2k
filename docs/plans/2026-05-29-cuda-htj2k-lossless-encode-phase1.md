# CUDA HTJ2K Lossless Encode — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the strict CUDA HTJ2K lossless encoder (`encode_j2k_lossless_with_cuda`) produce a codestream **byte-identical to `signinum_j2k_native::encode_htj2k`** for the in-scope lossless matrix, with no silent CPU fallback and tests that genuinely run (not skip) on the CUDA runner.

**Architecture:** The resident GPU engine already round-trips; Phase 1 adds (a) a build-determinism flag, (b) the missing 4-component resident MCT code path, (c) a byte-parity test matrix as the acceptance gate plus stage-level parity tests, (d) native scalar-reference exports + native precondition tests, and (e) fail-closed CI. Codestream framing stays on host (native `write_codestream`) — already byte-identical by construction.

**Tech Stack:** Rust (workspace), CUDA C++ kernels compiled to PTX via `nvcc`, `signinum-j2k-native` (CPU reference), `signinum-cuda-runtime` (FFI + kernels), `signinum-j2k-cuda` (accelerator + facade), GitHub Actions self-hosted CUDA runner.

**Source spec:** `docs/specs/2026-05-29-cuda-htj2k-lossless-encode-completion-design.md` (Approach C, v2 post-review).

**Verification reality:** No GPU/nvcc exists in the dev environment. For every CUDA-gated task, *local* verification = `cargo build`/`clippy`/`fmt` + the CPU-side (native) logic; *authoritative* verification = the `cuda-x86_64-compatibility` job on the runner with `SIGNINUM_REQUIRE_CUDA_RUNTIME=1`. Each task states both.

---

## Branch & file structure

**Base branch:** `codex/cuda-htj2k-runner` (the only branch with the CUDA encode pipeline). **Task 0 resolves how we get there** — do not start Task 1 until it's settled.

Files touched in Phase 1:
- `crates/signinum-cuda-runtime/build.rs` — add `--fmad=false` (Task 1).
- `crates/signinum-j2k-native/src/j2c/encode.rs` + `lib.rs` — new `pub(crate)`/`pub` scalar-reference exports for forward DWT 5/3, RCT, reversible quantize, deinterleave (Task 4); native precondition tests (Tasks 2–3).
- `crates/signinum-j2k-cuda/src/encode.rs` — 4-component resident MCT (Task 7); facade typed-rejection contract + no-`Ok(None)` invariant (Task 8).
- `crates/signinum-cuda-runtime/src/lib.rs` — resident RCT wrapper accepting 3-of-N planes (Task 7).
- `crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs` *(new)* — stage-level parity (Task 5) + facade byte-parity matrix (Task 6) + negative tests (Task 8) + tripwire (Task 9).
- `.github/workflows/gpu-validation.yml` — fail-closed gates + trigger (Task 9).
- `docs/architecture.md`, `docs/wsi-decode-api.md` — parity statement + non-goals (Task 10).

---

## Task 0: Resolve the implementation base branch

**This is a decision, not code.** The encode pipeline lives on `codex/cuda-htj2k-runner`; the current branch `codex/maturity-hardening` does not have it. Pick one (user decides):
- **(A)** Implement on `codex/cuda-htj2k-runner` directly.
- **(B)** Merge `codex/cuda-htj2k-runner` into the working branch first, then implement.
- **(C)** Merge the working branch's in-flight test fixes into `codex/cuda-htj2k-runner`, then implement there.

- [ ] **Step 1:** Confirm the choice with the user; record it here.
- [ ] **Step 2:** Get onto that base; confirm the encode pipeline files exist:

Run: `git rev-parse --abbrev-ref HEAD && ls crates/signinum-j2k-cuda/src/encode.rs crates/signinum-cuda-runtime/src/htj2k_encode_kernels.cu`
Expected: the chosen branch name, and both files listed (no "No such file").

- [ ] **Step 3:** Baseline green (local, non-CUDA): `cargo build --workspace` succeeds (PTX falls back to checked-in `.ptx` when nvcc is absent). Record the baseline commit hash.

---

## Task 1: Build determinism — disable FMA contraction (C7)

**Files:** Modify `crates/signinum-cuda-runtime/build.rs` (the `compile_or_copy_ptx` nvcc args).

Rationale: native (Rust/LLVM) does **not** contract `a*b+c` into a single-rounding FMA by default; `nvcc` **does**. The 5/3 DWT update `floorf((left+right)*0.25f + 0.5f)` and RCT `(r+2g+b)*0.25f` would then diverge from native at high magnitude / deep transforms, silently breaking byte parity.

- [ ] **Step 1: Change the nvcc args.** In `compile_or_copy_ptx`, change:

```rust
    let compiled = Command::new(nvcc)
        .args(["--ptx", "-O3", "--std=c++14"])
```
to:
```rust
    let compiled = Command::new(nvcc)
        // --fmad=false: native (Rust/LLVM) does not contract a*b+c into a single-rounding
        // FMA; nvcc does by default. Disabling it keeps the f32 DWT/RCT lossless path
        // bit-identical to the native reference (byte-parity requirement).
        .args(["--ptx", "-O3", "--std=c++14", "--fmad=false"])
```

- [ ] **Step 2: Verify it compiles (local).** Run: `cargo build -p signinum-cuda-runtime`
Expected: success (nvcc absent → fallback PTX path; the arg change is inert locally but recorded for the runner).

- [ ] **Step 3: Note for the runner.** On the runner, the PTX is rebuilt with `--fmad=false`. The high-bit-depth parity test in Task 6 (16-bit signed, multi-level) is what actually proves this; reference it here.

- [ ] **Step 4: Commit.**
```bash
git add crates/signinum-cuda-runtime/build.rs
git commit -m "build: nvcc --fmad=false for byte-exact f32 parity with native"
```

---

## Task 2: Native determinism precondition (CPU-only, runs everywhere)

The byte-parity oracle is `native::encode_htj2k`, which uses `rayon` (`encode.rs:2205,2237`). If its output is non-deterministic across thread counts, parity is meaningless. Prove determinism.

**Files:** Test in `crates/signinum-j2k-native/src/j2c/encode.rs` (`#[cfg(test)] mod tests`).

- [ ] **Step 1: Write the failing test.**
```rust
#[test]
fn encode_htj2k_is_deterministic_across_thread_counts() {
    // 3-component 8-bit, sizes chosen to exercise multi-codeblock subbands.
    let (w, h, c, depth) = (96u32, 80u32, 3u8, 8u8);
    let pixels: Vec<u8> = (0..(w * h * c as u32))
        .map(|i| ((i * 2654435761) >> 13) as u8) // deterministic pseudo-random
        .collect();
    let opts = EncodeOptions::default();
    let run = || encode_htj2k(&pixels, w, h, c, depth, false, &opts).expect("encode");
    let baseline = run();
    for threads in [1usize, 2, 4, 8] {
        let out = std::thread::scope(|_| {
            // rayon honors RAYON_NUM_THREADS via a scoped pool if configured;
            // if the crate builds its own pool, assert determinism of repeated calls instead.
            run()
        });
        assert_eq!(out, baseline, "non-deterministic at {threads} threads");
    }
}
```
(If the crate pins its own rayon pool, simplify to asserting `run() == run()` across 8 repeats — the point is byte-stable output.)

- [ ] **Step 2: Run (local).** Run: `cargo test -p signinum-j2k-native encode_htj2k_is_deterministic -- --nocapture`
Expected: PASS. If it FAILS, **stop** — byte-parity is not well-defined; escalate to the user before proceeding.

- [ ] **Step 3: Commit.**
```bash
git add crates/signinum-j2k-native/src/j2c/encode.rs
git commit -m "test: native encode_htj2k determinism precondition for byte parity"
```

---

## Task 3: Native round-trip precondition for 2- and 4-component (CPU-only)

Native has no existing 2-/4-component HTJ2K round-trip coverage. Before those become CUDA parity targets, prove native itself round-trips them; otherwise they are out-of-scope.

**Files:** Test in `crates/signinum-j2k-native/src/j2c/encode.rs`.

- [ ] **Step 1: Write the failing tests.**
```rust
#[test]
fn native_htj2k_roundtrips_two_component_lossless() {
    let (w, h, c, depth) = (32u32, 24u32, 2u8, 8u8);
    let pixels: Vec<u8> = (0..(w * h * c as u32)).map(|i| (i % 251) as u8).collect();
    let bytes = encode_htj2k(&pixels, w, h, c, depth, false, &EncodeOptions::default())
        .expect("encode 2-component");
    let decoded = Image::new(&bytes, /* args per existing helpers */).unwrap().decode_native().unwrap();
    assert_eq!(decoded.data, pixels);
}

#[test]
fn native_htj2k_roundtrips_four_component_lossless() {
    let (w, h, c, depth) = (32u32, 24u32, 4u8, 8u8);
    let pixels: Vec<u8> = (0..(w * h * c as u32)).map(|i| (i % 251) as u8).collect();
    let bytes = encode_htj2k(&pixels, w, h, c, depth, false, &EncodeOptions::default())
        .expect("encode 4-component");
    let decoded = Image::new(&bytes, /* args */).unwrap().decode_native().unwrap();
    assert_eq!(decoded.data, pixels);
}
```
(Use the existing round-trip helpers `lossless_htj2k_roundtrip_u8` / `validate_htj2k_codestream` as the template for the exact `Image::new` arguments.)

- [ ] **Step 2: Run (local).** Run: `cargo test -p signinum-j2k-native native_htj2k_roundtrips_ -- --nocapture`
Expected: PASS for both. If either FAILS, mark that component count **out-of-scope** in the spec (§3) and skip its CUDA parity cells; report to the user.

- [ ] **Step 3: Commit.**
```bash
git add crates/signinum-j2k-native/src/j2c/encode.rs
git commit -m "test: native 2-/4-component HTJ2K lossless round-trip precondition"
```

---

## Task 4: Native scalar-reference exports for stage parity (CPU-only)

To test CUDA stages against native at stage granularity (not just whole-codestream), native must expose reference functions for forward DWT 5/3, RCT, reversible quantize, and deinterleave — mirroring the existing `encode_ht_code_block_scalar` / `encode_j2k_packetization_scalar` exports.

**Files:** Modify `crates/signinum-j2k-native/src/j2c/encode.rs` (add `pub(crate)` thin wrappers) and re-export via `crates/signinum-j2k-native/src/lib.rs` behind a `pub mod scalar_reference` or existing test-support surface.

- [ ] **Step 1: Add reference wrappers.** Expose (names illustrative; match existing export style):
```rust
pub fn forward_dwt53_reference(samples: &[f32], width: usize, height: usize, levels: u8) -> J2kForwardDwt53Output { /* call fdwt::forward_dwt */ }
pub fn forward_rct_reference(planes: &mut [Vec<f32>]) { forward_mct::forward_rct(planes) }
pub fn quantize_reversible_reference(coeffs: &[f32]) -> Vec<i32> { /* quantize::quantize_subband reversible path */ }
pub fn deinterleave_reference(pixels: &[u8], w: u32, h: u32, comps: u8, depth: u8, signed: bool) -> Vec<Vec<f32>> { /* deinterleave_to_f32 */ }
```

- [ ] **Step 2: Compile (local).** Run: `cargo build -p signinum-j2k-native`
Expected: success.

- [ ] **Step 3: Sanity test (local).** Add one trivial test per export asserting it equals the already-tested internal path on a 4-value input. Run: `cargo test -p signinum-j2k-native _reference`
Expected: PASS.

- [ ] **Step 4: Commit.**
```bash
git add crates/signinum-j2k-native/src/j2c/encode.rs crates/signinum-j2k-native/src/lib.rs
git commit -m "feat: native scalar-reference exports for CUDA stage parity tests"
```

---

## Task 5: Stage-level CUDA-vs-native parity tests (runner)

**Files:** New `crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs`.

Each test is gated on `SIGNINUM_REQUIRE_CUDA_RUNTIME` (use the existing `cuda_runtime_required()` helper / pattern), runs the CUDA stage on device, and asserts byte/value identity vs the Task-4 native reference on the **same** input.

- [ ] **Step 1: Write the failing tests** (one per stage: deinterleave, RCT, DWT53, quantize). Example for DWT53:
```rust
#[test]
fn cuda_forward_dwt53_matches_native_reference_when_required() {
    if !cuda_runtime_required() { return; }
    let (w, h, levels) = (40usize, 24usize, 2u8);
    let samples: Vec<f32> = (0..w * h).map(|i| (i as f32 % 37.0) - 18.0).collect();
    let expected = signinum_j2k_native::forward_dwt53_reference(&samples, w, h, levels);
    let ctx = CudaContext::system_default().expect("ctx");
    let actual = ctx.j2k_forward_dwt53(&samples, w as u32, h as u32, levels).expect("cuda dwt");
    assert_eq!(actual.transformed(), expected.coefficients_as_slice()); // exact f32 bits
}
```
Repeat for RCT (3-plane), reversible quantize, and deinterleave (8/16-bit, signed/unsigned).

- [ ] **Step 2: Compile + clippy (local).** Run: `cargo test -p signinum-j2k-cuda --no-run --features cuda-runtime` and `cargo clippy -p signinum-j2k-cuda --features cuda-runtime -- -D warnings`
Expected: compiles clean; tests early-return locally (no GPU).

- [ ] **Step 3: Runner verification.** On the runner (`SIGNINUM_REQUIRE_CUDA_RUNTIME=1`): all four PASS with real assertions. This is where `--fmad=false` (Task 1) is first exercised.

- [ ] **Step 4: Commit.**
```bash
git add crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs
git commit -m "test: CUDA stage-level parity vs native (dwt53/rct/quantize/deinterleave)"
```

---

## Task 6: Facade byte-parity matrix — the acceptance gate (runner)

**Files:** Extend `crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs`.

The core deliverable: `encode_j2k_lossless_with_cuda` output **equals** `native::encode_htj2k` byte-for-byte, then round-trips.

- [ ] **Step 1: Write the matrix harness.**
```rust
struct Cell { w: u32, h: u32, comps: u8, depth: u8, signed: bool, levels: u8 }

fn parity_cells() -> Vec<Cell> {
    let mut v = vec![];
    for &(comps) in &[1u8, 2, 3, 4] {
        for &depth in &[8u8, 16] {
            for &signed in &[false, true] {
                for &levels in &[0u8, 1, 3] {
                    v.push(Cell { w: 64, h: 48, comps, depth, signed, levels });
                }
            }
        }
    }
    v
}

#[test]
fn cuda_facade_byte_matches_native_across_matrix_when_required() {
    if !cuda_runtime_required() { return; }
    for c in parity_cells() {
        let pixels = synth_pixels(&c); // deterministic per-cell bytes
        let bytes_native = native_encode_htj2k(&c, &pixels); // helper -> encode_htj2k
        let bytes_cuda = encode_j2k_lossless_with_cuda(samples(&c, &pixels), &opts(&c))
            .expect(&format!("cuda encode {c:?}"));
        assert_eq!(bytes_cuda.as_bytes(), bytes_native.as_slice(), "byte mismatch for {c:?}");
        let decoded = decode_native(&bytes_cuda);
        assert_eq!(decoded, pixels, "round-trip mismatch for {c:?}");
    }
}
```
(Skip 2-/4-component cells if Task 3 found native doesn't round-trip them; `log!` the skip — never silently drop.)

- [ ] **Step 2: Compile + clippy (local).** Run: `cargo test -p signinum-j2k-cuda --no-run --features cuda-runtime`
Expected: compiles. Locally early-returns.

- [ ] **Step 3: Runner verification.** On the runner: PASS for every cell. **This is the Definition-of-Done gate.** Failures here drive Tasks 7–8.

- [ ] **Step 4: Commit.**
```bash
git add crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs
git commit -m "test: facade byte-exact parity matrix vs native (acceptance gate)"
```

---

## Task 7: 4-component resident MCT (C2) — driven by the failing matrix cell

**Files:** `crates/signinum-cuda-runtime/src/lib.rs` (resident RCT wrapper, currently requires exactly 3 planes, ~:2480); `crates/signinum-j2k-cuda/src/encode.rs` (`encode_htj2k_tile` returns `Ok(None)` for `use_mct && num_components != 3`, ~:2113).

Native semantics to match: RCT on planes 0–2, plane 3 passed through; `guard_bits = guard_bits.max(2)` under MCT; quant params already computed by native and handed to the accelerator (no recompute).

- [ ] **Step 1: Confirm the red.** The 4-component cells in Task 6 must currently fail (today `encode_htj2k_tile` returns `Ok(None)` → strict facade errors). Note the exact failure from a runner run (or reason it from `encode.rs:2113`).

- [ ] **Step 2: Widen the resident RCT wrapper.** In `lib.rs`, change the wrapper that requires exactly 3 component planes to accept `num_components >= 3` and apply the 3-plane RCT kernel to planes 0–2 only, leaving any 4th plane untouched. Keep the existing 3-plane kernel; only the host-side plane-count guard changes.

- [ ] **Step 3: Remove the tile-body `Ok(None)`.** In `encode.rs`, replace `if job.use_mct && job.num_components != 3 { return Ok(None); }` with: handle `num_components == 4` on-device (RCT 0–2 + passthrough 3) returning `Some(...)`; for any genuinely unsupported count return a typed `Err`, **never `Ok(None)`**.

- [ ] **Step 4: Compile + clippy (local).** Run: `cargo build -p signinum-cuda-runtime -p signinum-j2k-cuda --features cuda-runtime` and clippy `-D warnings`.
Expected: clean.

- [ ] **Step 5: Runner verification.** The 4-component matrix cells in Task 6 now PASS byte-exact.

- [ ] **Step 6: Commit.**
```bash
git add crates/signinum-cuda-runtime/src/lib.rs crates/signinum-j2k-cuda/src/encode.rs
git commit -m "feat: 4-component resident MCT (RCT planes 0-2, passthrough 3) for byte parity"
```

---

## Task 8: Facade contract — typed rejections, no silent `Ok(None)` (C1)

**Files:** `crates/signinum-j2k-cuda/src/encode.rs`; negative tests in `tests/htj2k_encode_parity.rs`.

- [ ] **Step 1: Write the failing negative tests** asserting the **specific** typed error (not a generic one):
```rust
#[test]
fn cuda_facade_rejects_classic_tier1_with_typed_error_when_required() {
    if !cuda_runtime_required() { return; }
    let err = encode_j2k_lossless_with_cuda(classic_samples(), &classic_opts()).unwrap_err();
    assert!(matches!(err, Error::UnsupportedCudaRequest { reason } if reason.contains("HTJ2K")));
}
// + lossy 9/7, num_layers!=1, subsampling!=(1,1)
```

- [ ] **Step 2: Audit every accelerator hook** for in-scope inputs that return `Ok(None)`; convert each to `Some` (handled) or typed `Err`. The 4-comp path is Task 7; ensure subsampling and any other in-scope `None` become typed `Err`.

- [ ] **Step 3: Compile + clippy (local).** Run: `cargo build -p signinum-j2k-cuda --features cuda-runtime`; clippy clean.

- [ ] **Step 4: Runner verification.** Negative tests PASS (specific errors); the matrix (Task 6) still green.

- [ ] **Step 5: Commit.**
```bash
git add crates/signinum-j2k-cuda/src/encode.rs crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs
git commit -m "feat: strict facade typed rejections, no silent Ok(None) for in-scope inputs"
```

---

## Task 9: Fail-closed CI so parity can't false-green

**Files:** `.github/workflows/gpu-validation.yml`; tripwire test in `tests/htj2k_encode_parity.rs`.

Today: the `cuda-x86_64-compatibility` job sets `SIGNINUM_REQUIRE_CUDA_RUNTIME=1` and runs the parity tests, but the workflow is `workflow_dispatch`-only and nothing prevents a silently-skipped test from reading as green.

- [ ] **Step 1: Add an ungated tripwire test** (compiled always, no GPU needed):
```rust
#[test]
fn cuda_runtime_required_implies_feature_compiled() {
    if std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some() {
        assert!(cfg!(feature = "cuda-runtime"),
            "SIGNINUM_REQUIRE_CUDA_RUNTIME set but cuda-runtime feature not compiled — tests would silently skip");
    }
}
```

- [ ] **Step 2: Add fail-closed CI steps** on `cuda-x86_64-compatibility`, before/after the parity test step:
  - assert the env is set: `- run: test -n "$SIGNINUM_REQUIRE_CUDA_RUNTIME"`
  - run the parity tests with nextest JSON and assert an executed-count floor (e.g. ≥ number of matrix cells): a step that parses `cargo nextest run -p signinum-j2k-cuda --features cuda-runtime --message-format libtest-json` and fails if executed `< EXPECTED_MIN`.
- [ ] **Step 3: Add a merge trigger** (policy — confirm with user): add `pull_request:`/`merge_group:` (or a required-status equivalent) so the parity job runs on merge, not only manual dispatch. If the team prefers manual-only for cost, instead document in `gpu-validation.yml` that `cargo xtask ci` is **not** the CUDA parity gate and the parity job must be run before merge.

- [ ] **Step 4: Verify (local).** `cargo test -p signinum-j2k-cuda cuda_runtime_required_implies_feature_compiled` PASS (ungated). YAML lints/parses.

- [ ] **Step 5: Commit.**
```bash
git add .github/workflows/gpu-validation.yml crates/signinum-j2k-cuda/tests/htj2k_encode_parity.rs
git commit -m "ci: fail-closed CUDA parity gate (env assert, executed-count floor, feature tripwire)"
```

---

## Task 10: Docs

**Files:** `docs/architecture.md`, `docs/wsi-decode-api.md`.

- [ ] **Step 1:** State that CUDA HTJ2K **lossless** encode is byte-exact with native across the supported matrix (1–4 comp, 8–16 bit, signed/unsigned, multi-level DWT, single tile/layer), note the tag-tree bound, and record non-goals (tier-1, lossy 9/7, layers, multi-tile, subsampling, SigProp/MagRef).
- [ ] **Step 2:** `cargo xtask typos` clean (if available locally) or note for runner.
- [ ] **Step 3: Commit.**
```bash
git add docs/architecture.md docs/wsi-decode-api.md
git commit -m "docs: CUDA HTJ2K lossless encode at native byte-parity (scope + non-goals)"
```

---

## Self-review checklist (run before handoff)
- **Spec coverage:** C1→Task 8; C2→Task 7; C3→Tasks 5–6; C7→Task 1; native preconditions→Tasks 2–4; CI fail-closed→Task 9; docs→Task 10. (C4/C5/C6 + tag-tree growth = Phase 2/3, separate plans.)
- **No placeholders:** kernel-body specifics in Task 7 are intentionally approach-level because exact bit-perfect code must be developed against the live file under the Task-6 gate (TDD red→green); all *test* code is concrete.
- **Type consistency:** `encode_j2k_lossless_with_cuda`, `Error::UnsupportedCudaRequest`, `cuda_runtime_required()`, `encode_htj2k`, `*_reference` exports used consistently across tasks.
