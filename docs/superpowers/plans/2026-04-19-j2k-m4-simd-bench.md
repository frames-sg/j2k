# J2K-M4 SIMD / Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add SIMD acceleration to the J2K hot path and verify it against the scalar decoder on deterministic inputs.

**Architecture:** keep the scalar decoder as the reference behavior, add NEON and AVX2 kernels for the hot loops, and use Criterion only as a regression and sanity check.

**Tech Stack:** Rust, Criterion, `slidecodec-j2k`

---

### Task 1: Add SIMD kernel scaffolding

**Files:**
- Modify: `crates/slidecodec-j2k/src/backend/mod.rs`
- Modify: `crates/slidecodec-j2k/src/lib.rs`
- Modify: `crates/slidecodec-j2k/tests/decode.rs`

- [ ] **Step 1: Add failing tests that exercise the SIMD dispatch points**
- [ ] **Step 2: Run the tests to verify the SIMD paths are missing**
Run: `cargo test -p slidecodec-j2k`
- [ ] **Step 3: Add dispatch scaffolding for scalar, NEON, and AVX2 kernels**
- [ ] **Step 4: Re-run the targeted tests**
Run: `cargo test -p slidecodec-j2k`
- [ ] **Step 5: Commit**
Commit message: `feat: add j2k simd dispatch scaffolding`

### Task 2: Implement the hot SIMD kernels

**Files:**
- Modify: `crates/slidecodec-j2k/src/backend/scalar.rs`
- Modify: `crates/slidecodec-j2k/src/backend/x86.rs`
- Modify: `crates/slidecodec-j2k/src/backend/neon.rs`
- Modify: `crates/slidecodec-j2k/tests/decode.rs`

- [ ] **Step 1: Add parity tests for SIMD vs scalar decode outputs**
- [ ] **Step 2: Run them to verify they fail on the missing kernels**
Run: `cargo test -p slidecodec-j2k`
- [ ] **Step 3: Implement the SIMD DWT, Tier-1, and color kernels**
- [ ] **Step 4: Re-run the parity tests**
Run: `cargo test -p slidecodec-j2k`
- [ ] **Step 5: Commit**
Commit message: `feat: add j2k simd kernels`

### Task 3: Confirm bench coverage as a regression check

**Files:**
- Modify: `crates/slidecodec-j2k/benches/common/mod.rs`
- Modify: `crates/slidecodec-j2k/benches/compare.rs`

- [ ] **Step 1: Keep the compare bench focused on exercising the SIMD path**
- [ ] **Step 2: Run bench compilation**
Run: `cargo bench -p slidecodec-j2k --bench compare --no-run`
- [ ] **Step 3: Run workspace verification**
Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] **Step 4: Commit**
Commit message: `bench: verify j2k simd path`
