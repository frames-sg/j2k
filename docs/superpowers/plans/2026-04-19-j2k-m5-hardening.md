# J2K-M5 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** close the J2K work with external parity checks, stress regressions, and CI hooks.

**Architecture:** OpenJPEG CLI differential tests for classic J2K, focused edge-case regressions for the public API, and CI updates that keep the J2K bench/fuzz/build surfaces compiling.

**Tech Stack:** Rust, GitHub Actions, local OpenJPEG CLI

---

### Task 1: Add OpenJPEG differential tests

**Files:**
- Create: `crates/slidecodec-j2k/tests/openjpeg_parity.rs`

- [ ] **Step 1: Add parity tests for full/region/scaled classic J2K decode**
- [ ] **Step 2: Skip cleanly when OpenJPEG binaries are unavailable**
- [ ] **Step 3: Run targeted parity tests**
Run: `cargo test -p slidecodec-j2k --test openjpeg_parity`

### Task 2: Add stress regressions

**Files:**
- Modify: `crates/slidecodec-j2k/tests/decode.rs`

- [ ] **Step 1: Add out-of-bounds ROI regression coverage**
- [ ] **Step 2: Re-run targeted decode tests**
Run: `cargo test -p slidecodec-j2k --test decode`

### Task 3: Update CI and final verification

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Cargo.lock`

- [ ] **Step 1: Add J2K bench-build and fuzz-check coverage**
- [ ] **Step 2: Run final milestone verification**
Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo bench -p slidecodec-j2k --bench compare --no-run`
- [ ] **Step 3: Commit**
Commit message: `test: harden slidecodec-j2k`
