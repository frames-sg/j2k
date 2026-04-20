# J2K-M4 SIMD / Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a J2K compare bench that exercises the slidecodec public API and an OpenJPEG reference path on deterministic J2K/HTJ2K inputs.

**Architecture:** runtime-generated bench codestreams, Criterion groups for the primary decode surfaces, and an env/default-path OpenJPEG CLI harness.

**Tech Stack:** Rust, Criterion, `slidecodec-j2k`, `dicom-toolkit-jpeg2000`, local OpenJPEG CLI

---

### Task 1: Add J2K bench scaffolding

**Files:**
- Modify: `crates/slidecodec-j2k/Cargo.toml`
- Create: `crates/slidecodec-j2k/benches/common/mod.rs`
- Create: `crates/slidecodec-j2k/benches/compare.rs`

- [ ] **Step 1: Add the bench target and shared helpers**
- [ ] **Step 2: Generate deterministic J2K/HTJ2K inputs at runtime**
- [ ] **Step 3: Add an OpenJPEG CLI harness**
- [ ] **Step 4: Run bench compilation**
Run: `cargo bench -p slidecodec-j2k --bench compare --no-run`

### Task 2: Document the signoff path

**Files:**
- Modify: `docs/bench.md`

- [ ] **Step 1: Add a J2K benchmark section**
- [ ] **Step 2: Document OpenJPEG comparator expectations and local binary discovery**
- [ ] **Step 3: Re-run workspace verification**
Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

### Task 3: Commit

- [ ] **Step 1: Commit**
Commit message: `bench: add j2k compare harness`
