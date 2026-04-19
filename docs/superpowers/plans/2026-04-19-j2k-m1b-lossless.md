# J2K-M1b Lossless Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make reversible 5-3 / RCT support an explicit, tested milestone on top of the committed J2K-M1 decode path.

**Architecture:** keep the existing backend decode adapter; add the missing reversible-transform parser state and assert the lossless behavior with focused inspect/decode tests.

**Tech Stack:** Rust, `slidecodec-core`, `slidecodec-j2k`, `dicom-toolkit-jpeg2000`

---

### Task 1: Add the failing reversible inspect test

**Files:**
- Modify: `crates/slidecodec-j2k/tests/inspect.rs`
- Modify: `crates/slidecodec-j2k/src/parse/mod.rs`
- Modify: `crates/slidecodec-j2k/src/parse/codestream.rs`

- [ ] **Step 1: Write a reversible raw-codestream inspect test**

Add a raw codestream fixture where:
- `components == 3`
- `MCT` is enabled
- wavelet transform is reversible

Assert `Info.colorspace == Colorspace::Rct`.

- [ ] **Step 2: Run the inspect test to verify it fails**

Run: `cargo test -p slidecodec-j2k --test inspect`

- [ ] **Step 3: Parse and store the reversible-transform bit**

Extend `ParsedCod` and colorspace inference so reversible+MCT becomes `Rct`.

- [ ] **Step 4: Re-run the inspect tests**

Run: `cargo test -p slidecodec-j2k --test inspect`

- [ ] **Step 5: Commit**

Commit message: `fix: infer reversible j2k codestream colorspace`

### Task 2: Re-verify lossless decode coverage and workspace green state

**Files:**
- Modify: `crates/slidecodec-j2k/tests/decode.rs` only if any reversible decode coverage is still missing

- [ ] **Step 1: Confirm the exact reversible decode tests exist for grayscale and RGB native-depth paths**

If needed, add or tighten tests so reversible native-depth output remains byte-exact.

- [ ] **Step 2: Run milestone verification**

Run:
- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`

- [ ] **Step 3: Commit**

Commit message: `test: lock in j2k lossless decode coverage`
