# M8 Tilecodec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add `slidecodec-tilecodec` with Deflate, Zstd, LZW, and Uncompressed tile decompression plus tests, bench wiring, and release-gate integration.

**Architecture:** a new workspace crate exposing four `TileDecompress` implementations with typed pools and one shared error type.

**Tech Stack:** Rust, `flate2`, `zstd`, `weezl`, Criterion

---

### Task 1: Create the crate surface

**Files:**
- Create: `crates/slidecodec-tilecodec/Cargo.toml`
- Create: `crates/slidecodec-tilecodec/src/lib.rs`
- Create: `crates/slidecodec-tilecodec/src/error.rs`
- Create: `crates/slidecodec-tilecodec/src/pool.rs`
- Create: `crates/slidecodec-tilecodec/src/deflate.rs`
- Create: `crates/slidecodec-tilecodec/src/zstd.rs`
- Create: `crates/slidecodec-tilecodec/src/lzw.rs`
- Create: `crates/slidecodec-tilecodec/src/uncompressed.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the workspace member and crate metadata**
- [ ] **Step 2: Define pools, error type, and public re-exports**
- [ ] **Step 3: Implement `TileDecompress` for all four codecs**

### Task 2: Add tests and fuzz scaffold

**Files:**
- Create: `crates/slidecodec-tilecodec/tests/decompress.rs`
- Create: `crates/slidecodec-tilecodec/fuzz/Cargo.toml`
- Create: `crates/slidecodec-tilecodec/fuzz/fuzz_targets/decompress_fuzz.rs`

- [ ] **Step 1: Add roundtrip tests for Deflate, Zstd, and LZW**
- [ ] **Step 2: Add output-too-small and pool-reuse tests**
- [ ] **Step 3: Add a fuzz target that exercises all codecs on arbitrary bytes**
- [ ] **Step 4: Run targeted tests**
Run: `cargo test -p slidecodec-tilecodec`

### Task 3: Add compare bench and release-gate wiring

**Files:**
- Create: `crates/slidecodec-tilecodec/benches/compare.rs`
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/bench.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `crates/slidecodec-cli/Cargo.toml` and/or CLI docs only if needed for workspace packaging

- [ ] **Step 1: Add a throughput compare bench against direct library usage**
- [ ] **Step 2: Extend CI bench/fuzz-build coverage to the new crate**
- [ ] **Step 3: Update top-level docs to describe the full codec stack**
- [ ] **Step 4: Run final milestone verification**
Run:
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo check --manifest-path crates/slidecodec-tilecodec/fuzz/Cargo.toml`
- `cargo bench -p slidecodec-j2k --bench compare --no-run`
- `cargo bench -p slidecodec-tilecodec --bench compare --no-run`
- `cargo deny check`

### Task 4: Release-gate cleanup

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/slidecodec-cli/src/main.rs`

- [ ] **Step 1: If the JPEG + native J2K release gate is actually complete, bump workspace package version to `1.0.0`**
- [ ] **Step 2: Update CLI banner/help to match the 1.0 workspace state**
- [ ] **Step 3: Commit**
Commit message: `feat: add slidecodec tile decompression crate`
