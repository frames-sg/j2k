# JPEG Hybrid Viewport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make a hybrid `CPU+Metal` JPEG viewport pipeline beat the `CPU-only` viewport pipeline on a real region+scaled composite benchmark.

**Architecture:** CPU keeps the serial JPEG stages and writes scaled component rows into Metal shared buffers. Metal packs those planes directly into a composited viewport in one command buffer. Benchmarks compare end-to-end `decode -> resize -> composite` instead of raw decode-only timings.

**Tech Stack:** Rust, Metal, Criterion, `slidecodec-jpeg`, `slidecodec-jpeg-metal`

---

## File Structure

- Create: `crates/slidecodec-jpeg-metal/src/viewport.rs`
  - owns viewport request types and CPU/hybrid viewport helpers
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
  - add viewport composite kernel and command-buffer encoding helpers
- Modify: `crates/slidecodec-jpeg-metal/src/lib.rs`
  - expose the new internal viewport module and helper entry points
- Create: `crates/slidecodec-jpeg-metal/tests/viewport.rs`
  - parity tests for CPU vs hybrid viewport assembly
- Modify: `crates/slidecodec-jpeg-metal/benches/compare.rs`
  - add viewer-style region+scaled composite benchmark group
- Modify: `crates/slidecodec-jpeg-metal/Cargo.toml`
  - add any benchmark-only dependencies if needed

## Task 1: Add viewport request types and CPU-only baseline

**Files:**
- Create: `crates/slidecodec-jpeg-metal/src/viewport.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/lib.rs`
- Test: `crates/slidecodec-jpeg-metal/tests/viewport.rs`

- [ ] Add `ViewportTileRequest` and `ViewportLayout` types.
- [ ] Implement a CPU-only helper that runs `decode_region_scaled_into_with_scratch` per tile and composites into one RGB viewport.
- [ ] Add a parity test covering a 2x2 viewport mosaic built from a real JPEG fixture.
- [ ] Run `cargo test -p slidecodec-jpeg-metal --test viewport`.
- [ ] Commit.

## Task 2: Add Metal viewport composite kernel

**Files:**
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
- Create: `crates/slidecodec-jpeg-metal/src/viewport.rs`
- Test: `crates/slidecodec-jpeg-metal/tests/viewport.rs`

- [ ] Add a `jpeg_pack_into_viewport` Metal kernel that packs one tile's planes directly into the viewport output buffer.
- [ ] Add helpers to build a plane stage without immediately finishing it into an intermediate surface.
- [ ] Add a `compose_viewport_hybrid(...)` helper that decodes all tile component rows, encodes all tile pack/composite passes into one command buffer, commits once, and returns the final viewport surface.
- [ ] Extend the viewport test to assert byte-for-byte parity between CPU-only and hybrid on the same tile layout.
- [ ] Run `cargo test -p slidecodec-jpeg-metal --test viewport`.
- [ ] Commit.

## Task 3: Add viewer-style compare benchmark

**Files:**
- Modify: `crates/slidecodec-jpeg-metal/benches/compare.rs`

- [ ] Add a benchmark group for `viewer_region_scaled_composite_rgb`.
- [ ] Use a deterministic viewport layout with multiple region+scaled requests into the same source JPEG.
- [ ] Benchmark both:
  - CPU-only viewport helper
  - Hybrid CPU+Metal viewport helper
- [ ] Ensure the group can run against real local corpora from `SLIDECODEC_BENCH_INPUTS`.
- [ ] Run `cargo bench -p slidecodec-jpeg-metal --bench compare --no-run`.
- [ ] Commit.

## Task 4: Tune until hybrid wins a real case

**Files:**
- Modify: `crates/slidecodec-jpeg-metal/src/compute.rs`
- Modify: `crates/slidecodec-jpeg-metal/src/viewport.rs`
- Modify: `crates/slidecodec-jpeg-metal/benches/compare.rs`

- [ ] Measure restart-coded and non-restart viewer composite workloads on local corpora.
- [ ] If hybrid still loses, reduce overhead in this order:
  - remove intermediate surfaces
  - keep one command buffer per viewport
  - reuse plane/output buffers within the helper
  - increase viewport tile count to reflect real viewer work rather than decode-only latency
- [ ] Re-run the compare bench after each material tuning change.
- [ ] Stop only when hybrid beats CPU-only on at least one real viewer configuration.
- [ ] Commit.

## Task 5: Final verification

**Files:**
- No new files required unless benchmark wording/doc text needs tightening

- [ ] Run:
  - `cargo fmt --all --check`
  - `cargo test -p slidecodec-jpeg`
  - `cargo test -p slidecodec-jpeg-metal`
  - `cargo clippy -p slidecodec-jpeg -p slidecodec-jpeg-metal --all-targets -- -D warnings`
  - `cargo bench -p slidecodec-jpeg-metal --bench compare --no-run`
- [ ] Capture the final CPU-only vs hybrid benchmark numbers used to justify the architecture.
- [ ] Commit any last wording fix if needed.
