# Metal DCT 9/7 Transcode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Metal-first hybrid accelerator for JPEG DCT to HTJ2K 9/7 transcode while preserving the scalar Rust path as oracle and fallback.

**Architecture:** Add a new `signinum-transcode-metal` crate that implements `DctToWaveletStageAccelerator`. The crate accelerates only `dct_grid_to_dwt97` first, leaves `dct_grid_to_dwt53` as scalar fallback, and keeps CPU/Rayon ownership of JPEG extraction and tile scheduling.

**Tech Stack:** Rust 1.94, Cargo workspace, `metal` crate on macOS, `rayon` for CPU orchestration where useful, existing `signinum-transcode` scalar oracle, Criterion benchmarks.

---

## File Structure

- `Cargo.toml`: add `crates/signinum-transcode-metal` workspace member.
- `crates/signinum-transcode-metal/Cargo.toml`: new crate metadata, macOS-only `metal` dependency, `criterion` dev dependency.
- `crates/signinum-transcode-metal/src/lib.rs`: public API, error type, accelerator struct, constructors, non-macOS fallback behavior.
- `crates/signinum-transcode-metal/src/weights.rs`: CPU construction of reusable 9/7 projection weight rows. This mirrors scalar lifting semantics.
- `crates/signinum-transcode-metal/src/metal.rs`: macOS host runtime, Metal buffer upload/download, dispatch routing.
- `crates/signinum-transcode-metal/src/dct97.metal`: Metal kernel computing one output coefficient per thread.
- `crates/signinum-transcode-metal/tests/dct97.rs`: scalar-vs-Metal coefficient tests and non-macOS fallback tests.
- `crates/signinum-transcode-metal/tests/jpeg_to_htj2k.rs`: JPEG 4:2:0 to Metal 9/7 to HTJ2K integration test.
- `crates/signinum-transcode-metal/benches/dct97_metal.rs`: Criterion scalar vs Metal benchmark groups.
- `crates/signinum-transcode-metal/tests/bench_harness.rs`: stable benchmark-name checks.
- `docs/dct-to-htj2k-notes.md`: document Metal routing, validation, and benchmark evidence.

## Task 1: Crate Skeleton And Non-macOS Fallback

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/signinum-transcode-metal/Cargo.toml`
- Create: `crates/signinum-transcode-metal/src/lib.rs`
- Create: `crates/signinum-transcode-metal/tests/dct97.rs`

- [ ] **Step 1: Write the failing fallback/API test**

Add this to `crates/signinum-transcode-metal/tests/dct97.rs`:

```rust
use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use signinum_transcode_metal::{MetalDctToWaveletStageAccelerator, MetalTranscodeError};

#[test]
fn explicit_metal_reports_unavailable_on_non_macos() {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let blocks = vec![[[0.0; 8]; 8]];
    let result = accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 1,
        block_rows: 1,
        width: 8,
        height: 8,
    });

    #[cfg(not(target_os = "macos"))]
    assert_eq!(
        result.expect_err("explicit Metal is unavailable off macOS"),
        MetalTranscodeError::MetalUnavailable.as_static_str()
    );

    #[cfg(target_os = "macos")]
    let _ = result;
}

#[test]
fn auto_metal_falls_back_for_tiny_jobs() {
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();
    let blocks = vec![[[0.0; 8]; 8]];
    let output = accelerator
        .dct_grid_to_dwt97(DctGridToDwt97Job {
            blocks: &blocks,
            block_cols: 1,
            block_rows: 1,
            width: 8,
            height: 8,
        })
        .expect("auto accelerator can decline tiny job");

    assert!(output.is_none());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97
```

Expected: package `signinum-transcode-metal` does not exist.

- [ ] **Step 3: Add crate skeleton and fallback accelerator**

Create the crate and implement:

```rust
pub const METAL_UNAVAILABLE: &str = "Metal is unavailable on this host";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTranscodeError {
    MetalUnavailable,
    UnsupportedJob(&'static str),
    Kernel(&'static str),
}

impl MetalTranscodeError {
    pub const fn as_static_str(self) -> &'static str {
        match self {
            Self::MetalUnavailable => METAL_UNAVAILABLE,
            Self::UnsupportedJob(reason) | Self::Kernel(reason) => reason,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalDctToWaveletStageAccelerator {
    mode: MetalDispatchMode,
    min_auto_samples: usize,
    dwt97_attempts: usize,
    dwt97_dispatches: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalDispatchMode {
    Explicit,
    Auto,
}
```

Implement `DctToWaveletStageAccelerator` so `dct_grid_to_dwt97` returns
`Err(METAL_UNAVAILABLE)` in explicit non-macOS mode and `Ok(None)` in auto mode.
Leave `dct_grid_to_dwt53` inherited from the trait default.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97
cargo clippy -p signinum-transcode-metal --all-targets -- -D warnings
```

Expected: fallback tests pass and clippy is clean.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/signinum-transcode-metal
git commit -m "feat: scaffold Metal DCT transcode accelerator"
```

## Task 2: 9/7 Projection Weights

**Files:**
- Create: `crates/signinum-transcode-metal/src/weights.rs`
- Modify: `crates/signinum-transcode-metal/src/lib.rs`
- Modify: `crates/signinum-transcode-metal/tests/dct97.rs`

- [ ] **Step 1: Write failing weight tests**

Add tests requiring `Dwt97WeightRows::for_len(8)` to expose low/high row
counts and to produce deterministic non-empty weights for 8, 13, and 16 sample
lengths.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97 weight_rows
```

Expected: missing `weights` module or `Dwt97WeightRows`.

- [ ] **Step 3: Implement weight construction**

Implement public crate-internal rows:

```rust
pub(crate) struct Dwt97WeightRows {
    pub(crate) low: Vec<Vec<f32>>,
    pub(crate) high: Vec<Vec<f32>>,
}
```

Use the same 9/7 lifting constants and boundary handling as
`signinum-transcode/src/dct97_2d.rs`, but output `f32` rows for Metal upload.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97
cargo clippy -p signinum-transcode-metal --all-targets -- -D warnings
```

Expected: tests pass and clippy is clean.

- [ ] **Step 5: Commit**

```bash
git add crates/signinum-transcode-metal
git commit -m "feat: add 9/7 projection weight builder"
```

## Task 3: Metal 9/7 Kernel

**Files:**
- Create: `crates/signinum-transcode-metal/src/dct97.metal`
- Create: `crates/signinum-transcode-metal/src/metal.rs`
- Modify: `crates/signinum-transcode-metal/src/lib.rs`
- Modify: `crates/signinum-transcode-metal/tests/dct97.rs`

- [ ] **Step 1: Write failing macOS coefficient test**

Add a macOS-only test that compares explicit Metal output against scalar
`dct8x8_blocks_to_dwt97_float_linear_with_scratch` for 8x8, 13x11, and 16x16.
Use tolerance `2.0e-3` for each coefficient.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97 metal_dct97_matches_scalar_for_structured_grids
```

Expected on macOS: explicit accelerator returns unavailable or fallback because
no Metal kernel exists. Expected off macOS: test is cfg-gated and skipped.

- [ ] **Step 3: Implement host runtime and kernel**

Implement `metal.rs` with:

- runtime initialization from `metal::Device::system_default`
- `new_library_with_source(include_str!("dct97.metal"), ...)`
- four output buffers for LL, HL, LH, HH
- one kernel dispatch per band
- download into `Dwt97TwoDimensional<f64>`

Implement `dct97.metal` with one kernel:

```metal
kernel void transcode_dct97_project_band(...)
```

One thread computes one coefficient by iterating contributing x/y sample weights
and the 8x8 IDCT basis.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p signinum-transcode-metal --test dct97
cargo clippy -p signinum-transcode-metal --all-targets -- -D warnings
```

Expected on macOS: scalar-vs-Metal tests pass. Expected off macOS: fallback
tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/signinum-transcode-metal
git commit -m "feat: add Metal 9/7 DCT projection kernel"
```

## Task 4: JPEG 4:2:0 Integration

**Files:**
- Create: `crates/signinum-transcode-metal/tests/jpeg_to_htj2k.rs`

- [ ] **Step 1: Write integration test**

Add a test that uses `JpegToHtj2kOptions::lossy_97()`, a
`MetalDctToWaveletStageAccelerator::for_auto_with_min_samples(1)`, and
`JpegToHtj2kTranscoder::transcode_with_accelerator` on
`baseline_420_16x16.jpg`. Assert native decode succeeds and SIZ sampling is
`[(1,1), (2,2), (2,2)]`.

- [ ] **Step 2: Run test to verify failure if integration is missing**

Run:

```bash
cargo test -p signinum-transcode-metal --test jpeg_to_htj2k
```

Expected: fail until constructor/routing details are implemented, or pass on
non-macOS fallback after explicit fallback assertions are added.

- [ ] **Step 3: Implement missing routing helpers**

Add `for_auto_with_min_samples(min_auto_samples: usize)` and dispatch counters.
Ensure macOS uses Metal when sample count meets the threshold, and non-macOS
falls back cleanly.

- [ ] **Step 4: Verify**

Run:

```bash
cargo test -p signinum-transcode-metal --test jpeg_to_htj2k
cargo test -p signinum-transcode --test jpeg_to_htj2k
```

Expected: Metal crate integration and CPU transcode tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/signinum-transcode-metal
git commit -m "feat: wire Metal 9/7 transcode integration"
```

## Task 5: Benchmarks And Documentation

**Files:**
- Create: `crates/signinum-transcode-metal/benches/dct97_metal.rs`
- Create: `crates/signinum-transcode-metal/tests/bench_harness.rs`
- Modify: `docs/dct-to-htj2k-notes.md`

- [ ] **Step 1: Write benchmark harness test**

Add stable-name checks for:

- `dct97_metal_projection`
- `scalar_direct_13x11`
- `metal_direct_13x11`
- `jpeg_to_htj2k_420_lossy97`
- `jpeg_to_htj2k_420_lossy97_metal_auto`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p signinum-transcode-metal --test bench_harness
```

Expected: missing benchmark file or group names.

- [ ] **Step 3: Add Criterion benchmarks**

Benchmark scalar direct 9/7 against Metal direct 9/7 for 13x11 and benchmark
JPEG 4:2:0 lossy 9/7 scalar vs Metal auto. Gate Metal benchmark execution with
runtime availability checks.

- [ ] **Step 4: Run verification and benchmark**

Run:

```bash
cargo test -p signinum-transcode-metal
cargo clippy -p signinum-transcode-metal --all-targets -- -D warnings
cargo bench --profile release-bench -p signinum-transcode-metal --bench dct97_metal
```

Expected: tests and clippy pass; benchmark reports scalar and Metal groups when
Metal is available, and skips Metal groups cleanly otherwise.

- [ ] **Step 5: Document benchmark evidence and commit**

Update `docs/dct-to-htj2k-notes.md` with the benchmark command and measured
numbers from this branch. Then commit:

```bash
git add crates/signinum-transcode-metal docs/dct-to-htj2k-notes.md
git commit -m "test: benchmark Metal 9/7 transcode projection"
```

## Final Verification

Run:

```bash
cargo test -p signinum-transcode-metal
cargo clippy -p signinum-transcode-metal --all-targets -- -D warnings
cargo test -p signinum-transcode
cargo clippy -p signinum-transcode --all-targets -- -D warnings
cargo test -p signinum-core --test repo_integrity architecture_dependency_graph_matches_cargo_metadata
cargo bench --profile release-bench -p signinum-transcode-metal --bench dct97_metal
```

Do not claim a Metal performance win unless the benchmark output shows one.
