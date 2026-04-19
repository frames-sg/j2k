# M0 Core Extraction + JPEG Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `slidecodec-core`, move the shared decode contracts into it, and refactor `slidecodec-jpeg` onto those contracts without regressing the verified JPEG behavior or WSI-oriented performance surface.

**Architecture:** Add a new `slidecodec-core` crate that owns shared value types, traits, and error helpers. Then adapt `slidecodec-jpeg` to consume those shared types while preserving its borrowed decoder model (`JpegView<'a>`, `Decoder<'a>`) and existing context/scratch/tile-batch structure. Keep the refactor incremental: add compat shims first, switch call sites/tests/benches next, then remove JPEG-local duplicates once the new core API is fully wired.

**Tech Stack:** Rust workspace crates, `thiserror`, existing `criterion` benches, existing `proptest`/integration tests, existing JPEG NEON/AVX2 paths.

---

### Task 1: Create `slidecodec-core`

**Files:**
- Create: `crates/slidecodec-core/Cargo.toml`
- Create: `crates/slidecodec-core/src/lib.rs`
- Create: `crates/slidecodec-core/src/sample.rs`
- Create: `crates/slidecodec-core/src/pixel.rs`
- Create: `crates/slidecodec-core/src/scale.rs`
- Create: `crates/slidecodec-core/src/types.rs`
- Create: `crates/slidecodec-core/src/row_sink.rs`
- Create: `crates/slidecodec-core/src/scratch.rs`
- Create: `crates/slidecodec-core/src/context.rs`
- Create: `crates/slidecodec-core/src/error.rs`
- Create: `crates/slidecodec-core/src/backend.rs`
- Create: `crates/slidecodec-core/src/traits.rs`
- Modify: `Cargo.toml`
- Test: `cargo check -p slidecodec-core`

- [ ] **Step 1: Add the new workspace member**

```toml
[workspace]
members = [
    "crates/slidecodec-core",
    "crates/slidecodec-jpeg",
    "crates/slidecodec-cli",
]
```

- [ ] **Step 2: Add the new crate manifest**

```toml
[package]
name = "slidecodec-core"
description = "Shared decode contracts and types for slidecodec"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
homepage.workspace = true
keywords.workspace = true
categories.workspace = true

[lib]
name = "slidecodec_core"
path = "src/lib.rs"

[dependencies]
thiserror = { workspace = true }

[lints.rust]
unsafe_code = "forbid"
unreachable_pub = "warn"
```

- [ ] **Step 3: Add the core module skeleton**

```rust
#![no_std]
#![warn(unreachable_pub)]

extern crate alloc;

pub mod backend;
pub mod context;
pub mod error;
pub mod pixel;
pub mod row_sink;
pub mod sample;
pub mod scale;
pub mod scratch;
pub mod traits;
pub mod types;

pub use backend::CpuFeatures;
pub use context::{CacheStats, CodecContext, DecoderContext};
pub use error::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
pub use pixel::{PixelFormat, PixelLayout};
pub use row_sink::RowSink;
pub use sample::{Sample, SampleType};
pub use scale::Downscale;
pub use scratch::ScratchPool;
pub use traits::{DecodeRowsError, ImageCodec, ImageDecode, ImageDecodeRows, TileBatchDecode, TileDecompress};
pub use types::{Colorspace, DecodeOutcome, Rect, TileLayout, WarningKind, Info};
```

- [ ] **Step 4: Add the shared value types and traits**

```rust
// sample.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleType { U8, U16 }

pub trait Sample: Copy + Default + Send + Sync + 'static {
    const TYPE: SampleType;
    const BITS: u8;
}

impl Sample for u8 { const TYPE: SampleType = SampleType::U8; const BITS: u8 = 8; }
impl Sample for u16 { const TYPE: SampleType = SampleType::U16; const BITS: u8 = 16; }
```

```rust
// pixel.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout { Rgb, Rgba, Gray }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelFormat { Rgb8, Rgba8, Gray8, Rgb16, Rgba16, Gray16 }
```

```rust
// scale.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Downscale { None, Half, Quarter, Eighth }
```

```rust
// row_sink.rs
pub trait RowSink<S: Sample> {
    type Error: core::error::Error + Send + Sync + 'static;
    fn write_row(&mut self, y: u32, row: &[S]) -> Result<(), Self::Error>;
}
```

```rust
// scratch.rs
pub trait ScratchPool: Send {
    fn bytes_allocated(&self) -> usize;
    fn reset(&mut self);
}
```

- [ ] **Step 5: Add the core decode traits**

```rust
pub trait ImageCodec {
    type Error: CodecError;
    type Warning: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    type Pool: ScratchPool;
}

pub trait ImageDecode<'a>: ImageCodec + Sized + 'a {
    type View: 'a;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error>;
    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error>;
    fn from_view(view: Self::View) -> Result<Self, Self::Error>;
}

pub trait TileBatchDecode: ImageCodec {
    type Context: CodecContext;
}
```

- [ ] **Step 6: Run the new crate check**

Run: `cargo check -p slidecodec-core`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/slidecodec-core
git commit -m "feat: add slidecodec-core crate skeleton"
```

### Task 2: Bridge `slidecodec-jpeg` to `slidecodec-core`

**Files:**
- Modify: `crates/slidecodec-jpeg/Cargo.toml`
- Modify: `crates/slidecodec-jpeg/src/lib.rs`
- Modify: `crates/slidecodec-jpeg/src/info.rs`
- Modify: `crates/slidecodec-jpeg/src/decoder.rs`
- Modify: `crates/slidecodec-jpeg/src/context.rs`
- Modify: `crates/slidecodec-jpeg/src/internal/scratch.rs`
- Modify: `crates/slidecodec-jpeg/src/backend/mod.rs`
- Test: `cargo check -p slidecodec-jpeg`

- [ ] **Step 1: Add the core dependency**

```toml
[dependencies]
slidecodec-core = { path = "../slidecodec-core" }
thiserror = { workspace = true }
memchr = { version = "2.7.6", default-features = false }
```

- [ ] **Step 2: Re-export the new core types from `slidecodec-jpeg`**

```rust
pub use slidecodec_core::{
    CacheStats, CodecContext, Colorspace as CoreColorspace, DecoderContext, DecodeOutcome,
    Downscale, ImageCodec, ImageDecode, ImageDecodeRows, PixelFormat, PixelLayout,
    Rect, RowSink, ScratchPool, Sample, SampleType, TileBatchDecode,
};
```

- [ ] **Step 3: Add JPEG compat shims before deleting old types**

```rust
pub type DownscaleFactor = slidecodec_core::Downscale;

pub trait RgbRowSink: RowSink<u8, Error = JpegError> {}

impl<T> RgbRowSink for T where T: RowSink<u8, Error = JpegError> {}
```

- [ ] **Step 4: Implement the core marker traits for JPEG**

```rust
pub struct JpegCodec;

impl ImageCodec for JpegCodec {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}

impl<'a> ImageCodec for Decoder<'a> {
    type Error = JpegError;
    type Warning = Warning;
    type Pool = ScratchPool;
}
```

- [ ] **Step 5: Implement borrowed decode traits without changing behavior**

```rust
impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> { Self::inspect(input) }
    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> { JpegView::parse(input) }
    fn from_view(view: Self::View) -> Result<Self, Self::Error> { Self::from_view(view) }
}
```

- [ ] **Step 6: Implement row/tile traits using the existing JPEG bodies**

```rust
impl<'a> ImageDecodeRows<'a, u8> for Decoder<'a> {
    fn decode_rows<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
    ) -> Result<DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>> {
        // adapt existing row-writer path
    }
}

impl TileBatchDecode for JpegCodec {
    type Context = DecoderContext;
}
```

- [ ] **Step 7: Run the JPEG compile check**

Run: `cargo check -p slidecodec-jpeg`  
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add crates/slidecodec-jpeg/Cargo.toml crates/slidecodec-jpeg/src
git commit -m "refactor: wire slidecodec-jpeg onto slidecodec-core"
```

### Task 3: Migrate call sites, tests, benches, and docs to `PixelFormat` + `Downscale`

**Files:**
- Modify: `crates/slidecodec-jpeg/tests/decode_into.rs`
- Modify: `crates/slidecodec-jpeg/tests/view_and_rows.rs`
- Modify: `crates/slidecodec-jpeg/tests/regressions.rs`
- Modify: `crates/slidecodec-jpeg/tests/batch.rs`
- Modify: `crates/slidecodec-jpeg/tests/external_wsi.rs`
- Modify: `crates/slidecodec-jpeg/benches/common/mod.rs`
- Modify: `crates/slidecodec-jpeg/benches/compare.rs`
- Modify: `crates/slidecodec-jpeg/benches/corpus_report.rs`
- Modify: `README.md`
- Modify: `crates/slidecodec-jpeg/README.md`
- Modify: `docs/bench.md`
- Test: `cargo test -p slidecodec-jpeg`

- [ ] **Step 1: Replace scaled `OutputFormat` call sites**

```rust
dec.decode_scaled_into(&mut out, stride, PixelFormat::Rgb8, Downscale::Quarter)?;
dec.decode_region_scaled_into(&mut out, stride, PixelFormat::Rgb8, roi, Downscale::Quarter)?;
```

- [ ] **Step 2: Replace `RgbRowSink` usage with `RowSink<u8>`**

```rust
impl RowSink<u8> for CollectRows {
    type Error = JpegError;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Self::Error> {
        self.rows.push((y, row.to_vec()));
        Ok(())
    }
}
```

- [ ] **Step 3: Keep top-level convenience APIs working**

```rust
pub fn decode_tile_into(...) -> Result<DecodeOutcome, JpegError> {
    JpegCodec::decode_tile(...)
}
```

- [ ] **Step 4: Update the benchmark helpers**

```rust
slidecodec_decode_scaled(&input.bytes, Downscale::Quarter);
slidecodec_decode_region_scaled(&input.bytes, 256, Downscale::Quarter);
```

- [ ] **Step 5: Run the JPEG test suite**

Run: `cargo test -p slidecodec-jpeg`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/slidecodec-jpeg/tests crates/slidecodec-jpeg/benches README.md crates/slidecodec-jpeg/README.md docs/bench.md
git commit -m "refactor: migrate jpeg call sites to core decode types"
```

### Task 4: Workspace verification and cleanup

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `.github/workflows/ci.yml` (only if needed for new workspace member)
- Test: full workspace checks

- [ ] **Step 1: Update the changelog for the core extraction**

```md
### Changed
- added `slidecodec-core` as the shared contract crate
- refactored `slidecodec-jpeg` onto borrowed core decode traits
- replaced JPEG-local scaled output variants with `PixelFormat` + `Downscale`
```

- [ ] **Step 2: Run the full workspace verification**

Run:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench -p slidecodec-jpeg --bench compare --no-run
```

Expected: PASS

- [ ] **Step 3: Run the external WSI regression if the local corpus is present**

Run:

```bash
SLIDECODEC_WSI_ROOT=/Users/user/Bench/SlideViewer/downloads cargo test -p slidecodec-jpeg --test external_wsi
```

Expected: PASS

- [ ] **Step 4: Commit the finished M0 refactor**

```bash
git add .
git commit -m "feat: extract slidecodec-core and complete M0 jpeg refactor"
```

---

**Self-review**

- Spec coverage: M0 core extraction, JPEG trait migration, PixelFormat/Downscale reshaping, row-sink migration, benches/tests/docs, and workspace verification are covered.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; every task has exact file paths and commands.
- Type consistency: the plan uses `PixelFormat`, `Downscale`, `RowSink`, `ImageDecode<'a>`, `ImageDecodeRows<'a, u8>`, and `TileBatchDecode` consistently.
