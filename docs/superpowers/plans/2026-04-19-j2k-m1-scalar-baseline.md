# J2K-M1 Scalar Baseline Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a working full-frame JPEG 2000 decode surface to `slidecodec-j2k` using the committed inspect parser plus the in-tree scalar decoder path, with `ImageDecode<'a>` support for the M1 pixel formats.

**Architecture:** keep `slidecodec-j2k`'s lightweight inspect parser and borrowed `J2kView<'a>` / `J2kDecoder<'a>` shape, but implement Part 1 decode in-tree. `slidecodec-j2k` owns the public API, buffer validation, error mapping, and output-format adaptation.

**Tech Stack:** Rust, `slidecodec-core`, `thiserror`

---

### Task 1: Harden the inspect parser before building decode on top of it

**Files:**
- Modify: `crates/slidecodec-j2k/src/parse/codestream.rs`
- Modify: `crates/slidecodec-j2k/src/parse/boxes.rs`
- Test: `crates/slidecodec-j2k/tests/inspect.rs`

- [ ] **Step 1: Add failing parser regression tests**

Write tests for:
- raw codestream missing `COD`
- raw codestream ending at EOF after main header
- JP2 with `jp2c` before `jp2h`

- [ ] **Step 2: Run the new parser tests to verify they fail**

Run: `cargo test -p slidecodec-j2k --test inspect`

- [ ] **Step 3: Tighten codestream and JP2 validation**

Change the parser so:
- `COD` is required for raw inspect
- EOF is not accepted as a complete header terminator
- required JP2 boxes must appear in sane order

- [ ] **Step 4: Re-run the parser tests**

Run: `cargo test -p slidecodec-j2k --test inspect`

- [ ] **Step 5: Commit**

Commit message: `fix: harden j2k inspect validation`

### Task 2: Add the public J2K decode surface

**Files:**
- Modify: `crates/slidecodec-j2k/Cargo.toml`
- Modify: `crates/slidecodec-j2k/src/lib.rs`
- Modify: `crates/slidecodec-j2k/src/error.rs`
- Modify: `crates/slidecodec-j2k/src/view.rs`
- Create: `crates/slidecodec-j2k/src/decode.rs`

- [ ] **Step 1: Add failing API tests for decode construction and output validation**

Add tests covering:
- `J2kDecoder::new(...).decode_into(..., PixelFormat::Rgb8)`
- `J2kDecoder::new(...).decode_into(..., PixelFormat::Gray8)`
- `J2kDecoder::new(...).decode_into(..., PixelFormat::Gray16)` from 8-bit input
- unsupported `Rgba16`
- small buffer / bad stride
- `decode_region_into` -> `NotImplemented`
- `decode_scaled_into` -> `NotImplemented`

- [ ] **Step 2: Run those tests to verify they fail**

Run: `cargo test -p slidecodec-j2k --test decode`

- [ ] **Step 3: Add the scalar decode module**

Create `src/decode.rs` with:
- scalar decode entry points
- pixel-format validation helpers
- output mapping helpers for 8-bit and native-depth paths
- buffer validation that maps to `slidecodec_core::BufferError`

- [ ] **Step 4: Extend `J2kError` for buffer/decode failures**

Add:
- `Buffer(BufferError)`
- `Unsupported(Unsupported)`

and update `CodecError` classification.

- [ ] **Step 5: Wire `J2kDecoder` to `ImageCodec` + `ImageDecode<'a>`**

Implement:
- `ImageCodec for J2kDecoder<'a>`
- `ImageDecode<'a> for J2kDecoder<'a>`

using the existing borrowed `J2kView<'a>` / `J2kDecoder<'a>` shape.

- [ ] **Step 6: Re-run the decode API tests**

Run: `cargo test -p slidecodec-j2k --test decode`

- [ ] **Step 7: Commit**

Commit message: `feat: add j2k scalar baseline decode api`

### Task 3: Cover 8-bit and 16-bit decode behavior with synthetic fixtures

**Files:**
- Create: `crates/slidecodec-j2k/tests/decode.rs`

- [ ] **Step 1: Generate synthetic J2K/JP2 test inputs inside tests**

Use committed fixtures or small inline-generated codestreams for:
- 8-bit RGB irreversible sample
- 8-bit grayscale irreversible sample
- 16-bit grayscale reversible sample
- 16-bit RGB reversible sample

- [ ] **Step 2: Add behavior-focused decode assertions**

Verify:
- `Rgb8`
- `Rgba8`
- `Gray8`
- `Rgb16`
- `Gray16`
- 8-bit input widened to 16-bit exact little-endian output

and check decoded pixel bytes exactly for small synthetic patterns.

- [ ] **Step 3: Run the J2K tests**

Run: `cargo test -p slidecodec-j2k`

- [ ] **Step 4: Commit**

Commit message: `test: cover j2k scalar decode outputs`

### Task 4: Add decode fuzzing and workspace verification

**Files:**
- Modify: `crates/slidecodec-j2k/fuzz/Cargo.toml`
- Create: `crates/slidecodec-j2k/fuzz/fuzz_targets/decode_fuzz.rs`

- [ ] **Step 1: Add the decode fuzz target**

The fuzz target should:
- call `J2kDecoder::new`
- if that succeeds, allocate a bounded valid output buffer for one supported
  pixel format and call `decode_into`

- [ ] **Step 2: Verify fuzz crate compiles**

Run: `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`

- [ ] **Step 3: Run full milestone verification**

Run:
- `cargo fmt --all --check`
- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo check --manifest-path crates/slidecodec-j2k/fuzz/Cargo.toml`
- `cargo deny check`

- [ ] **Step 4: Commit**

Commit message: `test: add j2k decode fuzz scaffold`
