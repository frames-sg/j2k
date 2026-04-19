# J2K-M0 Skeleton + Inspect Implementation Plan

Goal: add `slidecodec-j2k` with JP2/raw-codestream inspect support, parser
 tests, a parse-fuzz scaffold, and CLI inspect dispatch.

## Task 1: Create the crate skeleton

Files:

- `crates/slidecodec-j2k/Cargo.toml`
- `crates/slidecodec-j2k/src/lib.rs`
- `crates/slidecodec-j2k/src/error.rs`
- `crates/slidecodec-j2k/src/view.rs`
- `crates/slidecodec-j2k/src/parse/mod.rs`
- `crates/slidecodec-j2k/src/parse/boxes.rs`
- `crates/slidecodec-j2k/src/parse/codestream.rs`
- root `Cargo.toml`

Steps:

- add `slidecodec-j2k` to the workspace
- depend on `slidecodec-core` and `thiserror`
- expose `J2kView`, `J2kDecoder`, and `J2kError`

## Task 2: Implement inspect parsing

Files:

- `crates/slidecodec-j2k/src/error.rs`
- `crates/slidecodec-j2k/src/view.rs`
- `crates/slidecodec-j2k/src/parse/boxes.rs`
- `crates/slidecodec-j2k/src/parse/codestream.rs`

Steps:

- parse raw codestream `SOC` + main-header markers through `SOT`/`EOC`
- parse JP2 top-level boxes and required `jp2h` / `jp2c` structure
- map parsed metadata into `slidecodec_core::Info`
- implement `J2kDecoder::inspect`, `J2kView::parse`, `J2kDecoder::from_view`

## Task 3: Add tests, fuzz scaffold, and CLI dispatch

Files:

- `crates/slidecodec-j2k/tests/inspect.rs`
- `crates/slidecodec-j2k/tests/proptest_inspect.rs`
- `crates/slidecodec-j2k/fuzz/Cargo.toml`
- `crates/slidecodec-j2k/fuzz/fuzz_targets/parse_fuzz.rs`
- `crates/slidecodec-cli/src/main.rs`

Steps:

- add inline synthetic JP2 / raw codestream test coverage
- add proptest robustness coverage
- add parse-fuzz target scaffold
- dispatch CLI inspect by magic bytes to JPEG or J2K, with J2K-specific output formatting

## Verification

Run:

- `cargo fmt --all --check`
- `cargo test -p slidecodec-j2k`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo bench -p slidecodec-jpeg --bench compare --no-run`
- `cargo deny check`
