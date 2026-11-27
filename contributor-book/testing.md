# Testing

`cargo xtask test` runs the workspace tests the way CI does (see
[Getting started](getting-started.md#everyday-commands)). While you work,
`cargo test -p <crate>` is usually enough.

A few rules for tests:

- Test behavior callers can see: decoded samples, errors, output bytes. A bug
  fix comes with a test that fails without it.
- Don't delete a passing regression test as cleanup.
- Unsupported input and backend failures return errors. Don't add a silent
  fallback to make a test pass.

## Where tests live

- **Unit tests** sit next to the code in `#[cfg(test)]` modules, often in a
  sibling `tests.rs` or `tests/` directory under `src/`.
- **Integration tests** are in each crate's `tests/` directory.
- **Shared fixtures and builders** are in `j2k-test-support`
  (`crates/j2k-test-support/fixtures/` plus generated JPEG and JPEG 2000
  builders in `src/`). Transcode oracles are in `j2k-transcode-test-support`.
- **Crate-specific fixtures** are in the crate's `fixtures/` directory, for
  example `crates/j2k-native/fixtures/`.
- **libjpeg-turbo reference outputs** are in `corpus/conformance/`, listed with
  their hashes in `manifest.json`. `corpus/conformance/generate.sh` and the
  `*_matrix.py` scripts regenerate them; CI only checks the hashes.
- **Allocation counts** are tested through `j2k-alloc-probe`, which
  `cargo xtask test` runs on its own.

Keep committed fixtures small. Anything from a third party needs its source
and license in a README next to it and an entry in [NOTICES.md](../NOTICES.md).
Repo-lint checks this for the OpenHTJ2K fixtures.

Larger inputs stay outside the repository and are passed in through
environment variables such as `J2K_WSI_ROOT`, `J2K_TRANSCODE_WSI_ROOT`, and
`J2K_ISO_CONFORMANCE_DIR`. Those tests skip when the variable is unset, unless
the matching `J2K_REQUIRE_*` variable is set. `scripts/parity-corpus-fetch.sh`
downloads a hash-checked parity corpus from a JSON manifest you supply. Never
commit protected health information or restricted datasets.

## GPU tests

GPU tests skip by default, so `cargo xtask test` passes on any machine. Each
one checks a gate first with the helpers in `j2k_test_support::gpu_gate`: when
its `J2K_REQUIRE_*` variable is unset it prints a skip marker and returns, and
when the variable is set a missing device fails the test.

| Variable | Gates |
| --- | --- |
| `J2K_REQUIRE_METAL_RUNTIME` | Metal runtime tests |
| `J2K_REQUIRE_CUDA_RUNTIME` | CUDA runtime tests |
| `J2K_REQUIRE_CUDA_OXIDE_BUILD` | Building every CUDA kernel; CUDA tests fail instead of skip when a kernel is missing |
| `J2K_REQUIRE_METAL`, `J2K_REQUIRE_CUDA` | Python GPU tests |

On hardware, run the same validation the GPU runners do:

```bash
cargo xtask release-metal --mode quick   # macOS, Apple Silicon
cargo xtask release-cuda --mode quick    # Linux x86-64, NVIDIA
```

They set the gates, fail on any skip marker, and check that the required named
tests actually ran. `--mode full` is the release-grade run. When a pull request
needs them is in [CI and GPU validation](ci.md).

## Framework adapter tests

`j2k-ml` and `j2k-mpsgraph` have portable tests that run anywhere and GPU
tests behind the usual gates:

```bash
cargo test -p j2k-ml --features cpu
cargo test -p j2k-mpsgraph
```

Their hardware tests run as part of `cargo xtask release-cuda` and
`cargo xtask release-metal`. The full Metal run also runs the ignored
`one_thousand_direct_submissions_reuse_one_session` test in
`crates/j2k-mpsgraph/tests/metal.rs`, which pushes 1,000 submissions through
one MPSGraph session.

On Linux AArch64 the `j2k-ml` tests and benches use Burn's NdArray backend
instead of Flex, because Flex pulls in `gemm-f16`, whose debug AArch64
assembly needs FP16 target features some runners lack
([gemm#31](https://github.com/sarah-quinones/gemm/issues/31)). Benchmark
results from different Burn backends aren't comparable.

## Python tests

The Python tests in `crates/j2k-python/tests/` run against an installed wheel
and need NumPy and PyTorch. Build and install the wheel as in
[Getting started](getting-started.md#python-bindings), then:

```bash
cargo test -p j2k --test wavelets
python -m unittest discover -s crates/j2k-python/tests -v
```

The first command covers the Rust side of the wavelet band API that the Python
`decode_wavelets` exposes. The CUDA and MPS tests skip when the wheel lacks
that backend or there is no device. Set `J2K_REQUIRE_CUDA=1` or
`J2K_REQUIRE_METAL=1` to make that a failure. The PyO3 sources only run inside
Python, so these tests, not the Rust coverage gate, cover them.

## Comparator parity

Parity tests compare J2K's output with OpenJPEG, Grok, and libjpeg-turbo.
Install the tools from
[Getting started](getting-started.md#comparator-libraries), then:

```bash
J2K_REQUIRE_OPENJPEG=1 J2K_REQUIRE_GROK=1 J2K_REQUIRE_LIBJPEG_TURBO=1 \
  cargo xtask j2k-bench-signoff
```

It fails unless each comparator runs at least its expected number of parity
tests, so a missing tool can't pass by skipping. Full validation runs it on
Ubuntu.

## Fuzzing

Fuzz targets are in separate `fuzz/` workspaces under `j2k`, `j2k-native`,
`j2k-jpeg`, `j2k-jpegls`, `j2k-tilecodec`, `j2k-transcode`, and `j2k-t803`. They need nightly
and [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

```bash
cargo xtask fuzz-build                    # check every fuzz workspace builds
cargo xtask fuzz-run                      # run each target in FUZZ_TARGETS
cd crates/j2k-native && cargo +nightly fuzz run native_decode_fuzz
```

`fuzz-run` runs every target listed in `FUZZ_TARGETS` in
`xtask/src/quality_commands.rs` for `J2K_FUZZ_RUNS` iterations (1000 by
default). CI runs 512 iterations or 60 seconds per target on every full
validation, and 20,000 iterations or 15 minutes in the weekly scheduled run.

Seed corpora are in `crates/j2k/fuzz/corpus/` and
`crates/j2k-native/fuzz/corpus/`. The `j2k-native` seeds are explained in
[its fuzz README](../crates/j2k-native/fuzz/README.md). Add a target to
`FUZZ_TARGETS` so CI runs it.

## T.803 conformance

`j2k-t803` runs the ISO/IEC 15444-4 / ITU-T T.803 conformance tests. The
results we publish, and what they cover, are in
[Conformance](../book/conformance.md).

The official corpus is copyrighted. `fetch` downloads it only from the URL and
archive hash pinned in `corpus/j2k-conformance/t803-v3.toml`, and keeps it
under `target/t803/`. Never commit or upload it; only the JSON and Markdown
reports are kept.

```bash
cargo xtask t803 fetch
cargo xtask t803 run --iut cpu --suite all --development
cargo xtask t803 run --iut metal --suite all --development
cargo xtask t803 run --iut cuda --suite all --development
```

`--suite` takes `part1`, `part15`, or `all`, and `--out-dir` sets where the
reports go. `--development` allows a dirty tree; the release verifier rejects
its reports, so release runs on a clean commit leave it off. The CPU encoder
matrix includes an HT+RGN case that needs the pinned OpenHTJ2K decoder, so run
`scripts/prepare-openhtj2k-reference.sh` first and export what it prints.

`fetch` rejects unapproved redirects, changed archive or file hashes, unsafe
archive entries, duplicate paths, unexpected case names, and oversized input.

Each backend is verified separately:

```bash
cargo xtask t803 verify --scope cpu --candidate-sha "$RC_SHA" \
  --report cpu-linux.json --report cpu-macos.json --report cpu-windows.json
cargo xtask t803 verify --scope metal --candidate-sha "$RC_SHA" --report metal.json
```

`--scope all` checks all five reports together. Verification requires every
selected decoder case with no skips, every report passing, matching source and
corpus hashes, and the right implementation, platform, and backend in each
report. A report that labels a CPU-assisted case as GPU-only is rejected.

The encoder matrix and the CPU, CUDA, and Metal implementation statements are
in `corpus/j2k-conformance/` (see its README). Encoder results are informative
under T.803, not part of decoder conformance.

T.803 doesn't test coding styles set in tile-part headers: none of its
codestreams has a tile-part COD or COC.
`crates/j2k-native/tests/tile_header_markers.rs` covers tile-part COD, COC,
QCD, and QCC overrides and their precedence. T.803 also doesn't test
robustness, security, or speed; fuzzing and benchmarks do.

## Coverage

```bash
cargo xtask coverage host --base origin/main
```

`coverage` measures the Rust lines your branch changed, using
[cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov) (CI pins the
version in `scripts/ensure-cargo-llvm-cov.sh`). The lane is `host`, `metal`,
or `cuda`. Without `--base`, the base is `J2K_COVERAGE_BASE`, then
`origin/$GITHUB_BASE_REF`, then `HEAD^`. It refuses to run with untracked Rust
files.

- The host lane requires 80% of changed production lines covered, and 80% of
  changed release-critical lines.
- The Metal and CUDA lanes require 80% for routing, validation, allocation,
  ownership, public API, parsing, security, and error-handling code. Kernel
  correctness is checked by comparing GPU output with the CPU on real
  hardware, not by line coverage.
- `#[cfg(test)]` code and test, example, bench, and fuzz targets are reported
  separately, not as uncovered production code.
- Only code that can't be instrumented on the host is excluded: CUDA device
  code, generated CUDA-Oxide host scaffolds, the shared SIMT prelude, CUDA and
  NVTX FFI declarations, and the generated codec-math DWT fragment. The PyO3
  sources in `crates/j2k-python/src/` are excluded too (see
  [Python tests](#python-tests)).

Each lane builds into a fresh target directory and uses only that build's
build-script output to decide which custom `cfg`s are active, so an earlier
run can't leak in. A package with a build script that produces no output, or
conflicting output, fails the lane. If a custom `cfg`'s value is unknown, both
branches count as changed lines.

## Miri and no_std

```bash
cargo xtask miri     # nightly with the miri component
cargo xtask no-std
```

`miri` runs the `j2k-core` and `j2k-tilecodec` tests and the `inspect::` tests
of `j2k-native` without default features. `no-std` checks the `no_std` crates
on bare-metal targets and the single-threaded wasm32 build of `j2k-native`.

## Format support matrix

[Format support](../book/format-support.md) is checked against the code:

```bash
cargo xtask public-support
cargo xtask public-support --final
```

It checks that the page has every required row and comparator, that it names
every row of `corpus/j2k-conformance/support-inventory.tsv`, and that every
test backing the matrix (listed in `xtask/src/public_support.rs`) still
exists. It doesn't run those tests. `--final` also requires the core rows to
be marked done; full validation runs it. Update the page and the inventory in
the same pull request as a codec support change.

Published benchmark numbers are checked separately, by `adoption-report` (see
[Benchmarks](benchmarks.md#the-adoption-bundle)).

## Repository lints

```bash
cargo xtask repo-lint
```

This runs the policy tests in `xtask/tests/repo_lint_support/`. They check,
among other things, that:

- the [crate dependency graph](architecture.md#crate-dependency-graph) matches
  `cargo metadata` and follows the dependency rules;
- workflows keep their required structure and pinned actions;
- each codec constant has one owner, and Metal gets it through
  `cargo xtask codec-math-codegen`;
- committed conformance fixtures are listed with hashes, and third-party
  fixtures carry their license notices;
- no text file embeds a local home directory, and every shell script a file
  mentions exists;
- `SECURITY.md` lists the latest release tag.

Run one policy with
`cargo test -p xtask --test repo_lint <name>`, for example
`cargo test -p xtask --test repo_lint architecture`.
