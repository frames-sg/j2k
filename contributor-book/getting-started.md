# Getting started

## Toolchain

`rust-toolchain.toml` pins the Rust toolchain, so `rustup` installs and uses
the right one when you run `cargo` in the repository. The minimum supported
Rust version is `rust-version` in the workspace `Cargo.toml`, and CI checks
the whole workspace with it. `j2k-types` and `j2k-codec-math` are also
compiled into the CUDA kernels by an older compiler, so CI checks those two
with that compiler as well (the `device-crates-msrv` job in
`.github/workflows/ci.yml`).

A few tasks need a nightly toolchain: fuzzing, Miri, and the public API
snapshots. Each chapter says which.

## Building

```bash
cargo build
cargo test -p j2k
```

The CPU crates build on Linux, macOS, and Windows with no system
dependencies.

- **Metal** crates build only on macOS. Tests that need a GPU skip unless you
  set `J2K_REQUIRE_METAL_RUNTIME=1` on an Apple Silicon Mac.
- **CUDA** crates build everywhere, but the kernels are only compiled on Linux
  hosts that have CUDA-Oxide's `cargo oxide` subcommand and libclang (for
  bindgen). Each kernel project pins its own toolchain in a
  `rust-toolchain.toml` next to its source. Without these, the build prints a
  warning and embeds placeholder PTX; set `J2K_REQUIRE_CUDA_OXIDE_BUILD=1` to
  make that an error. Runtime tests skip unless `J2K_REQUIRE_CUDA_RUNTIME=1`.

The `J2K_REQUIRE_*` variables are listed in
[Environment variables](environment-variables.md#test-and-ci-gates).

## Everyday commands

`cargo xtask` is a Cargo alias (in `.cargo/config.toml`) for the `xtask`
crate, which runs the repository's checks. `cargo xtask help` lists every
task. The ones you'll use most:

| Command | What it does |
| --- | --- |
| `cargo xtask fmt` | `cargo fmt --all -- --check`. Run `cargo fmt --all` to fix. |
| `cargo xtask clippy` | Clippy on all targets with all features and warnings denied, plus the repo-lint test target. |
| `cargo xtask clippy-strict` | Pedantic and nursery Clippy on the packages that pass it. |
| `cargo xtask test` | Workspace tests with all features, the JPEG `scalar-only` build, doctests, examples, and the allocation probe. On macOS, `j2k-metal` runs single-threaded. |
| `cargo xtask doc` | Rustdoc with warnings denied, and `missing_docs` denied on published libraries. |
| `cargo xtask ci` | `fmt`, `codec-math-codegen`, `clippy`, `panic-surface`, `test`, `doc`, and `unsafe-audit`. |

Run `cargo xtask ci` before you open a pull request that touches Rust. For a
quick loop, `cargo test -p <crate>` is fine.

`cargo xtask test` writes every skipped GPU test to a log and prints how many
were skipped, so a quiet pass on a machine without a GPU is expected.

Other tasks you may run into:

- `cargo xtask codec-math-codegen` checks that the generated Rust and Metal
  fragments match `j2k-codec-math`. Add `--write` to regenerate them.
- `cargo xtask panic-surface` keeps `unwrap`, `expect`, and panic macros in
  library code from growing.
- `cargo xtask unsafe-audit` checks the JPEG SIMD unsafe boundary (see
  [Architecture](architecture.md#cpu-jpeg-simd)).
- `cargo xtask metal-compile` (macOS) compiles every Metal target and runs the
  tests that need no GPU.

Testing, coverage, fuzzing, and the repository lints are in
[Testing](testing.md).

## Comparator libraries

Parity tests and benchmarks compare J2K with other codecs. They skip a
comparator that isn't installed unless you set its `J2K_REQUIRE_*` variable.

- **OpenJPEG, Grok, libjpeg-turbo.** On Ubuntu, CI installs:

  ```bash
  sudo apt-get install grokj2k-tools libgrokj2k1-dev libopenjp2-tools \
    libturbojpeg0-dev pkg-config
  ```

  On macOS the parity tests look for the tools in `/opt/homebrew/bin`. Point
  `J2K_OPENJPEG_BIN`, `J2K_GROK_BIN`, and the related variables elsewhere if
  needed. libjpeg-turbo comparisons need the `bench-libjpeg-turbo` feature of
  `j2k-jpeg` and `libturbojpeg` visible to `pkg-config`.
- **OpenJPH and OpenHTJ2K.** `scripts/prepare-openjph-reference.sh` and
  `scripts/prepare-openhtj2k-reference.sh` build pinned versions under
  `target/` and print the variables to export.
- **Kakadu** is optional and used only as a command-line comparison.

The in-process OpenJPEG comparator in `j2k-compare` builds `openjpeg-sys`'s
vendored OpenJPEG and needs no install. [Benchmarks](benchmarks.md) covers how
to run the comparisons.

## Python bindings

`crates/j2k-python` builds the `j2k` Python wheel with
[maturin](https://www.maturin.rs/). You need CPython 3.11 or later. From the
repository root:

```bash
uvx maturin build --manifest-path crates/j2k-python/Cargo.toml --release --out target/wheels
python -m pip install --no-deps --no-index --find-links target/wheels j2k
python -m pip install 'numpy>=2' 'torch>=2.12'
```

Add `--features metal` (Apple Silicon) or `--features cuda` (Linux with the
CUDA requirements above) to the maturin command for GPU output. Running the
Python tests is covered in [Testing](testing.md#python-tests).

## C and C++ package

`crates/j2k-c` is the C ABI and C++ wrapper. You need Python 3, CMake 3.20 or
later, and pkg-config. `scripts/package.py` builds a relocatable package for
the host; it refuses to overwrite an existing output directory.

```bash
cargo test -p j2k-c --locked
python3 crates/j2k-c/scripts/package.py --output dist/j2k-c
cargo run -p j2k-c --locked --example make_fixture -- target/fixture.j2c
cmake -S crates/j2k-c/native/tests -B target/native-tests \
  -DCMAKE_PREFIX_PATH="$PWD/dist/j2k-c" -DJ2K_FIXTURE="$PWD/target/fixture.j2c"
cmake --build target/native-tests
ctest --test-dir target/native-tests --output-on-failure
```

Keep `include/j2k.h`, the Rust ABI definitions in `src/abi.rs`, the examples,
and `crates/j2k-c/docs/` in sync. An incompatible layout or ownership change
needs a new `J2K_ABI_VERSION`.

## Repository layout

```text
book/                user guide (mdBook)
contributor-book/    this book
corpus/              small committed fixtures; metadata for external corpora
crates/              workspace crates; see Architecture
docs/                the GitHub Pages website (HTML only)
scripts/             CI planner, release scripts, comparator build scripts
xtask/               cargo xtask
  api/               public API snapshots
  release-evidence/  per-release API reviews and CUDA waivers
  routing-promotion-evidence.json
                     accepted GPU routing measurements
release-crates.json  published crates in publish order
CHANGELOG.md         unreleased changes
```

Each crate keeps its integration tests in `tests/` and Criterion benches in
`benches/`. Crates with fuzz targets keep them in a separate `fuzz/`
workspace. [Testing](testing.md#where-tests-live) says where fixtures go.
