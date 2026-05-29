# Release Notes

## Current State

The repository is staged for the `signinum` facade release. The stable release
artifacts are `signinum`, `signinum-core`, `signinum-jpeg`, `signinum-j2k`,
`signinum-tilecodec`, and `signinum-cli`. `signinum-cuda-runtime`,
`signinum-profile`, and `signinum-j2k-native` are published support crates so
the public runtime crates that depend on them can be installed from crates.io.

Metal and CUDA adapter crates are published as pre-1.0 artifacts where their
APIs changed for the facade boundary.
Runtime backend selection defaults to `Auto`; supported compiled device paths
may run before CPU fallback.
CUDA explicit requests can produce CUDA device memory surfaces when built with
`cuda-runtime` on a host with a CUDA driver. `signinum-jpeg-cuda` can use
NVIDIA nvJPEG for full-frame RGB8 JPEG decode when `libnvjpeg` is installed;
unsupported JPEG shapes keep their documented JPEG fallback behavior. The J2K
CUDA adapter reserves explicit CUDA requests for strict CUDA-resident HTJ2K
codestream decode; CPU decode plus CUDA upload is available only through
explicit CPU-staged J2K APIs. NVIDIA performance claims require self-hosted GPU
benchmark evidence.

## Verification Gates

Hosted CI must pass before release staging:

1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo xtask test` on Linux x86_64, Linux aarch64, and Apple Silicon
   macOS runners
4. `cargo doc --workspace --all-features --no-deps` with rustdoc warnings
   denied
5. Benchmark compile checks for JPEG, JPEG Metal, J2K Metal, and tilecodec

For any release note that includes benchmark numbers or comparator language,
attach a benchmark publication report generated with:

```sh
cargo xtask bench-report --command "<exact benchmark command>" \
  --input-source "<manifest, generated input label, or external corpus hash>" \
  --out target/signinum-bench-report.md
```

The report must include input source, comparator versions, comparator paths,
thread settings, skipped rows, host/compiler metadata, and crate revision.

Runtime GPU validation is intentionally separate because hosted GitHub runners
do not provide the required devices. Run `.github/workflows/gpu-validation.yml`
on self-hosted runners before claiming Metal runtime validation:

1. Apple Silicon Metal runner labels: `self-hosted`, `macOS`, `ARM64`,
   `metal`
2. x86_64 CUDA runner labels: `self-hosted`, `Linux`, `X64`, `cuda`
3. Use the `run-timed-benchmarks` workflow input when a release needs measured
   GPU benchmark timing rather than compile-only coverage

Passing the CUDA self-hosted job validates `cuda-runtime` device-memory output,
strict CUDA-resident HTJ2K codestream decode and encode (tests, profiling, and
clippy), and the opt-in nvJPEG JPEG decode path on a CUDA runner. Timed NVIDIA
performance claims require the `run-timed-benchmarks` workflow input and
recorded benchmark output plus the benchmark publication report.

## Crates.io

Crates.io publication is staged because workspace crates depend on each other.
Before publishing, run `cargo xtask package` from a clean worktree. The package
preflight runs `cargo package --list` for every publishable crate,
then runs strict `cargo package --no-verify` only for crates that do not depend
on unpublished workspace versions. Downstream crates such as
`signinum-j2k-native`, `signinum-jpeg`, `signinum-tilecodec`, `signinum-j2k`,
adapter crates, `signinum-cli`, and `signinum` cannot pass strict pre-publish
packaging until the prior staged crates exist on crates.io, because Cargo
resolves their versioned path dependencies against the registry during
packaging.

This is an unpublished workspace dependencies limit, not a package content
failure. The publish workflow's dry-run mode mirrors that limit: it uses
`cargo publish --dry-run` for registry-independent crates and
`cargo package --list` for crates blocked only by unpublished workspace
dependencies. Real publishes still run `cargo publish` in dependency order.

The crates.io publish order uses the current manifest versions and is enforced
by `.github/workflows/publish.yml`:

1. `signinum-core`
2. `signinum-cuda-runtime`
3. `signinum-profile`
4. `signinum-j2k-native`
5. `signinum-jpeg`
6. `signinum-tilecodec`
7. `signinum-j2k`
8. `signinum-transcode`
9. `signinum-jpeg-metal`
10. `signinum-j2k-metal`
11. `signinum-transcode-metal`
12. `signinum-jpeg-cuda`
13. `signinum-j2k-cuda`
14. `signinum-cli`
15. `signinum`

Every package in this list must have a fresh manifest version before a
metadata-refresh release, because crates.io package metadata is immutable after
publication. `signinum-transcode` and `signinum-transcode-metal` remain
experimental API crates even when published; downstream applications should pin
minor versions and treat their reports and accelerator heuristics as evolving
surfaces. `signinum-j2k-compare` remains `publish = false`; it is a local parity
oracle helper, not a released runtime dependency.
