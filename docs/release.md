# Release Policy

The repository is staged for the `j2k` public crate release. Runtime backend selection defaults to `Auto`; CPU remains the portable baseline while supported device paths are selected only with validation and benchmark evidence.

## Versions and publish order

Release scripts must use manifest versions. Do not publish from stale hard-coded crate/version pairs.

Real publishes must run from tag `v<workspace.package.version>`. All publishable crates must share that workspace version. If a crate version is already on crates.io, the publish script fails by default; set `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent rerun.

Publish in this order:

1. `j2k-core`
2. `j2k-profile`
3. `j2k-types`
4. `j2k-codec-math`
5. `j2k-cuda-runtime`
6. `j2k-metal-support`
7. `j2k-native`
8. `j2k-jpeg`
9. `j2k-tilecodec`
10. `j2k`
11. `j2k-transcode`
12. `j2k-transcode-cuda`
13. `j2k-jpeg-metal`
14. `j2k-metal`
15. `j2k-transcode-metal`
16. `j2k-jpeg-cuda`
17. `j2k-cuda`
18. `j2k-cli`

Publish preflight must account for staged unpublished workspace dependencies.
Use package listing and dry-run checks according to dependency availability:

```bash
cargo package --list
cargo publish --dry-run
```

Some downstream packages may be validated with `cargo package --list` while
strict dry-run publishing is blocked by unpublished workspace dependencies.

Run this before publishing:

```bash
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask public-support --final
```

The codec-math codegen gate verifies generated Rust and Metal fragments against
the Rust source of truth. The integrity gate parses cargo metadata, manifests,
`.github/workflows/publish.yml`, and this release document. It fails if a
publishable workspace crate is missing from publish order, docs.rs metadata,
semver/doc gates, or release docs, or if a workspace crate is neither
publishable nor explicitly `publish = false`.

The public-support gate verifies that the JPEG 2000 Part 1, JP2, HTJ2K Part 15,
JPH, known-limitation, and publication-gate rows remain synchronized with tests
and the conformance manifest before a release can claim full scoped codec
support.

## Required gates

Hosted CI must pass before release staging:

- formatting
- tests
- clippy
- panic-surface ratchet via `cargo xtask panic-surface`
- codec math fragment freshness via `cargo xtask codec-math-codegen`
- release integrity
- package validation
- semver checks for stable packages
- docs and stable API inventory
- benchmark target compilation
- unsafe audit
- bounded fuzz run
- coverage via `cargo llvm-cov --fail-under-lines 80`

Benchmark compilation is a release build-health gate, not a performance
regression threshold. A release may claim performance only when the relevant
CPU, Metal, or CUDA benchmark artifacts are recorded in
[`docs/benchmark-evidence.md`](benchmark-evidence.md) or an attached run
bundle. `cargo xtask j2k-perf-guard` is available for explicit CPU Criterion
median regression signoff, but it is not part of the default release gate until
the release checklist supplies a baseline ref and artifact retention policy.
GPU performance signoff remains hardware-runner evidence, not hosted CI.

Metal runtime validation runs on macOS where available. J2K Metal Criterion
bench signoff is reset until new narrow profiling benches are added.

Rust currently reports a future-incompatibility warning for transitive
`block v0.1.6` through `metal v0.33.0`. Track this as Metal dependency debt
until upstream `metal` removes or updates the dependency; do not downgrade,
fork, or silence it without a replacement path and release note.

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use J2K-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Coverage exclusions are limited to hardware-only GPU paths that cannot execute on hosted CI: `j2k-cuda-runtime`, CUDA adapter crates, Metal adapter crates, and `j2k-metal-support`. Those paths still require the Metal/CUDA validation gates before release.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`j2k-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
