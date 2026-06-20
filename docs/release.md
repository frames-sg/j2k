# Release Policy

The repository is staged for the `j2k` public crate release. Runtime backend selection defaults to `Auto`; CPU remains the portable baseline while supported device paths are selected only with validation and benchmark evidence.

## Versions and publish order

Release scripts must use manifest versions. Do not publish from stale hard-coded crate/version pairs.

Real publishes must run from tag `v<workspace.package.version>`. All publishable crates must share that workspace version. If a crate version is already on crates.io, the publish script fails by default; set `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent rerun.

Publish in this order:

1. `j2k-core`
2. `j2k-profile`
3. `j2k-types`
4. `j2k-cuda-runtime`
5. `j2k-metal-support`
6. `j2k-native`
7. `j2k-jpeg`
8. `j2k-tilecodec`
9. `j2k`
10. `j2k-transcode`
11. `j2k-transcode-cuda`
12. `j2k-jpeg-metal`
13. `j2k-metal`
14. `j2k-transcode-metal`
15. `j2k-jpeg-cuda`
16. `j2k-cuda`
17. `j2k-cli`

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
cargo xtask release-integrity
```

The integrity gate parses cargo metadata, manifests, `.github/workflows/publish.yml`, and this release document. It fails if a publishable workspace crate is missing from publish order, docs.rs metadata, semver/doc gates, or release docs, or if a workspace crate is neither publishable nor explicitly `publish = false`.

## Required gates

Hosted CI must pass before release staging:

- formatting
- tests
- clippy
- release integrity
- package validation
- semver checks for stable packages
- docs and stable API inventory
- benchmark target compilation
- unsafe audit
- bounded fuzz run
- coverage via `cargo llvm-cov --fail-under-lines 80`

Metal runtime validation runs on macOS where available. J2K Metal Criterion
bench signoff is reset until new narrow profiling benches are added.

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use J2K-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Coverage exclusions are limited to hardware-only GPU paths that cannot execute on hosted CI: `j2k-cuda-runtime`, CUDA adapter crates, Metal adapter crates, and `j2k-metal-support`. Those paths still require the Metal/CUDA validation gates before release.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`j2k-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
