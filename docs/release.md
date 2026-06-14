# Release Policy

The repository is staged for the `signinum` facade release. Runtime backend selection defaults to `Auto`; CPU remains the portable baseline while supported device paths are selected only with validation and benchmark evidence.

## Versions and publish order

Release scripts must use manifest versions. Do not publish from stale hard-coded crate/version pairs.

Real publishes must run from tag `v<workspace.package.version>`. All publishable crates must share that workspace version. If a crate version is already on crates.io, the publish script fails by default; set `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent rerun.

Publish in this order:

1. `signinum-core`
2. `signinum-cuda-runtime`
3. `signinum-profile`
4. `signinum-j2k-types`
5. `signinum-j2k-native`
6. `signinum-jpeg`
7. `signinum-tilecodec`
8. `signinum-j2k`
9. `signinum-transcode`
10. `signinum-transcode-cuda`
11. `signinum-metal-support`
12. `signinum-jpeg-metal`
13. `signinum-j2k-metal`
14. `signinum-transcode-metal`
15. `signinum-jpeg-cuda`
16. `signinum-j2k-cuda`
17. `signinum-cli`
18. `signinum`

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

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use Signinum-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Coverage exclusions are limited to hardware-only GPU paths that cannot execute on hosted CI: `signinum-cuda-runtime`, CUDA adapter crates, Metal adapter crates, and `signinum-metal-support`. Those paths still require the Metal/CUDA validation gates before release.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`signinum-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
