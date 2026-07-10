# Release Policy

The repository is staged for the `j2k` public crate release. Runtime backend selection defaults to `Auto`; CPU remains the portable baseline while supported device paths are selected only with validation and benchmark evidence.

## Release status

| Version | Distribution state | Security support |
| --- | --- | --- |
| `0.6.x` | Latest publicly published crates and documentation. | Supported. |
| `0.7.0` | Staged workspace release candidate; its changes remain under `Unreleased` until the tag and crates are published. | Not yet published or security-supported. |
| `<0.6` | Historical releases. | Unsupported. |

The workspace version records the staged package target; it does not by itself
mean that `0.7.0` has shipped. GitHub Pages is served directly from `main/docs`,
so pushing a frozen candidate to `main` also deploys its staged documentation
before the release tag and crates exist. Those pages must continue to identify
`0.6.x` as published and `0.7.0` as staged; a hosted page is not publication
evidence. After the tag and crates are published, update the site status in a
separate post-release commit.

## Candidate freeze and exact-SHA evidence

Finish source, generated artifacts, documentation, changelog, and package
metadata before freezing a candidate. The freeze starts only from a clean
worktree:

```bash
test -z "$(git status --porcelain)"
RC_SHA=$(git rev-parse HEAD)
```

Move the intended protected `origin/main` tip to exactly `RC_SHA` through the
repository's normal reviewed push/merge workflow, then run hosted CI and both
self-hosted GPU workflows for that exact commit. Verify the aggregate only
after those jobs have completed:

```bash
test "$(git rev-parse origin/main)" = "$RC_SHA"
cargo xtask release-status --sha "$RC_SHA"
```

Any tracked edit creates a new candidate: commit it, choose a new `RC_SHA`, and
rerun all exact-SHA evidence. Only after the verifier succeeds may the release
maintainer create an annotated `v<workspace-version>` tag that peels to
`RC_SHA`. Push that tag explicitly; do not use `--follow-tags`, move an existing
release tag, or treat a GitHub Pages deployment as release evidence.

## Versions and publish order

Release scripts must use manifest versions. Do not publish from stale hard-coded crate/version pairs.

Real publishes must run from tag `v<workspace.package.version>`. All
publishable crates must share that workspace version. If a crate version is
already on crates.io, the publish script fails by default; set
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent
rerun. A valid partial retry may contain only an already-published prefix of
the dependency-ordered list below; a published crate after an available crate
is inconsistent state and fails closed.

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
Use the repo-owned package gate from a clean worktree:

```bash
cargo xtask package
```

That gate applies package listing and dry-run checks according to dependency
availability:

```bash
cargo package --list
cargo package --no-verify
cargo publish --dry-run
```

The gate lists all 18 package contents. It then constructs `.crate` archives
with `cargo package --no-verify` for the 14 staged packages whose workspace
dependencies are not yet available from crates.io. The four
registry-independent packages (`j2k-core`, `j2k-profile`, `j2k-types`, and
`j2k-codec-math`) run
`cargo publish --dry-run`, including Cargo's package verification build. Manual
publish-workflow dry runs use the same split; listing alone is not package
construction.

Before the first real publish job, the hosted preflight verifies that the
checkout `origin` is the exact workflow repository, no draft, prerelease, or
published GitHub Release exists for the tag, and every target crate version has
a determinate crates.io state. Only an exact HTTP 404 means a version is
available; authentication errors, rate limits, server failures, timeouts, and
malformed responses stop publication. On an intentional partial retry,
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` permits the already-published prefix and
the per-crate jobs skip that prefix without moving the tag.
`CRATES_IO_PUBLISH_ATTEMPTS` must be a positive decimal integer;
`CRATES_IO_RATE_LIMIT_RETRY_SECONDS` and `CRATES_IO_INDEX_SETTLE_SECONDS` must
be nonnegative decimal integers. Invalid release-control values stop the script
before any registry operation.

Run this before publishing:

```bash
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask public-support --final
```

The codec-math codegen gate verifies generated Rust and Metal fragments against
the Rust source of truth. The integrity gate parses lockfile-strict cargo
metadata with `cargo metadata --locked --no-deps`, manifests,
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
- coverage via `cargo xtask coverage`
- hosted macOS Metal compilation and pure tests via `cargo xtask metal-compile`

Changed-line coverage includes production Rust across CPU and accelerator
crates. The hosted lane separately enforces the 80% threshold for changed
accelerator-host lines so broad CPU coverage cannot mask untested routing,
validation, allocation, or error-classification logic. GPU-heavy changes also
need self-hosted `gpu-validation` evidence. The Metal job delegates to
`cargo xtask release-metal`, which requires macOS, forces the strict runtime
gate, rejects GPU skip markers, checks named runtime sentinels and count floors,
and runs the exact declared ignored hardware-test inventory.

Benchmark compilation is a release build-health gate, not a performance
regression threshold. A release may claim performance only when the relevant
CPU, Metal, or CUDA benchmark artifacts are recorded in
[`docs/benchmark-evidence.md`](benchmark-evidence.md) or an attached run
bundle. `cargo xtask j2k-perf-guard` is available for explicit CPU Criterion
median regression signoff, but it is not part of the default release gate until
the release checklist supplies a baseline ref and artifact retention policy.
GPU performance signoff remains hardware-runner evidence, not hosted CI.

Hosted macOS runs `metal-compile` and does not claim hardware validation. A
release requires `release-metal` on a self-hosted Apple Silicon Metal runner;
missing devices, zero selected tests, skipped runtime paths, and inventory drift
are failures. These checks retain the per-backend minimum test count floors and
named runtime sentinels for every Metal-facing package. J2K Metal Criterion
bench signoff is reset until new narrow profiling benches are added.

The workspace already patches transitive `block v0.1.6` through
`third_party/block-0.1.6-patched` to mitigate its future-incompatibility
warning while `metal v0.33.0` remains the current crates.io release. Validate
the patch with lockfile-strict metadata plus the normal Metal build and runtime
gates. Remove it only after upstream `metal` no longer depends on the affected
crate or an approved replacement is adopted, and record that removal in the
release notes. Do not downgrade or merely silence the warning.

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use J2K-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Whole accelerator crates are not coverage exclusions. The only exclusions are
named non-host-instrumentable regions: CUDA SIMT device Rust, generated
cuda-oxide host scaffolds, the shared SIMT prelude, CUDA/NVTX FFI declaration
spans, and the embedded MSL string body. Each exclusion is tied to named
integrity or runtime-parity evidence. Metal and CUDA lanes publish separate LCOV
and summary artifacts and remain required before release.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`j2k-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
