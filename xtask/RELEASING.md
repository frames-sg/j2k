# Release Policy

The `j2k` 0.11.3 public crate release is published and security-supported.
Its T.803 decoder conformance results are in
[`T.803 conformance`](../docs/t803-conformance.md). The
default backend is `Auto`: the CPU path is always available, and a GPU path is
used only for shapes where it was tested and measured faster.

## Release status

Version `0.11.3` is published. Its
[API report](release-evidence/public-api/reviewed-public-api-diff-0.11.3.md)
compares against `v0.11.2` and shows no public signature changes. The
[hosted checks](https://github.com/frames-sg/j2k/actions/runs/36392470784) and
[full CUDA/Metal validation](https://github.com/frames-sg/j2k/actions/runs/36396634283)
passed for the tagged commit, the release verifier checked all five T.803
reports, and the
[publish workflow](https://github.com/frames-sg/j2k/actions/runs/36402630060)
published all 25 crates. The reports are attached to the
[GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.3).

Version `0.11.2` is published. Its
[API report](release-evidence/public-api/reviewed-public-api-diff-0.11.2.md)
compares with `v0.11.1`: the stable APIs only gained items, and the five new
hidden native adapters do not change existing decoder behavior. The
[hosted checks](https://github.com/frames-sg/j2k/actions/runs/36353712175)
and [full CUDA/Metal validation](https://github.com/frames-sg/j2k/actions/runs/36355267186)
passed for the tagged commit, the release verifier checked all five T.803
reports, and the
[publish workflow](https://github.com/frames-sg/j2k/actions/runs/36358652009)
published all 25 crates. The reports are attached to the
[GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.2).

| Version | Distribution | Security support |
| --- | --- | --- |
| `0.11.3` | [crates.io](https://crates.io/crates/j2k/0.11.3), annotated tag `v0.11.3`; CPU, CUDA, and Metal reports on the [GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.3). | Yes |
| `0.11.2` | [crates.io](https://crates.io/crates/j2k/0.11.2), annotated tag `v0.11.2`; CPU, CUDA, and Metal reports on the [GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.2). | Yes |
| `0.11.1` | [crates.io](https://crates.io/crates/j2k/0.11.1), tag `v0.11.1`; notes in the [changelog](../CHANGELOG.md). | Yes |
| `0.11.0` | [crates.io](https://crates.io/crates/j2k/0.11.0); [GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.11.0). | Yes |
| `0.10.0` | crates.io, annotated tag `v0.10.0`. | Yes |
| `0.9.0` | crates.io, annotated tag `v0.9.0`. | Yes |
| `0.8.1` | crates.io, annotated tag `v0.8.1`. | Yes |
| `0.8.0` | crates.io, annotated tag `v0.8.0`. | Yes |
| `0.7.5` | crates.io. The `j2k-ml` `cuda` and `metal` features are broken (see below). | Yes, except those `j2k-ml` features |
| `0.7.0`–`0.7.3` | crates.io. | Yes |
| `0.6.x` | crates.io. | Security fixes only, until 1.0 |
| `<0.6` | Old releases. | No |

### Earlier releases

- `0.11.0` compares against `v0.10.0`
  ([API report](release-evidence/public-api/reviewed-public-api-diff-0.11.0.md),
  [review](release-evidence/public-api/public-api-review-0.11.0.yml)). It added
  the graph-submission support crate, JPEG classification and ICC APIs, and
  lossy HT quality-factor options. The experimental MPSGraph adapter dropped
  four demonstration helpers; build graphs with `MpsGraphProgram::new` and keep
  reference calculations in your own tests. CPU, CUDA, and Metal release checks
  passed, and the five conformance reports on the
  [0.11.0 release](https://github.com/frames-sg/j2k/releases/tag/v0.11.0)
  identify the tagged commit.
- `0.10.0` compares against `v0.9.0` (commit
  `b197f01ab4b9271f1cbc36921755a5b9d588bd5a`)
  ([API report](release-evidence/public-api/reviewed-public-api-diff-0.10.0.md),
  [review](release-evidence/public-api/public-api-review-0.10.0.yml)). It split
  the crates. Most removals in the report are changed defining paths whose root
  re-exports still exist. It also moved `transcode_kernels_built` from the CUDA
  runtime to the CUDA transcode engine and generalized the Metal resident
  codestream handoff to `DeviceCodestream`.
- `0.9.0` was published from annotated tag `v0.9.0` (commit
  `b197f01ab4b9271f1cbc36921755a5b9d588bd5a`) after
  [hosted validation](https://github.com/frames-sg/j2k/actions/runs/31427966052)
  and [CUDA/Metal validation](https://github.com/frames-sg/j2k/actions/runs/31427977279);
  the [publish workflow](https://github.com/frames-sg/j2k/actions/runs/31434104062)
  checked all five T.803 reports and published 19 crates. It compares against
  `v0.8.1` ([API report](release-evidence/public-api/reviewed-public-api-diff-0.9.0.md),
  [review](release-evidence/public-api/public-api-review-0.9.0.yml)) and breaks
  the expert Metal APIs on purpose: `metal-rs` device, queue, buffer, texture,
  descriptor, size, and pixel-format types became retained or borrowed
  `objc2-metal` objects. Texture descriptors are now built directly, and the
  old helper and unreachable raw-message-send errors were removed. The review
  file lists every removed item in the four Metal crates.
- `0.8.1` was published from annotated tag `v0.8.1` (commit
  `f92646d0e6f0d0ef6c1e60b60beaad29da1afd3b`) after
  [CPU validation](https://github.com/frames-sg/j2k/actions/runs/31140212203)
  and [CUDA/Metal validation](https://github.com/frames-sg/j2k/actions/runs/31141587695).
  It compares against `v0.8.0`
  ([API report](release-evidence/public-api/reviewed-public-api-diff-0.8.1.md),
  [review](release-evidence/public-api/public-api-review-0.8.1.yml)) and only
  adds items: exact-resolution and sRGB/ICC decode APIs, encode-stage context,
  and the shared irreversible midpoint calculation.
- `0.8.0` was published from annotated tag `v0.8.0` (commit
  `53e0ad3d4f75f492af55413e0dab5a5834bd09c6`) by the
  [publish workflow](https://github.com/frames-sg/j2k/actions/runs/30425822681),
  which checked all 19 crates. It is not source or behavior compatible with
  `0.7.5`: decoding is strict by default, lenient mode covers only the
  documented JP2/JPH metadata recoveries, warnings report recoveries that
  actually happened, and `J2kDecodeWarning::LenientDecodeMode` became
  `J2kDecodeWarning::LenientMetadataRecovery`. See the
  [API report][v0.8.0-api-report] and [review file][v0.8.0-api-review].
- `0.7.5` broke the patch-release rule once, on purpose, to remove
  pass-through wrappers. Migrations are under its heading in the
  [`CHANGELOG`](../CHANGELOG.md); its API report compares against `v0.7.3`.
  Its `j2k-ml` `cuda` and `metal` features do not compile for registry users
  because they call CubeCL and wgpu interop methods missing from the released
  versions of those crates; the `cpu` feature works. Since `0.8.0` the adapters
  decode on the GPU, copy pixels to host memory, and upload with Burn's public
  API. Do not use the 0.7.5 GPU features.
- The `0.7.x` releases removed parts of the `0.6.2` API and are not source
  compatible with `0.6.x`. The [`CHANGELOG`](../CHANGELOG.md) has migration
  notes and the [API report][v0.7.3-api-report] lists the changes.

GitHub Pages is served from `main/docs`.

[v0.7.3-api-report]: https://github.com/frames-sg/j2k/blob/v0.7.3/engineering/reviewed-public-api-diff-0.7.3.md
[v0.8.0-api-report]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/reviewed-public-api-diff-0.8.0.md
[v0.8.0-api-review]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/public-api-review-0.8.0.yml

## Freezing a release candidate

Finish code, generated files, docs, changelog, and package metadata first.
Then freeze from a clean worktree:

```bash
test -z "$(git status --porcelain)"
RC_SHA=$(git rev-parse HEAD)
cargo xtask release-integrity --publish
cargo xtask package
```

If either command fails, or anything tracked changes, commit the fix, take a
new `RC_SHA`, and rerun everything.

T.803 conformance is reported per backend. A CPU conformance statement needs
reports for the exact commit from Linux x86-64, macOS arm64, and Windows
x86-64. A CUDA or Metal statement needs that backend's own report from real
hardware. Every report must contain all selected cases with no skips. A missing
GPU report only blocks the statement for that GPU; it does not affect the CPU
result. `--scope all` is a convenience for checking everything at once. Current
results are in [`docs/t803-conformance.md`](../docs/t803-conformance.md).

While preparing a release, the changelog has an `## [Unreleased]` heading and a
staged-version line. As the last edit before freezing, replace the heading with
`## [<workspace-version>] - YYYY-MM-DD` using the real tag date, and update any
docs that still point at `Unreleased`. Changing the date or notes later means a
new candidate.

Move protected `origin/main` to exactly `RC_SHA` through the normal reviewed
merge. Wait for `full-validation.yml` on that push, then dispatch one
`gpu-validation.yml` run with `target=all` and `mode=full` for that commit
(CUDA and Metal run in parallel). When all three jobs finish:

```bash
test "$(git rev-parse origin/main)" = "$RC_SHA"
cargo xtask release-status --sha "$RC_SHA" --scope all
```

Only after this passes, create an annotated `v<workspace-version>` tag on
`RC_SHA` and push it by name. Do not use `--follow-tags`, move an existing
release tag, or count a GitHub Pages deploy as a release check.

Before freezing, fill in the reviewer and review date in the
`PATCH_PROVENANCE.md` of every `[patch.crates-io]` path override. The integrity
command finds these from the workspace manifest and fails if any is missing or
unapproved. The date must be a real `YYYY-MM-DD` date and must not be taken
from commit metadata. The check runs even when there are no path patches. A
repository admin must also enable GitHub private vulnerability reporting
(**Security** settings); the candidate and tag verifiers fail if it is off.

## Versions and publish order

[`release-crates.json`](../release-crates.json) is the single ordered list of
release crates. Release integrity, packaging, registry recovery, API tiers,
docs coverage, semver checks, and publishing all read it. Schema 2 stores each
crate's name and its `api_contract` tier (`stable`, `experimental`,
`implementation`, or `binary`). Release scripts must not keep their own
crate/version lists.

Every crates.io-publishable workspace member appears exactly once (a member
restricted to another registry is not publishable). Library tiers need a
library target; the binary tier needs a binary target and no library. Every
path dependency between release crates, including dev-dependencies, uses the
exact `=<workspace.package.version>` requirement and resolves to the workspace
crate, not a registry or Git crate of the same name. Normal and build
dependencies (including optional and target-specific ones) set the publish
order; dev-dependencies do not.

Real publishes run from tag `v<workspace.package.version>`, and all crates
share that version. If a version is already on crates.io the publish script
fails, unless `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` is set for a deliberate
rerun. A retry may only skip an already-published prefix of the list below,
and each published `.crate` SHA-256 must match the archive built locally from
the tag. A published crate after an unpublished one, or a checksum mismatch,
stops the publish.

Publish in this order:

1. `j2k-core`
2. `j2k-profile`
3. `j2k-types`
4. `j2k-codec-math`
5. `j2k-cuda-build-support`
6. `j2k-cuda-runtime`
7. `j2k-cuda-j2k-engine`
8. `j2k-cuda-jpeg-engine`
9. `j2k-cuda-transcode-engine`
10. `j2k-metal-support`
11. `j2k-mpsgraph-support`
12. `j2k-native`
13. `j2k-jpeg`
14. `j2k-tilecodec`
15. `j2k`
16. `j2k-transcode`
17. `j2k-transcode-cuda`
18. `j2k-jpeg-metal`
19. `j2k-metal`
20. `j2k-transcode-metal`
21. `j2k-jpeg-cuda`
22. `j2k-cuda`
23. `j2k-ml`
24. `j2k-mpsgraph`
25. `j2k-cli`

Run the package check from a clean worktree:

```bash
cargo xtask package
```

It runs these as each crate's dependencies allow:

```bash
cargo package --list
cargo package --no-verify
cargo publish --dry-run
```

It lists every crate's contents, works out from locked Cargo metadata which
crates depend on unpublished workspace crates, builds `.crate` archives for
those with `cargo package --no-verify`, and runs `cargo publish --dry-run`
(with Cargo's verification build) for the rest. A manually triggered publish
workflow is always a dry run and never receives the crates.io token.

After building `j2k-ml`, the check creates a project outside the workspace and
compiles the packaged crate against the registry CubeCL and wgpu crates: `cpu`,
`cuda`, and `cpu,cuda` on Linux, and `cpu`, `metal`, and `cpu,metal` on macOS
(the combined set also builds docs). The project creates the public decoders.
Temporary `[patch.crates-io]` entries are allowed only for unpublished J2K
crates. To run just this check:

```bash
cargo xtask j2k-ml-package-smoke
```

The Metal check also builds `j2k-mpsgraph` from its packaged archive:

```bash
cargo xtask package-consumer-smoke --target metal
```

A GPU feature that fails in this outside project blocks the release, even if it
builds inside the workspace.

Before publishing, the hosted preflight checks that `origin` is the workflow's
repository, that no GitHub Release (draft, prerelease, or published) exists for
the tag, that every crate version's crates.io status is known, and that all
archives package. Only an exact HTTP 404 means a version is unpublished;
authentication errors, authorization failures, malformed responses, and
checksum mismatches stop the publish. For a deliberate partial retry,
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` allows skipping the checksum-matched
already-published prefix without moving the tag.

After the `crates-io-publish` environment is approved, one runner rechecks the
tag and published prefix, packages every archive, and publishes the remaining
crates in order with `cargo publish --locked -p <crate>`, with Cargo's
verification build on. There are no fixed waits. Only transport errors, HTTP
429, and server errors are retried, after 5, 15, and 30 seconds, and the
published prefix is rechecked before each retry. Authentication,
authorization, package verification, manifest, version, and checksum errors are
not retried.

Run these before publishing:

```bash
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask release-integrity --publish
cargo xtask public-support --final
```

`codec-math-codegen` checks that the generated Rust and Metal fragments match
their Rust source. `release-integrity` reads `cargo metadata --locked
--no-deps`, `release-crates.json`, the manifests,
`.github/workflows/publish.yml`, and this page. It fails if a publishable crate
is missing from the manifest, docs.rs metadata, semver/doc checks, or this
page; if a tier does not match the crate's targets; if an internal dependency
is not pinned exactly; or if the order is wrong.

Plain `release-integrity` runs offline before a candidate. It accepts the
`Unreleased` changelog state before a candidate, the dated state while checking
a release commit, and a new `Unreleased` section above the dated release after
publishing. `--publish` is also offline but requires exactly one dated heading
for the workspace version, no provisional changelog markers, and completed
patch-review fields. The tag workflow then uses the authenticated GitHub
verifier to check private vulnerability reporting, the annotated tag, and the
hosted and GPU results for the exact commit.

Running `scripts/publish-crate.sh` directly also requires the annotated tag to
exist and point at `HEAD` (`GITHUB_REF_NAME` is only an extra consistency
check) and refuses to run with tracked or untracked changes. It takes the
repository identity from `[workspace.package].repository`, normalizes HTTPS,
scp-style SSH, and `ssh://` URLs, requires `origin` to match, and checks that
the remote tag object and commit match the local tag and `HEAD`. Git URL
rewrites must still resolve to the same repository. These checks run before any
Cargo or registry call, and errors never print remote URLs or transport
messages that could contain credentials. The script then reruns the strict
offline integrity check.

`public-support --final` checks that the JPEG 2000 Part 1, JP2, HTJ2K Part 15,
JPH, known-limitation, and benchmark rows in `docs/public-support.md` match the
tests and the support inventory. It is not a T.803 conformance check.

## Required checks

Hosted CI must pass for exactly `RC_SHA` before release:

- formatting
- tests
- clippy
- strict clippy via `cargo xtask clippy-strict`
- panic count via `cargo xtask panic-surface`
- codec-math fragment freshness via `cargo xtask codec-math-codegen`
- release integrity
- package checks
- semver checks for the stable packages
- docs and the stable API snapshot
- benchmark targets compile
- unsafe audit
- a bounded fuzz run
- coverage via `cargo xtask coverage`
- macOS Metal compilation and pure tests via `cargo xtask metal-compile`
- T.803 CPU reports on Linux x86-64, macOS arm64, and Windows x86-64 for the
  exact commit, with every selected case present and passing, if the release
  states CPU conformance
- a CUDA or Metal T.803 report from real hardware, with the CPU/GPU split shown
  per case, if the release states conformance for that backend; compiling is
  not enough
- packaged-crate checks for `j2k`, `j2k-cuda`, and `j2k-metal`
- route-parity tests for every fixed `Auto` decision, plus external Criterion
  results and their artifact hash for any new GPU threshold

Changed-line coverage covers production Rust in the CPU and GPU crates. The
host lane requires 80% of changed production lines and, separately, 80% of
changed release-critical lines. The GPU lanes report raw host-Rust coverage
and require 80% for routing, validation, allocation, ownership, public API,
parser, security, and error-handling code. GPU kernel correctness is checked by
exact CPU/GPU output comparison on real hardware, not by line coverage, so
GPU-heavy changes need a self-hosted `gpu-validation` run. The Metal job calls
`cargo xtask release-metal`, which requires macOS, forces the runtime tests on,
fails on GPU skip markers, checks named runtime tests and minimum counts, and
runs the exact list of ignored hardware tests.

Compiling the benchmarks only checks that they build; it is not a performance
check. Performance numbers in a release must come from CPU, Metal, or CUDA
results recorded in [`docs/benchmark-evidence.md`](../docs/benchmark-evidence.md) or an
attached run bundle. `cargo xtask j2k-perf-guard --lane host` can compare CPU
Criterion medians against a baseline, but it is not a default release check
until the checklist names a baseline ref and how long artifacts are kept. GPU
performance comes from hardware runners, not hosted CI.

Hosted macOS only runs `metal-compile`. A release also needs `release-metal` on
a self-hosted Apple Silicon runner; a missing device, zero selected tests,
skipped runtime paths, or a changed test list fails the job. Each Metal package
keeps its minimum test count and named runtime tests. J2K Metal Criterion
sign-off is paused until new profiling benches are added.

Since 0.9.0 the Metal crates use the pinned `objc2`, `objc2-foundation`, and
`objc2-metal` crates instead of `metal-rs`, so they no longer need a crates.io
patch or the unmaintained `block 0.1.6` crate. Release review checks that the
objc2 versions are unified, runs the dependency-tree check, and runs the normal
Metal build and runtime checks.

CUDA validation needs a self-hosted CUDA machine for runtime tests and
performance numbers. The CUDA paths use J2K's own kernels and CUDA device
memory for supported shapes.

No GPU crate is excluded from coverage. The changed-line count includes
production code and required build scripts. `#[cfg(test)]` code, test targets,
and example/bench/fuzz targets are reported separately instead of as
uncovered production code.

Only these are excluded because they cannot be instrumented on the host: CUDA
SIMT device code, generated cuda-oxide host scaffolds, the shared SIMT prelude,
CUDA/NVTX FFI declarations, the embedded MSL string, and the generated
codec-math DWT fragment. Each is covered by a freshness, integrity, or
runtime-parity check instead. The Metal and CUDA lanes publish separate LCOV
and summary files and must pass before release.

Each coverage lane points `CARGO_LLVM_COV_TARGET_DIR` and
`CARGO_LLVM_COV_BUILD_DIR` at the same new empty directory and only uses
build-script output from that run to decide which custom `cfg`s are active, so
results from an earlier run cannot leak in. Every package with a build script
must produce current output; missing or conflicting output fails the lane. A
custom `cfg` whose value is unknown keeps both branches in the changed-line
count.

## Published and unpublished crates

Published crates must have a package README and docs.rs metadata. Tooling and
test helpers are not published, even though they share the workspace version.

`j2k-test-support` is an unpublished dev helper. The comparator crates and
automation tools are not runtime API.

## Public API snapshots

The generated files are:

- `xtask/api/stable-api-1.0.public-api.txt`
- `xtask/api/stable-api-1.0.implementation-public-api.txt`

Check or regenerate them with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

This must run on macOS with `cargo-public-api` `0.52.0`
(`cargo install cargo-public-api --version 0.52.0 --locked`) and the
`nightly-2026-08-13` toolchain. Both passes target `aarch64-apple-darwin`, so
the Metal APIs are included and the output does not change with the host or
the nightly channel.

The first pass runs with `RUSTDOCFLAGS=-D warnings` and lists the normal public
API. The second pass adds `--document-hidden-items`; the implementation file
records only the items that the second pass adds. Rustdoc sometimes rewrites
re-export paths when hidden modules become visible, so the implementation file
keeps those rewritten paths too rather than dropping them. The command fails if
the first pass is empty; an empty hidden-only list is recorded as empty.
`#[doc(hidden)]` items are still public Rust API and are reviewed like any
other item.

Every `cargo xtask semver` run generates both passes, compares them with the
committed files, and checks the added/removed fingerprints and the hidden-item
count against
`xtask/release-evidence/public-api/public-api-review-0.11.3.yml`. A package with
hidden items must give a reason for them in that file.

Any removed item must be listed in the review file with its package, a summary,
and migration instructions, and that list must match the generated diff
exactly. Behavior changes that a signature diff cannot show are listed the same
way, without removed items. A pre-1.0 version bump alone does not excuse an
undocumented break.

The two snapshot files are written together and rolled back together on
failure. Generation refuses to run if compiler, rustdoc, target, wrapper,
deployment-target, or flag environment variables are set, and both passes run
through `rustup run` with the pinned toolchain. `cargo xtask semver` uses Rust
`1.99.0` and rejects the old `J2K_SEMVER_TOOLCHAIN` override.

The snapshots also cover the CLI exit codes described below. Do not copy the
item list into the docs.

### Release comparisons

| Release | Compared with | Report |
| --- | --- | --- |
| 0.8.0 | 0.7.5 | [API diff][v0.8.0-api-report]; review file lists every break |
| 0.8.1 | 0.8.0 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.8.1.md), [review](release-evidence/public-api/public-api-review-0.8.1.yml); additions only |
| 0.9.0 | 0.8.1 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.9.0.md), [review](release-evidence/public-api/public-api-review-0.9.0.yml) |
| 0.10.0 | 0.9.0 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.10.0.md), [review](release-evidence/public-api/public-api-review-0.10.0.yml) |
| 0.11.0 | 0.10.0 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.11.0.md), [review](release-evidence/public-api/public-api-review-0.11.0.yml); removes experimental MPSGraph items |
| 0.11.1 | 0.11.0 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.11.1.md), [review](release-evidence/public-api/public-api-review-0.11.1.yml) |
| 0.11.2 | 0.11.1 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.11.2.md), [review](release-evidence/public-api/public-api-review-0.11.2.yml); stable changes are additions only |
| 0.11.3 | 0.11.2 | [API diff](release-evidence/public-api/reviewed-public-api-diff-0.11.3.md), [review](release-evidence/public-api/public-api-review-0.11.3.yml) |

Each report also records every package's hidden-item count and fingerprint.
The CPU, CUDA, and Metal release checks passed for each published release.

[v0.8.0-api-report]: https://github.com/frames-sg/j2k/blob/v0.8.0/engineering/reviewed-public-api-diff-0.8.0.md
