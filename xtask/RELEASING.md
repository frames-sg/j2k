# Releasing

The `j2k` 0.14.0 public crate release is published and security-supported.

This page is how a release is frozen, checked, and published. It is the
procedure, not a release log:

- what each release changed: [`CHANGELOG.md`](../CHANGELOG.md)
- which versions get security fixes: [`SECURITY.md`](../SECURITY.md)
- T.803 results: [`docs/t803-conformance.md`](../docs/t803-conformance.md)
- each release's API diff and review:
  [`release-evidence/public-api/`](release-evidence/public-api/)
- each release's conformance reports and workflow runs: its
  [GitHub release](https://github.com/frames-sg/j2k/releases)

GitHub Pages is served from `main/docs`.

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
hardware. Every report must contain all selected cases with no skips.
`cargo xtask release-status --scope` can check one backend at a time, but the
tag-publish workflow requires all five reports and both GPU full-validation
jobs. Only a release with a reviewed CUDA waiver (below) is published without
the CUDA ones. Current results are in
[`docs/t803-conformance.md`](../docs/t803-conformance.md).

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

A release with a reviewed waiver at
`xtask/release-evidence/cuda-waiver-<workspace-version>.md` still dispatches
that `target=all`, `mode=full` run. Once the Metal full job passes, cancel the
run, so the CUDA full job never starts. `release-status --scope all` and the
publish preflight then skip only the CUDA full validation job and the CUDA
T.803 report. They accept the cancelled run only when the CUDA full job is
its one cancelled job. The verifier accepts a waiver only for a version listed
in the `CUDA_WAIVER_VERSIONS` allowlist in `scripts/github_actions_verify.py`,
which today holds only `0.13.0`, so a new waiver needs a reviewed code change.
For a tag, the waiver must also match the tag.

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
order. `cargo xtask package` also requires a release crate's dev-dependencies
on other release crates to come earlier in the list.

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

It lists every crate's contents and builds a `.crate` archive for every crate
with `cargo package --no-verify`. A crate that depends on no unpublished
workspace crate, dev-dependencies included, also gets `cargo publish
--dry-run`, with Cargo's verification build. A manually triggered publish
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

Hosted CI must pass for exactly `RC_SHA` before release. The `Release
candidate aggregate` job in `.github/workflows/full-validation.yml` lists
every required job; the main ones are:

- formatting
- tests
- clippy
- strict clippy via `cargo xtask clippy-strict`
- panic count via `cargo xtask panic-surface`
- codec-math fragment freshness via `cargo xtask codec-math-codegen`
- routing-table freshness via `cargo xtask promotion-codegen --check`
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
- packaged-crate checks for `j2k`, `j2k-cuda`, `j2k-metal`, and `j2k-mpsgraph`
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

The Metal crates use the pinned `objc2`, `objc2-foundation`, and `objc2-metal`
crates. Release review checks that the objc2 versions are unified, runs the
dependency-tree check, and runs the normal Metal build and runtime checks.

CUDA validation needs a self-hosted CUDA machine for runtime tests and
performance numbers. The CUDA paths use J2K's own kernels and CUDA device
memory for supported shapes.

No GPU crate is excluded from coverage. The changed-line count includes
production code and required build scripts. `#[cfg(test)]` code, test targets,
and example/bench/fuzz targets are reported separately instead of as
uncovered production code.

Only these are excluded because they cannot be instrumented on the host: CUDA
SIMT device code, generated cuda-oxide host scaffolds, the shared SIMT prelude,
CUDA/NVTX FFI declarations, and the generated codec-math DWT fragment. Each is covered by a freshness, integrity, or
runtime-parity check instead. The PyO3 extension sources in
`crates/j2k-python/src/` are also excluded from host coverage: they run only
inside a Python interpreter, which the Rust coverage lane does not load. The
named `test_adapter.py`, `test_cuda.py`, and `test_metal.py` cases in the
Python bindings workflow cover them instead. The Metal and CUDA lanes publish
separate LCOV and summary files and must pass before release.

Each coverage lane points `CARGO_LLVM_COV_TARGET_DIR` and
`CARGO_LLVM_COV_BUILD_DIR` at the same new empty directory and only uses
build-script output from that run to decide which custom `cfg`s are active, so
results from an earlier run cannot leak in. Every package with a build script
must produce current output; missing or conflicting output fails the lane. A
custom `cfg` whose value is unknown keeps both branches in the changed-line
count.

## Published and unpublished crates

Published crates must have a package README and docs.rs metadata.

These workspace members are `publish = false`: `j2k-alloc-probe`, `j2k-c`,
`j2k-compare`, `j2k-python`, `j2k-t803`, `j2k-test-support`,
`j2k-transcode-test-support`, and `xtask`. `j2k-c` ships as native packages
built by `crates/j2k-c/scripts/package.py`, and `j2k-python` as the `j2k`
Python wheel built by the `python-bindings.yml` workflow. No workflow creates
the GitHub release or attaches the T.803 reports; do that by hand after the
publish.

## Starting the next version

After a release, `cargo xtask stable-api` and `cargo xtask semver` hold `main`
to the published API: any public API change fails them. Do these steps
together before landing one. `cargo xtask semver` rejects a workspace version
that is not newer than its baseline, so the baseline cannot move without the
version bump.

1. Bump `[workspace.package].version` and every `version = "=X.Y.Z"` path pin
   in the member manifests. Refresh `Cargo.lock` and each
   `crates/*/fuzz/Cargo.lock`.
2. Add `## [Unreleased]` and the line ``Staged workspace version: `X.Y.Z`.``
   to the changelog; plain `release-integrity` requires both until the dated
   heading replaces them.
3. Point `cargo xtask semver` at the release just published. In
   `xtask/src/semver.rs`, set `SEMVER_BASELINE_VERSION`, `SEMVER_BASELINE_TAG`,
   and `SEMVER_BASELINE_COMMIT` (the commit the tag peels to), and point
   `API_DIFF_REPORT` and `API_REVIEW_CONFIG` at the new version's files. The
   tests in `xtask/src/semver/tests.rs` and `xtask/src/semver/tests/` name
   both versions; update them to match.
4. Add `public-api-review-<version>.yml` beside the earlier ones, with the new
   baseline and candidate and a reviewed entry for every published crate.
   Generate its diff with `cargo xtask semver --write-report`.
5. The repo-lint test `release_facing_docs_match_the_latest_release_tag`
   requires the release-status line in `README.md`, `SECURITY.md`, this page,
   `docs/stable-api-1.0.md`, and `docs/env-vars.md` to name the latest
   release tag.

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
`xtask/release-evidence/public-api/public-api-review-0.14.0.yml`. A package with
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

Do not copy the item list into the docs. Each release's API diff and review
file are in [`release-evidence/public-api/`](release-evidence/public-api/); the
review also records every package's hidden-item count and fingerprint.
