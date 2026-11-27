# Releasing

How a release is frozen, checked, and published. This page is the procedure;
it doesn't record past releases. For those:

- release notes: [GitHub Releases](https://github.com/frames-sg/j2k/releases);
- which versions get security fixes: [SECURITY.md](../SECURITY.md);
- T.803 results: [Conformance](../book/conformance.md);
- each release's API diff and review:
  [`xtask/release-evidence/public-api/`](../xtask/release-evidence/public-api/).

All crates share one version, `[workspace.package].version` in `Cargo.toml`,
and a release is published from tag `v<workspace.package.version>`.

## Overview

1. [Write the release notes](#release-notes) in `CHANGELOG.md`.
2. [Freeze a release candidate](#freezing-a-release-candidate) and run the
   offline checks.
3. Merge the candidate to `main` and [validate it](#validating-the-candidate)
   on hosted CI and the GPU runners.
4. [Tag it](#tagging). The tag push starts the publish workflow.
5. [Publish to crates.io](#publishing) once the environment is approved. The
   workflow then attaches the [C packages](#c-packages) to a draft GitHub
   release.
6. [Finish the GitHub release](#creating-the-github-release) by hand.
7. [Start the next version](#starting-the-next-version).

## Release notes

`CHANGELOG.md` holds only upcoming changes, under one `## Unreleased` heading,
grouped into `###` subject sections. It has no version sections. Published
release notes live on GitHub Releases. How to write entries is in
[Writing documentation](documentation.md#changelog).

`cargo xtask release-integrity` checks for exactly one `## Unreleased` heading
and no version headings. `cargo xtask release-integrity --publish` also
requires `Unreleased` to have at least one entry, so a release can't go out
without notes.

Before freezing a candidate, edit `Unreleased` into release-note shape:

- upgrade notes first: breaking changes, output changes, and security fixes;
- then the subject groups, merging related entries.

That edit is part of the frozen commit. Changing the notes later means a new
candidate.

## Freezing a release candidate

Finish code, generated files, docs, the release notes, and package metadata
first. Before freezing:

- Fill in the reviewer and review date in the `PATCH_PROVENANCE.md` of every
  `[patch.crates-io]` path override. `release-integrity` finds them from the
  workspace manifest and fails if any is missing or unapproved. The date must
  be a real `YYYY-MM-DD` date, not taken from commit metadata. The check runs
  even when there are no path overrides.
- A repository admin must enable GitHub private vulnerability reporting
  (**Settings > Security**). The candidate and tag verifiers fail if it's off.

Then freeze from a clean worktree:

```bash
test -z "$(git status --porcelain)"
RC_SHA=$(git rev-parse HEAD)
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask release-integrity --publish
cargo xtask public-support --final
cargo xtask package
```

If any command fails or changes a tracked file, commit the fix, take a new
`RC_SHA`, and run everything again.

- `codec-math-codegen` checks that the generated Rust and Metal fragments
  match their source.
- `release-integrity` is described [below](#what-release-integrity-checks).
- `public-support --final` checks [Format support](../book/format-support.md)
  against the tests and the support inventory (see
  [Testing](testing.md#format-support-matrix)). It isn't a conformance check.
- `package` is described under [Package checks](#package-checks).

## Validating the candidate

Move protected `origin/main` to exactly `RC_SHA` through a normal pull
request merge. Wait for `full-validation.yml` on that push. Then dispatch one
`gpu-validation.yml` run with `target=all` and `mode=full` for the same commit;
CUDA and Metal run in parallel (see [CI](ci.md#gpu-validation)). When all of
it has finished:

```bash
test "$(git rev-parse origin/main)" = "$RC_SHA"
cargo xtask release-status --sha "$RC_SHA" --scope all
```

`release-status` checks the hosted and GPU results and the T.803 reports for
that exact commit. `--scope cpu|cuda|metal` checks one backend at a time.

### Conformance reports

T.803 conformance is stated per backend. A CPU statement needs reports for
the exact commit from Linux x86-64, macOS arm64, and Windows x86-64, which full
validation produces. A CUDA or Metal statement needs that backend's own report
from real hardware, which the GPU full validation produces. Every report must
contain all selected cases with no skips. The publish workflow requires all
five reports and both GPU full-validation jobs.

### CUDA waiver

A release can go out without the CUDA jobs only with an approved waiver at
`xtask/release-evidence/cuda-waiver-<version>.md`. It still dispatches the
`target=all`, `mode=full` run; once the Metal full job passes, cancel the run
so the CUDA full job never starts. `release-status --scope all` and the
publish preflight then skip only the CUDA full validation job and the CUDA
T.803 report, and accept the cancelled run only if the CUDA job is its one
cancelled job.

The verifier accepts a waiver only for a version in `CUDA_WAIVER_VERSIONS` in
`scripts/github_actions_verify.py`, so a new waiver needs a code change that
goes through review. For a tag, the waiver must also match the tag.

### Required checks

The `Release candidate aggregate` job in `.github/workflows/full-validation.yml`
lists every hosted job a release needs; [CI](ci.md#full-validation) summarizes
them. On top of those:

- the GPU full jobs: `cargo xtask release-cuda --mode full` and
  `cargo xtask release-metal --mode full` on real hardware, the packaged-crate
  checks for `j2k-cuda`, `j2k-metal`, and `j2k-mpsgraph`, and the GPU coverage
  lanes (see [Testing](testing.md#coverage));
- route-parity tests for every fixed `Auto` decision, and verified routing
  results for any new GPU threshold (see
  [Benchmarks](benchmarks.md#auto-routing-tables)).

Compiling the benchmarks only shows that they build. Performance numbers in a
release come from results recorded in [Performance](../book/performance.md)
or an attached run bundle, and GPU numbers come from the hardware runners,
never hosted CI.

## Tagging

Only after `release-status` passes, create an annotated
`v<workspace.package.version>` tag on `RC_SHA` and push it by name:

```bash
VERSION=X.Y.Z   # [workspace.package].version
git tag -a "v$VERSION" "$RC_SHA" -m "J2K $VERSION"
git push origin "refs/tags/v$VERSION"
```

Don't use `--follow-tags`, don't move an existing release tag, and don't count
a GitHub Pages deploy as a release check. Pages is served from `main/docs`.

## Versions and publish order

[`release-crates.json`](../release-crates.json) is the one ordered list of
crates we publish. Release integrity, packaging, registry recovery, API tiers,
docs coverage, semver checks, and publishing all read it, and release scripts
must not keep their own lists. Each entry has the crate's `name` and its
`api_contract` tier.

- Every crates.io-publishable workspace member appears exactly once. A member
  restricted to another registry isn't publishable.
- Library tiers (`stable`, `experimental`, `implementation`) need a library
  target. The `binary` tier needs a binary target and no library.
- Every path dependency between release crates, dev-dependencies included,
  uses the exact `=<workspace.package.version>` requirement and resolves to
  the workspace crate, not a registry or Git crate of the same name.
- Normal and build dependencies, including optional and target-specific ones,
  set the publish order. `cargo xtask package` also requires a crate's
  dev-dependencies on other release crates to come earlier in the list.

The publish order is: `j2k-core`, `j2k-profile`, `j2k-types`, `j2k-codec-math`,
`j2k-cuda-build-support`, `j2k-cuda-runtime`, `j2k-cuda-j2k-engine`,
`j2k-cuda-jpeg-engine`, `j2k-cuda-transcode-engine`, `j2k-metal-support`,
`j2k-mpsgraph-support`, `j2k-native`, `j2k-jpeg`, `j2k-jpegls`, `j2k-tilecodec`, `j2k`,
`j2k-transcode`, `j2k-transcode-cuda`, `j2k-jpeg-metal`, `j2k-metal`,
`j2k-transcode-metal`, `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-ml`, `j2k-mpsgraph`,
and `j2k-cli`.

`j2k-mpsgraph-support` is also used by a JPEG XR adapter outside this
repository. Publish it before building that adapter from the registry; a local
`[patch.crates-io]` only validates development changes.

These workspace members aren't published to crates.io (`publish = false`):
`j2k-alloc-probe`, `j2k-c`, `j2k-compare`, `j2k-python`, `j2k-t803`,
`j2k-test-support`, `j2k-transcode-test-support`, and `xtask`. `j2k-c` ships as
native packages built by `crates/j2k-c/scripts/package.py` and attached to
the GitHub release (see [C packages](#c-packages)), and `j2k-python`
as the `j2k` wheel built by `python-bindings.yml`. Published crates need a
package README and docs.rs metadata.

## Package checks

Run from a clean worktree:

```bash
cargo xtask package
```

It lists every crate's contents with `cargo package --list` and builds a
`.crate` archive for every crate with `cargo package --no-verify`, in
dependency order. A crate that depends on no unpublished workspace crate,
dev-dependencies included, also gets `cargo publish --dry-run`, with Cargo's
verification build. A manual dispatch of the publish workflow is always a dry
run and never sees the crates.io token.

After building `j2k-ml`, it creates a project outside the workspace and
compiles the packaged crate against the registry versions of its
dependencies, with the `cpu`, `cuda`, and `cpu,cuda` features on Linux and
`cpu`, `metal`, and `cpu,metal` on macOS. The project creates the public
decoders. Temporary `[patch.crates-io]` entries are allowed only for
unpublished J2K crates. A GPU feature that fails to build in this outside
project blocks the release, even if it builds inside the workspace. To run
just these checks:

```bash
cargo xtask j2k-ml-package-smoke
cargo xtask package-consumer-smoke --target metal    # also builds j2k-mpsgraph
cargo xtask package-consumer-smoke --target cuda --cuda-runtime
```

## Publishing

Real publishes run only from tag `v<workspace.package.version>` through
`.github/workflows/publish.yml`.

Before publishing, the hosted preflight checks that `origin` is the workflow's
repository, that no GitHub release (draft, prerelease, or published) exists
for the tag, that the crates.io status of every crate version is known, and
that every archive packages. Only an exact HTTP 404 means a version is
unpublished; authentication errors, authorization failures, malformed
responses, and checksum mismatches stop the publish.

If a version is already on crates.io, the publish fails, unless
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` is set for a deliberate retry. A retry
may only skip an already-published prefix of the publish order, and each
published `.crate` SHA-256 must match the archive built locally from the tag.
A published crate after an unpublished one, or a checksum mismatch, stops the
publish. Retrying never moves the tag.

After the `crates-io-publish` environment is approved, one runner rechecks
the tag and the published prefix, packages every archive, and publishes the
remaining crates in order with `cargo publish --locked -p <crate>`, with
Cargo's verification build on. There are no fixed waits. Only transport
errors, HTTP 429, and server errors are retried, after 5, 15, and 30 seconds,
and the published prefix is rechecked before each retry. Authentication,
authorization, verification, manifest, version, and checksum errors are not
retried.

`scripts/publish-crate.sh`, when run directly, also requires the annotated
tag to exist and point at `HEAD` (`GITHUB_REF_NAME` is only an extra
consistency check), and refuses to run with tracked or untracked changes. It
takes the repository from `[workspace.package].repository`, normalizes HTTPS,
scp-style SSH, and `ssh://` URLs, requires `origin` to match, and checks that
the remote tag object and commit match the local tag and `HEAD`. These checks
run before any Cargo or registry call, and errors never print remote URLs or
transport messages that could contain credentials. The script then reruns the
strict offline integrity check.

### C packages

`publish.yml` calls `.github/workflows/c-packages.yml`, which builds a
C/C++ package on each platform with `package.py --archive`:

| Asset | Built on |
| --- | --- |
| `j2k-c-<version>-linux-x86_64.tar.gz` | `manylinux_2_28` x86-64 container, glibc 2.28 |
| `j2k-c-<version>-linux-aarch64.tar.gz` | `manylinux_2_28` arm64 container on an arm64 runner, glibc 2.28 |
| `j2k-c-<version>-macos-aarch64.tar.gz` | macOS arm64, `MACOSX_DEPLOYMENT_TARGET=11.0` |
| `j2k-c-<version>-windows-x86_64.zip` | Windows x86-64, MSVC |

Each archive has a `.sha256` file. A test job per platform downloads the
archive and runs `crates/j2k-c/scripts/test_package.py` on a plain runner: it
checks the checksum, unpacks the archive into a new directory, checks the
glibc or macOS floor, and builds and runs the native consumer tests (CMake,
and pkg-config outside Windows) and the packaged examples. A final job checks
that every archive is present and matches its `.sha256` file, and writes
`j2k-c-<version>-SHA256SUMS` with `scripts/c_release_assets.py manifest`. It
prints the manifest and its SHA-256 in the job summary.

The crates.io job waits for the C packages, so they are built, tested and
checksummed before the `crates-io-publish` environment is approved, and the
one approval covers both. After the crates are published, the
`c-release-assets` job runs `scripts/c_release_assets.py verify`: the tag must
be `v<version>`, the directory must hold exactly the four archives, their
`.sha256` files and the manifest, and the manifest must match the SHA-256
recorded before approval. It then creates a draft GitHub release for the tag,
titled `J2K X.Y.Z`, with those nine files. It is the only job with
`contents: write`, and it never replaces an existing release. If it fails
after creating the draft, delete the draft and re-run the failed job.

Full validation calls the same workflow on every push to `main`, and the
release candidate aggregate requires it, so a candidate that can't be
packaged and tested on every platform fails before it is tagged. To check
packaging on a branch without releasing, dispatch `c-packages.yml`; it only
uploads workflow artifacts and uses no secrets.

### Python wheels

The `j2k` wheel goes to PyPI separately from the crates. Dispatch
`.github/workflows/python-bindings.yml` by hand from `main` with `publish`
enabled; set `cuda` too to validate the CUDA wheel on the GPU runner first.
The publish job runs only after the wheel builds and tests pass, and any
requested CUDA check with them. It uses the `pypi` GitHub environment and
short-lived OIDC credentials, so no PyPI token is stored.

PyPI has to trust the workflow first: configure a Trusted Publisher with
project `j2k`, owner `frames-sg`, repository `j2k`, workflow
`python-bindings.yml`, and environment `pypi`. PyPI rejects a second upload of
the same version, so each PyPI release needs a new workspace version.

## What release-integrity checks

`cargo xtask release-integrity` runs offline. It reads
`cargo metadata --locked --no-deps`, `release-crates.json`, the manifests,
`.github/workflows/publish.yml`, `scripts/publish-crate.sh`, `CHANGELOG.md`,
and this page. It fails if:

- a publishable crate is missing from `release-crates.json` or this page, or
  lacks a package README or docs.rs metadata;
- a crate's version differs from the workspace version, or its tier doesn't
  match its targets;
- an internal dependency isn't pinned exactly, or the order is wrong;
- the changelog doesn't have exactly one `## Unreleased` heading, or has a
  version heading;
- a path override's patch review is missing or unapproved.

`cargo xtask release-integrity --publish` adds the checks that only apply to
a release commit, including at least one entry under `Unreleased`. The tag
workflow then uses the authenticated GitHub verifier to check private
vulnerability reporting, the annotated tag, and the hosted and GPU results for
the exact commit.

## Creating the GitHub release

After the crates.io publish, the publish workflow creates a draft release for
tag `v<workspace.package.version>`, titled `J2K X.Y.Z`, with the
[C packages](#c-packages) attached. Edit the draft by hand. Its body is:

1. a short summary paragraph;
2. the `Unreleased` text from `CHANGELOG.md` at the tag;
3. a final line:

   ```text
   **Full Changelog**: https://github.com/frames-sg/j2k/compare/vPREV...vX.Y.Z
   ```

Attach the T.803 reports from the full-validation and GPU validation runs,
then publish the draft. Don't paste workflow run IDs into the notes, and
don't replace the C package files.

## Starting the next version

After a release, `cargo xtask stable-api` and `cargo xtask semver` hold `main`
to the published API, so any public API change fails them. Do these steps
together, in the pull request that starts the next version, before landing
any API change. `cargo xtask semver` rejects a workspace version that isn't
newer than its baseline, so the baseline can't move without the version bump.

1. Bump `[workspace.package].version` and every `version = "=X.Y.Z"` path pin
   in the member manifests. Refresh `Cargo.lock` and each
   `crates/*/fuzz/Cargo.lock`.
2. Empty `Unreleased` in `CHANGELOG.md`, keeping the heading.
3. Point `cargo xtask semver` at the release just published. In
   `xtask/src/semver.rs`, set `SEMVER_BASELINE_VERSION`, `SEMVER_BASELINE_TAG`,
   and `SEMVER_BASELINE_COMMIT` (the commit the tag peels to), and point
   `API_DIFF_REPORT` and `API_REVIEW_CONFIG` at the new version's files. The
   tests in `xtask/src/semver/tests.rs` and `xtask/src/semver/tests/` name
   both versions; update them to match.
4. Add `public-api-review-<version>.yml` next to the earlier ones, with the
   new baseline and candidate and an entry for every published crate.
   Generate its diff with `cargo xtask semver --write-report`. See
   [Public API changes](public-api.md).
5. Add the release to the supported-versions table in `SECURITY.md`. The
   repo-lint test `security_policy_lists_the_latest_release_tag` requires that
   table to list the latest release tag. No other doc names the version.
