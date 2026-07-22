# Release Policy

The `j2k` 0.7.3 public crate release is published and security-supported.
Runtime backend selection defaults to `Auto`; CPU remains the portable baseline
while supported device paths are selected only with validation and benchmark
evidence.

## Release status

| Version | Distribution state | Security support |
| --- | --- | --- |
| `0.7.5` | Frozen source-incompatible release candidate; not published or tagged until the exact-SHA gates and annotated tag complete. | Not yet a published release line. |
| `0.7.3` | Latest publicly published crates and documentation. | Supported. |
| `0.7.2` | Previous published release line. | Supported. |
| `0.7.1` | Previous published release line. | Supported. |
| `0.7.0` | Previous published release line. | Supported. |
| `0.6.x` | Previous published release line. | Supported for security fixes during the 0.7 transition. |
| `<0.6` | Historical releases. | Unsupported. |

Version `0.7.3` is published from annotated tag `v0.7.3`, which peels to the
exact locally, hosted-CI, Metal, and CUDA verified release commit. GitHub Pages
is served directly from `main/docs`; this post-release state is documentation,
while the tag and crates.io records remain the publication evidence.

Version `0.7.3` retains the API contract introduced by `0.7.1`, which
intentionally contracts parts of the published pre-1.0 `0.6.2`
API. It does not claim source compatibility with `0.6.x`. The
[`CHANGELOG`](../CHANGELOG.md) provides migration notes, and the
[reviewed API report](../engineering/reviewed-public-api-diff-0.7.3.md)
records the additions, removals, and changed signatures. That report was
regenerated, independently reviewed, and verified for the published tag.
Any report prepared for a future release remains provisional until it is
regenerated and verified after that release's final source freeze.

The `0.7.5` candidate is an explicit source-compatibility exception to
the normal patch policy. Its wrapper-removal migrations are recorded under
the dated `0.7.5` heading in the [`CHANGELOG`](../CHANGELOG.md), and its reviewed
API evidence is compared directly with the published `v0.7.3` baseline. This
candidate statement does not authorize publication or assert that the candidate
has passed exact-SHA release gates.

## Candidate freeze and exact-SHA evidence

Finish source, generated artifacts, documentation, changelog, and package
metadata before freezing a candidate. The freeze starts only from a clean
worktree:

```bash
test -z "$(git status --porcelain)"
RC_SHA=$(git rev-parse HEAD)
cargo xtask release-integrity --publish
cargo xtask package
```

Both offline candidate gates run from that clean commit. A failure or any
tracked correction invalidates `RC_SHA`; commit the correction, choose a new
candidate SHA, and rerun the local and exact-SHA evidence.

During remediation, the changelog keeps a real `## [Unreleased]` heading and a
structured staged-version line. As the final release-preparation edit before
candidate freeze, replace that heading with `## [<workspace-version>] - YYYY-MM-DD` using the
actual intended tag date and update every staged-document reference that still
says the notes are under `Unreleased`. Do not guess the date early. Any later
date or note change creates a new candidate and requires the exact-SHA gates
again.

Move the intended protected `origin/main` tip to exactly `RC_SHA` through the
repository's normal reviewed push/merge workflow. Let `full-validation.yml`
finish for that push, then dispatch one `gpu-validation.yml` run with
`target=all` and `mode=full` for that exact commit. CUDA and Metal execute in
parallel within that run. Verify the evidence only after all three release jobs
have completed:

```bash
test "$(git rev-parse origin/main)" = "$RC_SHA"
cargo xtask release-status --sha "$RC_SHA"
```

Any tracked edit creates a new candidate: commit it, choose a new `RC_SHA`, and
rerun all exact-SHA evidence. Only after the verifier succeeds may the release
maintainer create an annotated `v<workspace-version>` tag that peels to
`RC_SHA`. Push that tag explicitly; do not use `--follow-tags`, move an existing
release tag, or treat a GitHub Pages deployment as release evidence.

Before final candidate freeze, complete both structured fields in every
`[patch.crates-io]` path override's `PATCH_PROVENANCE.md` record with the
actual reviewer identity and review date. The publish-integrity command
discovers these records from the workspace manifest and fails if any one is
missing or unapproved. The date must be a calendar-valid `YYYY-MM-DD`; never
infer either value from commit metadata. The patched `block`
[release approval record](../third_party/block-0.1.6-patched/PATCH_PROVENANCE.md)
remains the example for the required format.
Also have a repository administrator enable GitHub private vulnerability
reporting under **Security** settings before exact-SHA candidate verification.
The authenticated candidate verifier reads that repository setting and fails
closed unless it reports enabled; the later tag verifier reuses the same
prerequisite.

## Versions and publish order

[`release-crates.json`](../release-crates.json) is the ordered release manifest
and source of truth for release-integrity, package construction, registry
recovery, and publication. Release scripts must use manifest versions and must
not publish from stale hard-coded crate/version pairs.

Real publishes must run from tag `v<workspace.package.version>`. All
publishable crates must share that workspace version. If a crate version is
already on crates.io, the publish script fails by default; set
`CRATES_IO_ALLOW_PUBLISHED_RERUN=true` only for an intentional idempotent
rerun. A valid partial retry may contain only an already-published prefix of
the dependency-ordered list below, and every published `.crate` SHA-256 must
match the archive packaged locally from the exact tag. A published crate after
an available crate, or any checksum mismatch, is inconsistent state and fails
closed.

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
18. `j2k-ml`
19. `j2k-cli`

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

The gate lists all 19 package contents. It then constructs `.crate` archives
with `cargo package --no-verify` for the 15 staged packages whose workspace
dependencies are not yet available from crates.io. The four
registry-independent packages (`j2k-core`, `j2k-profile`, `j2k-types`, and
`j2k-codec-math`) run
`cargo publish --dry-run`, including Cargo's package verification build. Manual
publish-workflow runs remain dry-run-only: they validate the manifest and
construct every local archive without receiving the crates.io token.

Before publication, the hosted preflight verifies that the checkout `origin` is
the exact workflow repository, no draft, prerelease, or published GitHub
Release exists for the tag, every target crate version has a determinate
crates.io state, and all archives package locally. Only an exact HTTP 404 means
a version is available; authentication errors, authorization failures,
malformed responses, and checksum mismatches stop publication. On an
intentional partial retry, `CRATES_IO_ALLOW_PUBLISHED_RERUN=true` permits only
the checksum-matched already-published prefix without moving the tag.

After `crates-io-publish` environment approval, one runner repeats the canonical
tag and prefix proof, packages all 19 archives, and publishes the remaining
manifest entries sequentially with `cargo publish --locked -p <crate>`. Cargo's
verification build stays enabled. There are no unconditional registry sleeps;
only retryable transport, HTTP 429, or server failures are retried with bounded
5, 15, and 30 second delays. The publisher re-queries and checksum-validates the
entire prefix before each retry. Authentication, authorization, package
verification, manifest, version, and checksum failures are never retried.

Run this before publishing:

```bash
cargo xtask codec-math-codegen
cargo xtask release-integrity
cargo xtask release-integrity --publish
cargo xtask public-support --final
```

The codec-math codegen gate verifies generated Rust and Metal fragments against
the Rust source of truth. The integrity gate parses lockfile-strict cargo
metadata with `cargo metadata --locked --no-deps`, `release-crates.json`,
manifests, `.github/workflows/publish.yml`, and this release document. It fails if a
publishable workspace crate is missing from the dependency-ordered manifest, docs.rs metadata,
semver/doc gates, or release docs, or if a workspace crate is neither
publishable nor explicitly `publish = false`.

The ordinary integrity mode is an offline pre-candidate check and accepts the
structured `Unreleased` changelog state. `--publish` remains offline but
requires exactly one dated heading for the workspace version, rejects the
provisional changelog markers, and requires completed patch-review approval
fields. The tag workflow separately uses the authenticated GitHub verifier to
confirm private vulnerability reporting, the annotated tag, and exact-SHA
hosted/GPU evidence. A direct real invocation of `scripts/publish-crate.sh`
independently requires the expected annotated Git tag to exist and peel exactly
to `HEAD`, treats `GITHUB_REF_NAME` only as an additional consistency check,
and rejects tracked or untracked worktree changes. It derives the canonical
repository identity from `[workspace.package].repository`, normalizes secure
HTTPS, scp-style SSH, and `ssh://` checkout URLs, and requires the checkout
`origin` to match that identity. It then queries `origin` directly and requires
the exact remote tag object and its peeled commit to match the verified local
annotated tag and `HEAD`. Any Git URL rewrite must still resolve to the same
canonical identity. Origin and remote-tag failures stop before Cargo or any
registry operation, and diagnostics do not print remote URLs or transport errors
that could contain credentials. Finally, the script reruns the strict offline
integrity mode so it cannot bypass those source and metadata checks.

The public-support gate verifies that the JPEG 2000 Part 1, JP2, HTJ2K Part 15,
JPH, known-limitation, and publication-gate rows remain synchronized with tests
and the conformance manifest before a release can claim full scoped codec
support.

## Required gates

After the candidate is frozen and committed, hosted CI must pass for exactly
`RC_SHA` before release authorization:

- formatting
- tests
- clippy
- authoritative strict Clippy via `cargo xtask clippy-strict`
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

Changed-line coverage records production Rust across CPU and accelerator
crates. The host lane enforces 80% across all changed production Rust and an
independent 80% release-critical gate. Accelerator lanes report raw host-Rust
coverage as audit evidence and enforce 80% for release-critical routing,
validation, allocation, ownership, public-API, parser, security, and error
boundaries. Broad accelerator implementation correctness is enforced by exact
CPU/backend output parity and fail-closed hardware suites, not by tests written
only to execute lines. GPU-heavy changes therefore require self-hosted
`gpu-validation` evidence. The Metal job delegates to
`cargo xtask release-metal`, which requires macOS, forces the strict runtime
gate, rejects GPU skip markers, checks named runtime sentinels and count floors,
and runs the exact declared ignored hardware-test inventory.

Benchmark compilation is a release build-health gate, not a performance
regression threshold. A release may claim performance only when the relevant
CPU, Metal, or CUDA benchmark artifacts are recorded in
[`docs/benchmark-evidence.md`](benchmark-evidence.md) or an attached run
bundle. `cargo xtask j2k-perf-guard --lane host` is available for explicit CPU Criterion
median regression signoff, but it is not part of the default release gate until
the release checklist supplies a baseline ref and artifact retention policy.
GPU performance signoff remains hardware-runner evidence, not hosted CI.

Hosted macOS runs `metal-compile` and does not claim hardware validation. A
release requires `release-metal` on a self-hosted Apple Silicon Metal runner;
missing devices, zero selected tests, skipped runtime paths, and inventory drift
are failures. These checks retain the per-backend minimum test count floors and
named runtime sentinels for every Metal-facing package. J2K Metal Criterion
bench signoff is reset until new narrow profiling benches are added.

The workspace resolves `metal v0.33.0` and patches its transitive `block v0.1.6`
through `third_party/block-0.1.6-patched` to mitigate the dependency's
future-incompatibility warning. The
[patch provenance record](../third_party/block-0.1.6-patched/PATCH_PROVENANCE.md)
pins the source digests, documents the limited ABI spelling changes, and records
the candidate's maintainer approval. That approval alone is not release signoff
and does not replace validation with lockfile-strict metadata plus the normal
Metal build and runtime gates. Remove
it only after the resolved `metal` dependency no longer uses the affected crate
or an approved replacement is adopted, and record that removal in the release
notes. Do not downgrade or merely silence the warning.

This override protects repository builds only. Cargo [reads `[patch]` only
from the top-level workspace](https://doc.rust-lang.org/cargo/reference/overriding-dependencies.html#the-patch-section)
and ignores patch settings supplied by a dependency, so a crates.io consumer
of the published Metal adapters will still resolve upstream `metal 0.33.0` and
upstream `block 0.1.6` unless that consumer adds its own override. The current
upstream [`metal` manifest](https://github.com/gfx-rs/metal-rs/blob/master/Cargo.toml)
still declares `block 0.1.6`, and its
[README](https://github.com/gfx-rs/metal-rs/blob/master/README.md) marks
`metal` deprecated in favor of `objc2-metal`.
Do not describe the local patch as a downstream fix. The 0.7 package evidence
must record this resolution explicitly; migration to maintained `objc2-metal`
or another publishable dependency path remains tracked maintenance debt.

CUDA validation requires a self-hosted CUDA environment for runtime and NVIDIA performance evidence. CUDA paths use J2K-owned CUDA kernels, cuda-runtime integration, and CUDA device memory surfaces for supported shapes. NVIDIA performance claims require recorded self-hosted benchmark output.

Whole accelerator crates are not coverage exclusions. The changed-line
denominator covers executable production and required build-script Rust.
Syntax-level `#[cfg(test)]` code, Cargo test targets, and example/bench/fuzz
targets are reported as separate non-production source dispositions rather than
being mislabeled as uncovered production.

Reviewed non-host-instrumentable exclusions are exact and named: CUDA SIMT
device Rust, generated cuda-oxide host scaffolds, the shared SIMT prelude,
CUDA/NVTX FFI declaration spans, the embedded MSL string body, the generated
codec-math DWT fragment, and the vendored patched `block` FFI binding. Every
generated or reviewed-vendored line must match one of those exclusions and its
named freshness, integrity, or runtime-parity evidence. Metal and CUDA lanes
publish separate LCOV and summary artifacts and remain required before release.

Each coverage lane forces `CARGO_LLVM_COV_TARGET_DIR` and
`CARGO_LLVM_COV_BUILD_DIR` to the same unique empty directory and uses only
build-script outputs captured from that invocation for custom `cfg`
classification. This makes byte-identical build-script reruns valid current
evidence without admitting retained scopes from an earlier run. Every selected
package with a Cargo custom-build target must have current output; missing or
conflicting package evidence fails the gate. A custom cfg value not established
by current evidence remains unknown, so both it and its negation stay in the
changed-source denominator rather than disappearing as inactive.

## Published and unpublished crates

Published crates must declare package README files and docs.rs metadata.
Unpublished tooling and oracle helpers remain local even when versioned with the
workspace.

`j2k-test-support` is an unpublished dev helper. Comparator crates and
automation-only tooling are not runtime API.
