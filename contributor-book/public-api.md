# Public API changes

Every published crate has an API tier in `release-crates.json`: `stable`,
`experimental`, `implementation`, or `binary`. What each tier promises users is
in [API stability](../book/stability.md). This chapter covers the checks that
keep us to it.

The tier is a support promise; it doesn't change the checks. Every published
library is built with `missing_docs` denied (`cargo xtask doc`), appears in
both API snapshots, and is checked by `cargo xtask semver`, whatever its tier.

For internals, prefer real privacy (`pub(crate)`, private modules) over
`#[doc(hidden)]`. Hidden items are still public Rust API: they are recorded in
a snapshot and reviewed like any other item.

## What a pull request must update

If your change adds, removes, or changes a public item of a published crate,
or changes behavior callers can see, update these in the same pull request:

- rustdoc for the items you touched, and the [book](../book/README.md) page
  if users would do something differently;
- integration tests for the behavior callers see;
- the public API snapshots, with `cargo xtask stable-api --write`;
- the review file for the version in progress, and its generated diff (see
  [Review files](#review-files));
- a `CHANGELOG.md` entry under `Unreleased`, marked **Breaking:** if callers
  have to change code (see [Writing documentation](documentation.md#changelog));
- [Format support](../book/format-support.md) and
  `corpus/j2k-conformance/support-inventory.tsv` if codec support changed
  (`cargo xtask public-support` checks them);
- [Benchmarks](benchmarks.md) if you changed how something is measured.

The "Public API review" job in the PR checks runs `cargo xtask stable-api` and
`cargo xtask semver` whenever the change can reach a published crate's API.
Both fail until the snapshots and review file match the code, so the change
gets reviewed before it reaches `main`.

## API snapshots

Two generated files list every public item of the published library crates:

- `xtask/api/stable-api-1.0.public-api.txt`: the API rustdoc shows.
- `xtask/api/stable-api-1.0.implementation-public-api.txt`: the items that
  only appear with `--document-hidden-items`, which is mostly
  `#[doc(hidden)]` API.

```bash
cargo xtask stable-api           # check
cargo xtask stable-api --write   # regenerate both files
```

This only runs on macOS, so the Metal APIs are included. It needs the
`cargo-public-api` version in `CARGO_PUBLIC_API_VERSION` and the nightly
toolchain in `PUBLIC_API_TOOLCHAIN`, both in `xtask/src/stable_api.rs`. Both
passes target `aarch64-apple-darwin`, so the output doesn't depend on the host
or the nightly channel.

- The first pass runs rustdoc with warnings denied and records the normal
  public API. It fails if that list comes out empty.
- The second pass adds `--document-hidden-items`, and the implementation file
  records only what it adds. Rustdoc sometimes rewrites re-export paths when
  hidden modules become visible; those rewritten paths are kept too.
- The two files are written together and rolled back together on failure.
- Generation refuses to run if compiler, rustdoc, target, wrapper,
  deployment-target, or flag environment variables such as `RUSTFLAGS` or
  `RUSTC_WRAPPER` are set, and both passes run through `rustup run` with the
  pinned toolchain.

Don't copy item lists into the docs. The snapshots and review files are the
record.

## Semver checks

```bash
cargo xtask semver
cargo xtask semver --write-report
```

`semver` also runs only on macOS. It runs `cargo-semver-checks` (pinned in
`xtask/src/semver.rs`) with the pinned Rust toolchain against the last
published release. That baseline is set by `SEMVER_BASELINE_VERSION`,
`SEMVER_BASELINE_TAG`, and `SEMVER_BASELINE_COMMIT` in the same file, and the
tag must peel to that commit. The constants move when the next version
starts. It also
regenerates both snapshot passes, compares them with the committed files, and
checks the added and removed items and the hidden-item count of each package
against the review file. `--write-report` rewrites the generated diff.

It rejects a workspace version that isn't newer than the baseline, so the
baseline only moves together with a version bump (see
[Starting the next version](releasing.md#starting-the-next-version)).

## Review files

Each release has a review file and a generated diff in
`xtask/release-evidence/public-api/`:

- `public-api-review-<version>.yml`, written by hand;
- `reviewed-public-api-diff-<version>.md`, written by
  `cargo xtask semver --write-report`.

`API_REVIEW_CONFIG` and `API_DIFF_REPORT` in `xtask/src/semver.rs` name the
files for the version in progress. The review file has an entry under
`reviews` for every published crate, with the fingerprints of its added and
removed items, its hidden-item count and fingerprint, and a `rationale` for
the change. A crate with hidden items says why in `hidden_rationale`.

Breaking changes go in `break_ledger`:

- Every removed item is listed with its package, a `summary`, and `migration`
  instructions. The list must match the generated diff exactly.
- A behavior change that the signatures don't show gets an entry of kind
  `behavior`, without removed items.
- A version bump before 1.0 doesn't excuse an unlisted break.

Earlier releases' files stay in the directory as the record of what each
release changed.
