# Stable API Policy

The stable API inventory is generated. The human-maintained policy is small:
stable crates must preserve the public codec contracts, while experimental
adapters may evolve until promoted.

## Generated snapshot

The generated item-level companions are:

- `docs/stable-api-1.0.public-api.txt`
- `docs/stable-api-1.0.implementation-public-api.txt`

Regenerate or check them with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

This task must run on macOS with `cargo-public-api` `0.52.0` installed
(`cargo install cargo-public-api --version 0.52.0 --locked`) and the pinned
`nightly-2026-06-28` toolchain available. Both passes explicitly target
`aarch64-apple-darwin` so target-gated Metal APIs and rustdoc formatting do not
silently change with the runner host or floating nightly channel.

The ordinary snapshot uses `RUSTDOCFLAGS=-D warnings` so its comparison with
the published 0.6.2 snapshot keeps the same scope. A second pass adds
`--document-hidden-items` and records only the extra rustdoc-hidden items in
the implementation snapshot. Rustdoc can rewrite equivalent re-export paths
when hidden modules become visible, so the generator forms a conservative full
candidate inventory from the union of both passes and writes its lexically
sorted difference from the ordinary pass. This guarantees that the combined
inventory remains a superset of the ordinary contract while retaining rewritten
path variants for review rather than silently dropping reachable API. An empty
full cargo-public-api pass fails the gate; an empty per-package hidden-only
difference is recorded truthfully.
The 0.6.2
baseline comparison continues to use only the ordinary snapshot. Those
adapters are implementation-facing, but they are still reachable Rust API and
therefore remain in the reviewed inventory. Do not use `#[doc(hidden)]` as a
compatibility escape hatch.

The published 0.6.2 artifact did not record a hidden-enabled pass. The
implementation companion is therefore staged-0.7 full-candidate inventory and
change-review evidence, not a reconstructed historical hidden-API baseline.
That artifact also recorded the `cargo-public-api` version but not its exact
nightly rustdoc build or target triple. This historical provenance gap cannot
be reconstructed reliably; the explicit 0.7 pins prevent it from recurring,
and the ordinary diff must be reviewed with that limitation in mind.
The generated 0.7 semver report compares the ordinary inventory with 0.6.2 and
also records each package's complete hidden-inventory count and fingerprint.
Every semver invocation collects both live passes, compares both committed
companions, and requires exact ordinary added/removed fingerprints plus the
hidden count/fingerprint in `engineering/public-api-review-0.7.0.yml`.
Nonempty hidden inventories also require a package-specific hidden rationale.
Consequently, additions and removals remain blocked until the snapshots,
report, and review evidence are updated together and their diffs are reviewed.

The two snapshot files are staged, synchronized, and committed as one
rollback-capable transaction. API generation rejects ambient compiler,
rustdoc, target, wrapper, deployment-target, bootstrap, and encoded flag
overrides that could silently change either pass. Toolchain selection is not
taken from the ambient Cargo process: both passes execute through the pinned
`rustup run` toolchain. `cargo xtask semver`
uses Rust `1.96` and does not accept the former `J2K_SEMVER_TOOLCHAIN` override.

The snapshots record the staged workspace's public items and the CLI exit-code
contract expectations. Manual prose in this file must not duplicate that
inventory. The published-baseline comparison belongs in the generated
[`0.7.0` reviewed API report](../engineering/reviewed-public-api-diff-0.7.0.md).
All generated artifacts remain provisional until they are regenerated and
verified after final source freeze.

The published stable contract is the `0.6.x` line. The workspace's `0.7.0`
inventory is a staged semver-review target whose changes remain unreleased; it
must not be presented as a published API until the release completes. Version
`0.7.0` intentionally contracts parts of the published pre-1.0 `0.6.2` API and
does not claim source compatibility with `0.6.x`.

## Stability tiers

- Primary stable user-facing codec APIs: `j2k`, `j2k-core`, `j2k-jpeg`,
  and `j2k-tilecodec`.
- Semver-gated published libraries: `j2k`, `j2k-core`, `j2k-jpeg`,
  `j2k-tilecodec`, `j2k-jpeg-metal`, `j2k-metal`,
  `j2k-jpeg-cuda`, `j2k-cuda`, `j2k-transcode`, `j2k-transcode-cuda`,
  `j2k-metal-support`, `j2k-transcode-metal`, `j2k-native`, `j2k-types`,
  `j2k-cuda-runtime`, and `j2k-profile`.
- Adapter and transcode crates are semver-gated published libraries, but their
  supported runtime shapes remain limited by feature gates, hardware
  availability, and `docs/public-support.md`.
- `j2k-codec-math` is included in the staged `0.7.0` semver inventory but is
  not yet published. Its presence here is a release-review target, not
  a claim that a public crates.io baseline already exists.
- Unpublished tooling: test support, comparators, and xtask automation helpers.

Patch releases preserve the active `0.x` public contract. Before `1.0`, a minor
release may intentionally change that contract only when the reviewed API diff
enumerates the change and the changelog provides migration guidance or states
that no compatibility replacement exists. Starting with `1.0`, stable crates
follow the normal compatibility guarantees for the declared major version.

## CLI contract

`j2k-cli` currently supports:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`

Recognized argument-validation failures return exit code `2`; these include an
unknown subcommand, a missing `inspect` file operand, and malformed or
unsupported `transcode` arguments. Runtime failures, including unreadable files
and unsupported codec inputs, return exit code `1`. Successful operational
commands return exit code `0` and write a single summary line to stdout. Help
and an invocation with no subcommand also return `0`, but print usage to stderr.

Additional arguments after the `inspect` file are currently ignored. This is a
known CLI limitation, not a stable contract: callers must not supply trailing
arguments or rely on them continuing to be accepted.
