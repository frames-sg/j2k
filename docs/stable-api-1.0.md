# Stable API Policy

The list of public API items is generated. This page explains how it is
generated and what each stability tier promises.

## Generated snapshot

The generated files are:

- `docs/stable-api-1.0.public-api.txt`
- `docs/stable-api-1.0.implementation-public-api.txt`

Check or regenerate them with:

```bash
cargo xtask stable-api
cargo xtask stable-api --write
```

This must run on macOS with `cargo-public-api` `0.52.0`
(`cargo install cargo-public-api --version 0.52.0 --locked`) and the
`nightly-2026-06-28` toolchain. Both passes target `aarch64-apple-darwin`, so
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
`docs/release-evidence/public-api/public-api-review-0.11.3.yml`. A package with
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
`1.96` and rejects the old `J2K_SEMVER_TOOLCHAIN` override.

The snapshots also cover the CLI exit codes described below. Do not copy the
item list into this page.

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

### Breaking releases

The currently published stable contract is the `0.11.x` line.

- `0.7.0` removed parts of the `0.6.2` API; it is not source compatible with
  `0.6.x`.
- `0.7.5` was a one-time exception to the patch-release rule: it removed
  pass-through public wrappers, and its API diff and changelog list every
  removed item with migration notes.
- `0.8.0` changed strict-decoding behavior and one warning variant; it is not
  source or behavior compatible with `0.7.x`. The review file lists the changes
  and migrations.
- `0.8.1` added exact-resolution and sRGB/ICC decode APIs and encode-stage
  context without changing any 0.8.0 item.
- `0.9.0` replaced the `metal-rs` types and `*Ref` wrappers in the Metal expert
  APIs with retained or borrowed `objc2-metal` protocol objects. Texture
  descriptors are now built with objc2-metal directly, and only the typed
  allocation and availability errors remain. The 0.9.0 review file lists every
  removed signature in `j2k-metal-support`, `j2k-jpeg-metal`, `j2k-metal`, and
  `j2k-transcode-metal` with migration notes; there is no compatibility layer.

`0.8.0` could only be compared against `v0.7.5`, and `0.9.0` only against
`v0.8.1`. `0.10.0` was compared directly with `v0.9.0` under a one-time
exception for the crate split, which is now disabled. The current semver
baseline is `v0.11.2`.

## Stability tiers

`release-crates.json` gives every published crate one `api_contract` value. The
tier sets the support promise; it does not change Rust visibility or exempt any
item from patch-release review.

Every published library is built with missing-docs enforcement, appears in
both API snapshots, and is checked for patch compatibility. Prefer real module
or item privacy over `#[doc(hidden)]` for internals.

- `stable`: `j2k`, `j2k-core`, `j2k-jpeg`, and `j2k-tilecodec`. The supported
  APIs for applications. Their documented behavior is the long-term
  compatibility promise.
- `experimental`: `j2k-native`, `j2k-cuda`, `j2k-metal`, `j2k-jpeg-cuda`,
  `j2k-jpeg-metal`, `j2k-transcode`, `j2k-transcode-cuda`,
  `j2k-transcode-metal`, and `j2k-ml`. Patch releases keep their APIs. A pre-1.0
  minor release may change them, with every break listed in the review file and
  migration notes. What they support at runtime also depends on features,
  hardware, and `docs/public-support.md`.
- `implementation`: `j2k-codec-math`, `j2k-profile`, `j2k-types`,
  `j2k-cuda-build-support`, `j2k-cuda-runtime`, `j2k-cuda-j2k-engine`,
  `j2k-cuda-jpeg-engine`, `j2k-cuda-transcode-engine`, and
  `j2k-metal-support`. These are published so the other J2K crates can use
  them; they are not meant as general extension APIs. They are documented,
  snapshotted, and patch-checked like the others, and a pre-1.0 minor release
  may change them under the same review rules.
- `binary`: `j2k-cli`. Its commands, output, and exit codes are described in
  the CLI section below instead of an API snapshot.

`j2k-codec-math` and `j2k-ml` are part of the published `0.7.5` semver
baseline. Test-support crates, comparators, and xtask helpers are not
published and have no compatibility promise.

Before `1.0`, a minor release may break the API only with a generated diff, a
break list in the review file, and migration notes. From `1.0`, the stable
crates follow normal semver for the major version.

## CLI behavior

`j2k-cli` supports:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`

Argument errors return exit code `2`: an unknown subcommand, a missing
`inspect` file, or malformed or unsupported `transcode` arguments. Runtime
failures, such as unreadable files or unsupported input, return exit code `1`.
Successful commands return `0` and print one summary line to stdout. Help, and
running with no subcommand, also return `0` and print usage to stderr.

Extra arguments after the `inspect` file are currently ignored. Do not rely on
this; it may become an error.
