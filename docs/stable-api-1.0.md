# Stable API Policy

This page explains what each stability tier of the public API promises and
which releases broke it. The generated item list is maintained with the
release tooling; see [`xtask/RELEASING.md`](../xtask/RELEASING.md#public-api-snapshots).

## Breaking releases

The currently published stable contract is the `0.11.x` line. The `0.12.0`
candidate changes three rustdoc-hidden CUDA engine signatures and requires
consumers to retain pooled surfaces until their GPU work completes. See the
[review](../xtask/release-evidence/public-api/public-api-review-0.12.0.yml)
and [changelog](../CHANGELOG.md) for migration instructions.

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
baseline is `v0.11.3`.

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
