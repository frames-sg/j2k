# Stable API Policy

This page explains what each stability tier of the public API promises and
which releases broke it. The generated item list is maintained with the
release tooling; see [`xtask/RELEASING.md`](../xtask/RELEASING.md#public-api-snapshots).

## Breaking releases

The currently published stable contract is the `0.14.x` line.

Before `1.0`, a minor release may break the API. Each release's generated API
diff and review file are in
[`xtask/release-evidence/public-api/`](../xtask/release-evidence/public-api/),
and the [changelog](../CHANGELOG.md) has the migration notes. `cargo xtask
semver` compares against the `v0.13.1` baseline.

| Release | What breaks |
| --- | --- |
| `0.14.0` | Struct literals of `JpegToHtj2kOptions` must set the new `precision` field. Memory and image-size limit failures are `J2kError::ResourceLimit`, not `J2kError::NativeDecode`. Metal Tier-1 device failures are `Error::Decode` and `J2kError::Backend`, not `Error::MetalKernel`. |
| `0.13.0` | `J2kDirectGrayscalePlan` has a required `signed` field. `BatchDecodeOptions`, `j2k_jpeg::ColorTransform`, and the Burn tensor and error enums are non-exhaustive. Burn batch results are read through accessors. JPEG decode is strict by default. `BatchLayout` is defined in `j2k-core` (still re-exported by `j2k`). The `cuda-oxide-j2k-ml` feature of `j2k-cuda-j2k-engine` is gone. |
| `0.12.0` | Three rustdoc-hidden CUDA engine signatures changed. Pooled surfaces must be kept until their GPU work completes. |
| `0.11.0` | The experimental MPSGraph adapter dropped four demonstration helpers. |
| `0.10.0` | Crate split: defining paths changed, root re-exports kept. `transcode_kernels_built` moved to `j2k-cuda-transcode-engine`. |
| `0.9.0` | The Metal expert APIs take `objc2-metal` protocol objects instead of `metal-rs` types. There is no compatibility layer. |
| `0.8.0` | Decoding is strict by default. `J2kDecodeWarning::LenientDecodeMode` became `LenientMetadataRecovery`. |
| `0.7.5` | A one-time exception to the patch-release rule: it removed pass-through public wrappers. |
| `0.7.0` | Removed parts of the `0.6.2` API. |

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
  `j2k-transcode-metal`, `j2k-ml`, and `j2k-mpsgraph`. Patch releases keep their
  APIs. A pre-1.0 minor release may change them, with every break listed in the
  review file and migration notes. What they support at runtime also depends on
  features, hardware, and `docs/public-support.md`.
- `implementation`: `j2k-codec-math`, `j2k-profile`, `j2k-types`,
  `j2k-cuda-build-support`, `j2k-cuda-runtime`, `j2k-cuda-j2k-engine`,
  `j2k-cuda-jpeg-engine`, `j2k-cuda-transcode-engine`, `j2k-metal-support`, and
  `j2k-mpsgraph-support`. These are published so the other J2K crates can use
  them; they are not meant as general extension APIs. They are documented,
  snapshotted, and patch-checked like the others, and a pre-1.0 minor release
  may change them under the same review rules.
- `binary`: `j2k-cli`. Its commands, output, and exit codes are described in
  the CLI section below instead of an API snapshot.

The semver baseline covers all 24 published libraries. `j2k-c` (versioned by
its own C ABI revision), `j2k-python`, test-support crates, comparators, and
xtask helpers are not published to crates.io and have no Rust API promise.

Before `1.0`, a minor release may break the API only with a generated diff, a
break list in the review file, and migration notes. From `1.0`, the stable
crates follow normal semver for the major version.

## CLI behavior

`j2k-cli` supports:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`
- `j2k decode <opj_decompress options>`, and the binary invoked as
  `opj_decompress`

Argument errors return exit code `2`: an unknown subcommand, a missing
`inspect` file, or malformed or unsupported `transcode` or `decode` arguments.
Runtime failures, such as unreadable files or unsupported input, return exit
code `1`. Successful `inspect` and `transcode` commands return `0` and print
one summary line to stdout; a successful `decode` returns `0`, writes only the
`-o` file and prints nothing to stdout. Help, and running with no subcommand,
also return `0` and print usage to stderr.

Invoked as `opj_decompress`, the binary follows that tool's contract instead:
every failure, including argument errors, returns `1`, and `-h` prints usage
to stdout and returns `1`. Options and output formats are described in
`crates/j2k-cli/README.md`.

Extra arguments after the `inspect` file are currently ignored. Do not rely on
this; it may become an error.
