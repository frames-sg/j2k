# API stability

Every published J2K crate belongs to one of four tiers. The tier says how
much you can rely on its API.

| Tier | Crates | What we promise |
| --- | --- | --- |
| Stable | `j2k`, `j2k-core`, `j2k-jpeg`, `j2k-tilecodec` | The API for applications. Documented behavior is the long-term compatibility promise. |
| Experimental | `j2k-native`, `j2k-cuda`, `j2k-metal`, `j2k-jpeg-cuda`, `j2k-jpeg-metal`, `j2k-transcode`, `j2k-transcode-cuda`, `j2k-transcode-metal`, `j2k-ml`, `j2k-mpsgraph` | Patch releases don't break them. A minor release before 1.0 can, and the release notes say how to migrate. What they support at run time also depends on features and hardware; see [Format support](format-support.md). |
| Implementation | `j2k-codec-math`, `j2k-profile`, `j2k-types`, `j2k-cuda-build-support`, `j2k-cuda-runtime`, `j2k-cuda-j2k-engine`, `j2k-cuda-jpeg-engine`, `j2k-cuda-transcode-engine`, `j2k-metal-support`, `j2k-mpsgraph-support` | Published so the other J2K crates can depend on them. Don't build on them directly. They follow the same release rules as experimental crates. |
| Binary | `j2k-cli` | The commands, output, and exit codes described [below](#command-line). |

The tier of each crate is recorded in `release-crates.json` at the repository
root.

`j2k-c` (the C library) and `j2k-python` aren't published to crates.io, so
they have no Rust API promise. The C ABI has its own version number,
`J2K_ABI_VERSION`, which `j2k_abi_version()` returns at run time.

## Breaking changes

J2K is not at 1.0 yet. Until it is:

- A patch release (`0.x.y` to `0.x.(y+1)`) does not break the public API of
  any published crate.
- A minor release (`0.x` to `0.(x+1)`) can break it. Every break is listed in
  the release notes on [GitHub
  Releases](https://github.com/frames-sg/j2k/releases), with what to change.

From 1.0, the stable crates follow semantic versioning: only a major release
breaks them.

## Command line

`j2k-cli` installs the `j2k` binary, which has three commands:

- `j2k inspect <file>`
- `j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53`
- `j2k decode <opj_decompress options>`

Run under the name `opj_decompress` (for example through a symlink), the
binary takes `opj_decompress` options directly. The options and output
formats are described in [Python, C, and the command line](bindings.md).

Exit codes and output:

- `0` on success. `inspect` and `transcode` print one summary line to stdout.
  `decode` writes only the `-o` file and prints nothing to stdout.
- `1` on a runtime failure, such as an unreadable file or unsupported input.
- `2` on an argument error: an unknown command, a missing `inspect` file, or
  malformed or unsupported `transcode` or `decode` arguments.
- Help, and running with no command, print usage to stderr and return `0`.

Run as `opj_decompress`, the binary follows that tool's conventions instead:
every failure, argument errors included, returns `1`, and `-h` prints usage
to stdout and returns `1`.

Extra arguments after the `inspect` file are ignored today. Don't rely on
this; it may become an error.
