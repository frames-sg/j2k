# J2K

J2K is a JPEG 2000 and HTJ2K codec for Rust, with a JPEG codec, a
JPEG-to-HTJ2K transcoder, optional CUDA and Metal backends, and Python, C, and
command-line front ends. It decodes untrusted files, so correctness and bounded
resource use come before speed.

## Start here

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for the pull request rules and change
  ownership.
- The [contributor book](contributor-book/SUMMARY.md) covers architecture,
  testing, benchmarks, CI, and releases. The [J2K Book](book/SUMMARY.md)
  describes user-facing behavior.
- Check APIs, feature flags, and commands against this checkout. Don't rely on
  older releases or on other JPEG 2000 libraries.

## Repository layout

- `crates/j2k/`: the public JPEG 2000 and HTJ2K API. `crates/j2k-native/` is
  the codec engine underneath it. `crates/j2k-core/`, `j2k-types/`, and
  `j2k-codec-math/` hold types and math shared with the GPU crates.
- `crates/j2k-jpeg/` (JPEG), `crates/j2k-transcode/` (JPEG to HTJ2K), and
  `crates/j2k-tilecodec/` (TIFF tile decompression).
- `crates/j2k-cuda/`, `j2k-jpeg-cuda/`, `j2k-transcode-cuda/`: CUDA adapters
  over `j2k-cuda-runtime/` and the `j2k-cuda-*-engine/` kernel crates.
- `crates/j2k-metal/`, `j2k-jpeg-metal/`, `j2k-transcode-metal/`: Metal
  adapters over `j2k-metal-support/`.
- `crates/j2k-ml/` and `j2k-mpsgraph/`: Burn and MPSGraph adapters.
- `crates/j2k-python/`, `j2k-c/`, `j2k-cli/`: Python, C/C++, and the CLI.
- `crates/j2k-t803/`, `j2k-compare/`, `j2k-test-support/`: conformance
  runner, comparator harness, and test fixtures.
- `xtask/`: checks and release tooling (`cargo xtask --help`).
- `book/`, `contributor-book/`: documentation. `docs/` is the website only.

## Making changes

- Keep changes focused, and separate mechanical moves from behavior changes.
- Unsupported input and backend failures return errors. Don't add silent
  fallbacks, and never let an explicitly requested GPU backend run on the
  CPU.
- Bug fixes need a regression test. Don't delete passing tests.
- Every `unsafe` block needs a `// SAFETY:` comment.
- Run `cargo xtask fmt`, `cargo xtask clippy`, and the tests for the crates
  you touched. CUDA and Metal changes need GPU validation; see
  [CI and GPU validation](contributor-book/ci.md).
- Public API changes need updated snapshots; see
  [Public API changes](contributor-book/public-api.md).

## Pull requests

- When you open or update a pull request, read
  [.github/pull_request_template.md](.github/pull_request_template.md) and use
  its sections. `gh pr create --body` doesn't fill in the template for you.

## Writing documentation and changelog entries

Follow [Writing documentation](contributor-book/documentation.md). In short:

- Put each fact in one place: the book for users, the contributor book for
  contributors, rustdoc for API details, crate READMEs for the crates.io page.
- Write plainly. No audit language ("evidence", "promoted", "fail-closed",
  "signoff"), no CI or release process in user docs, no test function names in
  user docs, no "as of version X".
- Record user-visible changes under `Unreleased` in
  [CHANGELOG.md](CHANGELOG.md), grouped by subject, in one to three sentences
  each. Published release notes go on GitHub Releases, not in the changelog.
- Edit generated text down before committing it. If a sentence wouldn't
  survive a human reviewer asking "what does this mean for the user?", cut it.
