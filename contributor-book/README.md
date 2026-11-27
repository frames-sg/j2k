# The J2K contributor book

This book is for people changing J2K: fixing a bug, adding a codec feature,
tuning a GPU kernel, or cutting a release. If you want to use J2K, read the
[J2K Book](../book/README.md) instead.

Start with the root [CONTRIBUTING.md](../CONTRIBUTING.md) for how we review
pull requests. This book covers the rest: how the code is laid out and how to
build, test, benchmark, and release it.

## How the workspace is organized

J2K is one Cargo workspace. `j2k` is the crate most people use; it owns the
public JPEG 2000 and HTJ2K API and calls into `j2k-native`, the CPU engine.
`j2k-jpeg` is the JPEG codec. Each GPU backend is split into a codec-neutral
runtime (`j2k-cuda-runtime`, `j2k-metal-support`), kernel crates, and the
adapters users call (`j2k-cuda`, `j2k-metal`, and their JPEG and transcode
siblings). Framework integrations (`j2k-ml`, `j2k-mpsgraph`), the Python and C
bindings, and the `j2k` command line sit on top. Test, comparison, and
conformance tooling lives in `j2k-test-support`, `j2k-compare`, and
`j2k-t803`, and `xtask` runs the repository checks.
[Architecture](architecture.md) has the full crate list and the rules for how
they may depend on each other.

## Chapters

- [Getting started](getting-started.md): toolchain, building, everyday
  commands, the bindings, and a map of the repository.
- [Architecture](architecture.md): crates, dependency rules, and how CPU and
  GPU decoding are put together.
  - [Metal JPEG decoder](metal-jpeg-decoder.md): how `j2k-jpeg-metal` decodes
    and routes.
- [Testing](testing.md): test layout, fixtures, fuzzing, T.803 conformance
  runs, coverage, Miri, and the repository lints.
- [Benchmarks](benchmarks.md): corpora, comparators, harness rules, and how to
  report numbers.
- [CI and GPU validation](ci.md): what runs on a pull request and when you
  need a GPU run.
- [Public API changes](public-api.md): API snapshots, semver checks, and the
  review files.
- [Releasing](releasing.md): freezing, checking, and publishing a release.
- [Environment variables](environment-variables.md): variables read by tests,
  benchmarks, and xtask.
- [Writing documentation](documentation.md): where docs go and how to write
  them.
