# Contributing

Contributions should keep the workspace focused on practical JPEG 2000 / HTJ2K
codec infrastructure: safe parsing, predictable decode and encode behavior,
caller-owned scratch/context reuse, optional GPU acceleration where it is
measured to help, and reproducible benchmarks. Whole-slide imaging workloads are
important stress cases, but the public APIs are general codec APIs.

## Development Setup

Use the Rust toolchain pinned by `rust-toolchain.toml`.

```sh
cargo xtask fmt
cargo xtask clippy
cargo xtask test
cargo xtask doc
```

Comparator benchmarks may need optional system libraries. Benchmark rules are
in [`xtask/BENCHMARKS.md`](xtask/BENCHMARKS.md).

## Pull Requests

- Keep changes scoped to one codec, adapter, or documentation topic when
  possible.
- Add or update behavior-focused tests for decode, API, or data-flow changes.
- Do not remove passing regression tests as cleanup.
- Avoid hardcoded secrets, credentials, or local machine paths.
- Return an error for unsupported inputs and backend failures; do not add
  silent fallbacks.
- Run the relevant tests before opening a PR, and the workspace checks above
  before changes that affect a release.

## GPU Validation

The GPU validation workflow only runs on `workflow_dispatch`, not on
`pull_request` or `push`, because it uses paid self-hosted CUDA and Metal
runners.

Pull requests that touch CUDA, Metal, shared GPU-profile paths, or
`.github/workflows/gpu-validation.yml` must record a successful manual
`gpu-validation.yml` dispatch for the PR head SHA before merge. The normal CI
planner checks the PR diff, queries `gpu-validation.yml` runs by head SHA, and
fails until the required quick backend job names have succeeded:

- `CUDA quick validation` for CUDA or shared GPU changes.
- `Metal quick validation` for Metal or shared GPU changes.

A release candidate needs one `target=all`, `mode=full` dispatch for its exact
commit in which `CUDA full release validation` and `Metal full release
validation` both succeed. Performance and profiling runs go in the separate
`gpu-benchmarks.yml` workflow so they do not compete with validation.

Hosted macOS CI runs `cargo xtask metal-compile`, which only compiles and runs
the tests that need no GPU. The self-hosted Metal job runs
`cargo xtask release-metal` with either `--mode quick` or `--mode full`. It
fails on skipped runtime tests or a missing Metal device.

Do not add `pull_request` or `push` triggers to `gpu-validation.yml` without
agreeing on it first.

## Public API Changes

Changes to ROI, scaled decode, tile-batch, row-streaming, context, scratch-pool, or device
surface behavior should update:

- README quick-start or examples when user-facing behavior changes
- API docs for affected public items
- integration tests covering caller-visible behavior
- `xtask/BENCHMARKS.md` when benchmark methodology changes
- `xtask/api/stable-api-1.0.public-api.txt` and
  `xtask/api/stable-api-1.0.implementation-public-api.txt`, regenerated together
  with `cargo xtask stable-api --write`, when public items change
- `docs/public-support.md`, checked with `cargo xtask public-support`, when
  codec support changes
- `cargo xtask semver` for the stable published libraries

## Releases

The release process and its required checks are in
[`xtask/RELEASING.md`](xtask/RELEASING.md).
