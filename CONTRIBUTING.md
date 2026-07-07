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

Comparator benchmarks may need optional system libraries. The workspace README
defines benchmark publication and no-silent-skip behavior.

## Pull Requests

- Keep changes scoped to one codec, adapter, or documentation topic when
  possible.
- Add or update behavior-focused tests for decode, API, or data-flow changes.
- Do not remove passing regression tests as cleanup.
- Avoid hardcoded secrets, credentials, or local machine paths.
- Surface unsupported inputs and backend failures explicitly; do not add silent
  fallback paths.
- Run the narrowest relevant tests before opening a PR, then run the workspace
  checks above before release-facing changes.

## GPU Validation

The GPU validation workflow is intentionally `workflow_dispatch` only. It does
not run automatically on `pull_request` or `push` because it uses
cost-sensitive self-hosted CUDA and Metal runners.

Pull requests that touch CUDA, Metal, shared GPU-profile paths, or
`.github/workflows/gpu-validation.yml` must record a successful manual
`gpu-validation.yml` dispatch for the PR head SHA before merge. The normal CI
`gpu-path-policy` job checks the PR diff, queries `gpu-validation.yml` runs by
head SHA, and fails until the required backend job names have succeeded:

- `CUDA API compatibility on x86_64` for CUDA or shared GPU changes.
- `Metal validation on Apple Silicon` for Metal or shared GPU changes.

Do not add `pull_request` or `push` triggers to `gpu-validation.yml` without an
explicit policy decision.

## Public API Changes

Public decode APIs are part of the general codec integration surface. Changes to
ROI, scaled decode, tile-batch, row-streaming, context, scratch-pool, or device
surface behavior should update:

- README quick-start or examples when user-facing behavior changes
- API docs for affected public items
- integration tests covering caller-visible behavior
- the README benchmark policy when benchmark methodology changes
- `docs/stable-api-1.0.public-api.txt` via `cargo xtask stable-api --write`
  when semver-visible public items change
- `docs/public-support.md` plus `cargo xtask public-support` when codec support
  claims change
- `cargo xtask semver` for stable published library surfaces
