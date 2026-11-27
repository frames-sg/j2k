# CI and GPU validation

Most checks run on GitHub-hosted runners. CUDA and Metal tests need real
GPUs, which run on paid self-hosted machines, so they only start when someone
dispatches them by hand. This chapter says which jobs a pull request gets and
when it also needs a GPU run.

## Pull request checks

`.github/workflows/ci.yml` (named "PR checks") runs on every pull request. Its
first job, the change planner, runs `scripts/ci_plan.py plan` on the diff and
decides which of the other jobs are needed.

| Job | Runs | What it does |
| --- | --- | --- |
| Change planner | always | Classifies the changed paths (below). |
| GitHub Actions verifier tests | always | Unit tests for the Python scripts in `scripts/`. |
| Diff check | always | `git diff --check` on the pull request. |
| Secret scan | always | gitleaks over the repository. |
| Minimum supported Rust | always | `cargo check` of the workspace with the MSRV, and of the CUDA device crates with the CUDA-Oxide compiler. |
| Rust quality | Rust changes | `cargo xtask fmt`, `clippy`, `clippy-strict`, `test`, `release-integrity`, `repo-lint`, and `x86-codegen-audit`. |
| Documentation | documentation changes | `cargo xtask doc`, `release-integrity`, `public-support`, and `repo-lint`. |
| Metal compile and pure tests | changes that can affect Metal | `cargo xtask metal-compile` on hosted macOS. |
| Public API review | changes that can affect a published API | `cargo xtask stable-api` and `cargo xtask semver` on macOS (see [Public API changes](public-api.md)). |
| C/C++ API | changes that can affect `j2k-c` | Tests, packages, and builds the C and C++ consumers against the package on Ubuntu and macOS. |
| GPU evidence policy | always | Requires a GPU validation run for the head commit when the change can affect CUDA or Metal. |
| PR required checks | always | Fails unless every always-on job and every job the planner required succeeded. |

The planner classifies each changed path:

- **Documentation:** anything under `book/`, `contributor-book/`, or `docs/`;
  the root `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `AGENTS.md`,
  `CLAUDE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `NOTICES.md`, and license
  files; `.github/pull_request_template.md`; and Markdown under `corpus/`,
  `crates/`, or `xtask/`, except inside a `src/` directory or under
  `xtask/release-evidence/`.
- **Everything:** `Cargo.toml`, `Cargo.lock`, `deny.toml`,
  `rust-toolchain.toml`, `typos.toml`, and anything under `.cargo/`,
  `.github/workflows/`, `scripts/`, or `xtask/`. The API snapshots under
  `xtask/api/` and the review files under `xtask/release-evidence/public-api/`
  count as everything too, documentation job included.
- **A crate's files:** Rust quality, plus the public API job if a published
  crate depends on the crate (normal dependencies only), the C/C++ job if
  `j2k-c` does, and the CUDA or Metal lane if any CUDA or Metal crate does.
  `j2k-ml` and `j2k-python` count for their GPU lanes directly.
- Any other path is treated as everything, so an unknown file never skips a
  check.

`scripts/tests/test_ci_plan.py` has examples of each case.

The `python-bindings.yml` workflow also runs on pull requests that touch the
Python crate or the crates it builds on. It builds and tests CPU wheels on
Ubuntu and CPU/MPS wheels on macOS.

## GPU validation

`.github/workflows/gpu-validation.yml` only runs on `workflow_dispatch`, never
on `pull_request` or `push`, because it uses the self-hosted CUDA and Metal
runners. Only the maintainer account can dispatch it; ask in the pull request
if you need a run. Don't add `pull_request` or `push` triggers to it without
agreeing on it first.

A pull request needs a successful dispatch for its exact head commit before it
can merge when the planner marks CUDA or Metal as affected: a change to a CUDA
or Metal crate or anything they depend on (`j2k`, `j2k-native`, `j2k-core`,
and so on), or to one of the "everything" paths above. Documentation never
needs one.

```bash
gh workflow run gpu-validation.yml --ref <pr-branch> -f target=all -f mode=quick
```

`target` is `cuda`, `metal`, or `all`. The GPU evidence policy job queries the
dispatched runs for the pull request's head commit and fails until the
required jobs have succeeded:

- `GPU / CUDA quick validation` for changes that can affect CUDA;
- `GPU / Metal quick validation` for changes that can affect Metal.

When a GPU validation run succeeds, the `gpu-evidence-rerun.yml` workflow
reruns the failed jobs of the latest PR checks run for that commit, if the GPU
evidence policy job was the one that failed. You don't need to push again.

The quick jobs run `cargo xtask release-cuda --mode quick` and
`cargo xtask release-metal --mode quick`. Both fail on a skipped runtime test
or a missing device (see [Testing](testing.md#gpu-tests)). Hosted macOS only
runs `cargo xtask metal-compile`, which compiles everything and runs the tests
that need no GPU.

A release candidate needs one dispatch with `target=all` and `mode=full` for
its exact commit, in which `GPU / CUDA full release validation` and
`GPU / Metal full release validation` both succeed. The full jobs also build
the packaged GPU crates as an outside consumer, run the CUDA or Metal T.803
reports, and measure GPU-lane coverage. See [Releasing](releasing.md).

## GPU benchmarks

Performance and profiling runs go through the separate `gpu-benchmarks.yml`
workflow so they don't hold up validation. Its suites are `smoke`,
`criterion`, `profile`, `adoption`, and `routing`; they are described in
[Benchmarks](benchmarks.md).

## Full validation

`.github/workflows/full-validation.yml` runs on every push to `main`, every
Sunday, and on manual dispatch. Beyond the PR checks it runs, among others:

- strict Clippy, the panic-surface ratchet, the clone audit, `cargo machete`,
  `cargo deny`, and typos;
- comparator parity against OpenJPEG, Grok, and libjpeg-turbo
  (`cargo xtask j2k-bench-signoff`);
- semver checks and the API snapshots;
- codec-math and routing-table freshness, `public-support --final`, and
  `release-integrity`;
- release-mode CPU tests and T.803 CPU reports on Linux x86-64, macOS arm64,
  and Windows x86-64;
- `no-std`, Miri, the benchmark build, fuzz build, and a short fuzz run (a
  longer one on the weekly schedule);
- `cargo xtask package`, `j2k-ml` package smoke test, and host coverage;
- the prebuilt C/C++ packages for every release platform
  (`c-packages.yml`), each tested from its archive.

The `Release candidate aggregate` job lists every job a release needs and
fails unless all of them succeeded.

## Other workflows

- `publish.yml` runs on release tags, publishes to crates.io, and attaches
  the C/C++ packages to a draft GitHub release. See [Releasing](releasing.md).
- `c-packages.yml` builds the C/C++ packages on Linux x86-64 and arm64, macOS
  arm64, and Windows x86-64, and tests each one from its archive. Full
  validation and the publish workflow call it. A manual dispatch is a dry run:
  it only uploads workflow artifacts and uses no secrets.
- `python-bindings.yml` can also be dispatched by hand to validate a CUDA
  wheel on the GPU runner or publish wheels to PyPI.
- `secret-scan.yml` runs gitleaks on pushes and is reused by the PR checks.
