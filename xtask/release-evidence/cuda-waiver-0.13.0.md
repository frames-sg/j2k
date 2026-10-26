# CUDA evidence waiver for j2k 0.13.0

This waiver applies only to the 0.13.0 release. It lets the publish
preflight and `cargo xtask release-status --scope all` accept the release
candidate without two items:

- the `GPU / CUDA full release validation` job in `gpu-validation.yml`
- the CUDA T.803 report, `j2k-t803-cuda-linux-x86_64-<sha>`

Everything else is still required for the exact candidate commit. That
includes the push-triggered `full-validation.yml` run with
`Release candidate aggregate`, the `GPU / Metal full release validation` job,
and the three CPU T.803 reports plus the Metal T.803 report.

`scripts/github_actions_verify.py` accepts a waiver only for a version listed
in `CUDA_WAIVER_VERSIONS`. The file must sit at this path and start with this
title. For a tag, the waiver version must also match the tag. Any later
release needs fresh CUDA evidence or its own reviewed waiver.

The trusted GPU runner workflow accepts full mode only with `target=all`. So
the candidate still gets one `gpu-validation.yml` run with `target=all` and
`mode=full`. Once `GPU / Metal full release validation` passes and uploads its
T.803 report, that run is cancelled, so the CUDA full job never waits on the
offline runner. Under this waiver the verifier accepts that cancelled run only
when the CUDA full job is its one cancelled job and every other job succeeded
or was skipped. Without a waiver, a cancelled run is still rejected.

## Why

The self-hosted CUDA runner is the only CUDA machine available to CI. It went
offline on 2026-10-26 at about 01:10 EDT and cannot be recovered remotely. The
maintainer expects it back around 2026-11-02. The maintainer approved
releasing 0.13.0 on the CPU and Metal evidence for the exact commit, plus the
CUDA evidence below.

## CUDA evidence already collected

These checks ran on the CUDA runner, an x86-64 Linux host with a 12 GB NVIDIA
GPU, against train commit `76a0fab3012cce76a5fef333de673f88df4f1c73`
(`chore(release): open 0.13.0`). That commit has the same crate sources,
manifests and lockfile as the 0.13.0 candidate:

| Check | Result |
| --- | --- |
| `cargo xtask release-cuda --mode quick` | 755 passed, 0 failed |
| `j2k-cuda-j2k-engine` tests with `j2k-cuda`, all features | 490 passed |
| CUDA crate doctests | passed |
| Python CUDA wheel tests with `J2K_REQUIRE_CUDA=1` | 22 passed, including all 6 `test_cuda` tests on the device |
| CUDA decode and encode byte identity against `main` | 98 comparisons, all equal |

The identity run covers part of the full matrix. It stopped when the runner
went offline.

The candidate must keep these from `76a0fab3`. Check them before tagging with
`git rev-parse <sha>:<path>`:

| Path | Tree or blob at `76a0fab3` |
| --- | --- |
| `crates` | `8457a8e44bdf8c738442b67c6171470b4f0be2b8`, apart from test-only files |
| `Cargo.toml` | `61b5888630d5d4e0ec09012b03ff69392f08abe2` |
| `Cargo.lock` | `28dac642957f5476aa5df76939e105acee2f520c` |

After `76a0fab3`, the only change under `crates` is the test file
`crates/j2k-ml/tests/cpu.rs`, which adds a CPU test for the mixed batch
accessors. No crate source, manifest, or lockfile changed.
`git diff --stat 76a0fab3 <sha> -- crates` must list only that file.

## Not covered by this waiver

The waiver does not skip the full CUDA release validation or the CUDA T.803
run. It only moves them after publication. When the runner is back, run
`gpu-validation.yml` with `target=cuda` and `mode=full` on the tagged commit.
Then attach the CUDA T.803 `.json` and `.md` reports to the
[0.13.0 GitHub release](https://github.com/frames-sg/j2k/releases/tag/v0.13.0).
Until then, the 0.13.0 CUDA T.803 statement is pending.
