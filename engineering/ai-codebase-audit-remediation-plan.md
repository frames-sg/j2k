# J2K codebase audit and remediation register

This file is the living current-state register for repository quality. It is
not a release diary or a task transcript. Completed investigations and prior
release states remain available through Git history.

## Evidence identities

This register keeps two evidence sets separate:

- **Published baseline:** `v0.7.3`, version `0.7.3`, peeled to
  `494eebc3ef20895d331da86221b1d8c4bd4cabf8`. Its results are immutable
  historical release evidence.
- **Current candidate:** the untagged `0.7.4` remediation worktree. Candidate
  results apply only to the exact locally reviewed source-freeze revision and
  do not retroactively change the `0.7.3` baseline or authorize publication.

Any unqualified “candidate” below means `0.7.4`; any unqualified “baseline”
means the published `v0.7.3` tag.

## v0.7.3 verdict — published baseline

The authoritative source base is tag `v0.7.3`, peeled to commit
`494eebc3ef20895d331da86221b1d8c4bd4cabf8`. The preserved pre-0.7.1 local
snapshot at `486b6b1bc3d3c6bc29a686ca6adf73cd945ea621` is provenance evidence only;
its 99 paths must not be re-ported because upstream v0.7.3 already contains
that work and later fixes.

The 2026-07-15 clean-tag baseline is green for formatting, normal and strict
Clippy, workspace tests and doctests, macOS Metal tests, strict repository
lint, unsafe audit, production clone audit, panic surface, typos, dependency
use, `cargo deny`, generated codec math, stable API, and semver review.
Production clone density is 1.96% (5,254 duplicated lines across 1,180 scanned
files); the enforced ceiling is 2.01%.

This verdict supersedes all present-tense findings against the historical
0.7.0 worktree. It does not substitute for final source-freeze, package, or
real NVIDIA evidence.

## Audit definition

“AI slop” is assessed through evidence, not authorship guesses:

- duplicated behavior or verbose near-copies;
- modules with mixed responsibilities or unclear ownership;
- speculative abstractions, pass-through layers, and stale compatibility code;
- unchecked input, arithmetic, allocation, FFI, or accelerator boundaries;
- broad lint suppression, undocumented unsafe code, or fail-open hardware tests;
- stale documentation, policy, generated output, API evidence, or version claims;
- missing negative-path, parity, lifecycle, or integration tests.

File size is a review trigger, not a finding. Classification must separate
production code, inline tests, physical test targets, test-support crates,
fixtures, benches, generated sources, host orchestration, and device/SIMD code.
A split is justified only by responsibility, dependency direction, lifecycle,
or testability.

## Active debt

| Priority | Owner | Current issue | Completion evidence |
|---|---|---|---|
| P1 | Hardware | Linux NVIDIA compile/runtime and `wsi-rs` resident-surface validation are unavailable on the current macOS host | Fail-closed `cargo xtask release-cuda` and patched `wsi-rs` run on a Linux NVIDIA runner |

No item authorizes a public API break, a dependency expansion, a relaxed
security check, or a performance/zero-copy claim without real-device evidence.

## Accepted large-file register

Sizes below are production physical lines at the v0.7.3 baseline unless the
row is explicitly test-only. Reconsideration triggers are mandatory; crossing
one reopens review but does not predetermine a split.

| Path/family | Owner and responsibility | Current size | Why retained | Reconsider when |
|---|---|---:|---|---|
| `crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_encode/simt/src/main.rs` | CUDA HT Tier-1 device state machine | 1,968 | One allocation-free SIMT coding family | New coding mode or core state machine exceeds 450 lines |
| `crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_decode/simt/src/main.rs` | CUDA JPEG baseline device decode | 1,770 | One synchronized 4:2:0/4:2:2/4:4:4 ABI and checkpoint family | Progressive/lossless support, another output family, or 2,000 lines |
| `crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_decode/simt/src/main.rs` | CUDA HT cleanup/refinement device decode | 1,346 | Cohesive device entropy state machine | New refinement route or 1,500 lines |
| `crates/j2k-jpeg/src/backend/neon.rs` | JPEG AArch64 SIMD backend | 1,745 | Benchmark-sensitive kernel family with one backend responsibility | New sampling/output family or 2,000 lines; require benchmark parity before a split |
| `crates/j2k-metal/src/compute/tier1_encode.rs` | Metal classic/HT Tier-1 device orchestration | 1,150 production lines at the prior source-aware inventory | Cohesive two-mode device encode owner | Third coding mode or 1,300 production lines |
| `crates/j2k-transcode/src/accelerator_contracts.rs` | Codec-neutral transcode job/trait contracts | 865 before inline tests | One public contract and default-accelerator family | Third accelerator, job-schema change, or 1,000 production lines |
| `xtask/src/semver.rs` | Stable API capture/review/report workflow | 897 before inline tests | One fail-closed release-evidence workflow | New review schema/report format or 1,100 production lines |
| `xtask/src/perf_guard.rs` | Benchmark snapshot/run/compare workflow | 813 before inline tests | One performance-evidence lifecycle | Snapshot schema v2, another runner, or 900 production lines |
| `crates/j2k-test-support/src/jpeg_fixtures/builders.rs` | Explicit JPEG fixture construction | 3,478 test-support lines | Test data is intentionally explicit and excluded from production metrics | Repeated fixture bug fixes diverge or a reusable format owner emerges |

The focused CUDA registry, classic decode, direct-plan, and `j2k-ml` modules
remain covered by structural line and ownership ratchets.

## Accepted clone register

Production clones of at least 50 lines require consolidation or an entry here.
The entry must identify a real owner and a concrete reconsideration trigger.

| Pattern | Owner | Reason | Reconsider when |
|---|---|---|---|
| CUDA JPEG SIMT sampling loops (53 lines) | CUDA JPEG device decode | Device-mode sequencing and checkpoint exits remain explicit | Shared defect or a fourth sampling mode |
| CUDA HTJ2K 9/7 resident/readback bands (66 lines) | CUDA transcode runtime | Pooled-resident and owned-readback lifecycles differ | Allocation/timing divergence or an existing typed owner covers both |
| CUDA J2K store sample/channel variants (56 lines) | CUDA J2K store runtime | ABI-specific launches follow shared validation | Repeated validation defect or another sample family |
| JPEG/J2K CUDA image-device facades (57 lines) | CUDA codec adapters | Trait symmetry wraps codec-private decoders | Trait changes or a neutral existing request covers both |
| JPEG/J2K CUDA tile-device facades (81 lines) | CUDA codec adapters | Stable trait symmetry wraps codec-private contexts/surfaces | Another backend duplicates it or a neutral public owner emerges |

## Unsafe and suppression policy

`docs/unsafe-audit.md` must enumerate every current unsafe source with its
invariant and regression guard. `static_mut_refs` and shared-prelude `include!`
are permitted only in exact reviewed CUDA-Oxide device paths, backed by unsafe
ledger entries and fail-closed NVIDIA parity. New crate- or file-wide host
suppressions are forbidden. An unavoidable lint uses function-level
`#[expect(..., reason = "...")]` with an actionable reason.

`cargo deny` must pass against the checked-in v0.7.3 lock graph. License
exceptions remain exact and version-scoped. Advisory exceptions retain an
owner and review date and may not broaden. `spin 0.10.0` is forbidden; the
baseline resolves `spin 0.10.1`.

## `j2k-ml` contract

`j2k-ml` remains unpublished, absent from default features, and independent of
the stable `j2k` facade. Preserve fallible single/batch APIs, indexed failures,
strict accelerator requests, and explicit `CpuStaged`, `MetalStaged`, and
`CudaDirect` reporting. Preserve `CudaContext::retain_primary`.

Raw external CUDA allocation/context APIs remain implementation-only and
`#[doc(hidden)]`, with precise lifetime, context, alignment, extent, and
exclusive-mutation safety contracts. CPU parity covers layout, channel
selection, integer dtypes, normalization, ROI/scaling, ordering, empty batches,
corrupt inputs, shape mismatches, and overflow. Metal/CUDA performance and
zero-copy claims require real-device validation.

## `wsi-rs` integration contract

Validate a scratch copy of clean `wsi-rs` v0.5.0. Its real manifest and lock
must remain unchanged. The scratch Cargo patch covers every consumed J2K crate,
the scratch lock resolves only local v0.7.3 J2K packages, and the scratch copy
is removed with `trash` after results are captured.

Required coverage includes raw J2K/J2C, JP2, JPH/HTJ2K, single/batch ordering,
RGB, YCbCr 4:4:4/4:2:2/4:2:0, grayscale, ROI/scaled combinations, TIFF tiles
and associated images, passthrough, malformed inputs, CPU fallback versus
require-device, DICOM HTJ2K frame ordering, CPU/reference parity, Metal
resident ownership/readback/no-fallback, and CUDA classic/HT resident behavior.
Unavailable corpus or NVIDIA evidence is a reported gap, never an inferred
pass.

## Verification matrix

| Gate | Published `v0.7.3` baseline evidence | Current `0.7.4` candidate evidence |
|---|---|---|
| `cargo xtask fmt` | pass | pass |
| `cargo xtask clippy` | pass | pass |
| `cargo xtask clippy-strict` | pass | pass |
| `cargo xtask test` including macOS Metal tests | pass | pass |
| `cargo xtask doc` | covered by baseline test doctests; standalone pending | pass |
| `cargo xtask repo-lint --strict` | pass, 423 checks | pass, 428 checks |
| `cargo xtask unsafe-audit` | pass | pass |
| `cargo xtask clone-audit` | pass, 1.96% | pass, 1.96% at 2.01% ceiling |
| test/support clone audit | not yet implemented | pass, 4.09% at 4.14% ceiling |
| `cargo xtask panic-surface` | pass | pass |
| `cargo xtask typos` | pass | pass |
| `cargo xtask machete` | pass | pass |
| `cargo xtask deny` | pass with reviewed warnings | pass with reviewed warnings |
| `cargo xtask codec-math-codegen` | pass | pass |
| `cargo xtask stable-api` | pass | pass; no API drift |
| `cargo xtask semver` | pass | pass |
| changed-line coverage | not rerun for baseline | pass, 81.19% (246/303); critical paths 100% (4/4) |
| `cargo xtask release-metal` | baseline behavior covered by normal Metal tests only | pass, fail-closed release lane |
| `cargo xtask release-cuda` | unavailable on macOS | required on Linux NVIDIA |
| patched-source `wsi-rs` CPU/Metal | clean 0.5.0 validation passed before patching | pass; all 14 J2K packages resolved to local v0.7.3, full validation and real Metal tests passed |
| patched-source `wsi-rs` CUDA | unavailable on macOS | required on Linux NVIDIA |
| public corpus parity | corpus availability unknown | pass; J2K/reference, DICOM/OpenSlide, and five real-WSI behavior tests |

## Living-document rule

Keep this file below 500 lines and limited to the current verdict, active debt,
accepted-large and accepted-clone registers, invariants, and verification
matrix. Do not append completed task histories, command transcripts, release
diaries, private host details, or obsolete version narratives. Git history is
the archive.
