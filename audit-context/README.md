# Maintainability audit snapshot

Audit the source on `audit/maintainability-20261104`, including the repository-wide callers and tests. This branch combines the development work that was absent from public `main` when the audit was requested.

## What is included

| Input | State captured |
| --- | --- |
| Public main | `5f50b5f1637eeda270102807e709d2fb464b82e2`, including PR #320's codeblock-marker tests |
| Local main | `1b0e365d24dcc79a46cd2d4493486345599d4516`, with 47 local-only commits since common ancestor `968dbb4d` |
| Main working changes | 77 modified tracked files and 42 new source/fixture files, including `crates/j2k-c` |
| JPEG/native cleanup queue | All ten patches under [`queue/`](queue/), applied to the source; overlapping API records reconciled for the combined tree |
| Other issue worktrees | Three additional diffs under [`issue-worktrees/`](issue-worktrees/), each against `629dc0303d25a2f439224dfbf5f6ceb6befa2ff0` |

The code under `crates/`, `xtask/`, and `scripts/` is the integrated audit target. The issue-worktree patches are supplementary work at an older base, and may overlap subsequent changes; assess them separately rather than assuming they are applied to this tree. [`source-manifest.json`](source-manifest.json) records input paths, revisions, and hashes.

The supplementary diffs cover Metal sampled-colour batching (#300), native HT zero-bitplane accounting (#304), and JPEG coefficient-domain crop/re-emission (#308).

## Recent changes that matter to the audit

### Native JPEG 2000 CPU paths

- Expensive sample conversion, quantization, rounding, and wavelet loops run inside runtime SIMD dispatch, avoiding per-sample external math calls on baseline x86-64.
- Native and facade conversion paths share rounding, clamping, and bounds. The `no_std` software rounding handles edge values exactly.
- Measured x86 packing blocks remain ahead of shared tails. `cargo xtask x86-codegen-audit` checks for accidental external math calls and out-of-line wide SIMD operations.

### JPEG decoding

- Range-checked 16-bit AVX2 integer inverse transforms, wider fallbacks, sparse-block specializations, and vectorized colour conversion and fancy upsampling.
- Streaming 4:2:0 through one stripe with neighbouring context rows, plus dedicated 4:2:2 and 4:4:4 routes for full images and regions.
- Shared shape classification, restart-marker scanning, and cached restart indexes.
- Row/span entropy checkpoints and device transform-window seeking; checkpoint construction skips outside the cache lock.
- Two-symbol AC Huffman lookup entries with unchanged refill and restart behavior.

### CPU / CUDA-Oxide / Metal math

- `j2k-codec-math::jpeg::idct` owns Rust constants, gain tables, transforms, rounding, and sample conversion.
- CPU scalar decoding and transcode reference paths use it. CUDA-Oxide decode/transcode kernels call the shared math instead of duplicate implementations.
- SIMD implementations share constants and bounds. Metal constants are generated from Rust, while shader templates share full-size, reduced-size, and 12-bit transforms.
- Metal still needs a shader-language implementation; CPU/GPU parity tests cover that boundary.

### Main working-copy additions

- Correct reference-grid geometry for reduced images and regions with nonzero image/tile origins.
- Reference-grid inspection and decode-route planning APIs.
- Additional bounded streaming paths, early row-sink termination, concurrent tile-budget accounting, and CPU batch working-memory limits.
- Consistent lenient metadata recovery and warning reporting, plus backend-neutral Python GPU batch preparation.
- `j2k-c`: C ABI, C++17 wrapper, caller-owned decode buffers, row callbacks, lossless encode, thread control, relocatable native packaging, and consumer tests.

### Cleanup and quality gates

- Explicit simplification passes removed duplicated math, wrappers, copies, repeated work, and unnecessary generic parameters.
- Strict Clippy covers 32 crates; the whole-workspace expansion remains a follow-up involving the new C crate.
- Public and rustdoc-hidden API inventories and semver review records are refreshed. `Box<NamedError>` to `Box<Self>` text changes in the existing CUDA error enums are recorded as equivalent types.
- The review repaired a benchmark-loader regression that dropped requested images under `$HOME/repo/` and CMYK/YCCK inputs.

## Review expectations

Trace ownership, error propagation, cancellation, concurrency, and memory accounting across crate boundaries. Look for duplicate geometry/routing/conversion rules, excessive abstraction, unnecessary public interfaces, GPU resource-lifetime complexity, and drift between FFI contracts, packaging, tests, and documentation.

Every finding should name the source revision and paths, give a concrete case, and propose the smallest effective fix. Treat the earlier cleanup as context, not proof that no further issues remain.

Preserve exact decoded bytes and malformed-input behavior. Some unusual inlining, local-state copies, scalar fallbacks, and specialized loops have measured performance reasons. Further shared-quantizer cleanup was rejected after slowing region decoding by roughly 1–3%; hot-path restructuring needs equivalent-output checks and before/after timing.

Known follow-ups include a clearer restart-index lookup result and measured reuse of the MCU skipper in checkpoint construction and transform-window loops.

## Validation

The earlier isolated main-plus-queue review ran 1,786 passing CPU library tests (seven ignored, zero failures). Its final fixes passed formatting, benchmark regression checks, benchmark Clippy, API inventory checks, and the full semver gate. Earlier handoff reports also contain real-GPU parity and timing evidence.

The combined snapshot was checked separately on an Apple Silicon macOS build host:

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | Pass |
| `cargo test -p j2k-native -p j2k -p j2k-c -p j2k-jpeg --lib --tests --no-fail-fast` | 2,659 passed, 10 ignored, zero failures across 78 test binaries; default features |
| `python3 -m unittest discover -s scripts/tests -p test_ci_plan.py` | 21 passed |
| `cargo xtask stable-api --write` | Both inventories regenerated successfully with the pinned tools |
| `cargo xtask semver --write-report` | Report regenerated; live inventory, exact review records, and full published-package compatibility checks passed |

The generated API files published here match that checked tree. GPU execution, performance benchmarks, installed-package C/C++ consumer tests, and the supplementary issue-worktree diffs were not rerun as part of the combined-snapshot checks.
