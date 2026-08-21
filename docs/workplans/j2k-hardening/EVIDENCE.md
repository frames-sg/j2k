# Durable Evidence

Plan anchor: J2K-HARDENING-2026-08-18
Audit baseline: f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5

## Baseline Reconciliation

- Repository: `.` (repository root); branch `main`.
- Current HEAD: `f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5`.
- Current HEAD exactly equals the audit baseline. `git diff --stat
  f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5..HEAD` is empty, so no
  commit-level finding needed a forward-port reconciliation.
- Five native Tier-1/MQ files were already modified when G0 began. G0 did not edit
  them: `arithmetic_decoder.rs`, `bitplane/arithmetic.rs`, `bitplane/state.rs`,
  `bitplane/tests.rs`, and `mq.rs` under `crates/j2k-native/src/j2c/`.
- The dirty files change packed MQ state handling and zero-coding behavior and add
  tests. Baseline test and benchmark results therefore describe the audit commit
  plus those preserved working-tree changes.
- No repository-local or nested `AGENTS.md` exists. The user-supplied global
  engineering instructions apply.

## Environment

| Item | Baseline value |
|---|---|
| Date / timezone | 2026-08-20 / America/New_York |
| Host | macOS 26.5.2 (25F84), Darwin 25.5.0, arm64 |
| CPU / GPU | Apple M4 Pro / Apple M4 Pro, 16 GPU cores, Metal 4 |
| Memory | 51,539,607,552 bytes (48 GiB) |
| Rust | rustc 1.96.0 (ac68faa20), LLVM 22.1.2; cargo 1.96.0 |
| Toolchain | pinned `1.96-aarch64-apple-darwin` |
| Apple tools | Xcode 26.6 (17F113); Apple Metal 32023.883 |
| External codecs | OpenJPEG and Grok binaries in `/opt/homebrew/bin` |
| Clone tooling | Node 22.14.0; jscpd 4.0.5 |
| CUDA | unavailable: macOS host, no NVIDIA GPU, `nvidia-smi`, or `nvcc` |

CUDA runtime tests and CUDA benchmark execution are unavailable due to missing
hardware and toolchain; this is not recorded as a pass. CUDA-Oxide build lanes
report their Linux-only targets as skipped.

## Workspace and Dependency Inventory

Final-tree Cargo metadata reports 29 workspace packages: 23 publishable and 6
private.

Publishable packages:

```text
j2k j2k-cli j2k-codec-math j2k-core j2k-cuda j2k-cuda-build-support
j2k-cuda-j2k-engine j2k-cuda-jpeg-engine j2k-cuda-runtime
j2k-cuda-transcode-engine j2k-jpeg j2k-jpeg-cuda j2k-jpeg-metal
j2k-metal j2k-metal-support j2k-ml j2k-native j2k-profile j2k-tilecodec
j2k-transcode j2k-transcode-cuda j2k-transcode-metal j2k-types
```

Private packages:

```text
j2k-alloc-probe j2k-compare j2k-t803 j2k-test-support
j2k-transcode-test-support xtask
```

Across normal, build, and development workspace edges, current fan-in leaders
are `j2k-core` (19), `j2k-test-support` (19), `j2k-native` (14),
`j2k-codec-math` (13), and `j2k-profile` (10). Fan-out leaders are
`j2k-transcode-metal` (10), `j2k-t803` (9), `j2k-metal` (8),
`j2k-transcode` (8), and `j2k-transcode-cuda` (8). Counts are direct workspace
dependency edges from `cargo metadata`, not transitive edges.

The material forbidden production edge identified by the audit is removed:
`j2k-transcode-metal` consumes `j2k-metal-support` and neutral codestream
contracts, not the public `j2k-metal` adapter.

Material feature surfaces:

- `j2k-native` defaults to `std,simd,parallel`.
- `j2k-cuda` exposes `cuda-runtime` and `cuda-profiling`.
- `j2k-cuda-runtime` defaults to no features and exposes only low-level
  CUDA-Oxide copy/runtime and profiling feature flags; codec projects belong to
  the three engine crates.
- `j2k-ml` exposes `cpu`, `cuda`, and `metal`, with an empty default.
- `j2k-transcode-metal` exposes `bench-internals`.

## Audit-Finding Reconciliation

| Finding | Current status at baseline HEAD | Evidence |
|---|---|---|
| Prepared-plan type erasure | still present | `crates/j2k/src/owned_batch/prepared_plan.rs` imports `core::any::Any` and owns `adapter_view`; CUDA and Metal adapters downcast referenced native plans. |
| Duplicate host phase budgets | still present | Four definitions exist in `j2k-cuda`, `j2k-transcode-cuda`, `j2k-cuda-runtime`, and `j2k-jpeg-cuda`. |
| Duplicate encode geometry | still present | `j2k-metal/src/encode/plan.rs` owns `lossless_device_encode_levels` and a local minimum-DWT constant while the facade owns parallel policy. |
| Semantic graph enforcement | still absent | `xtask/tests/repo_lint_support/architecture_policy.rs` only checks that documented edges match Cargo metadata. |
| Shader clone coverage | still absent | clone audit stages production/test Rust and jscpd is configured for Rust; Metal is not scanned. |
| Hidden public boundary bypasses | still present | `#[doc(hidden)] pub` APIs are concentrated in core/runtime/adapters, especially `j2k-cuda-runtime` codec types and operations. |
| Handwritten routing evidence | still present | route eligibility thresholds remain constants in Metal/CUDA routing code despite an xtask evidence generator. |
| Transcode-to-full-adapter edge | still present | `j2k-transcode-metal -> j2k-metal`. |
| God files / long functions | still present | inventory below; many production `clippy::too_many_lines` allowances remain. |
| Missing stage benchmarks | still present | Metal decode-stage and CUDA transcode DWT97 targets do not exist. |

No finding was partially or fully resolved at commit level because current HEAD is
the audited commit. The preserved dirty native changes do not address these
architecture findings.

## Existing Guardrails and Routes

- The architecture test compares `docs/architecture.md` with workspace dependency
  metadata but has no forbidden-edge policy.
- Clone audit stages 1,505 production Rust sources and 1,151 test Rust sources.
  It does not currently cover Metal shader sources.
- Route evidence generation exists under `xtask/src/auto_routing*`; production
  route constants are nevertheless handwritten.
- Numerous `#[doc(hidden)] pub` items and `clippy::too_many_lines` allowances form
  a baseline inventory rather than a reviewed boundary/algorithm allowlist.

## Benchmark Target Inventory

- `j2k`: `public_api`.
- `j2k-cuda`: `auto_routing`, `encode_stages`, `htj2k_decode`, `htj2k_encode`.
- `j2k-jpeg`: `compare`, `corpus_report`, `decode_cpu`, `encode_cpu`,
  `fast420_breakdown`, `micro`.
- `j2k-jpeg-cuda`: `device_decode`.
- `j2k-jpeg-metal`: `compare`, `device_upload`, `encode_baseline`.
- `j2k-metal`: `auto_routing`.
- `j2k-ml`: `batch_decode`, `batch_decode_cuda`, `batch_decode_metal`.
- `j2k-native`: `direct_cpu`, `htj2k_sigprop_phase`, `tier1_bitplane`.
- `j2k-tilecodec`: `compare`; `j2k-transcode`: `dct53`;
  `j2k-transcode-metal`: `dct97`.

The audited `j2k-metal/decode_stages` and `j2k-transcode-cuda/dwt97` harnesses are
missing. `cargo xtask bench-build --lane metal` successfully compiled the current
Metal benchmark set.

## File and Function Inventory

Largest production Rust files by physical lines (test/tool crates and test
directories excluded; generated/table-like data called out):

| Lines | File |
|---:|---|
| 1,645 | `crates/j2k-jpeg/src/backend/neon.rs` |
| 1,249 | `crates/j2k-metal/src/engine/tier1_encode.rs` |
| 1,183 | `crates/j2k-types/src/lib.rs` |
| 1,178 | `crates/j2k-transcode/src/accelerator_contracts.rs` |
| 1,144 | `crates/j2k-cuda/src/encode/stage.rs` |
| 1,116 | `crates/j2k-metal/src/engine/abi.rs` |
| 1,088 | `crates/j2k-native/src/color.rs` |
| 1,042 | `crates/j2k-transcode/src/dct97_2d.rs` |
| 1,029 | `crates/j2k-native/src/math.rs` |
| 1,006 | `crates/j2k-metal/src/classic.rs` |
| 990 | `crates/j2k-native/src/j2c/ht_encode_tables.rs` (table/generated-like) |
| 984 | `crates/j2k-native/src/j2c/codestream_write.rs` |
| 961 | `crates/j2k-native/src/j2c/decode.rs` |

Largest Metal sources:

| Lines | File |
|---:|---|
| 2,022 | `j2k-metal/.../encode_bitstream_classic_symbol_plan.metal` |
| 1,913 | `j2k-metal/.../packetize.metal` |
| 1,812 | `j2k-metal/.../classic.metal` |
| 1,810 | `j2k-jpeg-metal/.../shaders_encode.metal` |
| 1,635 | `j2k-metal/.../classic_core.metal` |
| 1,371 | `j2k-metal/.../ht_cleanup.metal` |

Largest CUDA-Oxide Rust sources:

| Lines | Area |
|---:|---|
| 1,968 | HTJ2K encode SIMT main |
| 1,770 | JPEG decode main |
| 1,397 | HTJ2K decode main |
| 936 | classic decode |
| 918 | IDWT |
| 740 | JPEG encode |
| 605 | transcode exports |

Function-count leaders include JPEG Metal `fast_packets/pipelines.rs` (77),
`descriptors.rs` (71), transcode `accelerator_contracts.rs` (65), native
`math.rs` (64), core `traits.rs` (57), JPEG NEON (55), and CUDA encode stage
(54). This is a structural trigger, not a conclusion that each file must split.

Name-frequency analysis found broad generic repetition (`new`, `fmt`, `drop`,
`default`) and policy-like repetition including `retained_allocation_bytes` (15),
`live_bytes` (15), `with_cap` (13), `try_vec_with_capacity` (13), `try_vec` (13),
and `try_vec_filled` (10). These counts are discovery heuristics, not proof of
semantic duplication. Crate roots with operational logic and explicit long-line
allowances require reviewed G1 allowlists.

## Baseline Correctness and Static Validation

| Command | Result | Baseline interpretation |
|---|---|---|
| `cargo xtask fmt` | pass | Canonical format check. |
| `cargo xtask codec-math-codegen` | pass | Generated codec math is current. |
| `cargo xtask clippy-strict` | pass | Strict library lint lane passes. |
| `cargo xtask test` | pass | Canonical host/workspace tests, Metal debug tests, doctests, allocation probe, downstream examples, and external OpenJPEG/Grok parity pass. |
| `cargo xtask unsafe-audit` | pass | Canonical unsafe audit. |
| `cargo xtask release-integrity` | pass | Release-integrity check. |
| `cargo xtask clone-audit` | pass | Production: 197 clones / 5,444 duplicated lines / 1.73%; tests: 201 / 5,632 / 2.72%. |
| `cargo xtask doc` | pass | Documentation build/check. |
| `cargo xtask bench-build --lane metal` | pass | Current Metal benchmark targets compile. |
| `cargo xtask metal-compile` | pass | Both clippy phases and optimized all-feature Metal library/integration/doc tests pass. |
| `cargo xtask repo-lint` | fail (pre-existing) | 50 checks pass; environment-variable policy fails because the WSI SVS corpus-path variable is used in two files but absent from `docs/env-vars.md`. |
| `cargo xtask clippy` | fail (pre-existing/dirty) | Library phase passes; all-target phase reports `similar_names` and truncating casts in dirty `mq.rs`, an unchanged cast in `math.rs`, and unchanged float-equality assertions in native tests. |
| `cargo xtask panic-surface` | fail (pre-existing) | `clippy::expect_used` reports 87 against baseline 50. |
| `cargo xtask stable-api` | fail (pre-existing) | Public and implementation snapshots are stale. |
| `cargo xtask semver` | fail (pre-existing) | Stale ordinary package `j2k-codec-math` and hidden package `j2k-native`; points to `stable-api --write`. |
| `cargo xtask package` | unavailable on current worktree | Canonical command refuses any dirty tree. Five user changes plus workplan docs were present. |

No listed baseline failure was introduced by G0 documentation. `git diff --check`
passes after the checkpoint edits.

## Conformance

OpenHTJ2K v0.19.0 reference source, commit/archive pinned by the repository, was
prepared with `scripts/prepare-openhtj2k-reference.sh`. Both runs used suite
`all` and `--development`; a dirty worktree prevents treating them as release
evidence.

| IUT | Result | Decoder | Encoder | Routes | Artifacts / SHA-256 |
|---|---|---:|---:|---|---|
| CPU | pass | 160/160 | 56/56 | CPU 160 | `target/t803/reports/cpu-development.{json,md}`; JSON `783142ba...f959059`, MD `4f3a0b58...0c5691f` |
| Metal | pass | 160/160 | 35/35 applicable | hybrid 81, CPU 79, device-native 0 | `target/t803/reports/metal-development.{json,md}`; JSON `1f767515...ae9ef5a`, MD `68ab3230...72ff7` |

The Metal report proves exercised hybrid routing but no device-native decoder
case. CUDA T.803 is unavailable on this host.

## Baseline Performance

Criterion artifacts are under `target/criterion/`. Results below are measurements
on the recorded M4 Pro host, not performance claims against another commit.

| Experiment | Result | Representative measurements / limitation |
|---|---|---|
| Metal auto-routing | unavailable | `J2K_METAL_PROFILE_STAGES=summary cargo bench -p j2k-metal --bench auto_routing` compiled, then fail-closed because `J2K_AUTO_ROUTING_MANIFEST` and its external evidence/corpus configuration were not supplied. |
| Metal transcode DCT97 | partial, matrix defect | DCT97 CPU/Metal: 224 px 312.28 us / 1.4628 ms; 512 px 1.6682 / 6.0416 ms; 1024 px 7.7855 / 24.282 ms; 2048 px 43.311 / 94.531 ms. Full run failed at `p3_like_ybr444_224_batch_512`: requested 537,933,440 bytes exceeds 536,870,912-byte cap. |
| Metal tile transcode batch | partial | 224 px batch128: Rayon 150.30 ms, Metal auto 138.26 ms, explicit 134.23 ms. Batch256: Rayon 303.07 ms, Metal auto 262.42 ms, explicit 262.84 ms. Criterion found no significant change from its stored baseline. |
| JPEG Metal compare | partial, matrix defect | Planning: fast420 306.73 us, restart 318.97 us, fast422 397.97 us, fast444 557.02 us. Generated 256 decode CPU/Metal: fast420 315.09 us / 8.3757 ms; fast422 379.48 us / 8.3339 ms; fast444 500.44 us / 8.3523 ms. WSI restart420 batch: CPU 18.975 ms, Metal 9.2266 ms, auto 9.2778 ms. Run failed at texture batch64 because a 1,151,597,618-byte owner/metadata allocation exceeds the 512 MiB cap. |
| JPEG Metal encode baseline | pass | 512 RGB8 4:2:2 single: CPU 14.285-14.382 ms (52.283 MiB/s point), Metal 550.27-550.69 ms (1.3624 MiB/s). Batch8: CPU 115.96-116.49 ms (51.632 MiB/s), Metal 777.07-777.35 ms (7.7199 MiB/s). |
| CUDA benchmark set | unavailable | Targets exist; execution and meaningful CUDA compilation are unavailable without Linux/NVIDIA hardware/toolchain. |

The two partial Metal runs reveal benchmark-matrix correctness defects: declared
cells exceed the benchmark's own memory cap. Their successful rows remain useful
baseline samples, but neither matrix is decision-grade complete. Stage summaries
were enabled for transcode; no kernel register/private-memory statistics were
available in G0.

## G1 Guardrail Evidence

### G1.1 Forbidden dependency edges

`xtask/tests/repo_lint_support/architecture_policy.rs` now classifies and rejects:

- support or runtime crates depending on public codec adapters;
- transcode adapters depending on full CUDA/Metal codec adapters;
- publishable production crates depending on test-support crates.

Pure rule tests cover forbidden and allowed directions. The live graph is
ratcheted to exactly one reviewed migration violation:
`j2k-transcode-metal -> j2k-metal`. New violations fail; removing that edge
requires removing its inventory entry.

### G1.2 Prepared-plan type erasure

`prepared_plan_policy.rs` scans facade, CUDA-adapter, and Metal-adapter Rust
sources and ratchets occurrences of `core::any::Any`, `adapter_view`, and native
referenced-plan downcasts by file and count. The current seams remain violations
to migrate, not accepted architecture; additions fail and removals make the
allowlist stale.

### G1.3 Phase budgets

`allocation_policy.rs` scans every crate Rust source for struct, enum, or type
definitions named `HostPhaseBudget`. Exactly four backend-local struct definitions
are inventoried. A fifth definition fails; A3 must shrink the inventory to the
single shared owner.

### G1.4 Structural size triggers

`source_size_policy.rs` distinguishes publishable production crates from private
test/tool crates and excludes physical test-only source modules. It enforces:

- 400-line soft ceiling for crate roots, with five reviewed current ceilings;
- 1,200-line hard ceiling for ordinary Rust modules, with five reviewed current
  ceilings;
- 1,500-line hard ceiling for Metal shaders, with five reviewed current ceilings;
- 75-line soft ceiling for root free functions, with one reviewed 127-line
  operational function;
- a ceiling of 98 production `clippy::too_many_lines` expectations.

Every allowance includes a rationale and becomes a failure if stale or exceeded.

### G1.5 Clone coverage

The production Rust lane now uses 12 lines / 50 tokens rather than 20 / 50, so it
captures smaller repeated orchestration. CUDA-Oxide Rust remains included through
the source-aware production stage. A separate Metal lane stages all 28 `.metal`
sources and explicitly maps them to jscpd's OpenCL tokenizer.

| Lane | Staged / analyzed | Clones | Duplicated lines | Percentage | Fail threshold | Report SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| production Rust | 1,505 / 1,478 | 747 | 13,559 | 4.31% | 4.32% | `7a7cfdf4...81f643a` |
| Metal shaders | 28 / 26 | 236 | 4,976 | 25.08% | 25.09% | `1a7f7c09...bc2e335` |
| test Rust | 1,151 / 1,126 | 201 | 5,632 | 2.72% | 3.99% | `78ba9949...3e730f6` |

Reports are under `target/clone-audit/{report,metal-report,test-report}/`. The
three-lane `cargo xtask clone-audit` passes. The high Metal baseline is an
inventory and regression threshold, not evidence that duplication is acceptable.
Targeted route-threshold and error-taxonomy inventories remain G1.5 work because
lexical clone detection alone cannot establish shared policy ownership.

G1.5 subsequently added AST/text ownership ratchets for all 16 current
handwritten `AUTO_*_MIN*` production thresholds and duplicated error-variant
declarations (`HostAllocationFailed` 22, `HostAllocationTooLarge` 5,
`UnsupportedCudaRequest` 2, `UnsupportedMetalRequest` 2). New declarations fail;
later shared-policy migrations must lower the inventories.

### G1.6 Stage benchmark harnesses

- `j2k-metal/benches/decode_stages.rs` uses a reusable production prepared batch,
  reports actual `MetalDecodeDispatchReport` counters, and measures resident and
  explicit-readback end-to-end paths. On the M4 Pro, the 16-image 512x512 HTJ2K
  RGB8 batch reported Tier-1 3, dequantization 3, IDWT 9, inverse MCT 1, final
  store 1, and host-to-device 3 dispatches. Criterion 95% intervals were
  7.2883-7.3783 ms resident and 7.9607-8.0043 ms with readback.
- `j2k-transcode-cuda/benches/dwt97.rs` exposes fused and unfused column-lift plus
  quantization runs through the production kill switch, reports pack/upload,
  column-lift, quantization, and readback timings, and separates production
  kernel-timing projection from end-to-end timing. It compiles with
  `--features cuda-runtime`; execution is unavailable on this macOS/non-NVIDIA
  host because CUDA-Oxide correctly skips its Linux-only builds.

### G1.7 Experiment switches

Repository policy now requires the active CUDA DWT97 fusion experiment to retain
its documented production baseline switch, real fused/unfused benchmark labels,
and production selection path. G1 is complete; the baseline repo-lint failure is
unchanged and unrelated.

## A1 Shared Encode Geometry Evidence

`crates/j2k-types/src/encode_geometry.rs` is now the single owner of maximum
legal decomposition levels, default and progression-sensitive lossless policy,
explicit maximum handling, per-level low/high dimensions, Part 1 code-block
exponent validation, reversible total bitplanes, and packet ordering. The
required dimension, progression, maximum-level, and component-count matrices
are covered directly, including zero/asymmetric/u32 boundary checks.

The facade's established rule is now explicit: an override can select more than
the default number of levels, capped by legal geometry, but cannot force a
decomposition when the shorter axis is below 64. Metal previously allowed that
below-64 override; it now consumes the shared policy. Selected Metal plan cases
compare directly with the facade for all five progression orders. A CPU
integration test verifies that 63x128 with `Some(255)` writes zero levels in COD,
128x128 with `Some(5)` writes five, and both codestreams round trip.

Consumer migration covered:

- facade lossless/default policy and the matching lossy default calculation;
- native legal-level caps, packed DWT geometry, precomputed geometry validation,
  and code-block exponent validation;
- Metal resident level policy, DWT plans, code-block conversion, reversible
  bitplanes, and existing packet progression mapping;
- CUDA-runtime forward-DWT validation;
- semantically identical Metal-transcode one-level DWT shapes;
- the existing codec-math public maximum-level function via compatibility
  re-export rather than duplicate implementation.

Packet ordering moved from the `j2k-types` root into the shared module while its
existing root function path remains re-exported. This reduced the root from
1,183 to 1,125 lines. `encode_geometry_policy.rs` enforces the owner and required
consumers and rejects the old facade/Metal/GPU-local policy symbols.

### A1 validation

| Command | Result |
|---|---|
| `cargo test -p j2k-types` | pass, 18/18 |
| `cargo test -p j2k --test encode_lossless` | pass, 68/68 plus one fixture-gated ignore |
| Native FDWT, precomputed, and code-block focused tests | pass |
| CUDA-runtime DWT focused tests | pass, 13/13 |
| Metal plan parity tests | pass, 2/2 |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal encode::tests::kernels -- --nocapture` | pass, 19/19 on Apple M4 Pro |
| `cargo test -p j2k-transcode-metal` | pass |
| A1 ownership, source-size, and architecture policy tests | pass |
| `cargo xtask clippy-strict` | pass |
| Changed production libraries clippy with `-D warnings` | pass |
| `cargo xtask test` | pass after updating the G1 Metal benchmark inventory test to include `decode_stages` |
| `cargo xtask clone-audit` | pass: production 4.31%, Metal 25.08%, tests 2.72% |
| `cargo xtask repo-lint` | 68/69 pass; only the unchanged `J2K_WSI_SVS_PATH` documentation failure remains |

CPU and Metal `--suite all --development` T.803 runs both pass after A1. CPU is
160/160 decoder and 56/56 encoder; Metal is 160/160 decoder, 35/35 applicable
encoder, 81 hybrid and 79 CPU routes. The four report SHA-256 values are exactly
unchanged from G0 (`783142ba...f959059`, `4f3a0b58...0c5691f`,
`1f767515...ae9ef5a`, `68ab3230...72ff7`), providing byte-identical conformance
evidence across the migration. CUDA execution remains unavailable on this host.

## A2 Typed Prepared-Plan Evidence

The public Classic and HT prepared-plan wrappers now expose typed immutable
`geometry()` borrows under facade-owned type names. Their `Arc` storage remains
private, so backends cannot replace or mutate the geometry. Integration tests
clone each wrapper and assert that both typed borrows have the same address;
the existing Classic/HT payload reconstruction tests show that encoded bytes
remain ranges into the original `Arc<[u8]>` rather than plan-owned copies.

All prepared-plan consumers in CUDA and Metal, including persistent Metal plan
caches, now use direct typed borrows. The A2 completion search for
`adapter_view`, prepared-plan downcasts, and `core::any::Any` returns no results
under `crates/j2k`, `crates/j2k-cuda`, or `crates/j2k-metal`. The repo-lint
policy now requires an empty inventory.

Validation results:

| Command | Result |
|---|---|
| `cargo test -p j2k --test owned_batch --no-fail-fast` | pass, 34/34 |
| `cargo test -p j2k-cuda --all-features --no-fail-fast` | pass across all unit, integration, and doc targets |
| `cargo test -p xtask --test repo_lint repo_lint_support::prepared_plan_policy:: -- --nocapture` | pass, 2/2 |
| `cargo clippy -p j2k -p j2k-cuda -p j2k-metal --all-features --lib --no-deps -- -D warnings` | pass |
| A2 completion `rg` search | empty |

`J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features
--no-fail-fast` passed the 380 library tests (22 ignored), all prepared-plan,
cache, Classic color, RGBA, benchmark-inventory, shader, and doc targets, but
the `device` target finished 101/102. Its sole failure was
`independent_openht_sigprop_overlap_matches_openht_oracle_within_one_lsb`, in
the pre-existing user-modified HT refinement/Tier-1 arithmetic surface. A2 did
not edit those files or the entropy implementation; this remains an explicit
unverified full-suite risk rather than an A2 plan-contract failure.

## A3 Shared Host Phase-Budget Evidence

`j2k-core::HostPhaseBudget` is now the single implementation for J2K CUDA,
JPEG CUDA, CUDA runtime, and CUDA transcode. Its neutral `HostPhaseError`
distinguishes allocation failure from phase-limit failure and retains requested
bytes, cap, and static operation context. Adapter `From` implementations retain
their existing error variants and downstream stage classification.

The core tests cover exact cap, one byte over, saturated overflow, actual
allocator capacity, failure without accounting mutation, incremental growth,
and zero-sized types. Existing adapter tests cover CUDA runtime and transcode
classification and exact-cap behavior. The repository search finds only the
definition in `crates/j2k-core/src/host_allocation.rs`; the repo-lint policy pins
that single owner.

| Command | Result |
|---|---|
| Cross-crate all-feature allocation-focused library tests | pass: core 15, J2K CUDA 13, CUDA runtime 24, JPEG CUDA 8, CUDA transcode 5 |
| `cargo test -p xtask --test repo_lint repo_lint_support::allocation_policy:: -- --nocapture` | pass, 2/2 |
| Five affected all-feature library clippy targets with `-D warnings` | pass |
| `rg` for HostPhaseBudget declarations | one shared struct in `j2k-core` |

## A4 Unified Decode-Operation Evidence

CUDA now normalizes every image request to `DeviceDecodeRequest`, validates it
once with `DeviceDecodePlan`, selects CUDA/Auto/CPU once, and executes the
matching backend from `decode_op_to_surface_impl`. Its tile trait boundary has
one analogous operation function, and CPU-staged uploads reuse a single CPU
operation executor. A source-structure regression rejects a return to four
image entrypoints.

Metal retains its public `MetalDecodeRequest` API but now performs planning,
scaled Auto selection, backend validation, and CPU/Metal dispatch in one
`decode_op_to_surface_impl`. Batch and direct paths construct that same request
instead of calling geometry-specific route methods.

| Command | Result |
|---|---|
| `cargo test -p j2k-cuda --test host_surface --no-fail-fast` | pass, 37/37 |
| CUDA operation structure regression | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal decoder::tests -- --nocapture` | pass, 14/14 |
| CUDA and Metal all-feature library clippy with `-D warnings` | pass |
| Search for old geometry-specific `*_surface_impl` entrypoints | empty |

## A5 Unified JPEG Metal Batch-Plan Evidence

`Rgb8MetalBatchSource::Bytes` and `Rgb8MetalBatchSource::Decoders` now perform
only source-specific resolution. Both feed `ResolvedRgb8BatchSource` into the
single `build_rgb8_batch_plan` loop, which owns vector budgeting, output-shape
and sampling consistency, restart restrictions, plan-owner admission, retained
cache-byte preflight, insertion, and execution baseline stamping. Prepared
decoder owners are identity-deduplicated before entering the shared context.

The parity matrix compares normalized request keys and output dimensions for
full, quarter-scaled, and region-half-scaled operations. It also proves that raw
and prepared sources reject mismatched dimensions, mixed 4:2:0/4:4:4 sampling,
and restart-coded full-tile 4:2:2 inputs identically while accepting the scaled
restart shape. A focused cap test proves cache-retained bytes participate in the
shared builder's admission calculation, and a source regression pins the one
builder boundary.

| Command | Result |
|---|---|
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast` | pass: 228 library tests, 37 integration tests, and doc tests |
| `cargo clippy -p j2k-jpeg-metal --all-features --lib --no-deps -- -D warnings` | pass |
| `cargo fmt --all -- --check` | pass |
| `git diff --check` | pass |

## C1.1 Low-Level CUDA Kernel Interface Evidence

The first C1 seam removes the need for future engine crates to extend the
runtime's closed codec-kernel enum. A validated `CudaKernelSpec` carries static
PTX and entrypoint ownership into a generic cache key. Checked launch geometry,
the documented unsafe parameter-layout marker, and the synchronous launch
primitive form the low-level execution boundary. Runtime still contains the
legacy codec modules and keys, so this is migration evidence, not C1 completion.

Cache equality hashes the small module/entrypoint identity and static PTX
allocation identity rather than the full PTX bytes. Constructor validation
rejects empty IDs, unterminated PTX, and empty, unterminated, or internally-NUL
entrypoint names before any Driver API operation.

| Command | Result |
|---|---|
| External low-level API test before implementation | failed as expected: kernel spec and geometry exports absent |
| `cargo test -p j2k-cuda-runtime --test low_level_kernel_api --no-default-features -- --nocapture` | pass: 3/3 |
| No-feature and all-feature runtime library clippy with `-D warnings` | pass |
| `cargo check -p j2k-cuda-runtime --all-features --all-targets` | pass; expected Linux-only kernel-build skips on macOS |
| `cargo test -p j2k-cuda-runtime --all-features --no-fail-fast --quiet` | pass: 333 library plus 3 external tests |
| `git diff --check` | pass |

## M8 Classic Metal Shader-Ownership Evidence

`classic.metal` was mechanically partitioned at declaration boundaries. A
pre-deletion `cmp` proved that concatenating the nine units in composer order
matched the original file. The retained integrity test independently ratchets
the original 77,178-byte length and FNV-1a value 4,431,697,704,945,636,949.
Thus table values, arithmetic, declaration order, and kernel bodies are exact.

QE and context tables contain no kernels. ABI and constants are independent
units; state helpers, entropy primitives, pass/job logic, and public decode
kernels have focused files. `encode_kernels.metal` was deliberately not added:
classic encode was not in the deleted source and already belongs to
`encode_bitstream_classic_kernels.metal`.

| Command | Result |
|---|---|
| M8 classic-shader regression before extraction | failed as expected: `classic/abi.metal` absent |
| Mechanical concatenation `cmp` before deleting the monolith | pass: exact bytes |
| `cargo test -p xtask --test repo_lint source_size_policy -- --nocapture` | pass: 11/11 |
| `cargo test -p j2k-metal --all-features --test shader_integrity -- --nocapture` | pass: 4/4 |
| `cargo check -p j2k-metal --all-features --all-targets` | pass |
| `cargo clippy -p j2k-metal --all-features --lib --no-deps -- -D warnings` | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features --lib --quiet -- --test-threads=1` | pass: 380 executed, 22 ignored |

## R1 Routing and Promotion-Codegen Evidence

CUDA and J2K Metal routing now own separate eligibility, availability,
promotion, decision, typed-rejection, and telemetry modules. Validated
artifact IDs and exact promoted workload cells live in one checked-in manifest;
production tables are generated and carry their source identifier. The
generator rejects unsupported schemas, malformed hashes, incomplete
six-operation matrices, cross-backend sources, invalid boundaries, and
duplicate workload identities.

Unverified JPEG Metal batch and J2K Metal region-scaled batch Auto routes were
removed. Their explicit Metal paths remain available. The Metal resident
host-output policy now matches only RGB8 1024×1024 and Gray8/RGB8 2048×2048,
which corrects the prior unsupported Gray8 512×512 extrapolation.

| Command | Result |
|---|---|
| unqualified-route regression tests before production repair | failed as expected for JPEG Auto batch, J2K Metal region-scaled Auto batch, and Gray8 512×512 host output |
| `cargo test -p xtask --bin xtask promotion_codegen` | pass: 4/4 |
| `cargo xtask promotion-codegen --check` | pass |
| `cargo test -p xtask --test repo_lint routing_policy` | pass: 3/3 |
| `cargo test -p j2k-cuda --all-features --lib --quiet` | pass: 164/164 |
| `cargo test -p j2k-metal --all-features --lib --quiet` | pass: 377 executed, 22 ignored |
| `cargo test -p j2k-jpeg-metal --all-features --lib --quiet` | pass: 228/228 |
| xtask and affected-adapter strict clippy | pass |
| full repository policy | 86/93 pass; seven stale path/inventory policies from completed earlier decomposition phases remain open for plan-wide cleanup |

## P0 Performance Experiment Framework Evidence

The new experiment record turns the plan's measurement checklist into a
fail-closed input. It preserves missing vendor metrics as explicit `null`
values, but does not permit a measured decision without baseline/treatment
rows, a known workload, output hashes, exact parity, and conformance status.
Promotion additionally enforces the repository's confidence, regression, and
complexity requirements.

| Command | Result |
|---|---|
| P0 validator regressions before implementation | failed as expected: malformed corpus hash, missing treatment, parity failure, and unsupported promotion were accepted by the stub |
| `cargo test -p xtask --bin xtask gpu_experiment::tests` | pass: 4/4 |
| `cargo test -p xtask --test repo_lint performance_experiment_records_are_validator_owned` | pass: 1/1 |
| `cargo clippy -p xtask --all-targets --no-deps -- -D warnings` | pass |

## P1 Metal Reversible IDWT53 Fusion Evidence

The temporary ordinary/repeated prototype combined interleave and horizontal
lifting in one row-owned kernel. Before timing, fused and unfused output bits
matched for widths and heights 1, 2, 3, 31, 32, 511, 512, 513, 1023, 1024,
1025, and 2592, alternating even/odd origins. Native multi-level and repeated
batch parity also passed on the required Metal runtime.

Criterion used the deterministic reversible HTJ2K RGB8 512×512 batch-16
decode-stage fixture, ten samples, three-second warm-up, approximately
five-second measurement, and `release-bench` on the recorded M4 Pro. The
treatment regressed resident time from 7.3275–7.4667 ms to 7.7683–7.8618 ms
and readback from 7.9272–8.0958 ms to 8.3685–8.5390 ms. Criterion reported
resident +5.02% to +6.94% and readback +4.52% to +6.72%, p=0.00.

The experiment was rejected and every kernel, pipeline, switch, and
prototype-only test was removed. Compiler register/private-memory, occupancy,
spill, and cache counters were unavailable and no causal claim is made.
`docs/performance-experiments/P1-metal-idwt53.json` is the validated record.

## P2 Metal Irreversible IDWT97 Fusion Evidence

The temporary prototype staged one complete axis in a 4096-float threadgroup
array, preserved the existing scale multiplication and four `fma` lifting
operations, and retained the generic route. Fused and fallback output bits
matched for odd 5×3 geometry, all singleton-origin combinations, and the
representative 1023×767 focused fixture. The eligible per-level physical
sequence fell from interleave plus ten transform dispatches to interleave plus
two fused axis dispatches.

Criterion used deterministic irreversible HTJ2K RGB8 512×512 batch 16, ten
samples, three-second warm-up, approximately five-second measurement, and the
`release-bench` profile on the recorded M4 Pro. Resident time moved from
8.2194–8.3167 ms to 8.1828–8.2965 ms, a 0.34–1.57% improvement. Readback moved
from 8.5792–8.7091 ms to 8.5576–8.7163 ms, with a -2.65% to +0.05% change
interval that included no change. Both variants produced SHA-256
`81a04643cc6ea4aecd77c43e753e9f2cb11efbdffa7665371db3c913abb6d981`.

The experiment was rejected and its production candidate removed. It consumed
16 KiB of threadgroup memory, did not implement tiled halos beyond 4096
samples, and lacked vendor register/private-memory/occupancy/spill counters.
No unsupported causal claim is made. The validated record is
`docs/performance-experiments/P2-metal-idwt97.json`.

## P3 Metal Irreversible FDWT97 Fusion Evidence

The temporary base-fusion candidate gave one thread ownership of a complete
row or column, applied the existing alpha/beta/gamma/delta `fma` sequence with
floating-point reassociation and contraction disabled, then wrote the exact
low/high scaled layout. Both candidate and kill-switch fallback matched the
fractional CPU reference bit-for-bit across three levels and produced
byte-identical native single- and multi-level codestreams. The physical
two-axis sequence fell from ten dispatches to two per level.

Criterion used a deterministic fractional-f32 1024×768 three-level stage and
an irreversible HTJ2K Gray8 512×512 three-level full encode, ten samples,
three-second warm-up, approximately five-second measurement, and
`release-bench`. The stage baseline was 1.8013–1.9794 ms and the serial
treatment 2.5817–2.6222 ms; Criterion's saved-baseline change comparison was
noisy and did not establish a change. Full encode moved from
25.687–25.904 ms to 26.183–26.401 ms, a significant 1.23–2.38% regression
(p=0.00). Stage and codestream SHA-256 values matched across variants.

The host exposed only public GPU timestamp counters. Registers, private bytes,
occupancy, active SIMD groups, spill loads/stores, and cache metrics therefore
remain explicitly unavailable rather than inferred. The candidate was removed.
The validated record is
`docs/performance-experiments/P3-metal-fdwt97.json`.

## P4 Terminal IDWT/MCT/Store Preflight Evidence

The current eligible RGB8 path already performs inverse RCT/ICT, clamp/convert,
and store in one native-color kernel. The remaining proposed fusion would have
to consume three pre-vertical component states, but the component executor
currently completes and materializes each plane independently. Supporting both
5/3 and 9/7 terminal lifting would therefore create a second cross-component
execution graph and duplicate transform-specific boundary arithmetic.

The required-product probe observed nine logical IDWT stages, one inverse-MCT
stage, and one final-store stage and preserved SHA-256
`81a04643cc6ea4aecd77c43e753e9f2cb11efbdffa7665371db3c913abb6d981`.
Split-command profiling did not expose a stable per-stage timing row for this
repeated path, and public counters on the host exposed timestamps only. The
task's explicit excessive-duplication rejection criterion was met at preflight;
no prototype, switch, or production complexity was added and no performance
claim is made.

## P6–P10 Tier-1 Resource Evidence Gate

The public Metal counter inventory contained only timestamps. Pipeline state
can provide static threadgroup bytes, SIMD width, and maximum threads per
group, but not registers, private bytes, achieved occupancy, active SIMD
groups, spill loads/stores, or cache counters. Missing headless trace spill
events are not conclusive, and the installed translator's register-use option
is undocumented. The Linux/NVIDIA lane is unavailable.

`docs/performance-experiments/P6-private-memory.json` validates as `blocked`.
No source-only spill claim was made and P7–P10 received no new cooperative
redesign. Existing Classic style-0 cooperation and generic fallbacks remain.

## P5 Metal Combined Input/MCT Evidence

The defaulted combined job is offered only for three-component MCT input and
keeps all decline and fallback behavior. Exact Metal tests cover signed and
unsigned 1–16-bit loading, reversible RCT, nested-FMA ICT, and forced separate
stages. Full RGB8 512×512 encode preserved codestream SHA-256 and improved RCT
from 41.055–41.153 ms to 40.663–40.707 ms and ICT from 50.845–51.157 ms to
50.435–50.651 ms. Criterion reported supported 0.63–1.15% and 0.55–1.14%
improvements respectively (p=0.00). The validated promotion record is
`docs/performance-experiments/P5-metal-input-mct.json`.

## P11 Metal Cooperative Packetization Evidence

The temporary one-threadgroup-per-tile route retained ordered header, tag-tree,
inclusion, and L-block mutation on lane 0, then synchronized and copied packet
bodies across the group. Direct resident tests covered Classic and HT coding,
all five progression orders, first/prior inclusion, L-block growth, empty
packets, and multiple layers. Both variants produced identical framed batch
hashes (`f3c9859d…c08a5` Classic and `e298bfd4…dca2` HT) and decoded to the
original RGB8 pixels.

Criterion on the true resident RGB8 512×512 batch-16 product path measured
Classic baseline 46.597–46.812 ms versus treatment 48.948–49.168 ms
(+4.71–5.40%) and HT baseline 8.2991–8.3648 ms versus treatment
10.401–10.493 ms (+24.67–26.16%), both p=0.00. The candidate, switch, and
prototype-only tests were removed; the ordered legacy packetizer and parallel
payload-copy dispatch remain. The validated rejection record is
`docs/performance-experiments/P11-metal-cooperative-packetization.json`.

## P12 Metal Terminal Column/Quantization Evidence

The temporary fused route applied final 9/7 column lifting and wrote the exact
quantized code-block layout without four temporary float subband handoffs.
Real-Metal differential tests covered tiny, odd/even, wide/tall, and truncated
code blocks. Its isolated 512×512 batch-16 terminal-stage measurement improved
5.04–7.13%, but this was not the priority product path.

The retained full JPEG-to-HTJ2K batch-16 512×512 product benchmark verifies
exact decoded output and framed SHA-256
`3ab221...e663`, and reports stage and transfer metrics. Baseline was
30.402–31.470 ms and treatment 30.415–31.212 ms; Criterion reported
-4.94% to +1.38% with p=0.36. The candidate kernel, pipeline, route, and active
switch were therefore removed. The generic float-band/staged implementation and
the >1024 fallback regression remain. The record correctly identifies 64×64
code blocks and validates at
`docs/performance-experiments/P12-metal-column-quantize.json`.

## P13 CUDA Column-Lift/Quantization Evidence

The isolated source snapshot ran on Ubuntu 24.04.4 under WSL2 with an RTX 4070
SUPER (compute capability 8.9), NVIDIA driver 610.88, CUDA 13.2, Rust 1.96.1,
LLVM 22.1.2, and cargo-oxide 0.2.1. The strict CUDA release gate passed before
measurement with 26/26 commands and 918 tests. A separate correctness repair
also proved exact Classic irreversible output, OpenJPEG codestream IDWT
normalization, and native half-tie conversion on this device.

The P13 baseline and treatment ran as separate processes. The 512×512 batch-16
stage input SHA-256 was
`d0389c0a7f4d506aeb5dc1e30212164b8e26b9fd222101e00ddc40ee34dd7d60`;
the complete preencoded metadata/payload SHA-256 was
`7c0c2b7027155ea12eddc0b7951c399f8dfe1988ab6fda044ad4195e96f8b539`.
The priority sRGB→YBR420 JPEG input SHA-256 was
`a2e0a67a28e6acccd57455353134179e2ba7a462f21cfd892e574ef569f61234`;
all 16 output codestreams parsed and independently decoded, with identical A/B
aggregate SHA-256
`656306703538bb190e75431c0dc954008dc9e97a9164a0e86d43d42e745c2e04`.
The record's corpus digest is SHA-256 over the domain string
`P13-CUDA-DWT97-COLUMN-QUANTIZE\0`, followed by each binary workload digest
framed with an eight-byte little-endian length.

The stage resident-preencode interval changed from 9.1972–9.2708 ms to
9.0592–9.0810 ms, but the isolated column-plus-quantize interval regressed from
541.65–553.81 µs to 572.57–584.33 µs (+4.21% to +7.53%). The priority product
changed from 15.840–15.954 ms to 15.603–15.867 ms; those absolute intervals
overlap, so the fail-closed validator does not support promotion. Baseline
product float bands occupied 25,165,824 bytes and caused 50,331,648 bytes of
logical write/reread traffic; treatment removed both. Exactness and traffic
reduction were insufficient to justify a candidate that regressed its target
stage without a decision-grade product win.

The candidate kernel, ABI, launch route, active switch, and prototype-only code
were removed. Production retains staged i16/F32 routes, the generic >1024 row
fallback, a 1032×8 differential regression, and the reusable product benchmark.
The validated record is
`docs/performance-experiments/P13-cuda-column-quantize.json`. P14-P17 are also
complete on the same direct CUDA lane. None is described as hardware-blocked.

## P14 CUDA Wide-Axis IDWT Evidence

The same RTX 4070 SUPER lane executed an exact odd-origin batch-16 regression
for 2592-wide reversible 5/3 and irreversible 9/7. The tiled candidate used
five physical dispatches for 5/3 and eleven for 9/7; every output bit matched
the generic single-output oracle before timing.

Separate-process Criterion runs covered both transforms, batch 1 and 16, and
512×512 and 2592×1944 axes. Wide batch 1 improved from 2.3282–2.3350 ms to
0.9363–0.9509 ms for 5/3 and from 4.5862–4.5933 ms to 2.2534–2.2837 ms for
9/7. The required wide batch-16 cells moved in the opposite direction: 5/3
regressed from 15.077–15.099 ms to 18.333–18.370 ms (+21.44–21.73%), and 9/7
regressed from 32.022–32.060 ms to 44.737–44.817 ms (+39.63–40.03%). The
benchmark-only narrow force seam also exposed 111–432% regressions relative to
the existing whole-line cooperative routes. All A/B input and framed output
hashes matched.

Because production selection knew geometry but not batch, a wide-shape
promotion could not retain the batch-1 gain without routing the measured
batch-16 cliff into production. The tiled device kernels, host launch modes,
active switch, benchmark force seam, and candidate-only tests were removed.
Generic wide and existing bounded Cooperative53/97 routes remain, along with a
single-path benchmark and exact wide regression. The validated record is
`docs/performance-experiments/P14-cuda-wide-idwt.json`.

## P15 CUDA Irreversible FDWT97 Shared-Staging Evidence

The same RTX 4070 SUPER lane evaluated a temporary shared-staging forward 9/7
route. Each block staged 32 low/high output pairs across eight lines with a
four-sample halo; horizontal launches used 2,304 bytes of shared memory and
vertical launches used 3,072 bytes. A focused odd 67x71 two-level differential
test first proved f32 `to_bits` parity against the generic route and confirmed
that both launch orientations selected shared staging.

Separate-process Criterion runs covered three-level 512x512, 1024x1024, and
2592x1944 transforms at batch 1 and 16. All framed input and output hashes
matched across variants. The shared route improved 512 batch 16 by
6.31–7.25%, 1024 batch 1 by 1.92–3.84%, 1024 batch 16 by 0.93–3.33%, and wide
batch 1 by 0.16–1.39%. Those wins did not generalize: 512 batch 1 regressed
0.429–1.220%, and wide batch 16 crossed no change at -0.130% to +1.879%.
Static source-load accounting, which is not a hardware cache-counter
measurement, fell from 79,822,848 to 3,584,000 bytes for 512 batch 1 and from
24,549,267,456 to 1,111,159,296 bytes for wide batch 16.

The priority irreversible HTJ2K RGB8 512x512 batch-16 full encode independently
parsed and decoded every codestream and preserved aggregate SHA-256
`d578f223b9484070ba52c3f459f3ba00f176d5177b914e8c41707185f042b9b7`.
Its absolute interval changed from 5.573581–5.586208 s to
5.567594–5.575540 s. Although Criterion's relative interval was -0.282% to
-0.020%, p=0.06, the treatment upper bound exceeded the baseline lower bound;
the repository's fail-closed decision rule therefore did not support
promotion.

The record's corpus digest is SHA-256 over the domain string
`P15-CUDA-FDWT97-SHARED-STAGING\0`, followed by each binary workload digest
framed with an eight-byte little-endian length. The shared kernels, route,
switch, trace seam, and candidate-only tests were removed. Production retains
the generic route and a reusable single-path exact stage/product benchmark.
The validated rejection record is
`docs/performance-experiments/P15-cuda-fdwt97-shared.json`.

## P16 CUDA RGB Input-Fusion Evidence

The same RTX 4070 SUPER lane evaluated a temporary kernel that combined RGB
deinterleave, level shift, and RCT or nested-binary32-FMA ICT. Exact hardware
coverage included signed and unsigned 1-16-bit inputs, reversible and
irreversible transforms, contiguous and strided paths, eligible RGB routes,
and non-RGB/non-MCT fallbacks. The retained benchmark's repeated probes matched
the native RCT/ICT component-plane oracles bit-for-bit and reduced the physical
input dispatch tuple from separate `(1, 1, 0, 2)` to fused `(0, 0, 1, 1)` for
deinterleave, MCT, combined, and total input dispatches.

Separate-process Criterion runs used ten samples, one second of warm-up, and
three seconds of measurement. The 512x512 RGB8 RCT stage improved from
4.008295-4.061676 ms to 3.194531-3.258764 ms (18.70-21.35%), while ICT improved
from 3.996569-4.030652 ms to 1.883434-1.894825 ms (52.59-53.27%). Stage output
SHA-256 values matched across variants at
`5d7319bf65cefb7f6456364e605263bc485d63f1b1e712f3457974c9abb854e3`
for RCT and
`73547dcc5a7a34992f554fe324ce9220d27fb03e226e435454606d8d1b423478`
for ICT.

Those isolated wins did not carry into the decision-grade HTJ2K products.
Lossless RCT changed from 43.178660-43.383856 ms to
43.004971-43.371718 ms; the absolute intervals overlap. Lossy ICT changed from
194.879754-195.727981 ms to 194.967121-196.525571 ms; those intervals also
overlap and the treatment point estimate regressed. Logical product dispatch
totals remained 84 for RCT and 36 for ICT, while the physical input portion
fell from two dispatches to one.

Both product codestreams were deterministic, parsed, and independently
decoded. The lossless aggregate codestream SHA-256 was
`bce6104d4e1a9fef3279e51da2e2d22cfdab6a61874c9613f58cfa46ee46d471`;
decoded RGB8 SHA-256 was
`d9457826d1278615d20d3869e0789be6dc5e13ecdfec5d2d1adeb9ebf5bf992b`
with infinite PSNR. The lossy codestream SHA-256 was
`67ece61ed9f2a0b2850d59814e9558a11ce05c0e2d31463722a60fe08e28cc29`;
decoded RGB8 SHA-256 was
`a957cd376ec544d631fcabfc91055e9557c239dcc68b051ccacf9d4d0e1cdb0b`
at 25.388668687821028 dB PSNR.

All four workloads used the same framed binary input SHA-256
`b6d8f7430dd6419757a60888e8e4aa339d0e6f5febb760c4f4c4529eeff27fff`.
The record's corpus digest
`23d8c880bbee44adb07490844c525b6c858e04c44a20b22bb9ccfca07e9d8bc3`
is SHA-256 over the domain string `P16-CUDA-INPUT-FUSION\0`, followed by the
unique binary workload digest framed with an eight-byte little-endian length.
The fail-closed product evidence rejects promotion. The specialized kernel,
launch route, production selector, switch, and candidate counters were removed.
Separate deinterleave and RCT/ICT are the sole production route; doc-hidden
combined-input methods remain as two-dispatch compatibility wrappers. The
reusable single-path benchmark retains exact stage/product hashes, product
dispatch accounting, parsing, independent decoding, strict lossless output,
and lossy PSNR validation. Post-cleanup checks passed the adapter 165-test and
engine 172-test suites, repo-lint 99/99, all-feature checking, benchmark
compilation, strict library and benchmark clippy, formatting, diff checks, and
the rejected-candidate symbol policy. No additional remote/PTX cleanup check
was run; the local engine checks cannot validate CUDA-Oxide device compilation,
so the next canonical CUDA gate still owns that residual risk. The validated
record is `docs/performance-experiments/P16-cuda-input-fusion.json`.

## P17 CUDA Final IDWT/Store Profile-Preflight Evidence

The direct RTX lane ran the retained deterministic 512x512 RGB8 4:4:4 matrix
for Classic and HT codestreams, reversible 5/3 and irreversible 9/7, and batch
1 and 16. Every cell used one decomposition level, two IDWT dispatches, zero
separate inverse-MCT dispatches, and one fused MCT/store dispatch. CUDA-event
profiling retained aggregate IDWT time while splitting the final stage into
interleave-plus-horizontal and vertical synthesis. Exact output validation and
SHA-256 calculation ran outside Criterion timing.

Two correctness defects had to be resolved before the measurements were
admissible. The exact-native irreversible ICT store first returned 129 for a
centered +0.5 sample where the CPU ties-even-before-shift contract requires
128; its RGB/RGBA path was repaired and verified on RTX. P17 still exposed the
same mismatch because its actual route was the display-width
`j2k_store_rgb8_mct_batch` kernel. A second focused RED reproduced 129 versus
128 through that exact entry point; RGB8 and RGB16 irreversible MCT now round
centered results ties-to-even before level shift, while reversible behavior and
global sample conversion remain unchanged. Both focused repairs passed on RTX
before the final eight-cell run.

The final capture completed with status 0 at
`/dev/shm/j2k-p17-profile-display-green.log`. Criterion intervals are
`[lower, estimate, upper]` in milliseconds. Stage columns are GPU-event
microseconds except resident wall, which is the structured probe wall in
microseconds. Tail share is `(final vertical + fused store) / resident wall`.

| Codec / transform | Batch | Criterion CI ms | IDWT / inter+H / final V / store / wall µs | Tail share | Input SHA-256 | Output SHA-256 |
|---|---:|---|---|---:|---|---|
| Classic R53 | 1 | [12.358992, 12.391618, 12.422359] | 1888 / 575 / 849 / 2246 / 482588 | 0.641% | `5ee289635a450218c16a9155e98608fde707b2e8fa9003c0553487e37992d732` | `91df7cec8ba5ae6a21b2ff11b9001e400c419006b008cd13508255444cb7b1c6` |
| Classic R53 | 16 | [65.008932, 65.203867, 65.406390] | 9375 / 740 / 1049 / 1821 / 206509 | 1.390% | `5ee289635a450218c16a9155e98608fde707b2e8fa9003c0553487e37992d732` | `15768e041e43e1e6bbd6c0eb991ffe3a6fd78f7317aa72e16b4d3e513a2aca89` |
| Classic I97 | 1 | [13.665884, 13.685833, 13.705368] | 2019 / 748 / 753 / 1750 / 140335 | 1.784% | `a56bacd9e54eba1cc00e474c9334cb0f7c63385fda35bd019d1b781d1e637183` | `aa14f897dc2d144841be13cc131c4ff639dd241360acf0ebc2d87a0ffc45a75c` |
| Classic I97 | 16 | [76.537217, 76.824113, 77.217454] | 9077 / 797 / 972 / 1820 / 249017 | 1.121% | `a56bacd9e54eba1cc00e474c9334cb0f7c63385fda35bd019d1b781d1e637183` | `b3724859f88a9a81a32b5ed9f47ec73c9c12bcdb282fef042c028df0cf5e29ae` |
| HT R53 | 1 | [1.776727, 1.782841, 1.788460] | 2099 / 331 / 663 / 1481 / 153518 | 1.397% | `40fa9a74b0e8c66defc48f8f3be3de109606f61ca7ec58df0cee4798b76fb60f` | `91df7cec8ba5ae6a21b2ff11b9001e400c419006b008cd13508255444cb7b1c6` |
| HT R53 | 16 | [22.374372, 22.962771, 23.530375] | 10491 / 861 / 1224 / 1924 / 190822 | 1.650% | `40fa9a74b0e8c66defc48f8f3be3de109606f61ca7ec58df0cee4798b76fb60f` | `15768e041e43e1e6bbd6c0eb991ffe3a6fd78f7317aa72e16b4d3e513a2aca89` |
| HT I97 | 1 | [1.895507, 1.906904, 1.917212] | 1840 / 82 / 605 / 1340 / 147466 | 1.319% | `9acdbbb74075b788bc9476000f453e3c2248e3092a4f69404c66b545471f4076` | `aa14f897dc2d144841be13cc131c4ff639dd241360acf0ebc2d87a0ffc45a75c` |
| HT I97 | 16 | [23.047382, 23.119887, 23.188736] | 9696 / 873 / 1124 / 1819 / 189219 | 1.555% | `9acdbbb74075b788bc9476000f453e3c2248e3092a4f69404c66b545471f4076` | `b3724859f88a9a81a32b5ed9f47ec73c9c12bcdb282fef042c028df0cf5e29ae` |

The largest observed tail share was 1.784%, and the smallest was 0.641%.
Every cell is far below the plan's 10% GO gate, so P17 received a NO-GO before
a prototype. No final-IDWT candidate, selector, switch, fallback branch, or
candidate A/B exists. The split profiler, deterministic matrix, exact hashes,
and fail-closed benchmark policy remain reusable. Following ADR-P004's
preflight precedent, no experiment JSON was created because there was no
candidate or A/B experiment to record.

## P18 Metal Staged JPEG Encode Evidence

The promoted Metal encoder separates one MCU-parallel coefficient-precompute
dispatch from ordered per-tile entropy emission. The temporary A/B switch and
obsolete fused host pipelines and kernels were removed after promotion.
Subprocess-isolated exact coverage includes Gray and RGB 4:4:4/4:2:2/4:2:0,
quality 1/90/100, restart intervals absent and present, odd dimensions and edge
padding, byte stuffing and markers, determinism, and independent decoding. The
aggregate reviewed output SHA-256 is
`b9af03a1a522926f3bee386fd554f146db773cf8287cef91d9001335c88c3a26`.

For RGB8 4:2:2 512×512 batch 8, baseline was 776.67–777.35 ms and treatment
395.85–396.52 ms, a 48.96–49.06% improvement with p=0. Batch-1 512×512 and
64×64 cells improved by more than 61%. The validated promotion record is
`docs/performance-experiments/P18-metal-jpeg-staged-encode.json`.

## P18 CUDA Staged JPEG Encode Evidence

The promoted CUDA encoder uses one MCU-parallel coefficient-precompute dispatch
followed by ordered entropy. Five subprocess-isolated RTX cells covered RGB8
4:2:2 at 512×512 batch 8 and batch 1, 64×64 batch 1, and 512×512 batch 8
with restart intervals 16 and 32. Every probe was deterministic, preserved its
exact A/B codestream hash, and passed both the repository decoder and
independent `jpeg-decoder`.

| Cell | Serial baseline 95% CI | Staged 95% CI | Exact result |
|---|---:|---:|---|
| 512×512 batch 8, no restart | 6.795133–6.805057 s | 336.067–336.740 ms | 95.044–95.062% faster |
| 512×512 batch 1, no restart | 6.755327–6.756790 s | 236.520–236.663 ms | 96.497–96.500% faster |
| 64×64 batch 1, no restart | 105.703–105.913 ms | 7.028–7.057 ms | 93.324–93.365% faster |
| 512×512 batch 8, restart 16 | 6.809174–6.810617 s | 336.328–336.845 ms | 95.053–95.062% faster |
| 512×512 batch 8, restart 32 | 6.808454–6.809186 s | 336.292–336.887 ms | 95.052–95.061% faster |

The priority path moved from one dispatch and zero coefficient scratch to two
dispatches and 16,777,216 bytes of checked scratch. Measured staged scratch was
2,097,152 bytes at 512×512 batch 1 and 32,768 bytes at 64×64 batch 1. The
representative end-to-end gain justified that bounded resource cost. Nsight
Compute resource and traffic counters are `null`, not estimates: the driver
returned `ERR_NVGPUCTRPERM`.

The exact matrix covered fixed-order Gray and RGB, 4:4:4/4:2:2/4:2:0,
quality, restart, odd-edge/padding, marker/stuffing, determinism, and both
decoders. After promotion cleanup, the sole staged route reproduced the
domain-separated input digest
`5fbd44a6890bfe562d66709eda023f0b5b8f942f0e113824399cfd39f06fe570`
and output digest
`99b76d5a103ed958e4a4cdef80fb8e48cd8f2c6e28ababbf4ff787fea67ab314`.
The domains are `P18-CUDA-STAGED-JPEG-EXACT-INPUTS\0` and
`P18-CUDA-STAGED-JPEG-EXACT-MATRIX\0`; each of the fixed 16 records is framed
with its little-endian u64 byte length.

Restart-16 coverage exposed a CPU decoder defect rather than a CUDA codestream
defect. The frame contained all 127 ordered restart markers for 2,048 MCUs and
the independent decoder accepted it, but `BitReader::consume_restart_marker`
only probed an unprefetched marker when zero bits remained. A red regression
with four legal pad bits and an unprefetched RST0 reproduced
`UnexpectedEoi { 16/2048 }`. The repair probes when no marker is pending and at
most seven pad bits remain, while retaining the eight-bit stuffed-data and
wrong-marker rules. Bit-reader 17/17, full `j2k-jpeg` 499 plus integrations,
and RTX restart-16 batch 1 and 8 all passed.

Cleanup removed the serial kernels, host route, split-process switch, and
candidate-only tests. The staged route, exact matrix, restart regression,
single-path benchmark, and checked plan remain. The validated record is
`docs/performance-experiments/P18-cuda-jpeg-staged-encode.json`.

## P19 Metal JPEG Decode Defusion Evidence

The existing narrow split coefficient/IDCT route was exposed temporarily for a
profile-first A/B. Exact baseline and treatment output SHA-256 matched. Baseline
was 10.456–10.688 ms, treatment 10.621–13.562 ms, and Criterion reported
-0.57% to +14.14% with p=0.30. The treatment required 12,681,216 bytes of
coefficient scratch for 512×512 batch 16 and the texture path added five private
allocations. Production candidate and switch code was removed; the original
test-only diagnostic remains. The validated rejection record is
`docs/performance-experiments/P19-metal-jpeg-decode-defusion.json`.

The superseded P19-only limitation record at
`docs/performance-experiments/P19-cuda-jpeg-decode-historical-blocker.json`
documents the historical unavailable-lane checkpoint. The direct RTX lane
subsequently completed both CUDA P19 decisions below.

## P19 CUDA JPEG Adaptive Checkpoint Evidence

The initial fused profile showed that each independent entropy checkpoint was
launched as its own block with one thread, activating one lane per warp. The
candidate retained one logical thread per checkpoint but used an adaptive
launch: checkpoint counts below 128 preserve block 1; counts at or above 128
use block 128 and `ceil(checkpoints / 128)` blocks. Ten subprocess-isolated
cells covered 4:2:0/4:2:2/4:4:4, batch 1/16, restart-16, 64x64, 512x512, and
1024x1024. Every A/B pair preserved its input/output SHA-256, deterministic
repeat, checkpoint count, component workspace, dispatch/transfer accounting,
and CPU conformance. Correctness-only probes covered odd 4:2:0 and 4:2:2
seams/boundary repair, caller-owned padded output, strict CUDA region/scaled
rejection, and Auto CPU fallback. The CUDA JPEG adapter has no texture-output
API, so the applicable CUDA output boundary is full RGB8 device surface and
caller-owned device buffer rather than the Metal texture path.

| Cell | Block-1 baseline 95% CI | Adaptive 95% CI | Point change |
|---|---:|---:|---:|
| 4:2:0 512x512 batch 16 | 21.081–21.255 ms | 20.785–21.026 ms | -1.21% |
| 4:2:0 512x512 batch 1 | 1.726–1.740 ms | 1.670–1.685 ms | -3.15% |
| 4:2:2 512x512 batch 16 | 22.094–22.461 ms | 18.903–19.472 ms | -13.75% |
| 4:2:2 512x512 batch 1 | 1.788–1.822 ms | 1.584–1.610 ms | -11.69% |
| 4:4:4 512x512 batch 16 | 24.042–24.363 ms | 20.701–20.885 ms | -14.05% |
| 4:4:4 512x512 batch 1 | 1.904–1.942 ms | 1.694–1.709 ms | -11.52% |
| 4:2:0 1024x1024 batch 1 | 3.859–3.879 ms | 3.397–3.442 ms | -11.68% |

The priority absolute intervals do not overlap and all seven
geometry-changing cells improved. Restart-16 batch 1/16 and 64x64 were
below-threshold controls: both processes executed identical block-1 production
code, so their +0.58%, +1.21%, and +3.96% cross-process movement is recorded as
run noise rather than a geometry regression. The cleaned single-path priority
rerun was 20.398–21.151 ms and reproduced exact hashes, block-128 geometry,
zero coefficient scratch, diagnostics, and conformance. The validated
promotion record is
`docs/performance-experiments/P19-cuda-jpeg-packed-checkpoints.json`.

Restart profiling exposed a capability bug before the A/B could be accepted.
`summarize_device_batch` reused
`PreparedDecodePlan::matches_fast_tile_shape()`, whose CPU-oriented predicate
intentionally requires no restart interval. A restart-coded 4:2:0 fixture with
`restart_interval=Some(2)` therefore reported
`device.matches_fast_420=false`, despite CUDA fast-packet construction and
checkpoint decoding supporting restart markers. The RED regression reproduced
that mismatch. The fix adds a device-only fast-4:2:0 predicate with the same
sampling and geometry constraints but without the CPU no-restart restriction;
the CPU predicate and routing are unchanged. On RTX, the same restart-coded
input profiled twice and then passed strict session batch decode.

## P19 CUDA JPEG Coefficient/IDCT Defusion Evidence

The settled adaptive fused profile justified a narrow 4:2:0 prototype. Entropy
threads emitted exact MCU-major i32 coefficients, a 128-thread kernel deposited
parallel block IDCT results, and the existing conversion stage completed the
RGB8 output. Six eligible 4:2:0 cells and four unchanged fused 4:2:2/4:4:4
controls ran in isolated baseline/treatment processes. Exact output hashes,
determinism, CPU conformance, adaptive checkpoint geometry, workspace, and
transfer bytes were preserved.

| 4:2:0 cell | Settled fused 95% CI | Split 95% CI | Point regression | Split scratch |
|---|---:|---:|---:|---:|
| 512x512 batch 16 | 20.398–21.151 ms | 30.846–31.996 ms | +51.47% | 25,165,824 B |
| 512x512 batch 1 | 1.667–1.700 ms | 2.217–2.273 ms | +33.60% | 1,572,864 B |
| restart-16 batch 16 | 41.010–41.393 ms | 47.584–48.095 ms | +16.14% | 25,165,824 B |
| restart-16 batch 1 | 2.912–2.924 ms | 3.202–3.228 ms | +10.18% | 1,572,864 B |
| 64x64 batch 1 | 1.358–1.382 ms | 1.404–1.415 ms | +3.00% | 24,576 B |
| 1024x1024 batch 1 | 3.377–3.391 ms | 4.004–4.038 ms | +18.76% | 6,291,456 B |

All eligible products regressed. The split added one kernel dispatch per tile
and three logical scratch accesses (clear write, entropy write, IDCT read), or
73,728 to 75,497,472 logical scratch bytes. The four unchanged controls crossed
zero with p=0.54–0.77. The split kernels, route selection, scratch allocation,
switch, and split-only tests were removed. The validated rejection record is
`docs/performance-experiments/P19-cuda-jpeg-decode-defusion.json`.

Static `ptxas -arch=sm_89` evidence for the settled fused kernels reported
fast420/fast422/fast444 respectively at 72/64/48 registers, 896/768/704-byte
stack-local frames, zero spill bytes, and 488 bytes constant memory bank 0.
The rejected entropy-coefficient kernel used 52 registers, a 320-byte frame,
zero spills, and 488 bytes constant memory bank 0; IDCT deposit used 40
registers, a 576-byte frame, zero spills, 400 bytes constant memory bank 0,
and 24 bytes bank 2. Nsight Compute 2026.1.1 returned
`ERR_NVGPUCTRPERM`, so no dynamic traffic, occupancy, shared-memory, spill, or
cache counter is inferred.

After rejection cleanup, real CUDA-Oxide rebuilt the 837,213-byte PTX with
SHA-256
`b90b1a97152d08e0fe9e153304dd53bb7062119f86529475cc2dbbfefe4fc9e1`,
exactly restoring the pre-candidate artifact. The strict build, owned decode
8/8, pitched output 1/1, focused 4:2:2/4:4:4 1/1, and release-bench compilation
passed on the RTX lane. Rejected symbol, source, and switch searches were
empty; the preserved `/dev/shm/j2k-p19-cleanup-*` logs report overall status 0.
| `git diff --check` | pass |

## M7 Facade Encode-Ownership Evidence

The operational `encode.rs` is now a 49-line module root. Its 106-line stable
API module contains only documentation, signatures, and delegation. Lossless
and lossy result construction, ROI descriptor validation, accelerator stage
resolution, high-bit adaptation and guards, shared geometry, CPU/native calls,
and round-trip validation each have explicit ownership. Existing public names
remain re-exported from the private `encode` module through the crate root.

The prior private `native` and `routing` names became `cpu` and `accelerator`.
CPU fell from 502 to 369 lines after high-bit and ROI policy moved out. The
lossy rate-target search remains a cohesive 443-line owner rather than being
split solely for line count.

| Command | Result |
|---|---|
| M7 encode-ownership regression before extraction | failed as expected: `encode/mod.rs` absent |
| `cargo test -p xtask --test repo_lint source_size_policy -- --nocapture` | pass: 10/10 |
| `cargo check -p j2k --all-features --all-targets` | pass |
| `cargo clippy -p j2k --all-features --lib --no-deps -- -D warnings` | pass |
| `cargo test -p j2k --all-features --no-fail-fast` | pass across 115 library tests and all integration/doc targets |
| Focused encode unit/lossless/lossy rerun | pass: 43 unit, 68 executed lossless with one intentional ignore, 28 lossy |
| `git diff --check` | pass |

## M6 JPEG Capability-Ownership Evidence

The former 672-line `capabilities.rs` is now a private module hierarchy. Public
request and result contracts retain their crate-root exports. CPU rules own
12-bit, lossless, format, and sampling correctness; CUDA owns full-RGB8,
sequential, sampling, and addressability rules; Metal owns fast-surface and
resident-batch correctness; output geometry and planner-rejection handling are
separate; and resolution consumes the completed report.

`Auto` continues to resolve to `CpuHost` in this correctness-only layer.
Accelerator promotion thresholds remain in workload-aware adapter routing, so
eligibility cannot silently become a performance claim.

| Command | Result |
|---|---|
| M6 capability-ownership regression before extraction | failed as expected: `capabilities/mod.rs` absent |
| `cargo test -p xtask --test repo_lint source_size_policy -- --nocapture` | pass: 9/9 |
| `cargo check -p j2k-jpeg --all-features --all-targets` | pass |
| `cargo clippy -p j2k-jpeg --all-features --lib --no-deps -- -D warnings` | pass |
| `cargo test -p j2k-jpeg --all-features --no-fail-fast` | pass: 498 library tests plus all integration and doc targets |
| `cargo test -p j2k-jpeg-cuda --all-features --no-fail-fast --quiet` | pass: 98 tests across targets |
| JPEG CUDA and JPEG Metal strict library clippy | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast --quiet` | pass: 228 library tests plus 37 integration tests and doc tests |
| Broad all-target JPEG clippy | known failure: 288 pre-existing test-only disallowed-allocation violations; production library lint passes |
| `cargo clean` | removed 87.7 GiB of generated Cargo artifacts after disk exhaustion; no source files affected |
| `git diff --check` | pass |

## M4 JPEG Metal Codec-Batch Evidence

The 1,018-line `codec_batch.rs` is now a 27-line semantic root. Public request
types, normalized sources, eligibility inspection, source-neutral planning,
owner accounting, buffer targets, texture targets, and tile submission each
have a focused module. `batch.rs` remains the queue/group/flush/completion
owner, preventing overlapping batch hierarchies. All files are below 400 lines.

| Command | Result |
|---|---|
| M4 source-boundary regression before the split | failed as expected: modules absent |
| M4 source-size policy | pass: 7/7 |
| `cargo check -p j2k-jpeg-metal --all-features --all-targets` | pass |
| `cargo clippy -p j2k-jpeg-metal --all-features --lib --no-deps -- -D warnings` | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast` | pass: 228 library tests, 37 integration tests, and doc tests |

## A6 Shared Prepared-Plan Geometry Evidence

Classic and HT referenced plans now return the same borrowed
`J2kReferencedImageGeometry<'_>` representation without changing their stored
variant fields. The view owns the common semantics for empty tile sets,
grayscale/RGB/RGBA classification, single-tile component geometry, reduced
image dimensions, output rectangle, and uniform wavelet transform. Facade
wrappers expose the type as `PreparedImageGeometry<'_>` and preserve existing
`is_grayscale`, `is_color`, and `is_rgba` calls as compatibility delegates.
Classic payload fragments and HT cleanup/refinement records remain entirely
separate.

New parity tests build real Classic and HT grayscale and RGB codestreams, then
compare the same shared contract including component plan cardinality and
wavelet selection. Existing RGBA, multi-tile, request, cache, and execution
tests exercise the same delegates. A repo policy pins the single geometry owner
and rejects reintroduced facade-local classification or wavelet aggregation.

| Command | Result |
|---|---|
| `cargo test -p j2k --test owned_batch --no-fail-fast` | pass: 36/36 |
| `cargo test -p j2k-native --lib direct_plan --no-fail-fast` | pass: 12/12 focused |
| `cargo test -p j2k-cuda --all-features --no-fail-fast` | pass across all unit, integration, and doc targets |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features prepared -- --nocapture` | pass: 47 executed, 7 intentional ignores |
| `cargo test -p xtask --test repo_lint repo_lint_support::prepared_plan_policy:: -- --nocapture` | pass: 3/3 |
| Four affected all-feature library clippy targets with `-D warnings` | pass |

## A7 Packing and Sample-Conversion Evidence

`j2k-native/src/color/packing.rs` now owns the shared byte-output policy.
`SampleConversionPolicy` decides uniform-depth direct 8-bit conversion versus
per-component scaling, and performs the single rounding/quantization rule.
`SampleWindow` checks output length, ROI arithmetic, row bounds, and component
plane lengths before traversal. Full-image fallback and region packing use one
window iterator; the measured/common direct 8-bit full path keeps explicit
one-, two-, three-, and four-component loops.

The new differential matrix compares full output with an equivalent crop for
1-4 components, mixed 4/8/12-bit components, 12-bit scaling, signed 8/12-bit
samples, the complete image, a non-aligned interior ROI, and edge-aligned
regions. Exact byte equality is required. Short full and ROI buffers return
`OutputBufferTooSmall` rather than reaching an unchecked write.

| Command | Result |
|---|---|
| `cargo test -p j2k-native --lib --no-fail-fast` | pass: 636 executed, 2 ignored |
| `cargo test -p j2k-native --test component_planes --no-fail-fast` | pass: 24/24 |
| `cargo test -p xtask --test repo_lint repo_lint_support::packing_policy:: -- --nocapture` | pass: 2/2 |
| Native all-feature library and xtask all-target strict clippy | pass |
| Source-size policy and `git diff --check` | pass |

## A8 Typed Error-Taxonomy Evidence

`j2k-core::CapabilityRejection` is the single internal taxonomy for accelerator
request rejection. Its categories are unsupported format, sampling, bit depth,
operation, missing prepared plan, unsupported container, geometry mismatch,
resource limit, context mismatch, and checked contract violation. Each J2K and
JPEG CUDA/Metal adapter converts the typed value to its existing public static
reason variant only in `error.rs`, preserving source compatibility and exact
diagnostics.

The migration replaced 324 direct literal/constant constructions across
production control flow. The AST policy now reports an empty construction
inventory in the four adapters while allowing public pattern matches. Runtime,
native decoder, allocation, buffer, truncation, unsupported, and dual-cleanup
sources remain on their prior variants and were not flattened.

| Command | Result |
|---|---|
| `cargo test -p j2k-core --no-fail-fast` | pass: 77 tests |
| `cargo test -p j2k-cuda --all-features --no-fail-fast` | pass across all targets |
| `cargo test -p j2k-jpeg-cuda --all-features --no-fail-fast` | pass: 98 tests across targets |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast` | pass: 265 tests across targets |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features --lib --quiet -- --test-threads=1` | pass: 380 executed, 22 ignored |
| Full parallel J2K Metal suite | known Tier-1/MQ device failure plus hybrid global-cache counter race; focused tests pass |
| `cargo test -p xtask --test repo_lint repo_lint_support::error_taxonomy_policy:: -- --nocapture` | pass: 4/4, zero direct production constructions |
| Five affected all-feature library clippy targets with `-D warnings` | pass |
| `cargo xtask clone-audit` | pass: production 4.27%, Metal 25.08%, tests 2.71% |

## M1 `j2k-types` Module-Boundary Evidence

The `j2k-types` crate root fell from 1,125 lines to 76. Transform, Tier-1,
packetization, prepared-plan, dispatch-report, and dispatch-error contracts now
live with their owning behavior and compile-time move-only assertions. Every
prior root type remains re-exported. The already-public accelerator trait is
now named and documented as the low-level `dispatch` SPI; it was not placed
behind a feature because doing so would break downstream implementers.

The structural regression was run before the move and failed because all
required ownership modules were absent. It now passes and the old large-root
allowance is removed, so both re-monolithization and a stale exemption fail.
No packetization-specific output value existed, and M1 was behavior-neutral,
so the target's illustrative `output.rs` was not created as an empty or
speculative module.

| Command | Result |
|---|---|
| M1 source-boundary regression before the move | failed as expected: all ownership modules absent |
| `cargo test -p j2k-types --all-features` | pass: 18 unit tests and doc tests |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy:: -- --nocapture` | pass: 4/4 |
| `cargo clippy -p j2k-types --all-features --all-targets --no-deps -- -D warnings` | pass |
| `cargo check --workspace --all-targets` | pass; two known `j2k-cuda` dead-code warnings only |
| `cargo test -p j2k-native --all-features --lib` | pass: 636 executed, 2 ignored |
| `cargo test -p j2k --all-features --no-fail-fast` | pass across all facade targets |

## Final Gate 8 Current-Tree Evidence

The settled local tree completed every CPU/Metal validation that does not
require a release-version choice or NVIDIA hardware. No failure was waived and
no API-review or source-break ledger entry was added merely to make a gate pass.

| Command | Result |
|---|---|
| `cargo xtask ci` | pass; canonical formatting, clippy, panic policy, workspace tests, debug Metal tests, docs/examples, allocation probe, and unsafe audit |
| `cargo xtask metal-compile` | pass; strict clippy, optimized Metal integration/library tests, and docs after clearing 43.9 GiB of reproducible `target/` output that caused `ENOSPC` |
| `cargo xtask repo-lint` | pass: 101/101 |
| `cargo xtask clone-audit` | pass: production 753 clones / 13,504 lines / 4.26%; Metal 236 / 4,976 / 24.99%; tests 199 / 5,563 / 2.68% |
| `cargo xtask release-integrity` | pass |
| `cargo xtask public-support --final` | pass |
| `cargo xtask unsafe-audit` | pass |
| `cargo xtask panic-surface` | pass: `expect_used` 34/50, `unwrap_used` 13/16, explicit inventories unchanged |
| all 17 `cargo xtask gpu-experiment validate ...` invocations | pass |
| `cargo xtask stable-api` | pass |
| `cargo xtask semver` | pass after the user-approved 0.10.0 transition; all 18 baseline libraries pass and the review covers all 22 current libraries plus exactly 20 removed signatures |
| `cargo fmt --all -- --check` and `git diff --check` | pass after the final durable-evidence edit |

The 0.10.0 review records the removal of runtime-owned
`transcode_kernels_built` after CUDA transcode ownership moved into its engine,
and the generic neutral-device-codestream Metal transcode handoff that removed
the forbidden `j2k-transcode-metal -> j2k-metal` dependency. Preserving their
old signatures would reverse the required architecture. It also ledgers 18
canonical defining-path changes while explicitly preserving root compatibility
re-exports. The generated report is
`docs/release-evidence/public-api/reviewed-public-api-diff-0.10.0.md`; the exact
review configuration is
`docs/release-evidence/public-api/public-api-review-0.10.0.yml`.

The ordinary dirty-tree package command is fail-closed by design. Packaging was
therefore verified from a clean temporary copy of the exact post-CI source. The
validation-only snapshot was `53b8bc4`; `cargo xtask package` passed all release
archives and packaged J2K, CUDA, Metal, and ML consumer configurations. The
temporary copy was moved to Trash and is not repository history.

T.803 development evidence passed before the final `target/` cleanup: CPU
decoder 160/160 and encoder 56/56; Metal decoder 160/160 and encoder 35/35
applicable. Recorded report hashes were CPU JSON
`783142ba6f782315496c5616e6cab3bc3a55c54697631fde801cac8b4f959059`,
CPU Markdown `4f3a0b58152dfedd1dcc1887828c0f9d50c64ed03f0f1a5c2f3c380830c5691f`,
Metal JSON `1f767515be6c3831f8cb56b020fd9fc8b031dbe4e42b782b1a5945a4fae9ef5a`,
and Metal Markdown `68ab32308f33c2ea8b0cb3925bad13e667027ccfefd5e186826df86582372ff7`.
The report files were reproducible build artifacts and were removed by
`cargo clean`; the commands and hashes, not surviving `target/` paths, are the
durable evidence.

Gate 8 current-tree validation is complete. Current HEAD remains the audit
baseline `f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5`; no commit or push was
authorized.
Direct access to the user-provided Linux/NVIDIA lane completed P13-P19. P17
stopped at its profile gate without a prototype; P18 CUDA encoding promoted its
staged route; P19 CUDA decoding promoted adaptive checkpoint geometry and
rejected/removed coefficient-IDCT defusion. The public-API/version decision is
resolved at 0.10.0, and the settled host, Metal, and CUDA Gate 8 reruns pass.
The sole unmet evidence item is the final authorized implementation checkpoint
and its SHA. Clean-candidate exact-SHA release preparation remains separately
authorized work.

### Post-hardening development coverage and conformance

Changed-line coverage uses the audit baseline
`f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5` and lane-specific source ownership.
The host lane excludes the four CUDA accelerator crates now owned by the CUDA
lane. Accelerator lanes accumulate `j2k-ml` with only their own backend feature,
snapshot authoritative build-script cfg evidence before the narrower pass, and
enforce the independent 80% release-critical threshold.

| Lane | Overall changed lines | Critical changed lines | Result | Summary SHA-256 |
|---|---:|---:|---|---|
| host | 2985/3480, 85.78% | 276/299, 92.31% | pass | `60464383d47bb76f23848f75f39da92adedfc7e5304d644f597a877e934c4ea0` |
| Metal | 1863/2554, 72.94% informational | 395/484, 81.61% | pass | `a416e9c6868c053353c957cb27d1618c81517b7bdb2fd0bbacd9275168662aea` |
| CUDA | 3223/4430, 72.75% informational | 614/761, 80.68% | pass | `b1a9feed44f8cb724b8208ceba7a5ade5cbe51b00dae19291fb70638d811543b` |

The post-0.10.0 coverage artifacts were copied byte-for-byte from exact-source
Git overlays. Host LCOV/regions SHA-256 are
`ef320925d84d6f2972855d4bc95eb8cc0a1f01ba32f3d719425acbe894028846`
and `6fee497405555e09cb27a89251086a3cd8d822f5cab77c9300b6772fa0cf1eb6`.
Metal LCOV/regions are
`ecda031253987a3d4c5589db7cd7970eca4783716ba3b3008a3f98b9917e8579`
and `3ab036d02ed102878a730c64fa27358460d23e190fc81097e47d8123c80b3110`.
CUDA LCOV/regions are
`bae38146a4f421d99272378a6b3e159fccd0847a02a693be46841f286c294efd`
and `6e5fcc03d9e528772f3a3c885ebffd1f6572d360394e3bf47d067b815b32fe0c`.
Metal crates were not compiled in the CUDA lane. Every disposable overlay was
trashed and confirmed absent.

Development T.803 `--suite all` evidence passed on the settled architecture:
CPU decoder 160/160 and encoder 56/56; Metal decoder 160/160 and encoder 35/35;
CUDA decoder 160/160 and encoder 35/35. CUDA used the official pinned corpus
SHA-256 `ac04b52e1fe38404912036c14f215099ea9a785f38644fbe76ae8f3d1523c86d`
and OpenHTJ2K v0.19.0 commit
`e0f7ae853220d1e359c438b0bb6ad6cb2b3899db`. These dirty-tree development
runs prove codec behavior but are not substitutes for the five clean-candidate,
exact-SHA release reports required by `docs/release.md`.

Post-0.10.0 report SHA-256 values are CPU JSON/Markdown
`28deaec610f6b7311c8f72cd53efabb2f7e8ccbcced784099162a571c53c74b9` /
`6b6903afa955b39ed12b8be98c979f79b4af950cf6152c9c538f700135cb449e`,
Metal `1ffc72db2cd19f6a9109fb2cfa6542e285c9df276705f1a5462f395c56b684ec` /
`c64ab966e8929517a0afce9bdd5e31390c19f22d1f1c9f994453c37beb02f77e`,
and CUDA `8ba6b9a0bbac46b39bdd9adcf3046fbac4ea79f022b801615c8705152e653ecd` /
`ee336195aa81e403c99c32455c8618bf70b6e56f6130dceac1cd4630c68d46e6`.

### Post-0.10.0 Gate 8 validation

| Command | Result |
|---|---|
| `cargo xtask ci` | pass at workspace 0.10.0 |
| `cargo xtask clippy-strict` | pass |
| `cargo xtask repo-lint` | pass, 101/101 |
| `cargo xtask clone-audit` | pass: production 4.27%, Metal 24.95%, tests 2.67% |
| `cargo xtask codec-math-codegen` | pass |
| `cargo xtask public-support --final` | pass |
| `cargo xtask release-integrity` | pass in pre-candidate mode |
| `cargo xtask unsafe-audit` | pass |
| `cargo xtask panic-surface` | pass: `unwrap_used` 13/16 and `expect_used` 34/50 |
| `cargo xtask stable-api` | pass |
| `cargo xtask semver` | pass for all 18 v0.9.0 baseline libraries under the reviewed 0.10.0 transition |
| all 17 experiment record validations | pass |
| `cargo xtask bench-build --lane host` | pass |
| `cargo xtask release-cpu` | pass |
| `cargo xtask metal-compile` | pass |
| `cargo xtask release-metal --mode full` | pass, including all 21 required ignored runtime tests |
| `cargo xtask bench-build --lane metal` | pass |
| `cargo xtask release-cuda --mode full` | pass: 923 passed, zero failed or ignored across 46 result rows |
| `cargo xtask package-consumer-smoke --target cuda --cuda-runtime` | pass, including packaged nested CUDA-Oxide builds |
| `cargo xtask bench-build --lane cuda` | pass |
| clean-snapshot `cargo xtask package` | pass for all 23 archives and packaged consumers |
| `cargo xtask no-std` | pass for AArch64 none and WebAssembly |
| `cargo xtask fuzz-build` | pass for all five fuzz manifests |
| `cargo xtask machete` | pass |
| Python script unit discovery | pass, 80/80 |

The clean package snapshot was `c0fc0b0758b009c41f1e1b34e6ef0c65777a676a`
and was content-identical to the shared source at capture. The post-0.10.0 CUDA
runtime manifest covered 2,922 source files with SHA-256
`c78bfbc7ba09c278181a338c3e2d4b2cd7247d2436729190f6a5545bafd3486b`.
CUDA full validation, package smoke, benchmark compilation, T.803, and coverage
all returned status 0; no source edit was made on the RTX host.

## C1.2 Initial CUDA JPEG Engine Boundary Evidence

The unpublished `j2k-cuda-jpeg-engine` now depends inward on the low-level
runtime; the runtime has no reverse dependency. `JpegCudaEngine<'a>` is one
borrowed pointer wide and keeps the existing `CudaContext` identity used by
adapter sessions, buffers, pools, pinned transactions, and public constructors.
`j2k-jpeg-cuda` imports JPEG plans/types from the engine and invokes decode,
caller-output validation, diagnostics, single encode, and batch encode through
that boundary.

At this intermediate checkpoint the engine still delegated to runtime
compatibility methods and runtime still owned JPEG source and PTX packaging.
The later C1.2 extraction evidence below records removal of that delegation.

| Command | Result |
|---|---|
| C1.2 dependency test before scaffold | failed as expected: engine manifest absent |
| `cargo test -p j2k-cuda-jpeg-engine --all-features --no-fail-fast` | pass |
| Engine and JPEG CUDA strict all-feature library clippy | pass |
| `cargo check -p j2k-jpeg-cuda --all-features --all-targets` | pass |
| `cargo test -p j2k-jpeg-cuda --all-features --no-fail-fast --quiet` | pass: 98 tests across targets |
| `cargo test -p xtask --test repo_lint architecture_policy -- --nocapture` | pass: 5/5 |
| `git diff --check` | pass |

## C1.2 CUDA JPEG Family Extraction Evidence

The transitional boundary is now a complete family extraction. JPEG ABI and
domain types, checked host allocation, padding-free GPU byte views, decode and
encode validation, diagnostics, kernel entry-point inventory, launch bodies,
build flags, and both CUDA-Oxide projects live in the unpublished JPEG engine.
The low-level runtime contains no `CudaJpeg` symbol, JPEG feature, module,
re-export, legacy cache key, kernel variant, or JPEG PTX project.

Shared CUDA-Oxide staging is not duplicated: the unpublished
`j2k-cuda-build-support` crate owns template rendering, shared SIMT prelude
staging, toolchain invocation, real/placeholder PTX output, cfg metadata, and
strict-build behavior. Runtime and engine build scripts declare only their own
project inventories. The engine's non-Linux missing-build diagnostic is covered
after the move.

All 45 pre-existing JPEG implementation tests moved with their owner. The
runtime count changed from 333 to 288 and the JPEG engine count changed from 1
to 46, preserving the aggregate 334 implementation tests while adding no test
deletion. CUDA hardware and the CUDA-Oxide Linux toolchain remain unavailable
on this macOS host, so real PTX compilation/launch is not claimed.

| Command | Result |
|---|---|
| C1.2 ownership exit gate before extraction | failed as expected: runtime still owned JPEG kernel features |
| `cargo check -p j2k-cuda-jpeg-engine --all-features` | pass; expected non-Linux placeholder warnings |
| `cargo check -p j2k-jpeg-cuda --all-targets --features cuda-runtime` | pass |
| `cargo test -p j2k-cuda-jpeg-engine --all-features` | pass: 46/46 plus doc tests |
| `cargo test -p j2k-cuda-runtime --all-features` | pass: 288 library + 3 external tests plus doc tests |
| `cargo test -p j2k-jpeg-cuda --all-features` | pass: 50 library + 7 encode + 41 host-surface tests |
| `cargo clippy -p j2k-cuda-build-support -p j2k-cuda-runtime -p j2k-cuda-jpeg-engine --all-features --lib -- -D warnings` | pass |
| `cargo test -p xtask --test repo_lint repo_lint_support::architecture_policy` | pass: 6/6 |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy` | pass: 11/11 |
| `cargo fmt --all -- --check` | pass |
| `git diff --check` | pass |

## C1.3 Initial J2K Engine and J2K-ML Slice Evidence

The unpublished `j2k-cuda-j2k-engine` establishes the same one-pointer borrowed
context identity used by the completed JPEG engine. `j2k-cuda` now forwards its
J2K/HTJ2K kernel features through that engine and routes session classic/HTJ2K
table/resource uploads through its operation surface. Low-level context, pool,
buffer, error, diagnostics, and copy-kernel types remain runtime-owned.

J2K-ML is the first complete vertical slice: its four domain types, validation,
raw-pointer/context checks, synchronous generic-kernel launch, missing-build
diagnostic, test, feature, built cfg, build script inventory, and CUDA-Oxide
project moved to the J2K engine. The runtime no longer contains any J2K-ML
symbol, feature, source, PTX accessor, cache key, or closed kernel enum variant.
The existing runtime ML test moved with the implementation; the runtime suite
changed 288→287 while the new J2K engine suite changed 1→2.

The only added runtime capability is codec-neutral
`CudaContext::validate_device_pointer`, externally signature-tested and backed
by the existing direct/stream-ordered pointer provenance path. Real J2K-ML PTX
compilation and device launch remain unverified because this is a macOS host.

| Command | Result |
|---|---|
| C1.3 J2K engine dependency gate before scaffold | failed as expected: engine manifest absent |
| C1.3 J2K-ML ownership gate before move | failed as expected: runtime feature/source present |
| `cargo check -p j2k-cuda-j2k-engine --all-features` | pass; expected non-Linux ML placeholder warning |
| `cargo test -p j2k-cuda-j2k-engine --all-features` | pass: 2/2 plus doc tests |
| `cargo test -p j2k-cuda-runtime --all-features` | pass: 287 library + 3 external tests plus doc tests |
| `cargo check -p j2k-cuda --all-targets --features cuda-runtime` | pass |
| `cargo clippy -p j2k-cuda-runtime -p j2k-cuda-j2k-engine --all-features --lib -- -D warnings` | pass |
| `cargo test -p xtask --test repo_lint repo_lint_support::architecture_policy` | pass: 8/8 |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy` | pass: 11/11 |
| `cargo fmt --all -- --check` | pass |
| `git diff --check` | pass |

## C1.3 Classic Tier-1 Engine Extraction Evidence

The second J2K-engine vertical slice moves the complete classic Tier-1 family:
public and kernel ABI types, compile-time padding proofs and byte views, host
validation, static MQ/context tables, pooled coefficient allocation,
synchronous and queued launches, deferred status interpretation, all nine
pre-existing classic host/device tests, feature/build policy, and the
CUDA-Oxide source project. `j2k-cuda` component allocation, synchronous batch,
queued batch, session table upload, and parity tests now enter through
`J2kCudaEngine`.

The low-level runtime gained a codec-neutral queued compiled-kernel primitive.
`CudaQueuedExecution` owns pooled resources behind the existing reuse hold,
synchronizes on ordinary finish or Drop, returns resources for deferred
readback, and permits release without a second synchronization only after an
unsafe caller proves event-ordered context completion. Ownership/disjointness,
D32 memset, payload access, and diagnostics accessors expose existing runtime
semantics without exposing the private reuse guard. The runtime search contains
no classic feature, module, re-export, cache key, kernel variant, build flag, or
PTX project; only the transitional classic-only payload constructors remain in
the still-runtime-owned HTJ2K resource family.

Runtime library tests changed from 287 to 278 because nine classic tests moved;
the engine changed from 2 to 12 because those nine tests moved and the ABI
layout assertions gained a focused test owner. No test was deleted. CUDA
hardware and Linux CUDA-Oxide compilation remain unavailable on this macOS
host, so the adapter parity target exercised its availability gate but a real
classic kernel launch is not claimed.

| Command | Result |
|---|---|
| Queued low-level API regression before implementation | failed as expected: queued launch and retained-resource completion APIs absent |
| Classic ownership policy before extraction | failed as expected: runtime still owned the feature and sources |
| `cargo test -p j2k-cuda-runtime --test low_level_kernel_api --no-default-features` | pass: 3/3 |
| `cargo test -p j2k-cuda-j2k-engine --all-features` | pass: 12/12 plus doc tests; expected non-Linux PTX skips |
| `cargo test -p j2k-cuda-runtime --all-features` | pass: 278 library + 3 external tests plus doc tests |
| `cargo check -p j2k-cuda --all-targets --features cuda-runtime` | pass |
| `cargo test -p j2k-cuda --test classic_tier1_parity --features cuda-runtime` | pass: 1/1 availability-gated test |
| `cargo clippy -p j2k-cuda-runtime -p j2k-cuda-j2k-engine --all-features --lib -- -D warnings` | pass |
| `cargo test -p xtask --test repo_lint architecture_policy` | pass: 9/9 |
| `cargo test -p xtask --test repo_lint source_size_policy` | pass: 11/11 |
| `cargo fmt --all -- --check` | pass |
| `git diff --check` | pass |

## C1.3 HTJ2K Decode and Dequantization Engine Extraction Evidence

The third J2K-engine vertical slice moves HTJ2K decode and J2K dequantization
as one lifecycle family: public and kernel ABI types, compile-time layout
proofs, payload/table resources, output-region validation, cleanup/refinement
kernel selection, synchronous and queued launch paths, shared status groups,
deferred dequantization, completion/status interpretation, feature/build policy,
kernel inventory, and both CUDA-Oxide projects. Adapter decode, resident,
chunked-cleanup, pending-completion, session, and reconstruction paths now bind
these operations through `J2kCudaEngine`.

Runtime library tests changed from 278 to 245 as HT-owned tests moved. The J2K
engine suite increased from 12 to 43 and retains the moved ABI, kernel-source,
geometry, output-overlap, resource-reuse, empty-work, selection, dequantization,
and queued-completion coverage. The runtime search contains no HTJ2K decode or
J2K dequantize feature, module, re-export, cache key, kernel variant, build
flag, PTX project, or test symbol.

| Command | Result |
|---|---|
| HTJ2K ownership policy before extraction | failed as expected: runtime owned both features and source families |
| `cargo test -p j2k-cuda-runtime --lib --all-features` | pass: 245/245 |
| `cargo test -p j2k-cuda-runtime --test low_level_kernel_api --all-features` | pass: 3/3 |
| `cargo test -p j2k-cuda-j2k-engine --lib --all-features` | pass: 43/43 |
| `cargo test -p j2k-cuda --lib --features cuda-runtime` | pass: 164/164 |
| `cargo test -p j2k-cuda --test htj2k_decode_reconstruction --features cuda-runtime` | pass: 2/2; availability-gated on this host |
| architecture policy | pass: 10/10 |
| source-size policy | pass: 11/11 |
| runtime/J2K-engine all-feature strict library clippy | pass |
| `cargo fmt --all -- --check` | pass |
| `git diff --check` | pass |

Linux CUDA-Oxide compilation and NVIDIA kernel execution were unavailable on
macOS. The successful checks prove host ownership, feature wiring, no-GPU
compilation, preserved API call sites, and availability-gated behavior; they do
not claim device execution or performance.

## C1 Completion Evidence

The low-level CUDA runtime now has no JPEG, J2K, HTJ2K, ML, or transcode
module, feature, public type, cache variant, entrypoint inventory, build flag,
test, or PTX project. Codec adapters bind borrowed JPEG, J2K, or transcode
engines over the stable runtime context and allocation identities.

The final J2K slice moved the coupled IDWT/store/forward-transform/quantize,
classic encode, HT encode, compaction, and packetization family as one PTX
ownership unit. The final C1 slice moved all coefficient-domain transcode
models and execution into `j2k-cuda-transcode-engine`. Existing execution,
validation, ABI, and kernel metadata tests moved with their implementations.

| Command | Result |
|---|---|
| C1 transcode ownership regression before extraction | failed as expected: engine manifest absent |
| `cargo test -p j2k-cuda-j2k-engine --lib --all-features` | pass: 168/168 |
| `cargo test -p j2k-cuda-transcode-engine --lib --all-features` | pass: 8/8 |
| `cargo test -p j2k-cuda-runtime --lib --all-features` | pass: 103/103 |
| `cargo test -p j2k-cuda --lib --all-features` | pass: 164/164 |
| `cargo test -p j2k-transcode-cuda --lib --all-features` | pass: 23/23 |
| strict all-feature library clippy for the five affected crates | pass |
| C1 transcode ownership regression after extraction | pass |

CUDA-Oxide PTX compilation and NVIDIA execution were unavailable on the
aarch64 macOS host. Host compilation, feature forwarding, ownership, ABI,
validation, and unavailable-runtime behavior are verified; device execution
and performance remain for the Linux/NVIDIA lane.

## C2 Metal Runtime and Engine Boundary Evidence

`j2k-metal-support` remains the shared checked Metal runtime/resource owner.
The J2K-specific module tree is now private `j2k-metal::engine`, covering
transforms, Tier-1, packetization, stores, and resident encode/decode. Metal
transcode consumes the support layer and a neutral `DeviceCodestream` metadata
contract, not the public J2K Metal adapter.

| Command | Result |
|---|---|
| C2 transcode dependency regression before repair | failed as expected: manifest retained `j2k-metal` |
| C2 private-engine regression before rename | failed as expected: root still declared `mod compute` |
| `cargo test -p xtask --test repo_lint repo_lint_support::architecture_policy -- --nocapture` | pass: 14/14 |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy -- --nocapture` | pass: 11/11 |
| `cargo check -p j2k-metal --all-features --all-targets` | pass |
| `cargo test -p j2k-transcode-metal --all-features --test route_report` | pass: 6/6 |
| `cargo test -p j2k-transcode-metal --all-features --lib` | pass: 18/18 |
| changed-library all-feature strict clippy | pass: core, J2K Metal, Metal transcode |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features --lib --quiet -- --test-threads=1` | pass: 380 executed, 22 ignored |

## M5 Declaration-Oriented Accelerator Root Evidence

The J2K Metal root now delegates its four benchmark operations to private
`bench_support`; the JPEG Metal root delegates generic codec trait bodies to
`codec` and surface routing/fallback/upload behavior to `decode_surface`; and
the CUDA runtime root delegates its exported macro definitions to `macros`.
No ownership module is public. Existing root names, including doc-hidden bench
entry points and the public JPEG device-batch helper, remain re-exported.

The three roots fell from 206, 904, and 172 lines at the plan baseline to 108,
108, and 131 lines. The stale JPEG Metal large-root and long-root-function
allowances were removed. The structural regression failed before extraction
and now checks definition ownership rather than rejecting compatibility aliases.

| Command | Result |
|---|---|
| M5 root-ownership regression before extraction | failed as expected: J2K Metal root still defined benchmark operations |
| `cargo test -p xtask --test repo_lint source_size_policy -- --nocapture` | pass: 8/8 |
| `cargo check -p j2k-metal -p j2k-jpeg-metal -p j2k-cuda-runtime --all-features --all-targets` | pass; expected CUDA-Oxide Linux-build skips on macOS |
| `cargo clippy -p j2k-metal -p j2k-jpeg-metal -p j2k-cuda-runtime --all-features --lib --no-deps -- -D warnings` | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast` | pass: 228 library tests, 37 integration tests, and doc tests |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-metal --all-features --lib --quiet -- --test-threads=1` | pass: 380 executed, 22 ignored |
| `cargo test -p j2k-cuda-runtime --all-features --no-fail-fast --quiet` | pass: 333/333 |
| `git diff --check` | pass |

## M3 JPEG Metal Compute-Ownership Evidence

The 584-line `compute.rs` is now a 193-line domain root. Checked command-buffer
and encoder creation/completion live in `command.rs`; shader composition and 53
immutable pipeline states live in `pipeline_registry.rs`; device, queue,
default-session initialization, scratch, and viewport cache state live in
`runtime.rs`. Pipeline fields are accessed through `runtime.pipelines`, making
the immutable/mutable ownership split visible at use sites.

Existing entropy, pack, batch, region, texture/single-decode, status, and encode
modules retain their ABI validation, bindings, dispatch geometry, and status
interpretation. Crate-level callers now use named batch-entry, single-decode,
encode, or viewport modules. The shader-integrity test follows the registry
source owner rather than the deleted `compute.rs`.

| Command | Result |
|---|---|
| M3 source-boundary regression before extraction | failed as expected: runtime owners absent |
| `cargo check -p j2k-jpeg-metal --all-features --all-targets` | pass |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy:: -- --nocapture` | pass: 6/6 |
| `cargo clippy -p j2k-jpeg-metal --all-features --lib --no-deps -- -D warnings` | pass |
| `J2K_REQUIRE_METAL_RUNTIME=1 cargo test -p j2k-jpeg-metal --all-features --no-fail-fast` | pass: 228 library tests, 37 integration tests, and doc tests |
| `git diff --check` | pass |

## M2 Native Color Module-Boundary Evidence

The former 910-line `color.rs` is now a 65-line module root. Public color and
bitmap values, container/channel metadata, decoded output planes, retained
allocation accounting, shared packing, palette expansion, ICC ownership,
sYCC, and CIE Lab each have one focused owner. A7's `packing.rs` remains the
single sample-conversion policy owner.

The component-plane facade boundary now returns named structs rather than
positional tuples. Tests prove that native sample buffers and ICC profiles are
moved without copying and that actual capacities remain intact. Full native and
facade suites preserve color conversion, palette/channel behavior, high-bit and
signed samples, ROI packing, allocation classifications, and parity outputs.

| Command | Result |
|---|---|
| M2 source-boundary regression before the move | failed as expected: ownership modules absent |
| Focused `color::` library tests | pass: 17/17 |
| `cargo test -p xtask --test repo_lint repo_lint_support::source_size_policy:: -- --nocapture` | pass: 5/5 |
| `cargo clippy -p j2k-native -p j2k --all-features --lib --no-deps -- -D warnings` | pass |
| `cargo test -p j2k-native --all-features --lib` | pass: 636 executed, 2 ignored |
| `cargo test -p j2k-native --all-features --test component_planes` | pass: 24/24 |
| `cargo test -p j2k --all-features --no-fail-fast` | pass across all facade targets |
