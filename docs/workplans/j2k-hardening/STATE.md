# Current State

Plan anchor: J2K-HARDENING-2026-08-18
Audit baseline: f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5
Current HEAD: f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5
Current branch: main
Current task ID: Gate8-final-checkpoint
Current phase: final architecture checkpoint creation
Status: active on 2026-08-21; the user authorized two local commits with no
push, tag, publish, or deployment

## Completed Since Last Checkpoint

- Selected the user-approved 0.10.0 pre-1.0 transition instead of reversing
  architecture boundaries with incompatible 0.9.1 shims. Updated the workspace
  version, 97 exact internal pins, the root and five fuzz lockfiles, release
  prose, and current-version tests.
- Generated and validated the 0.10.0 public-API report against published
  `v0.9.0`. Its review covers all 22 libraries and exactly 20 removed signatures
  in three source-break ledgers with direct migration guidance.
- Closed the changed-line coverage gates after correcting lane ownership and
  hardware-specific accumulation: host 85.78% overall / 92.31% critical,
  Metal 81.61% critical, and CUDA 80.68% critical.
- Completed development T.803 on CPU, Metal, and CUDA: every decoder lane passed
  160/160 selected cases, with encoder matrices 56/56 on CPU and 35/35 on each
  accelerator.
- Completed G0 and persisted the canonical plan, baseline evidence, environment,
  validation, conformance, and available performance measurements.
- Completed G1.1 semantic dependency-edge rules with one exact migration ratchet.
- Completed G1.2 prepared-plan type-erasure occurrence ratchets.
- Completed G1.3 host phase-budget definition ratchets.
- Completed G1.4 reviewed source-size, long-root-function, and
  `too_many_lines` ceilings.
- Extended G1.5 production Rust clone detection to 12-line blocks and added a
  dedicated Metal/OpenCL-tokenized lane covering 28 staged shaders.
- Completed G1.5 route-threshold and duplicated error-taxonomy ratchets.
- Completed G1.6 Metal decode-stage and CUDA transcode DWT97 harnesses.
- Completed G1.7 production kill-switch enforcement and Gate 1.
- Completed A1 with one backend-neutral encode-geometry owner in
  `j2k-types`, migrated facade/native/Metal/CUDA-runtime/transcode consumers,
  preserved the codec-math compatibility API, and added an ownership ratchet.
- Moved packet-order implementation into the shared owner and lowered the
  reviewed `j2k-types` crate-root ceiling from 1,183 to 1,125 lines.
- Preserved the semantics and tests of the five pre-existing native Tier-1/MQ
  changes; only strict-lint mechanical conversions were applied in `mq.rs`.
- Completed A2 by replacing prepared-plan `Any` storage and adapter downcasts
  with neutral `j2k-types` decode-plan contracts and immutable facade borrows.
- Migrated facade tests, CUDA execution/tests, Metal plan caches, and Metal
  execution to the typed prepared-plan contract; the completion search is empty.
- Added pointer-identity assertions proving cloned plan wrappers share one
  geometry owner, while existing range tests prove compressed-source retention.
- Completed A3 with one `j2k-core::HostPhaseBudget` and one neutral
  `HostPhaseError`; removed all four backend-local implementations.
- Migrated J2K CUDA, JPEG CUDA, CUDA runtime, and CUDA transcode while
  preserving their public error classifications through narrow `From` impls.
- Completed A4 by routing full, region, scaled, and region-scaled requests
  through one typed decode-operation entrypoint in CUDA and Metal.
- CUDA tile submission now uses the same `DeviceDecodeRequest` model and one
  tile operation entrypoint; CPU-staged CUDA paths share one CPU executor.
- Completed A5 with one JPEG Metal RGB8 batch builder consuming normalized raw
  or prepared sources and owning all validation, owner, cache, and metadata
  accounting decisions.
- Added raw/prepared parity coverage for full, scaled, region-scaled, mixed
  dimensions, mixed sampling, restart restrictions, duplicate owners, and
  cache-retained bytes.
- Completed A6 with one borrowed `PreparedImageGeometry` contract shared by
  Classic and HT plans while leaving their codec-specific payload APIs and
  public enum field layouts unchanged.
- Moved component classification, tile emptiness, single-tile component access,
  full/output geometry, and uniform-wavelet selection to that shared owner.
- Completed A7 by extracting `color/packing.rs` as the single owner of bit-depth
  equality, scaling, rounding, 8-bit quantization, output bounds, and checked
  full/ROI traversal.
- Preserved the specialized one-, two-, three-, and four-component direct 8-bit
  loops and added a full-versus-ROI matrix for mixed/high/signed inputs.
- Completed A8 with one shared `CapabilityRejection` taxonomy covering format,
  sampling, precision, operations, plans, containers, geometry, resources,
  contexts, and checked contract violations.
- Migrated all production `UnsupportedCudaRequest`/`UnsupportedMetalRequest`
  construction in the four CUDA/Metal J2K/JPEG adapters through typed reasons,
  retaining the existing public error variants and exact messages.
- Completed M1 by replacing the 1,125-line `j2k-types` crate-root monolith with
  an 85-line declaration, documentation, and compatibility re-export surface.
- Moved transform, Tier-1, packetization, prepared-plan, dispatch-report, and
  stage-error ownership into focused modules; progression behavior and all
  move-only compile-time assertions remain with their owning value families.
- Classified the existing semver-visible accelerator contract as the stable
  low-level `dispatch` SPI instead of continuing to call it experimentation.
- Completed M2 by replacing the 910-line native color file with a 65-line
  module root and focused type, metadata, output-plane, allocation, packing,
  palette, ICC, sYCC, and CIE Lab owners.
- Replaced the doc-hidden tuple-shaped component handoff aliases and aggregate
  tuples with named structs, migrated the facade, and preserved allocation-free
  moves of pixel and ICC owners.
- Completed M3 by replacing the 584-line JPEG Metal `compute.rs` with a
  193-line domain root plus separate command, pipeline-registry, and runtime
  owners.
- Nested all 53 immutable pipeline states under one registry, moved shader
  compilation/integrity ownership with it, and kept device/session/cache state
  in `runtime.rs`.
- Migrated crate callers to batch-entry, single-decode, encode, and viewport
  domains while preserving every Metal execution path and runtime result.
- Completed M4 by replacing the 1,018-line `codec_batch.rs` with a 27-line
  semantic root and focused request, source, inspect, plan, accounting,
  buffer-target, texture-target, and submit owners.
- Kept `batch.rs` as the distinct queue/group/flush/completion execution owner;
  raw and prepared sources still share exactly one normalized planner.
- Completed M5 by extracting J2K Metal benchmark operations, JPEG Metal codec
  trait implementations and decode/upload routing, and CUDA runtime macros
  from their crate roots.
- Reduced the three operational roots to 108, 108, and 131 lines while
  preserving existing root entry points as controlled compatibility re-exports.
- Completed M6 by replacing the 672-line JPEG capability monolith with focused
  request, output-geometry, CPU, CUDA, Metal, rejection, and resolution owners.
- Kept backend eligibility as correctness policy and documented that `Auto`
  performance promotion remains outside the capability resolver.
- Completed M7 by replacing the 562-line facade encode file with a 49-line
  module root and thin stable API delegates.
- Separated CPU execution, accelerator stage routing, geometry, high-bit
  adaptation, lossless result construction, lossy targeting, ROI validation,
  and tests while preserving all public names and behavior.
- Completed M8 by partitioning `classic.metal` into nine ordered shader units
  while preserving the original 77,178 source bytes exactly.
- Isolated QE and context tables from handwritten control flow and removed the
  stale large-shader allowance; existing classic encode shader ownership was
  retained instead of creating an empty duplicate unit.
- Completed C1 migration step 1 with a codec-neutral static PTX/kernel spec,
  checked public launch geometry, typed parameter ABI contract, and generic
  cached synchronous launch primitive.
- Proved the low-level kernel surface from an external integration test under
  both no-feature and all-feature runtime configurations.
- Added the private `j2k-cuda-jpeg-engine` crate with dependency direction into
  the runtime and a borrowed operation boundary preserving `CudaContext` type
  identity.
- Migrated `j2k-jpeg-cuda` domain imports and all six runtime operation call
  families through the JPEG engine before moving their implementations.
- Completed the JPEG-family extraction: the engine now owns every JPEG ABI,
  validation, allocation, byte-view, diagnostic, encode/decode, PTX, build-flag,
  and launch responsibility, while the runtime has no JPEG codec concept.
- Extracted shared CUDA-Oxide project staging and PTX packaging into the private
  `j2k-cuda-build-support` crate so subsequent engines can own kernel projects
  without copying build mechanics.
- Moved all 45 JPEG runtime tests with their implementation: the runtime suite
  is now 288 library plus 3 external tests and the engine suite is 46 tests.
- Added the private borrowed `j2k-cuda-j2k-engine` boundary, routed adapter
  feature ownership and session table/resource uploads through it, and retained
  the public `CudaContext` identity.
- Completed the first J2K-engine vertical slice by moving J2K-ML types,
  validation, launch, missing-build policy, test, feature, and CUDA-Oxide project;
  the runtime now has no J2K-ML codec concept.
- Added codec-neutral raw device-pointer validation to the low-level runtime SPI
  and proved its external signature alongside kernel launch and memset.
- Added codec-neutral queued compiled-kernel submission with pooled-resource
  retention, synchronized resource recovery, and an externally-completed path
  for event-timed engines.
- Completed the classic Tier-1 vertical slice: ABI, byte views, validation,
  table ownership, sync/queued orchestration, tests, feature, and CUDA-Oxide
  project now belong to `j2k-cuda-j2k-engine`.
- Removed every classic feature, module, re-export, legacy cache key, kernel
  variant, build flag, and PTX project from the low-level runtime; adapter
  component, completion, session, and parity call sites bind through
  `J2kCudaEngine`.
- Completed the HTJ2K decode/dequantization vertical slice: ABI and byte views,
  payload/table resources, output-region validation, sync/queued completion,
  status groups, kernel selection, tests, features, build flags, and both
  CUDA-Oxide projects now belong to `j2k-cuda-j2k-engine`.
- Removed every HTJ2K-decode/dequantize feature, module, re-export, cache key,
  kernel variant, build flag, PTX project, and test symbol from the low-level
  runtime while preserving its codec-neutral launch/memory lifecycle SPI.
- Completed the remaining J2K-family extraction: IDWT, inverse MCT, final
  stores, forward transforms, quantization, classic/HT encode, compaction,
  packetization, tests, features, and four PTX projects now belong to
  `j2k-cuda-j2k-engine`.
- Completed C1 with the private borrowed `j2k-cuda-transcode-engine`; moved all
  transcode band models, validation, launch geometry, timings, orchestration,
  tests, feature policy, and the CUDA-Oxide project out of the runtime.
- Reduced `j2k-cuda-runtime` to codec-neutral driver/context, checked module
  launch, streams/events, completion, memory/pools, pinned staging,
  diagnostics, and external-allocation validation.
- Completed C2 by renaming the J2K Metal compute ownership layer to the private
  `engine` boundary while retaining `j2k-metal-support` as the shared runtime,
  resource, dispatch, completion, and allocation owner.
- Added backend-neutral `DeviceCodestream` metadata and implemented it for
  `MetalEncodedJ2k`, allowing `j2k-transcode-metal` to remove its full
  `j2k-metal` dependency while preserving resident handoff behavior.
- Completed R1 with separate CUDA/Metal routing domains and deterministic
  promotion tables generated from a validated checked-in evidence manifest.
- Removed unverified JPEG Metal Auto batch and J2K Metal Auto region-scaled
  batch promotions, and restricted resident HTJ2K host output to the exact
  benchmark-qualified cells.
- Completed P0 with a validator-owned performance experiment record and a
  documented mandatory GPU workload/metrics/acceptance matrix.
- Completed P1 as a measured rejection: exact fused 5/3 IDWT regressed both
  resident and readback end-to-end workloads, so all experiment code was removed.
- Completed P2 as a measured rejection: exact fused 9/7 axis kernels provided
  only a small resident improvement, inconclusive readback, and disproportionate
  whole-axis threadgroup complexity, so all candidate code was removed.
- Completed P3 as a measured rejection: the exact line-serial 9/7 forward
  transform reduced dispatches but significantly regressed full encode, so all
  candidate code was removed and the reusable transform benchmark retained.
- Completed P4 as a design-preflight rejection: the tail already fuses MCT and
  store, while fusing final vertical synthesis would require disproportionate
  cross-component ownership and transform duplication without stage timing.
- Completed and promoted Metal P5 combined RGB deinterleave, level shift, and
  RCT/ICT with exact full-range semantics, fallback, and supported end-to-end gains.
- Completed P6's tooling audit and validated the unavailable compiler-resource
  inventory. P7–P10 redesigns are blocked rather than inferred from source
  scratch arrays; existing production paths remain unchanged.
- Completed P11 as a measured rejection. The exact cooperative packetizer
  regressed Classic by 4.71–5.40% and HT by 24.67–26.16%, so the candidate,
  switch, and prototype-only tests were removed while the benchmark remained.
- Completed P12 as a measured rejection. The isolated terminal stage improved,
  but the priority JPEG-to-HTJ2K product interval crossed no change; candidate
  code and switch were removed while the product benchmark and fallback remain.
- Completed P13 as an RTX-measured rejection. Exact stage/product hashes and
  all-output decode parity held, but priority absolute intervals overlapped and
  isolated column-plus-quantize regressed; candidate code and switch were removed.
- Completed P14 as an RTX-measured rejection. The exact tiled route won wide
  batch 1 but regressed required wide batch 16 by 21–40%; candidate code and
  switch were removed while generic-wide and bounded cooperative routes remain.
- Completed P15 as an RTX-measured rejection. Shared staging preserved exact
  coefficients and product decodes and reduced statically counted source loads,
  but priority full-encode absolute intervals overlapped, wide batch 16 crossed
  no change, and 512 batch 1 regressed; candidate code and switch were removed.
- Completed P16 as an RTX-measured rejection. The exact fused RGB input route reduced
  two physical input dispatches to one and substantially improved isolated RCT
  and ICT stages, but both HTJ2K product absolute intervals overlapped and the
  ICT point estimate regressed. The specialized kernel, selector, switch, and
  candidate counters were removed; compatibility methods use separate stages.
- Completed P17 with an RTX-measured profile-preflight NO-GO. All eight exact
  Classic/HT, 5/3/9/7, batch-1/16 cells placed final vertical plus store at
  0.641-1.784% of resident probe wall, below the 10% prototype gate. No
  candidate, switch, A/B, or experiment JSON was created; the reusable profiler
  and harness remain. Two prerequisite irreversible half-tie store defects were
  repaired and verified on RTX before the final matrix.
- Completed and promoted Metal P18 staged JPEG encoding with exact required
  matrix coverage and a 48.96–49.06% representative batch improvement.
- Completed and promoted CUDA P18 staged JPEG encoding. The priority RTX
  512x512 batch-8 workload improved 95.044-95.062% with exact five-cell A/B,
  dual decoding, pinned 16-frame matrix digests, and restart-marker coverage.
  The serial route and switch were removed after the sole staged route passed
  post-cleanup RTX verification.
- Repaired the repository JPEG bit reader at exact restart boundaries: marker
  probing now accepts one to seven legal buffered pad bits without weakening
  eight-bit stuffed-data or wrong-marker checks. The red regression, bit-reader
  17/17, full JPEG suite, and RTX restart-16 batch 1/8 passed.
- Completed Metal P19 as a measured rejection and removed its production
  candidate/switch after an inconclusive interval and 12.7 MB scratch cost.
- Replaced the superseded combined CUDA blocker with truthful P19-only history;
  the direct RTX lane is available and P19 CUDA decode is complete.
- Completed CUDA P19 profile-first work. Promoted and cleaned up adaptive
  checkpoint packing (one thread per block below 128 checkpoints, 128 threads
  per block at and above 128) after exact ten-cell RTX A/B; all seven
  geometry-changing cells improved, including separated priority absolute CIs.
- Repaired the CUDA device-capability predicate exposed by restart profiling.
  Device eligibility now accepts restart-coded fast 4:2:0 packets while the
  CPU-only `matches_fast_tile_shape` predicate and CPU routing remain unchanged.
- Prototyped exact 4:2:0 entropy/coefficient and parallel-IDCT defusion only
  after the settled fused profile justified it. Every eligible product cell
  regressed 3.00–51.47%; the scratch route, kernels, switch, and split-only
  tests were removed, and the exact pre-candidate PTX was restored.

## Files Changed

- Durable workplan: `docs/workplans/j2k-hardening/{MASTER_PLAN,STATE,DECISIONS,EVIDENCE}.md`.
- Dependency/source guardrails: `xtask/tests/repo_lint_support/` policy modules
  and module registration.
- Clone coverage: `.jscpd.json`, new `.jscpd-metal.json`, and
  `xtask/src/clone_audit.rs` plus its config, report, staging, and tests.
- Stage harnesses and manifests in `j2k-metal` and `j2k-transcode-cuda`.
- Reusable Metal/CUDA transform, packetization, JPEG encode, and JPEG decode
  benchmarks plus 17 validated P1–P19 experiment, rejection, promotion, and
  limitation records.
- Shared encode geometry in `j2k-types`, with consumer migrations in the
  facade, native encoder, Metal encoder, CUDA runtime, and Metal transcode.
- Typed prepared-plan facade, CUDA/Metal consumers and tests, plus the completed
  no-type-erasure repository policy.
- Compatibility/dependency updates in `j2k-codec-math`, affected manifests,
  `Cargo.lock`, and `docs/architecture.md`.
- `j2k-types` ownership modules under `src/{transform,tier1,packetization,
  dispatch,prepared_plan}` plus the re-export-oriented crate root.
- `j2k-native/src/color/` ownership modules, native named handoff contracts,
  and the facade component-handoff consumer.
- JPEG Metal compute root/runtime/registry/command modules, pipeline-field
  accessors, domain callers, and shader-integrity source ownership.
- J2K Metal `bench_support`, JPEG Metal `codec`/`decode_surface`, and CUDA
  runtime `macros` owners plus declaration-oriented crate roots.
- JPEG capability modules under `j2k-jpeg/src/capabilities/` and their
  source-ownership regression.
- Facade encode owners under `j2k/src/encode/`, including renamed CPU and
  accelerator modules and the delegated public API.
- Classic Metal shader units under `j2k-metal/src/classic/`, the updated host
  composer/integrity inventory, and byte-equivalence/source guardrails.
- CUDA runtime low-level `kernel.rs`, generalized compiled-module cache,
  checked launch geometry/API exports, and external API regression.
- CUDA JPEG engine scaffold, adapter manifest/import/call-site migration,
  workspace graph, lockfile, and current architecture documentation.
- CUDA J2K engine classic Tier-1 sources/project, codec-neutral queued runtime
  SPI, adapter classic call sites, ownership guardrail, and architecture graph.
- CUDA J2K transform/store/encode families and the private CUDA transcode
  engine, adapter migration, PTX ownership, tests, and C1 exit guardrail.
- J2K Metal private engine boundary, core resident-codestream contract,
  transcode dependency repair, path-policy updates, and C2 architecture gates.
- Routing domain modules, generated CUDA/Metal promotion tables,
  `docs/routing-promotion-evidence.json`, promotion codegen, and routing policy.
- P0 experiment validator, command registration, tests, and
  `docs/performance-experiments/README.md`.
- Pre-existing user files under `crates/j2k-native/src/j2c/` remain modified but
  untouched by this work.

## Tests and Static Checks

| Command | Result | Notes |
|---|---|---|
| `cargo xtask fmt` | passed | Current guardrail/clone changes formatted. |
| `cargo clippy -p xtask --all-targets --no-deps -- -D warnings` | passed | New xtask and repo-lint code warning-free. |
| `cargo test -p xtask --test repo_lint -- --nocapture` | baseline failure only | 68/69 pass; only pre-existing undocumented WSI SVS corpus-path variable fails. |
| `cargo test -p xtask clone_audit:: -- --nocapture` | passed | 11 clone-audit unit tests. |
| `cargo xtask clone-audit` | passed | Production Rust, Metal shader, and test Rust lanes pass. |
| Metal `decode_stages` benchmark build/run | passed | Resident/readback paths and all required dispatch stages observed. |
| CUDA transcode `dwt97` runtime-feature build | passed | Linux-only CUDA kernels skipped on macOS; execution unavailable. |
| `cargo xtask clippy-strict` | passed | Canonical strict library lane after G1.1. |
| `cargo xtask test` | passed | Full workspace, docs, allocation probe, Metal, and downstream-example suite after A1. |
| Focused A1 unit/integration tests | passed | Required geometry matrix, facade/Metal parity, COD markers, native/CUDA DWT validation, code-block validation, and transcode geometry. |
| Changed-crate library clippy with `-D warnings` | passed | `j2k-types`, codec math, Metal, CUDA runtime, and Metal transcode libraries. |
| `cargo xtask clone-audit` | passed | Production 4.31%, Metal 25.08%, tests 2.72%. |
| CPU T.803 suite `all --development` | passed | Decoder 160/160; encoder 56/56. |
| Metal T.803 suite `all --development` | passed | Decoder 160/160; encoder 35/35; 81 hybrid routes. |
| `cargo test -p j2k --test owned_batch` | passed | 34/34 typed-plan, payload, ownership, ROI/reduction, RGB/RGBA, and reuse tests. |
| `cargo test -p j2k-cuda --all-features --no-fail-fast` | passed | 271 tests across unit/integration/doc targets on the non-CUDA host. |
| `cargo test -p xtask --test repo_lint repo_lint_support::prepared_plan_policy::` | passed | 2/2; prepared-plan type-erasure inventory is empty. |
| A2 changed-library clippy | passed | Facade, CUDA, and Metal libraries pass `-D warnings`. |
| Full Metal all-feature suite | unrelated failure | 101/102 `device` tests passed; `independent_openht_sigprop_overlap_matches_openht_oracle_within_one_lsb` failed in the pre-existing user-modified Tier-1/MQ surface. Other Metal targets and all A2 plan/cache tests passed. |
| A3 cross-crate allocation tests | passed | Core 15, J2K CUDA 13, CUDA runtime 24, JPEG CUDA 8, CUDA transcode 5. |
| A3 allocation ownership policy | passed | Exactly one `HostPhaseBudget` definition under `j2k-core`. |
| A3 changed-library clippy | passed | Five affected libraries pass all-feature `-D warnings`. |
| CUDA host-surface operation matrix | passed | 37/37, including CPU/CUDA/Auto, formats, ROI, scaling, unavailable runtime, and residency. |
| C1 final engine/runtime suites | passed | J2K engine 168/168, transcode engine 8/8, runtime 103/103, J2K adapter 164/164, transcode adapter 23/23. |
| C1 final changed-library clippy | passed | Runtime, J2K engine, transcode engine, J2K adapter, and transcode adapter all pass all-feature library `-D warnings`. |
| C2 architecture and source policies | passed | 14/14 dependency/ownership tests and 11/11 source-size tests. |
| C2 Metal validation | passed | J2K Metal all-target check, transcode route 6/6, transcode library 18/18, strict changed-library clippy, and required-runtime J2K Metal 380 passed/22 ignored. |
| R1 generator and routing policy | passed | Codegen 4/4, stale check, and routing policy 3/3. |
| R1 affected adapter libraries | passed | CUDA 164/164; J2K Metal 377 passed/22 ignored; JPEG Metal 228/228. |
| R1 strict clippy | passed | xtask all targets and CUDA/J2K Metal/JPEG Metal all-feature libraries with `-D warnings`. |
| Full repository policy after R1 | baseline failures | 86/93 pass; seven stale policies from earlier engine/facade moves remain for the subsequent plan-wide cleanup. |
| P0 validator tests and policy | passed | Validator 4/4 and benchmark-framework architecture gate 1/1. |
| P0 xtask strict clippy | passed | All targets with `-D warnings`. |
| P1 exact boundary/runtime parity | passed | Fused/unfused axes through 2592, native parity, and repeated hybrid batch before cleanup. |
| P1 same-host A/B | rejected | Resident +5.02% to +6.94%; readback +4.52% to +6.72%; p=0.00. |
| P1 cleanup validation | passed | Rejection record validates; J2K Metal 377 passed/22 ignored, strict library clippy, and `git diff --check`. |
| P1 all-target direct clippy | baseline invocation failure | Raw command lacks the repository's canonical test/bench lint allowances and reports 193 existing disallowed allocation uses; library clippy and `cargo xtask metal-compile` use the established lanes. |
| P2 exact runtime parity | passed | Fused/fallback odd, singleton-origin, and 1023×767 outputs were bit exact before cleanup. |
| P2 same-host A/B | rejected | Resident -1.57% to -0.34%; readback -2.65% to +0.05%; 16 KiB threadgroup state and incomplete long-axis design. |
| P2 experiment record | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P2-metal-idwt97.json`. |
| P3 fallback/treatment exactness | passed | Fractional three-level output bits and native single/multi-level codestream bytes matched. |
| P3 same-host A/B | rejected | Full encode +1.23% to +2.38%, p=0.00; stage comparison noisy and treatment slower. |
| P3 experiment record | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P3-metal-fdwt97.json`. |
| P4 eligible-route probe | passed | Irreversible RGB8 repeated path reported IDWT, inverse-MCT, and final-store stages with the expected output SHA-256. |
| P4 design preflight | rejected | Existing MCT/store is fused; remaining vertical fusion requires a second cross-component graph and duplicated 5/3/9/7 terminal arithmetic without timing evidence. |
| P5 correctness and fallback | passed | Native fake hook, signed/unsigned 1-16-bit Metal matrix, RCT/ICT exactness, forced fallback, and end-to-end parity. |
| P5 same-host A/B | promoted | RCT -1.15% to -0.63%; ICT -1.14% to -0.55%; both p=0.00 with exact codestream hashes. |
| P5 record validation | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P5-metal-input-mct.json`. |
| P6 compiler-resource inventory | blocked and documented | Metal exposes timestamps but not per-kernel registers/private/occupancy/spill/cache metrics; no compatible CUDA lane was available at the P6 checkpoint, and later JPEG-specific P19 evidence does not backfill the HT/Classic inventory. |
| P6 record validation | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P6-private-memory.json`. |
| P7-P10 redesign gate | blocked | No cooperative Tier-1 code added without the P6 baseline; existing specialized/fallback paths retained. |
| P11 exact Metal parity | passed | Classic/HT, all five progression orders, inclusion/L-block/empty/multilayer cases, exact codestreams, and native decode. |
| P11 same-host A/B | rejected | Classic +4.71% to +5.40%; HT +24.67% to +26.16%; both p=0.00. |
| P11 record validation | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P11-metal-cooperative-packetization.json`. |
| P11 rejected-candidate cleanup | passed | Cooperative identifiers are absent; benchmark-harness 7/7, shader-integrity 4/4, strict library/bench clippy, and release-bench linking passed. |
| P12 exact Metal differential | passed | Tiny, odd/even, wide/tall, and truncated-code-block fused/fallback outputs match; fused handoffs 0 versus 4. |
| P12 product A/B | rejected | Isolated stage improved 5.04–7.13%, but priority product interval was -4.94% to +1.38%, p=0.36; candidate removed. |
| P12 record validation | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P12-metal-column-quantize.json`. |
| P13 CUDA column/quantize | rejected | Priority product absolute CIs overlapped; isolated column+quantize regressed 4.21–7.53%; exact candidate removed and schema-v2 record validates. |
| P14 CUDA wide IDWT | rejected | Wide batch 1 improved 50–60%, but required wide batch 16 regressed 21–40%; exact tiled candidate removed and record validates. |
| P15 CUDA FDWT97 shared staging | rejected | Priority full-encode absolute CIs overlapped; wide batch 16 crossed no change and 512 batch 1 regressed; exact candidate removed and record validates. |
| P16 CUDA input fusion exact/A-B | rejected | Native-oracle stage parity and all product parse/decode checks passed on RTX; isolated stages won, but both product absolute intervals overlapped and ICT's point estimate regressed. |
| P16 experiment record | passed | `cargo xtask gpu-experiment validate docs/performance-experiments/P16-cuda-input-fusion.json`. |
| P16 rejected-candidate cleanup | passed | Adapter 165/165, engine 172/172, repo-lint 99/99, all-feature check, benchmark build, strict library/benchmark clippy, format, diff, and symbol policy passed. |
| P16 CUDA input fusion | rejected and removed | Exact isolated stages improved, but both product absolute CIs overlapped and ICT's point estimate regressed. |
| P17 CUDA final store | NO-GO at preflight; complete | All eight exact cells passed, but final vertical plus store was only 0.641-1.784% of resident wall versus the 10% GO gate; no prototype, switch, A/B, or experiment JSON was created. |
| P18 Metal JPEG encode | promoted | 512x512 batch 8 improved 48.96–49.06% with exact required matrix and obsolete fused route removed. |
| P19 Metal JPEG decode | rejected | Split route interval -0.57% to +14.14%, p=0.30, with 12,681,216 bytes scratch; production candidate removed. |
| P18 CUDA JPEG encode | promoted and cleaned up | Priority 512x512 batch 8 improved 95.044-95.062%; exact matrix/restart coverage passed and the serial route/switch were removed. |
| P19 CUDA packed checkpoints | promoted and cleaned up | Adaptive block-1/block-128 routing preserved exact ten-cell hashes; the priority absolute CIs separated and all seven geometry-changing cells improved. |
| P19 CUDA coefficient/IDCT split | rejected and removed | Exact 4:2:0 i32-scratch route regressed every eligible cell by 3.00–51.47%; unchanged 4:2:2/4:4:4 controls were neutral. |
| P19 CUDA restart capability regression | passed | Device-only fast-4:2:0 eligibility accepts restart-coded packets; CPU eligibility is unchanged; the same packet profiled twice and decoded through the strict session batch route on RTX. |
| P19 CUDA post-cleanup | passed | Strict real CUDA-Oxide build; owned decode 8/8, pitched output 1/1, 4:2:2/4:4:4 1/1; release-bench compile; candidate-symbol/source/switch scans empty. |
| All performance records | passed | All 17 JSON files under `docs/performance-experiments/` validate. |
| Final repository policy | passed | `cargo test -p xtask --test repo_lint -- --nocapture`: 101/101. |
| Canonical Metal release lane | passed | `cargo xtask metal-compile`: strict library/all-target clippy, optimized integration/library tests, and doc tests. |
| Final formatting/diff | passed | `cargo fmt --all -- --check` and `git diff --check`. |
| Metal decoder operation tests | passed | 14/14 with required runtime, including requests, routing, sessions, region/scaling, and cache reuse. |
| A4 CUDA/Metal strict library clippy | passed | Both all-feature libraries pass `-D warnings`. |
| Full JPEG Metal all-feature suite | passed | 265 tests across library and integration targets with required Metal runtime; 228 library tests. |
| A5 parity and ownership tests | passed | 6 planner parity/cache tests plus duplicate-owner exact-cap tests. |
| A5 JPEG Metal strict library clippy | passed | All-feature library passes `-D warnings`. |
| Facade owned-batch suite | passed | 36/36 including shared grayscale/RGB image-geometry parity. |
| CUDA all-feature suite after A6 | passed | 273 tests across unit, integration, and doc targets on the non-CUDA host. |
| Metal prepared-plan matrix after A6 | passed | 47 tests plus 7 intentional runtime-gated ignores; Classic, HT, RGB, RGBA, cache, reuse, and resident paths. |
| A6 prepared-plan ownership policy | passed | 3/3, including one shared image-geometry owner. |
| A6 changed-library clippy | passed | Native, facade, CUDA, and Metal all-feature libraries pass `-D warnings`. |
| Native full library suite after A7 | passed | 636/636 executed, 2 intentional performance ignores. |
| Native component-plane integration suite | passed | 24/24 across signed, mixed, and high-bit formats. |
| A7 packing behavior matrix | passed | 7/7 boundary tests; 1-4 components, mixed/high/signed, full and edge/non-aligned ROI. |
| A7 packing ownership policy | passed | 2/2; one policy and both entrypoints in `color/packing.rs`. |
| A7 native/xtask strict clippy | passed | All-feature native library and all xtask targets pass `-D warnings`. |
| Core rejection tests | passed | 77 tests across unit/API; typed kind and exact rendered text matrix. |
| CUDA and JPEG CUDA all-feature suites after A8 | passed | Full non-CUDA-host unit, integration, and doc targets. |
| JPEG Metal full runtime suite after A8 | passed | 265 tests across all targets with required Metal runtime. |
| J2K Metal library suite serialized | passed | 380 executed, 22 intentional ignores. |
| J2K Metal full parallel suite | baseline plus concurrency flake | Known Tier-1/MQ device parity failure remains; hybrid global-cache count tests also race under parallel full-suite execution but pass individually and with `--test-threads=1`. |
| A8 taxonomy ownership policy | passed | 4/4; AST inventory proves zero direct production static-rejection constructors. |
| A8 five-library strict clippy | passed | Core and all four affected adapter libraries pass `-D warnings`. |
| Clone audit after A8 | passed | Production 4.27%, Metal 25.08%, tests 2.71%. |
| M1 structural red test | failed as expected | Required ownership modules were absent from the 1,125-line crate root. |
| `cargo test -p j2k-types --all-features` | passed | 18 unit tests plus doc tests after decomposition. |
| M1 source-size policy | passed | 4/4; root is 85 lines and the stale 1,125-line allowance is removed. |
| `cargo clippy -p j2k-types --all-features --all-targets --no-deps -- -D warnings` | passed | Decomposed crate and tests warning-free. |
| `cargo check --workspace --all-targets` | passed with baseline warnings | Every workspace target compiles; only two pre-existing dead-code warnings in `j2k-cuda` allocation helpers. |
| `cargo test -p j2k-native --all-features --lib` | passed | 636 executed, 2 intentional ignores. |
| `cargo test -p j2k --all-features --no-fail-fast` | passed | Full facade unit, integration, parity, owned-batch, encode/decode, and doc targets. |
| M2 structural red test | failed as expected | Color ownership modules and `color/mod.rs` were absent. |
| M2 focused color tests | passed | 17/17 across packing, metadata, ICC, palette, output ownership, sYCC, and CIE Lab. |
| M2 source-size policy | passed | 5/5; color root is 65 lines and native root remains at its 431-line ceiling. |
| M2 native/facade strict clippy | passed | Both all-feature libraries pass `-D warnings`. |
| `cargo test -p j2k-native --all-features --lib` after M2 | passed | 636 executed, 2 intentional ignores. |
| Native component-plane suite after M2 | passed | 24/24. |
| Full facade suite after M2 | passed | All unit, integration, parity, owned-batch, encode/decode, and doc targets. |
| M3 structural red test | failed as expected | Compute root still owned commands, shader source, pipeline registry, and runtime. |
| M3 source-size policy | passed | 6/6; compute root is 193 lines and existing crate-root ceilings did not increase. |
| JPEG Metal all-target check after M3 | passed | All feature/target combinations compile on macOS. |
| M3 JPEG Metal strict library clippy | passed | All-feature library passes `-D warnings`. |
| Full JPEG Metal runtime suite after M3 | passed | 228 library tests plus 37 integration tests and doc tests. |
| M4 structural red test | failed as expected | Codec-batch ownership modules were absent. |
| M4 source-size policy | passed | 7/7; codec-batch root is 27 lines. |
| M4 JPEG Metal check/clippy | passed | All targets compile; strict library clippy passes. |
| Full JPEG Metal runtime suite after M4 | passed | 228 library tests plus 37 integration tests and doc tests. |
| M5 structural red test | failed as expected | J2K Metal still owned benchmark functions before extraction. |
| M5 source-size policy | passed | 8/8; accelerator roots are 108, 108, and 131 lines and stale root allowances are removed. |
| M5 accelerator all-target check | passed | J2K Metal, JPEG Metal, and CUDA runtime compile with all features/targets. |
| M5 accelerator strict library clippy | passed | All three all-feature libraries pass `-D warnings`. |
| Full JPEG Metal runtime suite after M5 | passed | 228 library tests plus 37 integration tests and doc tests. |
| J2K Metal serialized library suite after M5 | passed | 380 executed, 22 intentional ignores. |
| CUDA runtime suite after M5 | passed | 333/333 on the non-CUDA macOS host. |
| `git diff --check` after M5 | passed | No whitespace errors. |
| M6 structural red test | failed as expected | Capability owner directory and modules were absent. |
| M6 source-size policy | passed | 9/9; the module root is 115 lines and each backend owns its eligibility rules. |
| M6 JPEG all-target check | passed | Public and internal capability paths compile with all features/targets. |
| M6 JPEG strict library clippy | passed | Production library passes `-D warnings`. |
| M6 JPEG all-feature suite | passed | 498 library tests and all integration/doc targets passed. |
| M6 JPEG CUDA all-feature suite | passed | 98 tests across targets on the non-CUDA host. |
| M6 JPEG Metal required-runtime suite | passed | 228 library tests plus 37 integration tests and doc tests. |
| M6 broad all-target clippy | known test debt | Failed on 288 pre-existing test-only disallowed-allocation uses; production library lint passed. |
| M7 structural red test | failed as expected | `encode/mod.rs` and required responsibility owners were absent. |
| M7 source-size policy | passed | 10/10; root is 49 lines and API contains delegation only. |
| M7 facade all-target check | passed | All feature/target combinations compile. |
| M7 facade strict library clippy | passed | All-feature production library passes `-D warnings`. |
| M7 full facade suite | passed | 115 library tests and every integration/doc target passed. |
| M7 focused encode rerun | passed | 43 encode unit tests, 68/69 lossless tests with one intentional ignore, and 28 lossy tests. |
| `git diff --check` after M7 | passed | No whitespace errors. |
| M8 structural red test | failed as expected | Classic ownership units were absent. |
| M8 source-size policy | passed | 11/11; generated tables are isolated and the old 1,812-line allowance is gone. |
| M8 shader-integrity suite | passed | 4/4, including exact byte length/FNV and kernel wiring. |
| M8 J2K Metal all-target check | passed | All feature/target combinations compile. |
| M8 J2K Metal strict library clippy | passed | All-feature production library passes `-D warnings`. |
| M8 serialized J2K Metal suite | passed | 380 executed, 22 intentional ignores. |
| `git diff --check` after M8 | passed | No whitespace errors. |
| C1.1 low-level API red test | failed as expected | Kernel spec, checked geometry export, parameter ABI, and launch primitive were unavailable externally. |
| C1.1 no-feature API tests | passed | 3/3 external kernel-contract tests. |
| C1.1 no-feature/all-feature strict library clippy | passed | Both runtime configurations pass `-D warnings`. |
| C1.1 CUDA runtime all-feature suite | passed | 333 library plus 3 external tests on the non-CUDA host. |
| C1.2 engine dependency red test | failed as expected | JPEG engine manifest/root were absent. |
| C1.2 engine tests and strict clippy | passed | Borrowed boundary compiles under all features without warnings. |
| C1.2 JPEG CUDA all-target check | passed | Adapter compiles through the engine boundary. |
| C1.2 JPEG CUDA all-feature suite | passed | 98 tests across targets on the non-CUDA host. |
| C1.2 architecture policy | passed | 5/5; graph documents engine-to-runtime and adapter-to-engine edges. |
| C1.2 ownership exit-gate red test | failed as expected | Runtime still owned JPEG features, modules, re-exports, and PTX projects. |
| C1.2 JPEG engine extraction suite | passed | 46/46; all 45 moved tests plus the borrowed-engine identity test. |
| C1.2 narrowed CUDA runtime suite | passed | 288/288 library plus 3/3 external low-level tests. |
| C1.2 JPEG CUDA adapter suite | passed | 50 library + 7 encode + 41 host-surface tests. |
| C1.2 strict library clippy | passed | Build support, low-level runtime, and JPEG engine pass all-feature `-D warnings`. |
| C1.2 architecture/source policies | passed | 6/6 dependency/ownership and 11/11 source-size tests. |
| `git diff --check` after C1.2 extraction | passed | No whitespace errors. |
| C1.3 J2K engine dependency red test | failed as expected | J2K engine manifest/root were absent. |
| C1.3 J2K-ML ownership red test | failed as expected | Runtime still owned the ML feature and source. |
| C1.3 J2K engine suite | passed | 2/2: borrowed identity plus moved ML boundary test. |
| C1.3 narrowed runtime suite | passed | 287/287 library plus 3/3 external low-level tests. |
| C1.3 J2K CUDA all-target check | passed | Adapter compiles through engine-forwarded features and resource uploads. |
| C1.3 strict library clippy | passed | Runtime and J2K engine pass all-feature `-D warnings`. |
| C1.3 architecture/source policies | passed | 8/8 dependency/ownership and 11/11 source-size tests. |
| `git diff --check` after C1.3 ML slice | passed | No whitespace errors. |
| C1.3 queued SPI and classic ownership red tests | failed as expected | Runtime lacked the queued retained-resource API and still owned classic sources/features. |
| C1.3 classic J2K engine suite | passed | 12/12 plus doc tests; nine classic tests moved and one focused ABI-layout test was added. |
| C1.3 narrowed runtime suite after classic move | passed | 278/278 library plus 3/3 external low-level tests. |
| C1.3 classic adapter parity target | passed | 1/1 availability-gated test; CUDA execution remains unavailable on macOS. |
| C1.3 strict library clippy after classic move | passed | Runtime and J2K engine pass all-feature `-D warnings`. |
| C1.3 architecture/source policies after classic move | passed | 9/9 dependency/ownership and 11/11 source-size tests. |
| C1.3 HTJ2K/dequantize engine suite | passed | 43/43; moved ABI, source, geometry, overlap, resource, empty-work, queued, and status coverage. |
| C1.3 narrowed runtime after HT move | passed | 245/245 library plus 3/3 external low-level tests. |
| C1.3 HT adapter validation | passed | 164/164 library plus 2/2 availability-gated reconstruction tests. |
| C1.3 HT strict library clippy | passed | Runtime and J2K engine pass all-feature `-D warnings`. |
| C1.3 HT architecture/source policies | passed | 10/10 dependency/ownership and 11/11 source-size tests. |

## Benchmarks

| Experiment ID | Baseline | Treatment | Result | Evidence location |
|---|---:|---:|---:|---|
| G0-JPEG-METAL-ENCODE | CPU 14.285-14.382 ms | Metal 550.27-550.69 ms | Metal slower; baseline only | `EVIDENCE.md`; `target/criterion/` |
| G0-TRANSCODE-METAL-B128 | Rayon 150.30 ms | Metal explicit 134.23 ms | Sampled Metal faster; full matrix incomplete | `EVIDENCE.md`; `target/criterion/` |
| G1-RUST-CLONES | 20-line scan: 1.73% | 12-line scan: 4.31% | Pass at 4.32% ratchet | `target/clone-audit/report/` |
| G1-METAL-CLONES | no prior lane | 25.08% | Pass at 25.09% ratchet | `target/clone-audit/metal-report/` |
| G1-METAL-DECODE-STAGES | resident 7.2883-7.3783 ms | readback 7.9607-8.0043 ms | Required stages observed | `target/criterion/metal_decode_stages/` |
| P1-METAL-IDWT53 | resident 7.3275-7.4667 ms; readback 7.9272-8.0958 ms | resident 7.7683-7.8618 ms; readback 8.3685-8.5390 ms | Rejected and removed; significant 5-7% regression | `docs/performance-experiments/P1-metal-idwt53.json` |
| P2-METAL-IDWT97 | resident 8.2194-8.3167 ms; readback 8.5792-8.7091 ms | resident 8.1828-8.2965 ms; readback 8.5576-8.7163 ms | Rejected and removed; benefit too small/inconclusive for complexity | `docs/performance-experiments/P2-metal-idwt97.json` |
| P3-METAL-FDWT97 | stage 1.8013-1.9794 ms; encode 25.687-25.904 ms | stage 2.5817-2.6222 ms; encode 26.183-26.401 ms | Rejected and removed; full encode regressed significantly | `docs/performance-experiments/P3-metal-fdwt97.json` |
| P5-METAL-INPUT-MCT | RCT 41.055-41.153 ms; ICT 50.845-51.157 ms | RCT 40.663-40.707 ms; ICT 50.435-50.651 ms | Promoted; exact 0.6-1.1% full-encode improvement | `docs/performance-experiments/P5-metal-input-mct.json` |
| P11-METAL-COOPERATIVE-PACKETIZATION | Classic 46.597-46.812 ms; HT 8.2991-8.3648 ms | Classic 48.948-49.168 ms; HT 10.401-10.493 ms | Rejected and removed; significant 4.7-26.2% regressions | `docs/performance-experiments/P11-metal-cooperative-packetization.json` |
| P12-METAL-COLUMN-QUANTIZE | Product 30.402-31.470 ms | Product 30.415-31.212 ms | Rejected and removed; -4.94% to +1.38%, p=0.36 | `docs/performance-experiments/P12-metal-column-quantize.json` |
| P15-CUDA-FDWT97-SHARED-STAGING | Product 5.573581-5.586208 s | Product 5.567594-5.575540 s | Rejected and removed; absolute CIs overlap, wide b16 crosses no change | `docs/performance-experiments/P15-cuda-fdwt97-shared.json` |
| P16-CUDA-INPUT-FUSION | ICT product 194.879754-195.727981 ms | ICT product 194.967121-196.525571 ms | Rejected; absolute CIs overlap and treatment point estimate regresses | `docs/performance-experiments/P16-cuda-input-fusion.json` |
| P18-METAL-JPEG-STAGED-ENCODE | Batch 8 776.67-777.35 ms | Batch 8 395.85-396.52 ms | Promoted; exact 48.96-49.06% improvement | `docs/performance-experiments/P18-metal-jpeg-staged-encode.json` |
| P18-CUDA-JPEG-STAGED-ENCODE | Batch 8 6.795133-6.805057 s | Batch 8 336.067-336.740 ms | Promoted and cleaned up; exact 95.044-95.062% improvement | `docs/performance-experiments/P18-cuda-jpeg-staged-encode.json` |
| P19-METAL-JPEG-DECODE-DEFUSION | 10.456-10.688 ms | 10.621-13.562 ms | Rejected and removed; -0.57% to +14.14%, p=0.30 | `docs/performance-experiments/P19-metal-jpeg-decode-defusion.json` |
| P19-CUDA-JPEG-PACKED-CHECKPOINTS | 21.081-21.255 ms | 20.785-21.026 ms | Promoted and cleaned up; priority absolute CIs separated and all geometry-changing cells improved | `docs/performance-experiments/P19-cuda-jpeg-packed-checkpoints.json` |
| P19-CUDA-JPEG-DECODE-DEFUSION | 20.398-21.151 ms | 30.846-31.996 ms | Rejected and removed; priority +51.47%, all eligible cells +3.00% to +51.47% | `docs/performance-experiments/P19-cuda-jpeg-decode-defusion.json` |

## Decisions Added

- ADR-G001: current migration violations use exact ratchets that reject additions
  and stale inventory entries.
- ADR-G002: Metal clone analysis uses jscpd's OpenCL tokenizer with an explicit
  `metal` extension mapping.
- ADR-G003: stage harnesses measure production flows and expose existing telemetry.
- ADR-A001: `j2k-types::encode_geometry` is the single neutral encode-geometry owner.
- ADR-A002: neutral `j2k-types` plans expose typed immutable geometry borrows
  instead of runtime-erased or native-backend public aliases.
- ADR-A003: one neutral phase budget owns host live-set accounting; adapters
  translate its error once at their boundary.
- ADR-A004: `DeviceDecodeRequest`/`MetalDecodeRequest` are the typed operation
  models and each adapter has one internal operation entrypoint.
- ADR-A005: one source-neutral RGB8 batch builder owns JPEG Metal planning;
  source resolvers only normalize raw bytes or prepared decoders.
- ADR-A006: Classic and HT plans expose one borrowed image-geometry view while
  retaining separate payload types and compatible public enum layouts.
- ADR-A007: byte packing shares sample conversion and checked windows while
  retaining direct 8-bit component-count specializations.
- ADR-A008: internal capability/contract rejections are typed in core and
  rendered into compatible adapter errors at one boundary.
- ADR-M001: stable value families have focused private owners; the already-public
  accelerator trait is an explicit low-level dispatch SPI with compatible root
  re-exports.
- ADR-M002: native color owners are domain modules and facade handoffs use
  named structs while retaining allocation-free moves.
- ADR-M003: JPEG Metal command, immutable-pipeline, and mutable-runtime
  ownership are separate; callers enter named compute domains.
- ADR-M004: codec-batch owns public semantics/planning; batch owns execution.
- ADR-M005: operational crate roots delegate to private responsibility owners
  while retaining only declarations, small marker contracts, and re-exports.
- ADR-M006: JPEG capability inspection coordinates backend-specific correctness
  eligibility; path resolution does not encode performance promotion.
- ADR-M007: facade encode API functions are stable delegates; concrete encode
  phases own validation, execution, routing, and result construction.
- ADR-M008: classic decode shader units preserve exact source ordering and
  bytes; existing classic encode units remain their single owner.
- ADR-C001: engine-owned static PTX and entry points enter the runtime through
  a validated generic kernel/launch contract rather than codec enums.
- ADR-C002: codec engines borrow the stable low-level context rather than
  wrapping or replacing its public identity.
- ADR-C003: codec engines own their CUDA-Oxide projects while one private build
  support crate owns only the shared staging and PTX packaging mechanism.
- ADR-C004: queued pooled-resource lifetime belongs to the low-level runtime;
  classic Tier-1 semantics and status interpretation belong to the J2K engine.
- ADR-C005: HTJ2K decode lifecycle and dequantization belong to the J2K engine;
  codec-neutral allocation, launch, and completion primitives remain runtime-owned.
- ADR-P011: the one-threadgroup-per-tile cooperative packetizer is rejected and
  removed because exact output did not offset large end-to-end regressions.
- ADR-P012: terminal Metal column/quantization fusion is rejected and removed
  because the priority product interval crossed zero despite an isolated-stage win.
- ADR-P015: shared-staging CUDA FDWT97 is rejected and removed because the
  priority product absolute intervals overlap and required stage cells regress.
- ADR-P016: CUDA RGB input fusion is rejected because both product absolute
  intervals overlap and the priority ICT product point estimate regresses.
- ADR-P018: staged Metal JPEG encoding is promoted after exact matrix coverage
  and 49% representative product improvement.
- ADR-P019: Metal JPEG coefficient/IDCT defusion is rejected and removed from
  production after an inconclusive-to-regressive product interval.
- ADR-P019-CUDA-PACKED: CUDA JPEG checkpoint launches use adaptive block-1 or
  block-128 geometry after exact product evidence supported promotion.
- ADR-P019-CUDA-DEFUSION: CUDA JPEG coefficient/IDCT defusion is rejected and
  removed because every eligible 4:2:0 product cell regressed.

## Final Current-Tree Validation

| Command | Result | Notes |
|---|---|---|
| `cargo xtask ci` | passed | Formatting, both canonical clippy stages, panic-surface policy, workspace tests, debug Metal tests, docs/examples, allocation probe, and unsafe audit all passed. |
| `cargo xtask metal-compile` | passed | Passed after `cargo clean` removed 43.9 GiB of reproducible build output that had caused `ENOSPC`; optimized Metal tests and docs passed. |
| `cargo xtask repo-lint` | passed | 101/101 repository-policy tests. |
| `cargo xtask clone-audit` | passed | Production 4.26%, Metal 24.99%, tests 2.68%. |
| `cargo xtask release-integrity` | passed | Current-tree release dependency and ownership checks passed. |
| `cargo xtask public-support --final` | passed | The final public J2K/HTJ2K support matrix and publication-status policy passed. |
| `cargo xtask unsafe-audit` | passed | Current unsafe-source inventory passed. |
| `cargo xtask panic-surface` | passed | `expect_used` 34/50 and `unwrap_used` 13/16; explicit panic inventories unchanged. |
| all 17 `gpu-experiment validate` invocations | passed | Every JSON record under `docs/performance-experiments/` validated under the fail-closed policy at its recorded checkpoint. |
| `cargo xtask stable-api` | passed | Stable-API snapshots match the current tree. |
| `cargo xtask semver` | passed | The approved 0.10.0 transition validates all 18 published-baseline libraries; the review covers all 22 current libraries and exactly 20 removed signatures. |
| clean-copy `cargo xtask package` | passed | A temporary post-CI snapshot `53b8bc4` packaged all crates and consumers; it was validation-only and was trashed afterward. |
| CPU/Metal T.803 development suites | passed | CPU decoder 160/160 and encoder 56/56; Metal decoder 160/160 and encoder 35/35 applicable. |
| post-0.10.0 clean-copy `cargo xtask package` | passed | Snapshot `c0fc0b0758b009c41f1e1b34e6ef0c65777a676a` validated all 23 archives and packaged J2K/CUDA/Metal/ML consumers, then was trashed. |
| `cargo xtask release-cpu` | passed | Complete optimized CPU release lane at workspace 0.10.0. |
| `cargo xtask release-metal --mode full` | passed | Full runtime inventory, 21 required ignored tests, J2K Metal 407 plus device 102, transcode, ML, and facade stages. |
| `cargo xtask release-cuda --mode full` | passed | RTX lane: 923 passed, zero failed or ignored across 46 result rows. |
| host / Metal / CUDA changed-line coverage | passed | Critical coverage 92.31% / 81.61% / 80.68%, each above the independent 80% gate. |
| CPU / Metal / CUDA T.803 development | passed | Decoder 160/160 in each lane; encoder 56/56 CPU and 35/35 per accelerator. |
| host / Metal / CUDA benchmark builds | passed | Every registered benchmark target compiled in its canonical lane. |

## Known Failures or Risks

- The worktree remains dirty with the user's five pre-existing native
  arithmetic/bitplane edits and this workplan's implementation; no changes were
  committed or pushed. The final checkpoint inventory has 907 status entries:
  416 modified, 406 deleted, and 85 untracked. The untracked set contains the
  expected new source, manifests, shader units, experiment records, API review,
  and durable workplan files; no `target`, coverage-report, failed-diagnostic,
  credential-like, or key-like path appears in that set.
- The final architecture checkpoint is not yet authorized. Until it exists,
  `EVIDENCE.md` cannot name a final commit and the dirty-tree development runs
  cannot become clean-candidate exact-SHA release evidence.
- The repository has no registered CUDA Actions runner. P18 and P19 were
  completed on the user-provided direct RTX lane, including strict post-cleanup
  CUDA-Oxide builds and focused runtime tests; this remains manually operated
  hardware rather than CI coverage.
- Nsight Compute 2026.1.1 dynamic counters remain unavailable because the
  driver returns `ERR_NVGPUCTRPERM`. P19 therefore records exact ptxas resource
  evidence and measured product/stage timing, but no inferred occupancy,
  dynamic traffic, cache, shared-memory, or spill counters.
- Public Metal tooling cannot provide the P6 per-kernel register/private-memory,
  occupancy, active-SIMD-group, spill-load/store, or cache inventory, so P7–P10
  are explicitly blocked.
- Both the current debug Metal lane in `cargo xtask ci` and the optimized Metal
  release lane pass; the previously observed Tier-1/MQ debug failure did not
  reproduce on the settled tree.
- The user approved the 0.10.0 pre-1.0 transition. The generated review records
  18 canonical defining-path changes whose root compatibility re-exports remain,
  the CUDA transcode availability move into its engine, and the generic
  `DeviceCodestream` Metal handoff. The one-time transition must be disabled
  after a published `v0.10.0` becomes the next baseline.

## Exact Next Action

Create the authorized local architecture checkpoint from the complete validated
0.10.0 tree, including the five preserved pre-existing native Tier-1/MQ edits.
Then record the implementation SHA in durable evidence, create the evidence-only
finalization commit, and verify the resulting clean tree. Do not push, tag,
publish, deploy, or start release preparation without separate authorization.

## Exact Next Command

`git add -A`
