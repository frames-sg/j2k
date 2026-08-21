# Architecture Decisions

Plan anchor: J2K-HARDENING-2026-08-18
Audit baseline: f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5

## ADR-G001: Migration violations use exact ratchets

Status: accepted
Context: G1 guardrails must prevent new violations before current architecture is migrated.
Decision: Semantic dependency, plan-erasure, and phase-budget checks compare live findings with an exact reviewed inventory. Additions and stale entries both fail.
Alternatives rejected: Failing on all current violations would make the canonical lint unusable; unconstrained allowlists would normalize debt.
Consequences: Migration changes must remove their inventory entries in the same change. The inventory is not approval of the listed architecture.
Files affected: `xtask/tests/repo_lint_support/{architecture_policy,prepared_plan_policy,allocation_policy}.rs`
Tests proving the decision: focused policy tests in `cargo test -p xtask --test repo_lint`.

## ADR-G002: Metal clone analysis uses the OpenCL tokenizer

Status: accepted
Context: pinned jscpd 4.0.5 has no Metal grammar but supports OpenCL and explicit extension mappings.
Decision: Stage `.metal` files unchanged in a dedicated lane and invoke jscpd with `opencl:metal`; keep a separate config, report, and threshold.
Alternatives rejected: Treating shaders as plain text loses token semantics; renaming without an explicit mapping was ignored by jscpd; adding a parser dependency is unnecessary.
Consequences: Metal syntax is approximated by the closest supported C-family GPU grammar. The 25.08% baseline is a ratchet, not a quality claim.
Files affected: `.jscpd-metal.json`, `xtask/src/clone_audit.rs`, `xtask/src/clone_audit/{config,metal_stage,tests}.rs`
Tests proving the decision: Metal staging/config/argument unit tests and passing `cargo xtask clone-audit`.

## ADR-G003: Stage harnesses measure production flows

Status: accepted
Context: Existing APIs expose dispatch/timing telemetry around complete production flows, not safe standalone entry points for every internal kernel.
Decision: Metal benchmarks resident and readback end-to-end decode while reporting production dispatch/profile stages. CUDA runs the same production DWT97 code-block path in separate fused/unfused processes selected by its documented kill switch and reports backend timings.
Alternatives rejected: Publishing benchmark-only kernel APIs would widen hidden boundaries; labeling repeated full decodes as isolated stage timings would be misleading.
Consequences: Metal Criterion numbers are end-to-end; stage observability comes from dispatch/profile evidence. CUDA fused/unfused A/B runs must be separate because the production flag is process-cached.
Files affected: `crates/j2k-metal/benches/decode_stages.rs`, `crates/j2k-transcode-cuda/benches/dwt97.rs`, their manifests, and benchmark/experiment repository policy tests.
Tests proving the decision: both targets compile, focused clippy passes, the Metal benchmark runs on M4 Pro, and CUDA runtime-feature compilation passes.

## ADR-A001: Shared encode geometry belongs in `j2k-types`

Status: accepted
Context: Lossless level selection diverged between the facade and Metal for explicit requests on dimensions below 64, while legal-level, DWT-shape, code-block, bitplane, and packet-order calculations were spread across neutral and backend crates.
Decision: `j2k-types::encode_geometry` owns legal and lossless level policy, DWT level/subband dimensions, code-block exponent conversion and validation, reversible subband bitplanes, and packet ordering. Explicit requests remain disabled below 64 to preserve the tested facade contract. Backend adapters perform only typed error mapping, fallible allocation, and integer-width conversion. The old `j2k_codec_math::dwt::max_decomposition_levels` path remains a compatibility re-export of the shared owner.
Alternatives rejected: Keeping the policy in codec math would leave packet and plan geometry split; using the facade would create forbidden adapter dependencies; adopting Metal's below-64 override behavior would break the established public contract.
Consequences: Facade, native, Metal, CUDA-runtime, and semantically identical Metal-transcode paths depend inward on `j2k-types`. A repository policy rejects reintroduced GPU-local lossless-level algorithms. Packet-order implementation moved out of the type crate root, lowering its reviewed ceiling from 1,183 to 1,125 lines.
Files affected: `crates/j2k-types/src/{lib,encode_geometry}.rs`, codec-math compatibility export, facade/native/Metal/CUDA-runtime/transcode consumers, manifests, architecture docs, and the encode-geometry policy test.
Tests proving the decision: required A1 matrix tests, facade/Metal boundary parity, COD marker and round-trip integration, native/CUDA DWT tests, real Metal encode tests, full `cargo xtask test`, clone audit, and byte-identical CPU/Metal T.803 development reports.

## ADR-A002: Prepared-plan sharing has a neutral typed owner

Status: accepted
Context: `PreparedClassicPlan` and `PreparedHtj2kPlan` retained immutable, data-only, zero-copy geometry behind `Arc`, but the geometry was defined by `j2k-native` and exposed to device backends through `dyn Any`, `adapter_view`, and fallible downcasts. A facade alias to the native type removed erasure but still leaked a private backend through public API.
Decision: `j2k-types::decode_plan` owns the direct and referenced Classic/HT decode-plan contracts, payload jobs, geometry view, rectangle and wavelet values, and retained-capacity accounting. Native constructs and re-exports those neutral contracts. Facade `ClassicPreparedGeometry`, `Htj2kPreparedGeometry`, and `PreparedImageGeometry` aliases point to `j2k-types`; wrappers expose immutable typed borrows, and CUDA and Metal consume the same compile-time contract.
Alternatives rejected: Facade aliases to native types fail the public-boundary requirement; a second isomorphic facade plan would copy or convert the execution graph; retaining a downcast compatibility method would violate the completion gate.
Consequences: Cloning a public prepared-plan wrapper clones only its `Arc`; the same geometry address and original compressed `Arc<[u8]>` are retained. Unsupported adapter-type errors disappear because incompatibility is no longer representable, and the public facade no longer exposes native backend types. The guardrail inventory is empty and rejects reintroduced erasure or native ownership.
Files affected: `crates/j2k-types/src/decode_plan/`, native plan producers/re-exports, `crates/j2k/src/owned_batch/{prepared_plan.rs,prepared.rs}`, facade exports, CUDA input/execution/tests, Metal plan caches, facade integration tests, and `prepared_plan_policy.rs`.
Tests proving the decision: full owned-batch and CUDA suites, Metal prepared-plan/cache tests in the full suite, explicit pointer-identity and payload-range assertions, focused no-erasure policy tests, completion search, and changed-library clippy.

## ADR-A003: Host phase accounting has one neutral owner

Status: accepted
Context: Four CUDA-family crates wrapped the same `HostAllocationBudget` with independently evolving phase methods and error mapping.
Decision: Extend `j2k-core` with the complete `HostPhaseBudget` API and one `HostPhaseError`. It owns preflight, actual-capacity/live-byte accounting, fallible vector construction, copy/clone/collection, incremental growth, and checked product/sum. Each adapter implements `From<HostPhaseError>` for its existing local error and otherwise re-exports the shared type internally.
Alternatives rejected: Keeping four wrappers would preserve duplicated policy; a closure-carrying generic mapper would complicate every allocation; replacing adapter errors with the neutral error would break existing classification and source behavior.
Consequences: There is exactly one implementation and one semantics test matrix. Allocation purpose and phase context remain explicit, and adapter APIs retain their established errors. Named allocation methods exist only where an owner label differs from the phase label.
Files affected: `j2k-core/src/{host_allocation,lib}.rs`, four CUDA-family allocation modules and consumers/tests, their error enums, and `allocation_policy.rs`.
Tests proving the decision: cross-crate allocation-focused suites, error-classification tests, exact/one-over/overflow/ZST/growth/failure core tests, one-owner repo lint, and all-feature strict library clippy.

## ADR-A004: Decode geometry is one typed operation per adapter

Status: accepted
Context: CUDA exposed four parallel image paths and four tile paths with repeated validation, geometry planning, routing, allocation, and fallback logic. Metal had a typed request but still dispatched into four route implementations.
Decision: Use the existing `DeviceDecodeRequest` as CUDA's operation enum and `MetalDecodeRequest`/`MetalDecodeOp` as Metal's. CUDA image decode owns one `decode_op_to_surface_impl`, tile decode owns one boundary-equivalent `decode_tile_op_to_surface_impl`, and Metal owns one `decode_op_to_surface_impl`. Trait methods only construct an operation and delegate.
Alternatives rejected: Adding another shared enum would duplicate `DeviceDecodeRequest`; forcing image and tile APIs through one lifetime-heavy function would obscure their distinct ownership boundaries; retaining geometry-specific wrappers would keep the orchestration duplication.
Consequences: Backend validation, normalized dimensions, Auto selection, CPU fallback, and profiling labels are decided once per image adapter. CPU-staged CUDA methods share the same normalized plan and CPU operation executor. Metal preserves its benchmark-qualified scaled Auto rule inside its unified route.
Files affected: CUDA decoder API and tile codec, Metal decoder routes/core/direct paths and batch route entry.
Tests proving the decision: CUDA structure regression, 37-test host-surface operation matrix, Metal request/routing/session tests, existing all-format suites, and strict all-feature library clippy.

## ADR-A005: JPEG Metal batch planning is source-neutral after resolution

Status: accepted
Context: Raw JPEG bytes and prepared decoders used separate loops for vector allocation, output-shape validation, plan-owner admission, cache accounting, insertion, and execution baselines, allowing their batch semantics to drift.
Decision: Each source adapter resolves one item into `ResolvedRgb8BatchSource`. One `build_rgb8_batch_plan` loop owns dimension and sampling consistency, restart restrictions, destination budgeting, plan-owner and cache accounting, request insertion, and execution-owner stamping. Prepared-decoder retained bytes enter the shared context as an external live-set baseline.
Alternatives rejected: A codec-generic framework would abstract unrelated formats without a current need; retaining two builders behind a common helper would leave ordering and accounting policy duplicated.
Consequences: Raw and prepared requests have identical normalized keys and rejection behavior for full, scaled, and region-scaled operations. The source-specific closures contain only parsing/cache resolution or prepared-request extraction.
Files affected: `crates/j2k-jpeg-metal/src/codec_batch.rs`.
Tests proving the decision: six planner parity/structure/cache tests, duplicate-owner exact-cap tests, the 228-test JPEG Metal library suite, all integration targets with required Metal runtime, and strict all-feature library clippy.

## ADR-A006: Prepared image geometry is a borrowed codec-neutral view

Status: accepted
Context: Classic and HT prepared wrappers repeated component classification and wavelet aggregation even though both already use the same referenced tile geometry. Moving common fields into a new owned structure would change the public, doc-hidden native enum variant fields.
Decision: Add a copyable borrowed `J2kReferencedImageGeometry<'a>` view over the existing tile slice, full dimensions, and output rectangle. Both native plan types construct the view without allocation; facade plans expose it as `PreparedImageGeometry` and retain their compatibility methods as thin delegates. Codec-specific payload and range APIs remain separate.
Alternatives rejected: Replacing variant fields with an owned common structure would be a public-shape break; a generic payload trait would conflate different Classic fragment and HT continuation semantics; copying common facts into facade wrappers would duplicate state.
Consequences: Tile emptiness, grayscale/RGB/RGBA classification, legacy single-tile component access, common dimensions, and uniform wavelet selection have one implementation. The view cannot outlive or mutate its plan, and public enum layouts remain compatible.
Files affected: native referenced-plan geometry/export files, facade prepared-plan exports, owned-batch contract tests, and prepared-plan repo policy.
Tests proving the decision: 36-test facade owned-batch suite, native direct-plan tests, full CUDA all-feature suite, focused Metal prepared-plan runtime matrix, ownership policy, and four-library strict clippy.

## ADR-A007: Byte packing shares policy but retains fast-loop specialization

Status: accepted
Context: Full-image and region byte packers independently detected bit-depth equality and implemented the same scaling, rounding, and quantization formula, while only the full path retained useful unrolled 1-4 component loops.
Decision: `color/packing.rs` owns `SampleConversionPolicy`, checked `SampleWindow`, and generic window traversal. Both public packing paths use them. The full-image direct 8-bit path keeps its explicit one-, two-, three-, and four-component loops, with every loop calling the same quantizer.
Alternatives rejected: One fully generic iterator loop would discard clear common-case specialization; keeping traversal and bounds in the callers would leave ROI/full policy split; changing signed-display semantics during the refactor would alter established output behavior.
Consequences: Full and equivalent ROI projections are byte-identical for direct, scaled, mixed-depth, high-bit, and signed inputs. Short destinations now return the existing typed buffer error before writes. `color.rs` falls from 1,172 to 910 lines and the focused packing module is 275 lines.
Files affected: `j2k-native/src/color.rs`, `color/packing.rs`, boundary tests, image caller, and packing repo policy.
Tests proving the decision: required conversion matrix, full native library and component-plane suites, one-owner policy, source-size policy, and strict native/xtask clippy.

## ADR-A008: Internal rejections are typed before public rendering

Status: accepted
Context: Four accelerator adapters embedded hundreds of static rejection strings directly in production control flow. Their public error variants expose `&'static str`, so replacing field types or variants would break callers that construct or match them.
Decision: `j2k-core::CapabilityRejection` carries one of ten explicit categories plus exact stable diagnostic text. Each adapter has one `capability_rejected` boundary that renders the typed reason into its existing `UnsupportedCudaRequest` or `UnsupportedMetalRequest` variant. Production call sites construct typed reasons; public variants remain unchanged for source compatibility.
Alternatives rejected: One generic `Other` would erase semantics; changing public variant fields would be breaking; adapter-local taxonomies would duplicate categories; reclassifying checked invariants during this migration would change established `CodecError` behavior.
Consequences: Format, sampling, bit-depth, operation, prepared-plan, container, geometry, resource, context, and contract failures are distinguishable internally. Exact public display and unsupported classification are preserved. An AST guardrail permits public variant matching but rejects direct production construction outside adapter error boundaries.
Files affected: `j2k-core` rejection type/export/tests, four adapter error boundaries and production rejection sites, plus error-taxonomy policy.
Tests proving the decision: core taxonomy matrix, full CUDA/JPEG CUDA and JPEG Metal suites, serialized J2K Metal library suite, error/source/cleanup classification tests, five-library strict clippy, AST policy, and clone audit.

## ADR-M001: `j2k-types` separates stable values from dispatch SPI

Status: accepted
Context: Stable codec values, prepared encode owners, packet ordering, dispatch accounting, errors, and the accelerator trait shared one 1,125-line crate root and were described collectively as backend experimentation despite being semver-visible.
Decision: Give transform, Tier-1, packetization, prepared-plan, dispatch-report, and dispatch-error families focused private module owners. Keep every existing root name as a compatibility re-export. Treat the already-public accelerator trait as a supported low-level dispatch SPI; do not hide it behind a feature or require callers to adopt defining-module paths.
Alternatives rejected: Feature-gating the existing trait would be breaking; an internal crate cannot preserve downstream trait implementations; empty `packetization/output.rs` and duplicate aliases would create ownership without a domain type.
Consequences: The crate root remains a compact compatibility surface, production modules stay focused, progression behavior stays with progression values, and move-only assertions stay with their owners. Root paths remain valid while private defining-module paths do not become accidental public API.
Files affected: `j2k-types/src/{lib,transform/,tier1/,packetization/,dispatch/,prepared_plan/,encode_geometry.rs}` and the source-size repository policy.
Tests proving the decision: M1 red/green structure regression, 18 `j2k-types` tests, strict all-target clippy, workspace all-target compilation, full native library tests, and the full facade suite.

## ADR-M002: Native color ownership and handoff are named

Status: accepted
Context: Public bitmap values, component-plane owners, container metadata, channel ordering, palette expansion, ICC cloning, sYCC, CIE Lab, and packing shared a 910-line file. Facade transfer used positional tuples whose same-width fields were easy to misorder.
Decision: Use focused `color/` modules for each domain. Retain A7's established `packing.rs` name and policy owner. Replace doc-hidden tuple aliases and aggregate tuple returns with named handoff structs whose public fields move existing owners without allocation.
Alternatives rejected: Renaming `packing.rs` to match the illustrative `pack.rs` would churn the A7 guardrail; keeping a mixed `postprocess.rs` would continue coupling palette expansion and channel metadata; retaining tuple compatibility would leave the ambiguity M2 explicitly targets.
Consequences: `color/mod.rs` is 65 lines; every ownership module is at or below 396 lines. The doc-hidden implementation SPI changes shape, while stable decoded bitmap/component APIs and sample behavior remain unchanged. Facade handoff retains payload and ICC pointer identity.
Files affected: `j2k-native/src/color/{mod,types,metadata,output_planes,allocation,packing,palette,icc,sycc,cielab}.rs`, native re-exports, facade component handoff, and source-size policy.
Tests proving the decision: M2 red/green structure regression, 17 focused color tests, 636 native library tests, 24 component-plane tests, full facade tests, and strict native/facade clippy.

## ADR-M003: JPEG Metal runtime and pipeline registry are distinct owners

Status: accepted
Context: `compute.rs` owned checked command creation, shader concatenation, more than fifty immutable pipeline states, default-session initialization, mutable scratch/cache state, and broad decode/encode entrypoints despite an already extensive domain module graph.
Decision: Move checked command creation/completion to `command.rs`, shader source and immutable pipeline loading/selection to `pipeline_registry.rs`, and device/queue/session plus mutable cache ownership to `runtime.rs`. Nest pipeline fields under the registry and remove their redundant `_pipeline` postfix. Keep established entropy, pack, full-batch, region-batch, single-decode/texture, status, and encode pipeline modules; direct crate callers to their named entry modules.
Alternatives rejected: A universal compute prelude would preserve the same hidden coupling; moving every already-focused kernel file solely to match illustrative names would add churn without changing ownership; retaining pipeline fields on the mutable runtime would keep compilation and cache lifecycle coupled.
Consequences: `compute/mod.rs` is 193 lines and contains no command functions, shader source, or runtime definition. Shader integrity scans the registry owner. Runtime pipeline access makes ownership explicit as `runtime.pipelines`.
Files affected: JPEG Metal `compute/{mod,command,pipeline_registry,runtime}.rs`, pipeline access sites, named domain callers, and shader-integrity/source-size policies.
Tests proving the decision: M3 red/green structure regression, all-target compile, strict library clippy, shader-integrity test, and the full required-Metal 265-test suite.

## ADR-M004: JPEG Metal batch semantics and execution have different owners

Status: accepted
Context: Public RGB8 source/operation/target contracts, normalized raw/prepared planning, eligibility inspection, owner accounting, target resizing, tile submission, queue grouping, flushing, and completion were split ambiguously between a 1,018-line `codec_batch.rs` and `batch.rs`.
Decision: `codec_batch/` owns public request semantics, source normalization, inspection, planning, owner accounting, and buffer/texture target adapters. `batch.rs` retains queued request keys/shapes, grouping, execution owner stamps, flush, completion, and `MetalSubmission`.
Alternatives rejected: Merging both hierarchies would couple public reusable-output semantics to session execution; duplicating target planners would undo A5; empty wrapper modules would not change ownership.
Consequences: The semantic root is 27 lines, all files are under 400 lines, and raw/prepared parity still passes through one `build_rgb8_batch_plan`.
Files affected: `j2k-jpeg-metal/src/codec_batch/` and the M4 source-boundary policy.
Tests proving the decision: M4 red/green structure test, source policy, all-target check, strict library clippy, parity/accounting tests, and the full required-Metal suite.

## ADR-M005: Operational crate roots delegate without widening the API

Status: accepted
Context: The J2K Metal, JPEG Metal, and CUDA runtime roots mixed declarations and re-exports with benchmark implementations, codec trait bodies, decode/upload routing, and macro definitions. This obscured responsibility and made every internal operation appear root-owned.
Decision: Put benchmark-only J2K Metal operations in private `bench_support`, JPEG generic trait implementations in private `codec`, JPEG decode/routing/upload operations in private `decode_surface`, and CUDA macros in private `macros`. Preserve existing semver-visible root paths through controlled re-exports; do not expose the new modules.
Alternatives rejected: Deleting doc-hidden benchmark entry points would break repository benches; making ownership modules public would add unsupported API; leaving trait bodies in the root would retain operational coupling; replacing macros during a structural task would change behavior unnecessarily.
Consequences: The roots are 108, 108, and 131 lines and contain declarations, re-exports, the small JPEG codec marker, and its unavoidable `ImageCodec` associated-type implementation. Operational behavior and macro expansion remain unchanged.
Files affected: `j2k-metal/src/{lib,bench_support}.rs`, `j2k-jpeg-metal/src/{lib,codec,decode_surface}.rs`, `j2k-cuda-runtime/src/{lib,macros}.rs`, and source-size policy.
Tests proving the decision: M5 red/green root-ownership regression, all-target compilation, three strict library clippy targets, full required-Metal JPEG suite, serialized J2K Metal library suite, CUDA runtime suite, and diff check.

## ADR-M006: JPEG capability reports contain eligibility, not promotion policy

Status: accepted
Context: One 672-line file owned public requests, output geometry, parse/planner rejection handling, CPU format and sampling rules, CUDA addressability, Metal surface and resident-batch rules, and final path selection. Backend correctness constraints and route choice were difficult to review independently.
Decision: Give request, output-geometry, CPU, CUDA, Metal, rejection, and resolution concerns private focused modules behind unchanged crate-root re-exports. Capability reports aggregate correctness eligibility. Explicit backends resolve from those facts; `Auto` remains the portable CPU path in this resolver, while measured workload promotion stays in accelerator routing layers.
Alternatives rejected: A generic backend trait would add indirection to three concrete policies; combining CUDA and Metal as `device` would hide their different formats and bounds; putting performance thresholds in capability results would conflate “can execute correctly” with “should execute here.”
Consequences: `capabilities/mod.rs` is 115 lines and every backend rule can be tested and changed independently. Public type names, fields, methods, derived traits, and stable rejection text remain unchanged.
Files affected: `j2k-jpeg/src/capabilities/{mod,request,output_geometry,cpu,cuda,metal,rejection,resolve}.rs` and source-size policy.
Tests proving the decision: M6 red/green ownership regression, all-feature JPEG suite, JPEG CUDA and required-runtime JPEG Metal suites, all-target compile, downstream strict library clippy, and diff check.

## ADR-M007: Facade encode entry points delegate to phase owners

Status: accepted
Context: Eight public encode entry points, high-bit component adaptation, backend routing, decomposition geometry, round-trip validation, CPU execution, rate targeting, ROI conversion, result construction, and tests shared one 562-line facade file despite several partially focused submodules.
Decision: Keep every public signature and root re-export, but make `api.rs` a thin delegation surface. Lossless, lossy, ROI, accelerator, high-bit, geometry, CPU, and validation modules own their concrete phases. Rename the private `native` and `routing` modules to the responsibility-oriented `cpu` and `accelerator` names. Retain established contract, sample, allocation, resident, and rate-target modules rather than duplicating them to match an illustrative tree.
Alternatives rejected: Moving only physical line ranges would leave public functions coordinating every phase; adding a generic encode pipeline trait would obscure concrete hot paths; duplicating existing lossy and validation owners would create competing policy locations.
Consequences: `encode/mod.rs` is 49 lines and `api.rs` contains no validation, CPU calls, backend resolution, or result literals. ROI descriptor validation and high-bit guards have named owners. Public docs, signatures, error text, output metadata, and dispatch semantics are preserved.
Files affected: `j2k/src/encode/{mod,api,geometry,cpu,accelerator,high_bit,lossless,lossy,roi,validation,tests/}` plus source-size policy.
Tests proving the decision: M7 red/green ownership regression, all-target compile, strict library clippy, full facade suite, focused lossless/lossy/ROI/accelerator reruns, and diff check.

## ADR-M008: Classic Metal decode units preserve their composed bytes

Status: accepted
Context: One 1,812-line shader mixed ABI layouts, constants, generated QE and context tables, state helpers, entropy decoder primitives, three job implementations, and six decode/store kernels. Host compilation consumed the file as one string inside a larger generated Metal library.
Decision: Partition the source at declaration boundaries into ABI, constants, QE table, context tables, state support, MQ decoder, bypass/sign decoder, pass logic, and decode-kernel units. Compose them in their original order without separators, and ratchet the original byte length and FNV-1a digest. Keep established `encode_bitstream_classic_kernels.metal` as the encode owner rather than creating an empty duplicate file.
Alternatives rejected: Runtime `#include` resolution from `newLibraryWithSource` lacks a stable file/include root; reordering interwoven MQ/bypass helpers would weaken byte-equivalence evidence; duplicating encode kernels would create two owners; an empty target-layout file would add no responsibility.
Consequences: The generated tables are reviewable without control flow, every unit is below the shader hard limit, and the host compiles byte-for-byte the same classic source. The deleted monolith can be recovered from the local Trash until emptied.
Files affected: `j2k-metal/src/classic/*.metal`, `compute/shader_source.rs`, shader-integrity tests, and source-size policy.
Tests proving the decision: M8 red/green structural policy, exact 77,178-byte/FNV reconstruction, kernel-wiring inventory, all-target compile, strict library clippy, serialized 380-test Metal runtime suite, and diff check.

## ADR-C001: Codec engines submit validated static kernel specifications

Status: accepted
Context: The runtime's module cache key was a closed enum containing every JPEG, J2K, HTJ2K, ML, and transcode kernel. Moving engines out would otherwise require each new codec concept to modify the low-level runtime and its cache.
Decision: `CudaKernelSpec` validates a static module identifier, NUL-terminated PTX image, and non-empty NUL-terminated entry point. Cache identity includes the diagnostic ID, static PTX allocation identity/length, and entry point. `CudaLaunchGeometry` is a checked public value. External engines implement the unsafe `CudaKernelParam` ABI contract for their local `repr(C)` values and call an unsafe generic synchronous launch primitive with explicit safety requirements.
Alternatives rejected: Raw module/function handles would expose driver lifetime hazards; hashing every PTX byte on every lookup would put source size in the hot path; a new runtime enum variant per engine kernel would preserve reverse ownership; a generic safe variadic argument API is not expressible without tuple machinery or new dependencies.
Consequences: Engine crates can own PTX and entrypoint inventories while the runtime owns loading, caching, launch validation, context binding, synchronization, and module lifetime. Existing internal codec wrappers continue to use the legacy cache keys during migration. The unsafe boundary is narrow and documented.
Files affected: CUDA runtime `kernel.rs`, kernel cache, execution launch surface, launch geometry, driver lint annotations, root exports, tests, and kernel-geometry structure test.
Tests proving the decision: external red/green low-level API test, malformed spec matrix, checked geometry limits, typed parameter pointer construction, no-feature and all-feature strict clippy, and the 333-test runtime suite.

## ADR-C002: CUDA codec engines borrow the low-level context identity

Status: accepted for C1 migration
Context: Public adapter constructors and resident resource types already expose `j2k_cuda_runtime::CudaContext` and its buffers. Replacing it with engine-specific newtypes would break adapter APIs and fragment context, pool, external-allocation, and ordering identity.
Decision: Each codec engine is a borrowed operation object over `&CudaContext`. The internal JPEG engine establishes this boundary first. Adapters import JPEG domain contracts from the engine and invoke its operations, while low-level contexts, buffers, pools, pinned transactions, errors, and execution statistics retain their runtime type identity.
Alternatives rejected: Engine-specific context newtypes would change public adapter types; extension traits would require every downstream caller to import method traits and would expose a very broad method surface; a runtime dependency on the engine would reverse the required graph.
Consequences: JPEG implementation bodies moved behind `JpegCudaEngine` without a second adapter migration. The runtime no longer owns JPEG source, public domain re-exports, kernel variants, cache keys, features, or PTX projects. The same borrowed-context pattern is available for the remaining C1 engine families.
Files affected: new `j2k-cuda-jpeg-engine`, workspace/lockfile, JPEG CUDA manifest and operation/domain call sites, architecture graph, and dependency policy.
Tests proving the decision: dependency-direction red/green policy, engine unit/strict clippy, JPEG CUDA all-target compilation and 98-test suite, and full architecture graph check.

## ADR-C003: CUDA engines own projects; build support owns packaging mechanics

Status: accepted
Context: Moving JPEG PTX ownership out of the runtime required moving two CUDA-Oxide source projects and their feature/build flags. Copying the runtime's 700-line build script into each planned engine would create competing toolchain, placeholder, staging, codec-math path, and NUL-termination policies.
Decision: Add one unpublished `j2k-cuda-build-support` crate that stages the shared SIMT prelude, renders CUDA-Oxide project templates, invokes the pinned toolchain, emits cfg/rerun metadata, and writes NUL-terminated real or placeholder PTX. Each runtime or engine build script declares only its own feature-to-project inventory and extra source files. JPEG decode/encode projects and built cfgs belong exclusively to `j2k-cuda-jpeg-engine`.
Alternatives rejected: Duplicating build scripts would make packaging policy drift likely; keeping JPEG sources under the runtime for build convenience would preserve reverse ownership; making the helper a runtime dependency would put build mechanics in production binaries; moving kernel source into a generic build crate would erase codec ownership.
Consequences: Runtime and engines share one build mechanism without sharing codec inventories. Changing a codec's kernels changes that engine; changing CUDA-Oxide staging changes one private build dependency. Non-Linux builds retain structured missing-PTX diagnostics and placeholder behavior, and strict Linux builds still honor `J2K_REQUIRE_CUDA_OXIDE_BUILD`.
Files affected: new `j2k-cuda-build-support`, runtime and JPEG-engine build scripts/manifests, moved CUDA-Oxide JPEG projects, workspace/lockfile, source policy, and architecture documentation.
Tests proving the decision: ownership exit-gate red/green test, all-feature runtime/engine builds on the non-CUDA host, moved missing-build diagnostic test, strict library clippy, source-size policy, and adapter suite.

## ADR-C004: Runtime guards completion; the J2K engine owns classic Tier-1

Status: accepted for C1 migration
Context: Classic Tier-1 queued decode depended directly on private kernel dispatch, pool reuse guards, context synchronization, status-copy diagnostics, and HTJ2K payload ownership. Moving the codec module verbatim would either expose lifecycle internals or duplicate the runtime's uncertain-completion safety rules.
Decision: The low-level runtime exposes one unsafe queued compiled-kernel submission that consumes pooled owners and returns a `CudaQueuedExecution`. Its safe completion path synchronizes and returns retained resources for readback; an unsafe alternate path releases them only after the caller proves completion with the runtime's event timer. Codec-neutral buffer/pool ownership, disjoint-allocation validation, D32 memset, and status-copy accounting remain runtime responsibilities. `j2k-cuda-j2k-engine` owns classic ABI/types, byte views, semantic validation, table resources, status interpretation, timing orchestration, feature policy, tests, and PTX project.
Alternatives rejected: Publishing `CudaBufferPoolReuseGuard` would expose pool invariants; copying synchronization and quarantine logic into the engine would create two safety owners; making queued classic decode synchronous would regress the adapter completion graph; retaining classic cache variants or PTX under the runtime would leave reverse codec ownership.
Consequences: Classic queued work preserves pooled owners through success, launch failure, explicit finish, and Drop without engine access to runtime internals. The runtime no longer contains a classic feature, module, re-export, legacy cache key, kernel variant, or PTX project. HTJ2K payload resources remain a temporary runtime dependency until the next C1 slice.
Files affected: runtime execution/memory/resource accessors and external API test; `j2k-cuda-j2k-engine/src/classic_decode*` and classic CUDA-Oxide project; runtime cache/kernel/build inventory; J2K CUDA classic adapter call sites; architecture policy and documentation.
Tests proving the decision: queued-SPI external red/green test, classic-ownership red/green gate, 12-test engine suite, 278-test runtime suite plus 3 external tests, adapter classic parity target, strict library clippy, architecture/source policy, formatting, and diff checks.

## ADR-C005: The J2K engine owns HTJ2K decode lifecycle and dequantization

Status: accepted for C1 migration
Context: HTJ2K decode combined codec ABI, payload/table ownership, disjoint output validation, cleanup/refinement kernel choice, queued status groups, deferred dequantization, pool quarantine, and runtime-private kernel inventory. Moving only entry points would leave the runtime as the semantic owner.
Decision: Move the complete HTJ2K decode and J2K dequantization family into `j2k-cuda-j2k-engine`. Keep context identity, buffers, pools, synchronous/asynchronous D32 memset, generic compiled-kernel launch, synchronization outcomes, and reuse guards in the low-level runtime. Engine methods consume the copyable borrowed engine by value; resource and queued guards retain the stable runtime allocation types.
Alternatives rejected: Keeping payload/status types in the runtime would preserve codec concepts there; duplicating pool lifecycle in the engine would create a second uncertain-completion owner; making adapter-visible context or buffer types engine-specific would break stable APIs.
Consequences: The runtime contains no HTJ2K-decode/dequantize feature, source, re-export, kernel/cache variant, build flag, or PTX project. Existing sync, queued, empty-work, ABI, source-regression, status, and output-overlap tests moved with their owner. Real CUDA and CUDA-Oxide execution still require the unavailable Linux/NVIDIA lane.
Files affected: runtime memory/launch SPI and inventories; `j2k-cuda-j2k-engine/src/{htj2k_decode*,bytes,kernels,build_flags,context}.rs` and CUDA-Oxide projects; J2K CUDA decode/session call sites; architecture/source policies and documentation.
Tests proving the decision: HT ownership red/green policy, 43 engine tests, 245 runtime tests plus 3 external API tests, 164 adapter library tests, 2 reconstruction tests, strict clippy, architecture/source policy, formatting, and diff checks.

## ADR-C006: Coupled J2K encode kernels move as one engine-owned unit

Status: accepted
Context: J2K forward transforms and quantization, HTJ2K code-block encode, compaction, and packetization appeared separable at the Rust API but share generated PTX modules and launch metadata. Splitting their ownership would compile on the non-CUDA host while leaving Linux module loading coupled to the runtime.
Decision: Move the complete transform/store/encode family into `j2k-cuda-j2k-engine` together, including ABI views, validation, resources, launch inventory, tests, features, build flags, and PTX projects. Keep only generic compiled-kernel and memory lifecycle primitives in the runtime.
Alternatives rejected: Cross-crate ownership of one PTX module would hide Linux-only coupling; duplicate PTX packaging would create competing entrypoint owners; retaining encode compaction in the runtime would violate C1's codec-neutral goal.
Consequences: The J2K engine is the single owner of every J2K/HTJ2K/ML kernel family. Comprehensive stable-entrypoint and generated-PTX tests cover the moved inventory. Real PTX compilation remains a Linux/NVIDIA verification lane.
Files affected: J2K engine transform/store/encode modules and tests, runtime inventories, J2K and transcode adapters, manifests/build scripts, and architecture policy.
Tests proving the decision: transform/store ownership red/green gate, 168 engine tests, 164 J2K adapter tests, strict library clippy, and kernel entrypoint inventory.

## ADR-C007: Coefficient-domain CUDA transcode has its own borrowed engine

Status: accepted; completes C1
Context: The last runtime codec concepts were transcode-specific band models, DCT-grid validation, reversible/irreversible transforms, fused quantization, timings, and one CUDA-Oxide project. Moving these directly into the public adapter would combine stable routing APIs with low-level kernel ownership.
Decision: Add unpublished `j2k-cuda-transcode-engine`, borrowing `&CudaContext` and retaining runtime buffer/pool identities. It owns the transcode types, validation, launch geometry, orchestration, stage timings, tests, feature, and PTX project. The public adapter depends inward on this engine and the J2K engine used for resident HT encode.
Alternatives rejected: Keeping extension methods on `CudaContext` would leave codec semantics in the runtime; moving implementation directly into the public adapter would skip the required engine boundary; engine-specific context/buffer newtypes would break adapter contracts.
Consequences: The runtime source, manifest, cache, dispatch, tests, and build script contain no transcode symbol or feature. All CUDA codec families now follow adapter → engine → runtime dependency direction.
Files affected: new transcode engine, runtime cleanup, transcode adapter imports/calls/features, workspace graph, architecture policy, and durable workplan.
Tests proving the decision: failing-then-passing C1 ownership gate, 8 engine tests, 103 runtime tests, 23 adapter tests, 168 J2K engine tests, 164 J2K adapter tests, and strict library clippy for all five crates.

## ADR-C008: J2K Metal uses a private engine layer and a neutral codestream handoff

Status: accepted; completes C2
Context: `j2k-metal-support` already owned checked Metal resources and runtime primitives, while `j2k-metal::compute` owned every J2K-specific transform, Tier-1, packetization, store, and resident path. The only production dependency from Metal transcode to the full adapter was a resident-output helper accepting `MetalEncodedJ2k` directly.
Decision: Name the J2K-specific module tree `engine` and keep it private over `j2k-metal-support`. Add the backend-neutral `DeviceCodestream` metadata contract in `j2k-core`, implement it for `MetalEncodedJ2k`, and make transcode resident handoff generic over that contract. Remove the full-adapter dependency and its migration exception.
Alternatives rejected: A new crate would move more than one hundred tightly coupled private modules without a second engine consumer; retaining the direct type dependency would preserve the forbidden edge; duplicating Metal output metadata in transcode would create competing range validation.
Consequences: Adapter → private engine → Metal support ownership is explicit in source, all unsafe-audit and coverage paths follow the engine owner, and the forbidden architecture inventory is empty. Existing Metal output methods and resident handoff call syntax remain valid.
Files affected: J2K Metal engine module tree and call paths, core traits, Metal encoded output, Metal transcode route/manifest/tests, coverage and source policies, architecture docs, and workplan.
Tests proving the decision: two red/green C2 architecture gates, 14-test architecture suite, 11-test source policy, J2K Metal all-target check, 6 route tests, 18 transcode library tests, strict clippy, and the required-runtime J2K Metal suite with 380 passed and 22 ignored.

## ADR-R001: Auto promotion is generated from validated evidence only

Status: accepted; completes R1
Context: CUDA and Metal route files mixed correctness support, runtime availability, handwritten benchmark thresholds and hashes, decisions, rejection strings, and profiling. Separate JPEG/J2K Metal batch heuristics also promoted Auto work without an artifact accepted by the routing verifier.
Decision: Split adapter routing into eligibility, availability, promotion, decision, rejection, and telemetry modules. Keep validated source IDs and workload boundaries in `docs/routing-promotion-evidence.json`; validate schema, backend ownership, six-operation coverage, SHA-256 form, and unique workload identities before deterministically generating checked-in Rust. Auto policies with no accepted evidence remain CPU-routed.
Alternatives rejected: Hand-copying constants and artifact hashes allows drift; runtime calibration is irreproducible; assigning undocumented heuristics a synthetic evidence identifier would misrepresent their status; retaining dead prechecked paths would preserve unsupported complexity.
Consequences: CUDA and J2K Metal promotion tables are mechanically reproducible and stale-checkable. JPEG Metal batch and J2K Metal region-scaled batch Auto acceleration are disabled pending new verified evidence. Metal resident host-output routing no longer extrapolates Gray8 512×512 or arbitrary larger RGB dimensions beyond the exact qualified matrix.
Files affected: CUDA/Metal routing modules and generated tables, J2K/JPEG Metal batch routing, Metal encode routing, promotion manifest/codegen, architecture policy, benchmark documentation, and durable workplan.
Tests proving the decision: red/green unqualified-route regressions, generator validation 4/4, stale check, routing policy 3/3, CUDA 164/164, J2K Metal 377 passed/22 ignored, JPEG Metal 228/228, and strict clippy for xtask and affected adapter libraries.

## ADR-P000: Performance decisions use validator-owned records

Status: accepted; completes P0
Context: Benchmark prose and Criterion directories recorded useful timings but did not mechanically require the complete environment, workload, metric, parity, and decision information needed across P1–P19.
Decision: Add a versioned JSON record validated by `cargo xtask gpu-experiment validate`. Require complete environment identity, an explicit applicable workload matrix, baseline and treatment per measured workload, exact output hashes/parity, conformance status, and a rationale. Require confidence-interval support, bounded representative regression, and proportional complexity before `promoted` is valid. Optional hardware metrics remain nullable rather than estimated.
Alternatives rejected: Prose-only templates cannot fail closed; making every compiler metric mandatory would make honest records impossible when vendor tools are unavailable; accepting stage-only improvements would violate the plan's end-to-end policy.
Consequences: Every following performance task can end in promoted, rejected, measured, or genuinely blocked evidence without weakening the information contract. Split-command runs remain attribution-only.
Files affected: xtask command/validator/tests, benchmark harness policy, performance-experiment documentation, and durable workplan.
Tests proving the decision: three red validation cases followed by 4/4 validator tests, the framework ownership policy, and xtask all-target strict clippy.

## ADR-P001: Reject row-serial Metal IDWT53 interleave/horizontal fusion

Status: rejected and removed; completes P1
Context: The baseline issued interleave, horizontal 5/3 lifting, and vertical lifting per level. A fused ordinary/repeated kernel performed interleave and exact horizontal lifting in one row-owned dispatch, retaining the original fallback behind the required switch.
Decision: Remove the candidate after same-host A/B. Although it was bit-exact across the required edge widths/heights and odd/even origins, the row-serial fused work reduced parallelism enough to regress resident and host-readback product paths by roughly 5–7% with non-overlapping confidence intervals.
Alternatives rejected: Promoting based on dispatch count would ignore the end-to-end regression; retaining a dormant fused path and switch would leave failed production complexity; claiming register or cache causality is unsupported because vendor counters were unavailable.
Consequences: The original three-dispatch implementation remains the production baseline. A future P1 revisit requires a genuinely tiled/cooperative design and a new record; it may not reuse this rejected result as evidence.
Files affected: only the validated experiment record and durable documentation remain after cleanup.
Tests proving the decision: source-wiring red/green test during the prototype, exact fused/unfused axis matrix, native IDWT parity, repeated hybrid batch parity, same-host Criterion A/B, post-cleanup shader integrity, 377-pass Metal library suite, and strict library clippy.

## ADR-P002: Reject whole-axis Metal IDWT97 fusion

Status: rejected and removed; completes P2
Context: The generic irreversible path issued interleave, horizontal scale and four lifting steps, then vertical scale and four lifting steps. A temporary threadgroup-staged prototype fused each bounded axis into one kernel while retaining the generic fallback behind the required switch.
Decision: Remove the candidate after same-host A/B. It was bit-exact for odd, degenerate, and representative 1023×767 axes and reduced the eligible per-level sequence from eleven physical dispatches to three, but the resident end-to-end benefit was only 0.34–1.57% and the host-readback interval included no change. Sixteen KiB of threadgroup memory, missing vendor occupancy/register evidence, and fallback rather than tiled halos beyond 4096 samples made the complexity disproportionate.
Alternatives rejected: Dispatch-count promotion would violate the end-to-end gate; retaining the dormant candidate would preserve an incomplete long-axis design; estimating occupancy from source would misstate unavailable evidence.
Consequences: Production retains the generic scale-plus-four-lifts path. The irreversible Criterion workload and output SHA-256 remain reusable, while all candidate kernels, pipelines, routing, switch, and prototype-only tests are removed.
Files affected: only the reusable benchmark workload, validated experiment record, and durable documentation remain after cleanup.
Tests proving the decision: shader-wiring red/green test during the prototype, required-runtime bit-exact fused/fallback tests, deterministic output SHA-256, same-host Criterion A/B, validator, post-cleanup Metal tests, strict clippy, formatting, and diff checks.

## ADR-P003: Reject line-serial Metal FDWT97 base fusion

Status: rejected and removed; completes P3
Context: The baseline performs four per-pixel lifting dispatches and one scale/deinterleave dispatch per active axis. A temporary one-thread-per-line prototype retained arithmetic order and the generic fallback while combining each axis into one dispatch.
Decision: Remove the candidate after same-host A/B. Fused and fallback paths matched the fractional CPU reference bit-for-bit and produced byte-identical single- and multi-level native codestreams, but the representative full encode regressed 1.23–2.38% with p=0.00. The isolated stage comparison was noisy and slower at the treatment point estimate.
Alternatives rejected: Promoting from the ten-to-two dispatch reduction would ignore the end-to-end regression; retaining a dormant serial path would preserve a known lane-stride performance cliff; estimating compiler resource behavior would exceed the timestamp-only public counters available on this host.
Consequences: Production retains the four-lift-plus-deinterleave axis path. A future P3 experiment must use the reviewed 256-sample core, four-sample halo cooperative design and produce independent evidence. The general transform benchmark remains.
Files affected: reusable transform benchmark, benchmark ownership test, validated experiment record, and durable documentation remain after candidate cleanup.
Tests proving the decision: shader-wiring red/green test, fallback and treatment exact fractional parity, native codestream parity, same-host stage/full-encode Criterion A/B, validator, post-cleanup tests, strict clippy, formatting, and diff checks.

## ADR-P004: Reject terminal IDWT/MCT/store specialization at design preflight

Status: rejected before implementation; completes P4
Context: The supported RGB8 product tail already fuses inverse MCT, clamping/conversion, and native-color store. Only the final vertical synthesis plane write/read remains. Component execution currently owns one complete component at a time and materializes its final plane before the three-component color destination runs.
Decision: Do not add the specialization. Exposing three transform-specific pre-vertical states would create a second cross-component execution graph, duplicate reversible and irreversible terminal lifting logic, and retain scratch lifetimes outside their current owner. The repeated-product dispatch probe confirms the eligible route but provides no timing basis for claiming the remaining plane traffic dominates.
Alternatives rejected: A universal tail mega-kernel violates the task scope; supporting only one transform would leave an arbitrary partial surface; estimating register pressure, occupancy, or traffic benefit from source would overstate unavailable evidence.
Consequences: The existing exact native-color MCT/store kernel and complete-plane IDWT fallback remain the sole production path. No switch, kernel, or dormant route was added. A future revisit requires a measured tail-stage profile and a narrow pre-vertical ownership contract before shader work.
Files affected: durable workplan only.
Tests proving the decision: existing exact irreversible RGB8 and OpenJPEG color parity, repeated decode-stage dispatch probe, and the P3/P4 Metal counter capability audit.

## ADR-P005: Promote explicit combined Metal input/MCT dispatch

Status: accepted; completes Metal P5
Context: Native staged encode materialized deinterleaved RGB planes, then offered a separate RCT/ICT accelerator stage, causing an avoidable second transfer/dispatch boundary.
Decision: Add a defaulted combined accelerator job used only for three-component MCT inputs. Metal implements signed/unsigned 1–16-bit loading, level shift, exact RCT or existing nested-FMA ICT in one pass; decline, switch, non-RGB, and no-MCT cases retain the old path.
Alternatives rejected: Implicit accelerator state would hide transform semantics; burdening all deinterleave calls with MCT flags would conflate the simple API; removing the fallback would weaken portability.
Consequences: Existing implementers remain source compatible through the default method. Full encode improves consistently by about 0.6–1.1% on both measured transform cells with exact codestreams.
Files affected: shared dispatch SPI, native staged orchestration/tests, Metal kernel/runtime/stage adapter/tests, benchmark, environment docs, and evidence.
Tests proving the decision: fake-accelerator orchestration, exact signed/unsigned 1–16-bit Metal matrix, forced fallback, end-to-end parity, strict affected-library clippy, and same-host Criterion A/B.

## ADR-P006: Do not redesign Tier-1 kernels without compiler-resource evidence

Status: tooling limitation recorded; completes P6 and blocks P7–P10 redesign on this host
Context: P7–P10 require moving substantial thread-private HT and Classic state into cooperative lane/shared-memory designs. The plan forbids treating source scratch-array arithmetic as measured spill behavior.
Decision: Record the unavailable metrics and stop those redesigns on this host. Public Metal counters expose timestamps only; reproducible per-kernel registers, private bytes, occupancy, active SIMD groups, spill loads/stores, and cache counters are unavailable. CUDA requires the absent Linux/NVIDIA lane.
Alternatives rejected: Undocumented translator output is not stable evidence; absence of xctrace spill events is not proof of zero spilling; throughput alone cannot establish the resource bottleneck.
Consequences: P7–P9 add no prototype complexity. P10 retains its existing style-0 specialization and generic/repeated fallbacks. Work may resume on a lane that can populate the required inventory.
Files affected: validated limitation record and durable workplan only.
Tests proving the decision: Metal counter capability/tool audit, P6 record validation, and unchanged Tier-1 regression suites in the canonical Metal lane.

## ADR-P011: Reject one-threadgroup-per-tile Metal packetization

Status: rejected and removed; completes P11
Context: The resident packet path built ordered headers and payload-copy descriptors, then launched a separate parallel payload-copy kernel. A temporary prototype assigned one threadgroup per tile, retained header/tag-tree mutation on lane 0, and copied each packet body cooperatively after a barrier.
Decision: Remove the candidate after same-host A/B. Classic and HT outputs were byte-exact across all five progression orders and the required inclusion, L-block, empty-packet, and multilayer cases, but full resident batch encode regressed 4.71–5.40% for Classic and 24.67–26.16% for HT with p=0.00.
Alternatives rejected: Promoting from one fewer dispatch would ignore large end-to-end regressions; retaining a dormant switch would preserve failed production complexity; attributing the result to occupancy or cache behavior would overstate unavailable public Metal counters.
Consequences: Production retains ordered packetization and the established parallel payload-copy dispatch. The reusable resident packetization benchmark keeps exact hash/decode probes, while cooperative kernels, pipelines, routing, switch, and prototype-only tests are removed.
Files affected: reusable J2K Metal benchmark and ownership test, validated experiment record, and durable documentation remain after cleanup.
Tests proving the decision: shader-wiring red/green coverage during the prototype, exact Classic/HT parity across five progression orders, native decode parity, same-host Criterion A/B, validator, and post-cleanup Metal verification.

## ADR-P012: Reject terminal Metal column-lift/quantization fusion

Status: rejected and removed; completes P12
Context: The terminal HTJ2K97 code-block path materialized four resident float subbands after column lifting, then reread them for quantization and code-block layout.
Decision: Remove the fused candidate after the priority JPEG-to-HTJ2K product A/B failed the promotion gate. The isolated terminal stage improved 5.04–7.13%, but the batch-16 512x512 product interval was -4.94% to +1.38% with p=0.36. Retain the exact product benchmark, the generic float-band path, and the staged implementation.
Alternatives rejected: Promoting from a favorable isolated stage would violate the product-path evidence rule; retaining the dormant kernel and switch would preserve unproven complexity; claiming a product improvement from an interval crossing zero would overstate the measurement.
Consequences: Production has no fused kernel, pipeline, or active switch. The retained benchmark verifies exact output, reports stage and transfer metrics, and records the correct 64x64 code-block workload. A >1024-dimension regression test protects the staged fallback boundary.
Files affected: Metal transcode benchmark, fallback regression, experiment record, environment documentation, and durable workplan remain after candidate cleanup.
Tests proving the decision: exact differential and product hash checks, >1024 fallback test, focused product benchmark execution, strict library/benchmark clippy, validator, and post-cleanup symbol audit.

## ADR-P013: Reject CUDA terminal column-lift/quantization fusion

Status: rejected and removed; completes P13
Context: The staged CUDA i16 path materialized four float subband buffers after final 9/7 column lifting, then reread them for quantization and resident HT code-block layout. A temporary direct-column candidate removed those buffers behind a split-process switch.
Decision: Remove the candidate after RTX A/B. It preserved exact preencoded payload and full-product codestream hashes and removed 25,165,824 bytes of product temporary bands, but the priority product absolute intervals overlapped (15.840–15.954 ms baseline; 15.603–15.867 ms treatment). The isolated column-plus-quantize interval regressed 4.21–7.53%, so the complexity was not proportional under the fail-closed promotion rule.
Alternatives rejected: Promoting from Criterion's small negative change interval would contradict the repository's absolute-interval rule; retaining a dormant direct kernel and switch would preserve rejected complexity; inferring cache benefit from eliminated traffic would overstate counters that were not captured.
Consequences: The i16 and F32 production routes both retain staged float bands and the generic >1024 row-lift fallback. The reusable single-path stage/product benchmark, exact hashes, resident-handoff metrics, and a 1032×8 i16 differential regression remain; all candidate kernel, ABI, launch, route, and switch code is removed.
Files affected: CUDA transcode engine staged route/tests, CUDA transcode benchmark, experiment policies, environment/unsafe documentation, validated experiment record, and durable workplan.
Tests proving the decision: RTX exact resident/host codestream and resident-handoff regression, all-output product parse/decode, split-process Criterion A/B, 1032-wide differential coverage, record validator, post-cleanup engine/adapter suites, strict clippy, unsafe audit, formatting, and candidate-symbol audit.

## ADR-P014: Reject tiled cooperative CUDA wide-axis IDWT

Status: rejected and removed; completes P14
Context: Axes above 512 samples used the generic CUDA IDWT path. A temporary 256-sample tiled route used one launch per lifting phase, a one-sample global-edge halo, and shared staging so 2592-wide rows did not need whole-line shared memory.
Decision: Remove the candidate after the complete RTX matrix. It was bit-exact and improved wide batch 1 by 50–60%, but wide batch 16 regressed 21.44–21.73% for reversible 5/3 and 39.63–40.03% for irreversible 9/7. Benchmark-forced 512 cells regressed 111–432%. Production selected by geometry, not batch, so promoting the wide shape would ship the measured batch-16 cliff.
Alternatives rejected: Adding a benchmark-derived batch selector would encode a narrow synthetic threshold without product evidence; promoting only batch 1 would violate the required matrix; retaining a dormant tiled route and force seam would preserve rejected complexity.
Consequences: Axes above 512 retain the generic two-dispatch route; existing bounded Cooperative53/97 routes remain. The single-path benchmark and exact wide batch-vs-single regression remain, while candidate kernels, modes, launch geometry, active switch, and candidate-only tests are removed.
Files affected: CUDA J2K engine IDWT selection/launch/device sources/tests, retained benchmark, benchmark policy, historical environment documentation, validated experiment record, and durable workplan.
Tests proving the decision: RTX odd-origin batch-16 bitwise parity for both transforms, eight-cell split-process Criterion A/B, record validator, candidate-symbol policy, post-cleanup 176-test engine suite, strict library/benchmark clippy, formatting, and release-bench linking.

## ADR-P015: Reject shared-staging CUDA irreversible FDWT97

Status: rejected and removed; completes P15
Context: The generic CUDA 9/7 forward transform reread overlapping source neighborhoods from global memory. A temporary candidate staged 32 output pairs by eight lines with a four-sample halo, using 2,304 bytes of shared memory horizontally and 3,072 bytes vertically while retaining the generic fallback.
Decision: Remove the candidate after exact RTX A/B. Coefficient and full-product hashes matched through three levels across 512, 1024, and 2592-wide batch 1/16 cases, and static source-load accounting fell sharply. However, the priority RGB8 512x512 batch-16 full encode had overlapping absolute intervals (5.573581–5.586208 s baseline; 5.567594–5.575540 s treatment, p=0.06). Wide batch 16 crossed no change (-0.130% to +1.879%), and 512 batch 1 significantly regressed 0.429–1.220%.
Alternatives rejected: Promoting from static load reduction or favorable non-priority cells would substitute a proxy for product evidence; selecting only favorable shapes would leave the required wide-batch cliff unresolved; retaining a dormant switch and trace route would preserve rejected complexity.
Consequences: Production retains the generic FDWT97 route. The reusable single-path stage/product benchmark retains exact framed hashes, dispatch accounting, and independent codestream decoding, while shared kernels, host selection, switch, trace seam, and candidate-only tests are removed.
Files affected: CUDA J2K engine generic route/tests, retained CUDA benchmark, experiment policy and historical environment documentation, validated experiment record, and durable workplan.
Tests proving the decision: RTX 67x71 two-level generic/shared f32 bit parity, six-cell three-level stage matrix, independently decoded batch-16 product, split-process Criterion A/B, record validator, post-cleanup engine/adapter tests, strict library/benchmark clippy, formatting, and candidate-symbol audit.

## ADR-P016: Reject CUDA RGB input fusion

Status: rejected and removed; completes P16
Context: CUDA RGB encoding used a deinterleave/level-shift dispatch followed by a separate RCT or ICT dispatch. A temporary candidate combined those operations into one kernel while retaining full-range signed/unsigned 1-16-bit support, exact reversible arithmetic, nested binary32 FMA ordering for ICT, and fallback for ineligible layouts.
Decision: Reject the candidate after exact RTX A/B. The isolated 512x512 RGB8 RCT stage improved 18.70-21.35% and ICT improved 52.59-53.27%, with physical input dispatches falling from two to one. The decision-grade product did not confirm that benefit: lossless RCT HTJ2K absolute intervals overlapped at 43.178660-43.383856 ms baseline versus 43.004971-43.371718 ms treatment, and lossy ICT overlapped at 194.879754-195.727981 ms versus 194.967121-196.525571 ms while its point estimate regressed.
Alternatives rejected: Promoting from isolated-stage speedup or one fewer dispatch would substitute proxy evidence for full-product throughput; selecting only RCT would still rely on overlapping absolute product intervals; retaining a dormant switch and second production route would preserve complexity unsupported by the product evidence.
Consequences: The specialized combined kernel, launch route, production selector, switch, and candidate-only route accounting were removed. Production retains the separate deinterleave and RCT/ICT stages; doc-hidden combined-input methods remain as two-dispatch compatibility wrappers. The reusable benchmark retains deterministic framed hashes, native-oracle stage parity, product dispatch accounting, codestream parsing, independent decoding, exact lossless output, and lossy PSNR validation.
Files affected: CUDA J2K engine/adapter input route and tests during the prototype, retained CUDA benchmark, experiment policy and environment documentation during cleanup, validated experiment record, and durable workplan.
Tests proving the decision: RTX signed/unsigned 1-16-bit exact matrix, resident-route exactness, deterministic separate/fused stage probes, split-process stage and product Criterion A/B, all product codestream parse/decode checks, experiment-record validation, post-cleanup adapter 165/165 and engine 172/172 suites, repo-lint 99/99, all-feature check, benchmark build, strict library/benchmark clippy, formatting, diff, and rejected-candidate symbol policy.

## ADR-P017: Reject CUDA final IDWT/store specialization at profile preflight

Status: rejected before implementation; completes P17
Context: P4 rejected terminal IDWT/MCT/store specialization without timing evidence. P17 retained the existing fused MCT/store route and added final-stage CUDA-event attribution plus a deterministic 512x512 RGB8 4:4:4 matrix covering Classic and HT, reversible 5/3 and irreversible 9/7, and batch 1 and 16. Two pre-existing irreversible half-tie defects in exact-native ICT and display-width MCT stores were repaired before accepting the matrix as correctness evidence.
Decision: Record a NO-GO and do not prototype the specialization. Final vertical plus the existing fused store represented only 0.641-1.784% of resident probe wall across all eight exact cells, while the plan required at least 10% before prototype authorization. Retain the profiling fields and fail-closed benchmark harness, but add no final-IDWT candidate, selector, switch, or fallback branch.
Alternatives rejected: Lowering the gate after measurement would make the decision rule non-fail-closed; promoting from isolated microseconds would ignore the measured resident product wall; a universal final-store mega-kernel remains outside scope and disproportionate to the observed tail share.
Consequences: Production keeps separate IDWT completion and the existing fused MCT/store kernel. The reusable profiler and deterministic matrix remain for future architecture changes. Following the P4 no-candidate precedent, P17 has no experiment JSON because no prototype or candidate A/B was run.
Files affected: CUDA J2K engine and adapter profiling, retained CUDA decode benchmark and policy, prerequisite store-rounding correctness paths/tests, and durable workplan documentation.
Tests proving the decision: RTX exact native RGB/RGBA and display-width RGB8/RGB16 half-tie regressions, exact CPU/CUDA parity and deterministic hashes for all eight P17 cells, two IDWT dispatches plus zero separate MCT and one fused store dispatch in every cell, Criterion confidence intervals, local engine/adapter suites, strict clippy, benchmark no-run, formatting, and diff checks.

## ADR-P018: Promote staged Metal JPEG encoding

Status: accepted; completes the Metal half of P18
Context: The production Metal baseline assigned one serial thread to each tile for color conversion, subsampling, FDCT, quantization, and ordered entropy coding, leaving only tile-level parallelism.
Decision: Precompute quantized coefficients with one MCU-parallel dispatch, then perform ordered entropy emission per tile. Promote the staged route after exact differential coverage and same-host product A/B, and remove the temporary switch and obsolete fused pipelines/kernels.
Alternatives rejected: Keeping the serial baseline would retain the measured bottleneck; parallelizing entropy within a restart segment would complicate byte ordering and stuffing; retaining a production A/B switch after promotion would preserve dead routing complexity.
Consequences: Gray and RGB 4:4:4/4:2:2/4:2:0 encoding use the staged Metal path. Representative RGB8 4:2:2 512x512 batch 8 improved 48.96–49.06%; batch-1 small and large cells also improved by more than 61%. Exact marker, stuffing, restart, padding, quality, determinism, and independent-decode behavior is retained.
Files affected: JPEG Metal staged shader, pipeline registry, encode orchestration/tests, benchmark, experiment record, and environment documentation.
Tests proving the decision: exact subprocess-isolated Gray/RGB sampling-quality-restart-edge matrix, six Metal encode integration tests, full library suite, shader integrity, strict clippy, release-bench linking, validator, and same-host Criterion A/B.

## ADR-P018-CUDA: Promote staged CUDA JPEG encoding

Status: accepted and cleaned up; completes P18
Context: The CUDA baseline assigned one serial work item to sampling, FDCT, quantization, and entropy for an entire tile. P18 retained ordered entropy but moved independent MCU coefficient work into a checked parallel precompute dispatch. During exact restart coverage, all 127 ordered restart markers and independent decoding passed, but the repository decoder reported `UnexpectedEoi` at an exact restart boundary when one to seven legal pad bits remained buffered.
Decision: Promote staged coefficient precompute plus ordered entropy after five exact RTX A/B cells. The priority RGB8 4:2:2 512x512 batch-8 interval improved from 6.795133-6.805057 s to 0.336067-0.336740 s (95.044-95.062%, 20.21x; p=0), and every other measured cell improved 93.324-96.500%. Repair `BitReader::consume_restart_marker` to probe an unprefetched marker when at most seven pad bits remain, while preserving eight-bit stuffed-data and wrong-marker behavior. Remove the serial production route and split-process switch after promotion.
Alternatives rejected: Retaining the serial route would preserve a roughly 20x representative bottleneck; parallel entropy across restart-independent segments was not needed to prove the dominant win; retaining the A/B switch after promotion would preserve dead production complexity. Inferring the missing Nsight counters was rejected after the driver returned `ERR_NVGPUCTRPERM`.
Consequences: CUDA baseline JPEG encode now has one checked staged route. It adds one dispatch and 32,768 to 16,777,216 bytes of coefficient scratch across the measured cells. Fixed-order exact coverage reproduces input/output digests `5fbd44a6890bfe562d66709eda023f0b5b8f942f0e113824399cfd39f06fe570` and `99b76d5a103ed958e4a4cdef80fb8e48cd8f2c6e28ababbf4ff787fea67ab314`; the decoder accepts restart-boundary pad bits correctly.
Files affected: CUDA JPEG engine staged planning/launch/execution and device kernels, adapter exact and restart tests, JPEG bit reader regression, retained benchmark, repository policy, experiment/environment/unsafe-audit records, and durable workplan.
Tests proving the decision: split-process deterministic five-cell Criterion A/B with dual decoding, exact serial/staged 16-frame matrix, restart-marker/order regressions, post-cleanup pinned RTX matrix and 512x512 restart-16 batch-1/batch-8 runs, engine 48/48, adapter 100 total, bit-reader 17/17, full `j2k-jpeg` 499 plus integrations, repo-lint 100/100, strict clippy, benchmark/test no-run builds, formatting, diff, and serial/switch symbol audit.

## ADR-P019: Reject Metal JPEG coefficient/IDCT defusion

Status: rejected and removed from production; completes the Metal half of P19
Context: A test-only split path decoded entropy to coefficient scratch and launched parallel IDCT for the full 4:2:0 texture case. P19 required profile-first product evidence before generalizing the route.
Decision: Measure the narrow existing split route and reject promotion. Baseline was 10.456–10.688 ms, treatment 10.621–13.562 ms, and Criterion reported -0.57% to +14.14% with p=0.30. Restore the production code to its prior test-only diagnostic state.
Alternatives rejected: Generalizing to more sampling and ROI/scaled modes before the narrow route won would add scratch traffic and routing complexity without evidence; retaining a dormant production switch would preserve a failed candidate.
Consequences: Production JPEG Metal decode remains fused. The split diagnostic continues to support targeted correctness investigation, while no production switch or candidate identifiers remain. The measured 512x512 batch-16 route required 12,681,216 bytes of coefficient scratch and added five private texture-path allocations.
Files affected: JPEG Metal benchmark probe, validated experiment record, and durable documentation remain after production cleanup.
Tests proving the decision: exact split-output hash, explicit split parity test, production 4:2:0 texture/boundary test, candidate-symbol cleanup search, validator, and same-host Criterion A/B.

## ADR-P019-CUDA-PACKED: Promote adaptive CUDA JPEG checkpoint geometry

Status: accepted and cleaned up; completes the CUDA launch-geometry prerequisite for P19
Context: CUDA JPEG decode already assigned one independent entropy checkpoint to each logical work item, but launched one thread per block. That geometry activated only one lane per warp. A split-process candidate instead packed checkpoints into 128-thread blocks. Profiling also exposed a separate eligibility defect: device batch capability reused the CPU-oriented `PreparedDecodePlan::matches_fast_tile_shape()`, whose intentional `restart_interval.is_none()` requirement rejected restart-coded input even though the CUDA packet and checkpoint decoder support it.
Decision: Use one thread per block for fewer than 128 checkpoints and 128 threads per block at or above 128. The priority 4:2:0 512x512 batch-16 product improved from a 21.080675–21.254977 ms absolute interval to 20.784920–21.026135 ms; the intervals do not overlap. All seven cells whose launch geometry changed had lower treatment point estimates, including 11.52–14.05% gains for the measured 4:2:2, 4:4:4, and 1024x1024 cells. Preserve the below-threshold serial geometry. Add a device-only fast-4:2:0 predicate that retains the geometry and sampling constraints but permits restart intervals; leave CPU eligibility and routing unchanged. Remove the historical A/B switch after the adaptive route passes single-path verification.
Alternatives rejected: Packing below 128 checkpoints would change the restart and 64x64 controls without product evidence; retaining block-1 for large checkpoint sets preserves severe warp underutilization; changing the CPU predicate would broaden an unrelated route; retaining the switch would leave promoted-candidate complexity in production.
Consequences: Production has one adaptive checkpoint-launch policy, no additional scratch, and unchanged dispatch/transfer accounting. Every A/B pair retained its exact input/output SHA-256, checkpoint count, workspace, and conformance result. The cleaned PTX is byte-identical to the pre-candidate artifact: 837,213 bytes, SHA-256 `b90b1a97152d08e0fe9e153304dd53bb7062119f86529475cc2dbbfefe4fc9e1`.
Files affected: JPEG CUDA device capability and restart regression, CUDA JPEG engine decode geometry/device indexing and tests, retained P19 benchmark/profiler, validated experiment record, and durable workplan.
Tests proving the decision: deterministic ten-cell split-process RTX A/B with exact hashes and CPU conformance; odd 4:2:0/4:2:2 seams, padded caller output, strict region/scaled rejection, and Auto fallback probes; restart-coded 4:2:0 device predicate RED/GREEN and strict session batch RTX regression; packed-boundary geometry tests; record validation; and post-cleanup strict real CUDA-Oxide build and focused RTX decode gates.

## ADR-P019-CUDA-DEFUSION: Reject CUDA JPEG coefficient/IDCT defusion

Status: rejected and removed; completes P19
Context: After adaptive checkpoint packing settled the fused baseline, profile timing still justified testing the plan's proposed split for 4:2:0. The prototype emitted exact MCU-major i32 coefficient scratch from checkpoint entropy threads, then launched a 128-thread block-IDCT deposit kernel before the existing conversion stage.
Decision: Reject and remove the split. It regressed every eligible product cell: +51.47% for priority 512x512 batch 16, +33.60% for batch 1, +16.14%/+10.18% for restart-16 batch 16/batch 1, +3.00% for 64x64, and +18.76% for 1024x1024. Unchanged fused 4:2:2 and 4:4:4 controls crossed zero. Exact hashes and conformance passed, but the split added one dispatch per tile plus 24,576 to 25,165,824 bytes of i32 scratch and three logical scratch accesses.
Alternatives rejected: Promoting an exact but uniformly slower route would violate the product gate; generalizing the losing split to 4:2:2/4:4:4 would add scratch and routing complexity without evidence; retaining dormant kernels, allocation, tests, or a switch would leave failed experimental complexity in production.
Consequences: The adaptive fused decoder remains the sole production route. Split kernels, route selection, scratch allocation, switch, and split-only tests were removed. Static ptxas evidence for the rejected entropy and IDCT kernels is retained rather than inferred dynamic metrics; Nsight Compute returned `ERR_NVGPUCTRPERM`. Post-cleanup regenerated the exact pre-candidate PTX hash and passed strict real CUDA-Oxide and focused RTX decode verification.
Files affected: CUDA JPEG engine and adapter only during the prototype; retained benchmark/profiler, validated rejection record, and durable workplan remain after cleanup.
Tests proving the decision: ten-cell split-process RTX A/B with six exact 4:2:0 treatment cells and four unchanged controls, deterministic output hashes, CPU conformance and routing probes, record validation, strict post-cleanup CUDA-Oxide build, owned decode 8/8, pitched output 1/1, 4:2:2/4:4:4 1/1, release-bench compilation, and empty rejected-symbol/source/switch scans.

## ADR-REL001: Stage the architecture transition as 0.10.0

Status: accepted; final publication remains separately authorized
Context: The generated comparison with published v0.9.0 contains 18 canonical defining-path changes whose root compatibility re-exports remain, plus two substantive changes required by the completed crate boundaries. `transcode_kernels_built` moved from the codec-neutral CUDA runtime into the CUDA transcode engine, and the Metal resident-codestream handoff became generic over `DeviceCodestream` so the transcode adapter no longer depends on the full Metal adapter.
Decision: Stage the workspace as 0.10.0 under a one-candidate pre-1.0 intentional-break transition. Keep v0.9.0 as the comparison baseline, ledger all 20 removed signatures with direct migrations, cover all 22 current libraries, and require v0.10.0 to become the next baseline before any later candidate.
Alternatives rejected: A 0.9.1 runtime shim cannot preserve the old availability semantics without restoring codec ownership or a dependency cycle; restoring the concrete Metal parameter would reintroduce the forbidden adapter dependency; undoing canonical defining-module ownership would reverse the hardening plan.
Consequences: All 23 publishable crates and exact internal pins use 0.10.0. The current published and supported release remains 0.9.0 until an independently authorized exact-SHA release workflow publishes v0.10.0. The intentional transition must be disabled after that publication.
Files affected: workspace and crate manifests, root and fuzz lockfiles, semver policy/tests, generated API report, review ledger, changelog, release/API policy, and durable workplan.
Tests proving the decision: locked workspace check, six locked metadata graphs, build-support 9/9, semver focused 30/30, complete xtask 387 plus integrations, `cargo xtask stable-api`, `cargo xtask semver`, repo-lint 101/101, strict xtask clippy, formatting, and diff check.
