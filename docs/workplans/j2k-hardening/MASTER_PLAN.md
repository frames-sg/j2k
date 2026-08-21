PLAN_ANCHOR=J2K-HARDENING-2026-08-18
AUDIT_BASELINE=f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5

# 1. MISSION

Transform the repository into a codebase with:

1. Typed, compile-time-safe boundaries.
2. One owner for every shared policy.
3. Clean separation between codec contracts, CPU/native implementation, GPU runtime, GPU codec engines, and public adapters.
4. Cohesive modules instead of physical file splitting that still behaves like a monolith.
5. No duplicated resource-accounting machinery.
6. No hidden public APIs used merely to bypass internal crate boundaries.
7. No benchmark thresholds copied manually into production code.
8. No speculative performance claims.
9. Measured Metal and CUDA kernel improvements.
10. Preserved bit-exactness, conformance, safety, allocation limits, and public behavior.

The final code should be simpler to reason about than the current code. A refactor that adds more types, traits, wrappers, or indirection without removing more complexity than it adds is a failure.


# 2. AUDIT BASELINE

The architecture and performance audit was performed against:

```text
AUDIT_BASELINE_COMMIT=f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5
```

Your current `HEAD` may differ.

At the beginning of the work:

```bash
git status --short
git rev-parse HEAD
git log --oneline --decorate -20
git diff --stat f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5..HEAD
```

If the repository has changed since the audit:

1. Reconcile each finding against current `HEAD`.
2. Mark findings as:
   - still present;
   - partially resolved;
   - fully resolved;
   - superseded by newer code;
   - newly worsened.
3. Do not blindly reproduce an old file layout if the repository has already moved.
4. Preserve all pre-existing user changes.
5. Never reset, discard, or overwrite unrelated work.


# 3. SOURCE-OF-TRUTH ORDER

When instructions or assumptions conflict, use this order:

1. Security and memory safety.
2. Correctness and conformance.
3. Applicable `AGENTS.md` files.
4. Existing canonical repository commands, CI, manifests, and tests.
5. This master plan.
6. Recorded architecture decisions.
7. Current implementation patterns.
8. Delivery speed.

Do not treat existing code as correct merely because it exists. Do not treat this plan as permission to violate repository invariants.


# 4. DURABLE MULTI-CONTEXT WORKFLOW

Create this directory immediately:

```text
docs/workplans/j2k-hardening/
```

Create and maintain exactly these four files:

```text
docs/workplans/j2k-hardening/
├── MASTER_PLAN.md
├── STATE.md
├── DECISIONS.md
└── EVIDENCE.md
```

Do not create a sprawling documentation hierarchy.


## 4.1 `MASTER_PLAN.md`

Copy the substantive execution sections of this prompt into `MASTER_PLAN.md`.

At the top, include:

```text
PLAN_ANCHOR=J2K-HARDENING-2026-08-18
AUDIT_BASELINE=f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5
```

The master plan is canonical. Do not rewrite it from memory after compaction.

You may:

- check off completed tasks;
- append narrowly scoped clarifications;
- mark tasks obsolete with a reason.

You may not silently delete unresolved work.


## 4.2 `STATE.md`

Keep this file short and operational. Use this exact structure:

```markdown
# Current State

Plan anchor: J2K-HARDENING-2026-08-18
Audit baseline: f1fdfb4b0edeb6cd060766a5cb8fd96f157a88c5
Current HEAD:
Current branch:
Current task ID:
Current phase:
Status: not-started | active | blocked | validating | complete

## Completed Since Last Checkpoint

## Files Changed

## Tests and Static Checks

| Command | Result | Notes |
|---|---|---|

## Benchmarks

| Experiment ID | Baseline | Treatment | Result | Evidence location |
|---|---:|---:|---:|---|

## Decisions Added

## Known Failures or Risks

## Exact Next Action

## Exact Next Command
```

Before any context compaction, interruption, or phase transition, update `STATE.md`.

The “Exact Next Action” must be concrete.

Bad:

```text
Continue refactoring.
```

Good:

```text
Replace the second backend-local lossless decomposition calculation in
crates/j2k-metal/src/encode/plan.rs with shared encode_geometry::lossless_levels,
then run the focused geometry parity test.
```


## 4.3 `DECISIONS.md`

Record architecture decisions using stable IDs:

```markdown
## ADR-A001: Prepared plan ownership

Status:
Context:
Decision:
Alternatives rejected:
Consequences:
Files affected:
Tests proving the decision:
```

Do not write essay-length ADRs. Record decisions that a future agent would otherwise have to rediscover.


## 4.4 `EVIDENCE.md`

Record durable evidence:

- baseline commands;
- test results;
- benchmark hardware;
- compiler versions;
- GPU model;
- driver or OS version;
- workload matrix;
- benchmark confidence intervals;
- kernel register/private-memory statistics;
- conformance results;
- output hashes;
- rejected optimization experiments.

Do not place temporary progress narration here.


# 5. COMPACTION PROTOCOL

At the beginning of every new context, including after automatic compaction:

```bash
cat docs/workplans/j2k-hardening/MASTER_PLAN.md
cat docs/workplans/j2k-hardening/STATE.md
cat docs/workplans/j2k-hardening/DECISIONS.md
git status --short
git rev-parse HEAD
```

Then:

1. Confirm the current task ID.
2. Inspect the files named in `STATE.md`.
3. Confirm that the working tree matches the recorded state.
4. Resume from the exact next action.
5. Do not restart the audit from scratch.
6. Do not rely on remembered chat history.

Before compaction:

1. Finish the smallest safe unit possible.
2. Run the focused tests for that unit.
3. Update `STATE.md`.
4. Update `DECISIONS.md` when architecture changed.
5. Update `EVIDENCE.md` when measurement changed.
6. Commit a coherent checkpoint if the repository’s workflow permits local commits.
7. Record the exact next command.

The repository state must be sufficient for another agent with no chat history to continue correctly.


# 6. WORKING RULES


## 6.1 Change discipline

For every behavior change:

1. Write or identify a failing regression test first.
2. Make the smallest complete correction.
3. Run focused tests.
4. Run broader affected-crate tests.
5. Run workspace checks appropriate to the platform.
6. Record the result.

For pure file moves:

1. Make the move behavior-neutral.
2. Preserve public names through re-exports where needed.
3. Do not mix the move with algorithm changes.
4. Confirm no benchmark or output changes.
5. Follow with behavior changes in separate commits.

Never mix all of these in one commit:

- module decomposition;
- public API redesign;
- kernel rewrite;
- route promotion;
- benchmark threshold change.


## 6.2 No agent-shaped abstractions

Do not introduce:

- `Any` or runtime downcasting where a typed enum or struct works;
- a trait used by one implementation;
- a generic abstraction whose only purpose is avoiding ten lines;
- “manager,” “coordinator,” “context,” “plan,” or “adapter” types without clear ownership;
- `common.rs`, `utils.rs`, or `helpers.rs` dumping grounds;
- public APIs solely to share code between workspace crates;
- new `#[doc(hidden)] pub` escape hatches;
- new `clippy::too_many_lines` exceptions unless a cohesive algorithm genuinely requires it and the reason is recorded;
- dead fallback paths retained “just in case”;
- feature flags for failed experiments.


## 6.3 Abstraction admission test

A new shared abstraction is allowed only when all are true:

1. At least two real callers share the same semantics.
2. The abstraction has one coherent owner.
3. It removes duplicated policy, validation, or ownership logic.
4. It reduces total branches, states, or error cases.
5. It preserves compile-time type information.
6. It can be tested independently.
7. It does not hide performance-critical allocations or dispatches.
8. Its name describes a concrete domain concept.


## 6.4 File-boundary test

A module is well bounded when:

- its data and behavior change for the same reason;
- its callers need a narrow surface;
- it does not re-export a large internal universe;
- it does not coordinate unrelated pipelines;
- it owns one meaningful concept;
- its tests can be focused.

Do not split a file merely every 500 lines. That creates a distributed monolith.


## 6.5 Performance honesty

Never claim an optimization without measurement.

A theoretical memory-bandwidth calculation is a ceiling, not a measured gain.

A CUDA optimization result does not establish Metal ROI.

A Metal optimization result does not establish CUDA ROI.

One fusion does not predict every other fusion.

Failed experiments must be removed or clearly documented as rejected. Do not leave dead experimental paths indefinitely.


## 6.6 Numerical behavior

Do not casually change:

- floating-point contraction settings;
- explicit `fma` order;
- exact integer lifting order;
- quantization division behavior;
- Tier-1 pass ordering;
- tag-tree ordering;
- bitstream byte stuffing;
- strictness or truncation behavior.

Structural kernel fusion must preserve the existing arithmetic order unless a separately reviewed numerical change is explicitly intended and all exact-parity and conformance tests pass.


# 7. INITIAL REPOSITORY INVENTORY — TASK G0


## G0.1 Read repository policy

Inspect:

```text
AGENTS.md
all nested AGENTS.md files
CONTRIBUTING.md
Cargo.toml
all affected crate manifests
CI workflows
xtask command definitions
benchmark documentation
conformance documentation
```

Use the repository’s canonical commands. Do not invent replacements when existing commands exist.


## G0.2 Capture baseline architecture

Record in `EVIDENCE.md`:

- workspace members;
- production dependency graph;
- public versus private crates;
- feature graph;
- current hidden public APIs;
- existing benchmark targets;
- existing route evidence generation;
- existing clone and architecture checks.

Useful searches:

```bash
rg -n "adapter_view|downcast_ref::<|core::any::Any" crates
rg -n "struct HostPhaseBudget|enum HostPhaseBudget|type HostPhaseBudget" crates
rg -n "lossless_device_encode_levels|j2k_lossless_decomposition_levels|MIN_LOSSLESS_DWT_DIMENSION" crates
rg -n "clippy::too_many_lines" crates xtask
rg -n "#\[doc\(hidden\)\]" crates
rg -n "Unsupported.*Request|HostAllocationFailed|HostAllocationTooLarge" crates
rg -n "AUTO_.*MIN|artifact|sha256|SHA-256" crates/*/src/routing.rs xtask docs
```


## G0.3 File and function inventory

Produce a machine-readable or Markdown inventory of:

- largest production Rust files;
- largest Metal files;
- largest CUDA-Oxide files;
- files with the highest number of functions;
- functions with excessive line counts;
- modules with high fan-in;
- modules with high fan-out;
- duplicate symbol names;
- duplicate policy functions;
- crate roots containing operational logic.

Exclude generated lookup tables from naïve god-file judgments, but identify whether generated data is mixed with handwritten control flow.


## G0.4 Baseline correctness

Run the repository’s canonical equivalents of:

- formatting;
- lint;
- focused crate tests;
- workspace tests;
- doc tests;
- package checks;
- architecture lint;
- clone audit;
- unsafe audit;
- T.803 or other conformance checks;
- public API checks.

Platform-gated tests must be recorded as:

- passed;
- failed;
- unavailable due to missing hardware;
- unavailable due to missing toolchain.

Do not report unavailable hardware tests as passed.


## G0.5 Baseline performance

Before architecture or kernel changes, capture available baseline results for:

```bash
cargo bench -p j2k-cuda --bench htj2k_decode
cargo bench -p j2k-cuda --bench htj2k_encode
cargo bench -p j2k-cuda --bench encode_stages
cargo bench -p j2k-cuda --bench auto_routing

J2K_METAL_PROFILE_STAGES=summary \
cargo bench -p j2k-metal --bench auto_routing

J2K_TRANSCODE_METAL_PROFILE_STAGES=summary \
cargo bench -p j2k-transcode-metal \
  --features bench-internals \
  --bench dct97

cargo bench -p j2k-jpeg-metal --bench compare
cargo bench -p j2k-jpeg-metal --bench encode_baseline
```

Confirm that the targets still exist before running them.

Do not use stale nonexistent targets such as:

```text
j2k-cuda --bench decode
j2k-metal --bench decode
```

If hardware is unavailable, still establish:

- benchmark compilation;
- benchmark input inventory;
- profiler instrumentation;
- static kernel compiler statistics where possible.


## G0 completion gate

Do not begin broad refactoring until:

- the baseline state is recorded;
- existing failures are distinguished from introduced failures;
- current `HEAD` is reconciled with the audit baseline;
- the durable workplan files are committed or otherwise safely persisted.


# 8. ARCHITECTURE GUARDRAILS AND BENCHMARK INFRASTRUCTURE — TASK G1

This phase must occur before large architectural changes so the repository can prevent backsliding.


## G1.1 Forbidden architecture edges

Extend the existing architecture-policy checks. The current graph synchronization test proves only that the documentation matches Cargo metadata; it does not prove the graph is good.

Add explicit forbidden-edge rules.

At minimum, enforce:

```text
support crates must not depend on public adapters
runtime crates must not depend on public adapters
transcode adapters must not depend on full public codec adapters merely for internal engines
test-support crates must not enter production runtime dependencies
public facade crates must not expose native concrete plans through Any
GPU adapters must not own shared encode geometry policy
```

The exact crate names may change during the refactor. Keep the rule semantic.


## G1.2 Ban plan type erasure

Add a repository test that rejects, in prepared-plan modules and adapters:

```text
core::any::Any
adapter_view
downcast_ref::<j2k_native::J2kReferenced...
```

This test may initially be introduced as an expected-failure inventory or allowlist, then tightened as migration progresses. Do not permanently normalize the current violation.


## G1.3 Ban duplicate phase-budget implementations

Add a repository check that ensures only the intended shared phase-budget implementation exists.

Initially inventory all definitions. After task A3, require exactly one shared implementation.


## G1.4 Long-function and root-module policy

Create a reviewed allowlist for existing long cohesive algorithms.

Reject increases in:

- production `clippy::too_many_lines` exceptions;
- operational functions in selected `lib.rs` roots;
- root-module function bodies above an agreed threshold;
- non-generated production files above a hard threshold without allowlist.

Do not make line count the only criterion. Use it as a trigger for review.

Suggested soft limits:

```text
crate root lib.rs: approximately 400 lines
ordinary production Rust module: approximately 1,200 lines
ordinary handwritten shader module: approximately 1,500 lines
```

Generated tables may exceed these limits if clearly isolated and labeled.


## G1.5 Extend clone analysis

The existing lexical clone analysis is insufficient because it scans Rust only and ignores many locations.

Add or extend checks for:

- Metal shader sources;
- CUDA-Oxide Rust;
- small repeated orchestration patterns;
- duplicate policy function names;
- repeated allocation wrappers;
- repeated route thresholds;
- duplicate error taxonomies.

Do not chase tiny helpers such as `min_u32` before meaningful duplication.


## G1.6 Add missing benchmark harnesses

Before kernel work, add current equivalents of:

```text
crates/j2k-metal/benches/decode_stages.rs
crates/j2k-transcode-cuda/benches/dwt97.rs
```

Use repository naming conventions if newer equivalents exist.

The Metal decode-stage harness must expose at least:

- entropy/Tier-1;
- dequantization;
- IDWT;
- inverse MCT;
- final store;
- readback where applicable;
- dispatch counts.

The CUDA transcode DWT97 harness must expose:

- unfused column lift plus quantization;
- fused column lift plus quantization;
- host-to-device staging when relevant;
- kernel time;
- end-to-end time.


## G1.7 Experiment kill-switch convention

Each kernel experiment must have a temporary, documented kill switch:

```text
J2K_<BACKEND>_DISABLE_<EXPERIMENT_NAME>=1
```

The kill switch must:

- select a real production-equivalent baseline;
- not change unrelated scheduling;
- be documented in the benchmark evidence;
- be removed after a final decision unless it remains intentionally supported for diagnostics.


## G1 completion gate

Progress: complete on 2026-08-20. Exact validation and remaining baseline-only
failures are recorded in `STATE.md` and `EVIDENCE.md`.

Proceed only when:

- architecture checks can encode the intended future state;
- missing benchmark harnesses exist or equivalent harnesses are confirmed;
- current benchmark and conformance baselines are documented;
- no new structural task can silently reintroduce the same violations.


# 9. CORRECTNESS-CRITICAL ARCHITECTURE

These tasks precede broad cosmetic cleanup because they remove real policy divergence and runtime type failure.


# 9A. SHARED ENCODE GEOMETRY — TASK A1

Progress: complete on 2026-08-20. Shared ownership, migration, regression,
workspace, and conformance evidence are recorded in `STATE.md` and `EVIDENCE.md`.


## Problem

Lossless decomposition-level and geometry policy is implemented independently in the facade and Metal encode path. The implementations are similar but not equivalent, creating a potential backend behavior divergence.


## A1.1 Choose the owner

Place shared, pure encode-geometry policy in one neutral location.

Preferred options:

```text
a dedicated module in j2k-types
a dedicated internal j2k-plan crate
another existing no-adapter neutral crate
```

Do not put backend-independent encode policy in:

```text
j2k-metal
j2k-cuda
j2k-cuda-runtime
j2k-metal-support
```


## A1.2 Shared policy must own

- maximum legal decomposition levels;
- default lossless level policy;
- progression-sensitive level policy;
- explicit maximum-level override;
- level dimension derivation;
- low/high subband dimensions;
- code-block exponent validation;
- total-bitplane derivation where backend-neutral;
- component/resolution packet ordering where backend-neutral.


## A1.3 Required test matrix

Test at least:

```text
width/height: 1, 2, 3, 31, 32, 63, 64, 65, 127, 128, 512, 1024
representative: 640×480, 1024×1024, 2592×1944
progression: LRCP, RLCP, RPCL, PCRL, CPRL
max levels: None, 0, 1, 2, 5, 255
component counts: 1, 3, 4
```

Verify:

- no overflow;
- legal level cap;
- facade and Metal choose the same geometry;
- native and accelerator paths consume the same plan;
- output codestream markers remain consistent;
- explicit below-64-dimension behavior is intentional and tested.


## A1.4 Migration

Migrate one consumer at a time:

1. facade policy;
2. Metal resident encode;
3. native encoder policy;
4. CUDA encode policy, if duplicated;
5. transcode geometry, only where semantically identical.

Remove old backend-local policy functions after all callers move.


## A1 completion gate

```text
one encode-geometry policy owner
no duplicate lossless decomposition algorithm
all backends consume shared validated geometry
focused regression tests pass
codestream and conformance behavior is preserved
```


# 9B. TYPED PREPARED PLANS — TASK A2

Progress: complete on 2026-08-20. Backend-neutral Classic/HT decode-plan
contracts now belong to `j2k-types`; facade wrappers expose typed immutable
borrows, native constructs/re-exports the plans, and CUDA and Metal consume the
same `Arc`-owned data without `Any`, downcasts, or native public-type leakage.


## Problem

The facade owns concrete native plans, erases them to `Any`, and the adapters downcast them back into the same concrete native types. This creates runtime failure paths without real decoupling.


## A2.1 Design constraints

The new plan representation must be:

- typed;
- immutable;
- data-oriented;
- backend-neutral;
- safe to share through `Arc`;
- zero-copy with respect to compressed codestream bytes;
- expressed using offsets and ranges into the original `Arc<[u8]>`;
- usable by native, CUDA, and Metal;
- free of runtime type erasure.

Forbidden:

```rust
Any
dyn Any
adapter_view()
downcast_ref()
opaque trait-object plan storage
```


## A2.2 Preferred representation

Use an explicit enum or separate typed fields:

```rust
pub enum PreparedDecodePlan {
    Classic(Arc<ClassicPreparedPlan>),
    Htj2k(Arc<Htj2kPreparedPlan>),
}
```

Shared image-level geometry should not be duplicated between Classic and HT plans.

A possible layout:

```text
prepared_plan/
├── mod.rs
├── image_geometry.rs
├── tile_geometry.rs
├── color_model.rs
├── wavelet.rs
├── classic.rs
├── htj2k.rs
└── payload.rs
```


## A2.3 Shared geometry should own

- grayscale, RGB, and RGBA classification;
- tile inventory;
- component count;
- component dimensions;
- wavelet transform;
- signedness and precision where relevant;
- ROI/reduction compatibility;
- retained payload ranges;
- source-index mapping where applicable.

Classic-specific plan data should own:

- code-block payload fragments;
- segments;
- packet geometry;
- progression information.

HT-specific plan data should own:

- cleanup ranges;
- refinement ranges;
- HT pass metadata;
- HT code-block geometry.


## A2.4 Migration order

1. Add neutral plan types.
2. Make native preparation produce them.
3. Make facade wrappers own them.
4. Migrate CUDA consumers.
5. Migrate Metal consumers.
6. Migrate plan caches.
7. Remove `adapter_view`.
8. Remove downcast-related errors.
9. Add guardrail that prevents reintroduction.


## A2.5 Required tests

Cover:

- Classic grayscale;
- Classic RGB;
- Classic RGBA;
- HT grayscale;
- HT RGB;
- HT RGBA;
- multiple tiles;
- multiple decomposition levels;
- ROI and reduction;
- strict failure on unsupported ROI maxshift;
- exact payload counts;
- exact payload ranges;
- exact source-index retention;
- plan-cache reuse;
- no compressed input cloning;
- no additional plan copying;
- adapter output parity before and after migration.


## A2 completion gate

The following searches must return no prepared-plan violations:

```bash
rg -n "adapter_view|downcast_ref::<j2k_native::J2kReferenced|core::any::Any" \
  crates/j2k crates/j2k-cuda crates/j2k-metal
```

Any remaining result must have a documented unrelated reason.


# 9C. SHARED HOST ALLOCATION PHASE BUDGET — TASK A3

Progress: complete on 2026-08-20. `j2k-core` now owns the only
`HostPhaseBudget` implementation and neutral error; CUDA runtime, J2K CUDA,
JPEG CUDA, and CUDA transcode share it with narrow adapter error conversion.


## Problem

`HostPhaseBudget` or equivalent logic exists independently in multiple CUDA-related crates.


## A3.1 Preferred design

Keep the common accounting algorithm in a neutral core crate.

Prefer one shared error:

```rust
pub enum HostPhaseError {
    AllocationFailed {
        requested_bytes: usize,
        what: &'static str,
    },
    LimitExceeded {
        requested_bytes: usize,
        cap_bytes: usize,
        what: &'static str,
    },
}
```

Adapter errors should wrap or translate this at a narrow boundary.

Avoid a highly generic mapper object or stored closure merely to vary error types.


## A3.2 Shared implementation should own

- exact-cap preflight;
- allocator-reported capacity accounting;
- live-byte accounting;
- capacity growth;
- vector with capacity;
- filled vectors;
- clone from slice;
- exact iterator collection;
- result iterator collection;
- product and sum overflow checks;
- aggregate byte accounting.


## A3.3 Required tests

- exact cap succeeds;
- one byte over fails;
- logical overflow saturates safely;
- allocator overcapacity is counted;
- failed allocation does not mutate accounting;
- incremental growth accounts only actual growth;
- zero-sized types consume zero bytes;
- adapter error classification remains correct;
- original error sources remain inspectable;
- CUDA runtime and transcode error semantics remain intact.


## A3.4 Migration

Migrate and remove local copies from:

```text
j2k-cuda
j2k-jpeg-cuda
j2k-cuda-runtime
j2k-transcode-cuda
```

Do not leave local compatibility wrappers that simply rename every shared method unless the wrapper meaningfully converts an error at the crate boundary.


## A3 completion gate

One shared phase-budget implementation exists. Repository lint enforces it.


# 10. SHARED ORCHESTRATION AND REDUNDANCY


# 10A. ONE DECODE-OPERATION MODEL — TASK A4

Progress: complete on 2026-08-20. CUDA image and tile adapters now construct
`DeviceDecodeRequest` and delegate to one operation entrypoint; Metal routes
all `MetalDecodeRequest` values through one operation entrypoint.

Unify the repeated operation family:

```text
full
region
scaled
region-scaled
```

Use one typed operation enum and one internal submit/decode function per adapter.

Trait methods may remain separate because the trait requires them, but each should construct an operation and delegate immediately.

Example:

```rust
fn submit_op(
    &mut self,
    session: &mut Session,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: DecodeOp,
) -> Result<Submission, Error>
```

Consolidate duplicated:

- backend validation;
- output-dimension calculation;
- profiling labels;
- fast-packet lookup;
- session locking;
- request creation;
- unsupported-device handling;
- CPU fallback selection.

Required tests:

- all four operations;
- all supported output formats;
- explicit CPU;
- explicit Metal;
- explicit CUDA;
- Auto;
- unavailable backend;
- unsupported operation;
- ROI clipping and scale geometry;
- output dimensions;
- warnings.


# 10B. ONE JPEG METAL BATCH-PLAN BUILDER — TASK A5

Progress: complete on 2026-08-20. Raw-byte and prepared-decoder sources now
resolve into one normalized source record consumed by one shared builder. The
builder owns output/sampling/restart validation, allocation and plan-owner
accounting, insertion, and execution baselines; parity tests cover every
required operation and rejection class.

Unify raw-byte and prepared-decoder batch planning.

The source-specific layer should only resolve one source into a normalized request.

A shared builder should own:

- destination vector budgeting;
- plan-owner accounting;
- output-dimension consistency;
- sampling-family consistency;
- restart restrictions;
- request insertion;
- execution-owner baseline;
- retained-byte accounting.

Suggested shape:

```rust
fn build_rgb8_batch_plan<S>(
    sources: &[S],
    context: &mut BatchBuildContext,
    resolve: impl FnMut(
        &S,
        &mut BatchBuildContext,
    ) -> Result<ResolvedBatchSource, Error>,
) -> Result<Rgb8MetalBatchPlan, Error>
```

Do not turn this into a generic framework used by unrelated codecs.

Required parity tests:

- raw bytes and prepared decoders produce equivalent normalized requests;
- duplicate decoder owners are counted once;
- plan cache bytes are accounted;
- mismatched output dimensions fail identically;
- mixed sampling family fails identically;
- unsupported restart-coded shapes fail identically;
- full/scaled/region-scaled paths remain equivalent.


# 10C. SHARED PREPARED-PLAN GEOMETRY API — TASK A6

Progress: complete on 2026-08-20. Classic and HT plans preserve their existing
codec-specific payload shapes and public enum layouts while sharing one
immutable image-geometry view for tile presence, component classification,
single-tile component geometry, dimensions, output region, and uniform wavelet
selection.

After A2, remove duplicated Classic and HT methods for:

- `is_grayscale`;
- `is_color`;
- `is_rgba`;
- tile emptiness;
- wavelet transform;
- shared component geometry.

Do not force Classic and HT payloads into one artificial generic payload trait. Share only image-level geometry with identical meaning.


# 10D. PACKING AND SAMPLE-CONVERSION POLICY — TASK A7

Progress: complete on 2026-08-20. Full-image and ROI byte packing now share one
sample-conversion policy and checked window traversal in `color/packing.rs`.
The 1-4 component direct 8-bit loops remain specialized; mixed/high-bit and ROI
paths share scaling, rounding, quantization, and bounds behavior.

Unify duplicated policy between full-image and region packing:

- bit-depth equality detection;
- sample scaling;
- rounding;
- output quantization;
- output bounds;
- row and region traversal.

Preserve specialized unrolled fast loops for one, two, three, and four components when they are faster and clearer.

The shared unit should be sample-conversion policy, not necessarily one generic inner loop.

Required tests:

- 1, 2, 3, 4 components;
- mixed bit depth;
- 8-bit fast path;
- high-bit scaling;
- signed samples;
- full versus equivalent full-size ROI;
- edge-aligned and non-aligned regions;
- exact output equality.


# 10E. ERROR TAXONOMY — TASK A8

Progress: complete on 2026-08-20. `j2k-core::CapabilityRejection` now provides
ten explicit internal categories. J2K/JPEG CUDA and Metal adapters translate
those reasons to their compatible public error variants at one boundary; an AST
guardrail rejects any direct production construction. Exact text,
classification, sources, and dual-cleanup diagnostics remain unchanged.

Replace repeated static rejection strings embedded throughout control flow with typed internal rejection reasons.

Example:

```rust
enum CapabilityRejection {
    UnsupportedFormat,
    UnsupportedSampling,
    UnsupportedBitDepth,
    UnsupportedOperation,
    MissingPreparedPlan,
    UnsupportedContainer,
}
```

Render human-readable text at the public error boundary.

Preserve:

- `CodecError` classification;
- truncation classification;
- unsupported classification;
- buffer-error classification;
- nested runtime sources;
- dual-error cleanup diagnostics;
- exact context needed for debugging.

Do not collapse all errors into one generic `Other`.


# 11. GOD-FILE AND MODULE-BOUNDARY REPAIR

Perform these as behavior-neutral decompositions first. Do not combine all files in one commit.


# 11A. SPLIT `j2k-types/src/lib.rs` — TASK M1

Progress: complete on 2026-08-20. The crate root is now 85 lines of private
module declarations, documentation, and compatibility re-exports. Stable values
are owned by focused implementation modules; the existing semver-visible
accelerator contract remains at its compatible root path as the low-level
dispatch SPI. No packetization-only output type existed to move, so an empty
`output.rs` was not introduced.

Target layout:

```text
j2k-types/src/
├── lib.rs
├── decode_payload.rs
├── encode_geometry.rs
├── transform/
│   ├── mod.rs
│   ├── mct.rs
│   ├── dwt53.rs
│   ├── dwt97.rs
│   └── quantization.rs
├── tier1/
│   ├── mod.rs
│   ├── classic.rs
│   └── htj2k.rs
├── packetization/
│   ├── mod.rs
│   ├── jobs.rs
│   ├── progression.rs
│   └── output.rs
├── dispatch/
│   ├── mod.rs
│   ├── accelerator.rs
│   ├── report.rs
│   └── error.rs
├── prepared_plan/
├── resident/
└── limits.rs
```

Preserve public names through re-exports initially.

Separate stable public value types from experimental accelerator SPI. If the SPI is not intended to be stable, place it behind an explicitly named unstable/internal feature or internal crate.

Do not continue describing semver-visible types as merely “for backend experimentation.”


# 11B. SPLIT `j2k-native/src/color.rs` — TASK M2

Progress: complete on 2026-08-20. The former 910-line file is now a 65-line
`color/mod.rs` with focused owners for types, metadata/channel ordering, output
planes, allocation, packing, palette resolution, ICC profiles, sYCC, and CIE
Lab. The A7 `packing.rs` owner is retained instead of renaming it to `pack.rs`.
Doc-hidden tuple handoffs were replaced by named field structs and the facade
was migrated without copying payloads.

Target layout:

```text
color/
├── mod.rs
├── types.rs
├── metadata.rs
├── output_planes.rs
├── allocation.rs
├── pack.rs
├── palette.rs
├── icc.rs
├── sycc.rs
└── cielab.rs
```

Ownership:

- `types.rs`: `ColorSpace`, owned and borrowed output types.
- `metadata.rs`: container/channel/color-space resolution.
- `pack.rs`: interleaving and native-to-output packing.
- `palette.rs`: palette index and palette resolution.
- `icc.rs`: ICC ownership and profile validation.
- `sycc.rs`: SYCC conversion.
- `cielab.rs`: CIE Lab conversion.
- `allocation.rs`: retained-allocation accounting.

Remove tuple-shaped facade escape types where a named neutral struct is clearer.


# 11C. SPLIT JPEG METAL COMPUTE — TASK M3

Progress: complete on 2026-08-20. `compute.rs` became a 193-line
`compute/mod.rs`; checked command boundaries, shader/pipeline registration, and
device/session/cache runtime state now have separate owners. The established
entropy, pack, full-batch, region-batch, single-decode/texture, status, and
encode modules retain their focused pipeline bindings. Crate callers now enter
named batch, single-decode, encode, or viewport domains instead of universal
root operations.

Current physical file splitting still routes most internals through a broad central `compute` namespace.

Target ownership:

```text
compute/
├── mod.rs
├── runtime.rs
├── pipeline_registry.rs
├── command.rs
├── status.rs
├── entropy/
├── idct/
├── color/
├── pack/
├── texture/
├── batch/
├── region/
└── encode/
```

Each pipeline module should own:

- its ABI types;
- pipeline lookup;
- resource validation;
- buffer bindings;
- dispatch geometry;
- status interpretation;
- focused tests.

Avoid a root `compute.rs` with dozens of imports and re-exports.

A caller should import from a domain module, not from a universal compute namespace.


# 11D. SPLIT JPEG METAL BATCH LOGIC — TASK M4

Progress: complete on 2026-08-20. The 1,018-line `codec_batch.rs` is now a
27-line semantic root with focused request/source/inspect/plan/accounting,
buffer-target, texture-target, and submission modules. `batch.rs` remains the
non-overlapping owner of queued requests, grouping, flushing, completions, and
`MetalSubmission`.

Target layout:

```text
codec_batch/
├── mod.rs
├── request.rs
├── source.rs
├── inspect.rs
├── plan.rs
├── owner_accounting.rs
├── buffer_target.rs
├── texture_target.rs
└── submit.rs
```

Do not create both `batch` and `codec_batch` hierarchies with overlapping ownership. Decide which owns public batch semantics and which owns execution internals.


# 11E. CLEAN CRATE ROOTS — TASK M5

Completion: complete on 2026-08-20. Operational benchmark helpers, JPEG codec
trait implementations and decode/upload routing, and CUDA macro definitions
now have focused private owners. The three roots are 108, 108, and 131 lines;
existing root API names remain compatibility re-exports.

Move operational logic out of:

```text
j2k-metal/src/lib.rs
j2k-jpeg-metal/src/lib.rs
j2k-cuda-runtime/src/lib.rs
```

A crate root should primarily contain:

- crate documentation;
- module declarations;
- controlled re-exports;
- very small top-level constructors;
- unavoidable trait implementations that clearly belong to the public surface.

Move benchmark-only operational functions into:

```text
bench_support
internal benchmark modules
examples
```

Avoid new hidden public APIs.


# 11F. SPLIT JPEG CAPABILITIES — TASK M6

Completion: complete on 2026-08-20. Request contracts, output geometry, CPU,
CUDA, and Metal eligibility, rejection classification, and path resolution now
have distinct owners. Public root exports and stable rejection text are
unchanged; `Auto` resolution remains correctness-only CPU selection, separate
from accelerator performance promotion.

Split:

```text
j2k-jpeg/src/capabilities.rs
```

into:

```text
capabilities/
├── mod.rs
├── request.rs
├── output_geometry.rs
├── cpu.rs
├── cuda.rs
├── metal.rs
├── rejection.rs
└── resolve.rs
```

Keep correctness eligibility separate from performance promotion.


# 11G. SPLIT FACADE ENCODE — TASK M7

Completion: complete on 2026-08-20. The 562-line operational facade file is a
49-line module root. Stable public entry points in `api.rs` delegate to focused
lossless, lossy, ROI, accelerator, high-bit, geometry, CPU, and validation
owners; existing contract, sample, allocation, resident, and tests ownership is
retained where already cohesive.

Split:

```text
j2k/src/encode.rs
```

into:

```text
encode/
├── mod.rs
├── api.rs
├── geometry.rs
├── cpu.rs
├── accelerator.rs
├── high_bit.rs
├── lossless.rs
├── lossy.rs
├── roi.rs
├── validation.rs
└── tests/
```

The public entry points should delegate to focused internal operations rather than owning validation, routing, geometry, execution, and result construction in one function.


# 11H. SPLIT `classic.metal` — TASK M8

Completion: complete on 2026-08-20. ABI, constants, generated QE/context
tables, shared state support, MQ/bypass primitives, pass logic, and decode
kernels are separate byte-preserving units. The original 77,178-byte source is
reconstructed exactly by the host composer. No empty `encode_kernels.metal`
was created: classic encode kernels already have the distinct established owner
`encode_bitstream_classic_kernels.metal` and were never part of
`classic.metal`.

Target layout:

```text
classic/
├── abi.metal
├── constants.metal
├── qe_table.metal
├── context_tables.metal
├── mq_decoder.metal
├── bypass_decoder.metal
├── pass_logic.metal
├── decode_kernels.metal
└── encode_kernels.metal
```

Generated tables must be isolated from handwritten control flow.

Avoid byte-order-sensitive textual concatenation where possible. Use explicit includes and independently compilable shader units.

The split must not alter arithmetic or table contents.


# 12. CRATE-BOUNDARY REPAIR

Do not start these high-churn tasks until A1–A8 and the relevant module boundaries are stable.


# 12A. NARROW CUDA RUNTIME — TASK C1

Progress 2026-08-20: migration step 1 is complete. The low-level runtime now
exposes validated static `CudaKernelSpec`, checked `CudaLaunchGeometry`, the
unsafe `CudaKernelParam` contract, parameter-pointer construction, and a
generic synchronous cached launch primitive. The cache accepts engine-owned
PTX/entry points without adding codec variants. Codec-family extraction and
adapter dependency migration remain unresolved; C1 is not complete.

Progress 2026-08-20: migration step 2 established the private
`j2k-cuda-jpeg-engine` dependency boundary and migrated `j2k-jpeg-cuda` to its
borrowed operation surface and domain-type imports.

Progress 2026-08-20: the JPEG family extraction is complete. JPEG ABI/types,
host allocation, byte views, validation, diagnostics, encode/decode
orchestration, kernel inventory, CUDA-Oxide projects, build flags, and all 45
pre-existing JPEG runtime tests now live in `j2k-cuda-jpeg-engine`. A private
shared CUDA build-support crate owns project staging and PTX packaging mechanics.
The low-level runtime contains no JPEG feature, module, re-export, cache key, or
kernel variant. J2K/HTJ2K/ML and transcode family extraction still remain, so
C1 is not complete.

Progress 2026-08-20: J2K-family migration step 1 is complete. The unpublished
`j2k-cuda-j2k-engine` now provides the borrowed-context boundary, owns adapter
feature forwarding, and mediates the session's classic/HTJ2K resource uploads.
The first complete vertical slice, J2K-ML, moved its domain types, validation,
launch, test, feature/build flag, and CUDA-Oxide project into the engine. The
runtime has no J2K-ML symbol, feature, cache key, kernel variant, or source.
Classic, HTJ2K, J2K encode/decode, and packetization remain in the runtime, so
C1 is still active.

Progress 2026-08-20: J2K-family migration step 2 is complete. A codec-neutral
queued compiled-kernel guard now retains pooled resources through completion
and can return them for deferred status readback. Classic Tier-1 ABI/types,
padding-free byte views, validation, table resources, synchronous and queued
orchestration, tests, feature/build flag, and CUDA-Oxide project moved into
`j2k-cuda-j2k-engine`. The runtime no longer contains a classic feature,
module, re-export, cache key, kernel variant, or PTX project. HTJ2K decode and
encode, J2K transform/store/encode, and packetization remain in the runtime, so
C1 is still active.

Progress 2026-08-20: J2K-family migration step 3 is complete. HTJ2K decode and
J2K dequantization ABI/types, padding-free byte views, payload/table resources,
output-region validation, synchronous and queued orchestration, status
interpretation, tests, feature/build flags, kernel inventory, and CUDA-Oxide
projects moved into `j2k-cuda-j2k-engine`. The runtime contains no HTJ2K-decode
or dequantization concept. J2K transform/store/encode, HTJ2K
encode/packetization, and transcode still remain in the runtime, so C1 remains
active.

Progress 2026-08-20: J2K-family migration step 4 is complete. J2K IDWT,
inverse MCT, final stores, forward transforms, quantization, classic encode,
HTJ2K encode, compaction, and packetization now live together in
`j2k-cuda-j2k-engine`. The encode families moved as one unit because HTJ2K
compaction/packetization and classic transform encode share the same generated
PTX module. The runtime has no J2K transform/store/encode or HT encode concept.

Progress 2026-08-20: C1 is complete. The unpublished
`j2k-cuda-transcode-engine` owns coefficient-domain reversible 5/3,
irreversible 9/7, fused quantization, transcode ABI/types, launch geometry,
timings, validation, tests, feature policy, and CUDA-Oxide project. The public
transcode adapter binds operations through a borrowed engine while retaining
the stable low-level `CudaContext`, buffers, and pools. `j2k-cuda-runtime` now
contains only codec-neutral driver, context, launch, memory, pool, staging,
completion, event, and diagnostic responsibilities.


## Goal

Make `j2k-cuda-runtime` a real low-level CUDA runtime rather than the owner of every CUDA codec concept.

The low-level runtime should own:

```text
Driver API loading
device/context
streams
events
module loading
kernel launch primitives
memory
buffer pools
pinned staging
completion
execution statistics
external allocation validation
```

It should not own public codec-domain concepts such as:

```text
JPEG MCU plans
JPEG baseline encode jobs
JPEG 2000 code blocks
HTJ2K packetization
DWT geometry
MCT jobs
transcode-specific band models
```


## Suggested staged split

```text
j2k-cuda-runtime
j2k-cuda-j2k-engine
j2k-cuda-jpeg-engine
j2k-cuda-transcode-engine
```

Names may differ. Keep internal crates unpublished if appropriate.

Do not add crates merely to move files. Each engine must represent a clean dependency boundary.


## Required dependency direction

```text
public CUDA adapter
    ↓
codec-specific CUDA engine
    ↓
CUDA runtime
    ↓
j2k-core / math / types
```

Never reverse this direction.


## Migration order

1. Establish low-level runtime interfaces.
2. Move one codec family at a time.
3. Keep public adapter APIs stable.
4. Preserve feature gating.
5. Preserve kernel packaging.
6. Preserve no-GPU compilation.
7. Update package includes.
8. Update docs and architecture tests.


# 12B. METAL RUNTIME AND ENGINE BOUNDARY — TASK C2

Progress: complete on 2026-08-20. `j2k-metal-support` remains the shared owner
of checked Objective-C resource creation, command queues/buffers/encoders,
pipeline loading, completion, buffer/texture access, allocation accounting,
and route-independent support errors. The J2K adapter's former `compute`
module is now the private `engine` boundary owning DWT/IDWT, MCT, Tier-1,
packetization, final stores, and resident encode/decode. A backend-neutral
`DeviceCodestream` contract removed the final `j2k-transcode-metal ->
j2k-metal` dependency without changing `MetalEncodedJ2k` behavior. The
forbidden dependency inventory is now empty and architecture tests enforce
both boundaries.

Use:

```text
j2k-metal-support or renamed runtime layer
```

for:

- checked Objective-C resource construction;
- device and queue ownership;
- command buffers;
- buffers and textures;
- pipeline cache;
- completion;
- allocation accounting;
- route-independent runtime errors.

Create a JPEG 2000 Metal engine layer for:

- DWT/IDWT;
- MCT;
- Tier-1;
- packetization;
- final stores;
- JPEG 2000 resident encode/decode.

Keep JPEG-specific MCU, Huffman, IDCT, and sampling concepts in JPEG Metal.


## Required outcome

`j2k-transcode-metal` must not depend on the entire public `j2k-metal` adapter merely to reuse internal compute functionality.

It should depend on a narrow runtime or JPEG 2000 engine.


# 13. ROUTING AND BENCHMARK-EVIDENCE ARCHITECTURE — TASK R1

Progress: complete on 2026-08-20. CUDA and J2K Metal routing now separate
eligibility, compile/runtime availability, evidence-backed promotion, core
decision, typed rejection, and telemetry ownership. The checked-in
`docs/routing-promotion-evidence.json` manifest records validated artifact
identifiers and exact workload boundaries; `cargo xtask promotion-codegen`
validates and deterministically emits both adapter tables, while `--check`
fails stale output. Architecture policy rejects handwritten Auto thresholds,
requires the routing domains and generator ownership, and the exact boundary
tests cover the generated Part 1, Part 15, and Metal host-output cells.
Unverified JPEG Metal batch and J2K Metal region-scaled batch heuristics were
removed from Auto instead of being represented as benchmark-qualified.

Separate these four concepts:

```text
correctness eligibility
runtime availability
performance promotion
telemetry
```

Target layout:

```text
routing/
├── eligibility.rs
├── availability.rs
├── promotion.rs
├── decision.rs
├── rejection.rs
└── telemetry.rs
```


## R1.1 Eligibility

Answers:

```text
Can this backend produce a correct result for this request?
```

It must depend on:

- format;
- dimensions;
- component count;
- bit depth;
- signedness;
- transfer syntax;
- container;
- ROI/reduction support;
- prepared-plan support.

It must not contain benchmark thresholds.


## R1.2 Availability

Answers:

```text
Is the requested backend compiled and usable on this host?
```


## R1.3 Promotion

Answers:

```text
For Auto, has this exact workload family been benchmark-qualified?
```

Promotion data must be generated from validated evidence.

Do not hand-copy artifact hashes and threshold constants into production logic.

Generate a checked-in table such as:

```rust
pub const PROMOTION_CELLS: &[PromotionCell] = &[
    // generated from validated evidence
];
```

The generator should:

- validate evidence schema;
- validate workload identity;
- validate hashes;
- validate operation coverage;
- emit deterministic Rust;
- record the source evidence identifier;
- fail if generated output is stale.


## R1.4 Telemetry

Telemetry observes decisions. It does not participate in deciding correctness.

Do not interleave profile-field construction with the core route match.


## R1.5 Typed rejections

Use typed rejection reasons and render strings at the boundary.


## R1 completion gate

- no handwritten benchmark artifact hashes in route code;
- eligibility contains no performance thresholds;
- promotion is generated;
- telemetry does not influence routing;
- route tests cover exact boundary cells;
- architecture lint enforces generated-table ownership.

Gate result: complete. Generator unit tests reject malformed hashes,
incomplete operation coverage, and duplicate workload identities; the stale
check, routing architecture policy, all three affected adapter library suites,
and strict clippy pass.


# 14. PERFORMANCE EXPERIMENT FRAMEWORK — TASK P0

Progress: complete on 2026-08-20. `cargo xtask gpu-experiment validate`
enforces a versioned JSON experiment record covering environment identity,
applicable workload dimensions, baseline/treatment pairing, available compiler
and runtime metrics, output hashes, exact parity, conformance, and the final
promotion/rejection judgment. Promoted records additionally require
confidence-interval support, bounded representative regressions, and a
proportional-complexity decision. `docs/performance-experiments/README.md`
defines the matrix and explicitly excludes split-command attribution runs from
final throughput comparisons.

Every optimization follows this sequence:

```text
PREFLIGHT
BASELINE
INSTRUMENT
IMPLEMENT
A/B
CORRECTNESS
PERFORMANCE
PROMOTE OR REJECT
CLEANUP
DOCUMENT
```


## P0.1 Benchmark environment record

For every performance result, record:

```text
commit
branch
dirty/clean state
CPU
GPU
RAM
OS
driver/runtime
Rust version
LLVM version
Metal compiler version or CUDA toolchain
build profile
feature flags
environment variables
input corpus hash
sample count
warm-up
measurement duration
```


## P0.2 Required metrics

Capture when available:

- end-to-end wall time;
- GPU timestamp time;
- stage time;
- dispatch count;
- host-to-device bytes;
- device-to-host bytes;
- device read/write traffic;
- register count;
- private/local memory per thread;
- shared/threadgroup memory;
- achieved occupancy;
- spill loads and stores;
- cache behavior;
- output hash or exact parity;
- conformance result.


## P0.3 Workload matrix

At minimum:

| Dimension | Values |
|---|---|
| Transform | reversible 5/3, irreversible 9/7 |
| Entropy | HT, Classic |
| Code block | 32×32, 64×64 |
| Image | 512×512, 640×480, 1024×1024, 2592×1944 |
| Batch | 1, 16 |
| Components | 1, 3, 4 |
| Output | native, RGB8, RGBA8 |
| Operation | full, ROI, half-scale |
| Axis | below and above 512 and 1024 |
| JPEG sampling | 4:4:4, 4:2:2, 4:2:0 |
| JPEG restart | none, present |


## P0.4 Performance acceptance

Follow the repository’s existing same-host confidence-interval policy. Apply at least the same rigor to GPU changes.

A complex optimization should generally not land unless:

- the priority end-to-end workload improves beyond noise;
- the confidence interval supports the improvement;
- representative workloads do not regress materially;
- correctness remains exact;
- added complexity is proportional to the gain.

A large stage-local gain with no end-to-end effect is not automatically worth permanent complexity.

Use split-command profiling only for attribution. Do not use command-split mode as the final production throughput comparison because it changes command-buffer overhead and cache behavior.


# 15. METAL TRANSFORM OPTIMIZATIONS


# 15A. METAL REVERSIBLE IDWT 5/3 FUSION — TASK P1

Progress: complete, rejected, and cleaned up on 2026-08-20. An ordinary and
repeated fused prototype preserved exact results across the required 1/2/3,
31/32, 511/512/513, 1023/1024/1025, and 2592 axis boundaries and reduced the
eligible interleave/horizontal sequence to one dispatch. Same-host Criterion
measurement on the repeated RGB8 512×512 batch-16 product path showed a
statistically supported end-to-end regression: resident +5.02% to +6.94% and
readback +4.52% to +6.72% (p=0.00). The fused kernels, pipelines, switch, and
tests were removed. The validated rejection record remains at
`docs/performance-experiments/P1-metal-idwt53.json`.


## Hypothesis

Metal reversible IDWT currently performs:

```text
interleave
horizontal
vertical
```

per level. Fusing interleave plus horizontal can remove one global plane write/read.


## Required implementation

Create fused ordinary and repeated/batched paths.

The implementation must:

- preserve exact integer lifting behavior;
- support odd and even dimensions;
- support one-sample axes;
- support multiple decomposition levels;
- support wide rows such as 2592 samples;
- avoid requiring the entire line to fit in threadgroup memory;
- use tiled staging with sufficient overlap/halo;
- retain a correct fallback;
- reduce dispatch count from three to two per level on eligible paths.

Temporary switch:

```text
J2K_<BACKEND>_DISABLE_FUSED_IDWT53_INTERLEAVE_HORIZONTAL=1
```


## Required tests

```text
width: 1, 2, 3, 31, 32, 511, 512, 513, 1023, 1024, 1025, 2592
height: analogous edge values
levels: 0 through legal maximum
components: 1, 3, 4
batch: 1, 16
full, ROI, scaled
```

Exact output must match the unfused implementation.


## Promotion gate

Retain only when measured end-to-end benefit survives the repository’s performance policy. Do not retain merely because dispatch count decreased.


# 15B. METAL IRREVERSIBLE IDWT 9/7 FUSION — TASK P2

Progress: complete, rejected, and cleaned up on 2026-08-20. A threadgroup-staged
prototype preserved bit-exact results and reduced each eligible per-level
interleave/9/7 sequence from eleven physical dispatches to three. Same-host
Criterion on irreversible HTJ2K RGB8 512×512 batch 16 measured a resident
improvement of 0.34–1.57%, while the host-readback interval crossed no change
(-2.65% to +0.05%). The prototype required 16 KiB of threadgroup memory and
fell back beyond 4096 samples instead of implementing the required tiled-halo
long-axis path; vendor occupancy and register evidence was unavailable. The
kernels, pipelines, switch, and prototype-only tests were removed. The
validated record remains at
`docs/performance-experiments/P2-metal-idwt97.json`.


## Hypothesis

Metal 9/7 synthesis uses many separate global-memory lifting and scale dispatches per level.


## Target

Fuse into approximately:

```text
one horizontal synthesis kernel
one vertical synthesis kernel
```

per level.


## Implementation constraints

- preserve current `fma` and arithmetic order;
- include scaling in the fused axis kernel;
- use threadgroup staging;
- synchronize between dependent lifting steps;
- use tiled halos for long axes;
- handle boundaries identically;
- retain generic fallback;
- avoid occupancy collapse from excessive threadgroup memory.

Temporary switch:

```text
J2K_<BACKEND>_DISABLE_FUSED_IDWT97=1
```


## Required evidence

- dispatch count before and after;
- threadgroup memory;
- register/private memory;
- occupancy;
- stage timing;
- end-to-end timing;
- exact output parity required by current tests.

Do not loosen floating-point pragmas to make fusion easier.


# 15C. METAL IRREVERSIBLE FDWT 9/7 FUSION — TASK P3

Progress: complete, rejected, and cleaned up on 2026-08-20. A bit-exact
one-thread-per-line base-fusion prototype reduced the two-axis per-level
sequence from ten physical dispatches to two. Same-host Criterion measured the
1024×768 three-level stage as noisy and slower at the treatment point estimate,
and the representative irreversible HTJ2K Gray8 512×512 full encode regressed
1.23–2.38% (p=0.00). The fused kernels, pipelines, switch, and prototype-only
test were removed. A future revisit requires the reviewed cooperative
256-sample tiled design with four-sample halos and new evidence. The reusable
transform benchmark and validated rejection record remain at
`docs/performance-experiments/P3-metal-fdwt97.json`.


## Target

Fuse the four lifting steps plus scale/deinterleave into one kernel per axis.

First implement:

```text
horizontal fused transform
vertical fused transform
```

Only after the base fused transform is stable should direct terminal quantization be considered.


## Correctness

Preserve:

- current operation order;
- exact coefficient layout;
- subband geometry;
- quantization inputs;
- OpenJPEG parity expectations;
- codestream roundtrip behavior.

Temporary switch:

```text
J2K_<BACKEND>_DISABLE_FUSED_FDWT97=1
```


## Follow-on experiment

For terminal detail bands, quantize directly from the final vertical kernel when the downstream pipeline does not require temporary float bands.

Do not force direct quantization into LL data required for the next level.


# 15D. FINAL VERTICAL IDWT + INVERSE MCT + STORE — TASK P4

Progress: complete as a design-preflight rejection on 2026-08-20; no prototype
was retained. The eligible RGB8 tail already uses a fused inverse-MCT/store
kernel, so the remaining opportunity is only the final vertical synthesis
write/read. Reaching it would require the component executor to expose and
retain three transform-specific pre-vertical states across its current
per-component ownership boundary, duplicate both terminal lifting families,
and add a parallel fallback orchestration path. The existing repeated-product
profile reported nine logical IDWT stages, one MCT stage, and one final-store
stage but did not expose per-stage timing; the host's public Metal counters
provided timestamps only. Because the plan explicitly permits rejection when
code duplication becomes excessive, no unsupported traffic or occupancy claim
was made and no candidate complexity was added.

This is a specialized experiment, not a required generic rewrite.

Initial scope:

```text
three components
4:4:4 geometry
RGB8 output
full decode
supported bit-depth range
```

Hypothesis:

```text
final vertical synthesis currently writes component planes
inverse MCT/store then rereads those planes
```

Prototype a final-stage kernel that performs:

```text
final vertical synthesis
inverse MCT
clamp/convert
RGB8 store
```

Do not build a mega-kernel that handles every output type, alpha mode, ROI, scale, and component layout.

Use route-specific specialization and retain existing fallback.

Reject the experiment if:

- register pressure or occupancy erases the traffic win;
- code duplication becomes excessive;
- the benefit is limited to noise;
- exact output cannot be preserved.


# 15E. INPUT DEINTERLEAVE + LEVEL SHIFT + RCT/ICT — TASK P5

Progress: Metal complete and promoted on 2026-08-20. CUDA P16 completed its
independent RTX correctness and A/B decision on 2026-08-21 and was rejected;
candidate cleanup is complete.
A defaulted combined accelerator hook is offered only for three-component MCT
inputs. The Metal kernel supports signed/unsigned 1–16-bit samples, exact RCT,
and the existing nested-`fma` ICT order; all other inputs retain the separate
fallback. Same-host RGB8 512×512 full encode improved 0.63–1.15% for RCT and
0.55–1.14% for ICT (both p=0.00) with identical codestreams. Evidence:
`docs/performance-experiments/P5-metal-input-mct.json`.

For common interleaved RGB encode inputs, test one input pass that:

```text
loads interleaved pixels
performs level shift
performs RCT or ICT
writes transformed component planes
```

Do not perform:

```text
interleaved input → temporary planes → reread planes for MCT
```

Preserve:

- signedness handling;
- bit-depth behavior;
- exact reversible RCT;
- current ICT arithmetic order;
- non-RGB fallback.

Implement independently for Metal and CUDA. Do not infer one backend’s result from the other.


# 16. TIER-1 AND ENTROPY OPTIMIZATIONS


# 16A. MEASURE PRIVATE MEMORY FIRST — TASK P6

Progress: complete as a documented tooling limitation on 2026-08-20. Public
Metal counters expose timestamps only; required registers, private bytes,
occupancy, active SIMD groups, spill loads/stores, and cache counters are not
reproducibly available. The Linux/NVIDIA lane is also unavailable. The
validated record is `docs/performance-experiments/P6-private-memory.json`; no
spill claim is inferred from source arrays.

Before rewriting HT or Classic kernels, collect compiler statistics.

For each key kernel:

```text
HT cleanup decode
HT encode
Classic decode
Classic encode
```

Record:

- private/local bytes per thread;
- registers;
- spill loads/stores;
- shared/threadgroup bytes;
- occupancy;
- active warps/SIMD-groups;
- code-block throughput.

Do not proceed from source-level scratch-array arithmetic alone.

If tools are unavailable, record that limitation and do not claim spill behavior as measured fact.


# 16B. HT CLEANUP DECODE — TASK P7

Progress: blocked by the P6 evidence gate on this host. No cooperative rewrite
was added without the required compiler-resource and occupancy baseline.

Current architecture appears to assign substantial serial work and scratch state to one GPU thread per code block.

Prototype:

```text
one warp or SIMD-group per code block
```

Suggested division:

- lane 0 performs serial entropy parsing where ordering requires it;
- lanes cooperatively clear and initialize state;
- lanes cooperatively decode or materialize independent quads where safe;
- lanes cooperatively write final coefficients;
- shared/threadgroup memory holds block scratch;
- several code blocks may share a threadgroup when occupancy allows.

Do not parallelize serial entropy state incorrectly.

Preserve:

- cleanup-only;
- SigProp;
- MagRef;
- pass-bucket routing;
- strict/truncated behavior;
- code-block source identity;
- per-job status.


# 16C. HT ENCODE — TASK P8

Progress: blocked by the P6 evidence gate on this host. No lane-cooperative
ordered-emit prototype was added without compiler-resource evidence.

Prototype one warp/SIMD-group per code block.

Parallelizable work:

- maximum-magnitude reduction;
- coefficient preprocessing;
- quad metadata generation;
- initialization;
- final output movement.

Serial work may remain on lane 0:

- ordered bitstream assembly;
- stateful entropy emission.

Do not keep large private stream arrays per thread if shared or staged storage is more appropriate.


# 16D. METAL CLASSIC TIER-1 ENCODE — TASK P9

Progress: blocked by the P6 evidence gate on this host. Existing scalar and
token-phase paths remain unchanged; no shared-state redesign is claimed from
source-level scratch sizes.

This is a high-priority architecture/performance target if compiler evidence confirms private-memory pressure.

Move code-block state from large per-thread private arrays into threadgroup memory.

Use lanes for:

- clearing;
- magnitude analysis;
- pass preparation;
- state updates that are safely parallel;
- output movement.

Keep MQ arithmetic serial where required.

Do not blindly fuse all existing plan, emit, and pack phases. Existing phase boundaries may preserve useful parallelism.

Required tests:

- all code-block style flags;
- 32×32 and 64×64;
- bypass;
- reset;
- termination;
- vertically causal;
- segmentation symbols;
- multiple segments;
- strict mode;
- truncation;
- exact decoded coefficients and codestream validity.


# 16E. METAL CLASSIC DECODE — TASK P10

Progress: blocked for new redesign by the P6 evidence gate. The existing
style-0 cooperative clear/store specialization and generic/repeated fallbacks
remain production behavior.

Extend cooperative execution to additional general-style and repeated paths when profiling supports it.

Do not delete existing route selection merely because one cooperative kernel exists.

Retain specialized paths when they benchmark better.


# 17. PACKETIZATION AND TRANSCODE


# 17A. METAL COOPERATIVE PACKETIZATION — TASK P11

Progress: complete and rejected on 2026-08-20. A one-threadgroup-per-tile
prototype kept ordered header/tag-tree state on lane 0 and cooperatively copied
packet bodies. Exact Classic/HT codestream and native-decode parity passed for
all five progression orders and the required inclusion, L-block, empty-packet,
and multilayer cases. Same-host RGB8 512×512 batch-16 full encode regressed
4.71–5.40% for Classic and 24.67–26.16% for HT (p=0.00), so the candidate,
switch, and prototype-only tests were removed. The established ordered
packetization plus parallel payload-copy dispatch remains production behavior.
Evidence: `docs/performance-experiments/P11-metal-cooperative-packetization.json`.

Model the stronger cooperative structure:

```text
one threadgroup handles one packet job
lane 0 builds ordered packet header/tag-tree state
threadgroup synchronizes
all lanes cooperatively copy packet body
```

Goal:

- eliminate descriptor writes and rereads where possible;
- remove a separate payload-copy dispatch;
- retain ordered header semantics;
- keep bulk copy parallel.

Do not replace the copy kernel with a single serial thread.

Required tests:

- LRCP;
- RLCP;
- RPCL;
- PCRL;
- CPRL;
- Classic and HT block coding;
- first inclusion;
- prior inclusion;
- L-block growth;
- empty packets;
- multiple layers;
- exact codestream packet bytes.


# 17B. METAL COLUMN LIFT + QUANTIZATION — TASK P12

Progress: complete and rejected on 2026-08-20. The fused kernel was exact and
improved the isolated terminal transform/quantize stage by 5.04–7.13%, but the
priority 512×512 batch-16 JPEG-to-HTJ2K product interval crossed no change
(-4.94% to +1.38%, p=0.36). The candidate kernel, pipeline, ABI, and switch
were removed; the staged production route, >1024 fallback, and reusable
end-to-end benchmark remain. Evidence:
`docs/performance-experiments/P12-metal-column-quantize.json`.

Fuse final 9/7 column lifting with quantization and direct code-block coefficient layout when the downstream path does not require float subband buffers.

Goals:

- avoid writing temporary float LL/LH/HL/HH bands;
- avoid separate quantization rereads;
- write quantized coefficients directly.

Constraints:

- retain LL float data when needed by the next level;
- preserve quantization division semantics;
- preserve subband ordering;
- preserve exact code-block layout;
- maintain fallback above staged axis limits.

Temporary switch:

```text
J2K_<BACKEND>_DISABLE_FUSED_COLUMN_QUANTIZE=1
```


# 17C. CUDA EXISTING COLUMN-LIFT FUSION A/B — TASK P13

Progress: complete, rejected, and cleaned up on 2026-08-21. On an RTX 4070
SUPER, the candidate preserved exact stage and product hashes and removed
25,165,824 bytes of temporary product float bands. The priority batch-16
512×512 JPEG-to-HTJ2K absolute intervals overlapped (baseline
15.840–15.954 ms; treatment 15.603–15.867 ms), while the isolated
column-plus-quantize interval regressed 4.21–7.53%. The candidate kernel, ABI,
route, and switch were removed; the staged i16/F32 routes, generic >1024 row
fallback, exact 1032-wide regression, and reusable product benchmark remain.
Evidence: `docs/performance-experiments/P13-cuda-column-quantize.json`.

Use the existing CUDA fused column-lift plus quantization kill switch only to measure that specific optimization.

Do not treat its result as the expected gain for unrelated fusions.

Add a dedicated CUDA transcode benchmark if one does not exist.

Record:

- fused versus disabled;
- stage timing;
- end-to-end timing;
- affected dimensions;
- batch size;
- temporary traffic;
- output parity.


# 18. CUDA TRANSFORM OPTIMIZATIONS


# 18A. CUDA WIDE-AXIS COOPERATIVE IDWT — TASK P14

Progress: complete, rejected, and cleaned up on 2026-08-21. The tiled route
was bit-exact at odd origins for 5/3 and 9/7 through 2592-wide batch 16. It
improved wide batch 1 by 50–60%, but required wide batch 16 regressed
21.44–21.73% for 5/3 and 39.63–40.03% for 9/7; benchmark-forced narrow cells
regressed 111–432%. A shape-only production selector could not avoid that
batch cliff, so tiled kernels, launch modes, switch, and force seam were
removed. Generic wide and existing <=512 whole-line cooperative routes remain.
Evidence: `docs/performance-experiments/P14-cuda-wide-idwt.json`.

Current whole-line cooperative paths may be limited by a maximum sample count, causing wide WSI rows to use generic fallback.

Implement a tiled cooperative route with:

- sufficient overlap/halo;
- correct edge extension;
- support for 2592-wide images;
- no requirement that a full row fit in shared memory;
- a generic fallback;
- launch-mode instrumentation.

Measure separately for:

- reversible;
- irreversible;
- batch 1;
- batch 16;
- narrow and wide axes.


# 18B. CUDA FDWT 9/7 SHARED STAGING — TASK P15

Progress: complete, rejected, and cleaned up on 2026-08-21. A 32-pair by
8-line shared-tile candidate with a four-sample halo was bit-exact through
three levels at 512, 1024, and 2592-wide geometry for batch 1 and 16. Static
source-load accounting fell substantially, and several transform-stage cells
improved, but the priority RGB8 512x512 batch-16 full encode had overlapping
absolute confidence intervals (5.573581-5.586208 s baseline versus
5.567594-5.575540 s treatment). Wide batch 16 crossed no change and 512 batch
1 regressed significantly. The shared kernels, route, switch, trace seam, and
candidate-only tests were removed; the generic route and reusable exact
stage/product benchmark remain. Evidence:
`docs/performance-experiments/P15-cuda-fdwt97-shared.json`.

The current dispatch structure may already be compact, but overlapping source neighborhoods can cause redundant global loads and arithmetic.

Prototype:

- shared-memory source tiles;
- cooperative low/high output calculation;
- halo handling;
- coalesced output;
- bounded shared-memory footprint.

This is not primarily a dispatch-count optimization.

Promote only if the memory/load reduction produces real throughput improvement.


# 18C. CUDA INPUT FUSION — TASK P16

Progress: complete, rejected, and cleaned up on 2026-08-21. The combined RGB8
CUDA route was bit-exact against native RCT and
nested-binary32-FMA ICT oracles and reduced the isolated input stage from two
physical dispatches to one. The isolated RCT stage improved 18.70-21.35% and
ICT improved 52.59-53.27%, but neither independent HTJ2K product comparison
met the fail-closed promotion rule. RCT absolute intervals overlapped
(43.178660-43.383856 ms baseline versus 43.004971-43.371718 ms treatment),
while ICT also overlapped and its point estimate regressed
(194.879754-195.727981 ms versus 194.967121-196.525571 ms). Evidence:
`docs/performance-experiments/P16-cuda-input-fusion.json`.

The specialized kernel, launch route, production selector, switch, and
candidate counters were removed. Separate deinterleave and RCT/ICT remain the
sole production route; doc-hidden combined-input methods remain only as
two-dispatch compatibility wrappers. The reusable single-path stage/product
benchmark retains exact hashes and independent decode validation.

Implement the CUDA counterpart of P5:

```text
deinterleave + level shift + RCT/ICT
```

Use an independent A/B result.


# 18D. CUDA FINAL STORE SPECIALIZATION — TASK P17

Progress: complete with a NO-GO at profile preflight on 2026-08-21. The direct
RTX lane completed the exact 512x512 RGB8 4:4:4 Classic/HT, reversible 5/3 and
irreversible 9/7, batch 1/16 matrix. Final-vertical-plus-store accounted for
only 0.641-1.784% of resident probe wall time across all eight cells, far below
the 10% prototype gate. No candidate kernel, selector, switch, or A/B route was
created. Existing fused MCT/store routes remain, while the reusable split-stage
profiling instrumentation and deterministic harness are retained.

Two half-tie correctness defects were repaired before accepting the profile:
the exact-native ICT store and the display-width MCT batch store now round
irreversible centered samples ties-to-even before level shift. The final matrix
passed exact CPU/CUDA output hashes in all eight cells. Following the P4
preflight precedent, there is no experiment JSON because P17 never authorized
a candidate or candidate A/B.

The gate is closed. Revisit only if a future system profile places at least 10%
of resident wall in this tail; do not build a universal final-store mega-kernel.


# 19. JPEG PERFORMANCE ARCHITECTURE


# 19A. JPEG ENCODE PIPELINE — TASK P18

Progress: complete and promoted on both backends: Metal on 2026-08-20 and CUDA
on 2026-08-21. Both promoted routes run an MCU-parallel
sampling/FDCT/quantization pass followed by ordered per-tile entropy. Metal
RGB8 4:2:2 512×512 batch 8 improved from 776.67–777.35 ms to
395.85–396.52 ms (-49.06% to -48.96%, p=0). On the RTX lane, the same CUDA
workload improved from 6.795133–6.805057 s to 0.336067–0.336740 s
(-95.062% to -95.044%, p=0), and all four additional batch, size, and restart
cells improved 93.324–96.500%. Exact codestream parity covers the required
Gray/RGB, sampling, restart, quality, odd-edge, padding, marker, stuffing,
dual-decoder, and determinism matrix. Post-promotion cleanup removed both
temporary switches and obsolete serial routes. The pinned CUDA single-route
matrix input/output digests are respectively
`5fbd44a6890bfe562d66709eda023f0b5b8f942f0e113824399cfd39f06fe570`
and
`99b76d5a103ed958e4a4cdef80fb8e48cd8f2c6e28ababbf4ff787fea67ab314`.
Evidence: `docs/performance-experiments/P18-metal-jpeg-staged-encode.json` and
`docs/performance-experiments/P18-cuda-jpeg-staged-encode.json`.

The current GPU baseline encode architecture must be profiled for under-parallelization.

Target staged architecture:

```text
1. parallel RGB to YCbCr and chroma subsampling
2. cooperative/separable 8×8 FDCT
3. quantization
4. ordered Huffman encoding per independent restart segment
5. parallel segment compaction and final assembly
```

Without restart intervals, entropy ordering may remain serial across the scan. That does not justify serializing color conversion, subsampling, FDCT, and quantization.

Keep the existing path as the baseline until the new path proves itself.

Required tests:

- grayscale;
- RGB;
- 4:4:4;
- 4:2:2;
- 4:2:0;
- odd dimensions;
- edge MCU padding;
- restart intervals;
- no restart interval;
- quality extremes;
- marker correctness;
- byte stuffing;
- decoder compatibility;
- deterministic output where the contract requires it.


# 19B. JPEG DECODE DEFUSION EXPERIMENT — TASK P19

Progress: complete on both backends. Metal was rejected on 2026-08-20 after
the exact split route added 12,681,216 bytes of scratch and produced an
inconclusive -0.57% to +14.14% interval (p=0.30). CUDA completed on 2026-08-21
on the direct RTX lane. A prerequisite launch-geometry experiment promoted an
adaptive checkpoint route: fewer than 128 checkpoints retain one thread per
block, while 128 or more use 128 threads per block. The priority 4:2:0
512×512 batch-16 absolute confidence intervals separated at
21.080675–21.254977 ms versus 20.784920–21.026135 ms; all seven cells whose
geometry changed had lower point estimates, with 11.5–14.0% gains on the larger
4:2:2, 4:4:4, and 1024×1024 cells. The settled fused route then justified a
4:2:0 coefficient/i32-scratch split prototype, but every eligible cell
regressed 3.00–51.47%; unchanged 4:2:2/4:4:4 controls were neutral. The split
route, scratch, switch, and candidate-only tests were removed, and the exact
pre-candidate PTX was restored. Evidence:
`docs/performance-experiments/P19-metal-jpeg-decode-defusion.json`,
`docs/performance-experiments/P19-cuda-jpeg-packed-checkpoints.json`, and
`docs/performance-experiments/P19-cuda-jpeg-decode-defusion.json`. The
superseded Apple-host CUDA limitation remains as P19-only history in
`docs/performance-experiments/P19-cuda-jpeg-decode-historical-blocker.json`.

The existing fused JPEG GPU decode may minimize temporary traffic but serialize entropy, IDCT, chroma handling, and color conversion in one thread.

Profile first.

Prototype only if justified:

```text
serial entropy parser emits block coefficients
parallel block IDCT
fused upsample/color/store
```

This adds coefficient traffic but may unlock substantial transform parallelism.

Keep both paths during A/B.

Reject if temporary traffic outweighs transform parallelism.

Test:

- baseline;
- restart-coded;
- 4:4:4;
- 4:2:2;
- 4:2:0;
- full;
- region;
- scaled;
- texture output;
- boundary repair.


# 20. THINGS NOT TO “FIX” BLINDLY

Do not automatically consolidate or rewrite:

- CPU, CUDA, and Metal arithmetic into one abstraction;
- backend-specific lifting implementations;
- explicit `fma` chains;
- exact integer lifting order;
- unrolled packing loops;
- backend-specific entropy kernels;
- distinct benchmark-qualified route thresholds;
- independent conformance tests;
- pass-homogeneous HT dispatch buckets;
- specialized kernels that demonstrably outperform a generic path.

Cross-language duplication is sometimes necessary.

Control it through:

- canonical constants;
- generated tables;
- shared geometry contracts;
- differential tests;
- conformance vectors;
- explicit algorithm documentation.

The primary duplication target is shared meaning and policy, not every repeated arithmetic expression.


# 21. TESTING LAYERS

Every task must identify which layers apply.


## Layer 1: Focused unit tests

Run tests for the changed module.


## Layer 2: Affected crate

Run the full crate test suite and lint.


## Layer 3: Dependent crates

Run direct dependents affected by public or internal contracts.


## Layer 4: Workspace

Run canonical workspace checks.


## Layer 5: Conformance

Run T.803, OpenJPEG parity, corpus, or exact-output tests applicable to the changed codec path.


## Layer 6: GPU runtime

Run hardware-gated tests on relevant hardware.


## Layer 7: Benchmark

Run same-host A/B with unchanged inputs and build settings.

A task is not complete when only Layer 1 passes if it changes a shared contract.


# 22. COMMIT AND CHECKPOINT POLICY

Use one logical commit per complete unit.

Good commit separation:

```text
test(encode): lock shared decomposition policy edge cases
refactor(types): add shared encode geometry
refactor(metal): consume shared encode geometry
refactor(plan): replace Any prepared plans with typed plans
refactor(core): centralize host phase budgeting
perf(metal): fuse reversible IDWT interleave and horizontal pass
bench(metal): record fused IDWT53 evidence
routing(metal): promote measured repeated RGB8 cells
```

Bad commit:

```text
refactor architecture and optimize GPU
```

Do not push or open a pull request unless explicitly instructed.

Do not rewrite unrelated history.


# 23. PHASE GATES


## Gate 0 — Baseline

Required:

- durable state files;
- current-HEAD reconciliation;
- baseline checks;
- benchmark inventory;
- existing failures documented.


## Gate 1 — Guardrails

Required:

- architecture policy catches intended violations;
- benchmark harnesses exist;
- clone and long-file policy established;
- no new violations can silently enter.


## Gate 2 — Correctness architecture

Required:

- shared encode geometry;
- typed prepared plans;
- one allocation phase budget;
- no `Any` plan boundary;
- no backend-local decomposition policy.


## Gate 3 — Orchestration cleanup

Required:

- one decode operation model;
- one JPEG Metal batch builder;
- shared plan geometry;
- reduced error and packing duplication.


## Gate 4 — Module boundaries

Required:

- major god files decomposed by ownership;
- crate roots mostly declarative;
- shader tables isolated;
- public names preserved or intentionally migrated.


## Gate 5 — Crate boundaries

Required:

- CUDA runtime narrowed;
- Metal runtime/engine boundary clean;
- transcode adapter no longer depends on full public adapter;
- forbidden-edge tests pass.


## Gate 6 — Performance

Required:

- every candidate measured;
- successful candidates promoted;
- failed candidates removed or documented;
- no correctness regression;
- no unsupported performance claim.


## Gate 7 — Routing

Required:

- eligibility separated from promotion;
- promotion generated from evidence;
- typed rejection reasons;
- no handwritten provisional artifact hashes.


## Gate 8 — Final validation

Required:

- formatting;
- lint;
- tests;
- docs;
- packaging;
- conformance;
- architecture lint;
- clone audit;
- public API review;
- GPU benchmarks where hardware is available;
- final clean working tree or precisely documented remaining changes.

Status on 2026-08-21: the user-approved 0.10.0 post-version validation sweep is
complete. Gates 0–7, all architecture tasks, P1–P19, local/CPU/Metal/CUDA
development validation, clean packaging, and all three changed-line coverage
thresholds pass. Gate 8 now requires only the explicitly authorized final
repository checkpoint and its SHA in durable evidence. Publication, hosted
exact-SHA evidence, tagging, and pushing remain separately authorized release
actions rather than implicit plan work.


# 24. FINAL DEFINITION OF DONE

The program is complete only when all of the following are true.


## Typed architecture

- No prepared-plan `Any`.
- No adapter plan downcasts.
- Shared prepared geometry is typed.
- Native produces plans; CPU/CUDA/Metal consume them.
- No impossible plan-type compatibility errors remain.


## Shared policy

- One encode-geometry policy.
- One phase-budget implementation.
- One owner for capability rejection reasons.
- One owner for batch-plan ownership accounting.
- Full/region/scaled/region-scaled operations share one internal model.


## File boundaries

- `j2k-types/src/lib.rs` is a re-export-oriented root.
- native color responsibilities are separated.
- JPEG Metal compute is split by pipeline ownership.
- JPEG Metal batch planning is cohesive.
- crate roots do not contain large operational implementations.
- Classic Metal tables and logic are separated.
- New long-function exceptions have not proliferated.


## Crate boundaries

- CUDA runtime is low-level.
- Codec-specific CUDA engines own codec concepts.
- Metal runtime is low-level.
- JPEG 2000 Metal engine owns JPEG 2000 kernels.
- transcode adapters depend on narrow engines rather than full adapters.
- architecture tests enforce dependency direction.


## Routing

- correctness eligibility is separate from performance promotion;
- promotion data is generated from validated evidence;
- telemetry is observational;
- rejection reasons are typed;
- route boundary tests cover generated cells.


## Performance

- Metal 5/3 fusion was measured and either promoted or rejected.
- Metal 9/7 IDWT fusion was measured and either promoted or rejected.
- Metal 9/7 FDWT fusion was measured and either promoted or rejected.
- private-memory statistics were collected for HT and Classic kernels where tools permit.
- Tier-1 redesigns were measured rather than assumed.
- Metal packetization was evaluated.
- Metal transcode column-quantization fusion was evaluated.
- CUDA wide-axis and FDWT staging were evaluated.
- JPEG encode parallelization was evaluated.
- JPEG decode defusion was evaluated.
- no failed experiment remains as unexplained production complexity.


## Evidence

`EVIDENCE.md` contains:

- baseline commit;
- final commit;
- hardware;
- commands;
- correctness results;
- conformance results;
- before/after benchmarks;
- rejected experiments;
- known hardware-validation gaps.


## Final report

Produce a final concise report containing:

```text
Architecture problems removed
Files and crates reorganized
Duplicated logic removed
Public API impact
Correctness evidence
Performance improvements
Rejected optimizations
Remaining known limitations
Exact validation commands
```

Do not claim the repository is “fully optimized.” State what was measured and what remains unknown.


# 25. FIRST EXECUTION INSTRUCTION

Begin with task `G0`.

Do not begin by editing kernels or splitting files.

Your first repository changes must be:

1. create the durable workplan files;
2. reconcile current `HEAD` with the audit baseline;
3. record baseline architecture and validation;
4. identify existing failures;
5. confirm the exact next task;
6. update `STATE.md`.

After completing `G0`, continue autonomously through the task order and phase gates. Do not repeatedly ask for permission.

Stop only for:

- a genuine correctness blocker;
- an irreconcilable conflict with pre-existing user changes;
- an unavailable required secret;
- a public API decision that cannot be inferred safely.

In that case:

1. document the blocker precisely in `STATE.md`;
2. record the exact files and decisions affected;
3. complete all nonblocked work;
4. do not use the blocker as an excuse to abandon unrelated tasks.
