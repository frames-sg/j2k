# j2k 0.7 Full Remediation and Release Runbook

Last updated: 2026-07-09

This is the canonical execution record for the 0.7 quality sweep. Git history
preserves the previous audit diary; do not create competing fixed, new, or
revision-numbered plans.

Priority order is security, correctness, maintainability, then delivery speed.
The release is blocked until every P1 and P2 item is complete and every P3
item is either complete or explicitly accepted with evidence.

## 1. Release verdict

Status: **BLOCKED — remediation in progress**

Captured baseline:

- Branch: main
- HEAD: a32b0fcbd897446f0adc4c6478d98158302aa7a4
- origin/main: 472a22eb201bf43172bf8ad303c10fbd5ec6ab41
- Divergence: 26 commits ahead, 0 behind
- Local annotated v0.7.0 tag peel: a32b0fcbd897446f0adc4c6478d98158302aa7a4
- Remote v0.7.0 tag at audit time: absent
- crates.io j2k release at audit time: 0.6.2
- Dirty baseline: 36 modified tracked files and one untracked test module
- Dirty diff at audit time: +1,652 / -196
- cargo-public-api: 0.52.0
- cargo-semver-checks: 0.48.0

The local tag is stale because it excludes all working-tree changes. It must
not be pushed or reused as release evidence.

No P0 issue was confirmed in the audit. Any later critical security finding
stops feature and refactor work until triage and remediation are complete.

## 2. Handoff capsule

Update this section whenever a task changes state. Keep it short enough to read
without loading the rest of the file.

- Current task: REC-001
- Parallel tasks: BUILD-001/TEST-001, METAL-001, SAFE-001
- Last completed task: DOC-001
- Last completed commit: none for this remediation run
- Candidate state: unfrozen
- Worktree expectation: dirty; all changes are being reconciled in place
- Last known green broad gates: pre-delta audit gates only; not candidate proof
- Current blockers:
  - workspace Clippy failure
  - unresolved test-ignore inventory
  - Metal runtime false-green
  - stale local tag and changelog
  - comparator test module is intentional but remains uncommitted
- Exact next local command:

      git status --short --branch

- After candidate freeze, derive the immutable SHA with:

      RC_SHA=$(git rev-parse HEAD)

- After candidate freeze, reconstruct remote evidence without editing this
  document:

      cargo xtask release-status --sha "$(git rev-parse HEAD)"

The document is updated and committed during remediation. Once the candidate
is frozen, do not edit it. Any tracked change creates a new candidate and
invalidates prior exact-SHA CI and GPU evidence.

## 3. Operating rules

1. Preserve every user change until its purpose is reconciled.
2. Never reset the worktree or overwrite unrelated edits.
3. Make one bisectable commit per high-risk task.
4. Do not combine unsafe-memory changes, dead-code deletion, and hot-path
   restructuring.
5. Do not add lint suppressions, ignored tests, silent fallback, or placeholder
   implementations to make a gate green.
6. Never delete a passing test without equivalent or stronger behavior
   coverage.
7. Use trash for local file deletion.
8. Do not push remediation commits individually. Push only the complete release
   candidate before exact-SHA verification.
9. Run the narrowest relevant tests after each task, then the full matrix at
   release freeze.
10. Target at least 80% changed-path coverage for measurable code and document
    narrow exclusions.
11. Preserve public behavior, exact errors, byte output, command ordering,
    resource retention, and allocation reuse during structural refactors.
12. If a task changes a stable API, update the API-diff report and changelog in
    the same task.

## 4. Audit rubric

The sweep treats these as likely AI-codebase failure modes:

- False-green tests: ignored tests, conditional early returns, approximate
  test-count floors, and CI labels mistaken for hardware execution.
- Copy/paste policy holes: duplicated path lists, divergent workflow snippets,
  and checks that prove a command exists without proving it runs.
- Speculative or zombie code: large test-only implementations, unreferenced
  kernels, and inventories that positively require dead code.
- Unsafe abstraction drift: duplicated raw-buffer helpers with different trait,
  aliasing, or lifetime requirements.
- Comprehension debt: thousand-line orchestrators, deeply nested resource
  management, and mixed planning/execution/readback/reporting.
- Clone drift: independently copied error mapping, category inference, staging,
  and packet-plan logic that already differs.
- Release theater: stale tags, empty changelogs, permissive semver flags,
  unverified exact SHAs, and publish jobs that validate only a small preflight.
- Coverage theater: excluding complete accelerator crates instead of measuring
  host logic and requiring named hardware evidence for device paths.

Duplication is not automatically a defect. Domain symmetry is accepted when
abstraction would couple unrelated public APIs, hide backend constraints, or
make code harder to review.

## 5. Task dashboard

| ID | Severity | Status | Depends on | Outcome |
|---|---:|---|---|---|
| DOC-001 | P1 | complete | — | Canonical runbook replaces stale diary |
| REC-001 | P1 | in progress | DOC-001 | Every dirty file is explained and tested |
| REL-001 | P1 | pending | REC-001 | Stale tag and changelog reconciled |
| BUILD-001 | P1 | in progress | — | Workspace Clippy restored without allows |
| TEST-001 | P1 | in progress | — | All 38 new ignores have exact dispositions |
| METAL-001 | P1 | in progress | TEST-001 | Hosted compile and fail-closed runtime lanes |
| POLICY-001 | P1 | pending | BUILD-001 | Public API scan and strict lane repaired |
| CI-001 | P1 | pending | METAL-001 | Shared exact-SHA workflow verifier |
| PUB-001 | P1 | pending | CI-001, POLICY-001 | Publish requires all candidate evidence |
| SEM-001 | P1 | pending | REC-001 | Reviewed API diff replaces blanket major |
| COV-001 | P2 | pending | METAL-001 | Accelerator host logic is measured |
| SAFE-001 | P1 | in progress | BUILD-001 | One truthful Metal buffer-access boundary |
| ERR-001 | P2 | pending | BUILD-001 | Neutral native decode classification |
| DUP-001 | P2 | pending | ERR-001 | Genuine clones consolidated |
| ADAPT-001 | P2 | pending | DUP-001 | Test-only adaptive router removed |
| CUDA-001 | P2 | pending | ADAPT-001 | Five suspected kernel orphans resolved |
| STR-001 | P2 | pending | SAFE-001, CUDA-001 | Resident encoder split safely |
| STR-002 | P2 | pending | STR-001 | Direct stacked batch split safely |
| STR-003 | P2 | pending | STR-002 | Native single-tile encoder split safely |
| TOOL-001 | P3 | pending | DUP-001 | Adoption report model/render split |
| FINAL-001 | P1 | pending | all above | Clean local release matrix |
| RC-001 | P1 | pending | FINAL-001 | Immutable exact-SHA candidate |
| TAG-001 | P1 | pending | RC-001 | Annotated tag and guarded publication |

## 6. Phase 0 — reconcile the worktree

### DOC-001 — canonical runbook

Intent:

- Replace the stale 3,805-line status diary in place.
- Keep a single task-oriented source of truth.
- Make compaction and agent handoff possible from the first screen.

Acceptance:

- This file contains the baseline, handoff capsule, task dashboard, detailed
  acceptance criteria, gate matrix, and immutable release sequence.
- No competing plan file is created.
- A new agent can identify the current task and next command immediately.

Evidence:

- 2026-07-09: replaced the 3,805-line diary with this 703-line task runbook.
- 2026-07-09: git diff --check passed for this document.

### REC-001 — dirty-delta reconciliation

Actions:

1. Inventory each modified path by user intent and map it to a task.
2. Adopt crates/j2k-compare/src/fixture_compare/tests.rs as tracked work because
   the tracked module declaration already requires it.
3. Add the repository SPDX header to that test module.
4. Run the comparator unit and subprocess tests.
5. Revert only changes that cannot be tied to intended behavior, a regression
   test, or a release requirement. Reverts must be surgical and documented.

Acceptance:

- No unexplained tracked or untracked path remains.
- Every retained behavior change has regression protection.
- The comparator test module is tracked, licensed, and green.
- git diff --check passes.

Current evidence:

- The comparator module is intentionally retained and now has the repository
  SPDX header.
- cargo test -p j2k-compare --lib fixture_compare::tests -- --nocapture:
  6 passed, 0 failed, 0 ignored.

### Captured dirty-path ownership

This maps the audit-time delta. Later edits inherit the owning task unless the
dashboard records an explicit transfer.

| Paths | Owner | Intent |
|---|---|---|
| Cargo.lock | SAFE-001 | Metal support test/dependency reconciliation |
| j2k-compare encode_compare and fixture_compare modules | REC-001, DUP-001 | Preserve comparator behavior and canonical category/fixture ownership |
| j2k-compare bench_harness tests | TEST-001 | Restore CPU-capable subprocess coverage |
| j2k-jpeg-metal source and integration tests | TEST-001, METAL-001, SAFE-001 | Expanded pure/runtime coverage and checked buffer access |
| j2k-metal-support manifest and library | SAFE-001 | Shared ABI-checked buffer access |
| j2k-metal production modules | METAL-001, SAFE-001 | Runtime coverage and checked GPU readback/write paths |
| j2k-metal unit/integration tests | BUILD-001, TEST-001, METAL-001 | Compile cleanup and exact hardware-test inventory |
| j2k-transcode-metal integration tests | METAL-001 | Fail-closed runtime validation |
| j2k encode_lossless tests | TEST-001 | Restore CPU codec-distribution coverage |
| docs/env-vars.md | METAL-001 | Required-runtime environment contract |
| xtask main and repo-lint support | METAL-001, POLICY-001, CI-001 | Gate orchestration and policy enforcement |
| this runbook | DOC-001 | Canonical remediation and handoff record |

### REL-001 — release metadata truth

Before changing the tag, recheck:

- git ls-remote tags on origin
- GitHub Releases
- crates.io versions for every publishable package

Decision:

- If no remote/public 0.7.0 exists, delete only the stale local v0.7.0 tag and
  create it again after exact-SHA validation.
- If any public or remote 0.7.0 artifact exists, never move it. Change the
  target to 0.7.1 and update all workspace, changelog, plan, and workflow
  references.

Populate the changelog from the reconciled diff. Do not claim that Unreleased
is empty.

## 7. Phase 1 — eliminate release false-greens

### BUILD-001 — workspace Clippy

The failing test modules place runtime guards before local imports/items,
triggering clippy::items_after_statements.

Actions:

- Move item declarations before executable guards in:
  - j2k-jpeg-metal encode integration tests
  - j2k-jpeg-metal core-trait integration tests
  - j2k-metal encode unit tests
  - j2k-metal encode-kernel unit tests
- Do not add allow attributes.

Acceptance:

    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings

### TEST-001 — exact ignored-test inventory

The current delta adds 38 ignores. Resolve them as follows.

Restore 19 CPU-capable tests to normal execution:

- all 13 j2k-compare benchmark-harness subprocess tests
- all three j2k CPU codec-distribution tests
- these three CPU-only j2k-metal routing-policy tests:
  - auto_htj2k_large_host_output_stays_cpu_for_single_frame
  - auto_htj2k_kodak_sized_rgb_host_output_stays_cpu_for_single_frame
  - auto_htj2k_gray_host_output_stays_cpu_for_single_frame

Keep exactly 18 j2k-metal library tests ignored only because they require real
Metal hardware:

- all 11 direct-plan compute tests in the audited hardware group
- all five direct.rs cleanup-kernel tests
- auto_htj2k_padded_private_rgb8_single_host_output_stays_cpu
- auto_htj2k_padded_private_gray8_single_host_output_stays_cpu

The strict public-API architecture test may remain ignored in the default suite
only because a required pinned strict lane runs it.

Add a repository policy assertion for exact names and groups. Approximate count
floors alone are insufficient.

Acceptance:

    J2K_REQUIRE_METAL_RUNTIME=1 \
      cargo test -p j2k-metal --lib -- --ignored --nocapture

The command must run exactly 18 tests, pass all 18, and print no skip marker.
The 19 restored tests must execute in their default CPU-capable suites.

### METAL-001 — compile lane versus hardware lane

Developer interfaces:

- cargo xtask metal-compile
- cargo xtask release-metal

metal-compile must run hosted compilation, Clippy, and pure tests across:

- j2k-metal-support
- j2k-jpeg-metal
- j2k-metal
- j2k-transcode-metal
- facade integration with Metal features

release-metal must:

- run only on macOS with a usable Metal device
- set J2K_REQUIRE_METAL_RUNTIME=1
- include support, JPEG, J2K, transcode, and facade runtime tests
- run the exact 18-test ignored j2k-metal library inventory
- reject J2K_GPU_TEST_SKIPPED and equivalent markers
- reject missing devices, missing binaries, zero selected tests, cancellation,
  and partial suites
- report named tests, not just broad count floors

GPU workflows call xtask rather than duplicating long shell sequences.

### POLICY-001 — public API policy repair

Actions:

- Remove the duplicate crates/j2k/src scan entry.
- Add crates/j2k-jpeg/src to the fast public-surface scan.
- Keep cargo-public-api pinned to 0.52.0.
- Require strict repository lint on macOS.
- Make normal repository lint verify the exact workflow, tool version, command,
  and required-job wiring.

### CI-001 — one exact-SHA verifier

Move inline GitHub API verification into one repository-owned Python program
using only the standard library unless a dependency is already required.

Required inputs:

- repository
- workflow file or immutable workflow ID
- exact commit SHA
- required job names

Required behavior:

- paginate all relevant runs/jobs
- match exact workflow identity and head SHA
- require every named job to conclude success
- reject skipped, cancelled, stale, queued, in-progress, and missing jobs
- fail closed on authentication, API, rate-limit, and JSON errors
- verify peeled tag SHA during publish
- never print credentials

Reuse the verifier for PR policy, release status, and publication. Publish jobs
receive actions: read.

Fixture tests must cover pagination, duplicate names, stale runs, missing jobs,
failed jobs, incomplete jobs, malformed JSON, and API failure.

Expose:

    cargo xtask release-status --sha <sha>

This command is read-only and is the post-freeze handoff mechanism.

### PUB-001 — candidate evidence aggregation

The exact-SHA core/API aggregate must require:

- formatting and diff checks
- pinned workspace Clippy
- workspace tests
- panic and unsafe policy
- dependency denial and unused-dependency checks
- code-generation freshness
- public-support --final
- clean package validation
- stable API and reviewed API-diff evidence
- normal and strict repository lint

Release authorization additionally requires exact-SHA CUDA and Metal jobs.
Hosted macOS is compile evidence, not proof of real Metal execution.

Manual workflow_dispatch publication is always dry-run. Real publication starts
only from a pushed release tag with verified evidence.

### SEM-001 — reviewed semver/API report

Remove the unconditional --release-type major behavior.

For every stable publishable package:

1. Compare the candidate public API with its published/tagged 0.6.2 baseline.
2. Compute the actual release type from the version delta.
3. Record additions, changes, and removals in a committed report.
4. Fail on unapproved changed or removed items.
5. Require every approved compatibility change in the changelog.

List first-published 0.7 packages separately instead of inventing a baseline.
Normal semver verification must fail when the report is stale. Report
regeneration must be an explicit command.

### COV-001 — accelerator coverage

Measure modified host-side Rust in Metal and CUDA crates. Do not exclude entire
adapter crates.

- Require at least 80% changed-path coverage for measurable Rust.
- Allow only line-level, documented exclusions for generated kernels, shader
  bodies, or FFI-only code.
- Map each exclusion to a named hardware or integration test.
- Publish CPU and self-hosted accelerator coverage artifacts separately.

## 8. Phase 2 — safety, duplication, and dead code

### SAFE-001 — Metal buffer access

Current defect:

- j2k-metal-support has a GpuAbi-constrained helper.
- j2k-jpeg-metal and j2k-metal contain local variants.
- one local helper permits mutable access through a shared &Buffer.
- trait, aliasing, and lifetime requirements differ across copies.

Actions:

1. Inventory every POD type crossing the CPU/GPU boundary.
2. Require repr(C), Copy, GpuAbi, and exact shader size/alignment/field-offset
   parity.
3. Centralize bounds, overflow, visibility, alignment, and pointer checks in
   j2k-metal-support.
4. Remove arbitrary-T and safe mutable-from-shared-&Buffer APIs.
5. Prefer checked owned readback/copy operations after confirmed completion and
   checked writes with exclusive CPU access.
6. Use a zero-copy CPU-access guard only if ownership and GPU completion can be
   encoded truthfully.
7. If Metal cannot support a generally safe view, keep one explicitly unsafe
   low-level primitive with documented preconditions rather than a falsely safe
   wrapper.
8. Migrate all adapter call sites and update the unsafe audit.

Tests:

- overflow and multiplication overflow
- alignment and field offsets
- zero length
- non-CPU-visible storage
- undersized buffers
- command completion and concurrent access contract
- lifetime confinement
- contextual errors

If an existing published safe API is unsound, correcting it is an approved 0.7
compatibility change and must appear in the API report and changelog.

### ERR-001 — native decode error classification

Add a neutral DecodeErrorClass/DecodeErrorKind and stable labels in j2k-native.
Each facade or adapter keeps only a small local conversion into its J2kError.

Constraints:

- do not expose j2k_native error values through the public j2k facade
- do not add an internal dependency cycle
- do not use a public doc-hidden facade helper that takes a native error
- forbid direct matching on native inner variants outside the classifier

Add golden parity tests proving equivalent CPU, CUDA, and Metal classification
and message behavior.

### DUP-001 — real clone consolidation

Consolidate:

- corpus category inference in j2k-compare; adoption tooling reuses it
- viewport validation and staging population; finalizers remain backend-specific
- CUDA JPEG packet-to-checkpoint/plan construction as a pure helper

Table-test every corpus needle, precedence rule, and fallback. Ensure viewport
tests call production staging rather than a copied loop. Keep ownership wrappers
around the shared CUDA plan helper.

Do not abstract the small exact-tile batch symmetry between two stable public
APIs with backend-private types.

Clone objective:

- do not exceed the audited 1.93% production-clone ratio
- do not target zero
- every accepted clone has an owner, rationale, and reconsideration trigger

### ADAPT-001 — test-only adaptive router

The 1,105-line adaptive_route module and its 423 tests compile only under
cfg(test), while repository lint positively requires the zombie file.

Actions:

1. Build an assertion map from every old test to:
   - a real shipped-route test, or
   - obsolete-by-design with a reason.
2. Preserve coverage of Auto thresholds/fallback, strict-device refusal,
   observable reports, and counters.
3. Do not port imaginary policy merely to preserve test count.
4. Delete the file with trash.
5. Remove its module wiring.
6. Replace the positive lint with a lint forbidding its return.

### CUDA-001 — suspected orphan kernels

Candidates:

- J2kIdwtHorizontal
- J2kIdwtVertical
- Htj2kEncodeCodeblock
- J2kInverseDwtSingle
- J2kStoreRgb8Mct

For each name search:

- static host dispatch
- dynamic-name lookup
- build/code-generation input
- benchmarks and fuzz targets
- tests and documentation
- public/external ABI commitments

If unreachable, remove the enum/inventory member, host match arms, test mapping,
device entrypoint, and generated PTX reference atomically. If live, document
the consumer and add a dispatch test.

Add parity lint deriving built device entrypoints versus reachable non-test host
dispatch. A waiver requires a named external consumer and owner.

## 9. Phase 3 — structural debt

Each hot refactor is isolated in one commit. Preserve output bytes, statuses,
error strings, labels, command order/count, retention, allocations, reuse, and
public API.

Use the existing performance threshold when available. Otherwise record
five-run medians and reject a regression greater than 5%.

Line targets are review triggers, not substitutes for behavior or performance
evidence.

### STR-001 — resident codestream encoder

The current module is 2,778 lines and contains functions near 541, 702, and
1,128 lines.

Extract in order:

1. pure packet and capacity planning
2. resource and result structures
3. encoding dispatch
4. readback and error interpretation
5. profiling and reporting

Leave thin classic and HT coordinators. Do not genericize their scheduling only
to reduce lines.

Review targets:

- coordinator functions at or below 250 lines unless justified
- orchestration shell preferably below 600 lines
- focused child modules preferably below 800 lines

### STR-002 — direct stacked batch

Separate:

- validation and planning
- buffer/resource preparation
- command submission
- result assembly and reporting

Do this after STR-001 so regressions are attributable.

### STR-003 — native single-tile encoder

Split encode_impl into:

- validation/planning
- accelerator preparation
- tile encoding
- codestream finalization

Preserve accelerator-hook order, exact codestream bytes, fallback semantics,
and allocation behavior.

### TOOL-001 — adoption report

Separate data collection and report-model construction from text rendering.
Add golden report tests and serialized-schema tests. No performance gate is
required for this tooling-only path.

### Accepted large files

Do not split these solely because of line count:

- the 3,466-line fixture builder, whose 204 small builders are cohesive test
  data
- the native encode root, whose functions are already materially smaller and
  domain-focused

Reconsider only if they gain a new domain responsibility or sustain further
growth.

## 10. Accepted-clone register

The following symmetry is accepted unless its trigger occurs:

| Pattern | Rationale | Reconsider when |
|---|---|---|
| RCT and ICT transforms | Mathematical twins remain clearer side by side | A shared primitive removes branches rather than hiding them |
| Backend error enums | Public/backend context differs | Variants and classifications drift again |
| Sampling shader variants | Device specialization is explicit | Generated parity becomes enforceable |
| Host and SIMT pairs | Different execution constraints | One source can generate both without obscuring performance |
| Fixture builder families | Tests prioritize explicit fixtures | Bug fixes repeatedly diverge across copies |
| Exact tile batch facades | Stable APIs use backend-private types | A neutral public type already exists |

Every newly accepted clone must add a row with an owner and concrete trigger.

## 11. Required verification matrix

### Routine local gates

    cargo fmt --all -- --check
    git diff --check
    cargo check --workspace --all-features --lib --bins --examples --tests
    cargo clippy --workspace --all-features --lib --bins --examples --tests -- -D warnings
    cargo xtask test
    cargo test -p xtask --test repo_lint -- --nocapture

### Policy and release gates

- normal repository lint
- strict repository lint on macOS
- release-cpu
- release-metal on real Metal hardware
- panic-surface
- unsafe-audit
- stable-api
- reviewed semver/API diff
- cargo deny check
- cargo machete
- code-generation freshness
- public-support --final
- no-std
- documentation
- bench-build
- relevant fuzz build and execution
- changed-path coverage
- release-integrity
- clean package construction

### Lane contract

| Lane | Trigger | Required evidence |
|---|---|---|
| Default cross-platform | every PR/candidate | CPU and pure tests, including 19 restored tests |
| Hosted Metal compile | every PR/candidate | all Metal crates/facade compile, lint, pure tests |
| Strict public API | every candidate | macOS, cargo-public-api 0.52.0, fail closed |
| Self-hosted Metal | release candidate | exact SHA, real device, all runtime suites, 18/18 ignored suite, zero skips |
| Self-hosted CUDA | release candidate | exact SHA, real device, affected kernels/facades |
| Coverage | every candidate | at least 80% changed-path or narrow documented exclusion |
| Package/support | every candidate | clean package and final support policy |

Run all-feature and non-macOS stub checks so accelerator cleanup cannot break
unsupported targets.

## 12. Candidate freeze and publication

1. Recheck that the selected version has no remote tag, GitHub Release, or
   crates.io publication.
2. Complete code, workflows, documentation, changelog, API snapshot, and
   semver report.
3. Commit everything and require a clean worktree.
4. Set RC_SHA to the current HEAD.
5. Push that candidate as the intended protected origin/main tip.
6. Run hosted CI for exactly RC_SHA.
7. Run exact-SHA CUDA and Metal validation.
8. Require the shared verifier to prove every core/API/GPU job succeeded.
9. If any tracked change occurs, discard all evidence and restart at step 2.
10. Create an annotated release tag at RC_SHA.
11. Verify the tag peels to RC_SHA.
12. Push only the tag; never use --follow-tags.
13. Publish preflight rechecks tag, SHA, workflow identity, required jobs,
    origin, GitHub Release state, and crates.io.
14. Publish packages in dependency order.
15. If publication is partial, rerun against the immutable tag with the
    documented idempotent skip-already-published mode. Never move the tag.

Manual workflow_dispatch publication is always dry-run.

## 13. Interface and compatibility decisions

- No intentional breaking change to the stable j2k facade.
- j2k-native gains a neutral decode-error classification interface; record it
  as an additive API change.
- j2k-metal-support becomes the sole checked Metal buffer-access boundary.
- Correcting an unsound published helper is an explicitly approved 0.7
  compatibility change, not a hidden compatibility shim.
- Developer tooling gains metal-compile, fail-closed release-metal,
  release-status, exact API-diff reporting, and shared workflow verification.
- Workflows consume repository tooling instead of embedding divergent logic.
- Every stable API change appears in both the reviewed report and changelog.

## 14. Release stop conditions

Do not tag or publish when any of these is true:

- a P0 or unresolved P1/P2 finding exists
- the candidate tree is dirty
- the selected version already exists publicly
- exact-SHA evidence refers to another commit
- any required job is missing, skipped, cancelled, or incomplete
- a GPU suite prints a skip marker
- API drift is unexplained
- the semver report is stale
- package contents differ from the validated candidate
- a critical benchmark exceeds its approved regression threshold

## 15. Completion definition

The remediation is complete only when:

- every dashboard item is complete or explicitly accepted
- all user changes are reconciled
- no unexplained ignored test, clone, dead entrypoint, unsafe buffer view, or
  oversized mixed-responsibility orchestrator remains
- the full local release matrix is green from a clean tree
- hosted CI and both GPU workflows are green for the same immutable SHA
- the changelog and public API report describe the actual candidate
- the annotated release tag peels to that candidate
- publication preflight succeeds without bypasses
