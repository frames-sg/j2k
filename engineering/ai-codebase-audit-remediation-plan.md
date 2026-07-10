# j2k 0.7 Full Remediation and Release Runbook

Last updated: 2026-07-09

This is the canonical execution record for the 0.7 quality sweep. Git history
preserves the previous audit diary; do not create competing fixed, new, or
revision-numbered plans.

Present-tense status is limited to the release verdict, handoff capsule, and
live task dashboard. The captured baseline, numerical observations, and dated
evidence elsewhere are historical snapshots, not claims about the current
worktree unless promoted into one of those live sections.

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

The initial audit did not confirm a P0 issue. The later API review confirmed a
P0 Metal host-read aliasing class: safe readback could overlap mutation through
safe or publicly reachable raw `Buffer` aliases. SEC-001 closed the confirmed
class in commits beb4d4e5 and a78cd3e2; an independent equivalent-cache scan
found no remaining safe host-read/host-write alias path. P1 raw GPU-resource
ordering and API hardening closed in 46130e58. The release remains blocked on
the final mixed-responsibility module sweep, regenerated API/semver evidence,
coverage, and the clean release matrix.

## 2. Handoff capsule

Update this section whenever a task changes state. Keep it short enough to read
without loading the rest of the file.

- Current task: STR-015 core JPEG and J2K-Metal suppression/namespace closure
- Parallel tasks: remove or explicitly classify the JPEG manifest/file-wide
  lint overrides and test include seam; remove J2K-Metal high-risk manifest
  overrides and replace internal production wildcard re-exports
- Last completed task: native all-target pedantic promotion and JPEG-Metal
  manifest-wide `too_many_lines` cleanup
- Last completed implementation commits: 008baec8
  (`refactor(native): enable pedantic lint policy`) and 42b28fc6
  (`refactor(jpeg-metal): remove broad line lint allowance`); JPEGCOR-002 and
  the preceding JPEG-Metal naming cleanup are 23e75193, 51594b8e, and 35d66704
- Last completed evidence commits: 0e78229a performance guards and c0937284
  clone scanner/report
- Candidate state: unfrozen
- Worktree expectation: dirty; all changes are being reconciled in place
- Last known green broad gates: repository policy 158/158 runnable plus one
  intentional strict API check ignored, affected strict Clippy, JPEG Metal
  171/171 fail-closed library plus 5/5 real-Metal encode integration, J2K
  Metal device integration 54/54, Metal encode 102/102 runnable,
  native default/all-feature 277/277 plus one intentional ignore and
  no-default 267/267, transcode routing 6/6, whole-production clone ratio
  3.06% below the 3.34% ceiling, and structural performance <=5%
- Current blockers:
  - the focused semantic audit and independent residual red-team pass found
    mixed native/JPEG/GPU/tooling roots and five concrete clone/interpreter
    hotspots; completed work is in STR-004 through STR-009 and the remaining
    roots are tracked in STR-010 through STR-015
  - remaining non-native manifest/file-wide lint overrides and internal
    wildcard namespace seams are being closed or entered into a reviewed
    owner/trigger register under STR-015
  - the pinned clone scan and affected Metal performance guards must be rerun
    after those structural edits
  - stable-API and reviewed semver artifacts must be regenerated after the
    structural source freeze
  - the unsafe inventory, staged-release banners, and changelog section
    placement are being corrected in DOC-002
  - changed-path coverage and the clean final release matrix remain pending
  - exact-SHA CUDA hardware evidence requires the Linux/NVIDIA runner
  - provenance signoff requires the release maintainer's name/handle and date
  - GitHub private vulnerability reporting is currently disabled; the policy
    now fails closed to a detail-free public contact request when the private
    form is unavailable, but an approved working private channel is still
    required before the security and conduct policies can be release-ready
- Exact next local command after the three active extraction lanes stop editing:
  `git diff --check`

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
- Suppression debt: crate/file-wide lint allowances and production `include!`
  seams that hide the effective namespace, ownership, or newly introduced
  warnings from normal review.
- Clone drift: independently copied error mapping, category inference, staging,
  and packet-plan logic that already differs.
- Release theater: stale tags, empty changelogs, permissive semver flags,
  unverified exact SHAs, and publish jobs that validate only a small preflight.
- Coverage theater: excluding complete accelerator crates instead of measuring
  host logic and requiring named hardware evidence for device paths.

Duplication is not automatically a defect. Domain symmetry is accepted when
abstraction would couple unrelated public APIs, hide backend constraints, or
make code harder to review.

### Research grounding used for this audit

The rubric was checked against current primary or first-party sources on
2026-07-09:

- The 2026 MSR study [Speed at the Cost of
  Quality](https://www.cs.cmu.edu/~ckaestne/pdf/msr26.pdf) reports persistent
  increases in static-analysis warnings and code complexity after Cursor
  adoption. It did not find a significant whole-sample increase in duplicate
  line density, though heavy adopters showed a possible modest increase. This
  is why this audit measures and classifies clones instead of pursuing zero
  duplication.
- [Debt Behind the AI
  Boom](https://arxiv.org/abs/2603.28592) attributes 484,366 issues to sampled
  AI commits; code smells dominate, more than 15% of commits from every sampled
  assistant introduce an issue, and 22.7% of tracked issues persist. This
  grounds the focus on static warnings, dead/speculative code, error handling,
  and debt that survives beyond the generating change.
- GitHub's [responsible-use guidance for Copilot
  agents](https://docs.github.com/en/copilot/responsible-use/agents) explicitly
  warns about missed issues, false positives, inaccurate or insecure
  suggestions, and the need for careful human review and testing. Automated
  review output is therefore evidence to verify, never a release approval by
  itself.
- Rust Clippy's official [lint
  configuration](https://doc.rust-lang.org/clippy/lint_configuration.html)
  documents default review thresholds of 25 for cognitive complexity and 100
  lines for a function. This runbook uses size/complexity as review triggers,
  not automatic deletion or abstraction rules.
- Sonar's [metric
  definitions](https://docs.sonarsource.com/sonarqube-server/user-guide/code-metrics/metrics-definition)
  distinguish cognitive complexity from control-flow path count and define
  token-based duplicated blocks. The audit therefore pairs structural metrics
  with reference/dispatch searches and behavior tests before calling code dead
  or duplicated.
- The 2026 empirical study [Secure coding with AI — from detection to
  repair](https://link.springer.com/article/10.1007/s10664-026-10812-8)
  documents both security weaknesses and model limitations in detecting and
  repairing them. That supports the independent unsafe-boundary, input,
  dependency, fuzz, and hardware-path checks in the release matrix.

## 5. Live task dashboard (updated 2026-07-09)

| ID | Severity | Status | Depends on | Outcome |
|---|---:|---|---|---|
| DOC-001 | P1 | complete | — | Canonical runbook replaces stale diary |
| REC-001 | P1 | in progress | DOC-001 | Every dirty file is explained and tested |
| REL-001 | P1 | in progress | REC-001 | Local stale tag removed; staged changelog/docs pending |
| BUILD-001 | P1 | complete | — | Known Clippy failures fixed without allows |
| TEST-001 | P1 | complete | — | All 38 new ignores have exact dispositions |
| METAL-001 | P1 | in progress | TEST-001 | Hosted compile and fail-closed runtime lanes |
| POLICY-001 | P1 | complete | BUILD-001 | Public API scan and strict lane repaired |
| CI-001 | P1 | complete | METAL-001 | Shared exact-SHA workflow verifier |
| PUB-001 | P1 | complete | CI-001, POLICY-001 | Publish requires all candidate evidence |
| SEM-001 | P1 | in progress | REC-001 | Reviewed report complete; generated snapshot pending |
| COV-001 | P2 | pending | METAL-001 | Accelerator host logic is measured |
| SAFE-001 | P1 | complete | BUILD-001 | Shared checked Metal buffer primitives established |
| SEC-001 | P0 | complete | SAFE-001 | Safe Metal readback cannot overlap aliased CPU/GPU mutation |
| GPUORD-001 | P1 | complete | SEC-001 | Reusable texture writes are serialized; raw texture access is unsafe |
| APIHARD-001 | P1 | complete | SEC-001 | Resident private raw resources are private/unsafe and contracts are complete |
| ERR-001 | P2 | complete | BUILD-001 | Neutral native decode classification |
| DUP-001 | P2 | complete | ERR-001 | Genuine clones consolidated and behavior-tested |
| ADAPT-001 | P2 | complete | DUP-001 | Test-only adaptive router removed; shipped behavior retained |
| CUDA-001 | P2 | complete | ADAPT-001 | Five unreachable kernel entrypoints removed |
| STR-001 | P2 | complete | SAFE-001, CUDA-001 | Resident encoder split with focused parity checks |
| STR-002 | P2 | complete | SEC-001, STR-001 | Direct stacked batch split safely |
| STR-003 | P2 | complete | STR-001 | Native single-tile encoder split with byte/hook parity |
| STR-004 | P2 | complete | STR-003 | Split native roots, J2C encode/decode, and precomputed packet preparation |
| STR-005 | P2 | complete | STR-003 | Split facade encode and JPEG decoder responsibilities |
| STR-006 | P2 | complete | STR-001, STR-002 | Split Metal Tier-1, decode dispatch, and direct interpreters |
| STR-007 | P2 | complete | STR-004 | Split core/CUDA/Metal transcode orchestration by stage |
| STR-008 | P2 | complete | STR-004, STR-006 | Consolidate the remaining measured 50–63-line production clones |
| STR-009 | P2 | in progress | STR-005 through STR-008 | Independently classify every remaining 1,000+ line production file and 250+ line function |
| STR-010 | P2 | complete | STR-009 | Split mixed release-tooling roots (`xtask/main.rs`, coverage) |
| STR-011 | P2 | complete | STR-009 | Split mixed native Tier-1, DWT, and codestream implementation roots |
| STR-012 | P2 | complete | STR-009 | Sequential entropy, 12-bit rendering, baseline adapter, and stripe emission split with byte parity |
| STR-013 | P2 | complete | STR-009 | Split mixed encode/fixture comparison tooling roots |
| STR-014 | P2 | complete | STR-009 | Close actionable GPU/runtime findings from the independent large-file pass |
| STR-015 | P2 | in progress | STR-009 through STR-014 | Remove or narrowly justify broad lint suppressions and hidden production namespace seams |
| JPEGCOR-001 | P2 | complete | STR-012A | Fixed ordered-dither rounding; stored and live libjpeg-turbo output now matches byte-for-byte |
| JPEGCOR-002 | P2 | complete | JPEGCOR-001 | Metal 4:2:2 interpolation now matches the CPU/libjpeg ordered-rounding contract across all routes |
| TOOL-001 | P3 | complete | DUP-001 | Adoption report model/render split |
| CUDA-002 | P1 | complete | SEC-001 | One exact named release-cuda gate with zero skip markers |
| PKG-001 | P1 | complete | SEC-001 | Construct all packages and verify independent packages |
| CLONE-001 | P2 | in progress | STR-008 through STR-015 | Scanner/config committed; rerun after final structural source freeze |
| PERF-001 | P1 | in progress | STR-004 through STR-015 | Existing guards passed; rerun after final structural source freeze |
| PUB-002 | P1 | complete | PKG-001, CUDA-002 | Fail-closed origin, Release, and crates.io preflight |
| DOC-002 | P2 | in progress | SEC-001 | Reconcile public claims and keep this as the only plan |
| CONTACT-001 | P1 | blocked on maintainer action | DOC-002 | Publish and verify a working private vulnerability/conduct-reporting channel |
| PROV-001 | P1 | blocked on maintainer input | DOC-002 | Record release signoff identity and date |
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

Evidence snapshot (2026-07-09):

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

The 2026-07-09 baseline delta added 38 ignores. Resolve them as follows.

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

Audit-time defect (2026-07-09 baseline):

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

### SEC-001 — P0 Metal host-read aliasing incident

Confirmed after SAFE-001 during the public-API review:

- `j2k-metal::MetalEncodedJ2k` exposed its backing `Buffer` while retaining
  safe codestream readback.
- both Metal `Surface` types exposed raw buffers while retaining safe host
  readback.
- `j2k-jpeg-metal::MetalBatchOutputBuffer` could safely write a reusable
  allocation while older `Surface` values safely read the same allocation on
  another thread; `metal::Buffer` is `Send + Sync`.
- JPEG viewport/scaled CPU fallback cloned cache-owned shared plane buffers,
  dropped the cache lock, and could later return the cached Gray8 allocation as
  a safe-readable `Surface`. A subsequent safe decode could overwrite it.
- `MetalLosslessEncodeTile` accepted a safe caller-owned buffer that can reach
  a direct host-read fallback and deferred GPU submission.

Required resolution:

1. Make externally supplied or borrowed raw buffer construction/access unsafe
   with contracts that cover cloned handles, CPU work, GPU work, copies of
   descriptor values, deferred submissions, and actual command completion.
2. Keep validated metadata private and expose ordinary metadata through safe
   getters.
3. Keep an owned-buffer handoff unsafe unless the implementation can prove
   exclusive allocation ownership. Consuming one descriptor is insufficient:
   batch descriptors can own different ranges of the same allocation while
   siblings retain safe readback.
4. Give reusable JPEG batch output and every derived surface one shared
   allocation gate. Hold it across safe GPU submission and completion and
   across safe host readback. Replacing a resizable allocation creates a new
   gate; existing surfaces retain the old allocation and gate.
5. Lease cached JPEG plane buffers across their complete CPU populate and GPU
   consume interval, including across cloned sessions, and never return a
   cache-owned plane as a safe-readable surface. Copy/pack cached Gray8 output
   into a fresh allocation before releasing the lease.
6. Surface poisoned-gate errors through fallible APIs; do not silently proceed
   or turn a fallible readback into an unconditional panic.
7. Search all pattern-equivalent public Buffer/BufferRef/Texture exposures and
   internal reusable allocation caches.
   Distinguish confirmed Rust host-memory soundness from GPU ordering-only APIs
   and document both.
8. Update every internal, test, benchmark, example, transcode, changelog,
   unsafe-inventory, stable-API, and semver-report consumer.

Acceptance:

- independent review finds no remaining safe path that can overlap a raw host
  read with CPU/GPU mutation of the same allocation
- focused constructor/range, consuming-handoff, clone/gate, readback, encode,
  transcode, and policy tests pass
- all affected Metal targets check and Clippy with `-D warnings`
- `cargo xtask unsafe-audit` and stable-API/semver checks describe the corrected
  unsafe boundary
- no lower-priority implementation resumes before the incident is closed

Completion evidence (2026-07-09):

- beb4d4e5 made J2K Metal surface, encoded-output, and external encode-input
  buffer boundaries unsafe; private metadata is range-validated and batch
  sibling sharing is explicit.
- a78cd3e2 added one shared gate to reusable JPEG buffer outputs and derived
  surfaces, retained a cache lease through CPU population/GPU completion, and
  forced cached Gray8 output into a fresh public allocation.
- An independent static scan covered public raw resource APIs and internal
  reusable caches/pools in both Metal adapters and found no remaining safe
  host-read/host-write alias path.
- Fail-closed real-Metal regressions passed with `J2K_REQUIRE_METAL_RUNTIME=1`:
  cache serialization 2/2, fresh cached-Gray output 1/1, and reusable-output
  synchronization/poison handling 2/2. No skip marker was emitted.
- Affected all-target JPEG-Metal check and Clippy, focused J2K-Metal/transcode
  checks, both repository policies, and `git diff --check` passed.

Residual P1 work is tracked separately in GPUORD-001 and APIHARD-001; it is a
GPU ordering/raw-resource API issue, not a remaining safe Rust host-memory
alias path.

### GPUORD-001 — reusable private-texture ordering

`MetalBatchTextureOutput` is cloneable, exposes safe `TextureRef`s, and accepts
safe synchronous decode writes. Clones can therefore submit overlapping writes
or external GPU work without a shared ordering boundary.

Actions:

1. Give every clone of a reusable texture output one shared access gate.
2. Hold that gate across every safe texture-output submission and actual GPU
   completion, including viewport/test-only entry paths.
3. Make raw texture access unsafe with a contract explaining that external
   commands bypass the internal gate; retain crate-private trusted accessors.
4. Ensure slot subsets share the original gate and resized allocations receive
   a new gate.
5. Add cloned-output serialization behavior tests and repository policy.

### APIHARD-001 — completed private resource hardening

`ResidentPrivateJpegTile` is safely returned after completion but publicly
exposes raw output/status buffers and a command buffer as fields. It has no safe
host-readback path, so this was not SEC-001, but the API makes ordering and
invariants implicit.

Actions:

- make raw resource fields private
- expose safe metadata getters and only explicitly unsafe raw resource access
  needed for resident handoff
- state cloned-handle and later command-ordering obligations
- add device/session compatibility to `MetalLosslessEncodeTile::from_buffer`
- update tests, examples, API/semver reports, changelog, and unsafe inventory

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

At the 2026-07-09 baseline, the 1,105-line adaptive_route module and its
423-line test file compiled only under cfg(test), while repository lint
positively required the zombie file.

The baseline contained nine tests in a 423-line test file. Their completed
assertion map is:

| Retired test | Disposition | Shipped behavior or reason |
|---|---|---|
| `encode_backend_preference_helpers_select_clear_routes` | mapped | `backend_preference_helpers_select_clear_routes` now exercises the public lossless/lossy option helpers |
| `adaptive_planner_keeps_small_workloads_on_cpu_without_benchmark_gate` | mapped | facade Auto fallback plus the Metal small-host-output threshold tests exercise the real route |
| `stage_candidate_remains_cpu_when_end_to_end_gate_is_missing` | obsolete by design | production accepts no synthetic stage/end-to-end benchmark records; adapter-owned thresholds are the shipped policy |
| `stage_candidate_remains_cpu_when_end_to_end_gate_fails` | obsolete by design | production does not evaluate caller-supplied benchmark percentages |
| `approved_backend_is_not_masked_by_faster_stage_only_candidate` | obsolete by design | the facade receives one concrete accelerator and performs no imaginary cross-device arbitration |
| `rca_reclassification_is_exact_to_stage_and_backend` | obsolete by design | no production API accepts mutable RCA findings as routing input |
| `adaptive_planner_requires_stage_and_end_to_end_gates_before_default_gpu` | mapped | real required-stage completeness tests cover zero, partial, and complete dispatch; benchmark-gate policy was discarded |
| `logical_gpu_loss_requires_rca_before_reclassification` | obsolete by design | production fallback is determined by actual eligibility/dispatch, not a test-only logical-owner/RCA model |
| `strict_device_request_fails_when_backend_is_unavailable` | mapped | direct and accelerator facade tests require typed errors instead of silent CPU fallback |

Completion evidence:

- Four retired tests mapped to shipped behavior and five were explicitly
  obsolete by design.
- Public facade coverage now verifies option helpers, Auto zero-dispatch
  fallback, partial-dispatch counters while reporting CPU, complete-dispatch
  counters while reporting the requested device, and strict-device refusal.
- Existing Metal tests remain the source of truth for actual Auto workload
  thresholds, CPU fallback reports, and resident-surface reporting.
- Both obsolete files were deleted with `trash`, their cfg(test) module wiring
  was removed, and architecture policy now fails if either file or module
  declaration returns.
- Focused validation on 2026-07-09:
  - `cargo test -p j2k --test encode_lossless backend_preference_helpers_select_clear_routes -- --exact`: 1 passed
  - `cargo test -p j2k --test encode_lossless accelerator_facade_`: 6 passed
  - exact direct Auto-fallback and unavailable-device tests: 2 passed
  - `cargo test -p xtask --test repo_lint obsolete_adaptive_route_policy_model_cannot_return`: 1 passed
  - scoped rustfmt, `cargo clippy -p j2k --test encode_lossless -- -D warnings`, and `git diff --check`: passed

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

The 2026-07-09 audit snapshot measured this module at 2,778 lines, with
functions near 541, 702, and 1,128 lines.

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

### STR-004 — native roots and J2C implementation families

The final semantic pass found these still mixing distinct production axes:

- `j2k-native/src/lib.rs` (2,257 lines): scalar codec entry points,
  `DecodeSettings`, reference transforms, and the `Image` implementation
- `j2k-native/src/j2c/encode.rs` (3,166 lines): typed i64 entry paths,
  transforms, validation, packet preparation/rate control, and Tier-1 driving
- `j2k-native/src/j2c/encode/precomputed.rs` (1,991 lines): public wrappers,
  geometry validation, packet conversions, and accelerator adaptation
- `j2k-native/src/j2c/decode.rs` (2,884 lines): direct planning, decode state,
  subband routing, and output storage

Extract focused modules without changing public paths. In the same task:

- route `encode_typed_component_planes_53_i64` through the existing shared
  packet preparation instead of retaining its 63-line duplicate
- extract the 60-line immutable high-bit plan shared by single/multitile encode
- share subband parameter/style/required-block validation where J2C decode
  currently carries 26–46-line copies

Acceptance: byte-for-byte encode parity, decode fixture parity, accelerator-hook
order, exact errors, public API snapshot, and strict Clippy all remain stable.

### STR-005 — facade encode and JPEG decoder

Split by responsibility:

- `j2k/src/encode.rs`: option/sample contracts, native bridge, backend routing,
  lossy target search, and round-trip/metric validation
- `j2k-jpeg/src/decoder.rs`: plan construction, public routing, tile/batch API,
  lossless rendering, and plan validation

Extract the region profile-row emission and RGB routing from
`decode_region_into_output_format_with_scratch`. Preserve public paths, error
text, backend selection, allocation caps, and profile labels.

### STR-006 — remaining Metal execution orchestrators

Split:

- `resident_tier1.rs`: types, waits/readback, profile dispatch, counter
  validation, and result harvest
- `decode_dispatch.rs`: MCT/store, IDWT, classic cleanup, and classic/HT
  subband dispatch
- direct grayscale interpreters: shared lookup/per-step execution while keeping
  repeated and single-output semantics explicit
- `lossless_prepare.rs`: per-item preparation versus command strategy

Consolidate the measured 50-line classic-token dispatch clone by dispatching
into caller-provided shared/private buffers. Preserve command order, lifetime
retention, pooled allocations, status interpretation, profile counters, and
the already-recorded structural performance threshold.

### STR-007 — core and accelerator transcode orchestration

Split:

- `j2k-transcode/src/jpeg_to_htj2k.rs`: facade/validation, component planning,
  and integer/float reference transforms
- `jpeg_to_htj2k/batch.rs`: preparation/grouping, transforms, storage, and
  encode/report assembly
- `j2k-transcode-cuda/src/cuda.rs`: transform dispatch versus resident HT
  encode/result assembly
- `j2k-transcode-metal/src/metal.rs`: runtime/shader, reversible path,
  irreversible path, resident handoff, and buffer/geometry helpers

Preserve route reports, coefficient values, timing ownership, CPU fallback,
device residency, and public API. Run core behavior tests plus both accelerator
compile/policy suites; real CUDA execution remains exact-SHA external evidence.

### STR-008 — measured residual clone/interpreter closure

After STR-004 through STR-007, rerun the pinned scanner and inspect every pair
of at least 50 lines. The known concrete closures are:

- 63/60-line native encode preparation/plan copies
- 50-line Metal classic-token dispatch copy
- direct grayscale 31/36-line execution copies where one primitive can remove
  drift without merging distinct output semantics
- CUDA color-batch fused-store predicate duplicated instead of using
  `can_fuse_mct_store_for_stores`

Do not force mathematical, host/SIMT, shader specialization, or stable
backend-facade symmetry through a branch-heavy abstraction. Any retained pair
must be added to the accepted-clone register with owner and trigger.

### STR-009 — residual large-file and long-function red-team pass

The focused findings above do not make line count disappear as an audit signal.
After STR-005 through STR-008 settle, regenerate an inventory of every
production Rust file at or above 1,000 lines and every production function at
or above 250 lines. Independently review each item and classify it as one of:

- split now because it mixes planning, execution, storage, reporting, policy,
  or public-facade responsibilities
- cohesive algorithm/state machine retained with a named owner and concrete
  reconsideration trigger
- generated/device-specialized source with an authoritative generator or
  parity contract
- test/fixture-only source whose explicit form improves regression coverage

Do not accept a file solely because it already has child modules. Do not split
an allocation-sensitive codec or GPU state machine solely to meet a number.
The review must also flag new production wildcard imports, broad lint allows,
placeholder branches, duplicated error strings, and phase-order comments that
no longer match ownership.

Acceptance:

- the inventory command, counts, classifications, owners, and triggers are
  recorded in this runbook
- every actionable mixed-responsibility item is fixed and behavior-tested, or
  remains an explicit release blocker
- accepted items are added to the large-function/file register
- the final scanner and performance gates run only after this pass freezes the
  source tree

The independent CPU/tooling pass classified the following production roots
after excluding inline tests:

| Disposition | Files | Evidence/next action |
|---|---|---|
| Split (STR-010) | `xtask/src/main.rs` (2,096 production lines), `xtask/src/coverage.rs` (1,030) | Mixed dispatch/release/package/codegen/process concerns; mixed lane/LCOV/policy/render concerns |
| Split (STR-011) | native `ht_block_decode.rs` (1,693), `bitplane.rs` (1,651), `ht_block_encode.rs` (1,882), `bitplane_encode.rs` (1,445), `idwt.rs` (1,530), `codestream.rs` (1,311) | Multiple independent algorithm phases or planning/parsing axes; preserve hot inner loops and exact byte/error behavior |
| Split (STR-012) | JPEG `entropy/sequential.rs` (2,201), `decoder/extended12.rs` (1,887), `adapter/baseline_encode.rs` (951), `entropy/sequential/emit.rs` (1,010) | Mixed public drivers, specialized decode/render routes, planning/assembly, and color-emission responsibilities |
| Split (STR-013) | `j2k-compare/src/encode_compare.rs` (2,324) and `fixture_compare.rs` (2,204) | CLI, input/manifest loading, external tools, validation, measurement, decode, and report rendering are independent axes |
| Accept | native `packet_encode.rs` (794), `tile.rs` (896); JPEG NEON (1,753); `xtask/perf_guard.rs` (785), `xtask/semver.rs` (984); `j2k-types/src/lib.rs` (1,018); transcode accelerator (918) | Cohesive packet/tile/hot-kernel/workflow/public-contract families; triggers are recorded in the accepted register |

The two confirmed 250+ line production functions are the 296-line classic
Tier-1 segment encoder and the 252-line RGB stripe emitter. They are explicit
split targets, not accepted exceptions.

The follow-up suppression/namespace scan found three additional concealment
patterns that line-count-only inventory misses:

- `j2k-native` currently allows `clippy::too_many_arguments` for the entire
  crate, while the JPEG CPU, CUDA, and Metal encoder/runtime roots carry
  file-wide lint allowances. These must be removed, reduced to the smallest
  justified hot kernel or ABI boundary, or retained only with a recorded owner
  and trigger.
- `j2k-transcode/src/lib.rs` textually includes the 918-production-line
  accelerator contract family and `j2k-metal/src/compute.rs` textually includes
  the direct-execution namespace. JPEG Metal had eight equivalent production
  fragments before STR-014. Convert host production fragments to real modules
  with explicit re-exports/imports. Test-only source includes and the shared
  CUDA device prelude remain acceptable when their scope and parity policy are
  explicit.
- seven crate manifests (`j2k`, `j2k-jpeg`, `j2k-metal`, `j2k-cuda`,
  `j2k-jpeg-metal`, `j2k-jpeg-cuda`, and `j2k-compare`) suppress
  `too_many_lines` for every target, and `j2k-jpeg` also suppresses
  `similar_names` globally. Command-line escalation must inventory the hidden
  warnings before these manifest allowances are removed or narrowed.
- `j2k-native/Cargo.toml` warns on Clippy's base `all` group but suppresses the
  entire `pedantic` group, in addition to the crate-root
  `too_many_arguments` allowance. Treat this as a broad suppression: run an
  explicit pedantic escalation, fix correctness/comprehension warnings, and
  retain only named, narrow algorithm/ABI exceptions with rationales.
- `xtask/Cargo.toml` also suppresses the entire `pedantic` group. Release and
  policy tooling is not exempt from comprehension/static-analysis review; run
  the same explicit escalation and replace the manifest-wide suppression with
  focused, documented exceptions only where command/schema compatibility
  requires the existing spelling.

No `TODO`, `FIXME`, `HACK`, `XXX`, `todo!`, or `unimplemented!` marker remained
in the 2026-07-09 source scan. Panic/expect/unreachable sites are reviewed by
the existing panic-surface gate rather than inferred from text matches alone.

### STR-010 — release tooling roots

Split the `xtask` dispatcher from release/package integrity, benchmark/report
commands, codegen/API snapshots, and process/path helpers. Split coverage lane
execution from LCOV/diff parsing, exclusion policy, evaluation, and rendering.
Preserve command/help/error text, exit status, fail-closed behavior, report
schemas, and workflow-consumed command lines.

Post-split fail-closed review found that the panic-surface parser silently
discarded malformed Cargo JSON. Commit 2ca3ee5b now requires UTF-8, validates
every nonblank record, requires one successful terminal `build-finished`
record, rejects trailing records, and has three parser regressions; xtask
all-target check and strict Clippy pass.

### STR-011 — native algorithm families

Extract phase modules without genericizing the hot state machines:

- completed: classic Tier-1 encode facade reduced from 1,445 production lines
  to 539 total lines, with token packing (131), pass kernels (676), segment
  scheduling (434), and distortion accounting (55) separated; the former
  296-line segmented encoder is a 17-line delegate and four coding-style
  payload/segment fingerprints are locked by regression tests
- completed: codestream parser reduced from 1,399 physical lines to a 26-line
  coordinator with ten model/header/validation/marker modules at or below 423
  lines; 32 function signatures, 74 crate-visible items, 53 error tokens, and
  50 string literals match exactly, and the broad unused allowance is gone
- completed: IDWT reduced from 1,535 to a 20-line coordinator with ten focused
  production/test modules at or below 411 lines; 50/50 functions, 12/12 9/7
  constants, 2/2 strings, 7/7 error tokens, and 22/22 normalized hot-loop
  bodies match, with eight new bit-exact goldens, 261 library tests, and 24
  high-bit integrations passing in commit 9acb7a75
- completed: HT block decode reduced from 2,316 physical lines to a 33-line
  coordinator with twelve focused modules; 72 function signatures, 20 error
  tokens, one string literal, 13 unique hot bodies, and ten reader bodies match
  exactly. Sixteen focused tests, 265 library tests with one intentional
  ignore, 24 component-plane integrations, 14 coefficient integrations, and
  the Criterion smoke lane passed in commit d56f327f. One intentional 27-line
  hot-loop similarity remains; quantitative pre/post performance evidence is
  still required by the final PERF-001 rerun.
- completed: classic bitplane decode reduced from 2,124 production lines to a
  22-line facade with focused arithmetic, bypass, context, facade, observer,
  scheduler, state, and test modules. Exact 83/83 production signatures,
  11/11 constants/tables, 4/4 error tokens, and 30/30 hot bodies match; five
  style/pass/segment/coefficient goldens and a bypass-stuffing golden lock the
  bitstream and pass order. Default/no-default checks, strict Clippy, 270
  library tests with one intentional ignore, 38 integrations, and optimized
  benchmark smoke passed in commit 29bac3c3. One 37-line specialized
  arithmetic clone remains for codegen symmetry; quantitative pre/post timing
  is still required by PERF-001.
- completed: HT block encode reduced from 2,083 physical lines to a 17-line
  facade with focused writer, cleanup, quad, emission, refinement,
  distribution, and test modules. Exact 69/69 functions, 16/16 types, 6/6
  constants, 24/24 runtime strings, 48/48 hot/writer bodies, and all three
  public functions match. Eleven focused, 276 library, and 38 integration
  tests, no-default check, strict Clippy, and optimized benchmark smoke passed
  in commit 809caa34. Three codegen-sensitive encode-versus-instrumentation
  traversal pairs remain intentionally specialized; quantitative pre/post
  timing remains part of PERF-001.

- completed HT decode: segment validation/API, MEL/VLC readers, cleanup,
  significance, magnitude refinement, and benchmark instrumentation
- classic decode/encode: state/lookups, pass scheduler/observers, arithmetic
  versus bypass kernels, token packing, and distortion accounting
- HT encode: MEL/VLC/MagSgn writers, refinement, cleanup quad walk, and
  distribution instrumentation
- IDWT: full/ROI/direct orchestration, f32/i64 interleave, and scalar/SIMD
  filters
- codestream: header model/validation versus marker parsers; replace the
  module-wide unused allowance with narrow ownership

Preserve exact coefficients, segments, coding-pass order, bytes, profile
counters, SIMD selection, and no-std behavior.

### STR-012 — JPEG entropy and extended rendering

Split sequential public drivers, generic MCU decode, DCT-block extraction,
fast tile ROI/scale, RGB444/420 routes, extended-precision plane construction,
progressive/sequential rendering, four-component conversion, baseline planning
versus marker/frame assembly, and stripe color emission. Preserve output bytes,
restart state, scratch reuse/caps, profile row order, allocation behavior, and
backend fast-path selection.

Completed STR-012A in commit 882a5c6b: `sequential.rs` fell from 2,209 to 203
lines, with generic, DCT, RGB444, and two fast-420 modules at or below 617
lines. Shared restart validation removed the only newly exposed clone while
preserving marker/error/DC/counter order; the touched clone count and 295
duplicated lines match HEAD and the percentage improved. All-target/all-feature
check, strict Clippy, 12 sequential, 7 WSI/golden, 5 scratch, 9 DCT-route, 2
profile, and 1 partial-byte restart tests passed. The broad all-feature package
run passed 453 tests in eight binaries before being stopped for repeated
macOS 0%-CPU dynamic-loader stalls; focused suites cover every moved route.

STR-012A exposed a pre-existing, deterministic generic baseline 4:2:2 output
difference from the stored libjpeg-turbo RGB fixture: the current-output
FNV-1a fingerprint is `4a9be9f5ec1f80df`. On
`JPEG_BASELINE_422_16X8`, 16/384 RGB bytes differ, the maximum absolute delta is
2, and the R/G/B counts are 0/4/12. The route uses
`upsample_h2v1_fancy_row`, so interpolation/rounding semantics are a plausible
but unproven explanation. The moved decode/emit bodies are text-identical, so
the structural commit must lock current J2K bytes without treating libjpeg as
a normative oracle. JPEGCOR-001 must compare the documented upsampling
contract and independent implementations, then either fix with behavior tests
or record a justified compatibility difference before 0.7.

JPEGCOR-001 closed in commit 5ecbdb7e. Baseline JPEG does not prescribe final
RGB chroma interpolation, but this repository explicitly promises bit-exact
libjpeg ISLOW compatibility in its WSI parity suite. The implementation used a
`+2` quarter-filter bias for both output phases; libjpeg-turbo uses ordered
dither (`+1` for the left/even phase and `+2` for the right/odd phase) to avoid
systematic half-tie bias. Installed libjpeg-turbo 3.1.4.1 reproduced the stored
fixture byte-for-byte, the one-term fix now matches it, and edge/odd-width,
ROI/scaled-ROI, live TurboJPEG, strict Clippy, and baseline encode round-trip
tests pass. Independent Rust decoders that choose `+2`/`+2` remain standards-
valid but do not satisfy this repository's stronger compatibility contract.

JPEGCOR-002 closed in commit 23e75193 after the fail-closed JPEG-Metal library
run exposed 18 Fast422 CPU-parity failures. Three scalar/thread-local shader
paths and the paired path still used `+2` for the even output phase; the
cross-segment texture repair also needed its spatial right pixel to use the
even-phase `+1` bias. The fix preserves `+2` for odd/spatial-left pixels,
adds a source-policy ratchet for every formula, and passes all 171 library
tests with `J2K_REQUIRE_METAL_RUNTIME=1`; the two initially isolated wide and
mixed-table texture failures pass as part of that run.

Completed STR-012B in commit bc90529c: `decoder/extended12.rs` fell from 1,887
to a 21-line coordinator with focused plane, sampling, state, upsample,
ROI/scaling-writer, sequential, progressive, and four-component modules. The
production symbol multiset, diagnostic-literal hash, and generic upsample
rounding/edge goldens are unchanged. Pre/post `decode_into` passed 113/113;
batch/session passed 44/44; scratch reuse passed 5/5; all-target/all-feature
check and strict Clippy passed; and the pinned touched-path scan remained
exactly 7 clones/175 duplicated lines. Root independently reran the 113-test
decode/ROI/scaling/color suite before commit.

Completed STR-012C in commit f6d76f09: `adapter/baseline_encode.rs` fell from
1,099 to a 26-line facade with focused frame, orchestration, planning, table,
type, validation, and test modules. Production API/type, constant, standard
table, zigzag, and string-literal hashes match. Whole JPEG bytes remain exact
for Gray+restart and RGB 4:4:4/4:2:2/4:2:0; independent round trips,
restart/DCT parity, structural tests, all-target/all-feature check, strict
Clippy, and the zero-clone touched-family ratchet passed.

Completed STR-012D in commit f0497b61: `entropy/sequential/emit.rs` fell from
1,010 to a 21-line owner with focused output, RGB, 4:2:0 region, upsample,
four-component, RGB444, type, and structure-test modules. All nine top-level
functions, seven structs/field order, output/conversion call order, strings,
derives, and hot bodies match. Row-streaming parity now covers 4:2:0, 4:2:2,
and 4:4:4; 12/13 sequential tests and both 36-test default/all-feature byte,
restart, ROI/scaling, scratch, CMYK, and YCCK suites passed with strict Clippy.
The three pre-existing output-strategy clone pairs did not increase.

### STR-013 — comparison tooling

For both encode and fixture comparison tools, separate CLI/options, corpus and
manifest loading, external tool discovery/execution, validation, measurement,
and row/metadata/publication rendering. Preserve TSV/JSON schemas and order,
digests, tool commands, environment semantics, publication blockers, and
subprocess exit behavior.

### STR-014 — GPU/runtime large-file closure

The independent GPU pass found these actionable roots after excluding tests,
embedded PTX bytes, and shared generated preludes:

Completed priority 1: JPEG Metal now uses real `fast_packets`,
`pack_dispatch`, `single_decode`, `batch_entry`, `batch_full`, and
`batch_region` modules. The former seven production fragments were trashed,
all replacement leaves are under 800 lines, exact function/signature/string/cfg
parity passed, the real-Metal fail-closed suite ran without skips, and the
touched clone count improved from 24 to 23.

Completed priority 6: CUDA runtime `context.rs` is now a 174-line owner with
focused device creation, context lifetime/drop, pinned-host staging, kernel
cache/dispatch, band transfer, compact-result, and test-kernel modules. Exact
53/53 function-signature, 151/151 string-literal, and 9/9 public-type/field
parity passed; default/all-feature checks, strict Clippy, 85/85 default and
88/88 all-feature host tests, the structural ratchet, and the dependent
`j2k-cuda` compile passed in commit fbc56258. Linux PTX construction and real
NVIDIA execution remain exact-SHA release evidence.

Completed priority 2 in commit 2584a0ce: CUDA runtime `j2k_decode.rs` fell
from 2,142 to a 125-line owner with seven ABI/type, validation, tracing,
IDWT/launch, and store/launch modules at or below 629 lines. Exact 50/50
function, 17/17 type/field-order, 11/11 `repr(C)`, 23/23 kernel-selection, and
launch/resource multiset parity passed; default/all-feature checks, strict
Clippy, 86/86 and 89/89 host tests, the structural ratchet, and dependent
`j2k-cuda` compilation passed. PTX and NVIDIA execution remain external.

Completed priority 3 in commit 1ce31782: CUDA runtime `jpeg.rs` fell from
1,463 to a 93-line owner with focused types, encode, decode, diagnostics,
validation, and ABI-test modules at or below 520 lines. The file-wide
`similar_names` allowance is gone; the remaining exceptions are narrow,
documented expectations. Exact 32/32 function/body and 12/12 `repr(C)` parity,
default/all-feature checks, strict Clippy, 88/88 default and 91/91 all-feature
host tests, and dependent `j2k-jpeg-cuda` compilation passed. PTX and NVIDIA
execution remain external.

Completed the host half of priority 4 in commit 5ae75c75: CUDA runtime
`transcode.rs` fell from 1,665 to a 115-line owner with focused types,
validation, reversible 5/3, staged 9/7, resident/fused HTJ2K 9/7, readback,
launch, and ABI-test modules at or below 446 lines. Exact 29/29 original
function signatures/bodies, every public and internal type field/order, the
single `repr(C)` layout, all errors/operations, and the 15-kernel selection
multiset match. Default/all-feature checks, strict Clippy, 90/90 and 93/93 host
tests, and dependent `j2k-transcode-cuda` compilation passed. The device SIMT
half remains active; Linux PTX and NVIDIA execution remain external.

Completed the device half of priority 4 in commit 0e5dbdc3: CUDA Oxide
transcode `simt/src/main.rs` fell from 1,509 to a 13-line root with focused ABI,
constant, helper, reversible 5/3, irreversible 9/7, quantization, and export
modules at or below 605 lines. The build script now fail-closed stages and
rerun-tracks all seven device modules, and a simulated staged-tree comparison
is exact. All 48 functions, 15 exported kernels, constants/macros, `repr(C)`
fields/order, arithmetic, and the one shared-prelude include match; strict
checks/Clippy and 90/93 host-test lanes passed. Actual PTX and NVIDIA execution
remain external because this host lacks CUDA Oxide and CUDA headers.

Completed the host half of priority 5 in commit 5ac4771d: CUDA runtime
`j2k_encode.rs` fell from 1,630 to a 27-line owner with focused types,
preprocessing/MCT, DWT, quantization, launch, readback, validation, ABI-test,
and structure-test modules. The split follows the implementation that actually
exists; this host layer has no tag-tree, packetization, compaction, dispatch,
or ABI-struct responsibility to invent. Exact 64/64 function-body, 15/15
struct-field/order, seven kernel-argument-list, ten kernel-selection,
resource-order, runtime-string, derive, and documentation parity passed.
Default/all-feature checks, strict Clippy, 95/95 and 98/98 host tests, and
dependent `j2k-cuda` compilation passed. The six pre-existing 5/3-versus-9/7
and host-versus-resident clone pairs did not increase; actual PTX and NVIDIA
execution remain external. The completed CUDA Oxide device half follows.

Completed the device half of priority 5 in commit 46244c5a: CUDA Oxide J2K
encode `simt/src/main.rs` fell from 1,490 to a 21-line root with focused ABI,
constant, 5/3, 9/7, export, helper, packet-writer, packetization,
quantization, and tag-tree modules at or below 461 lines. The build script now
fail-closed stages and rerun-tracks all ten device modules. Exact 56/56
function bodies, 19/19 constants, 10/10 structs/field order, 12/12 kernel
names/order, 7/7 `repr(C)` layouts, derives, and the single export surface
match. Strict runtime Clippy, 98/98 host tests, the dependent `j2k-cuda`
all-target/all-feature check, and a source/staging/ABI ratchet passed. The one
26-line contiguous-versus-strided deinterleave pair is pre-existing. Actual
PTX and NVIDIA execution remain exact-SHA external evidence.

Completed priority 7 encode in commit 0193f8ef: Metal `encode.rs` fell from
1,819 to a 196-line facade with ten focused batch, resident, fallback, routing,
validation, submission, wait, unavailable-host, and structural-test modules;
the largest leaf is 363 lines. All 40 function bodies, both structs, 79
strings, documentation, cfg attributes, errors, and command ordering match.
Default/all-feature checks, strict Clippy, 102 runnable real-Metal tests, both
explicitly required ignored tests, and the resident parity/performance guard
passed with identical bytes.

Completed priority 7 decode in commit a8dd1ab2: Metal `decoder.rs` fell from
1,786 to a 25-line facade with focused adapter, core, direct-path, request,
route, surface, and test modules at or below 476 lines. Exact 64/64 function
bodies and command-order fingerprints, 24/24 visible signatures, 7/7 types,
3/3 constants, 61/61 semantic cfg attributes, 51/51 documentation entries,
and 85/85 runtime strings match. Strict Clippy, 9/9 fail-closed decoder tests,
and 54/54 fail-closed real-Metal device integrations passed; the touched
family remains at zero clones.

Completed priority 8 in commit bc560767: the 953-line test-only classic
Tier-1 token-pack block moved out of production `tier1_encode.rs`. The
production root is now 1,149 lines; a 24-line test-support shell owns focused
GPU pack (357), ordered pack (264), and split CPU pack (340) modules. All 27
production/test function bodies and the 229-string literal multiset match.
The all-target/all-feature Metal check, structural ratchet, relevant real-Metal
encode coverage, strict Clippy, and scoped clone audit passed.

| Priority | Split targets | Required boundary |
|---:|---|---|
| 1 | JPEG Metal `compute.rs` plus included `fast_packets_impl`, `pack_dispatch_impl`, `batch_decode_full`, `batch_decode_region`, `batch_decode_entry`, `batch_decode_impl`, and `single_decode_impl` (effective 8,511-line namespace) | Replace production `include!` fragments with real modules/explicit imports; split packet, pack, single, RGB/RGBA, and repeated/grouped route families |
| 2 | Completed: CUDA runtime `j2k_decode.rs` (2,142 physical lines at split) | ABI types, IDWT scheduling, store/MCT, tracing/validation |
| 3 | Completed: CUDA runtime `jpeg.rs` (1,463 physical lines at split) | Encode/decode ABI and pipeline versus entropy diagnostics; file-wide similar-name allowance removed |
| 4 | Completed: CUDA runtime `transcode.rs` (1,665 physical lines at split) and CUDA Oxide transcode source (1,509) | Matching reversible 5/3 versus irreversible 9/7/HT boundaries |
| 5 | Completed: CUDA runtime `j2k_encode.rs` (1,630) and CUDA Oxide J2K encode source (1,490) | Host types/results, preprocessing/MCT/DWT/quantization plus device packet/tag-tree/export stages with one export surface |
| 6 | Completed: CUDA runtime `context.rs` (1,013 physical lines at split) | Context/pinned memory, kernel cache/loading, and compact result types |
| 7 | Completed: Metal `encode.rs` (1,819 at split) and `decoder.rs` (1,786 at split) | Resident batch/single/host fallback; request/direct plan/core adapters/surface transfer |
| 8 | Completed: Metal Tier-1 test support (953 test-only lines inside production module) | Parity helpers moved into three focused test modules without altering hot production code |

Preserve every `repr(C)` field/order, CUDA entrypoint and generated-PTX
metadata check, Metal shader ABI, status/error value, profile label/order,
device command sequence, lifetime retention, and JPEG output order. Real CUDA
execution remains an exact-SHA Linux/NVIDIA gate after hosted compile/parity.

### STR-015 — suppression and production-namespace closure

Audit every production `#![allow(...)]`, broad `#[allow(...)]`, and
`include!(...)`, plus every crate-local manifest lint override, after the
structural splits settle.

Completed the xtask group escalation in commit 373f3d53. The manifest now
enables pedantic warnings; 97 unique initial sites were resolved. The remaining
fulfilled item expectations are 31 cohesive policy/orchestration/schema
functions with `too_many_lines`, four report-only row-count precision casts,
and two structs whose booleans model independent CLI or fail-closed gate state.
No xtask crate/module allow remains, and repo policy rejects a manifest-level
pedantic allow or line-leading crate/module `#![allow]`. All 104 xtask unit
tests, 158 runnable repository-policy tests, strict all-target/all-feature
Clippy, formatting, and diff hygiene passed.

The native explicit escalation is substantially larger and is being handled
as a separate code-sensitive project rather than hidden behind a replacement
allowlist. The 2026-07-09 production inventory is 892 pedantic diagnostics:
172 possible-truncation casts, 139 lossless casts, 123 `inline(always)` sites,
69 sign-loss casts, 51 similar names, 46 precision-loss casts, 42 must-use
candidates, 34 trivial-copy references, 33 missing error-doc sections, 30
unreadable literals, 29 overlong functions, and smaller categories. A forced
scan also found 52 functions above the argument-count lint. Hot Tier-1/math,
high-level color/image/container, and encode/decode orchestration have separate
ownership so checked input conversions can be fixed while codegen-sensitive
casts, inlining, and stable signatures receive only narrow fulfilled item
expectations with explicit reasons.

The native escalation closed in commit 008baec8. Its manifest now enables both
`all` and `pedantic` at warning level, the crate-wide
`too_many_arguments` allowance is gone, and scans across source, tests, and
benches find zero actual `#[allow(...)]` attributes. Normal strict Clippy
passes for default, all-feature, and no-default all-target builds; 363
item/statement expectations plus five cfg-sensitive expectations remain as
fulfilled, reasoned codec/API boundaries. Default and all-feature library
tests pass 277/277 with one intentional ignore, every integration target
passes, no-default passes 267/267, and the optimized reference codestreams
remain byte-exact. The escalation found and fixed two behavior defects: a
negative palette index could wrap into a huge `usize`, and no-std signed
38-bit packing could saturate through an `i32` rounding polyfill. The native
encode tests are now a normal `#[path = "encode_tests.rs"] mod tests` module
instead of a host `include!` seam.

- remove the crate-wide native `too_many_arguments` allowance and use focused
  request/plan types where parameter groups express one responsibility
- remove the native manifest-wide `pedantic = allow`; promote the group to the
  normal warning policy after the explicit escalation is clean, with focused
  exceptions only where codec math or a stable signature genuinely requires it
- remove the xtask manifest-wide `pedantic = allow` under the same fail-closed
  standard; release automation must not hide warnings that application crates
  surface normally
- narrow math/SIMD lint exceptions to the smallest hot function that genuinely
  requires the spelling, with a one-line rationale and owner/trigger in this
  runbook
- convert transcode accelerator and Metal direct-execution host fragments into
  real modules without changing root exports or public API fingerprints
- replace non-prelude production wildcard re-exports at Metal/JPEG-Metal
  module seams with explicit inventories; standard Rayon/proptest preludes and
  test-only `super::*` imports are not defects by themselves
- make repository policy reject new host-production source includes and new
  crate/file/manifest-wide lint allowances outside a reviewed allowlist
- retain the shared CUDA device prelude only because each standalone SIMT
  crate must compile the same no-std definitions; keep its source/parity ledger
  authoritative

Acceptance: command-line lint escalation proves no hidden warning survives;
normal and no-default/native checks pass; API snapshot and semver fingerprints
are unchanged unless explicitly reviewed; structural policy distinguishes
test/device-generation includes from host production includes.

Completed host fragment conversions: J2K Metal direct execution in 7b1f513b
and transcode accelerator ownership in 64e150d2. The transcode conversion kept
explicit root re-exports, passed all-target/all-feature check and strict
Clippy, 34 library tests, and all three transcode structure-policy tests.

### TOOL-001 — adoption report

Separate data collection and report-model construction from text rendering.
Add golden report tests and serialized-schema tests. No performance gate is
required for this tooling-only path.

### Accepted large files

Do not split these solely because of line count:

- the fixture builder (3,466 lines and 204 small builders in the 2026-07-09
  audit snapshot), whose builders are cohesive test data
- embedded Metal shader-source composition, where most lines are device source
  or `include_str!` wiring rather than a host orchestrator
- cohesive codec state machines listed below, after they are moved into the
  focused owning modules named above

Reconsider only if they gain a new domain responsibility or sustain further
growth.

| Function/family | Owner | Why retained | Reconsider when |
|---|---|---|---|
| HT/classic resident packet submission (423/397 lines) | Metal resident schedulers | Linear stage and resource-lifetime ordering | New stage, repeated ordering defect, or >500 lines |
| CUDA HT Tier-1 device encode (358 lines) | CUDA HT Tier-1 | Allocation-free SIMT state machine | New coding-pass mode or proven neutral phase boundary |
| Native classic Tier-1 segment encode (295 lines) | Native classic Tier-1 | Cohesive pass/segment state machine | New coding style/pass or repeated finalization defect |
| Resident packet-plan construction (288 lines) | Metal Tier-2 planning | Pure validated state/capacity planning | New progression/state layout |
| Classic profile dispatch (279 lines) | Metal profiling | Linear optional stage sequence | New stage or label-order defect |
| Layered packet construction (245 lines) | Native packet/rate control | Cohesive contribution/budget construction | Another rate-control family |
| Native multi-tile coordination (239 lines) | Native tile encode | One tile assembly responsibility | Another assembly mode |
| CUDA resident cleanup/dequant batches (230 lines) | CUDA resident decode | Queued resource ownership is clearest together | Third route or timing drift |
| Metal Tier-2 plan (228 lines) | Metal Tier-2 | Focused pure plan already separated | New descriptor semantics |
| Metal HT Tier-1 preparation (209 lines) | Metal HT Tier-1 | One job/buffer/dispatch preparation path | Second coefficient storage model |
| Main/tile header parsers (194/203 lines) | Native codestream parser | Marker state machines | New marker families or another consumer |
| Single-tile packet encode (194 lines) | Native single-tile encode | Focused subband-to-packet pipeline | Another packetization mode |
| JPEG stripe emit twins (240/251 lines) | JPEG sequential output | Distinct writer contracts and color dispatch | New color/scale or duplicated bug fix |
| Native packet encoder (794 production lines) | Native Tier-2 | Cohesive packet-header/body and marker-state family | Production exceeds 900 lines or another packet-header/marker mode |
| Native tile parser (896 production lines) | Native tile parsing | Tile-part parsing and geometry remain one bounded state transition | Next tile marker/POC/PLT feature, `parse_tile_part` reaches 250 lines, or production exceeds 1,000 |
| JPEG NEON backend (1,753 production lines) | JPEG AArch64 backend | Cohesive benchmark-sensitive SIMD kernel family | New sampling/output format or 2,000 production lines; require benchmark parity before any split |
| Performance guard (785 production lines) | Performance tooling | One snapshot/run/compare workflow | Snapshot schema v2, another execution backend, or 900 production lines |
| Semver workflow (984 production lines) | Release tooling | One API capture/review/report workflow | New review schema/report format or 1,100 production lines |
| Shared encode SPI registry (1,018 production lines) | `j2k-types` maintainers | Public contract registry with short functions and stable root exports | Another accelerator-stage/schema family or 1,200 lines; require API snapshot checks |
| Transcode accelerator contracts (918 production lines) | Transcode acceleration | Cohesive job/trait/default-accelerator family; Rayon glob replaced with explicit traits | Third concrete accelerator, job-schema change, or 1,000 production lines |
| CUDA HT Tier-1 encode host (1,320 production lines) | CUDA HT encode | Cohesive reserved-ABI-aware job/result/launch family | Third job layout/coding mode or 1,500 lines |
| CUDA HT Tier-1 decode host (1,315 production lines) | CUDA HT decode | Cohesive cleanup/dequantize launch family | Another decode mode or 1,500 lines |
| CUDA kernel registry (583 production lines; 517 test lines) | CUDA runtime registry | Entry-point/PTX parity ledger, not an orchestrator | 700 production lines or another registry mechanism |
| CUDA Oxide HT encode core (1,961 lines; 358-line core) | CUDA HT encode parity | One four-entrypoint hot state machine | New coding mode or core exceeds 450 lines |
| CUDA Oxide JPEG baseline decode (1,698 lines) | CUDA JPEG parity | One synchronized fast-baseline ABI across 420/422/444 | Progressive/lossless support or another output family |
| CUDA Oxide HT decode (1,326 lines) | CUDA HT decode parity | One cleanup/refinement kernel family | New refinement path or 1,500 lines |
| Metal Tier-1 production core (1,150 production; 951 test-only lines) | Metal Tier-1 | Cohesive classic/HT device encode; test support moves under STR-014 | Production exceeds 1,300 lines or a third coding mode |
| Metal compute ABI ledger (983 production lines) | Metal shader ABI | 56 short layout types/constants plus parity tests | Second device ABI or 1,200 lines |
| CUDA resident decode scheduler (1,265 production lines) | CUDA resident decode | Cohesive staged cleanup/dequantize pipeline | Scheduler reaches 250 lines, another output pipeline, or 1,500 lines |
| JPEG Metal viewport router (628 production lines; 517 test lines) | JPEG Metal viewport | Cohesive workload/surface selection | Another backend/surface strategy or 800 production lines |

## 10. Accepted-clone register

The following symmetry is accepted unless its trigger occurs:

| Pattern | Owner | Rationale | Reconsider when |
|---|---|---|---|
| RCT and ICT transforms | Native transform maintainers | Mathematical twins remain clearer side by side | A shared primitive removes branches rather than hiding them |
| Backend error enums | Each public backend adapter | Public/backend context differs | Variants and classifications drift again |
| Sampling shader variants | Owning GPU codec stage | Device specialization is explicit | Generated parity becomes enforceable |
| Host and SIMT pairs | Owning codec algorithm | Different execution constraints | One source can generate both without obscuring performance |
| Fixture builder families | Test-support maintainers | Tests prioritize explicit fixtures | Bug fixes repeatedly diverge across copies |
| Exact tile batch facades | Backend adapter maintainers | Stable APIs use backend-private types | A neutral public type already exists |
| JPEG stripe emit twins | JPEG sequential output | Writer contracts differ despite color-dispatch symmetry | New color/scale or the same bug must be fixed twice |
| CUDA/Metal adapter facades | Respective GPU adapter | Stable device/private types differ | A neutral public request type already exists |

Every newly accepted clone must add a row with an owner and concrete trigger.

## 11. Documentation reconciliation

This section consolidates the completed documentation-only audit. The former
untracked `engineering/documentation-remediation-plan.md` was a temporary agent
ledger, not a second source of truth; its unique decisions and evidence are
preserved here before that duplicate is removed with `trash`.

### Public contracts verified by the documentation audit

- `0.6.x` is the latest published and security-supported line; workspace
  `0.7.0` is staged under `Unreleased` until publication.
- GitHub Pages serves `main/docs`. Pushing a candidate can deploy staged docs,
  so hosted pages must continue to distinguish staged 0.7 from published 0.6;
  hosting is not publication evidence.
- coefficient-domain JPEG transcode output is HTJ2K-only
- `j2k` is the CPU-facing facade with planning/shared SPI; concrete resident
  surfaces come from `j2k-cuda` or `j2k-metal`
- strict CUDA-resident codec operations are HTJ2K-only; selected shared stages
  may accelerate other supported work
- CUDA-buffer operations require the `cuda-runtime` feature
- extra `j2k inspect` trailing arguments are currently ignored, not a stable
  interface promise
- unsafe code is isolated to the exhaustive inventory of FFI, GPU,
  SIMD/intrinsic, allocation, and bounded pointer/buffer boundaries

### Completed documentation batches

- Corrected provenance/lineage, retained license inventory, libjpeg-turbo CLI
  fixture generation, and OpenHTJ2K fixture-copy locations in `NOTICES.md`.
- Removed private operator/host paths and normalized present-tense versus dated
  historical evidence in this runbook.
- Corrected unsafe-code posture and expanded `docs/unsafe-audit.md`; SEC-001,
  GPUORD-001, and APIHARD-001 boundaries are now described, with the final
  source-path inventory gate still required after structural moves.
- Drafted the conduct/security policies around GitHub private vulnerability
  reporting without inventing an email address; the later live repository
  check found that private reporting is disabled, so CONTACT-001 supersedes
  that draft and blocks release readiness.
- Repaired adoption commands to use
  `cargo run -p xtask --features adoption -- adoption-*`.
- Aligned staged release state across the changelog, security policy, release
  guide, environment guide, stable-API guide, root README, and static site.
- Documented the current `third_party/block-0.1.6-patched` override and its
  removal conditions.
- Corrected CLI, facade, transcode, codec-math, types, CUDA, JPEG-CUDA, and
  Metal capability/package descriptions.
- Refreshed JPEG-Metal benchmark names, the HTJ2K `_09` fixture guide, Metal
  readback record, benchmark provenance disclosure, and four affected static
  pages; removed manual sitemap `lastmod` values.

### Documentation evidence snapshot (2026-07-09)

Before the later SEC-001 public-API changes, the documentation batch passed:

- normal and strict repository lint
- documentation and workspace doctests
- release-integrity, public-support final, unsafe-audit, and downstream-smoke
- typo and XML checks
- focused crate tests/checks and strict Metal Clippy
- 105 local Markdown/HTML targets with zero failures
- 71 external URLs with only identified crates.io/docs.rs HEAD behavior and
  bot-blocked/manual-verification cases
- all six static pages at 1280px and 375px with one header/nav/main/h1, no
  broken image, and no horizontal overflow

This is historical batch evidence, not candidate proof. Final documentation,
API, unsafe, link, render, and strict gates must be rerun after source freeze.
No page was manually deployed.

After adding the staged-0.7/published-0.6 warning to every static page, a local
browser pass rechecked all six pages at 1280px and 375px: each had one H1 and
one visible release warning, no broken image, and no horizontal overflow. The
source-freeze documentation gate still owns the final rerun.

### Remaining documentation work

- Enable GitHub private vulnerability reporting and verify the external form,
  or publish a maintainer-approved private contact; then align both
  `SECURITY.md` and `CODE_OF_CONDUCT.md` to the working channel.
  The 2026-07-09 read-only check
  `gh api repos/frames-sg/j2k/private-vulnerability-reporting --jq .enabled`
  returned `false`; GitHub's [private-reporting
  documentation](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)
  says the public submission form works only when this repository setting is
  enabled.
- Record the release maintainer's name/handle and approval date for provenance.
- Regenerate stable API and reviewed semver artifacts after the final structural
  source freeze; the pre-STR snapshot is not candidate proof.
- Rerun unsafe inventory after every final source move.
- Re-run all documentation gates and visual checks on the frozen candidate.
- Do not re-create a second documentation plan.

## 12. Required verification matrix

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

## 13. Candidate freeze and publication

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

## 14. Interface and compatibility decisions

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

## 15. Release stop conditions

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

## 16. Completion definition

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
