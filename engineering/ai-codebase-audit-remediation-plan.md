# j2k 0.7 Full Remediation and Release Runbook

Last updated: 2026-07-12

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

Status: **BLOCKED — remediation and pre-candidate verification in progress**

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

The audit-time local `v0.7.0` tag has been removed and remains absent locally
and on origin as of 2026-07-10. Recreate it only after immutable exact-SHA
validation.

The initial audit did not confirm a P0 issue. The later API review confirmed a
P0 Metal host-read aliasing class: safe readback could overlap mutation through
safe or publicly reachable raw `Buffer` aliases. SEC-001 closed the confirmed
class in commits beb4d4e5 and a78cd3e2; an independent equivalent-cache scan
found no remaining safe host-read/host-write alias path. P1 raw GPU-resource
ordering and API hardening closed in 46130e58. SEC-002 implementation closed in
d4be4d20 and SEC-003 in 343bbab3. The later CUDA red-team confirmed SEC-007, a
second P0 confidentiality class in the safe baseline-JPEG CUDA path: forged or
inconsistent decode plans could report success without covering every output
pixel, returning bytes from an uninitialized device allocation. The working
tree now fails those plans before driver work, initializes owned output as
defense in depth, and reports every device-side defensive rejection; exact
final-tree verification is still in progress. The release remains blocked on
source-aware coverage and clean-tree packaging, genuine API
and dependency-provenance review, private vulnerability reporting, the clean
matrix, and exact-SHA hosted/GPU evidence. CUDA resource-lifetime and
architecture implementation is source-complete; its remaining work is frozen
NVIDIA and exact-candidate proof rather than additional unowned source edits.

## 2. Handoff capsule

Update this section whenever a task changes state. Detailed history belongs in
the issue sections below; this capsule is only the current continuation state.

- Release state: **blocked** and unfrozen. The latest settled source commit
  before this ledger refresh is `cef2ba40`. No push, release tag, crate
  publication, or externally visible release action has been made. The local
  `v0.7.0` tag remains absent. The approved plan authorizes later exact-SHA
  movement through the normal reviewed workflow; it does not authorize tagging
  or publication.
- Current objective: close the remaining host changed-line and source-body
  coverage proof without exclusions or threshold changes, then rerun the
  clean-tree clone, package, API, dependency, corpus, performance, and release
  gates. Preserve idiomatic Rust ownership, typed errors, fallible allocation,
  transactional mutation, and actual allocator-capacity accounting.
- Immediate continuation point (2026-07-12): direct CPU coefficient owners are
  fallibly budgeted (`59055081`); the Metal resident pool ceiling is derived
  from its real default working set (`c901602c`); and resident encoding now
  submits at most the configured in-flight chunk count with transactional range
  validation and failure stop (`3c2d1793`). Host/Metal/CUDA coverage ownership
  is partitioned with schema v3 and exact-SHA/lane provenance (`f3823de9`), and
  covered macro invocations no longer fail merely because the AST treats the
  invocation as opaque (`d717f7c8`). Release/semver, adoption runner/artifact
  generation, benchmark/codegen, compare, transcode, JP2 metadata, typed
  stage-error, native SOT parsing, and JPEG decoder boundaries now have focused
  behavior tests through `cef2ba40`. The exact detached host run at
  `b34ab401` completed the full host matrix and failed the real gate at
  77.7123% (78,149 / 100,562), leaving a 2,301-line numeric gap plus 306
  uncovered functions, 433 uncovered executable bodies, 1,490 one-line
  deferred closure bodies, and 666 uncovered opaque macros. That exact run
  passed repo-lint 411/411 with one established ignore. The later `cef2ba40`
  JPEG tile tranche is not included in those aggregate numbers; its focused
  LLVM evidence reached 99.29% for `decoder/tile.rs` and 94.63% for the
  implicated fast-420 row source. Root owns the coverage proof, settled
  aggregate gates, ledger, and exact-SHA handoff; delegated owners are bounded
  to facade, JPEG planning/rendering, and coverage-tool behavior tests.
- Independent architecture closure in the follow-up:
  - the 1,079-line JPEG encoder is a 148-line facade over API, allocation,
    sample-plane, transform, profiling, and test owners. Shared baseline entropy,
    encode, and DCT contracts sit below both encoder and transcode, eliminating
    their dependency cycle and duplicate allocation primitive;
  - the CUDA resident `helpers.rs` catch-all is gone. Buffer access, error
    mapping, IDWT conversion, component validation, and color-store validation
    have downward-only focused owners and share one checked element-count
    primitive;
  - JPEG Metal `pack_dispatch/common.rs` is replaced by conversion, request,
    surface, grouped-output, texture, dispatch, and split-IDCT owners, with a
    policy that forbids the catch-all from returning;
  - Metal resident packet planning validates parallel input lengths and zips
    them before access, and the three duplicate CUDA HTJ2K multi-input launches
    share one private kernel launcher.
- Active security/correctness closure:
  - the safe `GpuAbi` byte-view contract now requires compile-time proof of a
    padding-free, fully initialized representation. Five padded CUDA host/device
    records use explicit initialized tail fields without changing their sizes or
    existing offsets; JPEG Metal checkpoint staging has the equivalent explicit
    tail and every generic typed upload now requires `GpuAbi`;
  - the public benchmark AVX2 IDCT wrapper performs its own runtime detection
    and uses the scalar implementation when AVX2 is absent;
  - the live unsafe inventory passes after the module moves. `cargo xtask miri`
    now passes its complete repository-owned lane, including the pure byte-view
    regressions; the earlier unavailable-Miri note is superseded;
  - the J2K facade, J2K-to-HTJ2K transcode boundary, and CUDA/Metal adapters use
    crate-owned opaque native-source wrappers. Public signatures no longer name
    `j2k-native`; `Error::source` still reaches each concrete native error and
    local typed classification preserves allocation, unsupported, validation,
    and invariant categories. Warning-denied all-target/all-feature Clippy is
    green, including the completed ALLOC-018 Metal batch-allocation work.
- JPEG allocation status:
  - ALLOC-008 parse-once fast-packet/checkpoint budgeting is complete and its
    combined gates are green.
  - ALLOC-009 through ALLOC-012 are complete. Header/progressive metadata,
    parsed-to-prepared construction, every multi-plane decode owner, batch
    scheduling/collection, warning-result transfer, segment rewrite, and TIFF
    assembly now use typed actual-capacity budgets. Progressive terminal
    validation is implemented, and large prepared JPEG payloads are move-only
    with a fallible `try_clone` replacement.
  - The pattern-equivalent public-owner sweep makes `EncodedJpeg`,
    `JpegDctImage`, `JpegDctComponent`, `RestartIndex`, and all four retained
    fast-packet payload types move-only. Workspace production callers did not
    clone them; CUDA shares cached packets through `Arc`, and the Metal owner is
    reconciled in ALLOC-018. The unused stringly
    `JpegEncodeError::Internal(String)` variant is removed in favor of typed
    invalid-input errors and allocation-free static invariants.
  - DQT zero values and invalid table classes/slots now return typed errors
    before state mutation.
  - `duplicate_table_policy` is now enforced across every DQT/DHT definition,
    including mixed 8/16-bit and combined DC/AC markers. `AllowIdentical`
    coalesces exact definitions, default `RejectConflicting` preserves them for
    byte parity, both reject conflicts, and malformed later definitions fail
    before assembly. The focused helper, public integration, full library,
    strict Clippy, and segment-policy suites are green.
- Active native encode work:
  - the blanket `From<&'static str>` pipeline conversion is removed and
    forbidden by policy. Precomputed, packet, rate-control, Tier-1, typed-i64,
    single-tile, and resident boundaries now classify caller input,
    unsupported capability, arithmetic, invariant, resource, accelerator, and
    validation failures explicitly;
  - move-only retained sessions, borrowed precomputed coefficient owners,
    fallible Tier-1/rate-control graphs, packed i64 preparation, direct
    multi-tile packet ownership, and scratch-free final writer handoffs are in
    focused modules with passing local architecture ratchets;
  - accepted accelerator HT/classic code-block results are validated against
    their jobs before conversion. Segment lengths, pass counts, zero/nonzero
    block presence, classic bitplanes, segment coverage, and coding modes fail
    as the originating accelerator operation; fused-HT and preencoded inputs
    share the same metadata invariants;
  - reconciliation removed a real six-byte ROI planning scratch overlap,
    corrected header-cap failures to typed `AllocationTooLarge`, and updated
    stale maxshift and large-SIZ expectations. The deterministic
    `multitile_tile_parts` mismatch was a decoder-side PPM cursor bug: the
    writer emits one packed-header entry per packet, while tile parsing selected
    one entry per tile part. Checked packet counts and a global PPM cursor now
    preserve PLT/PLM/PPM/PPT round trips. The complete native package and
    strict all-feature/no-deps Clippy are green with ALLOC-006 complete.
- Active transcode work:
  - the blanket `From<&'static str>` conversion for `TranscodeStageError` is
    removed and forbidden by policy. The production reversible-grid boundary
    and three test accelerators now choose `Unsupported` explicitly;
  - CPU JPEG-to-HTJ2K batch result ordering uses a checked private slot owner.
    Out-of-range, duplicate, and missing worker results are typed
    `InternalInvariant` failures instead of direct indexing, replacement, or a
    generic validation fallback;
  - all-feature check, strict all-target no-deps Clippy, 55 library tests, 37
    integration tests, four result-slot regressions, the grid taxonomy test,
    and both focused repository policies pass. Removing the public blanket
    conversion and adding a public enum variant are deliberate 0.7 breaking
    changes now described in the changelog; frozen API/semver reconciliation
    remains.
- Error-contract sweep:
  - no production `impl From<String>`, `impl From<&str>`, or
    `impl From<&'static str>` blanket error conversion remains. Facade backend,
    native encode, transcode, and CUDA packetization boundaries now choose a
    category explicitly;
  - `BackendError` no longer accepts blanket owned or borrowed strings. Facade
    adapters must construct a typed kind explicitly, and repository policy
    prevents the error-erasing conversions from returning;
  - CUDA packetization invalid plans still decline to the CPU path, while a
    host allocation failure remains a hard stage error. Full host CUDA tests
    pass 86 across targets, 15 packetization tests and 8 resident policies pass,
    and target strict Clippy/checks are green.
  - JPEG Metal group failures clone the original typed error into every
    affected output slot; decoder, encoder allocation, and buffer failures no
    longer become generic rendered kernel strings. Focused regressions, strict
    Clippy, and the source policy are green.
  - shared Metal runtime, completion, and buffer failures now retain
    `MetalSupportError` sources in both public adapters. Prepared-plan cache
    allocation and invariant failures remain distinct, the JPEG adapter stays
    cloneable, and no readback path invents a saturated byte count;
  - public baseline JPEG DCT re-emission now reports a non-exhaustive typed
    `JpegDctImageError` through `JpegEncodeError::InvalidDctImage`. Every used
    dimension, component position, sampling factor/MCU limit, block grid/count,
    coefficient entropy category, and quantization constraint is validated
    before capacity or entropy work; ignored extraction metadata remains
    byte-neutral by regression.
  - the facade no longer has a private generic string-to-backend constructor.
    Future variants of the non-exhaustive resident HTJ2K boundary retain their
    typed source, and a recode pixel mismatch carries
    `BackendErrorKind::Validation` instead of the generic `Other` category.
- Active container/recode work:
  - paired validation counts the encoded output, both parsed images, the first
    decoded result, and the second decode in one retained-baseline sequence;
    raw and JP2/JPH parsing now accept that baseline before allocation.
  - facade encode validation now counts the generated `Vec` capacity through
    parse and decode, preserves typed native decode resource errors, compares
    component samples without a second full reference owner, and prevents rate
    searches from retaining an earlier codestream during the next encode.
  - the reopened 1,102-line native `image.rs` owner is split into focused core,
    output, and direct-plan modules with structural ceilings.
  - borrowed and owned native component planes now use one authoritative
    palette-aware sampling rule. Postprocess and native-output owner accounting
    share one checked `DecodeOwnerBudget` component-owner and arithmetic
    primitive instead of maintaining parallel SIMD/integer/cap/overflow
    implementations; the serialized native and workspace gates are green.
  - adversarial review found palette validation could truncate mixed or
    25–38-bit values through the codestream index precision/f32 path. Palette
    comparisons now force component output, and >24-bit palette values retain
    an exact i64 shadow; mixed/signed/high-precision regressions pass.
- Active structural work:
  - the J2K facade view owner is now a 514-line core over focused 301-line row
    and 180-line trait/batch modules. Its all-target/all-feature check, strict
    no-deps Clippy, 323 passing tests plus one established ignore, and focused
    architecture policies are green;
  - the last STR-019 seam is closed: the former 711-line facade decode owner is
    a 187-line orchestration/warning root over explicit 8-bit and 16-bit output
    modules. View, decode, batch, wrapper, and recode owners all have focused
    structural ratchets;
  - post-allocation review reopened CUDA runtime HT encode/decode, CUDA
    resident decode, and native JP2 container owners above the 1,000-line
    production trigger. All four source owners now use thin explicit facades
    and focused semantic modules with lower ratchets; combined-tree and
    exact-source hardware verification remain in STR-009.
- Current verification: focused native production/test compilation, strict
  library Clippy, typed-error suites, Tier-1 metadata suites, high-bit tests,
  precomputed tests, and their architecture policies pass. The combined
  all-feature `j2k-core`/`j2k-native` no-deps library/test Clippy gate is now
  warning-free after reconciling `CodeBlockStyle` value ownership, six
  `expect_err` conversions, and one test-only `let`-`else` without suppressions;
  37 focused bitplane tests and five affected typed-error/allocation regressions
  pass. JPEG Metal ABI
  check, two no-padding/layout tests, strict lib/test Clippy, and its policy
  pass. The unsafe-audit command passes. The standard-library Python release
  tooling suite passes 49/49 tests, including exact workflow/tag identity,
  fail-closed API behavior, crates.io state, credential redaction, and publish
  preconditions; no project virtual environment exists, so this checkpoint
  used the system Python 3.14.4. `cargo deny check` passes advisories, bans,
  licenses, and sources, and `cargo machete --with-metadata` reports no unused
  dependencies. A mid-remediation RTX 4070 SUPER run
  built all ten strict `sm_89` cuda-oxide projects and passed 257/257 required
  CUDA-runtime library tests. This is supporting evidence only: the full
  native/workspace matrix and exact frozen-tree GPU reruns remain mandatory.
- Panic/source-audit ceilings are reduced, never raised: `unwrap` 16,
  `expect` 50, `panic!` 0, `unreachable!` 50, `assert!` 8, `assert_eq!` 3,
  `debug_assert!` 91, and `debug_assert_eq!` 66; the remaining zero-count
  categories stay at zero. The latest reductions remove an internal marker
  read unwrap and the unused public deinterleave panic wrapper; their focused
  behavior/policy tests pass. The typo gate passes after correcting one
  identifier and narrowly excluding generated `fnv1a64:` fingerprints. The
  combined panic and typo gates rerun on settled source.
- Next serialized gates after the source and tooling phase commits:
  1. finish compiler-grounded one-line closure evidence and add behavior tests
     until every source-body proof and the real 80% host gate pass;
  2. rerun clone, packaging, paired Metal performance, stable/hidden API,
     semver, dependency, corpus, and publish-mode integrity gates on the settled
     committed source;
  3. obtain maintainer API/provenance/security approval, date the changelog,
     run publish-mode offline integrity, and commit a clean candidate SHA;
  4. run exact-SHA hosted CI plus Metal and CUDA hardware validation.
- External release blockers: GitHub private vulnerability reporting is disabled
  (CONTACT-001); patched `block 0.1.6` provenance reviewer/date is pending
  (PROV-001); all 33 `PENDING` API rationales still require real maintainer
  review; release date and candidate SHA do not yet exist. A tag is deliberately
  outside this verified-RC endpoint.
- CUDA handoff: the maintainer supplied a private WSL/NVIDIA host with RTX 4070
  SUPER, CUDA 13.2, Rust 1.96, and libclang 18. Keep login/address out of tracked
  files. The repaired GPU ABI source passed a supporting `sm_89` build and
  257-test runtime run there, but later source changes invalidate it as release
  evidence. Rerun only the final exact frozen bundle/commit with the required
  runtime/build/hardware-decode flags.

## 3. Operating rules

1. Preserve every user change until its purpose is reconciled.
2. Never reset the worktree or overwrite unrelated edits.
3. When the maintainer authorizes commits, make one bisectable commit per
   high-risk task; audit/remediation work itself does not imply commit, push,
   tag, release, or publication authority.
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

## 5. Live task dashboard (updated 2026-07-12)

| ID | Severity | Status | Depends on | Outcome |
|---|---:|---|---|---|
| DOC-001 | P1 | complete | — | Canonical runbook replaces stale diary |
| REC-001 | P1 | in progress | DOC-001 | Every dirty file is explained and tested |
| REL-001 | P1 | in progress | REC-001 | Local stale tag removed; staged changelog/docs pending |
| BUILD-001 | P1 | complete | — | Known Clippy failures fixed without allows |
| TEST-001 | P1 | complete | — | All 38 new ignores have exact dispositions |
| METAL-001 | P1 | complete | TEST-001 | Hosted compile and fail-closed runtime lanes; exact-SHA rerun belongs to FINAL-001 |
| POLICY-001 | P1 | complete | BUILD-001 | Public API scan and strict lane repaired |
| CI-001 | P1 | complete | METAL-001 | Shared exact-SHA workflow verifier fails closed unless private vulnerability reporting is enabled |
| PUB-001 | P1 | complete | CI-001, POLICY-001 | Candidate aggregate requires both ordinary and authoritative strict Clippy without replacing either gate |
| SEM-001 | P1 | blocked on maintainer review | REC-001 | Frozen-source ordinary/hidden snapshots and reviewed diff are regenerated; exact fingerprints are recorded and the fail-closed PENDING rationales await real approval |
| COV-001 | P2 | in progress | METAL-001 | Exact schema-v3 host evidence at `b34ab401` is 77.7123% (78,149 / 100,562); close the 2,301-line numeric gap and compiler-grounded function/body/deferred/macro proofs without exclusions or threshold changes |
| ALLOC-001 | P2 | in progress | SEC-007 | Context-wide CUDA external/pinned/provisional authority, transactional actual-capacity phase ownership, and policy ratchets pass local gates; frozen NVIDIA and final combined-tree evidence remain |
| ALLOC-002 | P1 | in progress | STR-014 | Source-complete no-byte resident J2K descriptor and fail-closed whole-tile route; frozen-source NVIDIA parity remains |
| ALLOC-003 | P1 | complete | — | Native parse/tile, ROI/direct-plan, Tier-1, recode, postprocess, output, and reusable context owners share one actual-capacity decode budget |
| ALLOC-004 | P1 | complete | — | CPU/DCT/GPU planes, entropy, frame copies, batch metadata, returned chunks, and prior frames share actual-capacity high-water plans |
| ALLOC-005 | P1 | in progress | ALLOC-001 | Source-complete CUDA coefficient-transcode staging/readback/widening ownership reconciles allocator-reported capacity; frozen NVIDIA and final combined-tree evidence remain |
| ALLOC-006 | P1 | complete | ALLOC-003 | Transactional actual-capacity tile/tile-part ownership, exact owner-graph handoff, failure rollback, and PPM packet-header mapping pass the full native package, strict Clippy, and focused policies |
| ALLOC-007 | P1 | complete | ALLOC-004 | CPU JPEG-to-HTJ2K composes actual external owners with lowered native caps, retained batch outputs, and fallible Vec-backed validation metrics |
| ALLOC-008 | P1 | complete | ALLOC-004, ALLOC-007 | JPEG entropy, restart offsets, terminated scan, checkpoint/cache representations, and decoder metadata share one actual-capacity contract |
| ALLOC-009 | P1 | complete | ALLOC-007 | Versioned parse arenas, consuming prepared-plan construction, progressive script/terminal validation, and parsed/scan/result warnings share typed actual-capacity budgets |
| ALLOC-010 | P1 | complete | ALLOC-009 | Progressive, lossless, extended-12, sequential/DCT, output, row, entropy, and retained-frame owners feed each allocator-returned capacity into the next reservation |
| ALLOC-011 | P1 | complete | ALLOC-009, ALLOC-010 | JPEG batch sessions enforce one 512 MiB codec domain plus one collective 64 MiB metadata domain with cap-driven concurrency and typed outer failures |
| ALLOC-012 | P1 | complete | ALLOC-010 | SOF rewrite and TIFF normalization retain borrowed segments, reserve one exact output, and expose fallible rather than infallible owned-payload duplication |
| ALLOC-013 | P1 | complete | ALLOC-003 | Native encode uses typed aggregate phase ownership through transform, Tier-1/2, multi-tile, accelerator, final writer, and batch paths; full dependent gates pass |
| ALLOC-014 | P1 | complete | ALLOC-003, ALLOC-011 | Four authoritative native worker claims plus one 64 MiB metadata domain, typed scoped scheduling, heap-free decode errors, dual-limit collection, and removal of the unused panic/infallible collector are verified |
| ALLOC-015 | P1 | complete | ALLOC-010 | J2K row-decode scratch reserves fallibly, distinguishes over-cap from OOM, and reconciles allocator-reported capacity before each simultaneously live owner |
| ALLOC-016 | P1 | complete | ALLOC-003 | Facade native-component decode moves owned plane/ICC payloads and reserves borrowed/owned metadata under the cached-Image plus native live budget |
| ALLOC-017 | P1 | complete | ALLOC-013, ALLOC-016 | JP2/JPH parse, inspection, wrap, passthrough/recode, paired validation, metadata, and final output share capped fallible owner plans |
| ALLOC-018 | P1 | in progress | ALLOC-001, ALLOC-011, ALLOC-014 | JPEG Metal cache/queue/execution metadata and retained host-surface ownership are transactionally reconciled; full package, strict Clippy, and combined policies pass, with the final matrix/hardware evidence remaining |
| METALCACHE-001 | P1 | complete | SEC-009, ALLOC-018 | Move-only Arc-shared direct plans, actual host/device cache weights, deterministic flat LRU, fallible metadata, and typed optional admission are verified |
| JPEGCACHE-001 | P1 | in progress | ALLOC-001, ALLOC-008, ALLOC-018, ERR-015 | One neutral 8-entry/64-MiB inspect-once cache is integrated under clone-shared cache, queue, pinned-staging, result, and CPU-fallback ledgers; package/policy gates pass and frozen hardware evidence remains |
| METALPOOL-001 | P1 | complete | ALLOC-018, SAFE-001 | Commit `3fd23cf2` uses separate flat private/shared `VecDeque` pools and move-only once-validated capacity owners; default private working-set reuse, full Metal gates, and alternating direct/resident performance comparisons pass |
| CUDAPOOL-001 | P1 | complete | ALLOC-001, CUDAERR-001 | CUDA buffer pools apply shared actual-byte/count/bucket limits, deterministic completed-buffer eviction, deferred safety accounting, and clone-shared high-water diagnostics; strict RTX 4070 SUPER retention tests passed |
| CUDAPIN-001 | P1 | in progress | ALLOC-001, CUDAPOOL-001, CUDAERR-001 | Context-authoritative page-locked staging pre-reserves growth, external owners use unlocked full-headroom RAII replacement transactions, and exact retention/quarantine/compound errors pass local race and package gates; frozen NVIDIA evidence remains |
| CUDAHANDLE-001 | P2 | complete | CUDAPIN-001, CUDAERR-001 | One typed boundary validates device allocations and every context/module/function/event/stream out-parameter; successful-null function lookup unloads its module |
| METALALLOC-001 | P1 | complete | SAFE-001 | Metal coefficient transcode uses checked aggregate phase plans, fallible actual-capacity host ownership, checked device construction, typed sources, and linear sparse weights; full local package and policy gates pass |
| PROFILE-001 | P2 | in progress | ALLOC-003, AUDIT-001 | Source-complete typed limits, transactional fallible ownership, move-only summaries, shared diagnostics, and bounded typed callers; frozen API/semver review and final matrix remain |
| CUDASESSION-001 | P2 | in progress | ALLOC-002 | Source-complete context-bound resident resource cache, external-context binding, clone/batch reuse, and pre-upload mismatch rejection; frozen-source NVIDIA evidence remains |
| SAFE-001 | P1 | complete | BUILD-001 | Shared checked Metal buffer access primitives established; fallible resource construction is tracked separately by METALBUF-001 |
| METALBUF-001 | P0 | complete | SAFE-001, ALLOC-018 | One checked support boundary owns nil-safe retained buffer, texture, queue, command-buffer, encoder, library, and pipeline construction; all three adapters and real-Metal package gates pass |
| SEC-001 | P0 | complete | SAFE-001 | Safe Metal readback cannot overlap aliased CPU/GPU mutation |
| SEC-002 | P0 | complete | STR-015 | d4be4d20 ensures JPEG owned output never exposes uninitialized `u8` storage; candidate rerun belongs to FINAL-001 |
| SEC-003 | P1 | complete | STR-013, STR-015 | 343bbab3 hardens comparator FFI bounds/ownership/initialization; candidate rerun belongs to FINAL-001 |
| SEC-004 | P1 | complete | CUDAERR-001 | Safe CUDA store jobs validate destination/source u32 bounds and initialize every unwritten output byte; candidate rerun belongs to FINAL-001 |
| SEC-005 | P1 | complete | CUDAERR-001 | Safe single-IDWT jobs validate every band allocation and device index before launch; candidate rerun belongs to FINAL-001 |
| SEC-006 | P1 | complete | CUDAERR-001 | Safe CUDA JPEG encode jobs validate input/entropy ranges and reject overlapping batch outputs; candidate rerun belongs to FINAL-001 |
| SEC-007 | P0 | complete | CUDAERR-001, CUDAGRID-001 | Safe CUDA JPEG decode proves a complete non-overlapping half-open MCU partition and initializes the full u32-addressable output extent before every successful safe decode; strict CUDA-Oxide RTX 4070 SUPER evidence passed |
| SEC-008 | P1 | complete | SEC-007 | Safe JPEG fast-packet builders bounds-check malformed SOF quant selectors and defensive Huffman selectors without panicking |
| SEC-009 | P1 | complete | SAFE-001 | J2K Metal prepared-plan caches use randomized digest buckets plus owned full request identity and equality before every hit; candidate rerun belongs to FINAL-001 |
| SEC-010 | P1 | complete | APIHARD-001 | Safe GPU ABI byte views require compile-time no-padding proofs; five CUDA and one JPEG-Metal padded records now use initialized explicit tail fields with preserved ABI sizes/offsets; frozen-tree GPU/Miri evidence belongs to FINAL-001 |
| SEC-011 | P1 | complete | STR-012 | The safe public AVX2 benchmark wrapper performs runtime feature detection and falls back to scalar; host/x86 checks, parity, strict Clippy, and policy are green |
| GPUORD-001 | P1 | complete | SEC-001 | Reusable texture writes are serialized; raw texture access is unsafe |
| APIHARD-001 | P1 | complete | SEC-001 | Resident private raw resources are private/unsafe and contracts are complete |
| ERR-001 | P2 | complete | BUILD-001 | Neutral native decode classification |
| ERR-002 | P1 | complete | ERR-001, ALLOC-003 | CUDA and Metal delegate native decode mapping to the facade; typed resource-source regressions, architecture policy, and strict adapter gates pass |
| ERR-003 | P2 | complete | ALLOC-007 | Transcode stage boundaries classify legacy strings explicitly, and checked batch result slots report missing, duplicate, or out-of-range worker output as internal invariants |
| ERR-004 | P2 | complete | STR-009 | CUDA packetization explicitly distinguishes invalid-plan CPU fallback from hard host-allocation failure; the last blanket production string-to-error conversion is gone |
| ERR-005 | P2 | complete | STR-019 | Facade backend errors require explicit classification; strict Clippy debt, mixed view ownership, recoverable batch assertions, and mapped-component recode fallback are reconciled |
| ERR-006 | P2 | complete | ERR-005 | JPEG Metal clones typed batch failures into every affected output slot instead of rendering codec/buffer failures into generic kernel strings |
| ERR-007 | P2 | complete | ERR-006 | J2K Metal resident batch preparation propagates its original typed error instead of wrapping same-crate failures in a rendered kernel string; focused behavior/policy and combined warning-denied Metal Clippy pass |
| ERR-008 | P2 | complete | ERR-005 | Tile-codec decoder I/O retains typed sources and stable codec classification; the obsolete string backend variant is removed and the public enum is non-exhaustive |
| ERR-009 | P2 | complete | ERR-006, ERR-007 | J2K/JPEG Metal support and prepared-plan cache crossings preserve typed sources, exact routing, and operation diagnostics without fake allocation byte counts |
| ERR-010 | P1 | complete | BUILD-001 | Public baseline JPEG DCT re-emission validates every consumed caller field, including entropy categories, and the unowned `Internal(String)` variant is removed in favor of typed input errors/static invariants |
| ERR-011 | P2 | complete | ERR-005 | Facade resident-encode fallback retains the non-exhaustive native source, and recode sample mismatch is explicitly classified as backend validation instead of generic backend text |
| ERR-012 | P2 | complete | ERR-009, ERR-011 | Scalar code-block helpers and every Metal token-pack route preserve typed native encode sources; focused policies and strict dependent gates pass |
| ERR-013 | P2 | complete | ERR-011 | The unused doc-hidden public scalar deinterleave compatibility wrapper that panicked on caller geometry is removed; the checked typed entry point is the only exported contract |
| ERR-014 | P1 | complete | ERR-009, ALLOC-018 | J2K/JPEG Metal surface byte access is fallible; host borrowing, typed range/readback sources, poisoned access gates, callers, policies, and strict device gates are verified |
| ERR-015 | P1 | complete | ALLOC-008, ALLOC-018 | JPEG Metal selects, builds, shares, and caches one typed fast-packet family; only capability mismatch becomes absence and hard failures retain their sources |
| ERR-016 | P1 | complete | ALLOC-013, ERR-012 | Core, Metal, and CUDA encode-stage contracts preserve typed categories, concrete sources, resource failures, and ordinary decline semantics |
| ERR-017 | P2 | complete | ALLOC-005, ALLOC-007, METALALLOC-001 | Core, Metal, and CUDA transcode crossings preserve concrete sources and distinct host/device resource categories; dependent gates pass |
| ERR-018 | P2 | complete | ERR-012, ERR-017 | Core helpers and both GPU adapters use exhaustive typed HT segment/option errors with full variant and policy coverage |
| DUP-001 | P2 | complete | ERR-001 | Genuine clones consolidated and behavior-tested |
| ADAPT-001 | P2 | complete | DUP-001 | Test-only adaptive router removed; shipped behavior retained |
| CUDA-001 | P2 | complete | ADAPT-001 | Five unreachable kernel entrypoints removed |
| CUDAERR-001 | P1 | complete | CUDA-001 | Typed completion guards and lifecycle quarantine prevent pooled reuse after uncertain asynchronous work; candidate rerun belongs to FINAL-001 |
| CUDAERR-002 | P2 | in progress | ALLOC-005 | CUDA transcode preserves typed allocation and runtime operation/detail/unavailability classification; isolated local gates pass and final combined/NVIDIA evidence remains |
| CUDAGRID-001 | P2 | complete | CUDAERR-001 | Centralized CUDA grid/block limits reject deterministic over-limit safe requests before allocation; candidate rerun belongs to FINAL-001 |
| STR-001 | P2 | complete | SAFE-001, CUDA-001 | Resident encoder split with focused parity checks |
| STR-002 | P2 | complete | SEC-001, STR-001 | Direct stacked batch split safely |
| STR-003 | P2 | complete | STR-001 | Native single-tile encoder split with byte/hook parity |
| STR-004 | P2 | complete | STR-003 | Split native roots, J2C encode/decode, and precomputed packet preparation |
| STR-005 | P2 | complete | STR-003 | Split facade encode and JPEG decoder responsibilities |
| STR-006 | P2 | complete | STR-001, STR-002 | Split Metal Tier-1, decode dispatch, and direct interpreters |
| STR-007 | P2 | complete | STR-004 | Split core/CUDA/Metal transcode orchestration by stage |
| STR-008 | P2 | complete | STR-004, STR-006 | Consolidate the remaining measured 50–63-line production clones |
| STR-009 | P2 | in progress | STR-005 through STR-008 | Runtime HT, adapter resident decode, encode/packetization, and native JP2 splits are source-closed; final combined inventory and exact-source hardware verification keep the task open |
| STR-010 | P2 | complete | STR-009 | Split mixed release-tooling roots (`xtask/main.rs`, coverage) |
| STR-011 | P2 | complete | STR-009 | Split mixed native Tier-1, DWT, and codestream implementation roots |
| STR-012 | P2 | complete | STR-009 | Sequential entropy, 12-bit rendering, baseline adapter, and stripe emission split with byte parity |
| STR-013 | P2 | complete | STR-009 | Split mixed encode/fixture comparison tooling roots |
| STR-014 | P2 | in progress | STR-009 | Runtime/color and CUDA encode structural closures are host-verified; exact-source NVIDIA and final combined evidence remain |
| STR-015 | P2 | complete | STR-009 through STR-014 | Host implementation and combined-tree structural policy verification are green without raised ratchets |
| STR-016 | P3 | complete | SEC-007, SEC-008 | JPEG fast-packet ABI/build/checkpoint/entropy ownership and the device-plan integration suite are split, ratcheted, and verified |
| STR-017 | P2 | complete | STR-010, ALLOC-002 | Oversized repository-policy owners are responsibility-split into explicit child modules with lower structural ratchets and unchanged policy behavior |
| STR-018 | P2 | complete | STR-009, CUDASESSION-001 | CUDA encode regressions use seven focused real test modules with an exact 43-test inventory and lower structural caps |
| STR-019 | P2 | complete | ALLOC-014 through ALLOC-017 | J2K facade view/decode/batch/wrap/recode owners are responsibility-split and protected by focused structural ratchets |
| STR-020 | P2 | complete | ALLOC-018, ERR-015 | The 1,145-line JPEG Metal viewport god file is a 157-line facade over focused model, policy, CPU, resident, and test owners with allocation/error parity and passing real-Metal gates |
| STR-021 | P2 | complete | METALBUF-001 | Split the incident-expanded 1,901-line Metal support god file into focused route, error, runtime, pipeline, allocation, access, dispatch, and test owners with unchanged public paths |
| JPEGCOR-001 | P2 | complete | STR-012A | Fixed ordered-dither rounding; stored and live libjpeg-turbo output now matches byte-for-byte |
| JPEGCOR-002 | P2 | complete | JPEGCOR-001 | Metal 4:2:2 interpolation now matches the CPU/libjpeg ordered-rounding contract across all routes |
| JPEGCOR-003 | P2 | complete | ALLOC-012 | TIFF `JPEGTables` normalization enforces duplicate policy over every DQT/DHT definition, preserves default byte parity, and rejects malformed/conflicting later tables |
| TOOL-001 | P3 | complete | DUP-001 | Adoption report model/render split |
| CUDA-002 | P1 | complete | SEC-001 | One exact named release-cuda gate with zero skip markers |
| PKG-001 | P1 | in progress | SEC-001 | Staged unpublished dependency closure is repaired and the clean package gate passed at `4c947c2b`; rerun after `3fd23cf2` and every later source commit |
| CLONE-001 | P2 | in progress | STR-008 through STR-015 | Repository-owned source-aware scanner passed at 1.97% duplicated production lines before `3fd23cf2`; rerun after the latest committed Rust files and at candidate freeze |
| AUDIT-001 | P2 | complete | STR-010, COV-001 | Clone and panic quality gates use shared production-source classification, explicit reviewed ratchets, fixtures, and fail-closed repository policy |
| PERF-001 | P1 | in progress | STR-004 through STR-015 | Alternating `0e78229a`/candidate process medians pass on the source committed as `3fd23cf2`: direct -3.1%, resident buffer -6.7%, resident host -15.9%; exact committed-candidate repetition remains |
| PUB-002 | P1 | complete | PKG-001, CUDA-002 | Fail-closed canonical origin, exact remote annotated tag, Release, and crates.io preflight |
| DOC-002 | P2 | in progress | SEC-001 | Reconcile public claims and keep this as the only plan |
| CONTACT-001 | P1 | blocked on maintainer action | DOC-002 | Publish and verify a working private vulnerability/conduct-reporting channel |
| PROV-001 | P1 | blocked on maintainer input | DOC-002 | Record release signoff identity and date |
| METALDEP-001 | P3 | complete | PKG-001 | Packaged Metal contents exclude the workspace patch, a standalone downstream graph resolves registry `metal 0.33.0 -> block 0.1.6`, and the maintained-binding migration has an explicit owner/trigger |
| FINAL-001 | P1 | in progress | all above | Clean local release matrix |
| RC-001 | P1 | pending | FINAL-001 | Immutable exact-SHA candidate |
| TAG-001 | P3 | deferred outside verified-RC endpoint | RC-001 | Annotated tag and guarded publication require separate authorization |

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

- 2026-07-09: replaced the 3,805-line diary with the initial 703-line task
  runbook; subsequent evidence expanded that same file in place.
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
- 2026-07-11 reconciliation found a background `reasonix` process regenerating
  an untracked local permission file containing developer-specific absolute
  paths with incorrect casing. It has no repository consumer or portable
  release purpose, so the root ignore policy now excludes it rather than
  repeatedly deleting a live tool's state. The same policy excludes the
  generated nested `Cargo.lock` under the vendored `block` source. No
  workstation/CUDA login or address is present in release-tracked source.
- The same reconciliation found 17 extracted, untracked Rust modules without
  the repository SPDX line. Each now has the canonical
  `MIT OR Apache-2.0` header, and a complete untracked-Rust scan reports no
  missing header. Legacy tracked roots do not consistently carry file-level
  SPDX comments and are not being churned solely for that pre-existing style
  difference. The active publish shell script and JPEG Metal
  shared shader were the only other changed source files missing the same
  header; both are normalized, with shell syntax and diff hygiene green.

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

### Current pre-candidate dirty-path ownership

The live 2026-07-12 worktree has grown far beyond the captured audit-time
delta. Every current path family is grouped below; this is ownership evidence
for handoff, not a claim that the tree is frozen or that generated artifacts
are current.

| Current path group | Owner | Reconciled purpose |
|---|---|---|
| `crates/j2k-native` | ALLOC-003/006/013, STR-003/004/009/011, ERR-001/012/018 | Typed bounded decode/encode owner graphs, structural splits, and source-preserving native contracts |
| `crates/j2k-jpeg` | ALLOC-004/008–012, STR-005/012/016, ERR-010/015, JPEGCOR-001/003, JPEGCACHE-001 | Bounded JPEG parse/decode/encode/packet ownership, correctness, structure, and the neutral accelerator cache |
| `crates/j2k-cuda-runtime` | ALLOC-001, CUDAERR-001, CUDAGRID-001, CUDAPOOL-001, SEC-004–007/010, STR-009/014 | Bounded asynchronous runtime ownership, safe execution geometry/lifetimes, ABI validation, pools, and focused modules |
| `crates/j2k-metal` | SAFE-001, ALLOC-018, METALCACHE/POOL/BUF-001, SEC-001/009/010, STR-001/002/006 | Checked Metal access/construction, bounded caches/pools/batches, ownership, and focused execution paths |
| `crates/j2k` | ALLOC-014–017, ERR-005/011/013/016/018, STR-005/019 | Fallible facade batches/containers/recode, explicit error contracts, and focused public orchestration |
| `crates/j2k-cuda` | ALLOC-002, CUDASESSION-001, ERR-002/004/016, PROFILE-001, STR-009/014/018 | Honest resident input/session resources, typed routing, profiling, and adapter/test structure |
| `crates/j2k-jpeg-metal` | ALLOC-018, ERR-006/009/014/015, JPEGCACHE-001, JPEGCOR-002, STR-020 | Fallible Metal JPEG batches/surfaces, one typed packet route, bounded cache integration, parity, and viewport split |
| `crates/j2k-transcode` | ALLOC-007, ERR-003/017/018, STR-007, PERF-001 | Bounded JPEG-to-HTJ2K ownership, typed stages/validation, structural split, and performance evidence |
| `crates/j2k-transcode-metal` | METALALLOC-001, ERR-017, STR-007 | Bounded Metal transcode workspace, source-preserving errors, and focused accelerator stages |
| `crates/j2k-jpeg-cuda` | ALLOC-001/008, SEC-002/007/008, ERR-015, JPEGCACHE-001, STR-014 | Safe owned CUDA JPEG decode/encode, bounded packet ownership/cache integration, and adapter structure |
| `crates/j2k-transcode-cuda` | ALLOC-005, CUDAERR-002, ERR-017, STR-007 | Actual-capacity CUDA transcode phase ownership, complete diagnostics, and focused routes |
| `crates/j2k-profile` | PROFILE-001 | Bounded typed profile text, move-only transactional summaries, and shared diagnostics |
| `crates/j2k-metal-support` | SAFE-001, ALLOC-018, METALBUF-001, ERR-009, STR-021 | Sole checked Metal resource/access boundary, shared fallible submission queue, typed sources, and focused owners |
| `crates/j2k-core`, `crates/j2k-types`, `crates/j2k-codec-math` | ALLOC-001/014/018, ERR-016/018, DUP-001, STR-008, METALALLOC-001 | Neutral allocation/SPI contracts and shared codec/math definitions without backend duplication |
| `crates/j2k-tilecodec` | ERR-008 | Typed source-preserving tile codec I/O errors |
| both test-support crates | TEST-001, JPEGCOR-001/002/003, PERF-001 | Restored regression fixtures, parity oracles, and benchmark evidence |
| `xtask` and `xtask/tests` | COV-001, CLONE-001, AUDIT-001, POLICY-001, SEM-001, PUB/REL-001, STR-010/017 | Source-aware coverage/clone/panic/API/release gates and responsibility-split policy tests |
| `.github/workflows` | CI-001, COV-001, PUB-001, CUDA-002 | Exact-SHA, source-aware, fail-closed host/GPU/publication orchestration |
| `docs`, `engineering`, and `CHANGELOG.md` | DOC-001/002, REC-001, REL-001, SEM-001, CLONE-001, PERF-001 | Canonical handoff, honest staged-release/API/unsafe/performance documentation, and final evidence placeholders |
| `scripts` | CI-001, PUB-001, REL-001 | Checked publication and workflow support scripts |
| `third_party/block-0.1.6-patched/PATCH_PROVENANCE.md` | PROV-001, METALDEP-001 | Hash-pinned ABI-delta provenance; maintainer identity/date signoff remains external |
| root Cargo/config/release files | COV-001, CLONE-001, REC-001, REL-001, DOC-002, CONTACT-001, POLICY-001, METALDEP-001 | Dependency/tooling pins, source classification, ignores, release metadata, contact policy, and patch scope |

Moving-tree reconciliation snapshot on 2026-07-12 after independent review: 65
tracked paths are modified and 28 reviewed Rust/policy modules are untracked,
with zero staged or renamed paths and two intentional tracked module deletions.
The deletions remove the CUDA resident `helpers.rs` catch-all and JPEG Metal
`pack_dispatch/common.rs`; focused downward-dependency owners replace them.
Every untracked Rust file is wired and covered by package plus structural
policy tests. API snapshots, the semver report, exact review fingerprints, and
the unsafe inventory are regenerated. No backup, reject, temporary, `_new`, or
`_fixed` variant exists, and no private CUDA connection detail appears in the
tree.

REC-001 remains in progress until this review follow-up is committed and the
worktree is clean. Changed-line coverage and staged dependency-aware packaging
are the next clean-tree gates.

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

The no-silent-failure follow-up found that release integrity built the Cargo
`workspace_members` set with `filter_map`, defaulted missing package IDs and
dependency names to empty strings, and treated a missing dependencies array as
empty. Malformed metadata could therefore omit an unpublished workspace member
or dependency from validation. The command now validates every package ID,
requires every unique string workspace-member ID to resolve to exactly one
package record, requires string versions and dependency names/requirements,
and rejects invalid dependency kinds. Eleven focused release-integrity tests,
including malformed/duplicate/unmatched metadata cases, pass; warning-denied
all-target/all-feature xtask Clippy also passes after splitting the oversized
DCT source-policy assertion into focused helpers without a lint allowance.

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
- require authenticated private vulnerability reporting before accepting a
  no-tag exact-SHA candidate
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
- pinned workspace Clippy and authoritative `cargo xtask clippy-strict`
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

The 2026-07-10 worktree replaces filename and terminal-test-module heuristics
with Cargo target roots, a canonical module graph, and Syn-based source
analysis. It classifies production, build-script, syntax-test-only, Cargo test,
example/bench/fuzz, generated, and reviewed-vendored roles; evaluates cfg state
fail closed; includes build scripts; treats shared accelerator routing as
accelerator code; and rejects changed functions without a positively covered
body record. Each lane now forces `CARGO_LLVM_COV_TARGET_DIR` and
`CARGO_LLVM_COV_BUILD_DIR` to the same unique empty directory, accepts
custom-cfg output only from that invocation, requires current output for each
selected Cargo custom-build package, and treats missing cfg values
conservatively in both polarities. It pins cargo-llvm-cov 0.8.7 and report
schema v2. COV-001 remains in progress until the split modules, policy
regressions, and frozen-candidate host/Metal/CUDA artifacts pass. No current
percentage is candidate evidence.

The 2026-07-12 nonterminal external-test-module failure was a stale hard-coded
line in the real-source regression after `backend/mod.rs` gained a production
module declaration; the reported line was the actual `#[cfg(test)]` attribute.
The analyzer required no production change. The regression now locates unique
production markers, and its focused check, all 63 coverage tests, coverage
structure policy, and strict xtask Clippy pass.

The first clean-tree measurement after the module-root and clone-fixture
repairs ran `cargo xtask coverage host --base v0.6.2` through all prerequisite
tests and policies, then failed the real 80% gate: 89,519 of 170,971 changed
executable lines (52.36%) overall and 14,436 of 70,565 accelerator lines
(20.46%). It reported 81,452 uncovered lines, 70,291 residual unmeasured lines,
2,485 functions without a covered body, 1,508 executable bodies, 2,240
one-line deferred bodies, and 2,433 opaque macros across 1,820 changed Rust
files. Reaching 80% requires 47,258 additional covered lines overall and
42,016 accelerator lines. This is a genuine release blocker, not permission to
exclude accelerator crates, macros, or deferred bodies. The next correction
must partition and aggregate authoritative host, Metal, and CUDA lane evidence
while preserving fail-closed function/body accounting, then add missing host
tests where the host lane itself remains below threshold.

The authoritative follow-up host measurement at
`114275ebdfe07f2bc3b99b130035b63c00f1efd4` uses schema
`j2k-changed-line-coverage-v3` and scope `non-accelerator-production`. It ran
the full host test/parity matrix from a detached clean worktree and failed at
76,459 of 100,541 measurable lines (76.0476%). Relative to the earlier exact
`d717f7c8` evidence, focused behavior tests added 1,315 covered lines, reduced
uncovered instrumentable functions from 450 to 390, and reduced opaque macro
blockers from 900 to 791. The remaining 80% gap is 3,974 covered lines. The
same report lists 462 executable bodies without a covered body, 1,490
one-line deferred closure bodies whose LCOV line is shared with their creation
site, and 791 uncovered opaque macros. The closure count did not move when
tests executed the surrounding lines, confirming that line-only LCOV cannot
prove these bodies. The active tooling correction must use compiler-emitted
instrumentable source regions with exact-SHA/lane provenance, require positive
counts for real closure-body regions, and fail closed on missing or stale
region evidence; it must not accept the shared creation-site line, add an
exclusion, or lower the threshold.

The next exact committed checkpoint at
`b34ab401100cf25a3f3d5fd3a138cf34049667b5` also uses schema
`j2k-changed-line-coverage-v3` and scope `non-accelerator-production`. Its
detached clean worktree ran the full host test/parity and 411-test repository
policy matrix, then failed only the genuine coverage gates at 78,149 of
100,562 measurable lines (77.7123%). This is a gain of 1,690 covered lines
over `114275eb`; the remaining numeric gap is 2,301 lines. Uncovered
instrumentable functions fell from 390 to 306, executable bodies from 462 to
433, and opaque macros from 791 to 666. The 1,490 deferred-body count remained
unchanged, so compiler-region evidence remains a separate blocker rather than
something that can be cleared by shared source-line execution. Commits after
this exact checkpoint, including `a809caeb` and `cef2ba40`, require the next
detached aggregate run before their gains count as release evidence.

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

### METALBUF-001 — P0 checked Metal resource construction

The follow-up constructor audit found a distinct memory-safety boundary that
SAFE-001's checked *access* work did not cover. `metal` 0.33's safe
`DeviceRef::new_buffer`, `new_buffer_with_data`,
`new_buffer_with_bytes_no_copy`, and `new_texture` wrappers immediately create
foreign owning handles from the Objective-C result. `CommandQueueRef`'s
`new_command_buffer` and `CommandBufferRef`'s compute/blit encoder constructors
likewise create foreign references without a null check. They do not expose a
nil result as `Result`; device allocation/resource pressure can therefore reach
the foreign-types non-null invariant instead of a typed Rust error. The
repository still had 132 production direct buffer-constructor calls and one
texture call across `j2k-metal` and `j2k-jpeg-metal`, plus 207 production direct
command-buffer/compute/blit-encoder calls across the three Metal adapters. The
five public `j2k-metal-support` allocation helpers were infallible. Metal
coefficient transcode independently implemented the correct raw-selector/nil-
check buffer pattern, creating a second unsafe allocation boundary rather than
one shared source of truth.

Migration then confirmed the lifetime half of the same P0: J2K Metal
`decode_store_component_and_capture` wrapped caller `&mut [f32]` with
`newBufferWithBytesNoCopy` and retained/returned that `Buffer` inside
`MetalStoreDecoder` after the Rust borrow ended. The device handle could
therefore outlive or alias mutable caller storage. That path must use owned
shared Metal storage, synchronously copy the completed result into the caller
slice, and retain only the owned capture; a nil-only constructor patch would
not close the dangling-owner defect.

Treat this as P0 because otherwise safe, bounded public decode/encode calls can
enter undefined behavior under ordinary device-memory pressure. Feature work
does not take precedence over closing the pattern-equivalent constructors.

The follow-up ownership review found a second issue in the checked command
boundary. Apple's [Objective-C memory-management
policy](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/MemoryMgmt/Articles/mmRules.html)
says objects returned from methods outside the `alloc`/`new`/`copy`/
`mutableCopy` families are not caller-owned and must be retained when they need
to survive the receiving method scope. `commandBuffer` and the command-encoder
factories are in that non-owning family. Apple's [`commandBuffer`
documentation](https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer%28%29?language=objc)
states that the command buffer strongly retains resources encoded into it; that
does not make a Rust borrow of the returned command-buffer object valid for the
queue's entire lifetime. The support API must therefore retain these non-null
results and return owned `CommandBuffer`/encoder handles rather than inventing
borrow lifetimes tied to their parents.

Required closure:

1. Keep the only raw `newBufferWithLength:options:`,
   `newBufferWithBytes:length:options:`, `newTextureWithDescriptor:`,
   `commandBuffer`, `computeCommandEncoder`, and `blitCommandEncoder` sends in
   `j2k-metal-support`; check Objective-C dispatch and nil before creating an
   owning foreign handle or borrowed foreign reference. Apply the same rule to
   `MTLTextureDescriptor`/`MTLCompileOptions` factories plus shader-library and
   compute-pipeline creation: foreign-types 0.5 uses a non-null representation,
   so a post-construction `as_ptr()` check is already too late.
   Retain non-owning command-buffer/encoder results before returning owned Rust
   handles; do not expose a parent-lifetime borrow across an autorelease pool.
2. Enforce checked `usize` multiplication/conversion, the selected device's
   `max_buffer_length`, and the repository's per-allocation cap before every
   request. Zero-byte logical requests may use one real byte, but zero-sized GPU
   ABI element types must remain errors.
3. Remove the old public infallible support constructors. Migrate every
   production J2K/JPEG/transcode Metal direct allocation, command buffer, and
   encoder construction plus the transcode allocation duplicate to typed
   checked helpers; resource errors remain hard errors and must never be
   converted into Auto fallback, unsupported input, absence, or panic.
4. Preserve caller context and `MetalSupportError` in each adapter's source
   chain. Account allocator-reported buffer/texture size in the surrounding
   live budget and pool/cache policy rather than trusting requested size.
5. Do not expose a no-copy buffer factory whose returned cloneable `Buffer`
   erases a Rust borrow. The two former borrowed paths now copy into
   Metal-owned storage, and the unused low-level no-copy selector/API has been
   removed rather than retained as public unsafe surface.
6. Add a source ratchet rejecting direct safe Metal buffer/texture constructors
   outside the support boundary and explicit, reviewed test fixtures.

Acceptance requires pure exact-cap/one-over/overflow/zero-size tests, testable
nil-result seams, shared/private/upload/typed/texture and command/encoder Metal
tests, adapter source-chain and non-fallback tests, borrowed-input ownership regressions,
strict Clippy, updated unsafe/API/semver/changelog reports, and final serialized
Apple Metal execution. The centralized support boundary is implemented and
passes 18/18 all-target tests plus strict no-deps Clippy, including buffer,
command-buffer, and encoder nil rejection plus command resources retained across
their creation autorelease pools. An injected-nil regression covers texture descriptors,
textures, and compile options; shader libraries and pipeline states now also
cross raw checked boundaries. The transcode Metal
duplicate allocator and all of its production direct command/encoder calls are
removed, with typed support errors retained as sources; its package verification
initially waited for the concurrently changing encode-stage error API and is
superseded by the final closure evidence below.
The incident-expanded 1,901-line support root was not accepted as new debt:
STR-021 replaced it with a 53-line facade over 49-line route, 256-line error,
271-line runtime/command, 263-line pipeline, 269-line allocation, 192-line
buffer-access, 78-line dispatch, and 366-line test owners. All 18 real-Metal
tests, strict no-deps Clippy, rustdoc, and diff hygiene remain green after the
split, with public names re-exported from the crate root.
All production, test, example, and benchmark Rust sources under the J2K, JPEG,
and transcode Metal adapter crates now contain zero direct safe Metal buffer,
texture, descriptor, queue, command-buffer, compute-encoder, or blit-encoder
constructors. The three-part repository ratchet passes: adapters may not call
those constructors or form foreign handles, raw selectors stay in their
focused support owners, borrow-erasing no-copy APIs remain absent, and the
support facade/children retain explicit size caps. The updated unsafe inventory
also passes `cargo xtask unsafe-audit`; the serialized adapter closure follows.

Final local closure on the stable Apple source boundary: all 22 support tests
pass, including nil buffer/texture/descriptor/compile/library/pipeline seams,
retained command/encoder ownership across autorelease pools, exact allocation
caps, and typed range/readback behavior. The six constructor/resource source
ratchets and warning-denied support Clippy pass. Full all-feature packages pass
for J2K Metal (255 library and 54 device tests plus integrations/docs), JPEG
Metal (196 library plus 37 integration/device tests and docs), and Metal
transcode (58 tests and docs); warning-denied all-target/all-feature Clippy
passes for all three adapters, as does no-default J2K Metal Clippy. METALBUF-001
is complete. FINAL-001 owns the immutable-candidate repetition and does not
weaken this closed P0 boundary.

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

### SEC-002 — JPEG owned-output initialization

The explicit JPEG lint escalation surfaced a `clippy::uninit_vec` warning in
`decoder/output_format.rs`. The safe `allocate_output_buffer` helper reserved a
`Vec<u8>`, called `set_len` before any element was initialized, and relied on
downstream writers to overwrite every byte. That violates `Vec::set_len`'s
initialized-element safety precondition and creates an information-disclosure
surface if any successful route leaves a byte unwritten. The helper feeds
public full, region, scaled, lossless, and 12-bit owned-output decode paths, so
this is a P0 release stop even though no leaking fixture was reproduced.

Required closure:

1. Replace the preinitialized-length pattern with safe initialized storage for
   0.7; do not suppress `uninit_vec` or encode the assumption in a safe helper.
2. Compare owned output against caller-owned buffers prefilled with a nonzero
   sentinel across every distinct 8-bit, 12-bit, lossless, full, region, and
   scaled writer family.
3. Search all production `set_len`, `MaybeUninit`, and `assume_init` sites for
   the same ordering error. Retain only sites that write the complete spare
   capacity before changing length.
4. Make repository policy reject expectations for `uninit_vec`,
   `uninit_assumed_init`, `invalid_value`, undocumented unsafe blocks, and
   unsafe operations in unsafe functions.
5. Rerun strict all-target JPEG Clippy, the full JPEG suite, owned-output
   regressions, the structural performance guard, and changed-path coverage.

Completion evidence: d4be4d20 replaced the preinitialized-length helper with
initialized storage and added the sentinel parity matrix. The equivalent-
pattern scan retained only bounded spare-capacity writes. Implementation status
is complete; all pre-current suite/performance numbers are historical and
FINAL-001 must rerun them on the combined tree.

### SEC-003 — external comparator FFI hardening

The comparison-tool lint escalation exposed several related bounds and
concurrency defects at the optional OpenJPEG/Grok boundary:

- OpenJPEG and Grok ROI endpoints used unchecked addition; OpenJPEG then
  narrowed coordinates to signed C integers without validation.
- Grok constructed and copied a Rust slice from the shim's pointer and
  reported length before independently checking dimensions, the shared 512 MiB
  cap, and exact length equality.
- the Grok C shim used an unsynchronized `static int` pseudo-once even though
  `jp2k_roi_batch_compare` intentionally performs concurrent Grok decodes.
- the shim silently truncated a public `u32` reduction factor to `u8`, wrapped
  unsigned 32-bit component values through `int32_t`, and reinterpreted unknown
  component layouts as signed 32-bit samples.
- OpenJPEG component precision could drive an invalid shift, and raw callbacks
  dereferenced null state/buffer arguments without a defensive error return.

Current source uses checked ROI endpoints, checked reduction and precision,
RAII-owned Grok output, pre-slice length/cap validation, checked C sample
extraction, null-safe callbacks, and a Rust `Once` around a dedicated Grok
initializer. The C shim is built with warnings and extra warnings promoted to
errors plus supported conversion/sign-conversion diagnostics. Boundary unit
tests and an eight-worker Grok decode test cover the repaired contracts.

Implementation completed in 343bbab3, with comparator lint closure in
a6d1f370. Candidate proof still requires fresh optional-library unit/
integration tests, strict Clippy, the C warning gate, and public-API
verification under FINAL-001; never reuse binaries built before the FFI edits.

### SEC-007 — P0 safe CUDA JPEG output-coverage incident

The 2026-07-10 CUDA equivalent-pattern scan confirmed that the doc-hidden but
safe public `CudaJpegRgb8DecodePlan` boundary trusted caller-supplied MCU grid,
checkpoint, bit-reader, and Huffman metadata. Three concrete failure classes
made the existing path release-blocking:

- a first checkpoint after MCU zero, duplicate/decreasing checkpoint ranges,
  or a sampling/grid mismatch let a decode thread store an initial OK status
  and return without writing its assigned pixels; the owned path allocated
  uninitialized device bytes and could return those bytes as successful RGB8
  output
- host allocation math was `usize`, while kernel output indices are `u32`; a
  valid 65,500 by 65,500 RGB8 request could exceed 4 GiB, wrap device row
  offsets, overwrite early rows, and leave an uninitialized tail
- public Huffman metadata admitted value counts beyond the fixed 256-entry
  device array and noncanonical offsets; Rust device array indexing retains a
  bounds check, so the demonstrated consequence is a kernel trap/denial of
  service rather than a proven raw out-of-bounds read, but the safe API still
  failed to reject the plan deterministically

The same review found process-abort allocation debt on the reachable high-level
path: checkpoint construction reserved one record per MCU even when cadence
caps the actual nonrestart plan near 2,048, successive checkpoint conversions
used infallible collection, scan termination duplicated entropy bytes, and
destuffing/restart-offset growth was infallible.

Required closure:

1. Before context binding, allocation, upload, or kernel lookup, derive the
   exact sampling grid (4:2:0 = 16x16, 4:2:2 = 16x8, 4:4:4 = 8x8), require the
   plan grid to match, and require checkpoint ranges to partition `[0,
   total_mcus)` strictly from MCU zero without overlap or gaps.
2. Validate every checkpoint entropy position, left-aligned accumulator,
   63-bit producer bound, strictly advancing consumed-bit position, initial
   predictor state, and reserved field before driver work.
3. Require the last active RGB byte, including pitch, to fit the kernel's
   `u32` address domain. Keep owned RGB output zero-initialized as defense in
   depth even after the coverage proof. Reject a full-tile RGB8 request at
   capability routing when its tight output already exceeds that address
   domain, before checkpoint or packet construction.
4. Make CUDA Huffman representation fields crate-private, reconstruct and
   validate canonical code progression, reject the JPEG-prohibited all-ones
   code, and enforce baseline DC/AC symbol roles for all six decode and
   diagnostic tables.
5. Defensively repeat grid, output, checkpoint, bit-state, Huffman-index, and
   coefficient bounds in the device module. Every malformed branch must store
   a nonzero status instead of returning with the initial OK row.
6. Replace caller-sized infallible status/diagnostic/checkpoint vectors with
   shared fallible allocation. Reserve checkpoint counts from actual cadence or
   restart semantics, borrow already terminated entropy prefixes, and cap
   required copies/growth with the repository's 512 MiB host-allocation policy.
7. Add pure adversarial boundary tests, source-order/structure policy, real
   4:2:0/4:2:2/4:4:4 owned and caller-buffer parity, strict Clippy, CUDA-Oxide
   compilation, and exact-source NVIDIA execution.

Closure evidence on 2026-07-11: checkpoint validation now models every worker
range as `[checkpoint[i], checkpoint[i + 1])` and the final range as
`[last, total_mcus)`. Requiring boundary zero, strictly increasing shared
boundaries, and a final start below `total_mcus` proves complete,
non-overlapping coverage. Both safe public decode entrypoints converge on one
validated launch path that zeroes the full u32-addressable output extent,
including inter-row pitch padding, before any upload or decode launch. Pure
validation tests cover exact odd grids, nonuniform complete partitions,
coverage failures, malformed bit state, and the 4-GiB boundary; host/device
source policy also requires each sampling kernel to advance exactly once per
MCU in its validated half-open range.

The exact transferred source built both CUDA-Oxide JPEG modules for `sm_80`
with skip gates made fatal, then passed the sentinel-backed caller-owned pitched
output regression on the supplied RTX 4070 SUPER under
`J2K_REQUIRE_CUDA_RUNTIME=1`, `J2K_REQUIRE_CUDA_OXIDE_BUILD=1`, and
`J2K_REQUIRE_CUDA_JPEG_HARDWARE_DECODE=1`. Local validation tests, focused
CUDA safety policies, adapter check, and strict Clippy were also green. SEC-007
is complete; the repository-wide frozen-tree rerun remains tracked only by
FINAL-001.

The Huffman boundary is deliberate and grounded in
[ITU-T T.81 Annex C](https://www.w3.org/Graphics/JPEG/itu-t81.pdf). Annex C
reserves the all-1-bits code for encoder compliance. Generic CPU canonical
derivation remains decoder-compatible with complete prefix tables found in
existing inputs, while the strict CUDA safe-plan validators reject an
all-1-bits assignment in caller-supplied backend metadata before launch. The
device-side validator repeats that rejection defensively.

### SEC-008 — P1 fast-packet table-selector panic

The 2026-07-10 safe-boundary review found that `quant_for_component` accepted
the SOF component's untrusted `Tq` byte and indexed a four-entry quantization
table array directly. A malformed selector above three could therefore panic
through the safe grayscale fast-packet builder. The color builder happened to
return an earlier typed decoder error, but that ordering did not make the
shared table-selection helper safe.

Closure is complete:

- quantization lookup now uses `.get(usize::from(slot))`, returning the
  existing `FastPacketError::MissingQuantTable { slot }` without changing the
  public error contract
- defensive Huffman lookup uses the same bounds-checked pattern and preserves
  `MissingHuffmanTable { kind, slot }`
- a pattern-equivalent scan confirmed decoder planning already uses `.get()`;
  SOS Huffman selectors are also bounded by parsing before fast-packet lookup
- the malformed-selector integration regression mutates a grayscale SOF to
  `Tq = 255`, calls `build_gray_packet`, and proves the typed error without a
  panic

The focused fast-packet suite passes 12/12, the device-plan suite passes
70/70, the library suite passes 234/234, and strict all-target/all-feature
`j2k-jpeg` Clippy plus the focused repository structure/security policies pass.

### SEC-009 — P1 Metal prepared-plan cache collision isolation

The 2026-07-10 residual Metal review found four prepared-plan cache instances
that treated a 64-bit `DefaultHasher` result as complete request identity:
session direct-grayscale, session direct-color, session region-scaled color,
and global region-scaled color. A digest collision could therefore return a
prepared plan containing another request's decoded coefficient data and GPU
resources. The forced-collision proof demonstrates the wrong-result and
cross-request data-selection consequence, but it does not demonstrate a
practical collision against the former SipHash-1-3 digest without infeasible
search. The release severity is therefore P1 rather than P0.

Closure is complete in the working tree:

- every cache uses a per-instance randomized digest builder and retains
  digest buckets rather than replacing caching with a linear scan
- every entry owns the full compressed input bytes and records format, ROI,
  scale, and prepared-plan kind; a bucket candidate is returned only after
  equality across every field
- direct and hybrid cache scopes share one focused cache implementation, remain
  capped at 128 entries per existing session/global limit, retain no more than
  the shared 512 MiB host-allocation ceiling in owned key bytes, explicitly
  skip this optional cache for a single key above that ceiling, reserve bucket
  metadata fallibly, and evict to the byte budget before copying a new key so
  insertion never transiently owns two full budgets; a subsequent
  copy-allocation failure may reduce cache residency but is surfaced and cannot
  change decoded output
- allocation/invariant failures and poisoned locks are surfaced as typed Metal
  errors instead of becoming silent misses; no borrowed input or pointer
  identity is retained
- constant-digest tests prove that distinct inputs and distinct ROIs coexist
  without cross-hit, and separate tests prove exact-key reuse, owned identity,
  replacement, entry-count eviction, aggregate-byte eviction, and fail-closed
  cache bypass for one oversized key without rejecting an otherwise valid
  decode

Focused evidence on 2026-07-10: all six pure cache tests passed; the full
`j2k-metal` library suite passed 213 tests with 18 established runtime ignores;
and two fail-closed real-Metal cache-reuse tests passed on an Apple M4 Pro for
repeated and distinct region-scaled color requests. Strict all-feature,
all-target crate-local Clippy passed with `--no-deps`. The dependency-inclusive
command's former native import finding is now closed under ALLOC-003; a fresh
combined-tree Metal rerun remains FINAL-001 evidence rather than being claimed
here.

### SEC-010 — P1 padding-free safe GPU ABI byte views

The 2026-07-11 safe-boundary review found that `GpuAbi::as_bytes` and
`slice_as_bytes` exposed complete object representations through safe APIs,
while several `repr(C)` implementors contained implicit tail padding. Rust does
not initialize padding during ordinary construction, so reading those bytes as
`u8` is unsound and could also copy stale host bytes into a device upload.
JPEG Metal independently reinterpreted a padded entropy-checkpoint array and
accepted arbitrary `T` at a generic typed-upload boundary.

Closure is complete in the working tree:

- `GpuAbi` now requires no internal or tail padding, complete initialization,
  validity of every bit pattern, and a compile-time field-offset/end proof;
  size-only tests or safety comments are explicitly insufficient
- all 31 CUDA implementations plus two nested records are declared through one
  focused proof macro; policy forbids direct implementations outside it
- `CudaJpegEntropyCheckpoint`, `CudaHtj2kCleanupMultiKernelJob`,
  `CudaHtj2kDequantizeKernelJob`, `CudaJ2kIdwtMultiKernelJob`, and
  `CudaJ2kStoreRgb8MctBatchJob` occupy their former implicit tails with
  initialized fields mirrored by the device definitions. Their respective
  40/64/40/128/128-byte sizes and every existing field offset are unchanged
- JPEG Metal gives its 40-byte checkpoint the same explicit final word, mirrors
  it in Metal, constrains reusable typed uploads to `T: GpuAbi`, and routes
  restart-offset, checkpoint, and status byte views through the safe trait
- the unsafe inventory now follows the moved CUDA proof module and passes

Local default/all-feature checks, strict CUDA-runtime/JPEG-CUDA and JPEG-Metal
lib/test Clippy, pure byte-view/layout tests, ABI tests, and both CUDA/Metal
policies pass. A supporting mid-remediation RTX 4070 SUPER run built all ten
strict `sm_89` cuda-oxide projects and passed the required CUDA-runtime suite
257/257. It must be repeated after source freeze. The complete repository Miri
lane now passes, including the pure ABI byte-view regressions; FINAL-001 retains
only the frozen-source repetition requirement.

### SEC-011 — P1 runtime-safe public AVX2 benchmark wrapper

The same review found that safe public `bench_idct_avx2_block` unconditionally
called a `#[target_feature(enable = "avx2")]` function. Existing benchmark and
test callers happened to feature-detect, but an arbitrary safe caller could
violate the callee precondition on an older x86-64 CPU.

The wrapper keeps its public safe signature, performs its own runtime AVX2
detection, and calls the scalar IDCT when the feature is absent. A pure dispatch
selector covers both decisions; the x86 parity suite now always exercises the
safe wrapper. Host and x86 cross-checks, five IDCT parity tests, strict JPEG
Clippy, unsafe-audit, and the new SIMD-boundary policy pass. The x86 environment
reported no AVX2 and therefore verified the formerly unsafe fallback case; the
actual AVX2 branch remains covered by the final real-x86 candidate lane.

### ALLOC-001/002 — fallible GPU staging and honest resident input

The equivalent-allocation scan found two distinct problems that require
different remedies:

- safe CUDA runtime encode, packetization, diagnostics, and readback methods
  construct caller/metadata-sized host vectors with `vec!`, `with_capacity`,
  `to_vec`, or infallible collection even though their APIs already return
  `Result`; capacity failure must become a typed error before launch/readback
- `encode_lossless_cuda_tile_with_report` allocates a zero-filled host image of
  `output_width * output_height * bytes_per_pixel` solely to construct
  `J2kLosslessSamples`, while the real input is already CUDA-resident and the
  whole-tile accelerator ignores those fake bytes. This is avoidable
  O(image-size) host-memory amplification, not merely a missing error mapping.

Required closure:

1. Provide one focused crate-level fallible allocation utility per error
   domain and use exact reservation before initialization or copy. Preserve
   empty fast paths and distinguish arithmetic validation from allocation
   failure.
2. Migrate runtime HTJ2K encode/packetization, J2K/JPEG readback staging, and
   ownership-sensitive device/pinned-buffer and kernel caches. During
   in-flight work, failed host cache growth must retain rather than destroy a
   referenced CUDA allocation. Then scan the CUDA adapter for equivalent safe
   `Result`-returning paths.
3. Introduce an explicit resident-input whole-tile encode contract carrying
   validated geometry and format but no host pixel slice. It must either
   dispatch the resident accelerator or return an error; it must never expose
   an invalid/dangling slice or silently enter CPU fallback.
4. Remove the full-image dummy allocation. Preserve marker bytes, stage
   accounting, external-validation requirements, and host/CUDA-resident output
   behavior.
5. Add impossible-capacity unit tests, source policy preventing reintroduction
   of fake resident pixels, strict all-target Clippy, host behavior tests, and
   final exact-source NVIDIA parity.

Status: ALLOC-001's original infallible-allocation migration is complete, but
the host-side source was reopened by the allocator-capacity red-team below.
`j2k-core` owns the neutral fallible-Vec primitive; CUDA runtime and adapter
errors preserve typed allocation failures. Caller/image-derived runtime
encode, packetization, decode, JPEG, readback, and adapter collections reserve
before mutation. Pool,
pinned-staging, and kernel-cache growth is ownership-safe on allocation
failure. Async HTJ2K cleanup and IDWT allocate and populate metadata-retention
vectors before launch, so host OOM cannot strand queued work before its guard
exists. Adapter packetization uses typed allocation classification rather than
string matching, rejects sparse public descriptor state indices before
allocation, and checks state-count arithmetic.

The equivalent-pattern census leaves only explicitly bounded CUDA encode
vectors: one resolution vector capped by validated JPEG 2000 decomposition
levels; fixed one/three-subband vectors; a 16-entry tag-tree path; and three
tag-tree arrays capped at 2,048 nodes. Remaining `Vec::new()` constructors
either are empty fast paths or grow only after a fallible reserve. Source
policy records those exceptions and forbids image-derived `collect`, `to_vec`,
and infallible capacity growth. Owned quantized, packetized, and code-block
results now move through consuming accessors instead of cloning payloads.

Local evidence is `j2k-core` 6 unit plus 35 API tests, CUDA runtime 237/237,
`j2k-jpeg-cuda` 8/8, and `j2k-cuda` 95/95. Strict all-target/all-feature
Clippy passes for the runtime and both CUDA adapters; `j2k-cuda` also passes
the no-feature strict lane. Allocation/ownership source policies pass. Full
repository lint is 203 passed, 1 ignored, with one unrelated JPEG policy path
failure pending reconciliation after its module split. Panic-surface remains
within its 17 unwrap / 106 expect ratchet. Exact-source NVIDIA execution is
still required before ALLOC-001 is marked complete.

Post-checkpoint actual-capacity reopening (2026-07-11): the neutral helper
returns the allocator's `Vec` but intentionally does not enforce a codec cap.
The generic wrappers in `j2k-cuda`, `j2k-jpeg-cuda`, and `j2k-cuda-runtime`
map reserve failure but do not inspect returned `capacity()`; adapter growth
also uses `try_reserve`, whose geometric over-allocation can exceed the logical
request further. Existing call-site preflights therefore prove requested
counts, not the actual simultaneously live host owners. ALLOC-001 is source-
open until each image/caller-derived wrapper is either incorporated into an
aggregate phase budget with allocator-reported capacity or proven to have a
fixed specification bound recorded by policy. Add model-driven overcapacity,
existing-vector growth, exact-cap, and one-over regressions; keep host and
device allocation domains separate; and rerun the exact-source NVIDIA lane.

Actual-capacity source closure (2026-07-11): the neutral
`HostAllocationBudget` now reconciles allocator-reported capacity rather than
requested length, and runtime/adapter `HostPhaseBudget` owners carry every
simultaneously live host allocation through the relevant phase. Runtime HTJ2K
cleanup/dequantization and queued completion, IDWT sequences, color store,
encode planning/status/compaction/readback, packetization launch/completion,
DWT, and transcode readback all use live-byte-aware handoffs. Adapter direct
and color plans account their nested owner graphs, while packetization accounts
descriptor state, tag trees, runtime descriptors, and temporary snapshots
before launch. Growth remains fallible and pre-launch; status/error cleanup
retains the asynchronous ownership and quarantine guarantees. Exact-boundary,
allocator-overcapacity, and existing-vector-growth regressions plus focused
source-ordering and file-focus policies ratchet the contract.

Local closure evidence is 255/255 CUDA-runtime tests and strict
all-target/all-feature Clippy; the CUDA adapter passes 122 library tests plus
7/5/39/15/7 integration tests, its strict all-feature and no-default-feature
Clippy lanes, and 57 no-default-feature library tests. The CUDA runtime safety
suite passes all allocation, lifecycle, queued, HTJ2K-output, submit, and ABI
checks; its only two current failures are concurrent JPEG-policy assertions
owned by JPEGCACHE-001. ALLOC-001 remains in progress only for the frozen-source
NVIDIA lane and the final combined-tree rerun.

ALLOC-002's no-byte resident input descriptor and resident-only whole-tile hook
are implemented. The path reuses native validation/planning/finalization,
converts a declined hook to an explicit error, cannot enter CPU component or
packet fallback, and has byte-parity, strict-option, dispatch-accounting,
invalid/huge-geometry, source policy, 285-test native regression, and strict
host Clippy evidence. Its final exact-source NVIDIA parity remains pending the
combined-tree run.

The 2026-07-11 closure audit found no remaining fake or dangling host sample
owner in this route. It added a real-device regression that starts with an
externally created tile context and a completely uninitialized `CudaSession`,
proves that the first encode binds that exact shared context, then proves a
cloned session reuses the same cached HTJ2K resource set without a second
upload. Batch reuse and cross-context rejection now live beside it in a
focused 168-line session test owner. The driver-independent state-machine
tests still cover failed initialization, compatible retry, and rejection
before cache lookup or initialization. The exact resident policies pass
12/12; the CUDA encode inventory policy owns all 43 tests; the all-feature
CUDA package passes 121 library tests plus every integration/doc target; and
warning-denied all-target Clippy passes for the types, native, facade, runtime,
and CUDA adapter, including the adapter no-default lane.

Follow-on encode architecture is deliberately separate from ALLOC-001.
CUDASESSION-001 closes the former per-tile HT-table upload and session-ownership
gap. The `to_cuda_buffer` route still first assembles host codestream bytes,
extracts stable metadata, and uploads the bytes, but the resident outcome now
retains only metadata plus the device codestream and drops the host payload.
`codestream_assembly_used=false` is therefore accurate. Avoiding the remaining
transient host assembly requires real device tier-2 assembly, not another
allocation-helper substitution.

This is the explicit ALLOC-001 overlap, not a reason to overstate ALLOC-002:
the fake input image is gone, but the transient host codestream owner, its CUDA
upload staging, and a later caller-requested download remain in ALLOC-001's
allocator-reported-capacity and aggregate-phase census. They must close under
that shared allocation contract rather than by reintroducing adapter-local
resident-input budget logic.

### ALLOC-003 — native tiny-ROI/full-tile decode workspace amplification

The native region API capped output-sized channel storage, but tile decode
called decomposition construction before deriving its ROI work plan. A crafted
single-tile SIZ as large as 60,000 by 60,000 combined with a 1x1 ROI therefore
kept the output allocation tiny while `build.rs` infallibly created full-tile
`f32` (and sometimes `i64`) coefficient storage. The same path expanded
precincts, code blocks, quality-layer entries, and tag-tree nodes from header
geometry. This was a small-input process-abort/resource-exhaustion class and is
tracked as P1.

Closure:

1. Decomposition construction now computes one checked aggregate plan before
   mutating storage. It accounts for both coefficient representations and the
   exact tile-decomposition, decomposition, sub-band, precinct, code-block,
   layer, and tag-tree element counts under the repository 512 MiB decode cap.
   Precinct roots and global code-block/layer counts are charged before any
   potentially large precinct walk.
2. Every planned vector is fallibly reserved before initialization. Build-time
   pushes are capacity-guarded, coefficient ranges and geometry use checked
   arithmetic, and the builder verifies that observed counts exactly match the
   plan. Allocation failure is the typed
   `DecodingError::HostAllocationFailed`; an over-cap request remains
   `ValidationError::ImageTooLarge`.
3. Packet-segment growth shares the unused portion of the same active-tile
   budget and records allocation failure across the legacy option-based packet
   parser so lenient mode cannot suppress it. Full/ROI/direct IDWT growth and
   channel/SIMD storage also reserve fallibly rather than moving the abort
   surface later in decode.
4. Regression coverage includes a 60,000-square SIZ with a 1x1 ROI, a
   4,096-square/4x4-code-block/32-layer metadata amplification case, exact
   tag-tree plan-versus-builder counts for empty, skinny, odd, and square
   trees, packet-segment remaining-budget enforcement, and SIMD padding/error
   behavior. Existing classic/HTJ2K ROI crop parity and pruning behavior remain
   covered by the full native suite.

Evidence on 2026-07-10:

- `cargo test -p j2k-native --lib`: 291 passed, 0 failed, 1 established
  diagnostic ignore.
- `cargo clippy -p j2k-native --all-targets --all-features -- -D warnings`
  passed.
- `cargo check -p j2k-native --no-default-features` passed.
- `cargo fmt -p j2k-native --check`, scoped `git diff --check`, and semantic
  reference checks for the reserve helper, active structural budget, packet
  allocation error, and typed host-allocation error passed.

The later ALLOC-006 pattern-equivalent review invalidated the complete status
above. Header and all parsed tile metadata remain live while decomposition,
segments, output channels, ROI plans, direct-plan jobs, and parallel decoded
blocks are built, but several of those owners reset their accounting to a fresh
512 MiB ceiling. `RoiPlan::build`, direct-plan job arrays, parallel subband
collections, and combined code-block payload copies also retained production
infallible growth. ALLOC-003 is therefore reopened until one retained baseline
flows through normal decode, both direct-plan routes, reversible recode,
decomposition, segment growth, ROI planning, and block-result/payload phases;
every Result-returning owner must reserve fallibly, and reused contexts must
not hide old capacity outside that baseline.

Working-tree correction checkpoint (2026-07-10): an independent JP2 boundary
review found that `ImageBoxes` and the public `Image` ICC clone remained live
while normal decode, both direct-plan paths, reversible coefficient recode,
palette/channel postprocessing, and owned output each started from a fresh
cap. One exact parsed-Image metadata baseline now sums actual header, JP2/JPH
box, nested payload, and ICC capacities at construction. That baseline is
threaded into tile metadata planning and every listed consumer; output and
postprocessing budgets include it before their next allocation, while a
borrowed component handoff deliberately removes it only from the returned
post-`Image` live baseline. Header accounting now uses allocator capacities
rather than logical lengths. A follow-up parse-time check now feeds already
retained JP2/JPH box capacity into the codestream-header marker ledger before
SIZ allocation, reconciles actual component/override/flattened capacities, and
builds the raw-codestream synthetic color owner fallibly under the same header
baseline. The native library compiled warning-free before this parse-time
increment; the focused parsed-Image source ratchet passes, while updated
boundary/behavior and strict evidence remain before this increment can count
toward ALLOC-003 closure.

Decoder-context reuse regression (2026-07-11, P1 reopened): the post-allocation
review found that normal decode clears every reusable component owner at call
start and several owned-output adapters replace `channel_data` with a fresh
empty vector after packing. That contradicts the public `DecoderContext`
contract promising allocation reuse and can turn repeated decode into repeated
large allocation. Existing reuse regressions inspect the cleared vector with
`iter().all(...)` but never require it to be nonempty, so they pass vacuously.
Closure must retain reusable capacity without returning stale samples, include
that retained capacity honestly in the next decode budget, and add nonempty
pointer/capacity-before-and-after behavior tests for packed and component
outputs. This is an active release blocker; changing the documentation to hide
the regression is not an accepted substitute.

Tile-part cursor correction (2026-07-11): the move-only owner sweep found that
every segment parse cloned the retained `TilePart` before reading it. For
separated PPM/PPT packet headers this transitively deep-cloned the complete
`Vec<BitReader>` and packet-length vector once per normal decode, direct-plan
build, or reversible recode. That duplicate was outside the active retained
baseline and could approach the marker-metadata cap while the original remained
live. Segment parsing now constructs a per-operation, non-allocating
`TilePartCursor`. It borrows the retained header and packet-length slices and
owns only cheap `BitReader` cursor values plus header/length indices. The
retained `TilePart` graph contains no mutable reader index, so a fresh cursor
can repeat normal decode, grayscale direct-plan construction, and reversible
coefficient recode without mutation or allocation.

`Clone` was removed from `Tile`, both tile-part variants, `TilePart`, packet-
length metadata, PPM/PLM/PLT marker aggregates, ROI plans, SIMD/component sample
owners, and the nested component coding/quantization graph. The same sweep made
native encode parameters and component ROI plans move-only. Header override
initialization no longer requires `T: Clone`: it reserves fallibly and fills
`None` with `resize_with`. Test fixtures construct independent owners
explicitly instead of preserving production deep-clone contracts. Multi-reader
separated-header and PLT reset/mismatch tests, repeated decode/recode and direct-
plan tests, the six-test PLT/PLM/PPM/PPT integration suite, all 512 native
library tests (one established ignore), all native integration/bench targets,
warnings-denied all-feature/no-deps Clippy, six native-decode policies, and all
25 native-encode allocation policies pass in the isolated verification target.
This closes the tile-part clone remainder; ALLOC-003 remains open only for its
other recorded aggregate-baseline owners.

ALLOC-003 closure (2026-07-11): the remaining aggregate-baseline and decoder-
reuse owners are reconciled. Normal decode accounts retained parsed image/tile
metadata before decomposition, ROI, packet segments, serial or parallel Tier-1,
and output growth. Both direct-plan routes, reversible coefficient recode, and
palette/channel postprocess use the same actual-capacity budget. Reused decode
contexts release stale tile scratch but retain real component capacity, count it
as the next call's baseline, reset every sample, and keep nonempty pointer/
capacity snapshots across packed, component, and exact-i64 output paths; the
former vacuous clear-vector regression is forbidden by policy.

The full all-feature native package passes 598 tests with one established
diagnostic ignore, strict all-target/all-feature Clippy passes with warnings
denied, and the no-default-feature library check passes. All ten focused
`native_decode` repository checks and all three tile-metadata ownership checks
pass. The stale 707-line tile-metadata owner was tightened rather than waived:
four independent exact-boundary/rollback tests now live in an 83-line child,
the parent is 632 lines under a reduced 640-line ceiling, and all six affected
tests pass. ALLOC-003 is complete; exact immutable-candidate repetition belongs
to FINAL-001.

### ALLOC-004 — bounded CPU JPEG entropy and frame assembly

The post-ALLOC-001 equivalent scan found a separate host-only P1 denial-of-
service path. A cap-valid grayscale request with restart interval one could
create 8,396,640 independently allocated entropy vectors. Their 64-bit vector
descriptors alone required about 192 MiB before payloads; `BitWriter`, entropy
merge, and final frame copies then used infallible `Vec` growth. A failed large
allocation could therefore abort a service process instead of returning the
existing typed encoder error.

Closure is complete in the host tree:

- the shared `DEFAULT_MAX_HOST_ALLOCATION_BYTES` is the single encoded-frame
  policy limit; a conservative MCU/block/restart bound is checked before
  sample-length validation, plane conversion, DCT work, or GPU allocation
- `CappedBytes` performs checked geometric growth with `try_reserve_exact`;
  allocation failure maps to `JpegEncodeError::HostAllocationFailed`, while a
  logical cap breach maps to `MemoryCapExceeded`
- restart encoding uses at most 64 ordered Rayon chunks, resets DC prediction
  at every restart segment, writes restart markers without entropy stuffing,
  and preserves byte identity with the one-chunk serial reference
- CPU pixel encode, DCT-domain re-emission, both frame assemblers, per-tile GPU
  planning, and combined GPU entropy planning share the same bound
- entropy orchestration is split from the 870-line encoder owner into a
  ratcheted 413-line module; the reusable capped-byte owner is 138 lines

Evidence on 2026-07-10: the all-feature `j2k-jpeg` package suite passed (244
library tests plus every integration and doc-test target before the two final
GPU-plan cap regressions), followed by the final 246/246 library suite; strict all-feature/all-
target and no-default-feature Clippy passed with warnings denied; no-default
library check passed; and all-feature `j2k-jpeg-cuda` plus `j2k-jpeg-metal`
downstream checks passed on macOS. Final combined-tree gates remain part of
FINAL-001, but there is no hardware-dependent ALLOC-004 behavior left open.

A later whole-lifecycle red-team invalidated the final sentence above for the
CPU and resident-GPU orchestrators. CPU RGB encode retains three converted
component planes while allocating entropy and then a copied final frame;
grayscale and DCT-domain re-emission still retain entropy while reserving the
second frame. Even the resident-GPU single-tile route retains the
adapter-returned entropy vector while frame assembly reserves and copies a
second near-cap frame; each allocation is capped, but their sum is not. Each
contiguous same-source batch group also has a combined entropy cap, while
`encode_jpeg_baseline_gpu_batch` retains every prior `EncodedJpeg` before
planning the next group. Within one group, all returned entropy chunks remain
live while frame assembly copies those payloads into an accumulating output
vector. Multiple individually capped groups, or one group near the cap, can
therefore exceed the aggregate live contract. The outer encoded-result vector
and temporary GPU-tile collection also use infallible caller-length allocation.
`CappedBytes` also checks the requested logical target but not the allocator's
returned capacity, and geometric growth can transiently retain the old and new
contiguous buffers at once. Planned entropy/frame writers should reserve their
known phase capacity up front (or reject an unprovable growth peak), recheck
actual capacity, and include that owner in the surrounding phase budget.
ALLOC-004 is reopened until single and batch high-water are preflighted across
retained outputs/current chunks/current copies, all derived metadata is
fallible, and ordinary grouping/byte-identity behavior remains covered.

The private resident-adapter sublane is now source-focused-green. Metal no
longer copies the complete entropy buffer before per-tile output; CUDA and
Metal preflight parameter conversion, validation, status, outer-result, and
payload phases while excluding device allocations from the host-only cap.
Root review additionally required post-reserve checks of allocator-returned
parameter/status/result/chunk capacities and preserved both CUDA cap excess
and allocator failure as nested typed runtime sources. Nineteen focused CUDA
runtime validation/allocation tests, 13 CUDA-adapter library tests, and four
Metal allocation tests pass after those corrections; the runtime all-feature
library check is green. Strict combined adapter Clippy, full post-correction
Metal/runtime suites, and exact-current-tree NVIDIA execution remain. This
does not close the still-open CPU/frame/group ownership described above.

Public-owner follow-up (2026-07-11): `EncodedJpeg` no longer derives `Clone`.
Its codestream can approach the 512 MiB frame cap, and the exhaustive workspace
audit found no production clone consumer. Batch and downstream adapters already
move results or borrow them, so inventing an unused fallible duplicate API
would add surface without an owner. This closes accidental infallible result
duplication only; it does not close the CPU/frame/group high-water work above.

ALLOC-004 closure (2026-07-11): the reopened whole-lifecycle paths now use one
actual-capacity host contract. CPU grayscale/RGB and DCT re-emission preflight
borrowed input, converted planes, entropy workspace/output, and the exact frame
copy together; every plane and output reserve is fallible and the allocator's
returned capacity is reconciled before the next owner. `CappedBytes` checks both
old-plus-replacement transient growth and the returned allocation. GPU single,
same-source group, and multi-group batch orchestration account parameter/status/
result outer capacities, every entropy chunk, current frame assembly, and all
prior retained `EncodedJpeg` capacities; malformed adapter outer capacity is
rejected before frame copy.

The complete all-feature `j2k-jpeg` package is green with 388/388 library tests
plus every integration and doc-test target, including exact CPU byte goldens,
restart fanout, allocator-capacity, retained-group, and adapter-capacity
regressions. Strict all-target/all-feature Clippy passes with `-D warnings`, the
no-default-feature check passes, and the combined JPEG encode output allocation
policy passes its CPU, DCT, frame, entropy, GPU group/batch, move-only, and
structural checks. ALLOC-004 is complete; final accelerator hardware repetition
remains in FINAL-001 rather than reopening the host ownership design.

### ALLOC-005 / CUDAERR-002 — bounded CUDA coefficient transcode and complete diagnostics

The 2026-07-10 equivalent-allocation/error scan found that
`j2k-transcode-cuda` trusted caller block grids before staging, allocated and
copied several geometry-derived host vectors independently, widened device
readback from `f32` to `f64` without an aggregate phase budget, built nested
resident output metadata through infallible `vec!`/`collect`, and flattened
runtime failures to static labels. Legal-looking batch geometry could therefore
amplify host memory well beyond the shared ceiling, while a backend failure
lost the operation and driver/kernel diagnostic needed to distinguish
unavailability from execution failure.

Working-tree closure is implemented:

- block-grid products and slice lengths are validated before staging, with
  checked arithmetic and typed cap errors on overflow
- f64-to-f32 and i16 DCT staging, f32-to-f64 readback widening, quantized band
  reslicing, grouped outputs, resident jobs/shapes, and nested component,
  resolution, subband, and code-block metadata all reserve fallibly
- phase-wide preflights include simultaneously live input staging, readback,
  widening, source/destination metadata, and all four resident subband plans;
  staging vectors are explicitly dropped before the next allocation-heavy
  phase
- grouped output assembly writes into one fallibly reserved result vector,
  validates returned group indices/duplicates/missing groups, and no longer
  performs a second infallible caller-sized collection
- `CudaTranscodeError` exposes typed host-cap/allocation variants and a
  feature-independent `CudaRuntimeFailure` carrying operation, full rendered
  backend detail, and unavailability classification; Auto mode recovers only
  from genuine unavailability/unsupported jobs, while allocation and execution
  failures propagate
- allocation failures retain `TranscodeStageError::MemoryCapExceeded` or
  `HostAllocationFailed` classification instead of becoming backend strings
- component conversion, grouped resident dispatch, and resident encoding are
  split into focused modules: the resident facade is 84 lines and its
  planning/output/orchestration children are 378/361/356 lines, all below the
  existing 425-line child ceiling without raising a ratchet

Focused compile evidence on 2026-07-10: the combined tree passes
`cargo check -p j2k-transcode-cuda --features cuda-runtime --lib`; the focused
feature-enabled library suite passes 18/18, including no-GPU aggregate and
runtime-classification regressions. Strict feature/no-feature Clippy,
repository policy, and exact-source NVIDIA parity are still required before
these dashboard rows can move to complete.

Post-checkpoint red-team reopening (2026-07-11): both
`j2k-transcode-cuda::cuda::allocation::try_transcode_vec_with_capacity` and
the corresponding Metal transcode helper validate requested element bytes and
then delegate to `Vec::try_reserve_exact` through `j2k_core`. Neither helper
reconciles `Vec::capacity()` after the allocator returns. The same omission
flows into CUDA staging/output metadata and Metal staging plus dense/sparse
weight owners. `try_reserve_exact` is not an exact-capacity guarantee, so a
request at the logical boundary can still exceed the shared cap or invalidate
a cumulative phase model through allocator overcapacity. This reopened
ALLOC-005 and METALALLOC-001 at that checkpoint. Closure required every
returned capacity to feed the owning phase ledger, including growth of an
existing vector, plus allocator-model exact/one-over tests before the frozen
CUDA/Metal runs; the later source-closure entries record that implementation.

CUDA host-owner checkpoint (2026-07-11): the adapter packetization tag tree no
longer uses infallible level/node vectors or a heap path stack. Its six backing
arrays reserve/initialize through typed fallible helpers, reconcile aggregate
allocator capacities, and retain the fixed 16-level/2,048-node kernel bounds;
allocator-failure and exact/one-over capacity tests are source-ratcheted. The
tag-tree, subband state, aggregate state, flattened packet plan, direct decode
plan, runtime host-band downloads, and JPEG diagnostic report are move-only
after an exhaustive production-clone search found construction/move/borrow
use only. At that checkpoint, generic CUDA transcode allocation wrappers, the
buffer-pool high-water issue, and exact NVIDIA evidence still kept ALLOC-005
open; the closure below supersedes that source-open status.

CUDA transcode actual-capacity closure (2026-07-11): its `HostPhaseBudget`
reconciles each returned capacity, including growth of existing vectors. The
f32 staging phase begins with every already-live input owner; runtime readback
adds its bands to that same live graph; staging is dropped before widening; and
the widening phase re-accounts runtime bands plus completed outputs. Uniform
batch/code-block paths follow the same ownership model, and reversible batches
carry prior outputs into each following phase. Runtime transcode readback uses
one aggregate actual-capacity budget rather than independent vector caps.
Focused policy tests enforce the owner handoffs, allocation-before-launch
ordering, exact boundaries, and module-size ratchets.

Local evidence is 22/22 all-feature and 10/10 no-default-feature transcode
library tests plus every applicable integration/doc target. Warning-denied
all-target Clippy passes for both configurations when scoped to the transcode
crate; its combined dependency lane is temporarily blocked by concurrent
JPEGCACHE-001 source changes in `j2k-jpeg`, not by this workstream. The runtime
dependency itself passes 255/255 tests and strict all-target/all-feature
Clippy. ALLOC-005 and CUDAERR-002 remain in progress only for the final
combined-tree gate and frozen-source NVIDIA execution.

### ALLOC-006 — deep native tile and tile-part metadata amplification

The 2026-07-10 pattern-equivalent native review found that the structural tile
preflight counted `ComponentInfo` and `ProgressionChange` only at their shallow
sizes, while `Tile::new` deep-cloned each component's quantization step-size and
precinct-exponent vectors for every tile through an infallible `collect`. A
small, Part-1-bounded header with many components and up to 65,536 addressable
tiles could pass that estimate and amplify into gigabytes of owned metadata
before entropy decode. Tile-part parsing also retained untrusted PPT/PLT/PLM
header and packet-length vectors through infallible push/collect/clone paths.
This is a P1 service-availability issue and a pattern-equivalent remainder of
ALLOC-003.

Source remediation is complete in the working tree. `TileMetadataBudget`
retains the cheap whole-graph logical preflight, then routes every outer tile,
inherited component/progression vector, COD/COC/QCD/QCC replacement, POC
extension, PPT/PLT temporary, flattened packet-length vector, separated header
reader, and retained tile part through one actual-capacity ledger. Growth
preflights the old-plus-new transient peak, reconciles `Vec::capacity()`
immediately after the fallible reserve, and keeps the final owner and ledger in
sync even when reserve or post-reserve validation fails. A scoped transaction
rolls temporary marker claims back on every early return, and successful parse
handoff requires exact equality with a fresh walk of the complete retained
tile graph.

The same review exposed and fixed the package-level PPM regression: main-header
packed headers are emitted per packet, so indexing them by tile-part number
reused the first packet header in the second image tile. Checked progression
packet counts and a global PPM cursor now assign the correct header readers.
Pure exact-cap, one-over, allocator-overcapacity, reserve-failure, replacement,
logical-amplification, and owner-graph tests pass; the malformed-PPT rollback
test and PLT/PLM/PPM/PPT multi-tile regressions pass. The serialized full native
package is green (508 library passes, one established ignore, all integration
and doc tests passing), as are strict all-feature/no-deps native Clippy and all
32 focused tile/container/native-decode/batch/handoff/scratch/error policy
checks. No cap or suppression ratchet was raised. The full repository-policy
aggregate remains globally red on unrelated concurrent API snapshot, module
inventory, and line-ratchet work; none of its remaining failures names the
ALLOC-006 source or tests.

### ALLOC-007 — bounded CPU JPEG-to-HTJ2K coefficient/reference workspace

The 2026-07-10 CPU transcode review confirmed that a tiny legal JPEG header
could advertise up to 65,535 x 65,535 samples and reach full-grid DCT extraction
before entropy failure. The path built separate quantized/dequantized planes
with infallible geometry-derived allocation (roughly 8.6 GiB each for one
maximal grayscale grid), while integer/float reference paths further amplified
coefficients into full `i64`/`f64` storage. Reversible Rayon batches and grouped
prepare/report vectors also used independent infallible collection. This is a
P1 service-availability issue reachable through untrusted JPEG input.

Host remediation is implemented and under final verification: the transcode
workspace is checked before DCT extraction; dequantized-only extraction avoids
the unused duplicate plane; typed cap/allocation errors flow through the stage
contract; integer/float reference storage, rounded validation outputs, Rayon
reversible results, and batch prepare/encode/report growth reserve fallibly;
group metadata uses an aggregate budget. Public 5/3 and 9/7 transforms now
separate truthful grid validation from execution failures through
`DctTransformError`, reject aggregate simultaneously-live scratch/output bytes
before reserving, and release geometry-mismatched reusable scratch. The former
5/3 sparse-row builder's input-driven O(N^2) unit-basis loop is replaced by an
allocation-free shared codec-math primitive that derives at most five taps per
row in constant work; a 65,535-axis regression covers the legal JPEG maximum,
and Metal consumes the same primitive. Progressive extraction preflights the
i32 accumulator, component metadata, and retained i16 planes together, uses
fallible allocation, and now honors `DctExtractOptions::dequantized_only()`.
The reference/output/group helpers live in focused 220/65/32-line modules,
with their former roots restored below existing caps; component geometry
grouping is a separate 49-line fallible owner. Verification currently includes
248/248 `j2k-jpeg` library tests, warning-clean all-target `j2k-jpeg` Clippy,
11/11 codec-math unit tests, warning-clean codec-math all-target Clippy, and an
all-feature `j2k-transcode` library check. The serialized transcode package,
repository policies, and combined final matrix remain before ALLOC-007 can move
to complete.

A post-handoff red-team then found that public baseline/sequential
`extract_dct_blocks` still created caller-sized quantized/dequantized planes
with infallible nested `vec!`/`collect` inside entropy decode. The outer
JPEG-to-HTJ2K preflight protects that caller but not direct public extraction.
ALLOC-007 product source is reopened until sequential extraction performs its
own aggregate plane/metadata preflight and fallible allocation, with a direct
API regression and policy ratchet. Whole-lifecycle review also found that the
returned component-metadata vector was still infallible and omitted from the
aggregate, while restart-index construction used unbounded `Vec::push` after
all coefficient planes were resident. Closure must cover those retained owners,
not stop at the entropy decoder boundary.

Those sequential and fixed batch-metadata fixes are now source-complete and
focused-green. A broader output-lifecycle pass found a further ALLOC-007
dependency: `validate_jpeg_transcode_workspace` models extraction and transform
peaks but not codestream payload capacity. During single encode, retained JPEG
DCT planes, precomputed wavelet coefficients/validation results, and the new
codestream overlap. Batch routes additionally retain prior codestreams while
encoding the next tile, and the fixed-metadata model does not charge any output
payload. Exact closure therefore depends on ALLOC-013's bounded native encoder
output plan; the transcode validator must then add single and batch codestream
high-water before ALLOC-007 can close.

The current composition audit made that dependency concrete. Native encode
knows its precomputed coefficient input but not the simultaneously retained
JPEG bytes, decoder scratch, validation owners, report metadata, or prior batch
codestreams. Transcode must measure those actual capacities immediately before
each native phase and pass only the remaining global budget through a native
session cap that can lower, never raise, the 512 MiB ceiling. Drops and moves
must update the external baseline before the next phase. Do not expose a
forgeable retained-byte token or duplicate native's internal ownership model.

The same audit found optional `error_metrics_i32` constructs a public
`BTreeMap<i64, usize>` through infallible per-node allocation while both full
coefficient inputs remain live. `BTreeMap` offers no fallible reservation or
inspectable allocation capacity, so the validation feature can bypass the
contract even after encode composition is fixed. Replace it in 0.7 with a
move-only sorted Vec-backed histogram built by one checked, fallible reserve,
allocator-capacity reconciliation, in-place sort, and in-place coalescing.
The metrics error must distinguish length mismatch, cap excess, and allocator
failure and remain a typed source in `JpegToHtj2kError`. Exact/one-over,
allocator-overcapacity, all-unique-bucket, lookup/iteration, and accumulated
batch-output regressions are required.

Source checkpoint (2026-07-11): native precomputed 5/3, borrowed/owned 9/7,
preencoded, prequantized, compact, and owned-batch adapters now accept a
doc-hidden cap that is clamped to the process ceiling. CPU transcode measures
allocator capacities for reusable DCT scratch, extracted JPEG coefficient and
restart owners, component reports, validation owners, native inputs, generated
codestreams, Rayon preparation collections, result-slot conversion, and prior
batch outputs. It drops JPEG coefficient graphs before native encode, passes
only the remaining cap, and serializes CPU tile encodes so parallel jobs cannot
each claim an independent 512 MiB allowance. The former `BTreeMap` histogram
is a move-only sorted `Vec<ErrorHistogramBucket>` reserved fallibly once and
coalesced in place; `MetricsError` preserves length, cap, and allocator
categories as an error source. Exact/one-over, allocator-capacity,
all-unique-bucket, lowered-native-cap, and accumulated-output source tests and
policy ratchets are present. ALLOC-007 remains in progress until the serialized
transcode/native/JPEG/CUDA-host tests, strict Clippy, repository policies, and
combined final matrix are green; this checkpoint is not a completion claim.

ALLOC-007 closure (2026-07-11): the dependency chain is now closed. ALLOC-004
provides the aggregate JPEG extraction/entropy/frame contract and ALLOC-013
provides lowered-cap, actual-capacity native encode sessions through final
codestream ownership. The complete all-feature native package passes 598 tests
with one established diagnostic ignore, `j2k-jpeg` passes 388/388 library tests
plus every integration/doc target, and `j2k-transcode` passes 132/132 across all
targets. Strict all-target/all-feature Clippy passes for all three packages.
All 18 JPEG transcode allocation policies pass, covering transform geometry,
progressive extraction, grouped/reference owners, lowered native caps, retained
batch outputs, Vec-backed metrics, typed error sources, and focused module
ratchets. ALLOC-007 is complete; immutable-candidate repetition remains under
FINAL-001.

### ALLOC-008 — aggregate JPEG fast-packet ownership

Tracing the sequential restart-index allocation exposed a pattern-equivalent
P1 in the backend-neutral fast-packet builder. Entropy bytes and restart offsets
are each independently allowed to consume the full 512 MiB host cap. Color
packet construction then builds a `Vec<DeviceCheckpoint>` and a second
`Vec<JpegEntropyCheckpointV1>` while both packet vectors remain live; during
conversion both checkpoint arrays are simultaneously resident. Individual
fallible reserves therefore do not enforce the codec's aggregate live-memory
contract, and several reserve failures are currently reported as cap breaches
instead of the existing typed `HostAllocationFailed` category.

The generic checkpoint planner has the same phase-model defect outside fast
packets: it reserves `Vec<DeviceCheckpoint>` before a missing-EOI scan may be
copied into an owned, terminated reader buffer. `build_device_plan` can retain
that scan copy and the checkpoint vector together. Fast-packet extraction itself
requires EOI, so its reader remains borrowed, but the shared helper and device
plan still require the checkpoint-plus-terminated-scan boundary.

Required closure is one pre-allocation phase model covering destuffed entropy,
restart offsets, device checkpoints, packet checkpoints, and simultaneous
conversion metadata, plus the generic checkpoint/owned-reader transient.
Derived capacities must use checked arithmetic and exact fallible reservation,
direct grayscale/color/device-plan builders must reject the boundary before
large growth, and allocation failure must preserve its typed category. Focused
regressions and repository policy must prove aggregate entropy-plus-offset,
entropy-plus-both-checkpoint, and checkpoint-plus-reader rejection as well as
ordinary packet/checkpoint behavior before ALLOC-008 can close.

Working-tree correction (2026-07-10): fast-packet construction now consumes the
decoder's single prepared header instead of calling `parse_header` again.
Entropy and restart vectors reserve fallibly through one live-byte sequence,
and each allocator-returned capacity becomes the baseline for the next owner.
Color materialization includes retained decoder metadata, entropy, restart
offsets, terminated-scan storage, device checkpoints, and public packet
checkpoints in one aggregate; grayscale uses the corresponding smaller phase.
The generic lazy CPU checkpoint cache obtains the decoder baseline before
locking, preflights old-plus-replacement growth, reconciles actual capacity,
and clears the attempted cache growth when the final aggregate postcheck fails.
Exact/one-over tests cover entropy-plus-offset, decoder-plus-packet owners,
checkpoint conversion, and cache replacement. Fast-packet source ratchets
forbid the duplicate parse and private owner cloning. A public-owner follow-up
removes `Clone` from `JpegFast{420,422,444}PacketV1` and `JpegGrayPacketV1`:
entropy, restart, and checkpoint vectors can collectively approach the same
512 MiB budget. CUDA's supported reusable cache shares these move-only packets
through `Arc`; its manual cache-entry clone deliberately clones only the
`Arc`s and does not add a `T: Clone` bound. Metal's equivalent Arc migration
belongs to ALLOC-018. The follow-up owner sweep also makes
`DeviceDecodePlan<'a>` move-only: its owned `Cow` scan, warnings, components,
and checkpoint graph must not expose an infallible deep copy. Small fixed/POD
device component and batch-summary metadata remains `Clone`/`Copy`. ALLOC-008
remains in progress until the combined
CPU/DCT/batch/transcode high-water audit and gates below are complete.

ALLOC-008 closure (2026-07-11): the dependent CPU/DCT/batch/transcode high-
water work is complete under ALLOC-004 and ALLOC-007. The full all-feature JPEG
package is green, including 70 device-plan tests and 12 fast-packet integration
tests, with strict all-target/all-feature Clippy and the no-default-feature
check passing. The focused high-level JPEG allocation policy confirms one
prepared parse, fallible actual-capacity entropy/restart/checkpoint growth,
terminated-reader and decoder baselines, transactional checkpoint-cache
replacement, typed cap-versus-allocator failures, move-only packet/plan owners,
and focused production/test modules. ALLOC-008 is complete; CUDA/Metal retained
cache byte limits remain separately tracked by JPEGCACHE-001.

### ALLOC-009 — bounded JPEG header and progressive-plan metadata

The final pattern-equivalent JPEG pass found two small-input amplification
paths before coefficient allocation. `parse_header` appends one `Warning` for
each relevant untrusted APP marker through infallible `Vec::push`. Progressive
collection also appends one `ParsedProgressiveScan` per SOS through an
infallible vector. Each scan retains full inline Huffman and quantization table
snapshots plus its own heap-allocated component vector, so a short repeated scan
script amplifies far beyond its input bytes and can abort before the existing
progressive coefficient cap is reached.

Decoder construction then builds a second `PreparedProgressivePlan` while the
entire parsed scan script remains live. Its scan vector and one component vector
per scan are also infallible, and compiled table `Arc`s are retained in the new
plan during conversion. This is a distinct P1 metadata/lifecycle issue; the
ALLOC-007 coefficient-plane preflight does not protect it.

Closure must cap warnings, parsed scan snapshots, component metadata, and the
simultaneously live prepared representation under one typed host-allocation
contract. Repeated per-scan tiny heap owners should be replaced by fixed-capacity
or flattened storage appropriate to JPEG's at-most-four frame components.
Reserve failure must report `HostAllocationFailed`, cap/overflow must remain
`MemoryCapExceeded`, and adversarial repeated-APP/repeated-SOS tests plus source
ratchets must prove rejection before unbounded growth without changing valid
progressive decode behavior.

The fast color-packet path also exposes an ownership symptom of this issue: it
constructs `Decoder::new(bytes)` and separately calls `parse_header(bytes)`,
retaining two parsed/prepared metadata graphs before entropy/checkpoint
materialization. ALLOC-008 and ALLOC-009 must meet at one parse/planning handoff
or an explicitly budgeted consuming conversion; two independent 512 MiB
claims are not acceptable. The final policy should forbid the duplicate parse
and private heap-owner `Clone` derives in addition to the raw allocation
tokens.

Partial source work has already made warning/scan growth fallible, stores each
scan's at-most-four component selectors inline, and preflights the logical
parsed-plus-prepared peak. The remaining defect is that prepared components,
flattened scan components, scan descriptors, and compiled table owners are then
allocated through individually capped helpers; allocator-returned capacities
do not feed one running conversion budget before the next owner is created.
Treat the earlier logical preflight as necessary but insufficient. Add actual-
capacity exact/one-over conversion tests and preserve cap versus allocator
failure categories. `parse/header.rs` is also still a 1,071-line mixed owner;
land the budget correction by extracting progressive-script collection and
metadata allocation/conversion responsibilities into real modules, with tests
separate from production, rather than extending the existing `too_many_lines`
state machine.

The stable-Rust allocation audit makes the table-owner redesign mandatory:
Rust 1.96 still gates fallible `Arc`/`Box` construction behind unstable
allocator APIs, so the current `Arc::new` compiled/quant tables and boxed
4,096-entry Huffman arrays cannot honestly return `HostAllocationFailed`.
The accepted implementation versions each DHT/DQT definition once in fallible
raw arenas; each progressive SOS stores compact active slot-to-ID snapshots
instead of cloning complete tables. Parsed frame component metadata becomes
inline. A consuming conversion compiles each referenced raw Huffman version
once into one fallibly reserved arena of fully inline table values, stores
checked prepared IDs in sequential/progressive descriptors, releases parsed
owners phase by phase, and records actual retained bytes on the decoder.
Context hits copy compiled inline values into the already budgeted plan arena;
this intentionally gives up zero-copy cross-decoder `Arc` sharing in favor of a
safe stable typed-allocation contract. Benchmark the context-hit/fast-4:2:0
copy cost before release. Parser growth must count old plus replacement peaks,
and cache-hit as well as cache-miss conversion must use the same ledger.

Source checkpoint (2026-07-10): Stage 1 is implemented as focused header,
marker, progressive-script, validation, walker, allocation-ledger, and raw
table-version modules. Warnings, scans, DHT versions, and DQT versions grow
through one actual-capacity ledger; frame component metadata and SOS table
snapshots are inline. Stage 2 replaces boxed/`Arc` decode tables with fully
inline compiled values in one fallibly reserved arena per active prepared plan.
Sequential and progressive descriptors carry checked IDs, quant tables are
inline, hot 4:2:0/4:4:4 paths resolve table references before their MCU loops,
and `prepare_header` consumes the parsed owner before returning decoder
metadata. The exact decoder ledger method
`retained_allocation_bytes_excluding_cpu_checkpoint_cache` includes context
coexistence, warning capacity, all prepared-vector capacities, and compiled
arenas while deliberately excluding the existing CPU checkpoint cache; the
checkpoint growth path obtains this baseline before taking its mutex. Source
ratchets cover both stages. The former 959-line progressive entropy owner is
now a 53-line facade over explicit model, allocation, scan, render, and test
modules, each with its own low line-count ratchet; the split adds no lint
suppression or threshold increase. Warning-free compile, strict Clippy, and
behavior parity are satisfied by the final reconciliation below. Comparative
performance measurement remains tracked by PERF-001 rather than hidden as an
allocation-contract residual.

Prepared-construction correction (2026-07-10): one ledger now begins at
`parsed_retained + actual_context_retained`. For each prepared vector it checks
the next requested replacement against the current live total, reserves
fallibly, and advances the total by the allocator-reported capacity before any
later reserve. Budgeted Huffman-cache growth reconciles
`live - old_context + new_context`; a decode-plan cache operation rebases to
`parsed_retained + context_after_cache + returned_plan_retained`. Progressive
components, flattened scan components, scan descriptors, and the companion host
plan therefore share one sequence. The final aggregate prepared-metadata check
remains defense in depth. Pure exact/one-over tests force a vector whose capacity
exceeds its length, and a 230-line ratchet covers the focused construction
ledger (currently 215 lines; `decoder/plan.rs` is 697 lines under its unchanged
800-line cap). Source is frozen for focused compile/policy verification; no new
lint suppression or threshold increase was introduced.

Progressive DQT correction (2026-07-10): a focused heap-free planner now binds
each frame component to the quantization-table values resolved from that
component's first scan snapshot. Every later scan containing the component must
resolve equal values or preparation returns
`ProgressiveQuantTableChanged { offset, component, table_id }`; byte-identical
new versions remain valid. A different definition before a component first
appears, a redefinition after its last appearance, and changes to unused slots
remain legal. Both the progressive plan and its companion host plan copy the
latched values rather than terminal active DQT slots. Behavior tests cover all
of those cases and preserve the offending scan entropy offset. The focused
latch/test owners are 77/114 lines under new 90/170-line ratchets, while the
simplified `decoder/plan.rs` is now 697 lines under its unchanged 800-line cap.

Progressive script correction (2026-07-10): a heap-free
`[[u8; 64]; 4]` state machine validates every selected component/coefficient
across SOS segments. Initial scans may initialize a coefficient only once;
refinement requires the previous Al to equal the new Ah and advances to the new
Al. Validation is two-pass so a rejected scan cannot partially mutate state.
EOI requires an initial DC scan for every frame component without inventing a
requirement that every AC coefficient appear. Typed errors retain SOS/EOI
offset, marker, component, coefficient, and duplicate/before-initial/skipped
state. Focused tests cover valid DC/partial-AC refinement, duplicate and
overlapping initial scans, coefficient-63 refinement boundaries, skipped
levels, and missing component DC. The collector facade remains 223/225 lines;
script and tests are 151/141 under 165/155-line ceilings. Physical EOF/EOI,
marker-fill, residual-EOB, invalid-padding, excess-entropy, and parser/decoder
terminal mismatch validation now live in a focused terminal owner with exact-
offset behavior and structure regressions.

Final reconciliation (2026-07-11): the existing parse/prepared policies were
green, but the emitted warning owner still allocated independently after the
workspace planner assumed its capacity. A focused `warning_ownership` module
now carries parsed-warning capacity, the at-most-one scan-warning owner, and
the merged public result through one actual-capacity ledger. Decode workspace
and batch retained-result planning import those same formulas, and exact/
one-over tests prevent allocator spare capacity from exceeding the peak. The
header and prepared-table policy groups (2 and 5 tests), warning/decode policy
group (5 tests), 388 all-feature library tests, every package integration/doc
test, and warning-denied all-target/all-feature Clippy pass in the isolated
target. Existing fast-path/profile behavior tests pass; no comparative
Criterion run was performed in this bounded correctness lane.

### ALLOC-010 — fallible JPEG output and decode scratch

The final JPEG allocation sweep found that several paths correctly reject
logical sizes above the 512 MiB ceiling but still perform the accepted
caller-derived allocation with infallible `vec!` or `Vec::resize`. A cap-valid
request under memory pressure can therefore abort the process instead of
returning the newly established `JpegError::HostAllocationFailed` contract.
Affected owners include codec-owned output allocation, `JpegOutputBuffer`, all
sequential `ScratchPool` stripe/upsample/sink rows, progressive component-image
and output-row storage, sampled lossless full-frame planes, and extended-12
component planes.

The reusable scratch pool also grows monotonically. Its byte counter exists,
but preparation neither reserves fallibly nor reconciles previously retained
capacity with the next decoder's metadata/scratch budget. Batch sessions can
multiply that retained state across workers; their cross-worker problem is
tracked separately.

Closure requires fallible reserve-before-resize helpers with checked aggregate
phase formulas, typed cap versus allocator errors, and mutation only after all
required reservations succeed where partial state would be observable. The
public reusable output error type must be able to distinguish an over-cap shape
from allocator failure. Progressive/lossless/extended paths must count all
simultaneously live planes and row scratch together with retained plan metadata,
and reused pools must release or budget stale capacity before growth. Focused
cap-boundary, allocator-category, reuse/shrink, and ordinary decode parity tests
plus source policies are required.

Working-tree closure now uses typed, fallible host allocation for every listed
owner and removes the infallible `Clone` contract from `JpegOutputBuffer`.
Output growth clears the previous allocation before reserving a larger one, and
the reusable scratch pool releases disposable owners before growth, checks the
allocator-returned capacities after every reserve, and feeds those actual bytes
into the next live-budget decision. Progressive 12-bit planning counts both the
coefficient storage and simultaneous `u16` render planes. Verification passed
35 allocation units, 6 scratch units, 9 output-buffer units, 5 reuse tests, 114
decode-into tests, 14 view/row tests, 4 source-policy tests, the all-target/
all-feature package check, and all-target/all-feature strict Clippy. The stable
API snapshot still must record `BufferError::HostAllocationFailed` and removal
of `JpegOutputBuffer: Clone`; exact frozen-tree and CUDA evidence remain under
FINAL-001.

Independent review after those gates found a narrower residual: several
multi-vector owners checked aggregate allocator-returned capacities only after
all logical allocations had succeeded. The correction must feed each actual
capacity into a running phase budget before allocating the next coefficient,
image, row, sampled, or extended-12 plane, and the shared allocation helpers
must postcheck their own returned capacities. The evidence above is retained as
the last green checkpoint, not a current closure claim; focused and strict gates
must rerun after the correction.

Current correction checkpoint (2026-07-10): progressive/lossless/extended-12
planes, sequential DCT storage, CPU entropy/frame owners, restart writers, GPU
batch metadata/results, and prior retained frames now feed allocator-returned
capacities into the next live-budget decision. Sequential decode is a 65-line
facade over focused plan/scratch/stripe/DCT owners; extended-12 plane allocation
is separate from decode/render orchestration; resident GPU single-tile and batch
orchestration are separate 89/166-line owners under unchanged 190-line caps.
The all-feature/all-target JPEG check and all 331 library tests pass, including
the exact aggregate and architecture ratchets. The five focused decode
allocation policies pass. The GPU policy run exposed a formatting-sensitive
matcher and the separately open ALLOC-008 checkpoint owner above its existing
line cap; those are policy/next-lane work, not permission to raise a threshold.
Final reconciliation (2026-07-11): ALLOC-009's overlapping prepared and
warning owners are frozen, the five decode-allocation policies pass, exact DCT
re-emission bytes and coefficient parity remain unchanged, and the full
all-feature package plus warning-denied all-target/all-feature Clippy pass.
No multi-vector decode owner discovered in the bounded rescan allocates its
next caller-sized vector before reconciling the preceding actual capacity.

### ALLOC-011 — aggregate JPEG batch-session ownership

`JpegBatchSession` currently allocates per-chunk indexed results, Rayon
collections, explicit thread handles, ordered result slots, final outputs, and
worker slots through infallible caller-length vectors. The session's worker
count is bounded by the job count but an explicit worker option can still make
that count caller-driven. Each `WorkerSlot` then retains an independent
`DecoderContext` and monotonically grown `ScratchPool`; a per-decode 512 MiB
check therefore permits the session or one concurrent call to retain many
multiples of the intended codec-owned ceiling.

Most public batch methods can return `TileBatchError`, but
`decode_prepared_jpeg_tiles_rgb8` returns a bare per-tile result vector and has
no representation for failure to allocate the outer result itself. This API
must become honestly fallible or adopt another explicit, typed batch-level
failure contract; allocating an equally large vector of per-tile errors is not
a valid recovery.

Closure requires a fallible scheduling/result collector, checked worker-slot
growth, and an operation/session budget that covers every concurrently active
and retained context/scratch pool plus result/handle metadata. Scheduling may
reduce concurrency to stay inside the cap, but must preserve input-order error
selection and caller-requested worker semantics where feasible. Stale worker
capacity must be released or evicted before a new call when required. Add
boundary tests that use pure byte formulas rather than huge allocation,
multi-worker retained-cap tests, typed outer-allocation tests, and parity for
Rayon and explicit scoped-worker modes.

Working-tree closure splits the session into focused planning, allocation,
runtime, scheduler, worker, and collection modules. The corrected two-domain
contract keeps the authoritative JPEG codec allowance at 512 MiB and gives all
batch-owned metadata one collective 64 MiB allowance, for a checked 576 MiB
maximum. Plans, worker slots, retained summaries, ordered worker slots, final
results, scoped handles, and deep warning owners all share that single metadata
allowance rather than receiving independent caps. Before the plan vector is
allocated or any tile is parsed, stale scratch and all secondary contexts are
evicted; at most one retained context participates inside the planning
decoder's authoritative codec claim. Allocator-returned plan/summary/result/
handle/output capacities are reconciled immediately.

Execution separately caps the aggregate active/retained worker codec claims and
reduces concurrency until both domains fit. Each successful tile's actual
warning-vector capacity is checked against its per-job planned metadata claim
before the outcome can enter an ordered result slot. The scheduler writes
disjoint result slices directly and converts spawn, panic, poison, missing-slot,
capacity, and result-integrity failures into typed infrastructure errors rather
than a fake tile-zero failure. Prepared batches expose a typed outer `Result`;
the shared core dual-limit collector defensively validates both collection and
aggregate ownership while preserving the first codec error in input order.

Final-source verification passes the 3 focused batch-allocation policies, all
44 JPEG batch integration/parity tests, the full all-feature JPEG package, and
warning-denied all-target/all-feature JPEG Clippy. Successful warning owners
are checked against the same shared result-capacity formula before entering
ordered slots. Stable API/semver snapshot regeneration remains under SEM-001
and frozen-candidate evidence under FINAL-001; neither is an unresolved batch
allocation path. J2K adoption of the shared collector is closed by ALLOC-014.

### ALLOC-012 — bounded JPEG segment rewrite and TIFF assembly

The public marker utilities contain a separate caller-byte ownership gap.
`rewrite_sof_dimensions` clones the complete input with infallible `to_vec`.
Abbreviated TIFF/WSI tile preparation grows its output infallibly, while
`collect_normalized_segments` first clones every retained table segment into a
separate vector and `push_segment_dedup` copies it again into the final JPEG.
Large `JPEGTables` plus tile slices can therefore exceed the shared ceiling or
abort, and the temporary per-marker ownership roughly doubles the table
payload high-water before the output is complete.

Closure should keep normalized segment descriptors borrowed as ranges/slices,
preflight tables plus tile body plus SOI/EOI and metadata under one checked byte
formula, reserve the final output fallibly once, and return
`HostAllocationFailed` separately from cap/overflow. Dimension rewrite should
use the same capped fallible copy helper. Duplicate-table policy, offsets,
zero-dimension repair, restart validation, and borrowed-no-change behavior must
remain byte-for-byte compatible. Boundary/duplicate/repair tests and a source
ratchet must prove that no per-segment payload clone returns.

Working-tree closure now performs two allocation-free marker passes with a
fixed table-key set, retains accepted segment bytes as borrowed slices, checks
the exact SOI + unique tables + tile body + EOI length, and reserves the final
vector once through the typed JPEG allocation helper. Owned zero-dimension
repair mutates that vector in place instead of allocating a second full copy;
the public SOF rewrite uses one checked fallible copy. Final reconciliation
also removes infallible `Clone` from the potentially near-cap `PreparedJpeg`
owner and provides `PreparedJpeg::try_clone`: borrowed inputs remain borrowed,
while owned payloads use the same typed copy helper. Allocation tests were
split from the segment facade rather than raising its structural threshold.
The 5 segment policies, 34 `inspect` integration tests, exact canonical DCT
byte goldens, full all-feature package, and warning-denied all-target/
all-feature Clippy pass.

The pattern-equivalent retained-owner sweep also covers coefficient extraction
and restart metadata. `JpegDctImage`, `JpegDctComponent`, and `RestartIndex`
are move-only: the two coefficient planes and untrusted-count restart segment
vector are included in the actual-capacity DCT ledger and can approach the
512 MiB host cap. No production caller cloned these owners; transcode callers
borrow or move them, so shared consumers should use `Arc` rather than an
infallible deep duplicate. The DCT ownership policy preserves `Debug`,
`PartialEq`, and `Eq` while forbidding the large-owner `Clone` derives.

### ALLOC-013 — typed, phase-bounded native encode and codestream assembly

The pattern-equivalent scan of native encoding found that the decode-side live
budget work has no encode-side counterpart. Single/multi-tile and precomputed
5/3/9/7 paths, Tier-1 job/result vectors, rate-control layers, prepared packet
trees, packet header/body merges, tile-part lists, extracted PLT/PPM metadata,
tile bodies, and final codestream writers use widespread infallible capacity,
collection, and copy operations. Batch precomputed output also accumulates a
`Vec<Vec<u8>>` of codestreams without a shared payload limit.

Many public native encode functions still return `Result<_, &'static str>`, so
they cannot preserve arithmetic overflow, cap excess, allocator failure, or
backend failure as distinct typed categories. This is both availability debt
and an error-architecture blocker for JPEG-to-HTJ2K and facade callers.

Closure requires a public non-exhaustive native encode error with source-aware
backend diagnostics and typed size/cap/OOM variants; one checked phase plan per
encode family; fallible reserve-before-fill for all geometry-derived metadata,
Tier-1/packet payloads, tile bodies, and final bytes; and aggregate single,
multi-tile, batch, preencoded, and resident high-water accounting. Avoid merely
wrapping individual vectors at independent 512 MiB ceilings. Byte-exact CPU and
accelerator behavior, rate-control/layer semantics, PLT/PPM/tile-part output,
and existing passing tests must remain. This is a large lane and should be
split by planning/Tier-1, packetization, and codestream-finalization ownership
with structure ratchets rather than another god allocation helper.

Execution is divided into four reviewable ownership lanes; each lane must land
with pure boundary tests before the next layer treats its contract as trusted:

1. **Error and allocation foundation.** Add one public, non-exhaustive
   `EncodeError` that keeps invalid input, arithmetic overflow, shared-cap
   excess, allocator failure, codestream validation, and accelerator-stage
   failure distinct. Existing accelerator traits may continue to expose their
   static backend detail internally, but every public native/facade/transcode
   boundary must map it with the failed operation instead of flattening all
   failures to another string. Shared helpers perform checked byte arithmetic
   and fallible reserve/resize only; phase-specific formulas remain with their
   owners. The shared ledger must accept an explicit retained-input baseline
   for facade recode owners and precomputed coefficient images. Its allocation
   owner and capacity claim must be inseparable, and its final-handoff state
   transition must be race-free: a claim that observed an unsealed state may
   not commit after another thread seals the ledger.
2. **Input, transform, and Tier-1 ownership.** Single-tile, typed-i64,
   multi-tile, and precomputed routes must preflight simultaneously live sample
   planes, transformed coefficients, subband/code-block metadata, ROI plans,
   accelerator job/result arrays, encoded segments, and rate-control layers.
   Parallel collection must use an explicitly fallible bounded strategy, and
   moving stage outputs must replace cloning wherever ownership permits.
3. **Tier-2 packet ownership.** Split production packet formation from its
   large in-file test module. Replace the current always-live merged + header +
   body triple with an interleaved-or-separated representation, make tag-tree
   and bit-writer growth fallible, and budget packet states, descriptor arrays,
   headers, bodies, lengths, SOP/EPH bytes, and prior tile payload together.
   PLT/PLM/PPM/PPT and multi-layer behavior require byte-exact regressions.
4. **Tile-part, codestream, and batch finalization.** Compute exact marker and
   payload lengths before copying, reserve final codestream output once, keep
   tile parts borrowed until that copy where possible, and include prepared
   markers plus retained tile bodies in the final high-water. Batch 9/7 encode
   must cap the aggregate outer metadata and all retained codestream payloads,
   not merely each image independently.

Active ownership review adds the following handoff constraints. Subband
preparation currently copies full transformed stores into one coefficient Vec
per code block, then Tier-1 builds an additional downcast coefficient Vec per
job, a job-descriptor Vec, and a complete encoded-result Vec while the prepared
subbands remain live. Parallel `collect` makes that peak infallible. Multi-layer
rate control then copies the encoded payload into per-layer contribution Vecs
before releasing the original encoded block. Prefer coefficient views/ranges
and one encoded backing owner with contribution ranges where ordering permits;
do not simply place independent caps on every duplicate. The packet lane's new
borrowed contribution view should become the seam for this later conversion.
`tile_parts.rs` likewise clones tile data, packet lengths, and nested headers
even for the single-part case; codestream finalization should consume borrowed
part descriptors and copy payload bytes only into the planned final output.

Structural closure is part of the allocation work: `packet_encode.rs` (1,567
lines) and `codestream_write.rs` (1,071 lines) mix production and tests, while
the 2,355-line encode test owner covers many unrelated families. Split packet
and codestream planning/writing from their tests and divide the encode tests by
single/multi-tile, classic/HT, layered/marker, and precomputed/resident behavior
with exact test-inventory ratchets. Large generated lookup-table modules are
data owners, not orchestration god files, and should not be split mechanically.

Acceptance requires no production `Vec::with_capacity`, caller-derived
`vec!`, `to_vec`, infallible `collect`, or unchecked payload extension in these
owners unless a source ratchet documents a fixed specification bound. Boundary
tests must cover exact-cap acceptance and one-byte/count rejection without huge
allocation. Full classic/HT, reversible/irreversible, single/multi-tile,
precomputed/preencoded/resident, progression, marker, rate-control, facade, and
JPEG-to-HTJ2K suites plus strict Clippy and API/semver regeneration remain
mandatory before ALLOC-013 can close.

Working-tree Tier-2 checkpoint (2026-07-10): the packet owner is now an
idiomatic facade over form/header/ownership/state/view modules and separate
tests. Packet inputs, one AoS code-block state owner, tag trees, header payloads,
lengths, final tile data, and optional separated headers share a race-free
atomic encode ledger with an explicit retained baseline. The checked writer
cannot grow, implicit multidimensional progression is rejected in favor of
explicit descriptors, legacy infallible packet-writer adaptation is test-only,
and classic pass counts 36/37/164 have bit-exact regressions. Warning-free lib
and lib-test compilation, 20 packet tests, nine ledger tests including a forced
stale-CAS seal interleaving, six tag-tree tests, and five source-policy tests
pass. This is a local foundation/Tier-2 checkpoint, not ALLOC-013 closure:
transform/Tier-1, contribution/rate-control, call-site retained baselines,
tile-part/codestream finalization, batch aggregation, and public typed-error
migration remain.

Retained-input checkpoint (2026-07-10): native encode now exposes a move-only
`NativeEncodeRetainedInput<'a>` whose lifetime borrows the actual caller-owned
allocation graph. `RawBitmap`, `DecodedNativeComponents`, and
`Reversible53CoefficientImage` derive their token from allocator-returned data,
outer-vector, nested-plane, level, and coefficient capacities; bit-packed
`Vec<bool>` capacity is converted from bits to bytes. One
`NativeEncodeSession` propagates the baseline through single/multi-tile, typed
i64, precomputed, and scalar packet paths, and scalar packet formation combines
it with the actual packet-owner capacities before adapter/output work. Existing
public entrypoints use the zero token, while doc-hidden typed adapters let
recode paths carry retained owners. Exact/cap-minus-one and owner-aggregation
tests plus source ratchets cover this seam. This is not a whole-operation
ledger yet: transform/Tier-1, accumulated multi-tile bodies, accelerator
outputs, and final codestream ownership remain separately open.

Precomputed 5/3 correction (2026-07-10): the retained coefficient route no
longer allocates a full dummy pixel image or clones the complete DWT tree.
Private `DwtComponentSource` views borrow LL/HL/LH/HH storage directly, a
focused single-tile orchestrator reuses validation, planning, Tier-1,
packetization, and finalization, and a narrow accelerator adapter exposes only
the quantization/Tier-1/packet hooks that are meaningful after sample/MCT/DWT
stages are skipped. Source/component and decomposition-level mismatches fail
before indexing. Tests prove the retained coefficient exact boundary, skipped
deinterleave/DWT hooks, preserved HT hooks, and byte-identical codestreams
against the ordinary pixel-to-reversible-5/3 path. Source is frozen for the
combined compile/behavior gate; accelerator-returned output, general
transform/Tier-1 ownership, multi-tile accumulation, and final codestream
high-water remain open.

Accelerator-output checkpoint (2026-07-11): session-backed packetization and
whole-tile shortcuts now reconcile allocator-returned output capacity before
acceptance. Packet acceleration counts the retained input, owned Tier-2 packet
tree and descriptors, actual outer/nested public packet-view metadata, and the
returned tile data plus packet-length/header owners. Host-pixel and
backend-resident whole-tile routes count the complete nested `SingleTilePlan`
capacity and returned tile body before codestream finalization. A shared
`NativeEncodePhase` preserves allocation/arithmetic errors separately from
`EncodeError::Accelerator`; exact-cap, cap-minus-one, zero-copy parity,
decline/failure, nested-metadata, and resident resource-category regressions
ratchet these seams.

Compact preencoded 9/7 checkpoint (2026-07-11, source-only):
`encode_preencoded_htj2k_97_compact_owned_with_accelerator` now derives a
lifetime-bound baseline from every actual payload/component/resolution/subband/
code-block vector capacity, keeps borrowed payload ranges zero-copy through
Tier-2, and counts the prepared packet tree, descriptors, and outer/nested
public accelerator metadata exactly once. Accepted backend output is reconciled
before return from the hook; an over-cap accepted result returns typed
`AllocationTooLarge` without scalar fallback, while backend failures remain
typed `Accelerator`. After packetization, the source image and borrowed metadata
are dropped before final assembly. Construction now uses direct progression-
ordered packets with phase checks before and after every fallible exact reserve,
so no intermediate component-vector/flattening allocation escapes the cap.
Unsupported PPM/PPT, multi-layer, explicit tile, ROI, and conflicting sampling
options fail instead of being silently ignored; guard-bit and precinct-exponent
marker arithmetic is validated before plan allocation. A scratch-free single-
tile codestream seam computes the exact marker/output length without allocation,
checks the requested high-water, reserves fallibly, reconciles the allocator-
returned capacity, and only then writes markers. Exact-cap, cap-minus-one/no-
fallback, construction, nested-owner, true scalar-decline fallback, zero-copy,
writer-preflight, and independent byte-parity regressions plus structural
ratchets are present. Rustfmt and diff checks pass; no Cargo command was run in
this parallel lane. This is not full
ALLOC-013 closure: general transform/Tier-1, multi-tile accumulation, and
non-compact final-codestream propagation remain open.

Transform/multi-tile/finalization source checkpoint (2026-07-11): a nested
`NativeEncodeSession` now requires an actual lifetime-bound phase-owner borrow;
each child tile and Tier-2 child session therefore keeps the owners behind its
byte claim immutable and alive without counting the original retained input
twice. The ordinary single-tile path preflights fallible deinterleaved plane
owners, the packed-plus-extracted CPU DWT overlap, actual nested accelerator DWT
capacities, and prepared component/resolution/subband/code-block trees. Staged
accelerator failures retain typed operation names for deinterleave, RCT, ICT,
and 5/3 or 9/7 DWT. After packetization, transform owners are dropped and the
large preparation plan is consumed into a marker-only final plan before output
assembly.

Standard interleaved multi-tile encode now has explicit loop-plan and
final-plan ownership transitions. The loop retains only a fallibly copied child
option owner; duplicate default/component quantization marker graphs are
recomputed after tile accumulation under requested and allocator-returned
capacity checks. ROI and precinct input validation happens before tile work.
Each child is forced to emit one unsplit tile, while only the parent applies the
requested tile-part packet limit; a true multi-tile, multi-tile-part decode
round-trip regression protects that contract. Accumulation counts existing and
incoming nested payload/length/header capacities, distinguishes growth from a
no-op reserve, reconciles actual outer capacity, checks the clone peak, then
drops child pixels, ROI, codestream, and extracted metadata before append.
Final planning drops the child option owner before fallibly building marker
metadata and drops step graphs before writer handoff. Exact/cap-minus-one tests
cover loop/final planning, nested sessions, transform owners, growth and
spare-capacity append, and borrowed finalization.

The scratch-free codestream writer now owns the single general assembly route
for ordinary, marker-bearing, split-tile, and multi-tile output. It computes the
exact SOC-through-EOC length before allocation, checks that requested high-water,
reserves fallibly, reconciles allocator-returned capacity, and adds the writer
peak exactly once. PLT/PLM packet lengths and PPM/PPT separated headers stream
directly from borrowed tile-part views; there is no prepared-marker, flattened
length, concatenated-header, or PPT-chunk allocation. Single-tile split
finalization uses borrowed ranges/views and releases range metadata before
output; multi-tile header views are skipped unless PPM/PPT validation needs them
and are released before part views. Exact-cap/cap-minus-one regressions cover
marker-bearing multi-tile assembly and the caller-visible finalization phase
while preserving marker bytes.

Precinct ownership checkpoint (2026-07-11, source-only): prepared Tier-1
coefficient and preencoded payload vectors are now move-only through precinct
partitioning. The splitter reserves only the destination component/packet/
subband/code-block owner arrays, distributes each source block exactly once in
the existing row-major packet order, and preserves the allocator and payload
pointers instead of cloning the complete graph. One focused ownership tracker
counts invariant payload capacity plus the exact simultaneously live source
and destination structural capacities. Every destination reserve is checked
before allocation and reconciled against allocator-returned capacity; released
source owner arrays reduce the tracked overlap as their consuming iterators
finish. The ordinary single-tile route includes its transform/plan baseline,
and typed-i64 uses the same session-backed seam; the legacy precomputed-batch
string boundary maps the typed resource result without changing that public
error contract. Pointer/order behavior and measured exact-peak/cap-minus-one
regressions plus source/line ratchets are present.

This checkpoint is source-only and does not close ALLOC-013. Tier-1 still needs
one fallible phase across quantized stores, per-block `i64` coefficients, the
duplicate `i32` downcast graph, job descriptors, backend/CPU encoded results,
and result-to-packet metadata. Multi-tile still serializes each child
codestream, reparses SOD/PLT/PPM, and clones body/metadata; those extraction/
split allocations remain infallible. Multi-layer rate-control duplication,
typed-i64 multi-tile, precomputed/batch families, and the post-encode HT
self-validation decode peak remain open. `multitile.rs` and
`single_tile/tile_encode.rs` also retain long coordinators that need a further
one-tile/Tier-1 state split. No Cargo evidence is claimed for this precinct
lane; compile, behavior, strict Clippy, policy, and combined-tree gates remain
mandatory.

High-bit transform/multi-tile checkpoint (2026-07-11, source-only): the typed
25–38-bit facade is now a 67-line dispatcher over focused geometry, plan,
single-tile, multi-tile, input, packed-transform, subband, and accounting
modules. Plan construction copies options, sampling, precincts, quantization,
and component metadata with fallible exact reserves while checking both the
requested and allocator-returned capacities. The plan is consumed into an
execution owner and later into marker-only `EncodeParams` plus quantization;
no cloned planning graph remains live at final writer handoff.

Typed and raw high-bit samples now share one packed in-place i64 5/3 DWT seam.
It owns one fallible line scratch, borrows validated packed subband rectangles,
and copies directly into fallibly reserved i64 code blocks. The shared
`PreparedCodeBlockCoefficients` representation preserves i64 through Tier-1,
so the former decomposed-band clone and per-block i64-to-i32 downcast graph are
not part of these routes. ROI maxshift is applied once while each block is
copied from its packed view. Raw single-tile encode moves the complete
`SingleTilePlan` into the route, deinterleaves under its retained baseline,
consumes one component plane at a time, then drops the transform graph before
packetization and consumes the plan again before session-backed finalization.

Typed multi-tile encode no longer writes and reparses a child codestream. Each
tile extracts component planes fallibly, releases tile scratch after packed
preparation, packetizes under the execution-plan plus accumulated-part
baseline, and consumes the packetized tile into final parts. The unsplit case
moves payload, length, and nested header allocations without copying; the split
case checks the complete source-plus-destination overlap, performs only
fallible exact copies, releases the packetized source, and appends through the
existing actual-capacity accumulation seam. Final PPM/PPT/PLT output uses the
same borrowed multi-tile finalizer and general exact-size writer.

Focused regressions cover exact/cap-minus-one plan construction, raw
deinterleave values and capacity, packed i64 preparation and coefficient
representation, consuming split overlap, pointer-preserving unsplit handoff,
and a 25-bit multi-tile PLT/PPT/tile-part round trip. A source policy freezes
the module ceilings and forbids the legacy writer, clone-heavy DWT/subband
path, `Vec::with_capacity`, `to_vec`, infallible collection, and no-op writer
peak callback from these owners. Rustfmt, semantic parse, and diff checks pass
in this lane. The first combined compile exposed internal visibility only; the
exports were narrowed to encode/typed-i64 scope and refrozen. No Cargo command
was run in this parallel lane, so compile, behavior, strict Clippy, policy, and
combined-tree evidence remain mandatory. ALLOC-013 also remains open on the
standard interleaved child-codestream copy seam and any failures found by those
combined gates.

Legacy precomputed 9/7 and batch checkpoint (2026-07-11, source-only): the
non-compact 9/7 coefficient entrypoint now implements the same borrowed
`DwtComponentSource` contract as precomputed 5/3. It no longer allocates a
dummy full-resolution pixel image, clones every LL/HL/LH/HH vector, or removes
precomputed DWT outputs from an adapter queue. One shared precomputed-stage
adapter exposes only quantization, Tier-1, and packetization hooks, and the
normal session-aware single-tile plan, packet, and scratch-free finalizer own
the remainder of the route.

Borrowed prequantized and borrowed preencoded packet inputs now derive exact
baselines from every component/resolution/subband/code-block owner and nested
coefficient or encoded-payload capacity. Their destination graphs use checked
fallible reservations and count the unavoidable borrowed copies together with
the source. The owned preencoded route first allocates and reconciles a
shape/payload-owner skeleton while the source graph is still retained, then
releases that construction session and moves each encoded payload vector into
its reserved destination without reallocating or cloning. Validation,
arithmetic/cap/OOM, Tier-1, packetization, accelerator, and writer failures stay
typed through the native pipeline and public `EncodeError` boundary.

Precomputed batch preparation is now a bounded fallible sequence rather than
an infallible Rayon collection. Prepared packets from every image are moved
into one shared Tier-1 call, then regrouped without copying encoded payloads.
Per-image Tier-2 and finalization count the complete source or owned-input
baseline, plan and group outer allocations, remaining encoded packet graphs,
current metadata, all prior codestream capacities, the output outer owner, the
current packetized tile, and the scratch-free writer peak under one cap. The
owned batch adapter releases the complete DWT source graph before Tier-1 and
output growth; JPEG-to-HTJ2K now uses that consuming adapter. Progression,
PLT/SOP/EPH, tile-part, accelerator-batch, decline/failure, and byte output
remain on the shared packet/finalization machinery.

Focused regressions cover the direct DWT pointer, exact/cap-minus-one direct,
prequantized, preencoded, and multi-image peaks, owned preencoded payload
pointers, one-call Tier-1 batching, per-image byte parity, and marker/tile-part
parity. Strong source and line-count ratchets cover the new allocation,
orchestrator, packet-construction, batch preparation, batch finalization, and
transcode call-site boundaries.

Precomputed architecture and Clippy closure (2026-07-11): borrowed construction
and packet child sessions now end at lexical helper boundaries instead of
using `drop` on non-`Drop` lifetime carriers. The owned preencoded handoff and
owned multi-image batch preparation each return only their fully owned plan,
so the borrowed source/session lifetime ends before payload movement or shared
Tier-1/output growth. Batch finalization is split into a short batch iterator,
one per-image finalizer, and an explicit remaining-owner view; it preserves the
same live-byte ledger while removing the monolithic loop and non-idiomatic
session handoff. The private precomputed subband boundary passes the copyable
quantization step by value and borrows one local request into the shared
subband preparation seam.

Current focused evidence is warning-free
`cargo check -p j2k-native --lib --all-features`; a successful normal
all-feature native Clippy run with no finding in the precomputed files; 7/7
precomputed typed-error, accelerator, direct-DWT, exact-cap, batch-cap, and
batch byte-parity tests; 3/3 owned-payload-move and prequantized/preencoded
byte-parity tests; and the marker/tile-part parity regression. The standalone
`encode_coefficients` integration target remains blocked by two stale
string-versus-typed-`EncodeError` assertions outside this lane. Targeted
rustfmt and `git diff --check` pass. These focused results supersede the
source-only qualification above but do not close ALLOC-013 until the serialized
full behavior, policy, and strict workspace gates are green.

Precomputed typed-error boundary closure (2026-07-11): every non-compact,
compact, and batch precomputed native-result boundary now classifies legacy
static-string helpers explicitly instead of depending on a blanket
`From<&'static str>` conversion. Public option and deep image validation map to
`InvalidInput`; JPEG 2000 component/bit-depth limits remain `Unsupported`;
checked index, dimension, and packet-count conversions map to
`ArithmeticOverflow`; and post-validation payload-owner, skeleton, grouping,
and image-count mismatches map to `InternalInvariant`. Helpers that are still
useful as private `Result<_, &'static str>` validation primitives retain that
focused contract and are mapped only where they cross into the native typed
pipeline. `try_metadata` now delegates quantization owner construction to a
named `QuantizationOwners` helper; the shared precinct validator remains the
single authority for code-block exponent and area checks, and the orchestrator
is back below the strict 100-line function ceiling without a suppression. One
shared option helper now validates quality-layer structure before route
capability: zero layers and nonempty target-count mismatches are `InvalidInput`,
while structurally valid multi-layer or targeted-rate requests remain
`Unsupported`. Compact input similarly rejects simultaneous PPM/PPT and a zero
tile-part packet limit as `InvalidInput` before its broad unsupported marker
branch. The helper lives in the focused options module, keeping the
orchestrator, options, and compact files at 349/350, 85/100, and 499/500 lines.

The all-feature native library check and strict native Clippy both pass
warning-free. Five focused classification tests pass, covering public shallow
and deep input, legacy packet-layer precedence, compact option precedence,
batch image-count invariants, and subband dimension overflow. The refreshed
precomputed ownership repository policy passes with lexical input-session and
typed `BatchTailOwners` ratchets. Targeted rustfmt and diff checks pass for this
slice.

Typed-i64 typed-error boundary closure (2026-07-11, source-complete): every
native-result boundary below `encode/typed_i64/` now classifies private
static-string helper failures explicitly instead of depending on the removed
blanket `From<&'static str>` conversion. Checked tile, dimension, code-block,
index, and sample conversions map to `ArithmeticOverflow`; public plane data
validation remains `InvalidInput`; Part 1 guard/exponent and coded-bitplane
limits map to `Unsupported`; and post-plan missing quantization steps, marker
indices, packed-view bounds, and owner-count mismatches map to
`InternalInvariant`. Focused private geometry, reversible-step, ROI-scale, and
coefficient-shift helpers retain their useful `Result<_, &'static str>`
contracts and are mapped only where they enter the native typed pipeline.
Regressions now pin missing planned steps as an invariant failure and excessive
ROI coded bitplanes as an unsupported capability. Targeted rustfmt and diff
checks pass. A fresh all-feature native library check passes warning-free, and
the focused typed-i64 library subset passes 5/5, including exact/cap-minus-one
plan and packed-preparation ownership plus the 25-bit multi-tile round trip.
These are live-tree checkpoint results; the serialized combined release gate
is still required after source freeze.

General Tier-1 and rate-control source checkpoint (2026-07-11): prepared
code-block coefficients now have explicit `I32`, `I64`, and payload-free
representations. Ordinary quantized input stays `i32` through job construction;
only an actual `i64` source that fits the selected coding path creates a
fallible downcast owner. One Tier-1 phase tracker counts the actual prepared
graph, packet shells, downcasts, job descriptors, CPU or accelerator result
owners, payload capacities, and final packet owners against the encode session.
Preencoded HT payloads and completed results move into packet ownership instead
of cloning. Bounded Rayon execution fills an already fallibly reserved result
owner, and accepted accelerator batches are reconciled before return; an
over-cap accepted result cannot silently retry on the CPU. Accelerator failures
retain their classic- or HT-Tier-1 operation category.

Multi-layer construction now uses one owned rate-control state for classic and
HT candidates, locations, byte claims, and block indices. Every candidate,
assignment, segment-layer, contribution, and payload owner is reserved through
the same session-aware tracker with requested and allocator-returned capacity
checks. The assignment phase consumes and releases its temporary state before
per-layer packet construction, reducing the live peak instead of carrying four
dead arrays through output. Classic PCRD ordering, classic/HT monotonicity, and
segment-boundary semantics remain explicit. The former 1,189-line rate-control
and 1,082-line layered owners are now focused model, assignment, classic/HT
contribution, state, ownership, handoff, output, and test modules with line
ratchets; no `too_many_arguments` suppression remains in the layered path.

The general packetization accelerator adapter no longer constructs nested
resolution/subband/code-block metadata through infallible `collect` calls
before checking its phase. It reserves each borrowed metadata owner fallibly,
tracks the complete already-built graph plus encoded packet ownership, and
passes actual prepared-packet, resolution-packet, and descriptor capacities at
the internal boundaries. Exact-peak/cap-minus-one regressions now cover the
Tier-1 phase, layered rate control, and nested packet accelerator metadata;
additional regressions protect ordinary-i32 borrowing, payload parity, typed
backend failure, and accepted-output no-fallback behavior. A dedicated native
encode policy freezes these modules and forbids production `with_capacity`,
`vec!`, `to_vec`, and infallible collection in this ownership lane.

Packet, layered rate-control, and Tier-1 typed-error closure (2026-07-11): every
native-result boundary in these owners now classifies private static-string
helpers explicitly instead of relying on the removed blanket
`From<&'static str>` conversion. Caller quality-layer counts and cumulative
targets map to `InvalidInput`; mixed Tier-1 modes and unsupported i64 HT input
map to `Unsupported`; checked counts, indices, geometry, shifts, and byte/pass
arithmetic map to `ArithmeticOverflow`; and post-validation packet, candidate,
assignment, segment, job, accelerator-output, and coefficient-owner mismatches
map to `InternalInvariant`. Focused pure test/compatibility helpers retain
`Result<_, &'static str>` only where they do not cross the native pipeline.

The explicit classifications were kept reviewable without lint exceptions:
classic and HT budget application, precinct shell planning and owner movement,
HT assignment workspace construction, and classic contribution planning and
payload construction are separate cohesive helpers. Fresh strict all-feature
native Clippy passes with `-D warnings`; all-feature library-test compilation is
warning-free; targeted rustfmt and `git diff --check` pass; and 12/12 focused
precinct move/exact-cap, layered classification/payload/exact-cap, and Tier-1
classification/accelerator/exact-cap regressions pass. The tests were executed
from the freshly linked library-test binary after a shared-target rust-analyzer
race left direct Cargo-launched binaries sleeping in dyld; this is execution
evidence, not a substitute for the final serialized combined gate. ALLOC-013
therefore remains open on the native allocation policy, the full combined tree,
and the separately tracked standard interleaved multi-tile closure. This focused
slice does not claim an instrumented changed-path coverage percentage; the
repository coverage gate remains required before release.

Tier-1 scratch closure and compiler checkpoint (2026-07-11): classic MQ,
selective-bypass, HT cleanup/refinement, and shared bit-writer growth now use
fallible bounded reservations derived from one validated JPEG 2000 code-block
geometry contract. The contract rejects zero axes, axes above 1024 samples,
blocks above 4096 samples, and arithmetic overflow before fixed-table indexing.
Classic i32 and i64 coefficient views share one generic segmented encoder, so
the ordinary path no longer builds an i32-to-i64 coefficient graph. HT and
classic output buffers reconcile their exact retained length instead of
carrying the full conservative worker bound after completion.

CPU Tier-1 scheduling now executes bounded worker waves. Each preflight counts
the output and scratch bound of every simultaneously live worker plus actual
payload retained from earlier waves; it does not reject ordinary images by
summing the worst-case output of every future code block. Result slots are
fallibly reserved before Rayon dispatch, missing slots and worker failures are
typed, and the serial i64 route uses the same direct coefficient view. The
serial external-accelerator route additionally checks the complete public
classic output, including transient segment-vector capacity, while that owner
is still live and before metadata is discarded during packet adaptation.

The public selective-bypass token seam now requires exact contiguous coding-
pass coverage and the normative MQ/raw schedule: early and cleanup segments
are arithmetic, raw segments start on the significance-propagation pass, span
at most two passes, and never cross a cleanup boundary. Malformed first-pass,
interior-gap, missing-tail, wrong-mode, overlong-raw, and cleanup-crossing
schedules fail before token reads or output allocation. Focused module splits
keep the MQ tests, segmented encoder, CPU waves, accelerator regressions, HT
writers, and legacy test-only facades out of their production roots; the native
Tier-1 policy covers every new owner and freezes these boundaries.

Exact evidence for this checkpoint is now green: `cargo check -p j2k-native
--lib --all-features` and `cargo clippy -p j2k-native --lib --all-features --
-D warnings` pass without diagnostics. The refreshed Tier-1 allocation and
structure policy passes 2/2. Focused behavior passes 2/2 token schedules, 1/1
serial accelerator segment accounting, 2/2 layered exact-cap and payload
parity, 4/4 PCRD ordering/distortion, 3/3 HT layer assignment/contribution,
and 17/17 HT allocation/geometry/byte-golden tests with one diagnostic report
test intentionally ignored. `rustfmt --check` and `git diff --check` are clean.

Strict Clippy drove focused classic/HT layered packet, contribution-output,
candidate-queue, worker-allocation, token-reader, and serial-driver module
boundaries rather than suppressions or raised existing ceilings. The 49
combined-tree findings observed at the earlier checkpoint were reconciled by
their owners; the final strict native run has zero findings. This closes the
Tier-1 scratch design, P1/P2 review defects, and local Tier-1/rate-control
architecture gate. It does not independently close ALLOC-013 or claim release
readiness until the root-owned full behavior and serialized workspace/release
matrix also pass.

Transform and single-tile test boundary closure (2026-07-11): the former
589-line transform owner is now a 33-line real-module facade over focused MCT,
accelerated-DWT, component-sample, typed-5/3-output, and reversible-marker
owners. The largest child is 317 lines. The accelerated DWT boundary now takes
one move-only `ForwardDwtRequest` plus the accelerator instead of nine
positional arguments; the request keeps owned coefficients, geometry, session,
retained baseline, and mutable line scratch explicit without a lint
suppression. Existing encode-module visibility remains available through the
facade, and the only caller constructs the typed request directly.

The oversized 557-line single-tile regression owner is now a 294-line public
behavior/pipeline-order module plus a 273-line whole-tile accelerator ownership
module. Repository policy lowers the former 480-line ceiling to 320, adds a
300-line child ceiling, requires both transform and test module wiring, forbids
whole-tile fixtures from returning to the parent, and includes the child in
the accelerator-regression name coverage. Focused evidence is green:
warning-free `cargo check -p j2k-native --all-features --lib`; strict
`cargo clippy -p j2k-native --all-features --lib -- -D warnings`; all eight
filtered single-tile tests; both the transform-boundary and accelerator-output
repository-policy tests; focused `rustfmt --check`; and lane
`git diff --check`. This closes the transform/test structural lane only; the
serialized workspace and release matrix still own combined-candidate status.

Native typed-error and accelerator-boundary closure (2026-07-11): the blanket
`From<&'static str>` pipeline conversion is gone and forbidden by repository
policy. All 216 exposed call sites were assigned an explicit input,
unsupported, arithmetic, invariant, typed resource, accelerator, or validation
category. Resident encode has a centralized classifier, so invalid options no
longer masquerade as resource failures and arithmetic/codestream invariants
remain backend failures. Precomputed option precedence now rejects malformed
layer/marker/tile-limit combinations as input before applying route capability
limits.

Accepted accelerator Tier-1 output is now validated before ownership conversion
or packetization. One shared metadata module covers HT payload segments and
preencoded inputs; job-relative validation additionally proves requested HT
passes and zero/nonzero block presence. Classic output proves coefficient-
relative bitplanes/pass counts, payload and segment presence, contiguous byte
and pass coverage, and coding-mode boundaries. Batch, serial, and fused-subband
failures retain their precise accelerator operation. Focused module splits keep
the driver, layout, output validation, metadata, layered assignment, and
classic/HT contribution owners within their existing or lower ratchets.

Current evidence for this closure is a warning-free native production/test
compile, strict library Clippy, both Tier-1 policies, nine metadata-category
tests, nine accelerator tests, five preencoded HT tests, fused-HT coverage, and
the focused layered/HT/classic rate-control suites. The broad library run had
495 passes, one established ignore, and five stale/accounting failures now
owned by the combined-regression lane; therefore ALLOC-013 remains open and no
full-suite green claim is made.

Combined native regression and standard multi-tile closure (2026-07-11): the
earlier five broad failures are reconciled. The standard interleaved multi-tile
route no longer writes or reparses a child codestream and its remaining
230-line coordinator no longer carries `too_many_arguments` or
`too_many_lines` exceptions. A lifetime-bound request feeds a 72-line grid
orchestrator, a 226-line per-tile extraction/packetization/ownership transition,
and a 100-line checked grid-geometry owner behind a 39-line facade. Pixel and
ROI capacity are checked before allocation, reconciled after allocation, kept
inside the child-session baseline, then released before packet owners move into
the aggregate parent tile-part graph. The compact preencoded accelerator tests
were also responsibility-split rather than raising a stale 460-line ceiling;
the parent is 302 lines and the focused accelerator child is 165.

Fresh evidence is green: the complete all-feature native package passes 598
tests with the one established diagnostic report ignore; strict all-feature
native all-target/all-feature Clippy and strict xtask all-target/all-feature
Clippy pass with `-D warnings`; all 27 native encode allocation policies pass,
including six standard multi-tile ownership/structure ratchets; and the four
moved compact accelerator acceptance/failure/fallback regressions pass under
their new real module. This supersedes the 495-pass checkpoint above.
The dependent gates then passed on the same source tree: the complete
all-feature `j2k` package passed 327 tests with one external-fixture ignore and
the complete `j2k-transcode` package passed 132/132; both packages pass strict
all-target/all-feature Clippy with `-D warnings`. ALLOC-013 is complete. The
eventual immutable-candidate rerun belongs to FINAL-001 and does not reopen the
implemented ownership contract.

### ALLOC-014 — aggregate J2K/HTJ2K facade batch ownership

The pattern-equivalent review of ALLOC-011 found the same availability class in
`j2k/src/batch.rs`. Each call creates caller-count-derived scoped-thread handles,
per-chunk indexed result vectors, and a second ordered result vector through
infallible allocation. Every worker also owns a native decode context and
scratch pool whose individual decode validation assumes it may use the full
512 MiB allowance. Region batches can retain a shared direct plan at the same
time. Increasing worker count therefore multiplies both the permitted decode
workspace and the untracked collection metadata.

The shared core collector itself allocates infallibly and documents panics for
missing or out-of-range worker indices. `TileBatchError` can represent only a
tile-indexed codec error, so it cannot honestly report an outer allocation or
scheduler invariant failure. Resuming a worker panic and using `expect` for
result completeness also makes the library boundary depend on internal worker
behavior rather than a typed infrastructure contract.

Closure should introduce one shared fallible batch failure model usable by JPEG
and J2K without flattening infrastructure failures into a fictitious tile-zero
decode error. J2K batch planning must count shared direct-plan bytes, worker
contexts/scratch, handles, per-chunk and ordered result metadata, and any
simultaneously live warnings; reduce concurrency or reject before spawning when
the aggregate cannot fit. Worker-result collection must preserve the first
input-order codec failure while returning typed OOM and invariant failures.
Pure exact-boundary tests, worker-count reduction/rejection tests, full/region/
scaled parity, panic-surface ratchets, facade API evidence, and strict Clippy are
required before ALLOC-014 can close.

Independent native review confirmed that a header/geometry-only worker estimate
would be unsound. Exact native peaks depend on tile overrides, packet segment
counts, ROI-required blocks, Tier-1 mode, and Rayon fan-out discovered during
tile/packet parsing. Until native exposes a consumed
`PreparedNativeDecode`/`NativeDecodeAllocationClaim`, every generic facade
worker must therefore claim the full 512 MiB native allowance. Do not label an
output-size heuristic as an exact bound. The repeated direct-color route may use
a tighter claim only after the already-built direct plan exposes separately
checked shared-plan and per-worker CPU-scratch/workspace bytes. This preserves a
safe 0.7 boundary while leaving a clean prepared-decode optimization seam.
The facade must read that worst-case value from the authoritative doc-hidden
`j2k_native::DEFAULT_MAX_DECODE_BYTES`; duplicating the numeric constant would
allow the two crate policies to drift.

Working-tree closure now limits generic batch concurrency to four and charges
each active worker the authoritative 512 MiB native claim. A separate, single
64 MiB metadata domain covers result slots, ordered outcomes, worker/handle
owners, and warnings; checked aggregate accounting is therefore four native
claims plus metadata (roughly 2.06 GiB), while the claims themselves allocate
no memory. Scoped workers use fallible `Builder::spawn_scoped`, write disjoint
slices of one preallocated ordered slot vector, and return typed spawn, panic,
and invariant failures. Workers and a repeated direct-color shared plan are
dropped before dual-limit ordered collection. Native/header decode failures are
preserved in heap-free typed facade variants, and an unexpected legacy
string-owning decode error fails closed as a static internal invariant before
result retention. The root is split into allocation, direct, planning,
scheduler, worker, and focused test modules.

Verification passed 9 batch units, 11 full/region/scaled/region-scaled and raw/
wrapped HTJ2K parity tests, and 3 source/structure policy tests. All-target/
all-feature facade check, strict all-target/all-feature Clippy with warnings
denied, and scoped diff checks pass. `TileBatchError` intentionally now aliases
`BatchDecodeError<J2kError>`; the stable API snapshot and semver report must
record that pre-1.0 source break and the additive non-exhaustive heap-free error
variants under SEM-001/FINAL-001.

Post-closure audit found the former doc-hidden
`collect_indexed_batch_results` still exported despite having no production
caller. Its result type could encode only a tile error, so the helper retained
an infallible `Vec::with_capacity`, an out-of-range assertion, and an `expect`
for result completeness beside the authoritative fallible collector. The
legacy helper/export are removed in 0.7; its three passing regressions now
exercise the fallible API and require ordered success, lowest-index typed tile
failure, and typed out-of-range infrastructure failure without unwinding. A
repository policy forbids restoring the panic/infallible collector. The three
converted core API regressions and focused policy pass. Warning-denied,
all-feature library/test Clippy for `j2k-core` and `j2k-native` also passes
after reconciling the unrelated native test idioms without suppressions, so
ALLOC-014 is closed. Frozen public-API/semver evidence remains under
SEM-001/FINAL-001 and does not reopen this allocation contract.

### ALLOC-015 — fallible reusable J2K row-decode scratch

The J2K facade's reusable `J2kScratchPool` is outside the native tile workspace
planner. Bounded row decode computes checked stripe and u16-row lengths, then
calls helpers that use `Vec::resize` directly and return bare mutable slices.
A cap-valid request can therefore abort on allocator failure, and a prior wide
row remains retained when a later narrow request arrives. `reset` clears lengths
without releasing or accounting for that capacity.

Closure requires fallible reserve-before-resize using the shared typed buffer
error contract, an aggregate packed-byte plus u16-row live formula, and reuse
logic that counts retained capacity or releases stale owners before growth.
Row-decode callers must propagate cap, arithmetic, and allocator categories
without partial sink callbacks. Add exact-boundary and stale-reuse pure tests,
ordinary 8/16-bit row parity, source ratchets, strict facade Clippy, and public
API evidence before ALLOC-015 can close.

Working-tree closure uses a pure aggregate byte plan, drops a stale allocation
before any required growth so allocator reallocation cannot transiently retain
old and new capacities, verifies actual returned capacities, and maps reserve
failure to `BufferError::HostAllocationFailed`. The u16 route reduces its packed
stripe allowance by the simultaneously live converted-row bytes.

Closure reconciliation (2026-07-11) checks the allocator-reported packed
capacity immediately after packed growth and before any u16-row reservation.
If overcapacity consumes the row budget, stale row storage is released and the
planned row is rechecked; an impossible aggregate clears both disposable owners
and returns the typed cap error without making the second allocation. Four
scratch unit tests, five full/region row parity tests, and three source-policy
tests pass. The affected-package all-target/all-feature test matrix and
warnings-denied no-deps Clippy pass, so ALLOC-015 is complete. Its changed
helpers are crate-private; frozen-candidate API evidence remains a release-wide
FINAL-001 concern and does not reopen this allocation contract.

### ALLOC-016 — move-based facade native-component decode

The facade's owned native-bit-depth component APIs currently receive an owned
`j2k_native::DecodedNativeComponents`, borrow every plane, clone every complete
payload with `to_vec`, and collect a second outer plane vector while all native
owners remain live. A native decode accepted at its own workspace boundary can
therefore reach roughly two full decoded payloads and abort in the facade
conversion. Borrowed component decode avoids payload copies but still collects
caller/input-derived plane metadata infallibly.

Closure should add a narrow consuming native handoff that moves plane payloads
without exposing fields broadly. The facade then reserves its destination
metadata fallibly, counts the simultaneously live native and facade outer
vectors, and moves one plane at a time. Borrowed metadata collection must use
the same typed cap/OOM contract. Full/region, mixed precision/signedness,
arbitrary component count, payload pointer/move evidence, exact boundaries,
source ratchets, and strict native/facade Clippy are required before ALLOC-016
can close.

Working-tree implementation now provides `#[doc(hidden)]` consuming native
adapter seams, moves plane payload and ICC owners without copying, and keeps
the facade types isolated in `decode/component_handoff.rs`. Native results
expose actual retained-capacity evidence; borrowed results record the exact
decoder-channel, SIMD-padding, integer-shadow, metadata, and ICC baseline at
construction. The facade preflights source plus destination metadata, reserves
fallibly, and rechecks the allocator-returned destination capacity before
moving owners.

Closure reconciliation (2026-07-11) found that the facade handoff allocated its
second outer metadata vector while `J2kDecoder` still retained the parsed native
`Image`, but counted only the decoded result. All four full/region and
borrowed/owned entry points now carry `Image::retained_allocation_bytes()`
through both logical and allocator-capacity destination checks. The same audit
removed unused infallible `Clone` implementations from facade component/color
owners and native `ColorSpace`; the two real Metal Gray/RGB call sites now
reconstruct only heap-free variants and reject ICC/unsupported color spaces
before plane-stage ownership transfer. The ICC regression preserves the
caller's profile pointer and capacity.

Three exact-boundary/pointer-preservation unit tests, eight full/region
component behavior tests, three handoff source-policy tests, the Metal
color-ownership regression, the complete affected-package test matrix, and
warnings-denied all-target/all-feature no-deps Clippy pass. The breaking
pre-1.0 Clone removals are recorded in `CHANGELOG.md`; the later frozen API
report must enumerate them under SEM-001/FINAL-001, but that serialized release
evidence does not reopen the complete ALLOC-016 ownership contract.

### ALLOC-017 — bounded JP2/JPH parse, wrap, and recode ownership

The container/recode sweep found complete-input `to_vec` copies for passthrough
and codestream-preserving paths, nested temporary JP2/JPH box payload vectors,
caller-count metadata `collect`/`with_capacity`, sampled-plane compaction, and a
final file vector that grows through unchecked `extend_from_slice`. Individual
box lengths are validated for format width, but no phase plan covers temporary
metadata, retained codestream bytes, wrapper overhead, and final output at the
same time. Cap-valid recode branches can therefore duplicate a near-cap input
or abort during an otherwise valid wrapper append.

Closure requires fallible parse metadata ownership, checked exact box/file
length planning, and one final reserve before writing. Fixed small box payloads
should use stack arrays; variable metadata should be written directly or kept
borrowed until the final copy. Passthrough and codestream-preserving recode must
avoid an unnecessary full copy before wrapper assembly and must include native
encode output plus wrapper output in their phase peak. Exact-cap, over-cap,
large ICC/palette/mapping/channel metadata, JP2/JPH parity, passthrough/recode,
source-policy, and strict facade evidence are mandatory before ALLOC-017 closes.

The implementation split should also remove the current dual parse ownership:
`parse_jp2_container_with_strict` builds private `ImageBoxes` and then clones
ICC, palette, mapping, and channel data into a second public metadata graph even
when image decode immediately discards that graph. Internal image parse should
retain only its decode boxes; public inspection should consume/convert the
internal representation once under a shared JP2 allocation budget. `colr`,
`pclr`, `cmap`, and `cdef` counts and nested palette rows need fallible exact
reservation plus actual-capacity accounting. On the facade side, extract
allocation/length planning, metadata validation/iteration, and final box writing
from `wrap.rs`; extract output selection/wrapping from `recode.rs`. The final
writer should stream fixed headers and borrowed payloads directly into one
pre-sized fallible output instead of nesting temporary box vectors.

Discovery checkpoint (2026-07-10): the original parse path could create three
metadata graphs for one container: private native `ImageBoxes`, native public
inspection metadata, and the facade public representation. This established
the consuming-move requirement for ICC and nested palette payloads, including
the retained source outer metadata until each move occurred. Codestream
inspection had the same requirement for `CodestreamInfo` and its component
vector. It also established that malformed-box diagnostics must retain a raw
four-byte tag or static category rather than allocate tag strings. The
working-tree progress below supersedes this discovery snapshot; preserve
unknown-box and ordering behavior and do not discard metadata to satisfy a cap.

The recode high-water is explicitly coupled to ALLOC-013. Packed fallback
retains decoded pixels while native transform/Tier-1/output allocations run;
component fallback can additionally retain compacted component grids and their
outer metadata; coefficient fallback retains the complete precomputed
coefficient image while native encode runs. These owners must enter the native
encode ledger as allocator-returned-capacity baseline bytes, or be consumed and
dropped before the next phase. A locally bounded one-pass wrapper is necessary
but is not sufficient to close ALLOC-017. The retained-input session and direct
precomputed 5/3 correction below now carry packed, component, and coefficient
owners into native encoding, but the general transform/Tier-1, accelerator,
multi-tile, and final-codestream phases still need the same aggregate domain.

Working-tree progress (2026-07-10): native JP2 decode now retains only the
private box graph, while inspection consumes it into public metadata and moves
ICC profiles and nested palette rows. The facade consumes that public graph and
codestream component metadata without cloning. All parser-side metadata growth
uses fallible exact reservation, actual capacities, transactional non-strict
PCLR/CDEF attempts, and typed resource failures. Private heap-owning JP2 types
are no longer `Clone`. Wrapper output now has an allocation-free validation and
exact-length plan plus a checked direct writer that reserves one final vector;
recode passthrough copies fallibly, codestream-preserving wrap reads borrowed
input directly, and owned encode output plus JPH wrapper capacity share one
aggregate peak. Pixel fallback also counts actual decoded owners, compacted
planes, outer metadata, and the returned codestream, then drops those sources
before wrapper allocation. Source ratchets forbid reintroducing clone/to-vec or
nested payload-vector assembly across these boundaries.

ALLOC-017 intentionally remains open on the already-recorded ALLOC-013 seam:
packed, component, and coefficient recode owners now expose actual aggregate
capacity and pass a lifetime-bound baseline into native encode, and the direct
precomputed 5/3 path removes its former dummy image and coefficient-tree clone.
However, the current session proves the retained baseline and scalar packet
phase, not every simultaneously live transform, Tier-1, accelerator-returned,
multi-tile, and final-codestream owner. A later post-allocation facade check is
typed defense in depth, not a substitute for those remaining pre-allocation
claims.

Paired-validation correction (2026-07-10): recode drops the facade parsed
metadata before validation, then parses the source while counting the encoded
output vector's actual capacity and parses the encoded image while additionally
counting the first image's actual metadata. Raw-codestream and JP2/JPH parser
budgets now accept that external baseline before header, synthetic COLR,
nested ICC/palette/mapping/channel, codestream-header, and color-profile
allocation; final parsed-image construction rechecks the actual aggregate.
The first native decoded result's actual packed or nested component capacity is
then added to both parsed images and the encoded output before the second
decode. Mixed-precision images use component-plane comparison; uniform images
also compare per-component signedness and packed sample metadata. Decode errors
are propagated instead of being treated as an alternate format or pixel
mismatch. Helper-level exact/one-over arithmetic tests, full-path
retained-baseline rejection tests, paired-budget tests, and source ratchets
cover the handoff. The reopened 1,102-line `image.rs` owner was split into a
738-line core, 352-line output API, and 104-line direct-plan API with explicit
800/400/130-line ceilings. Current native/facade behavior, strict-Clippy, and
policy gates remain required before this correction is accepted as release
evidence.

Facade encode-validation correction (2026-07-11, source checkpoint): every
lossless, lossy, sampled-component, typed-component, and PSNR round-trip now
keeps the allocator-returned capacity of the generated codestream inside both
parse and decode budgets. Two doc-hidden native decode adapters combine that
external capacity with actual parsed metadata before packed or owned-component
decode. The shared retained decode-owner budget reports cap excess and
arithmetic overflow as structured `DecodeError::AllocationTooLarge`; the facade
preserves native resource failures and stable operation context in the
validation-phase `J2kError::NativeValidation` variant instead of formatting
them into backend strings or misclassifying generated EOF/unsupported output as
caller input/capability failure. Output geometry, metadata, and sample
mismatches use `BackendErrorKind::Validation` rather than blaming valid caller
samples.

Component validation uses owned native planes and compares canonical samples
streamingly. It masks unused precision bits, sign-extends non-byte-aligned
signed samples, handles either component-grid or resolved reference-grid
output, and does not allocate a second full canonical reference image. Byte-
target and PSNR searches retain only scalar best-candidate state; each selected
scale is encoded once after the search so an earlier codestream cannot escape
the next native encode phase budget. The returned attempt rechecks its byte or
PSNR postcondition, and accelerator backend/RequireDevice resolution uses only
that final attempt's dispatch rather than discarded probes. Focused exact/over-
cap, stateful final-attempt, signed-metadata, and canonical-sample tests are
present. Five repository-policy tests freeze retained-capacity use,
typed error propagation, allocation-free component comparison, rate-search
ownership, and module ceilings and pass locally. A warning-free all-feature
`j2k-native` library check and warning-free all-feature `j2k` library check
passed at the pre-scratch checkpoint; later Tier-1 and facade source edits make
those compile results checkpoints rather than final release evidence. Focused
behavior tests, strict Clippy, full facade/recode suites, stable API review, and
the serialized final matrix remain mandatory.

Palette-validation correction (2026-07-10): a valid JP2 palette can resolve a
uniform index codestream into columns with different precision and signedness,
so the codestream header is not proof that packed comparison is lossless.
Paired validation now always uses component comparison when palette resolution
is active. Palette columns above the exact 24-bit integer range of `f32` also
retain a fallibly allocated, budgeted i64 shadow; native packing reads that
shadow instead of the rounded float representation. Regressions compare mixed
unsigned-8/signed-16 palettes and adjacent 25-bit values above `2^24`, where
the former packed/index-precision path could make distinct valid images appear
equal. Full JP2 baseline tests additionally cover implicit CMAP allocation and
an owned ICC profile plus the cloned `Image` color profile.

The same 2026-07-10 boundary review corrected a separate decode-side seam:
parsed JP2/JPH boxes, nested ICC/palette owners, and the `Image` color-profile
clone now enter native decode/direct-plan/recode/postprocess/output allocation
budgets as one actual-capacity baseline. This does not close ALLOC-017's native
encode or paired-validation blockers, but prevents the container metadata graph
from hiding outside the decode ceiling while those later seams are repaired.

Architecture reconciliation (2026-07-11, source-only): borrowed component
planes and owned native component planes now both call
`component_plane_sampling_at`; the duplicate palette/header sampling method was
removed. Decode-side metadata, palette postprocess, integer-shadow, and native
output accounting now delegate component-owner, overflow, and exact-cap
arithmetic to one `DecodeOwnerBudget`, while `NativeOutputBudget` remains
a focused domain facade for output-owner composition. Regressions retain subsampled borrowed
sampling and now require identical owned sampling, resolved-palette display-grid
sampling, premultiplied-alpha propagation, high-precision integer-shadow
packing, and exact/one-over allocation boundaries. Rustfmt and diff hygiene are
the only evidence recorded here; the focused source policy and Cargo gates are
still pending the root agent's serialized run.

The lightweight raw-codestream inspector was a remaining parallel metadata
owner outside the full parser budgets: after validating `Csiz` it still used
`Vec::with_capacity` for up to 16,384 public component descriptors. It now
computes the exact descriptor bytes, uses `try_reserve_exact`, and returns the
additive non-exhaustive `J2kCodestreamHeaderError::HostAllocationFailed`
variant. The maximum Part-1 component-count regression and repository policy
lock the fallible path; focused native execution and strict Clippy remain part
of ALLOC-017's combined verification.

Move-only reconciliation (2026-07-11): public facade/native JP2 metadata and
container results, `ReencodedHtj2k`, native reversible coefficient images,
classic/HT encoded blocks, forward-DWT outputs, precomputed/prequantized/
preencoded coefficient and payload graphs, compact batch outputs, and public
transcode DWT results no longer derive infallible `Clone` over image-sized
vectors. No production clone caller existed. Test-only duplicate fixtures now
construct independent owners explicitly. Source policy pins that closed owner
list, and all native/facade/transcode dependents compile and pass their
all-target/all-feature tests and warnings-denied no-deps Clippy.

`J2kPacketizationResolution`/`J2kPacketizationSubband` joined the move-only list
after J2K Metal changed resident planning and submission to move code-block,
packet descriptor, and resolution owners instead of cloning nested metadata.
The native direct-plan/subband/code-block graph remains a documented exception:
Metal prepared-plan caches still clone its nested entropy payloads. That cache
ownership belongs to METALCACHE-001 and requires a move, `Arc`, or borrowed
redesign before the derives can be removed; it does not justify restoring Clone
to already-closed payload owners.

ALLOC-017 closure (2026-07-11): ALLOC-013's retained native-encode lifetime
seam is complete, and the same-source dependent matrix is green. In addition to
the 598-pass native package and strict native Clippy, the full 327-pass facade
suite exercises parse, wrap, encode validation, passthrough, coefficient and
pixel recode, JP2/JPH metadata, palette, ICC, and high-bit paths; the full
132-pass transcode suite covers retained JPEG-to-HTJ2K owners. Strict
all-target/all-feature Clippy passes for all three packages, all 27 native
encode ownership policies pass, and all 12 container/recode allocation policies
pass. ALLOC-017 is complete; immutable-candidate repetition is owned by
FINAL-001.

### ALLOC-018 — fallible Metal adapter batch metadata

The post-remediation infallible-capacity scan found that the CPU JPEG/J2K batch
cores are now fallible, but their Metal adapters still allocate several
caller-count-derived owners with `Vec::with_capacity`. JPEG Metal public request
builders in `lib.rs`/`codec_batch.rs`, packet planning, entropy payload/offset/
length/checkpoint flattening, grouped full/region/texture result vectors, and
surface/texture collections can abort instead of returning an error. The
entropy helper checks total byte addition but then uses unchecked
`tile_count * segment_count` for checkpoint capacity and does not preflight or
fallibly reserve the simultaneously live byte/offset/length/checkpoint owners.
J2K Metal has the same pattern in direct-batch request specs, CPU fallback
metadata, prepared plans, output collections, resident preparation, and
surface assembly. Fixed three-plane/tiny profiling vectors and syntax-proven
test-only vectors are not findings.

Closure requires one shared, typed adapter-batch allocation mapper per crate,
using `j2k_core` fallible reservation and `BatchInfrastructureError` rather
than a fictitious tile or encode error. Count multiplication and byte-size
arithmetic must be checked before allocation; aggregate phase preflight must
include every metadata vector that coexists with borrowed/retained packet
owners, and actual allocator capacities must be reconciled before GPU buffer
creation. Builders should allocate only after request validation where
possible, and partial failures must retain the original tile/index or
infrastructure classification. Do not replace these vectors with unbounded
iterator `collect` calls or assign host OOM to `MetalKernel`.

Acceptance requires source-aware inventory/disposition of every production
`Vec::with_capacity` in both adapters, exact-cap/one-over/overflow tests for
entropy/checkpoint and grouped-result planning, typed allocation-source tests,
full CPU-fallback and Metal behavior parity, strict no-deps Clippy, repository
ratchets forbidding caller-derived infallible capacity, and final Metal
hardware execution. The closure evidence below satisfies this product gate;
FINAL-001 owns only immutable-candidate repetition.

Product remediation is source-complete as of 2026-07-11. Both adapters now
delegate checked count/byte arithmetic, simultaneous-owner preflight, fallible
host reservation, and allocator-capacity reconciliation to one backend-neutral
doc-hidden `j2k_core::BatchAllocationBudget`; thin crate-local aliases preserve
adapter-specific contexts and typed error mapping without duplicating the
allocator. JPEG exposes the additive typed `BatchInfrastructure` error variant;
J2K uses its existing variant. Public/lazy tile-batch queue growth reserves
before session mutation, entropy bytes/offsets/lengths/checkpoints share one
plan, and grouped packet/result, resident preparation/Tier-1/packet-plan, and
surface/texture handle owners retain infrastructure classification and source
chains. Allocation failures are not routed through `Encode`, `Buffer`, or
`MetalKernel`, and ERR-009's Metal-support/cache routing is unchanged. Resident
J2K ownership now moves the original code-block allocation through all three
prepare routes, transposes component-resolution plans without nested-vector
clone, and moves packet descriptors/resolutions from metadata into compute
submission. The internal resident resolution type is no longer cloneable;
pointer/capacity and source-policy regressions pin the move-only handoffs.

The source-aware raw-capacity inventory leaves only explicit non-ALLOC-018
owners: fixed 16-tile viewport planning, fixed one/three-plane/component and
syntax-bounded encoder plans, profiling bookkeeping, single-tile Tier-2
descriptors, and coefficient-transform payloads tracked by ALLOC-001/
ALLOC-013. Test-only and benchmark-tool vectors are excluded by role rather
than substring replacement. A second scan for `Vec::new` growth and iterator
collection, rather than only `with_capacity`, closed the remaining direct
classic/HT flatteners, nested stacked/repeated band graphs, retained/status/
scratch resource lists, color component preparation, cropped HT payloads,
lossless coefficient jobs, encoded classic segments, and token-pack result
metadata. The two codec tile-batch implementations now share one
`FallibleSubmissionQueue` owner in `j2k-metal-support`; codec-specific session
and packet behavior remains in the adapters instead of a duplicated macro.
Focused allocation helpers were split out when the added contracts crossed the
existing file-size ratchets.

ALLOC-018 closure evidence (2026-07-11): the focused allocation policy passes
9/9, including the raw-capacity inventory and no-`collect` ratchets. Exact-cap
and one-byte-over tests cover shared submission/output capacity, distinct
classic and HT submissions, nested stacked band graphs, direct execution
resources, entropy/checkpoint ownership, and grouped results; typed allocator
source and overflow tests remain green. Warnings-denied all-target/all-feature
Clippy passes for `j2k-metal-support`, `j2k-metal`, and `j2k-jpeg-metal`.
Their complete all-target/all-feature test matrix passes on Apple Metal:
22 support tests; 196 JPEG Metal library tests plus all integration and bench
targets; and 255 ordinary J2K Metal library tests, 54 device integration tests,
and all remaining targets. The 18 ignored J2K Metal release-lane tests also
pass with `J2K_REQUIRE_METAL_RUNTIME=1`, so hardware execution is fail-closed.
The production clone audit passes at 2.11% duplicated lines. ALLOC-018 is
complete; cache and pool retention limits remain
separately owned by METALCACHE-001, JPEGCACHE-001, and METALPOOL-001.

### METALCACHE-001 — byte-bounded move-only prepared-plan caches

The move-only and map-owner scans found that J2K Metal session caches still
deep-clone the public native direct-plan graph. `J2kDirectGrayscalePlan` and
`J2kDirectColorPlan` contain nested subband/code-block jobs and full entropy
payload vectors; `session.rs` clones a plan into the cache and clones it again
on every hit. Three session caches and the global region-scaled cache each
permit 128 entries. `PreparedPlanCache` limits aggregate input-key *length* but
not allocator-returned key capacity, hash/bucket metadata, retained native-plan
payload/metadata, prepared host owners, or Metal buffer lengths. Entry count is
therefore not a meaningful memory bound, and a cache hit itself can abort on a
deep infallible clone.

Closure requires move/`Arc` ownership for native direct plans and prepared
values, followed by removal of infallible `Clone` from the public direct-plan,
owned-subband, and code-block owner graph. Each cache entry must report actual
host and device weight: owned-key capacity, native nested Vec capacities,
prepared host owners, and allocator-reported Metal buffer lengths. Insertion
must evict or explicitly skip before separate configured host/device limits
are crossed, reconcile fallible cache-metadata capacity, and avoid retaining a
second full owner during replacement. Preserve SEC-009's randomized digest,
owned full identity, and equality-before-hit guarantee; a flat bounded entry
owner is acceptable if it keeps collision lookup predictable. True lock or
invariant failures and any operation-required allocation failure must retain
ERR-009's typed source contract; ordinary disabled/oversized cache admission is
an explicit non-error outcome.

Acceptance requires pointer/owner-sharing hits, no deep plan clone, actual key
overcapacity, host-value/device-value exact and one-over limits, replacement
peak, 128-distinct-entry eviction, forced digest collision, disabled/oversized
admission, allocation-source, and global/session-isolation regressions; source
policies covering the complete direct-plan owner graph and cache weights;
strict native/Metal checks and Clippy; API/changelog review; and exact-source
Metal stress with a stable cache high-water.

Closure evidence (2026-07-11): native and prepared direct-plan owner graphs are
move-only; decoder fields and all three session routes share native/prepared
owners through `Arc`, and the global/session ROI+scaled color routes share the
same weighted cache policy. The flat cache keeps a randomized digest plus owned
full identity, updates deterministic LRU recency only after full equality, and
uses explicit 64 MiB host and 256 MiB device ceilings. Host weight includes the
allocator-returned entry-vector and key capacities, native component/step/job/
entropy/segment capacities, prepared step/group/member/payload capacities, and
stable CPU Tier-1 cache owners. Device weight sums every retained Metal
buffer's reported length. Cached hybrid plans disable later coefficient
retention so admitted weight cannot grow behind the cache ledger.

Insertion computes value weight fallibly, reuses an exact replacement's owned
key, reserves cache metadata fallibly, reconciles allocator-returned capacity,
evicts deterministically before commit, and declines disabled or individually
oversized optional admission without converting it into a decode failure.
Allocator and invariant failures still route through ERR-009's
`PreparedPlanCacheAllocation`/`PreparedPlanCacheInvariant` variants. Regressions
cover full-key forced collisions, Arc pointer-sharing hits, actual key capacity,
host/device exact and one-over boundaries, replacement eviction, deterministic
128-entry eviction, disabled/oversized admission, metadata allocation source,
and distinct session cache ownership. The complete native/prepared owner graph
and cache routes are pinned by three focused repository policies.

Verification passed the full `j2k-native` all-feature package (522 library
tests plus all integration/doc groups) and full `j2k-metal` all-feature package
(255 library tests passed, 18 explicitly ignored hardware-lane tests, plus 54
device tests and all integration/doc groups). The exact Metal lane passed the
native/prepared pointer-sharing cache-hit regression with
`J2K_REQUIRE_METAL_RUNTIME=1`. Strict all-target/all-feature Clippy passed for
both packages, strict no-default library Clippy passed for both packages, and
all three `metal_plan_cache_policy` tests passed. API review records the
intentional removal of public deep-`Clone` implementations and the new
doc-hidden retained-capacity query in the changelog.

### JPEGCACHE-001 — byte-bounded JPEG accelerator input/packet plans

The ERR-015 follow-through found that `j2k-jpeg-metal::SessionState` has three
independent eight-entry caches: input aliases, batch shapes, and fast packets.
Each retained entry holds or shares a full `Arc<[u8]>` input, and the fast-packet
entry can additionally retain a frame-scale entropy/checkpoint owner graph.
`Arc::from(input)`, `VecDeque::push_back`, and packet construction are currently
infallible or have their typed errors erased. The equivalent scan found the
same design in `j2k-jpeg-cuda::CudaSession`: separate 4:2:0, 4:2:2, and 4:4:4
eight-slot caches each retain a full input plus packet owner. A fixed entry
count is not a byte limit: individually cap-valid inputs plus packet graphs can
pin several gigabytes in one long-lived accelerator session. This is P1
service-availability debt.

Required closure:

1. Replace each backend's loosely synchronized/per-family caches with one
   per-input cached-plan owner (or one authoritative identity table referenced
   by small derivative entries). Prefer a shared cache identity/budget policy
   in the neutral JPEG/core layer so Metal and CUDA cannot drift. Preserve
   randomized/strong-enough digest plus full-byte equality; pointer identity
   may only accelerate a hit after content identity is still proven under the
   existing alias-reuse threat model.
2. Enforce one collective retained host-byte limit covering actual input bytes,
   matching fast-packet entropy/checkpoint/vector capacities, keys, and cache
   metadata. Reserve metadata fallibly and inspect allocator-reported
   capacities. Evict deterministically before commit; an entry larger than the
   cache limit may serve the current request under its live ALLOC-008/018 budget
   but must not be retained.
3. Integrate ERR-015's inspect-once route in Metal and preserve CUDA's typed
   family selection: cache at most the one matching 4:4:4, 4:2:2, or 4:2:0
   packet family plus its shape. Unsupported capability is cacheable as a small
   typed state; cap, allocator, parse, and invariant errors are returned and
   never cached as absence.
4. Make input interning and cache insertion typed/fallible. Move or share
   packet owners through `Arc`; no cache hit or insertion may deep-clone input,
   entropy, checkpoint, or packet vectors. Queued-request and cache baselines
   must compose when they simultaneously retain the same or distinct inputs.

Acceptance requires repeated-hit/full-equality, pointer-reuse-with-new-bytes,
one-family-only build, oversized-entry non-retention, byte-bound LRU/eviction,
actual-capacity overage, metadata allocation failure, hard-error non-caching,
shared-input non-double-counting (or a documented conservative upper bound),
and long-session stable-high-water regressions for both backends; source policy
forbidding the three-builder `.ok()` pattern, per-family full-input caches, and
infallible cache growth; strict Clippy; and exact-source Metal and NVIDIA
stress.

The backend-neutral foundation is source-complete in `j2k-jpeg` without yet
changing either adapter. One doc-hidden `JpegCachedPlan` owns a fallibly copied
`SharedJpegInput`, the canonical cadence-4 `DeviceBatchSummary`, and an explicit
unsupported-or-ready one-family packet state. `JpegPlanCache` is one flat
randomized-digest/full-byte-equality LRU with 8-entry/64-MiB defaults, fallible
metadata reservation, allocator-capacity accounting, typed allocation and
invariant errors, non-error disabled/oversized admission, and current/peak
diagnostics. Its `resolve` method is the single backend-neutral full-equality
hit, fallible input copy, inspect-once build, typed admission, and current-plan
return boundary. Input and packet hits clone only small `Arc` handles; retained
weights charge each cache entry's complete input and one selected packet graph
once, using exact `Vec` capacities plus the documented stable-Rust estimate for
the small `Arc` allocation and control block. Parse/decoder, fast-packet, and
cache failures remain distinct typed sources; only capability mismatch becomes
the explicit negative state, whose summary fast-family flags are cleared.

The foundation subsequently gained caller-owner-aware shared-input decoder and
CPU-decode entry points. Fresh-output allocation charges the exact output
capacity, retained scratch, and any pre-existing CPU entropy-checkpoint cache;
checkpoint replacement charges the old and replacement capacities while both
are live. The new routing and cache-build responsibilities are being split into
real child modules under the existing source-size ratchets rather than raising
those ratchets.

Moving-tree checkpoint on 2026-07-12:

- The neutral flat cache and its inspect-once APIs have passed focused cache,
  external-output, and prepopulated-checkpoint exact/one-over regressions. The
  final full neutral package, both strict Clippy configurations, rustdoc, and
  structure policies must be rerun after the current module split settles.
- JPEG Metal has replaced the three independent caches with the neutral owner,
  added identity-aware queued-plan accounting, and made queue growth
  transactional: replacement vectors are allocated and admitted off-state and
  committed only after every fallible check. It still must carry actual
  compatible-group vector capacities throughout execution, account host
  surfaces retained in completion slots, compose generic submission/result
  vectors, and pass the existing ledger/session size ceilings without widening
  them.
- JPEG CUDA clones now share the packet cache, operation gate, runtime context,
  output pool, and active-host ledger. The context binding is single-context
  and retryable after initialization failure. Remaining closure is to validate
  resident encode metadata before first binding, reuse one constructed CPU
  decoder on generic owned-decode routes, keep batch/result metadata under the
  same ledger, attach exact leases to session-produced host surfaces where the
  session retains responsibility, and finish the existing `owned_decode`
  structural split.

JPEGCACHE-001 remains in progress until those source items, long-session
high-water tests, strict local gates, and exact-source Apple/NVIDIA stress pass.

### METALPOOL-001 — bounded, fallible Metal scratch retention

The map-backed owner scan found `j2k-metal::MetalBufferPools` is attached to a
long-lived runtime and stores exact-size private and shared buffers in two
`HashMap<usize, Vec<Buffer>>` values. Every completed request recycles all
eligible scratch buffers with `entry(bytes).or_default().push(buffer)`. Those
map/vector growth operations are infallible even though recycle returns
`Result`, and there is no retained-byte, buffer-count, size-bucket, or eviction
limit. A sequence of individually valid requests with distinct dimensions can
therefore retain unbounded device memory and host pool metadata after all work
has completed.

Closure requires a per-runtime pool state with explicit, separate private and
shared retained-byte and entry-count limits derived from device/runtime policy,
not the host allocation cap. Prefer a small flat, fallibly pre-reserved slot
owner over nested exact-size hash buckets unless measured lookup behavior
requires a map. Reuse may remain exact-size, but recycling beyond the limit or
after metadata-reserve failure must safely drop the completed buffer instead of
failing the codec or retaining it. Taking a buffer must decrement accounting;
recycling must add the allocator-reported Metal buffer length, reject overflow,
and never cache a buffer whose recorded size disagrees with its actual length.
Keep command-completion/lifetime ordering intact and preserve typed poisoned-
state or Metal allocation errors where the requested operation itself fails.

Acceptance requires repeated-size reuse, many-unique-size boundedness,
oversized-buffer decline, private/shared isolation, take/recycle accounting,
metadata-reserve-failure, size-mismatch, and completion-order regressions; a
source policy forbidding infallible map/vector growth in pool recycle; strict
Metal checks/Clippy; and exact-source hardware stress proving the pool reaches
a stable retained high-water rather than monotonic growth.

The original 64-record private retention checkpoint below was insufficient for
the real 16-by-512 RGB8 resident working set and caused repeated Metal
allocation. Commit `3fd23cf2` supersedes it: the private record ceiling is
4,096, derived above the default 3,083-record resident chunk (three components,
per-component DWT scratch, base batch owners, and four classic split-token
owners), while the independently derived shared ceiling remains 64. A flat
fallible `VecDeque` preserves deterministic oldest eviction. More importantly,
the old `(usize, Buffer)` convention is replaced by a move-only
`PooledBuffer`: allocator-reported capacity is validated once when the Metal
allocation enters the typed lifecycle and cannot disagree when that owner is
later recycled. This preserves actual-capacity accounting without an
Objective-C length query for every warm recycle.

The committed source passes 265 Metal library tests with the established 18
release-lane ignores, 54 device tests, all integration/doc targets, 22 support
tests, warning-denied all-target/all-feature Clippy, all 411 ordinary policies,
stable API, unsafe audit, and the unchanged panic ratchets. Alternating
three-process comparison against the benchmark harness at `0e78229a` records
candidate versus baseline medians of 8.199/8.458 ms for direct decode,
7.952/8.520 ms for resident-buffer encode, and 8.083/9.607 ms for resident-host
encode. All deltas are improvements and pass the unchanged 5% limit.

Product remediation is source-complete on the local Apple host. The two nested
`HashMap<usize, Vec<Buffer>>` owners are replaced by separate mutex-protected
flat private/shared ledgers with independent 256 MiB byte limits and the 4,096
private/64 shared record limits described above, each clamped by the device's
maximum buffer length. Admission starts from the once-validated capacity held
by `PooledBuffer`, reserves metadata fallibly, evicts the oldest completed
buffers deterministically, and treats an oversized buffer or cache-only
metadata failure as a safe non-retention decision. Poisoned state and a
requested new Metal allocation retain their existing typed hard-error paths.
Public session diagnostics expose current allocator-capacity ownership, peaks,
evictions, rejections, and metadata failures without exposing buffers.

Focused evidence is six serialized real-Metal tests covering completed command
work before exact-size reuse, actual-byte take/recycle accounting, unique-size
eviction under both limits, exact/oversized and injected metadata-failure
admission, typed size mismatch, private/shared isolation, and public high-water
diagnostics. Three repository source/structure ratchets also pass. After
METALCACHE-001 reached a stable source boundary, the combined package passed
255 library tests and 54 device tests plus all non-benchmark integrations and
docs; its 18 library ignores remain the exact release-Metal inventory rather
than silent skips. Warning-denied all-target/all-feature Clippy,
warning-denied no-default library Clippy, warning-denied rustdoc, focused policy,
formatting, and diff hygiene pass. METALPOOL-001 is complete; immutable-source
release stress remains under FINAL-001 rather than reopening this product fix.

### CUDAPOOL-001 — byte-bounded completed CUDA buffer retention

The equivalent cache/pool sweep found that the public
`j2k_cuda_runtime::CudaBufferPool` retains every successfully completed device
buffer returned to it. Both first-fit and sorted size-bucket modes grow their
metadata fallibly, but neither has a retained device-byte, buffer-count, or
distinct-size limit, eviction policy, or cached-byte diagnostic. A caller can
therefore take many simultaneous buffers or cycle through distinct sizes and,
after every allocation is safe to release, leave the pool holding a monotonic
fraction of device memory indefinitely. This is P1 service-availability debt;
the existing CUDAERR-001 completion quarantine is correct and must not be
weakened to fix retention.

Required closure:

1. Add explicit per-pool completed-cache limits for actual device bytes,
   buffer count, and—when using size buckets—distinct bucket count. Public
   construction uses safe defaults; any configurable limits are validated and
   visible in the API rather than environment-driven. Cache accounting uses
   `CudaDeviceBuffer::byte_len()`, checked arithmetic, and one common ledger for
   first-fit and size-bucket modes.
2. On recycle after completion is established, admit or evict deterministically
   under the limits. An over-limit buffer is safely dropped, not leaked, and a
   metadata reservation failure drops the completed buffer while returning the
   typed host-allocation error. Preserve reuse quality with a documented
   best-fit/size-aware policy instead of allowing a single oversized buffer to
   crowd out all useful entries.
3. Do not evict or free buffers in `deferred` state while a reuse hold protects
   possibly queued work. Track deferred actual bytes/count for diagnostics and
   overflow checking; after the final hold establishes completion, transfer
   each buffer through the bounded completed-cache admission path. Preserve the
   existing intentional leak/quarantine behavior only when completion or
   lifetime safety is genuinely uncertain.
4. Expose typed cached/deferred count and byte diagnostics so release stress can
   assert a stable high-water. Clone of the pool remains a cheap `Arc` alias to
   the same limits/ledger; it must not create an independent unbounded cache.

Acceptance requires hardware-neutral retention-policy tests for repeated size,
many unique sizes, oversize decline, deterministic eviction, first-fit versus
bucket parity, metadata reservation failure, checked byte overflow, nested
reuse holds, final-hold bounded transfer, and Arc aliasing; a source policy that
forbids unbounded completed-cache push/insert; strict runtime checks/Clippy;
and exact-source NVIDIA stress showing cached bytes/count reach a stable high-
water without premature release, leak, or performance-route regression.

Closure evidence on 2026-07-11: first-fit and sorted size-bucket caches now
share immutable limits for actual `CudaDeviceBuffer::byte_len()`, buffer count,
and distinct size count. Admission rechecks the shared state under the mutex,
then releases deterministic oldest/largest victims only after unlocking.
Oversized completed buffers are not retained; metadata failures preserve their
typed error while completed allocations drop outside the lock. Deferred
buffers have checked actual-byte/count high-water accounting and do not enter
the bounded cache until the final reuse hold establishes completion. Pool
clones expose the same `Arc`-shared diagnostics, including cached/deferred
bytes, peaks, evictions, rejections, and metadata failures.

Hardware-neutral exact/one-over, disabled/oversize, checked-ledger, and nested
hold tests passed with the CUDA runtime check and strict Clippy. With
`J2K_REQUIRE_CUDA_RUNTIME=1`, three device tests passed on the supplied RTX
4070 SUPER: first-fit actual-byte/count high-water, best-fit deterministic
bucket eviction, and oversize deferred retention followed by bounded
post-completion rejection. CUDAPOOL-001 is complete; final frozen-tree package
and performance gates remain under FINAL-001.

### CUDAPIN-001 — bounded page-locked CUDA upload staging

The post-pool owner sweep found a separate P1 host-memory retention path:
`CudaContext::upload_pinned` and `CudaBufferPool::upload_pinned` created a new
`cuMemHostAlloc` allocation for each upload and freed it afterward without a
bounded reuse owner, aggregate host-cap admission, or a transaction type that
could safely survive early returns and unwinding. Adding reuse without explicit
ownership would also risk losing or freeing a raw page-locked token after a
compound upload/recycle failure.

Required closure is one clone-shared per-context operation gate and pool with:

1. a 512-MiB current page-locked aggregate cap plus default completed-retention
   limits of 64 MiB and eight buffers; actual allocation lengths, best-fit take,
   and deterministic largest-oldest eviction
2. a move-only RAII checkout that owns the raw token until consuming upload or
   recycle; abandoned/unwound and failed-release tokens enter pre-reserved
   uncertain quarantine, every later operation fails closed, and a successful
   nonzero CUDA allocation must produce a non-null pointer before safe Rust can
   construct a slice over it
3. both public pinned-upload routes using the same transaction and preserving
   the primary plus release error without early pool reuse
4. exact current/peak diagnostics whose documentation distinguishes completed,
   quarantined, and checked-out bytes; adapter admission uses the post-checkout
   pool-plus-checkout total exactly once
5. focused source ownership below the existing `operations.rs <275` and
   `pool.rs <400` ceilings, with semantic policy checks rather than stale text
   matching or widened ratchets

Closure evidence on 2026-07-12: page-locked growth now reserves exact bytes in
a context-owned authority before `cuMemHostAlloc` and releases the charge only
after confirmed free. Every adapter session registers its complete cache plus
active-owner graph with that same authority. Cache replacement and actual-
capacity host allocation use an unlocked full-headroom RAII reservation: peer
sessions and external context clones see the provisional charge and fail before
allocation, while caller work never executes under the authority mutex.
Pre-bind owners migrate transactionally when a context binds; wrong-context
guards reject; early returns/unwinds roll back; and compound operation plus
accounting failures preserve both typed sources.

The independent-session/same-context race, context-clone pinned growth,
owner-outlives-context, pinned-first/owner-first, exact/one-over, growth/shrink,
drop, rollback, quarantine, null allocation, and non-reentrant replacement
regressions pass. The runtime pinned lane passes 21 tests, context authority 6,
JPEG CUDA all-feature library 50, and no-default library 12. The combined seven
affected packages, workspace warning-denied Clippy, strict Clippy, 411 ordinary
repository policies, and strict rendered-API policy pass. `operations.rs` and
the context authority are responsibility-split below their existing ceilings;
the unsafe inventory follows the focused `operations/growth.rs` owner.
Exact-source NVIDIA execution remains under FINAL-001.

### CUDAHANDLE-001 — validate successful CUDA out-parameters

The pinned-staging soundness review found the same unchecked FFI shape at the
remaining CUDA creation boundaries. A successful `cuCtxCreate_v2`, nonzero
`cuMemAlloc_v2`, `cuModuleLoadData`, `cuModuleGetFunction`, `cuEventCreate`, or
test-only `cuStreamCreate` currently promotes the out-parameter directly into a
safe Rust owner. The CUDA contract should not return a null or zero resource on
success, but the safe abstraction must validate that postcondition instead of
encoding it only in a safety comment. Pinned host memory is P1 because a null
pointer reached `from_raw_parts`; CUDAPIN-001 closes that immediate soundness
path. The remaining handles are P2 correctness and RAII hardening.

Required closure is one focused typed validation boundary for pointer handles
and nonzero device allocations, invoked before construction of every safe
owner. A null function after successful module load must take the existing
module-unload rollback path and preserve both lookup and unload diagnostics.
Empty device buffers may retain the documented zero sentinel, but a nonempty
buffer must never do so. Pure regressions must cover each validator and the
partial module/function transaction; source policy must enumerate the creation
sites so a new unchecked promotion fails. Do not widen raw-handle visibility or
add a parallel wrapper hierarchy solely for the policy scanner.

Closure evidence on 2026-07-12: nonzero device allocations and successful
context, module, function, event, and test-stream creation validate their
out-parameters before safe owner construction. A successful null function
takes the existing unload transaction and retains compound rollback failures.
The pure validator/rollback tests, focused source policy, 282 runtime library
tests, and warning-denied all-target/all-feature Clippy pass on the host tree;
exact-source NVIDIA execution remains part of FINAL-001.

### METALALLOC-001 — bounded Metal coefficient-transcode workspace

The equivalent-pattern scan found that coefficient transcode validated logical
block grids but then derived sparse weights through dense quadratic matrices,
materialized whole Rayon batches, staged full `f64` grids through temporary
`f32` vectors, duplicated full readbacks during `f32`-to-`f64` conversion, and
called infallible Metal buffer constructors. A cap-valid caller input could
therefore amplify into uncapped host/device workspace or abort on allocation;
runtime and resident-handoff failures also lost their backend diagnostics.
This is a P1 service-availability issue.

Product remediation and the serialized local package evidence are complete;
the frozen release-candidate matrix remains under FINAL-001:

- all caller-derived host vectors and Metal buffers share the 512 MiB core
  policy, use checked products, reject before session/runtime initialization,
  reserve fallibly, and return typed host/device allocation errors; Metal
  buffer creation uses a nil-checked Objective-C boundary
- reversible IDCT batches use a fallible shared Rayon helper plus bounded
  chunks, DCT upload uses fixed-size chunks, and readback materializes one
  bounded item at a time without a full `f32` plus full `f64` batch duplicate
- 5/3 and 9/7 sparse rows are built in linear work from fixed-capacity symbolic
  lifting rows instead of a dense square matrix; dense reference rows are also
  capped/fallible and exact sparse-versus-dense parity remains covered through
  2,048-sample axes
- projected, reversible, and prequantized outputs preflight coefficient and
  metadata counts; code-block growth and copies are fallible; driver,
  command-buffer, checked-buffer, and resident-handoff diagnostics retain the
  operation and backend detail. Auto still falls back only for unavailable or
  unsupported work, while allocation/runtime failures remain hard errors.
- prequantized code-block readback and assembly share one pre-launch/readback
  aggregate formula covering retained final coefficients, one current
  component's four readbacks, all code-block/component/resolution/subband
  metadata, and the bounded readback chunk; component/resolution/subband arrays
  use exact fallible reservation rather than nested `vec!`
- the former 875-line crate root is a 126-line facade over 230-line route and
  312/213-line accelerator owners; reversible, irreversible, geometry, buffer,
  resident, code-block-output, and weight owners are all below their existing
  or new-child 425-line ceilings without raising an established ratchet

Verified on the local Apple M4 Pro/Metal 4 host: the all-feature package suite
passed 51 tests with explicit 5/3, reversible 5/3, staged 9/7, code-block, route,
resident-handoff, shader, and JPEG-to-HTJ2K hardware paths; focused adversarial
tests reject overflow, over-cap grids/batches/code-block metadata, and a sparse
axis beyond the cap before allocation, while a 16,384-sample sparse axis builds
without dense workspace. Package-isolated all-target/all-feature Clippy passes
with warnings denied. Dependency-inclusive Clippy and combined repository policy
are rerun after the concurrently edited shared transcode and Metal cache crates
settle; FINAL-001 still owns frozen-source combined evidence.

Post-closure focused evidence is 12/12 library tests, 2/2 bench-harness tests,
and 11/11 DCT53 integration tests, plus an all-feature/all-target package check
and scoped formatting/diff hygiene. The DCT97 binary was stopped before test
execution during the known concurrent macOS loader stall, so no result is
claimed for it; root owns the serialized rerun, final strict Clippy, and the
combined structure/allocation policies.

The same post-handoff red-team found two Metal follow-ups. Float batch
conversion moves coefficient vectors without copying them, but its exact phase
model omitted the simultaneously live final
`Vec<Dwt97TwoDimensional<f64>>` outer metadata while the source
`Vec<ProjectedBands>` metadata remains allocated. Current device/chunk bounds
appear to dominate the missing term, so this is P2 latent accounting debt rather
than a demonstrated cap bypass. The working tree now models readback and
conversion as distinct peaks, counts both outer metadata arrays during
conversion, and has an exact boundary regression. Separately, host cap and
allocator failures now map directly to `TranscodeStageError::MemoryCapExceeded`
and `HostAllocationFailed` without allocating a backend string. Nine focused
source-policy assertions cover both fixes; product compile/runtime tests were
initially blocked behind the active native dependency extraction.

The dependency extraction then returned to a coherent state. A fresh
dependency-inclusive `cargo check -p j2k-transcode-metal --all-targets
--all-features` passed, and `cargo test -p j2k-transcode-metal --lib` passed
15/15, including both typed host-error mappings and the simultaneous outer-
metadata boundary. The exact Metal allocation/error source policy passes 9/9.
The dependency-inclusive warning-denied Clippy gate and the complete serialized
Metal package now pass; no product failure is open in this lane.

Final local closure on 2026-07-11: the all-feature package passed 58 tests
(18 library, 2 benchmark-harness, 11 DCT53, 11 DCT97, 9 JPEG-to-HTJ2K, 6 route,
and 1 shader-integrity), all-target/all-feature Clippy passed with warnings
denied, and 12 focused allocation/structure policies passed. The serialized
run also caught a stale benchmark-harness marker and a second infallible
profile-row `format!`; the harness now matches the emitted schema and the
prefix is streamed after bounded fallible field formatting without allocating
a duplicate row string. METALALLOC-001 is complete.

### PROFILE-001 — bounded, fallible profile text and summaries

The equivalent-pattern owner sweep found a public diagnostics-only allocation
seam in `j2k-profile::ProfileSummary`. Callers can currently supply arbitrary
codec, operation, path, label, label-value, and numeric-field strings and an
arbitrary number of distinct rows. Recording copies those values into nested
`BTreeMap`, `Vec`, and `String` owners through infallible `collect`, `entry`,
`to_owned`, and `push` operations; formatting constructs another unbounded
`Vec<String>`. The type also implements a deep `Clone` over the complete
summary even though the only workspace clone is a crate-local test of that
implementation. The equivalent public `parse_profile_key_value_fields` and
`format_profile_key_value_fields` boundaries also return unbounded owned
collections/strings through infallible iteration and formatting; parsed field
owners derive deep `Clone`. This is P2 service-availability/API debt because
profiling is explicitly opt-in and is not on a normal decode or encode path,
but the public contract must not silently retain an unbounded text or
cardinality sink.

Required closure:

1. Replace map-backed rows and numeric sums with deterministic, small
   Vec-backed owners that reserve fallibly and account allocator-reported
   capacity. Define explicit per-summary row, label, numeric-field, and retained
   text-byte limits; reject a new distinct key before mutating state when any
   limit would be exceeded. Existing-key counter updates remain allocation-free.
2. Add a non-exhaustive typed profile allocation/input error distinguishing
   invalid limits/input, size overflow, configured-cap exhaustion, and
   allocator failure. Fallible label/summary constructors, record methods,
   public key/value parsing/formatting, and summary formatting/take methods
   return that source instead of aborting, truncating, or dropping a row.
   Internal emission helpers must surface a diagnostic error explicitly without
   turning optional profiling failure into codec success/failure ambiguity.
3. Preflight and reserve every owned key/value and formatted output before
   writing it. Formatting must not rely on `write!(String).expect(...)` or an
   underestimated buffer that can reallocate after validation. Clearing or
   draining formatted rows must preserve a coherent summary when formatting
   fails partway through.
4. Remove `Clone` from `ProfileSummary`, parsed owned field sets, and unbounded
   nested owner types. Keep small copyable modes and borrowed views copyable. If
   callers need a snapshot, expose an explicitly fallible, capped operation
   whose cost is visible in the name and error type.
5. Adapt all workspace profiling call sites and public documentation. Because
   this changes public return types, regenerate the API snapshot and semver
   report and record the 0.7 migration in the changelog.

Implementation checkpoint (2026-07-11): the profile owner itself is now
source-complete. Parsing, formatting, typed fields, summary labels, summary
records, and transactional format/take operations use explicit `ProfileLimits`
and the non-exhaustive `ProfileError`. Summary rows and numeric sums are sorted
Vec owners with allocator-reported capacity accounting; existing-key updates
with an unchanged numeric schema allocate nothing. Deep `Clone`, `BTreeMap`,
infallible string formatting, saturating profile totals, and silent malformed-
token truncation are removed. `TranscodeBatchProfileRow` is move-only and
fallible with the same error source. The profile package passes 35 all-feature
tests plus a no-default-features check, and native, JPEG, and xtask consumers
check successfully. CPU JPEG callers now build fixed typed `ProfileField`
arrays inside the env-gated emitter instead of preformatting durations,
counters, and labels into infallible owned strings; duration formatting uses
the same bounded fallible numeric contract. Profile construction, summary
initialization, and drop-time failures route through the shared explicit
diagnostic helper.

Metal/CUDA caller checkpoint (2026-07-11): J2K Metal batch, direct, hybrid-plan,
and route rows plus JPEG Metal fast-batch/route rows now construct fixed typed
`ProfileField` arrays. Debug values stream through `Display` wrappers; no caller
preformats backend, pixel-format, duration, count, stage, or label values with
infallible `format!`, `to_string`, `collect`, or deep clones. The decode-label
environment boundary sanitizes through a bounded writer, reports non-Unicode
and limit failures explicitly, and cannot silently grow an unbounded token.
Batch-result slot context is a fallible, aggregate-budgeted owner; optional
profile allocation failure emits the shared diagnostic and does not change the
codec result. JPEG CUDA and both Metal adapters use that one diagnostic helper,
so its format has one implementation. The focused profile policy passes 8/8,
the complete Metal policy filter passes 79/79, `j2k-profile` passes 35/35, J2K
Metal passes 238 with 18 established runtime ignores, JPEG Metal passes 195/195,
and JPEG CUDA passes 15/15. Warning-denied all-target/all-feature no-dependency
Clippy passes for all four product crates and `xtask`. Frozen-source API/semver
report regeneration and the combined final-tree matrix remain before this item
can be marked complete.

Acceptance requires exact-limit and limit-plus-one tests for parsed fields,
rows, labels, numeric fields, text bytes, and formatted output;
allocator-overcapacity and synthetic reservation-failure coverage;
duplicate-key aggregation without new allocation; parse rejection without
partial success; formatting rollback/drain behavior; `no_std` compilation; a
source policy forbidding `BTreeMap`, infallible collection growth, formatting
expects, and deep `Clone` in this owner graph; full `j2k-profile` and affected
profiling integration tests; strict Clippy; API/semver review; and the
serialized final matrix.

### CUDASESSION-001 — context-bound resident encode session ownership

The public CUDA-resident tile path previously accepted a `CudaSession` for
submission accounting while each tile uploaded a fresh HTJ2K encode table
resource set from the tile buffer's context. That made the session a counter,
not the resource owner promised by the API, and provided no early diagnostic
when a caller mixed buffers from independent CUDA contexts.

Required closure:

1. Bind an uninitialized session to the first resident tile's `CudaContext`,
   cache one `Arc<CudaHtj2kEncodeResources>`, and reuse it across compatible
   single and batch submissions.
2. Expose only the minimum runtime identity primitive needed across crates:
   whether two `CudaContext` handles share the same driver context. Do not
   expose raw driver handles or identifiers.
3. Reject a tile from another context before cache initialization, upload, or
   kernel launch. Preserve the existing per-attempt submission count, decode
   table cache, decode buffer pools, session cloning, and retry after a failed
   first upload.
4. Replace the resident-input constructor's raw string validation result with
   the non-exhaustive `J2kResidentEncodeInputError`, covering empty geometry,
   component range, precision range, and address-space overflow. Keep stable
   `reason()` text and map it explicitly at adapter/string-SPI boundaries.
5. Add driver-independent cache-state tests, typed-constructor tests, source
   policy, public batch reuse coverage, cross-context NVIDIA behavior coverage,
   strict feature/no-feature Clippy, and exact-source CUDA verification.

Status: host implementation is complete and final exact-source NVIDIA evidence
is pending. `CudaSession` now retains the first tile context and one shared
encode-resource allocation; the mismatch guard precedes both cache reuse and
initialization. Compatible batch submissions record each attempt while sharing
one upload. The context identity API is doc-hidden and compares only shared
runtime ownership. Decode caches and accounting are unchanged. Failed resource
initialization leaves the session bound but uncached so a compatible retry can
succeed.

The 2026-07-11 final host checkpoint also covers the two ownership cases that
the original device tests did not distinguish: a fresh session binds to a
context supplied by the tile rather than manufacturing its own, and a cloned
bound session shares the cached `Arc<CudaHtj2kEncodeResources>`. Session-specific
device tests are isolated from general resident encode behavior in a 168-line
real module; the source policy requires the external-context/clone, batch
reuse, cross-context, and driver-independent retry regressions. All-feature and
no-default strict Clippy plus the complete host CUDA package are green. The
dashboard remains in progress only for root's frozen-source NVIDIA batch.

ERR-016 subsequently replaced the static-string accelerator SPI with
`J2kEncodeStageResult<T>`. Resident constructor failures now cross that
boundary as typed invalid-request errors, backend failures retain their
concrete sources, and native resident orchestration distinguishes an ordinary
decline from every hard stage failure without parsing `reason()` text.

Subsequent output-contract work made `to_cuda_buffer` metadata-only on the host:
the assembled host codestream is uploaded and then dropped, so the outcome no
longer retains duplicate host and device payloads. Host assembly is still
transiently required; eliminating it requires the real device tier-2 contract
described above and is not part of the session-cache closure.

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
The facade owns the one conversion into `J2kError`; sibling GPU adapters must
delegate rather than duplicate or stringify it.

Constraints:

- keep native implementation modules out of facade re-exports and rendered
  public documentation; the typed `NativeDecode` source added by ALLOC-003 is
  the deliberate diagnostic exception
- do not add an internal dependency cycle
- keep the sibling-adapter conversion as a documented `#[doc(hidden)]` SPI and
  verify that it is absent from rendered `cargo-public-api` output
- forbid direct matching on native inner variants outside the classifier

Add golden parity tests proving equivalent CPU, CUDA, and Metal classification
and message behavior.

### ERR-002 — GPU adapters preserve native decode resource sources

The 2026-07-11 adapter review found that the main facade retained
`AllocationTooLarge` and `HostAllocationFailed` as typed heap-free native
sources, while CUDA and Metal duplicated the older classifier and converted
every remaining error to an allocated generic backend string. This erased the
resource category on the GPU routes that most need allocation diagnostics.

The facade mapper is now the single doc-hidden sibling-adapter SPI. CUDA and
Metal delegate to it; their `DecodeErrorClass` imports, local matches, and
`error.to_string()` catchalls are gone. Both adapters contain resource-source
parity regressions, and repository policy requires delegation while forbidding
category duplication/erasure. The focused policy, central facade resource
test, CUDA library check, strict CUDA no-deps Clippy, and rendered-public-API
check pass. Metal CPU batch fallback also now preserves non-resource
scheduler failures through the new transparent public
`Error::BatchInfrastructure` variant rather than formatting them as a kernel
string.

Closure checkpoint (2026-07-11): the post-split
`native_decode_resource_errors_preserve_typed_sources` regression passes in
both `j2k-cuda` and `j2k-metal`. The central facade parity regression and
`native_decode_error_mappers_delegate_to_the_facade_classification` repository
policy pass, as do the four focused architecture/error policies and
warning-denied all-target/all-feature no-dependency Clippy for both adapters.
No adapter-local native classifier or rendered resource catchall remains.
Frozen-candidate public-API and hardware reruns stay under SEM-001/FINAL-001
rather than keeping this implementation item open.

### ERR-003 — explicit transcode errors and checked batch result ordering

The 2026-07-11 transcode boundary review found a public blanket
`From<&'static str>` implementation that classified every legacy helper error
as `Unsupported`. It also found parallel JPEG-to-HTJ2K batch assembly writing
directly to `tile_results[index]`, permitting duplicate results to replace an
earlier outcome and assigning missing results a generic validation error.

The blanket conversion is gone. The one production reversible-grid boundary
and the three integration-test accelerator boundaries now construct
`TranscodeStageError::Unsupported` explicitly. A focused private
`BatchResultSlots<T>` owns fallibly allocated ordered slots, bounds-checks each
insert, rejects duplicate writes, and requires every slot at completion.
Missing, duplicate, and out-of-range worker results now produce the public
`JpegToHtj2kError::InternalInvariant` category, while allocation-cap and host
allocator errors retain their existing typed resource variants. Repository
policy forbids both the blanket conversion and direct result-slot indexing.

The all-feature transcode check, strict all-target no-deps Clippy, 55/55
library tests, 37/37 JPEG-to-HTJ2K integration tests, 4/4 result-slot tests,
the explicit reversible-grid taxonomy test, and both focused repository
policies pass; scoped diff hygiene is clean. Removing the public conversion and
adding an enum variant both affect downstream source compatibility. The
changelog now records both; they must still be captured in the frozen 0.7
API/semver review.

### ERR-004 — explicit CUDA packetization fallback errors

The residual error scan found the workspace's last blanket production string
conversion in the private CUDA HTJ2K packetization planner. Dozens of `?`,
`ok_or`, and `.into()` sites could silently assign any new static-string helper
failure to `Invalid`, whose stage behavior is CPU fallback, while only a
separate allocation mapper produced a hard failure.

Every flatten, state, and tag-tree crossing now constructs
`CudaHtj2kPacketizationPlanError::Invalid` explicitly. Fallible host ownership
continues to map only to `HostAllocation`. One stage classifier makes the
policy visible: invalid plans emit their stable decline reason and return
`None`; allocation failure propagates as an error and cannot retry on CPU.
Behavior tests pin both routes, and repository policy forbids recreating the
blanket conversion or raw string propagation. Two resident-encode policies
were retargeted to the current typed no-host-input validation boundaries after
review confirmed policy drift rather than a production regression.

All-target CUDA check and warning-denied Clippy pass. The host package passes
86 tests across targets, the packetization filter passes 15, and the complete
resident policy filter passes 8/8. Formatting and diff hygiene pass. The error
type is private, so there is no public API consequence.

### ERR-005 — explicit facade errors and Rust-quality closure

The facade audit found blanket public `From<String>` and `From<&str>`
implementations that silently assigned every adapter failure to
`BackendErrorKind::Other`. Both conversions are gone; downstream code must use
`BackendError::new` or a typed convenience constructor. Repository policy
requires the explicit constructors and forbids reintroducing error-erasing
string conversions. This is a deliberate pre-1.0 source-breaking API change
that belongs in the frozen public-API and semver review.

The same lane removed nineteen strict facade Clippy failures without adding
allowances. Component validation is split by responsibility, duplicate match
arms and stale lint expectations are gone, test fixtures are focused, and the
958-line `view.rs` owner is now a 504-line decode core over 301-line row and
180-line trait/batch modules with structural ratchets. Semantic symbol review
confirmed those boundaries follow decode responsibility rather than arbitrary
line-count partitioning.

Recoverable empty batch plans now return the typed public
`BatchInfrastructureError::EmptyBatchPlan` variant instead of relying on a
debug assertion. Compile-time layout invariants use type-level equality proofs,
and the redundant lossy-target debug assertion is gone. The new enum variant is
an additive public API change and has a behavior regression.

Full recode reconciliation exposed a real sampled-component regression: pixel
fallback removed JP2 palette/component mapping metadata without first
materializing the resolved components to the reference grid. A focused,
budgeted grid materializer now expands resolved planes while preserving each
plane's precision and signedness before metadata removal. All 19 recode
integration tests pass.

The all-target/all-feature facade check and warning-denied no-deps Clippy pass.
The full facade suite passes 323 tests with one established environment-gated
ignore and no failures; the focused native allocation extraction passes 6/6.
Facade behavior, allocation, error-classification, recode, view-structure, and
strictness policies pass. Final frozen-tree workspace/API/semver evidence still
belongs to FINAL-001.

### ERR-006 — typed JPEG Metal batch failure replication

The JPEG Metal batch coordinator formerly rebuilt only a few routing/runtime
variants and converted every other group failure with `format!` into
`Error::MetalKernel`. A decoder truncation, encoder allocation failure, or
caller buffer error therefore lost both its variant and shared codec-error
classification when the same failed batch result was assigned to multiple
output slots.

`JpegEncodeError` and `j2k_jpeg_metal::Error` are now cloneable, and the batch
coordinator clones the original typed error directly for each affected slot.
Unit coverage locks decode, encode, buffer, backend-routing, and unavailable
variants; repository policy requires direct typed cloning and forbids the old
string-rendering helper. Both `Clone` implementations are additive public API
surface. The three focused batch regressions, the focused repository policy,
and warning-denied all-feature library/test Clippy for `j2k-jpeg` and
`j2k-jpeg-metal` pass in an isolated target. Frozen-tree Metal hardware
evidence still belongs to FINAL-001.

### ERR-007 — direct J2K Metal resident-batch error propagation

The resident Metal submit coordinator accepted a `crate::Error` from its own
batch preparation helper, rendered that error with `format!`, and constructed a
new `Error::MetalKernel`. That same-type wrap discarded non-kernel variants and
their typed codec/buffer classification without adding a recoverable boundary.
The coordinator now uses `?` to propagate the original error. The injected
failure regression asserts the exact original variant/message, and a focused
source policy forbids restoring the same-type rendering wrapper.

The product edit and source policy are formatted and pass scoped diff hygiene.
The exact injected-failure regression and focused repository policy now pass.
The subsequent combined all-target/all-feature Metal check and warning-denied
no-deps Clippy also cover this path and pass, so ERR-007 is closed. Frozen-tree
Metal hardware evidence remains part of FINAL-001 rather than this error-
propagation contract.

### ERR-008 — typed tile-codec I/O sources

The tile-codec error audit found that its only `Backend(String)` constructor
rendered a non-input `std::io::Error` together with operation context. The
`malformed_io_error` and `input_or_backend_io_error` helpers also rendered
`InvalidData`/`InvalidInput` errors into messages, discarding `ErrorKind` and
the source chain. No production caller outside those helpers constructed or
matched the string variant.

`TileCodecError` is now non-exhaustive. `Io { context, source }` owns
operational decoder/encoder I/O failures, while the existing `Malformed`
category owns the original typed source for malformed I/O. LZW statuses do not
provide an error object, so they become explicit `InvalidData` sources rather
than operational I/O failures. `UnexpectedEof` still becomes the shared
`InputError::TruncatedAt` category, and buffer/unsupported classifications are
unchanged. The unused `Backend(String)` variant and all production I/O error
stringification are gone. A repository policy locks owned handoff, typed
sources, classification branches, and behavior coverage.

The package library check, 5/5 error-source unit tests, all 12 tile-codec
integration tests, doc tests, focused repository policy, warning-denied
library/test no-deps Clippy, targeted formatting, and diff hygiene pass. The
changelog records the deliberate pre-1.0 enum/field change; the frozen public
API snapshot and semver review remain part of FINAL-001.

### ERR-009 — typed Metal support and prepared-cache crossings

The residual Metal error inventory found shared runtime, command-completion,
buffer access/readback, and benchmark operations formatting
`MetalSupportError` into `MetalRuntime` or `MetalKernel` strings. Direct
surface, resident codestream, host-fallback, and plane-upload routes repeated
the same erasure. The JPEG readback mapper additionally converted an element
count into a byte count with saturating multiplication, which could report a
fabricated `usize::MAX` request. Prepared-plan cache insertion in both session
and hybrid routes rendered allocation and invariant failures into one generic
kernel string.

`j2k_metal::Error` and the still-`Clone` `j2k_jpeg_metal::Error` now carry a
public `MetalSupport` variant with the original typed source and the existing
operation-specific rendered diagnostic. Shared support failures remain
`AdapterErrorKind::Other`; runtime/device unavailability still takes the
existing `MetalUnavailable` unsupported route, and the one host-fallback
operation that deliberately treats a private input buffer as unsupported still
uses `UnsupportedMetalRequest`. This preserves public fallback behavior while
making internal readback defects source-inspectable rather than accidentally
fallback-eligible.

The J2K adapter separately exposes `PreparedPlanCacheAllocation`, retaining the
original `TryReserveError` source without inventing a requested byte count, and
`PreparedPlanCacheInvariant`, retaining the cache's static reason. One central
mapper serves direct, session-region, and global-region cache routes. No
private prepared-cache implementation type appears in the public error API.

Both all-target/all-feature package checks and warning-denied no-deps Clippy
pass. The complete J2K Metal library suite passes 221 tests with 18 established
Metal-runtime ignores; the complete JPEG Metal library suite passes 179/179.
Focused support, clone, source-chain, classification, cache-category, and
no-fake-byte-count regressions pass, as do both focused repository policies.
The changelog records the additive pre-1.0 public variants; frozen public-API
and semver evidence remains part of FINAL-001. The formerly concurrent CUDA,
restart-policy, and policy-size lint debt is now reconciled; warning-denied
all-target/all-feature no-deps xtask Clippy passes on the settled source.

### ERR-010 — typed baseline JPEG DCT re-emission input contract

The public `j2k_jpeg::transcode::encode_baseline_dct_image` boundary accepted
caller-constructed `JpegDctImage` values but classified invalid coding modes,
component layouts, grids, and quantization tables as
`JpegEncodeError::Internal(String)`. Its max-only sampling check also missed a
zero factor on one component when another component kept the maximum nonzero,
and it never rejected factors above JPEG's permitted range. Unchecked grid
multiplication occurred before the shared frame-dimension validation, so
adversarial safe input could produce an invalid marker, wrap in optimized
builds, or panic in checked builds instead of receiving an input error.

The boundary now builds one pure validated re-emission plan before capacity
planning, entropy coding, or frame assembly. Public non-exhaustive
`JpegDctImageError` variants distinguish coding mode, empty/oversized frame
dimensions, component count/order, each horizontal and vertical factor, the
canonical grayscale 1x1 shape, ten-block MCU sum, checked grid arithmetic, grid
shape, quantized block count, baseline 1-through-255 quantization values, and
the current shared chroma-table limitation. It also enforces the baseline
DC-difference category maximum of 11 and AC magnitude-category maximum of 10.
Without the AC check, a category-16 value after fifteen zeros aliases the
`0xF0` zero-run-length symbol while still writing sixteen magnitude bits,
producing a malformed entropy stream. These constraints follow the frame,
scan, Huffman entropy, and DQT requirements in
[ITU-T T.81](https://www.itu.int/rec/T-REC-T.81-199209-I/en). A workspace-wide
owner audit found no production constructor or supported downstream seam for
`JpegEncodeError::Internal(String)`, so the pre-1.0 variant is removed rather
than retained as zombie API. Impossible encoder states use allocation-free
`InternalInvariant { reason: &'static str }`; caller-invalid coefficient input
uses `InvalidDctImage { reason }`. The three former test-only constructors now
exercise the typed static invariant instead.

Validation deliberately ignores metadata that the encoder does not consume:
the detected color-space label, source scan count, restart index, native sample
dimensions, dequantized blocks, and other extraction-only state. A parity
regression mutates all of those fields and requires identical output bytes.
Pre-change grayscale and 4:2:0 output lengths/FNV fingerprints remain exactly
323/`eb283c65094ab76a` and 661/`44dadc89e9272c00`; decoded quantized component
parity also remains exact. Checked-in canonical hex goldens additionally require
byte-for-byte output identity. Validation reuses the encoder's tested
magnitude primitive so category classification cannot drift from emission.

The nine focused public-contract tests, full all-feature `j2k-jpeg` package
suite, warning-denied all-feature no-deps library/test Clippy, and focused
source policy pass in the isolated target. The source policy also ratchets the
new validation/test owners, enforces validation-before-allocation/entropy
ordering, forbids `Internal(String)` construction in this product family, and
prevents validation from expanding into ignored metadata. The changelog records
both the typed additions and the pre-1.0 removal. The source policy forbids
either the string variant or its constructor from returning; frozen public-API
and semver evidence remains part of FINAL-001.

### ERR-011 — source-preserving resident fallback and recode classification

The final facade string-error scan found two explicit leftovers after blanket
conversions had been removed. `map_native_resident_encode_error` handled every
current native variant structurally but its required wildcard for the public
non-exhaustive source rendered any future variant into `BackendErrorKind::Other`.
Separately, a decoded-sample mismatch in J2K-to-HTJ2K round-trip validation used
the same generic backend constructor even though the failure is specifically
backend-output validation.

`J2kError::NativeResidentEncode { context, source }` is the typed fallback for
future native resident-boundary variants; all current invalid-input,
unsupported, declined, accelerator, resource, and backend variants retain
their narrower existing mapping. Recode sample mismatch now constructs
`BackendErrorKind::Validation`. With those call sites reconciled, the private
generic `J2kError::backend` and `BackendError::native` constructors are removed
rather than left as an attractive error-erasure path.

The complete 73-test facade library suite passes, including source-chain and
real unequal-codestream classification regressions. The focused repository
policy and warning-denied all-feature no-deps library/test Clippy pass in
isolated targets. The additive pre-1.0 public error variant is recorded in the
changelog; frozen public-API and semver reconciliation remains under
SEM-001/FINAL-001.

### ERR-012 — typed scalar code-block adapter failures

The post-ERR-011 scan found four doc-hidden but public native scalar adapter
helpers—classic code-block encode, classic token packing, HT cleanup encode,
and HT encode with explicit passes—still returning `Result<_, &'static str>`.
Their implementations already originate `EncodeError`; the legacy conversion
therefore discards invalid-input, arithmetic, cap, allocator, and invariant
categories. The production Metal classic token-validation path compounds the
loss by formatting that static text into `Error::MetalKernel`.

Closure must change the four scalar helper signatures to `EncodeResult`, keep
valid code-block bytes and segment metadata exact, and preserve the typed
native source plus stable operation context at the Metal crossing. Benchmark
and test-only callers may render diagnostics only at their final assertion or
reporting boundary. Add cap/allocation/category/source-chain regressions, a
source policy forbidding the legacy converter in these public functions, full
native/Metal checks, strict Clippy, and changelog/API review. Do not conflate
the separate fixed-size cleanup-distribution diagnostic helper with this
fallible encode contract.

Closure checkpoint (2026-07-11): all four helpers now return `EncodeResult`, and
their implementations propagate the originating native error without a string
compatibility shim. A deterministic oversized token plan proves the public
token-pack boundary retains `AllocationTooLarge`; invalid classic geometry,
HT bitplanes, and refinement-pass requests retain `InvalidInput`. Native and
facade all-target/all-feature checks, native no-default-features, strict
native/facade Clippy, the focused behavior test, and
`public_typed_helper_error_policy` pass. The J2K Metal production, profile, and
test-support token-pack crossings now retain operation context and the concrete
native `EncodeError`; their source-chain regression and repository ratchet
pass, as do all-target/all-feature check and strict no-deps Clippy. Final
public-API/semver reconciliation remains under SEM-001/FINAL-001 rather than
keeping this implementation item open.

### ERR-013 — remove the public scalar deinterleave panic wrapper

The post-ERR-011 safe-public-function scan found the doc-hidden
`j2k_native::deinterleave_reference` compatibility wrapper calling the checked
implementation with `expect`. Invalid component counts, bit depths, geometry,
or byte lengths could therefore panic through a safe public API. Semantic and
literal reference scans found no caller of the wrapper; every native, CUDA,
and Metal parity consumer already uses `try_deinterleave_reference` and handles
its typed `DecodeError`.

The unsupported pre-1.0 wrapper and both re-exports are removed rather than
retaining two names for one fallible operation. Existing invalid-geometry
behavior tests remain on the checked entry point. The repository policy now
requires that checked implementation and its two re-exports, forbids restoring
the panic wrapper, and keeps accelerator parity consumers on the checked API.
The two focused deinterleave tests and source policy pass. The subsequent full
native package run passed 508 tests with one established ignore, every
integration/doc target, and warning-denied all-target/all-feature/no-deps
Clippy, covering this removal and the marker-reader panic reduction together.
The frozen public-API/semver report remains under SEM-001/FINAL-001.

### ERR-014 — fallible public Metal surface byte access

The safe-public panic scan found that both `j2k_metal::Surface::as_bytes` and
`j2k_jpeg_metal::Surface::as_bytes` call a fallible storage/readback helper with
`expect`. These are not limited to impossible construction invariants. The J2K
path can receive Metal buffer availability and range errors; the JPEG path can
also receive safe-access mutex poisoning and checked readback failures. A
caller can therefore encounter a process panic for an operational accelerator
failure through a safe public method even though both crates already expose a
typed `Error` and `download_into` handles the same source fallibly.

Closure must change both 0.7 byte-access contracts to return typed results,
migrate tests, examples, and benchmarks without inserting replacement unwraps
in production paths, preserve borrowed host storage and owned Metal snapshots,
and retain the exact Metal-support source categories introduced under ERR-009.
Add host-backed success and injected range/readback/access-gate failure tests,
a source policy forbidding `expect` at this public seam, changelog/API migration
notes, strict Clippy, and exact-source Metal verification after source freeze.
Coordinate the surface edits with ALLOC-018 so capacity and error-contract work
do not race.

Closure checkpoint (2026-07-11): `j2k_metal::Surface::as_bytes` now returns
`Result<Cow<'_, [u8]>, Error>` and directly propagates the same checked range
and typed Metal-support failures as `download_into`. Host-backed success proves
the view remains borrowed; an inconsistent host-backed range proves the public
method returns an error without unwinding. All J2K Metal tests and benchmark
reporting callers handle the result explicitly, with one readback per diagnostic
comparison. The focused source ratchet, all-target/all-feature check, 234
passing library tests with 18 established Metal-release ignores, all 54 real
device integration tests, no-default-feature check, and strict no-deps Clippy
pass. `j2k_jpeg_metal::Surface::as_bytes` now has the same fallible contract;
its reusable-output path returns the dedicated `MetalStatePoisoned` category
for a poisoned safe-access gate and retains `MetalSupportError::BufferBounds`
as the source of an out-of-range readback. Host borrowing, gate serialization,
and both injected failures have focused regressions. All 65 JPEG Metal caller
sites handle the result explicitly. The JPEG surface policy, all-target/
all-feature and no-default checks, 188 library tests, 13 batch integrations, 18
core-trait integrations, and strict no-deps Clippy pass. Final public-API/semver
reconciliation remains under SEM-001/FINAL-001 rather than keeping this
implementation item open.

### ERR-015 — typed, single-build JPEG Metal fast-packet routing

The no-silent-failure and duplicate-work scan found production JPEG Metal
decoder, session, viewport, and capability paths calling all three
`build_fast444_packet`, `build_fast422_packet`, and `build_fast420_packet`
functions against the same bytes and immediately applying `.ok()` or
`.is_ok()`. Each builder reparses the JPEG, constructs a CPU decoder, plans
checkpoints, and allocates retained entropy/checkpoint owners. Only one
sampling family can match, so this performs repeated validation/allocation and
then silently converts every error—including `MemoryCapExceeded`,
`HostAllocationFailed`, and `InternalInvariant`—into a missing optional fast
packet. Explicit Metal requests can consequently report unsupported routing,
and Auto can retry CPU work, after a real resource or invariant failure.

Closure must inspect sampling once, build only the matching packet family,
retain/share that one packet through session and viewport routing, and
distinguish ordinary unsupported-capability variants from hard typed decode,
cap, allocator, and invariant errors. Only genuine capability mismatch may
become `None` or CPU fallback. Add injected hard-error classification tests,
single-build/cache behavior evidence, exact 4:4:4/4:2:2/4:2:0 routing parity,
a source policy forbidding production fast-packet `.ok()`/three-builder
probing, strict Clippy, and exact-source Metal verification. Reuse ALLOC-008's
packet allocation contract and coordinate edits with ALLOC-018.

Closure checkpoint (2026-07-11): `j2k-jpeg` now exposes one doc-hidden
metadata classifier for the mutually exclusive 4:2:0, 4:2:2, and 4:4:4 packet
families, including restart-coded 4:2:0. JPEG Metal stores that choice as one
`SharedJpegFastPacket` enum and carries the same owner through decoder,
submission, session-cache, capability, and viewport routes. The central
builder invokes only the selected packet constructor. `FastPacketError` is a
typed source: five ordinary capability variants alone map to `None`; malformed
scan/table/entropy, cap, allocator, nested decode, and invariant failures cross
as `Error::FastPacket` with the concrete JPEG source chain intact.

Behavior verification covers all three families, restart-coded 4:2:0,
unsupported grayscale, all five decline variants, four malformed variants,
three resource/invariant sources, and repeated cache hits. The full
all-feature package runs pass with 389 `j2k-jpeg` library tests and 195 JPEG
Metal library tests, plus JPEG Metal integration groups of 13 batch, 18 core
traits, 5 encode, and 1 host-surface test. All-target/all-feature and
no-default-feature warning-denied Clippy pass for both crates. Two focused
ERR-015 policies forbid production three-builder probing/error erasure, and
the structural policy passes after splitting the packet owner into focused
243-line build, 100-line error, 37-line family, and 264-line ABI type modules.
JPEGCACHE-001 may now add collective byte-bounded admission without changing
the single-owner or hard-error contract. Frozen API/semver reconciliation
remains under SEM-001/FINAL-001.

### ERR-016 — typed public encode-stage accelerator failures

The cross-crate static-string scan found that all 14 fallible methods on the
public, re-exported `J2kEncodeStageAccelerator` trait still return
`Result<_, &'static str>`. This is wider than ERR-012's four scalar helpers.
CUDA and Metal implementations use the trait for deinterleave, color
transforms, both DWTs, quantization, classic and HT Tier-1, fused subbands,
whole/resident tiles, and packetization. Several adapter helpers currently
map any typed backend failure—including host allocation and runtime errors—to
one static literal. Native call sites can then preserve only an operation name
and that literal in `EncodeError::Accelerator`; the original category and
source chain are irrecoverable.

Closure requires one neutral, non-exhaustive, `no_std`-compatible stage error
contract in `j2k-types` and migration of every fallible trait method,
implementation, wrapper, and test double. The contract must distinguish
invalid request, unsupported capability, arithmetic overflow, shared-cap
excess, host allocation failure, backend/runtime failure, and internal
invariant without introducing a blanket string conversion. Backend failures
must retain their concrete source when the type boundary permits it; resource
variants must not allocate merely to report allocation failure. `Ok(None)` or
`Ok(false)` remains the only ordinary capability decline and must never be
used for a hard typed failure. Native `EncodeError` mapping must preserve both
the failed operation and the stage category/source instead of re-rendering it.

Acceptance requires injected cap, allocator, backend, malformed-output, and
ordinary-decline regressions across CPU-only, Metal, and CUDA implementations;
a source policy forbidding `&'static str` on the public trait and literal-only
allocation-error adapters; warning-denied checks and strict Clippy for
`j2k-types`, native, facade, transcode, CUDA, and Metal dependents; and explicit
0.7 changelog, public-API, and semver review. Coordinate this migration after
ERR-012 and the active ALLOC-013 ownership work so source classification is
settled before the public SPI is changed.

Core checkpoint (2026-07-11): `j2k-types` now owns the non-exhaustive
`J2kEncodeStageError`/`J2kEncodeStageErrorKind` contract and all 14 fallible
hooks return `J2kEncodeStageResult<T>`. Native `EncodeError::Accelerator`
retains both operation and source; native/facade/transcode test doubles use the
same contract, and batch-wide native encode failure is returned once instead
of dishonestly cloning a move-only source into every tile. The source policy
is `encode_stage_error_policy`. Full `j2k-types`/native/facade tests and strict
Clippy passed, as did native no-default-features. The Metal implementation now
preserves typed stage categories plus concrete backend sources; its dependent
J2K/transcode checks, strict Clippy, and source/resource regressions pass.
CUDA now has the same contract: six focused encode-stage source/resource
regressions cover invalid requests, cap and host-allocation failures, concrete
backend sources, malformed output, and ordinary decline. The source policy
passes, the complete `j2k-cuda` run passes 118 library tests together with its
integration and documentation targets, and warning-denied
all-target/all-feature Clippy is green. Core, native, facade, transcode, Metal,
and CUDA implementation parity is closed; frozen API/semver and exact-candidate
hardware evidence remain centralized in SEM-001/FINAL-001.

### ERR-017 — source-preserving transcode-stage and metric failures

The post-CUDAERR-002 source-chain scan found that the public
`TranscodeStageError` still stores backend execution failures as
`Backend(String)` and implements `Error` without `source()`. Both CUDA and
Metal adapters render their typed runtime errors into this string. The Metal
adapter also renders device allocation-too-large and device allocation-failed
variants into the same generic bucket. Host cap and host allocator failures
now retain their categories, but the originating GPU operation, concrete error
type, unavailability metadata, and device-resource classification disappear at
the shared accelerator boundary.

The same narrow conversion scan found `JpegToHtj2kError::Metrics(String)` is
constructed only by rendering the typed `MetricsLengthError`, after which its
`source()` implementation deliberately returns `None`. The source type is
already public and carries the structured mismatch, so the string allocation
and erased chain provide no compatibility or abstraction benefit.

Closure must make the non-exhaustive transcode-stage contract source-aware,
move concrete CUDA/Metal errors into the public boundary instead of formatting
them, and add typed device cap/allocation variants that do not allocate merely
to report resource failure. Replace the metrics function's length-only result
with a typed length/cap/allocator error, store it directly in the metrics
variant, and expose it through `source()`. Ordinary unsupported or unavailable
Auto-mode declines remain distinct from explicit execution failure. Remove infallible
`Clone`/equality requirements from the error if retaining a boxed source makes
those traits dishonest; update callers and tests to match behavior and source
chains rather than copied rendered text. Add CUDA/Metal runtime, kernel, device
cap, device allocator, host cap, and host allocator regressions; a source
policy forbidding `Backend(String)` and `.to_string()` in adapter conversions;
strict dependent checks; changelog/API/semver review; and frozen-tree GPU
verification.

Core and Metal checkpoint (2026-07-11): `TranscodeStageError` now owns a boxed
concrete backend source together with stable backend and operation labels. It
is deliberately move-only and exposes that source through `Error::source`.
Dedicated device-cap and device-allocation variants keep GPU resource failures
separate from host cap/allocation errors and from ordinary unsupported or
unavailable declines. Metrics construction now returns non-exhaustive
`MetricsError` length, cap, and allocator variants, and
`JpegToHtj2kError::Metrics` retains the concrete source. Core focused behavior
tests, source/shape policies, all-target/all-feature check, and strict no-deps
Clippy pass. The Metal runtime, support, and kernel crossings retain concrete
sources, its device mappings use the new resource categories, four focused
regressions pass, and Metal check plus strict Clippy are green. CUDA runtime,
kernel, device-cap, device-allocator, host-cap, and host-allocator crossings
now use the same move-only source-preserving contract. The
`j2k-transcode-cuda` package passes all 20 library tests plus parity,
integration, and documentation targets. Its all-feature and
no-default-feature warning-denied Clippy gates pass, as do the four transcode
source/category policies. Core, Metal, and CUDA implementation migration is
closed; frozen-tree API/semver and device reruns remain under
SEM-001/FINAL-001.

### ERR-018 — typed public HT validation and diagnostic helpers

The reachable-public-result inventory found three bounded static-string seams
outside the accelerator trait: doc-hidden
`j2k_native::packet_math::ht_segment_lengths`, doc-hidden
`j2k_native::collect_ht_cleanup_encode_distribution`, and public
`j2k_transcode::validate_htj2k97_codeblock_options`. Their possible failures
are closed validation/arithmetic cases, yet callers currently receive only
`&'static str` and GPU adapters must reclassify prose. This is smaller than
ERR-016 but still part of the 0.7 public contract and should not remain an
attractive template for new APIs.

Closure must replace each boundary with an appropriate non-exhaustive typed
error (or the existing typed native encode error where it preserves the exact
category), implement `Display`/`Error`, and keep stable reason text only as a
presentation method. GPU and transcode callers must match variants for policy
decisions rather than compare strings. Add one behavior test for every variant,
cross-backend accept/reject parity for 9/7 options, a source policy that scans
reachable public functions for `Result<_, &'static str>`, and changelog/API/
semver review. Internal hot helpers may retain compact private error forms only
when the typed public boundary maps them exhaustively.

Core checkpoint (2026-07-11): `ht_segment_lengths` now returns the
non-exhaustive `HtSegmentLengthError`, and the CPU packetizer exhaustively maps
only length overflow into `EncodeError::ArithmeticOverflow`. Cleanup
distribution returns `EncodeResult` end to end, preserving the existing native
taxonomy. The shared 9/7 validator now returns non-exhaustive
`Htj2k97CodeBlockOptionsError` variants for numeric, quantization, exponent,
and decoded-dimension failures. Every new variant has a behavior regression;
focused native/transcode tests, native strict Clippy/no-default-features, the
transcode library strict Clippy gate, and `public_typed_helper_error_policy`
passed. CUDA packetization and CUDA/Metal 9/7 wrapper parity, frozen-tree
dependent gates, and final public-API/semver review remain. The Metal 9/7
wrapper now exhaustively maps all four typed option-error categories and its
four-category regression passes; Metal check, 18 library tests, and strict
Clippy are green. CUDA packetization now maps every one of the 10
`HtSegmentLengthError` cases exhaustively, and CUDA 9/7 validation maps all
four `Htj2k97CodeBlockOptionsError` categories with the same accept/reject
semantics as Metal. Full variant regressions and both typed-helper source
policies pass. Warning-denied dependent Clippy is green for `j2k-cuda`,
`j2k-transcode-cuda`, and `xtask`. HT helper implementation parity is closed;
the central frozen-candidate API/semver and hardware matrix remain under
SEM-001/FINAL-001.

### DUP-001 — real clone consolidation

Consolidate:

- corpus category inference in j2k-compare; adoption tooling reuses it
- viewport validation and staging population; finalizers remain backend-specific
- CUDA JPEG packet-to-checkpoint/plan construction as a pure helper
- DWT maximum decomposition-level policy in no-std `j2k-codec-math`; native,
  facade, and CUDA validation import the same const helper

Table-test every corpus needle, precedence rule, and fallback. Ensure viewport
tests call production staging rather than a copied loop. Keep ownership wrappers
around the shared CUDA plan helper.

Do not abstract the small exact-tile batch symmetry between two stable public
APIs with backend-private types.

The DWT clone closure is complete in the working tree. The shared helper owns
the zero/unit-axis, shorter-axis, every-power-of-two, and `u32::MAX` contract;
focused codec-math, native no-default-feature, facade, CUDA validation, strict
Clippy, and structure-policy checks passed. RPCL progression, sampled-tile
minimum reduction, bit-width math, and high-bit recode logic were reviewed and
retained because their contracts are distinct rather than clones.

Clone objective:

- remain below the pinned 3.34% duplicated-line ceiling under the canonical
  source-aware `cargo xtask clone-audit` scope; the former 1.93% objective is
  invalid because it came from a scan that omitted files over 1,000 lines,
  and the later direct-tree 3.33% snapshot is not final evidence because it
  included inline test syntax
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
- share subband parameter/style/required-block validation where the audit-time
  J2C decode source carried 26–46-line copies

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

The 2026-07-09 independent CPU/tooling pass classified the following pre-split
production roots after excluding inline tests:

| Disposition | Files | Evidence/next action |
|---|---|---|
| Split (STR-010) | `xtask/src/main.rs` (2,096 production lines), `xtask/src/coverage.rs` (1,030) | Mixed dispatch/release/package/codegen/process concerns; mixed lane/LCOV/policy/render concerns |
| Split (STR-011) | native `ht_block_decode.rs` (1,693), `bitplane.rs` (1,651), `ht_block_encode.rs` (1,882), `bitplane_encode.rs` (1,445), `idwt.rs` (1,530), `codestream.rs` (1,311) | Multiple independent algorithm phases or planning/parsing axes; preserve hot inner loops and exact byte/error behavior |
| Split (STR-012) | JPEG `entropy/sequential.rs` (2,201), `decoder/extended12.rs` (1,887), `adapter/baseline_encode.rs` (951), `entropy/sequential/emit.rs` (1,010) | Mixed public drivers, specialized decode/render routes, planning/assembly, and color-emission responsibilities |
| Split (STR-013) | `j2k-compare/src/encode_compare.rs` (2,324) and `fixture_compare.rs` (2,204) | CLI, input/manifest loading, external tools, validation, measurement, decode, and report rendering are independent axes |
| Accept | native `packet_encode.rs` (794), `tile.rs` (896); JPEG NEON (1,753); `xtask/perf_guard.rs` (785), `xtask/semver.rs` (984); `j2k-types/src/lib.rs` (1,018); transcode accelerator (918) | Cohesive packet/tile/hot-kernel/workflow/public-contract families; triggers are recorded in the accepted register |

That CPU/tooling pass found two 250+ line production functions: the 296-line
classic Tier-1 segment encoder and the 252-line RGB stripe emitter. They were
explicit split targets, not accepted exceptions.

A 2026-07-10 residual CUDA pass found one unclassified 51-line cleanup
launch/status clone, one 252-line mixed color finisher, and one cohesive
1,040-line encode-stage adapter. The worktree consolidates the runtime path,
splits color orchestration from a real store module, adds pure RGB/RGBA,
bit-depth, geometry, reversible/irreversible, fused/separate plan tests, and
adds deterministic NVIDIA store accounting. It also ratchets color ownership
and file size. The affected host behavior and source policies are green;
STR-009 and STR-014 remain in progress only for final combined-tree and
exact-source NVIDIA evidence.

The 2026-07-10 follow-up inventory found two newly oversized, unclassified
adapter owners after the resident-input and fallible-allocation work:
`j2k-cuda/src/encode/htj2k.rs` had reached 1,052 production lines and
`encode/packetization.rs` had reached 1,044 physical lines, including its two
inline boundary tests. They mixed result types, validation, launches, resident
DWT/subband orchestration, tile packet assembly, tag-tree state, descriptor
flattening, and runtime ABI conversion. They are now 20- and 25-line explicit
facades. HTJ2K owns five focused leaves of 74–528 lines; packetization owns six
production leaves of 97–364 lines plus a 46-line test module. No production
`include!` or wildcard import was introduced. Lower per-child caps and exact
ownership checks cover every new boundary, and the allocation census follows
the complete source families rather than only their facades.

The 2026-07-11 post-allocation inventory reopened four production owners after
masking inline tests: CUDA runtime `htj2k_encode.rs` (1,350 lines), CUDA runtime
`htj2k_decode.rs` (1,296), CUDA adapter `decoder/resident.rs` (1,339), and
native `jp2/mod.rs` (about 1,006 production lines before its test module). The
first three still combine resource/type contracts, validation, launch setup,
completion/readback, and result construction; the JP2 root combines public
metadata models, container traversal, metadata conversion, and codestream
consistency validation. Existing child modules do not close this finding.
Split each along those named responsibilities, preserve public paths and exact
diagnostics/bytes, add lower child ratchets, and rerun host behavior plus final
GPU evidence. STR-009 remains open for this source work in addition to its
already recorded final combined/hardware gates.

The CUDA runtime HT pair is now closed. `htj2k_encode.rs` is a 23-line facade
over API (260), completion/result assembly (339), context validation (43),
launch (176), planning/compaction (298), and ABI/resource types (332).
`htj2k_decode.rs` is a 27-line facade over API (148), completion (486), context
validation (100), launch (216), output-region validation/sweep (110/164),
planning (174), queued ownership (141), status (45), and ABI/resource types
(344). Public root re-exports and inherent `CudaContext` method paths are
unchanged; repr(C) field order/size/offset contracts remain under compile-time
and runtime ABI tests. No wildcard sibling import or textual include seam was
introduced.

Default and all-feature runtime checks, warning-denied no-deps library/test
Clippy, 236 default host tests, 245 all-feature host tests, and four exact new
structure/ABI/behavior policies pass in the isolated runtime target. Existing
lifecycle, queued, output-layout, submit, ABI, and result-ownership policies
were retargeted to the authoritative children and pass static contract review;
their executable rerun waits for the combined freeze because the unrelated
external workspace build is stalling new repo-lint binaries before `main`.

The CUDA adapter resident source lane is now closed. The former 1,339-line
mixed owner is a 34-line explicit facade over routing (356), component
resource/work lifecycle (171), cleanup/dequant batching (283), IDWT/color
batching (296), final grayscale surface assembly (119), and pure
conversion/validation helpers (152). Existing decoder-sibling paths remain
available through explicit facade re-exports. CUDA context, session pool,
queued guard, cleanup-before-dequant, IDWT-before-store, profiling-counter,
error, and fallback bodies moved without phase reordering or semantic edits.
No wildcard sibling import, textual include seam, or public API/ABI change was
introduced. Lower root/child ratchets and source policies now cover semantic
ownership, fallible host collections, submit-only allowlisting, IDWT output
preflight ordering, and queued completion/error cleanup. Default and
all-feature all-target checks, strict all-feature no-deps library/test Clippy,
86 default host tests, 179 all-feature host tests, four decoder-family policies,
and the submit-only unsafe allowlist policy pass in the isolated adapter
target. Exact-source NVIDIA execution remains part of the final frozen-tree
gate rather than this structural lane. A broader CUDA safety-policy sweep
passed 22 of 23 tests; its only failure is outside this adapter split:
`validation/htj2k_output.rs` is 203 lines against its existing below-200
policy-module ratchet and remains with that policy owner.

The native JP2 source lane is now closed. The former 1,201-line mixed root is a
27-line explicit facade over 382-line container traversal/orchestration,
459-line public/native metadata and move conversion, 55-line image-header
parsing, and 140-line codestream-consistency modules. Its five inline
allocation/ownership regressions moved unchanged to a real 199-line test
module. Existing allocation, box, COLR, PCLR, CMAP, CDEF, and ICC modules remain
focused and now use explicit imports throughout the family; no wildcard import
or `include!` seam was introduced. Lower per-module ratchets cover the complete
source family.

Every public and crate-visible `jp2` root path is preserved by explicit
re-export, and the crate-root public re-exports are unchanged, so this
structural split has no changelog or stable-API consequence. Production parse,
validation, allocation-accounting, diagnostics, and conversion bodies moved
without behavioral changes. The isolated all-target/all-feature native check
passes without warnings, and the focused JP2 test target compiles. Execution of
that binary and the focused repository-policy binary was deferred because a
concurrent external workspace build stalled newly linked processes in dyld
before `main`; neither reached a test failure. Strict native Clippy reaches
unrelated active encode/test warnings outside the JP2 family, while the split
itself is rustfmt-clean and has no JP2 lint diagnostic. Repeat the focused JP2,
policy, full-library, and strict gates after the combined source freeze.

The 2026-07-09 follow-up suppression/namespace scan found additional
concealment patterns that line-count-only inventory missed:

- `j2k-native` allowed `clippy::too_many_arguments` for the entire crate, while
  JPEG CPU, CUDA, and Metal encoder/runtime roots carried file-wide lint
  allowances. Closure removed them, narrowed them to justified kernels/ABI
  boundaries, or registered an owner and trigger.
- `j2k-transcode/src/lib.rs` textually included the 918-production-line
  accelerator family, `j2k-metal/src/compute.rs` included the direct-execution
  namespace, and JPEG Metal had eight equivalent fragments. Closure converted
  host production fragments to real modules with explicit imports/exports;
  only reviewed test/device-generation seams remain.
- seven crate manifests suppressed `too_many_lines` for every target, and
  `j2k-jpeg` also suppressed `similar_names` globally. Explicit escalation
  inventoried and closed the hidden warnings before narrowing/removal.
- `j2k-native/Cargo.toml` suppressed the entire `pedantic` group in addition
  to a crate-root argument-count allowance. Its explicit escalation retained
  only named, narrow algorithm/ABI expectations with rationales.
- `xtask/Cargo.toml` also suppressed the entire `pedantic` group. Its explicit
  escalation replaced that manifest-wide suppression with focused documented
  expectations where command/schema compatibility required the spelling.

A 2026-07-10 strict-Clippy rerun exposed six new transcode orchestration
regressions after the allocation and typed-error work: two eight-argument
component batch functions and four 101–131-line single/batch encode drivers.
They are now split around typed component requests, batch route validation,
integer validation owners, explicit 9/7 input-family selection, and prepared/
completed single-transcode state. No lint suppression or threshold increase was
added. The former 472-line mixed facade is 238 lines; a 254-line single-tile
orchestrator owns admission/extract/transform/encode/report phases, the batch
encode owner is 517 lines with a 103-line 9/7 input selector, and the 69-line
result-slot owner keeps its 57-line regressions separate. The structural policy
ratchets each new owner and rejects wildcard/import-by-include shortcuts.
`j2k-transcode` passes all-target/all-feature check, strict no-deps
Clippy with `too_many_lines` and `too_many_arguments` denied, and its complete
test/doc-test matrix (65 unit tests plus all integration suites).

No `TODO`, `FIXME`, `HACK`, `XXX`, `todo!`, or `unimplemented!` marker remained
in the 2026-07-09 source scan. The panic-surface gate retains fail-closed Clippy
ratchets for production-library `unwrap` and `expect` use and now adds a
token-aware static inventory for `panic!`, `unreachable!`, `assert!`,
`assert_eq!`, `assert_ne!`, `debug_assert!`, `debug_assert_eq!`, and
`debug_assert_ne!`. Both inventories derive the publishable library set from
Cargo metadata. The explicit-macro inventory reuses the coverage tool's
Syn/span cfg analyzer, masks only syntax proven test-only, and ratchets every
category against its reviewed production baseline instead of relying on grep
counts or silently including inline tests.

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

### AUDIT-001 — source-aware quality-gate integrity

The production clone audit is now repository-owned as
`cargo xtask clone-audit`. It discovers eligible Rust under `crates/`, stages
repository-relative files under `target/clone-audit/production`, and replaces
exact test-only Syn spans with spaces while preserving byte and line positions.
Ambiguous cfg syntax remains in scope. The command validates the checked-in
3.34%/20-line/50-token/20,000-line policy, invokes exactly `jscpd@4.0.5`, and
requires a structurally valid JSON report. CI archives that report and treats
the job as a release-candidate dependency. Fixture regressions prove that a
20-line inline-test clone disappears from the production view while the same
production clone remains visible.

The same shared production-source layer now feeds panic-surface's explicit
macro inventory. The implementation-validation baseline is `panic!` 0,
`unreachable!` 51, `assert!` 8, `assert_eq!` 3, `assert_ne!` 0,
`debug_assert!` 91, `debug_assert_eq!` 76, and `debug_assert_ne!` 0. The
production `expect` ceiling was first tightened from 106 to 82, then 81, and
is now 63; the `unwrap` ceiling moves from 17 to 16. These are ceilings, not
targets. An increase in any category fails, and an explicit-macro overage now
reports the actionable repository-relative file/line/column site for every
invocation in each exceeded category.
Repository policy prevents a second cfg parser or broader coverage-analyzer
visibility and pins the location-bearing scanner contract.

AUDIT-001 is complete in the implementation tree. The 2026-07-11 scoped panic
closure passed all 155 xtask unit tests, the focused panic/source-inventory and
ABI-proof repository policies, and strict all-target xtask Clippy. That settled
production-source run passed all 17 selected publishable libraries at
`unwrap` 17/17, `expect` 82/82, `panic!` 0, `unreachable!` 51, `assert!` 9,
`assert_eq!` 3, `assert_ne!` 0,
`debug_assert!` 91, `debug_assert_eq!` 76, and `debug_assert_ne!` 0. No panic
ratchet was raised. Subsequent audited removals lower the configured ceilings
to `unwrap` 16 and `expect` 63; the explicit production `assert!` ceiling is
also reduced from 9 to 8. A post-ERR-014 moving-tree rerun passed all 17
publishable libraries at `unwrap` 16/16, `expect` 63/63, `panic!` 0,
`unreachable!` 51, `assert!` 8, `assert_eq!` 3, `assert_ne!` 0,
`debug_assert!` 91, `debug_assert_eq!` 76, and `debug_assert_ne!` 0. ALLOC-006
is complete, but ALLOC-018 and other moving-tree work still prevent treating
this as candidate evidence. A final whole-tree formatting and panic-surface
rerun is deferred to combined source freeze. Clone-audit likewise remains
non-candidate evidence, and CLONE-001 stays open until the frozen-SHA rerun.

The latest moving-tree rerun after typed profile/error and structural cleanup
passes all 17 packages at `unwrap` 16/16, `expect` 50/50, `panic!` 0,
`unreachable!` 50/50, `assert!` 8/8, `assert_eq!` 3/3, `assert_ne!` 0,
`debug_assert!` 91/91, `debug_assert_eq!` 66/66, and `debug_assert_ne!` 0.
The configured ceilings were reduced immediately to those counts; no removed
panic surface remains as unused headroom.

Moving-tree clone checkpoint (2026-07-11): the repository-owned scan passes
across 1,085 staged production Rust sources after masking 2,421 test-only
syntax nodes. It reports 196 exact groups, 5,354 duplicated lines (2.13%),
and 46,193 duplicated tokens (2.15%), below the 3.34% fail threshold. The
required review of every 50-plus-line pair did not treat that aggregate pass as
disposition: duplicated generic/RGB JPEG scan orchestration, the shared Metal
tile-submission ownership protocol, and AVX2/NEON safe input normalization were
classified as actionable shared owners. The SIMD normalization is now a
focused safe module with three regressions, local 19/19 backend tests,
x86_64 cross-target compilation, both-architecture warning-denied Clippy, and
a source ratchet; architecture-specific kernels remain separate. The 58-line
generic/RGB JPEG scan clone is also closed: 149-line typed output entry points
delegate geometry, restart seek/skip, rolling-stripe decode, and finish order to
one 233-line monomorphized `ScanSetup`/`ScanBuffers`/`StripeEmitter` driver,
while the 207-line MCU-row kernel remains a focused codestream-order owner. Both
entry-point `too_many_lines` suppressions are gone; restart-region output-mode,
WSI region/restart, and profiled/unprofiled parity pass, and the generated JSON
contains no clone pair involving the generic scan family. The Metal queue
extraction remains active in its owning lane. CUDA codec trait implementations,
runtime/store kernels, and adapter allocation wrappers remain
under ALLOC/CUDASESSION review: explicit trait/ABI boundaries must not be
hidden behind a macro merely to reduce scanner output. This measurement is
diagnostic only and must be replaced by the frozen-SHA report.

Moving-tree ownership checkpoint (2026-07-10): the public J2K Metal `Surface`
derived `Clone` while host storage was a `Vec<u8>`, so an apparently cheap
surface clone performed an infallible full-image allocation and copy. Host
storage is now one immutable `Arc<Vec<u8>>` owner constructed through the
private storage boundary. A pointer-identity regression proves that cloning a
host surface shares the allocation, the fallible borrowed byte-access behavior
is unchanged, and the Metal surface source policy rejects loss of that shared
owner contract. The focused surface suite passes 3/3 and the exact repository
policy passes; warning-denied all-feature `j2k-metal` library Clippy also
passes. The pattern-equivalent JPEG Metal surface conversion remains in its
active ALLOC-018 lane; CLONE-001 still requires the final frozen-source
inventory after all such ownership changes settle.

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

The 2026-07-11 all-target package rerun exposed one post-remediation structural
regression hidden by library-only checks: classic Tier-1 `passes.rs` had grown
to 729 lines against its existing below-700 gate. Padded coefficient/state
preparation is now a focused 67-line module and the pass-kernel owner is 673
lines; no threshold moved. The architecture test was also reconciled with the
already-real token-reader and segmented-encoder child modules instead of
matching their obsolete pre-split locations. The exact structure test, all 37
bitplane tests, and warning-denied all-feature `j2k-core`/`j2k-native`
library/test Clippy pass.

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

The 2026-07-11 pre-0.7 reconciliation found that later progressive, sequential
DCT, lossless/extended-precision, and checkpoint work had reintroduced the
same restart-marker transaction at six call sites across five files. The
transaction now belongs to `BitReader::consume_restart_marker`: it performs
the boundary probe, marker extraction, expected-RST validation, diagnostic
offset capture, sequence advance, and reset after successful validation.
Caller-specific state remains at the caller: lossless preserves its required
pre-probe padding reset, sequential/progressive reset predictors only after
success, progressive also clears `eob_run`, and checkpoint construction keeps
generic error conversion and snapshot timing. The raw lossless restart-index
scanner remains separate because it scans the complete byte stream and reports
marker-start offsets rather than consuming entropy-reader state. A focused
repository policy forbids the ignored `ensure_bits(1)` probe and duplicate
restart validation in those consumers. `cargo check -p j2k-jpeg --lib` and all
16 `BitReader` tests pass, including correct sequence, wrong marker/offset,
missing/truncated marker, FF fill/stuffing, and buffered-padding cases. The
focused repository policy now compiles on stable Rust and executes cleanly;
broad JPEG tests and strict Clippy remain part of the combined freeze rerun.
No accepted-clone register row was added because this clone was removed, not
accepted.

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

JPEGCOR-003 closed in the 2026-07-11 working tree. The old abbreviated-TIFF
normalizer recorded only the first selector in each DQT/DHT marker and never
consulted the public `duplicate_table_policy`. A focused allocation-free helper
now walks every quantization and Huffman definition, distinguishes DC from AC,
tracks all four legal identifiers, and rebuilds only partially deduplicated
markers with a corrected length. `AllowIdentical` removes byte-identical
redefinitions; default `RejectConflicting` keeps them so a conflict-free source
retains byte parity; both reject differing bytes for the same class/id.
Malformed class/id/precision fields and truncated later definitions return
typed errors before assembly. DRI conflict behavior remains unchanged.

Evidence is 7/7 focused normalization tests, 377/377 JPEG library tests, 33/33
public inspect/assembly integrations, 5/5 JPEG segment repository policies,
and warning-denied `j2k-jpeg --lib --tests` Clippy. An adjacent primary-parser
regression now validates DQT precision before deriving payload length, so an
invalid `Pq` cannot be mislabeled truncation. `segment.rs` is 901 lines;
the 367-line production helper and 213-line test owner hold the parsing and
normalization responsibility. Rustfmt and scoped diff hygiene pass. The public
type shape is unchanged, but duplicate-policy behavior is now observable and
malformed inputs can fail earlier.

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

Completed priority 9 in the 2026-07-10 working tree: CUDA adapter
`encode/htj2k.rs` fell from 1,052 production lines to a 20-line facade over
types/results (84), validation (74), code-block/subband launches (255),
resident DWT/subband orchestration (528), and tile packet assembly (143).
`encode/packetization.rs` fell from 1,044 physical lines to a 25-line facade
over plan types (97), tag-tree state (166), packet state/math (299), descriptor
flattening (364), runtime ABI conversion (109), and tests (46). Existing
caller-visible entrypoint signatures, error strings, fallible allocation order,
packet bytes, cfg behavior, and the crate-visible session table accessor remain
at their original parent paths. The 102-test all-feature and 40-test no-feature
`j2k-cuda` library suites pass; both strict all-target Clippy configurations,
the focused structural/allocation policies, strict repo-policy Clippy, and diff
hygiene pass. Exact CUDA execution remains part of FINAL-001.

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
| 9 | Completed in worktree: CUDA adapter `encode/htj2k.rs` (1,052) and `encode/packetization.rs` (1,044 physical) | Types/validation/launch/resident-DWT/tile assembly plus plan/tag-tree/state/flatten/runtime conversion/test boundaries |

Preserve every `repr(C)` field/order, CUDA entrypoint and generated-PTX
metadata check, Metal shader ABI, status/error value, profile label/order,
device command sequence, lifetime retention, and JPEG output order. Real CUDA
execution remains an exact-SHA Linux/NVIDIA gate after hosted compile/parity.

### CUDAERR-001 — pooled asynchronous CUDA resource lifetime

The 2026-07-10 Rust architecture red-team found that several error paths could
drop `CudaPooledDeviceBuffer` metadata while default-stream kernels might still
reference it. The first confirmed paths were HTJ2K cleanup launch/status
readback and queued color IDWT work; the equivalent-pattern scan also reached
the timing-disabled error contract and multi-stage IDWT enqueue. Recycling a
device allocation before completion is a correctness and memory-safety release
stop even when the normal success path remains byte-identical.

Required closure:

1. Centralize default-stream completion. Every safe timed or timing-disabled
   helper establishes completion before returning; true submission-only work
   is exposed only through the explicit unsafe helper and a typed resource
   guard that retains all reachable allocations until real completion.
2. Retain queued IDWT resources behind a typed guard. On every ordinary error,
   surface synchronization failure and release resources only after completion;
   disarm only after a safely ordered dependent store has been submitted.
3. Protect every post-launch failure in multi-stage enqueue and every pooled
   status readback consumer before its owning buffers can recycle.
4. Pattern-scan all CUDA pool take/upload, queued execution, launch, readback,
   kernel-status, early-return, and Drop paths. Record why each survivor is
   synchronous, explicitly guarded, or safely ordered.
5. Serialize context binding and driver/resource operations. A panic while the
   gate is held must poison the atomic health state before the gate unlocks.
   Because NVIDIA documents that [memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html),
   [module](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html),
   and [launch](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html)
   calls may surface earlier asynchronous errors, an ordinary operation error
   may keep the context usable only after an in-gate
   [`cuCtxSynchronize`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
   succeeds. Resource creation, destruction, or ownership-transfer failures
   remain state-uncertain and quarantine the context even if synchronization
   succeeds. A binding failure, synchronization failure, or unwind also
   remains permanently poisoned, and compound failures must preserve both the
   primary operation and completion diagnostics.
6. Validate same-context ownership and writable-region disjointness before
   pointer construction, allocation, context binding, and zero-work returns.
   Keep ordered cross-stage aliases only where the stage sequence proves they
   are not concurrent.
7. Add behavior/source-policy regressions, run strict Clippy and all host CUDA
   suites, then require exact-SHA NVIDIA parity/error injection where the local
   host cannot execute the path.

Status: implementation complete; final combined-tree policy, error-injection,
and frozen-candidate NVIDIA reruns remain. No pre-fix CUDA count or hardware
run is candidate evidence.

### CUDAGRID-001 — CUDA execution-geometry limits

The post-security launch review found that several pure geometry helpers bound
grid axes only by the Driver API's `c_uint` parameter type. That admits values
the hardware contract rejects: current supported compute capabilities allow a
maximum grid x dimension of `2^31 - 1` and y/z dimensions of `65,535`.
For example, a `2x1,048,576` single DWT/IDWT shape produces grid y `65,536`
with the fixed `16x16` launch. The driver rejects the launch safely, but only
after avoidable validation, upload, and allocation work.

Required closure:

1. Centralize nonzero grid/block validation against the documented CUDA axis
   and 1,024-threads-per-block limits; every direct launch must pass it.
2. Make all pure geometry builders fail at the exact boundary plus one instead
   of relying on integer conversion alone.
3. Where a safe public request determines geometry before driver work, reject
   over-limit DWT, IDWT, store-batch, JPEG-batch, and copy requests before
   upload or allocation.
4. Retain defensive launch-time validation for internal/direct geometries and
   add boundary, precedence, and source-order regression policy.
5. Run exact-SHA NVIDIA tests because host-only geometry tests cannot prove the
   real driver accepts the maximum supported shapes.

Grounding: NVIDIA's current [compute-capability
table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
specifies the grid limits, and the Driver API exposes the corresponding
[`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_*`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
attributes. The repository uses the documented limits for its supported modern
CUDA compute capabilities and keeps a single defensive launch boundary.

Status: implementation complete. Central geometry validation and pre-driver
request checks pass focused host tests and strict Clippy; frozen-candidate
maximum-boundary execution remains part of FINAL-001.

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

The J2K-Metal suppression/namespace lane closed in f1611ec0. Seven high-risk
manifest overrides and all actual source/test `#[allow(...)]` attributes are
gone; only `must_use_candidate` and `missing_errors_doc` remain as an explicit
public-API documentation policy pending the release fingerprint review. All
35 non-prelude production wildcard re-exports were replaced by an explicit
symbol inventory without broadening visibility; `compute.rs` remains under
its 450-line structure ratchet. Normal and forced strict all-target Clippy,
public-API rendering, 209 library tests, 54 device integrations, 18 required
runtime tests, benchmark budgets, shader integrity, and focused repository
policies pass on real Metal with no skip evidence.

The J2K-Core escalation closed in ee80a430. All five manifest-wide Clippy
overrides are gone. The 64 diagnostics they hid were resolved with concrete
`must_use` contracts, public `# Errors` documentation, elided needless trait
lifetimes, and one fulfilled item-level complexity expectation preserving the
separate codec/sink error channels. Strict all-target/all-feature Clippy, four
library tests, 35 API/behavior integration tests, doc tests, formatting, and
diff hygiene pass. The simplified `cargo-public-api` rendering remains
byte-identical to the pre-change snapshot.

The small shared-contract lanes closed in eecb0dfd and 7073a99c. J2K-Types
replaced its last actual source allowance with a fulfilled expectation tied to
the five independent COD flags; strict all-target Clippy and both packet-order
tests pass. Metal support removed both documentation-wide manifest overrides
and now documents all 14 public error paths. Strict all-target Clippy, 10/10
real-Metal buffer/queue tests with no skips, doc tests, and rustdoc with denied
warnings pass; its simplified public-API rendering is unchanged.

The CUDA host-suppression lane closed in 804ff6ae. J2K-CUDA, JPEG-CUDA, and
the CUDA runtime now have zero actual host/test `allow(...)` attributes; their
high-risk manifest overrides are removed, forced back on, and green for both
all-feature and no-default all-target builds. The only actual source
allowances left in this lane are the reviewed standalone SIMT device roots and
one shared mutable-pointer helper that is consumed by only a subset of the ten
device crates. Each has an inline device-ABI/toolchain reason and a runbook
owner/trigger. Independent review also removed a zero-consumer const-pointer
helper that repository policy had incorrectly required, then corrected the
policy to protect only live shared primitives. Host tests pass 157/76 for
J2K-CUDA, 40/26 for JPEG-CUDA, and 98/95 for the runtime across all/no-default
features; API rendering and focused device-source policies pass. Exact CUDA
execution remains the frozen-SHA Linux/NVIDIA gate.

Reviewed CUDA device-generation exceptions:

| Scope | Owner | Retained lint reason | Removal/review trigger |
|---|---|---|---|
| `cuda_oxide_simt_prelude.rs::simt_mut_ptr_at` | CUDA runtime/SIMT ABI | Shared include is consumed by HT decode and transcode but unused in the other standalone device crates | Device crates gain per-helper inclusion, every owner consumes it, or the last two consumers stop using it |
| `cuda_oxide_htj2k_encode/simt` | HTJ2K CUDA encode | Explicit ceiling division for the device toolchain; flat kernel ABI arguments; monolithic entropy control flow preserves PTX/register behavior | Supported device compiler gains the integer helpers, ABI is packed into stable job structs, or the kernel is split with exact PTX/register and NVIDIA parity evidence |
| `cuda_oxide_j2k_encode/simt` | Classic/HT J2K CUDA encode | Device-compatible division/parity spelling, flat export/packet ABI, and shared-memory statics | Device compiler support changes, ABI packing changes, or shared-memory access receives a safe device primitive with exact NVIDIA parity |
| `cuda_oxide_j2k_idwt/simt` | CUDA IDWT | Five row/column shared-memory arrays require device-scoped static references | Shared-memory access receives a safe device primitive with exact NVIDIA parity |
| `cuda_oxide_jpeg_encode/simt` | JPEG CUDA encode | Bounded DCT/entropy narrowing, conventional coordinate names, and flat entropy/kernel ABI | A typed bounded ABI removes the casts, coordinate refactor stays readable, or arguments are packed without PTX/register/performance regression |

The shared fixture-support and transcode lanes closed in b61f83f4 and
d814ef7a. Test-support cast overrides were replaced by checked fixture
serialization and narrowly documented oracle conversions; a policy confines
its three intentional fixture-catalog wildcard exports. Transcode removed both
cast-wide manifest overrides and every actual source/test/bench allowance,
replaced its public wildcard facade with explicit exports, removed one
unreferenced duplicate fixture module via `trash`, and extracted the in-place
reversible 5/3 primitive so benchmark/test path reuse no longer imports an
entire implementation module. Dependency-inclusive strict all-target Clippy,
benchmark construction, exact simplified public-API hash parity, support-crate
tests, and 110 transcode tests pass. One duplicate path-module execution
disappeared, but the underlying regression test remains and passes in the
library target.

Residual STR-015 host closure landed through 2488da70, 196f8317, a6d1f370,
0c659ad8, 5a53c7ea, 3000c96a, f1b0bc30, and 369a469d, with path/policy
follow-ups in f0bb18c7 and 78ac9ae1. At fcf99d27 no unregistered host
allow/include/public-glob seam remained. The host implementation is complete,
but STR-015 remains in progress until the combined STR-009/STR-014 tree passes
the final policy rerun.

The final source scan now recognizes conditional attributes, not only direct
`#[allow(...)]` spelling. A follow-up audit found six host cases hidden in
`cfg_attr(..., allow(...))` across native, J2K-Metal, and JPEG-Metal. Those
were removed, converted to fulfilled item expectations, or replaced with
platform gating. The new repository policy rejects direct and conditional
host allowances, file-level expectations, unreviewed manifest overrides,
host-production `include!` seams, and unreviewed public wildcard exports. It
permits only the exact registered CUDA `(file, lint)` device-generation pairs
and the intentional fixture-catalog facade; deleting an exception never
requires a replacement. It also rejects unexplained item expectations and
memory-safety expectations, recognizes multiline public globs, and keeps
generated `target` trees outside repository source inventories.

### STR-017 — repository-policy responsibility split

The mixed 2,696-line
`xtask/tests/repo_lint_support/docs_and_workflows_policy.rs` owner is now an
eight-line real-module shell. Its 42 existing policy tests retain their names,
assertions, and `repo_root()`-relative path behavior under six explicit
responsibility modules: documentation/API evidence (252 lines),
workflow/coverage policy (467), structural ratchets (351), duplication policy
(625), encoder architecture (658), and decoder/fixture architecture (445).
The new structural test keeps the shell below 25 lines, applies lower
per-child ceilings, and rejects actual wildcard imports and `include!`
module seams.

The same verification exposed ALLOC-002's resident-encode source policy above
its existing 150-line ceiling. That owner is now a 42-line coordinator over
explicit allocation checks (147 lines), batch allocation policy (33), a
25-line resident-contract coordinator with a 129-line ownership child,
image-derived allocation policy (9), typed-input policy (55), and
session-resource policy (99). The ownership split also keeps each policy test
below the audit-integrity 100-line function cap. The long image-allocation test
delegates to focused check functions instead of gaining a lint suppression.
Its policy follows the typed `CudaHtj2kPacketizationPlanError::HostAllocation`
route rather than matching the former location of an error string.

Evidence on 2026-07-10:

- `cargo test -p xtask --test repo_lint repo_lint_support::docs_and_workflows_policy`:
  43 passed, 0 failed, 0 ignored.
- `cargo test -p xtask --test repo_lint repo_lint_support::gpu_adapter_policy::resident_encode_policy`:
  7 passed, 0 failed, 0 ignored.
- `cargo clippy -p xtask --all-features --all-targets -- -D warnings` passed.
- `cargo fmt --all -- --check`, focused `rustfmt --check`, and
  `git diff --check` passed.

The 2026-07-11 moving-tree self-ratchet sweep closed every policy-size failure
without raising a ceiling. CUDA decoder architecture moved from its 365-line
mixed owner into a 226-line focus/color/allocation parent plus a 161-line
runtime-ownership child. HTJ2K output planning moved from the 203-line
validation owner into a 161-line region/focus parent plus a 48-line planning
leaf. The aggregate gate then exposed masked debt: JPEG allocation checks now
use a 37-line coordinator over explicit adapter (46), structure (121),
checkpoint (80), and packet (77) leaves; JPEG owned-output policy is a 43-line
child of the 395-line decoder-structure owner; and the xtask-wide lint-policy
test is a 27-line child of its 313-line structural owner. The earlier JPEG
restart policy also moved call-site cardinality checks into a 55-line child so
strict Clippy needs no long-function exception. Parent caps were retained or
lowered, every child is in the aggregate inventory, and no wildcard import,
`include!` module seam, or lint suppression was added.

All assigned CUDA decoder and HTJ2K policies, their child inventories, the
newly exposed JPEG/xtask policies, and the aggregate repository-policy size
gate pass. Default-feature all-target warning-denied no-deps xtask Clippy,
focused rustfmt check, and diff hygiene pass. After the independent native
preparation split settled, warning-denied all-target/all-feature no-deps xtask
Clippy also passed in an isolated target.

### STR-018 — CUDA encode test ownership split

The former 2,453-line `crates/j2k-cuda/src/encode.rs` mixed its production
facade with nearly 2,000 lines of inline tests spanning unrelated routing,
packetization, resident-buffer, transform, and HTJ2K concerns. The production
facade now ends in a real `mod tests;` boundary and is 481 lines.

The current 43 test functions retain their feature gates and required-runtime
behavior under seven explicit owners:

- routing and backend/fallback policy (291 lines)
- packetization plan/state regressions (577 lines)
- public resident-buffer behavior (218 lines)
- resident session/context/resource behavior (168 lines)
- transform-stage behavior and reference parity (296 lines)
- resident tile/DWT pipeline behavior (213 lines)
- HTJ2K code-block, batch, quantized-subband, and strided-region behavior
  (270 lines)

`encode/tests/mod.rs` is a 109-line coordinator containing only explicit module
declarations, imports, and the shared encode fixture helpers. No child uses
`include!` or wildcard imports. The structural policy keeps the production
facade below 550 lines, every test child below its owner-specific ceiling (and
all below 600), and asserts the exact one-owner inventory for all 43 tests.

Evidence on 2026-07-11:

- `cargo test -p j2k-cuda --all-features --lib`: 121 passed, 0 failed,
  0 ignored.
- `cargo clippy -p j2k-cuda --all-targets --all-features -- -D warnings`
  passed.
- `cargo clippy -p j2k-cuda --all-targets --no-default-features -- -D warnings`
  passed.
- the focused CUDA encode-test structure policy and resident encode policies
  passed; focused formatting and diff checks passed.

### STR-019 — J2K facade responsibility split

The facade view, batch, wrapper, and recode roots are already thin
coordinators over explicit responsibility modules with allocation and size
ratchets. The remaining 711-line `decode.rs` owner still combined decode
orchestration and warning policy with both 8-bit component conversion and
16-bit native channel-layout conversion.

`decode.rs` is now a 187-line orchestration/warning root. An explicit
`decode/output.rs` coordinator delegates to a 336-line 8-bit layout owner and
a 229-line 16-bit layout owner. The 8-bit child owns direct-layout eligibility,
component-plane validation/scaling, row packing, and alpha add/drop behavior;
the 16-bit child owns native sample widening, channel preservation/drop,
opaque-alpha synthesis, and row conversion. Imports and visibility are
explicit, with no wildcard or `include!` module seams.

Evidence on 2026-07-11:

- `cargo check -p j2k --all-targets --all-features` passed.
- `cargo clippy -p j2k --all-targets --all-features --no-deps -- -D warnings`
  passed.
- all five focused output-module unit tests and all 43 facade decode
  integration tests passed.
- `j2k_decode_structure_policy` passed and ratchets the root below 220 lines,
  the coordinator below 25, the 8-bit child below 375, and the 16-bit child
  below 260. Existing view, batch, wrapper, and recode structure policies also
  pass.

### STR-020 — JPEG Metal viewport responsibility split

The post-remediation structural inventory found
`crates/j2k-jpeg-metal/src/viewport.rs` at 1,145 lines. It currently owns the
public workload data model and geometry checks, backend route selection,
three separately built fast-packet probes, explicit-Metal validation, fixed
workload suggestion, CPU region decode and multi-tile composition, host-surface
adapters, reusable Metal buffer/texture dispatch, platform stubs, blitting,
and inline tests. This is a real god file rather than a long cohesive table or
state machine. Its repeated platform wrappers and CPU/Metal route families make
allocation and error behavior difficult to audit consistently.

Required closure follows ALLOC-018 and ERR-015 so the split preserves one
correct contract:

1. Keep `viewport.rs` as a small public facade/data-model module. Extract
   workload geometry/suggestion, route and capability selection, CPU
   allocation/composition, surface adaptation, and reusable Metal
   buffer/texture execution into explicit sibling modules with narrow
   visibility and no wildcard or `include!` seams.
2. Build fast-packet capability once through the typed parse/cache boundary;
   route selection must consume that result instead of rebuilding all three
   packet families or turning allocation/invariant errors into absence.
3. Centralize checked viewport/output geometry and one live allocation budget
   used by CPU byte output, one current tile, workload metadata, surface upload,
   and resident Metal outputs. Platform-specific wrappers may select an
   implementation but must not duplicate geometry or error classification.
4. Move inline tests into behavior-focused modules without deleting or merging
   passing regressions. Preserve public and platform-specific coverage for
   overlap/gap detection, scaled rects, route choice, CPU composite byte
   placement, explicit backend failure, reusable buffer/texture behavior, and
   non-macOS fallbacks.

Acceptance requires the facade below 180 lines, each production child below
350 lines and each test child below 600 lines; a source-aware one-owner function
inventory and no wildcard/`include!` policy; unchanged public API except the
planned move-only/fallible 0.7 corrections; focused CPU and exact Metal parity;
strict Clippy; and the serialized repository structural matrix. Do not raise an
existing ceiling or preserve duplicate wrappers merely to make the line counts
pass.

Closure evidence on 2026-07-11: `viewport.rs` is a 157-line public facade over
174-line model/geometry, 245-line routing/policy, 258-line CPU composition,
340-line resident Metal, and 151-line behavior-test owners. The focused
architecture/resource sweep passes all four viewport policies, including the
one-owner module inventory, shared plane-row target, aggregate move-only
allocation contract, and fallible invalid-state handling. The complete JPEG
Metal package passes 196 library tests plus 37 device/integration tests and
docs on the local Apple host, with exact CPU/Metal viewport byte parity,
reusable buffer/texture, sparse/contiguous routing, and unsupported-shape
behavior covered. Warning-denied all-target/all-feature Clippy and diff hygiene
pass. ALLOC-018 and ERR-015 are closed, no ceiling was raised, and STR-020 is
complete; frozen-candidate repetition remains under FINAL-001.

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
| CUDA encode stage adapter (1,040 production lines) | CUDA encode adapter maintainers | One public accelerator state/counter/timing object and one indivisible `J2kEncodeStageAccelerator` trait implementation; per-stage methods remain bounded | New encode stage, second state/timing owner, any method exceeds 150 lines, file exceeds 1,200 production lines, or upstream trait gains separable subtraits |
| CUDA kernel registry (583 production lines; 517 test lines) | CUDA runtime registry | Entry-point/PTX parity ledger, not an orchestrator | 700 production lines or another registry mechanism |
| Retired by STR-021: Metal support root (formerly 1,901 lines) | Shared Metal support | No accepted-large exception remains; the largest focused production owner is 271 lines and tests are 366 lines | Any child exceeds 425 production lines/600 test lines or the facade regains implementation ownership |
| CUDA Oxide HT encode core (1,961 lines; 358-line core) | CUDA HT encode parity | One four-entrypoint hot state machine | New coding mode or core exceeds 450 lines |
| CUDA Oxide JPEG baseline decode (1,943 lines after shared defensive validation) | CUDA JPEG parity | One synchronized fast-baseline ABI across 420/422/444 with one shared grid/checkpoint/Huffman defense | Progressive/lossless support, another output family, or 2,000 production lines |
| CUDA Oxide HT decode (1,326 lines) | CUDA HT decode parity | One cleanup/refinement kernel family | New refinement path or 1,500 lines |
| Metal Tier-1 production core (1,150 production; 951 test-only lines) | Metal Tier-1 | Cohesive classic/HT device encode; test support moves under STR-014 | Production exceeds 1,300 lines or a third coding mode |
| Metal compute ABI ledger (983 production lines) | Metal shader ABI | 56 short layout types/constants plus parity tests | Second device ABI or 1,200 lines |
| JPEG Metal viewport router (former 628 production lines; trigger exceeded at about 1,000 production lines) | JPEG Metal viewport | **No longer accepted:** CPU allocation/composition, fast-packet routing, reusable Metal buffer/texture paths, and platform wrappers became separate responsibilities | STR-020 must split it before source freeze; do not re-accept the 1,145-line owner |

Moving-tree trigger check: a raw physical-line scan reported
`accelerator_contracts.rs` at 1,167 lines, but the first production test seam is
at line 866 and the remaining 302 lines are two explicit ground-truth/allocation
test modules. The production owner is therefore about 865 lines, below its
recorded 1,000-line reconsideration trigger. It still owns the same public job,
trait/default-accelerator, counter, and reversible-oracle family, so it is not a
god-file split target. Final STR-009 evidence must use the source-aware scanner,
not raw `wc`, so inline-test growth cannot create a false production finding.

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

This section consolidates completed documentation batches and the current
pre-candidate reconciliation. The former untracked
`engineering/documentation-remediation-plan.md` was a temporary agent ledger,
not a second source of truth; its unique decisions and evidence are preserved
here before that duplicate is removed with `trash`.

### METALDEP-001 — workspace-only `block` patch scope

Closure evidence on 2026-07-11 proves the dependency boundary rather than
inferring it from the working lockfile. `cargo package -p j2k-metal
--allow-dirty --list` succeeds and its complete package inventory contains no
`third_party/block-0.1.6-patched` path or patch metadata. A standalone
downstream manifest under `target/`, with no enclosing workspace patch and an
exact `metal = "=0.33.0"` dependency, resolves:

```text
block v0.1.6
└── metal v0.33.0
    └── j2k-metaldep-downstream-proof v0.0.0
```

Locked Cargo metadata identifies both `metal` and `block` as crates.io
registry packages and points `block` at the registry source, not the vendored
workspace directory. The provenance hash/ABI policy and a new patch-scope
policy pass; the latter requires the patch only at the workspace root,
requires the vendored crate to remain outside workspace membership, forbids
member manifests from implying a publishable patch, and pins the release
guide's downstream warning.

The post-0.7 maintenance owner is the Metal adapter/shared-support lane. Review
or replace the dependency when upstream `metal` changes its `block` edge, when
`objc2-metal` supplies the required buffer/texture/command/session APIs on the
supported Rust baseline, or before the next release that materially expands
Metal public API. Migration acceptance requires exact device behavior, ABI,
autorelease/ownership, public-API, benchmark, and package-resolution parity;
the local patch must not be silently removed or described as protecting
published consumers. Maintainer reviewer/date signoff for the vendored ABI
delta remains PROV-001, not METALDEP-001.

### Public contracts verified by the documentation audit

- `0.6.x` is the latest published and security-supported line; workspace
  `0.7.0` remains staged and unsupported until publication. Its notes stay
  under `Unreleased` during remediation and receive the actual dated heading
  only in the final pre-freeze edit.
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
- the unsafe inventory is intended to cover every FFI, GPU, SIMD/intrinsic,
  allocation, and bounded pointer/buffer boundary; exhaustive current-source
  status is not claimed until the post-freeze inventory gate passes

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
- Hardened direct real publication around the workspace repository identity,
  clean `HEAD`, and an exact annotated tag whose local object, remote object,
  and peeled commit agree; URL-rewrite and credential-bearing diagnostics fail
  closed before Cargo or registry access.
- Documented the current `third_party/block-0.1.6-patched` override and its
  removal conditions. Follow-up review established that Cargo ignores a
  dependency's `[patch]` table: published consumers still resolve upstream
  `metal 0.33.0 -> block 0.1.6`. The release guide now says this directly and
  records upstream's `objc2-metal` migration direction instead of presenting
  the workspace patch as downstream remediation.
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
  The 2026-07-10 read-only recheck
  `gh api repos/frames-sg/j2k/private-vulnerability-reporting --jq .enabled`
  returned `false`; GitHub's [private-reporting
  documentation](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)
  says the public submission form works only when this repository setting is
  enabled.
- Record the release maintainer's name/handle and approval date for provenance.
- The packaged downstream-consumer resolution check is complete: without the
  workspace patch it resolves `metal 0.33.0 -> block 0.1.6`. Retain the
  maintained-binding migration as an owned post-0.7 task unless the dependency
  is replaced before freeze.
- As the final release-preparation edit before candidate freeze, replace the
  changelog's real `Unreleased` heading with the dated `0.7.0` heading and
  update every staged-document reference that still says the notes are under
  `Unreleased`. Do not guess the date early; a later change creates a new
  candidate.
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
- source-aware clone-audit
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

The maintainer has supplied an out-of-band Linux/WSL NVIDIA host for the CUDA
lane. Do not record its private address or login in this tracked document. Once
the source and evidence tree is frozen, transfer the exact local commit with a
Git bundle, clone that bundle into a fresh temporary directory on the host, and
run the repository-owned CUDA release command there. This permits exact-commit
hardware validation without pushing. Treat that run as pre-publication evidence
only until the same commit is the protected `origin/main` tip and the hosted
exact-SHA verifier succeeds.

## 13. Candidate freeze and publication

1. Recheck that the selected version has no remote tag, GitHub Release, or
   crates.io publication.
2. Complete code, workflows, documentation, the dated `0.7.0` changelog
   heading, API snapshot, and semver report.
3. Commit everything and require a clean worktree.
4. Set RC_SHA to the current HEAD, then run
   `cargo xtask release-integrity --publish` and `cargo xtask package` from
   that clean commit. Any correction restarts the candidate at step 2.
5. Before push authorization, optionally transfer RC_SHA to the maintainer's
   CUDA host with a Git bundle and run the exact-commit CUDA gate from a fresh
   clone. Do not substitute a copied dirty worktree.
6. Push that candidate as the intended protected origin/main tip through the
   normal reviewed workflow; this step requires explicit maintainer authority.
7. Run hosted CI for exactly RC_SHA.
8. Run exact-SHA CUDA and Metal validation. A pre-push CUDA bundle run may be
   retained as supporting evidence, but it does not replace the hosted
   exact-SHA verification contract.
9. Require the shared verifier to prove every core/API/GPU job succeeded.
10. If any tracked change occurs, discard all evidence and restart at step 2.
11. Stop at the verified-RC endpoint for the current execution. Do not create
    or push a tag and do not publish crates without separate authorization.
12. A later authorized release execution creates an annotated release tag at
    RC_SHA and verifies that it peels to RC_SHA.
13. Push only the tag; never use --follow-tags.
14. Publish preflight rechecks the canonical origin, exact local and remote
    annotated-tag object and peeled SHA, workflow identity, required jobs,
    GitHub Release state, and crates.io.
15. Publish packages in dependency order.
16. If publication is partial, rerun against the immutable tag with the
    documented idempotent skip-already-published mode. Never move the tag.

Manual workflow_dispatch publication is always dry-run.

## 14. Interface and compatibility decisions

- Version 0.7 intentionally contracts the published pre-1.0 0.6.2 `j2k`
  facade and several adapter/runtime surfaces. It makes no 0.6.x source-
  compatibility claim; the reviewed report and changelog migration notes must
  enumerate the changes before freeze.
- j2k-native gains a neutral decode-error classification interface; record it
  as an additive API change.
- j2k-metal-support becomes the sole checked Metal buffer-access boundary.
- Correcting an unsound published helper is an explicitly approved 0.7
  compatibility change, not a hidden compatibility shim.
- Developer tooling gains metal-compile, fail-closed release-metal,
  release-status, exact API-diff reporting, and shared workflow verification.
- Workflows consume repository tooling instead of embedding divergent logic.
- Before candidate freeze, every stable API change must appear in both the
  regenerated reviewed report and the changelog.

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

The current verified-RC remediation is complete only when:

- every dashboard item is complete or explicitly accepted
- all user changes are reconciled
- no unexplained ignored test, clone, dead entrypoint, unsafe buffer view, or
  oversized mixed-responsibility orchestrator remains
- the full local release matrix is green from a clean tree
- hosted CI and both GPU workflows are green for the same immutable SHA
- the changelog and public API report describe the actual candidate
- `cargo xtask release-integrity --publish` and `cargo xtask package` succeed
  without bypasses while no `v0.7.0` tag exists locally or remotely

Tag creation, tag verification, tag-dependent `publish-crate.sh --preflight-all`,
crate publication, and post-publication documentation remain a later explicitly
authorized release execution.
