# Source-aware clone audits

Release line: 0.7.3

Production baseline: 1.96% duplicated lines (5,254 lines, 192 exact
clones, 1,193 jscpd sources). The enforced ceiling is 2.01%.

Test/support baseline: 4.09% duplicated lines (7,300 lines, 198 exact
clones, 905 jscpd sources). The enforced ceiling is 4.14%.

## Canonical command

```bash
cargo xtask clone-audit
```

The source-aware command pins `jscpd@4.0.5`, validates both checked-in configurations,
stages both scopes below `target/clone-audit/`, independently validates both
JSON reports, and fails at or above either ceiling. Configuration drift,
malformed output, unreadable Rust, unsupported source symlinks, or scanner
failure also fails closed.

Reports:

- production: `target/clone-audit/report/jscpd-report.json`
- tests/support: `target/clone-audit/test-report/jscpd-report.json`

## Scope ownership

The production lane covers Rust below `crates/` while excluding physical
tests, test-support crates, benches, examples, fuzz targets, generated code,
and build scripts. Its Syn-based source analysis replaces proven test-only
syntax with spaces while preserving byte and newline positions. Ambiguous cfg
expressions remain production.

The test/support lane includes physical test targets, test helper modules,
test-support crates, and test-only syntax extracted from mixed production
files. For mixed files it masks production syntax, preserving the same byte
and newline positions. This keeps production and test duplication separately
actionable without counting inline tests twice.

Both lanes require Rust detection, mild matching, at least 20 lines and 50
tokens per clone, and a 20,000-line source ceiling. The higher source ceiling
prevents large device algorithms and fixture owners from silently disappearing
from reports.

## Production clones of at least 50 lines

The v0.7.3 cleanup report contains five reviewed pairs:

| Lines | Owners | Disposition | Reconsideration trigger |
|---:|---|---|---|
| 53 | CUDA JPEG SIMT sampling loops | Retain explicit device-mode sequencing and checkpoint exits | Shared bug fix or a fourth sampling mode |
| 66 | CUDA HTJ2K 9/7 resident/readback bands | Retain distinct pooled-resident and owned-readback lifecycles | Allocation or timing logic diverges, or an existing typed owner can express both |
| 56 | CUDA J2K store sample/channel variants | Retain ABI-specific launch and output ownership after shared validation | Repeated validation defect or another output sample family |
| 57 | JPEG/J2K CUDA image-device facades | Retain trait-protocol symmetry around codec-private decoders | Trait method set changes or a neutral existing request type covers both |
| 81 | JPEG/J2K CUDA tile-device facades | Retain stable trait symmetry around codec-private contexts and surfaces | Another backend duplicates the block or the public trait gains a neutral owner |

New production pairs of at least 50 lines require consolidation or an entry in
the living remediation register with an owner, reason, and trigger. The
ratchets may be lowered after sustained improvement; they must not be raised to
accept regression.
