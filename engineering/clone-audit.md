# Production Clone Audit

Audit date: 2026-07-09
Release line: 0.7.0
Baseline: `v0.6.2` (`f5c1bf05ea33df8c8dec04251ca38a266171a05d`)
Candidate production source: `0e78229a` plus documentation/metadata-only
worktree changes

## Method

The repository pins `jscpd` 4.0.5 through the exact command below and pins its
analysis settings in `.jscpd.json`. The scan covers production Rust under
`crates/`, including nested CUDA SIMT sources. It excludes tests, benchmarks,
examples, fuzz targets, generated files, build scripts, and test-support
crates. A reported clone must span at least 20 lines and 50 tokens.

The `maxLines` setting is deliberately 20,000. An earlier ad hoc result near
1.74% used jscpd's default 1,000-line file ceiling, which omitted the largest
production files—the exact files this audit was supposed to assess. That
number and the derived 1.93% objective are invalid release evidence. The
corrected candidate ceiling is 3.34%; jscpd fails when line duplication reaches
or exceeds that value.

Reproduce the comparison from the repository root:

```bash
trash /tmp/j2k-clone-v062 /tmp/j2k-jscpd-v062 2>/dev/null || true
mkdir -p /tmp/j2k-clone-v062 /tmp/j2k-jscpd-v062
git archive v0.6.2 crates | tar -x -C /tmp/j2k-clone-v062
npx --yes jscpd@4.0.5 /tmp/j2k-clone-v062/crates \
  --config .jscpd.json --threshold 100 --output /tmp/j2k-jscpd-v062 --silent
npx --yes jscpd@4.0.5 crates --config .jscpd.json --output target/jscpd --silent
jq '.statistics.total' /tmp/j2k-jscpd-v062/jscpd-report.json
jq '.statistics.total' target/jscpd/jscpd-report.json
```

The baseline overrides only the failure threshold so its higher historical
ratio still produces a report. All detection and scope settings remain those
in `.jscpd.json`.

## Result

| Metric | v0.6.2 | 0.7 candidate | Change |
|---|---:|---:|---:|
| Analyzed production lines | 203,999 | 205,827 | +1,828 (+0.9%) |
| Analyzed sources | 272 | 394 | +122 (intentional module splits) |
| Clone pairs | 290 | 241 | -49 (-16.9%) |
| Duplicated lines | 9,011 | 6,847 | -2,164 (-24.0%) |
| Duplicated-line ratio | 4.42% | 3.33% | -1.09 percentage points |
| Duplicated tokens | 81,086 | 60,690 | -20,396 (-25.2%) |
| Duplicated-token ratio | 4.49% | 3.36% | -1.13 percentage points |

The candidate passes the pinned 3.34% line-duplication ceiling. The line ratio
fell by 24.7% relative to v0.6.2 while production lines grew slightly.

## Review of the largest remaining pairs

The machine report is a review queue, not an instruction to force every pair
through one abstraction. The largest remaining categories were checked against
the accepted-clone register in the remediation runbook:

| Largest observed pattern | Lines | Disposition |
|---|---:|---|
| CUDA and JPEG-CUDA device codec facades | 81 | Accepted stable facade symmetry; backend-private types make a shared public abstraction worse |
| CUDA transcode kernel branches | 66 | Accepted SIMT specialization; reconsider if one generated source can retain device clarity and performance |
| Native encode/finalization families | 63, 60 | Algorithm-family symmetry; the mixed-responsibility single-tile coordinator was split separately |
| CUDA transcode geometry variants | 60 | Device-shape specialization; retain until a branch-free common primitive exists |
| JPEG sequential entropy paths | 58 | Format-state symmetry; retain while state transitions differ |
| CUDA and Metal error classification | 57 | Public backend context differs; neutral native classification was consolidated to prevent semantic drift |
| JPEG-CUDA and CUDA decode adapters | 57 | Accepted adapter symmetry; reconsider when a neutral public request type already exists |
| CUDA/JPEG Metal shader and architecture paths | 51–53 | Accepted host/SIMT or device specialization with parity tests |

Genuine drift-prone clones found during the audit—corpus inference, viewport
staging, CUDA checkpoint planning, and native decode error classification—were
consolidated and behavior-tested. The remaining pairs are owned by explicit
reconsideration triggers rather than a zero-duplication target.

## Freeze rule

Rerun the candidate command after the final source commit. A ratio of 3.34% or
higher, a new unclassified pair of at least 50 lines, a changed scan scope, or
a non-zero scanner exit blocks the candidate until reviewed and recorded.
