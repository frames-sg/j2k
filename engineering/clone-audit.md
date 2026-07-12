# Production Clone Audit

Release line: 0.7.0

Final candidate ratio: pending. The command has been implementation-tested on
the moving worktree, but its numeric result is not release evidence until the
0.7 source tree is frozen and the command is rerun at the candidate SHA.

## Canonical command

Run the repository-owned gate from the repository root:

```bash
cargo xtask clone-audit
```

The command stages source-aware production Rust under
`target/clone-audit/production/`, preserving repository-relative paths, byte
length, and line positions. It then invokes exactly `jscpd@4.0.5` with the
checked-in `.jscpd.json` and validates the JSON report before succeeding. The
report is generated at:

```text
target/clone-audit/report/jscpd-report.json
```

The command accepts no arguments. Configuration drift, malformed or missing
scanner output, a scanner failure, an unsupported source symlink, an unreadable
or unparsable Rust source, or a result at or above the configured threshold
fails the gate.

## Production-source scope

The scan covers Rust below `crates/`, including nested CUDA SIMT sources, while
excluding physical test targets and helpers, benches, examples, fuzz targets,
generated sources, build scripts, and test-support crates. Source selection and
staging are repository-owned; invoking jscpd directly on `crates/` is not the
canonical audit.

Inline test syntax is excluded structurally rather than by text matching. The
clone audit reuses the coverage tool's Syn-based cfg and span analyzer through
a narrow `pub(crate)` facade. Exact spans proven to be test-only, including
`#[cfg(test)]` modules and `#[test]` items, are replaced with spaces while line
breaks remain in place. Ambiguous cfg expressions remain in production scope,
and mixed production/test lines are reported by the staging summary. This keeps
the policy conservative and prevents duplicated inline tests from inflating the
production clone count.

Repository fixtures lock down both sides of that contract: a 20-line clone
inside inline test modules falls below the production threshold after masking,
while an equivalent production clone remains visible. Separate tests verify
path preservation, unchanged byte/newline positions, conservative cfg handling,
the pinned scanner invocation, the checked-in configuration, and fail-closed
report validation.

## Pinned policy

`.jscpd.json` requires Rust format detection, mild matching, at least 20 lines
and 50 tokens per clone, a 20,000-line source ceiling, and a 3.34% duplicated-line
failure threshold. The high source ceiling is intentional: jscpd's historical
1,000-line default omitted the largest production files and produced misleading
results.

The generated stage lives below `target/`, so the canonical configuration does
not ignore `**/target/**`. The xtask validates this invariant and supplies only
the staged production directory to jscpd. This configuration must not be reused
for an unscoped repository-root scan.

## Superseded measurements

The 2026-07-09 manual snapshot reported 3.33% duplicated lines. It scanned the
working `crates/` tree directly and therefore included inline test syntax in
production files. That number, the earlier 1.74% scan that used jscpd's default
file ceiling, and objectives derived from either number are not valid final
0.7 release evidence. They are retained only as historical context in version
control; this document intentionally does not promote a replacement ratio from
the moving worktree.

## Freeze rule

After the final structural source commit, run `cargo xtask clone-audit` at the
frozen candidate SHA and archive the JSON report from CI. A duplicated-line
ratio of 3.34% or higher, a newly unclassified clone of at least 50 lines, any
scope/configuration drift, or any non-zero command exit blocks the candidate
until the cause is reviewed and recorded in the remediation runbook. Record the
frozen SHA, source/file totals, clone totals, duplicated line/token totals and
ratios, and dispositions for newly material pairs in this document.
