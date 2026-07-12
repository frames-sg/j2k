// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::fs;

mod accelerator_ownership;
mod build_outputs;
mod evaluation;
mod exclusion_policy;
mod lane;
mod model;
mod parsing;
mod source_analysis;
mod summary;

use accelerator_ownership::validate_shared_accelerator_registry;
use evaluation::{coverage_violations, evaluate_changed_coverage};
use exclusion_policy::validate_exclusion_policy;
use lane::run_lane;
use model::parse_options;
use parsing::{
    ensure_no_untracked_rust_sources, git_output, parse_changed_lines, parse_lcov,
    resolve_diff_base,
};
use source_analysis::SourceIndex;
pub(crate) use source_analysis::{
    analyze_test_only_syntax, SourceAuditSyntax, SourceAuditTestSpan,
};
use summary::{print_summary, write_summary, CoverageSummaryInput};

pub(crate) fn coverage(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = parse_options(args)?;
    let root =
        env::current_dir().map_err(|err| format!("failed to locate repository root: {err}"))?;
    ensure_no_untracked_rust_sources()?;
    validate_shared_accelerator_registry(&root)?;
    validate_exclusion_policy(&root)?;

    let base = resolve_diff_base(options.base.as_deref())?;
    let head_sha = git_output(&["rev-parse", "HEAD"])?;
    let merge_base = git_output(&["merge-base", "HEAD", &base])?;
    let diff = git_output(&[
        "diff",
        "--unified=0",
        "--no-ext-diff",
        "--diff-filter=ACMR",
        &merge_base,
        "--",
        "*.rs",
    ])?;
    let changed = parse_changed_lines(&diff)?;
    let lcov_path = root.join(options.lane.lcov_path());
    let lane_run = run_lane(&root, options.lane, &lcov_path)?;
    // Source analysis intentionally follows the lane build. Its build-script
    // cfg evidence comes only from the lane's unique current-build target;
    // missing or conflicting evidence cannot silently remove changed source.
    let source_index = SourceIndex::build(
        &root,
        options.lane,
        &changed,
        &lane_run.build_output_evidence,
    )?;
    let lcov = fs::read_to_string(&lcov_path)
        .map_err(|err| format!("failed to read {}: {err}", lcov_path.display()))?;
    let report = parse_lcov(&lcov, &root)?;
    if report.lines.is_empty() {
        return Err(format!(
            "{} did not contain any Rust coverage records",
            lcov_path.display()
        ));
    }

    let result = evaluate_changed_coverage(options.lane, &root, &changed, &report, &source_index)?;
    let violations = coverage_violations(options.lane, &result);
    let summary_path = options
        .output
        .unwrap_or_else(|| root.join(options.lane.summary_path()));
    write_summary(&CoverageSummaryInput {
        path: &summary_path,
        lane: options.lane,
        base: &base,
        merge_base: &merge_base,
        head_sha: &head_sha,
        lcov_path: &lcov_path,
        cargo_llvm_cov_version: &lane_run.cargo_llvm_cov_version,
        result: &result,
        violations: &violations,
    })?;
    print_summary(options.lane, &summary_path, &result);

    if violations.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "changed-line coverage failed:\n{}",
            violations
                .iter()
                .map(|violation| format!("- {violation}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

#[cfg(test)]
mod tests;
