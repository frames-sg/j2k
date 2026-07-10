// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::fs;

mod evaluation;
mod exclusion_policy;
mod lane;
mod model;
mod parsing;
mod summary;

use evaluation::{coverage_violations, evaluate_changed_coverage};
use exclusion_policy::validate_exclusion_policy;
use lane::run_lane;
use model::parse_options;
use parsing::{git_output, parse_changed_lines, parse_lcov, resolve_diff_base};
use summary::{print_summary, write_summary};

pub(crate) fn coverage(args: impl Iterator<Item = String>) -> Result<(), String> {
    let options = parse_options(args)?;
    let root =
        env::current_dir().map_err(|err| format!("failed to locate repository root: {err}"))?;
    validate_exclusion_policy(&root)?;

    let lcov_path = root.join(options.lane.lcov_path());
    run_lane(options.lane, &lcov_path)?;

    let base = resolve_diff_base(options.base.as_deref())?;
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
    let lcov = fs::read_to_string(&lcov_path)
        .map_err(|err| format!("failed to read {}: {err}", lcov_path.display()))?;
    let report = parse_lcov(&lcov, &root)?;
    if report.lines.is_empty() {
        return Err(format!(
            "{} did not contain any Rust coverage records",
            lcov_path.display()
        ));
    }

    let result = evaluate_changed_coverage(options.lane, &root, &changed, &report)?;
    let violations = coverage_violations(options.lane, &result);
    let summary_path = options
        .output
        .unwrap_or_else(|| root.join(options.lane.summary_path()));
    write_summary(
        &summary_path,
        options.lane,
        &base,
        &merge_base,
        &lcov_path,
        &result,
        &violations,
    )?;
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
