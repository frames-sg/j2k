// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::path::{Component, Path};

use crate::process::{self, CommandContext};

use super::model::LcovReport;

pub(super) fn ensure_no_untracked_rust_sources() -> Result<(), String> {
    let untracked = git_output(&["ls-files", "--others", "--exclude-standard", "--", "*.rs"])?;
    validate_no_untracked_rust_sources(&untracked)
}

pub(super) fn validate_no_untracked_rust_sources(untracked: &str) -> Result<(), String> {
    let paths = untracked
        .lines()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>();
    if paths.is_empty() {
        return Ok(());
    }
    Err(format!(
        "changed-line coverage cannot classify untracked Rust sources; add or stage them before running the gate:\n- {}",
        paths.join("\n- ")
    ))
}

pub(super) fn resolve_diff_base(explicit: Option<&str>) -> Result<String, String> {
    if let Some(base) = explicit {
        verify_git_revision(base)?;
        return Ok(base.to_string());
    }
    if let Ok(base) = env::var("J2K_COVERAGE_BASE") {
        if base.trim().is_empty() {
            return Err("J2K_COVERAGE_BASE must not be empty".to_string());
        }
        verify_git_revision(&base)?;
        return Ok(base);
    }
    if let Ok(base_ref) = env::var("GITHUB_BASE_REF") {
        if !base_ref.trim().is_empty() {
            for candidate in [format!("origin/{base_ref}"), base_ref] {
                if verify_git_revision(&candidate).is_ok() {
                    return Ok(candidate);
                }
            }
            return Err(
                "GITHUB_BASE_REF is not available locally; coverage checkout must use fetch-depth: 0"
                    .to_string(),
            );
        }
    }

    let fallback = "HEAD^";
    verify_git_revision(fallback).map_err(|_| {
        "cannot resolve a changed-line coverage base; pass --base or set J2K_COVERAGE_BASE"
            .to_string()
    })?;
    Ok(fallback.to_string())
}

fn verify_git_revision(revision: &str) -> Result<(), String> {
    git_output(&["rev-parse", "--verify", &format!("{revision}^{{commit}}")]).map(|_| ())
}

pub(super) fn git_output(args: &[&str]) -> Result<String, String> {
    let output = process::command_output(OsString::from("git"), args, CommandContext::new())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(format!(
            "`git {}` exited with {}{}",
            args.join(" "),
            output.status,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub(super) fn parse_changed_lines(diff: &str) -> Result<BTreeMap<String, BTreeSet<usize>>, String> {
    let mut changed = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut current_path = None::<String>;

    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            current_path = Some(path.to_string());
            continue;
        }
        if !line.starts_with("@@ ") {
            continue;
        }
        let path = current_path
            .as_ref()
            .ok_or_else(|| format!("diff hunk has no destination path: {line}"))?;
        let added = line
            .split_whitespace()
            .find(|part| part.starts_with('+'))
            .ok_or_else(|| format!("diff hunk has no added range: {line}"))?;
        let range = added
            .trim_start_matches('+')
            .split_once(',')
            .map_or((added.trim_start_matches('+'), "1"), |(start, count)| {
                (start, count)
            });
        let start = range
            .0
            .parse::<usize>()
            .map_err(|err| format!("invalid diff hunk start in `{line}`: {err}"))?;
        let count = range
            .1
            .parse::<usize>()
            .map_err(|err| format!("invalid diff hunk count in `{line}`: {err}"))?;
        if count == 0 {
            continue;
        }
        let end = start
            .checked_add(count)
            .ok_or_else(|| format!("diff hunk range overflows in `{line}`"))?;
        changed.entry(path.clone()).or_default().extend(start..end);
    }
    Ok(changed)
}

pub(super) fn parse_lcov(input: &str, root: &Path) -> Result<LcovReport, String> {
    let mut report = LcovReport::default();
    let mut current_path = None::<String>;

    for line in input.lines() {
        if let Some(path) = line.strip_prefix("SF:") {
            current_path = Some(normalize_coverage_path(path, root)?);
            continue;
        }
        let Some(data) = line.strip_prefix("DA:") else {
            continue;
        };
        let path = current_path
            .as_ref()
            .ok_or_else(|| format!("LCOV DA record has no source file: {line}"))?;
        let mut fields = data.split(',');
        let line_number = fields
            .next()
            .ok_or_else(|| format!("LCOV DA record has no line number: {line}"))?
            .parse::<usize>()
            .map_err(|err| format!("invalid LCOV line number in `{line}`: {err}"))?;
        let count = fields
            .next()
            .ok_or_else(|| format!("LCOV DA record has no execution count: {line}"))?
            .parse::<u64>()
            .map_err(|err| format!("invalid LCOV execution count in `{line}`: {err}"))?;
        report
            .lines
            .entry(path.clone())
            .or_default()
            .entry(line_number)
            .and_modify(|existing| *existing = (*existing).max(count))
            .or_insert(count);
    }
    Ok(report)
}

pub(super) fn normalize_coverage_path(path: &str, root: &Path) -> Result<String, String> {
    let path = Path::new(path);
    let relative = if path.is_absolute() {
        path.strip_prefix(root).map_err(|_| {
            format!(
                "LCOV source {} is outside repository root {}",
                path.display(),
                root.display()
            )
        })?
    } else {
        path.strip_prefix("./").unwrap_or(path)
    };
    let mut normalized = Vec::new();
    for component in relative.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part.to_string_lossy()),
            Component::ParentDir => {
                if normalized.pop().is_none() {
                    return Err(format!(
                        "coverage source {} resolves outside repository root {}",
                        path.display(),
                        root.display()
                    ));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "coverage source {} could not be normalized relative to repository root {}",
                    path.display(),
                    root.display()
                ));
            }
        }
    }
    Ok(normalized.join("/"))
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_no_untracked_rust_sources, git_output, normalize_coverage_path, resolve_diff_base,
        validate_no_untracked_rust_sources, verify_git_revision,
    };
    use std::path::Path;

    const MISSING_REVISION: &str = "j2k-coverage-test-revision-that-does-not-exist";

    #[test]
    fn coverage_paths_collapse_lexical_parents_without_escaping_the_repository() {
        let root = Path::new("/workspace/j2k");
        assert_eq!(
            normalize_coverage_path(
                "/workspace/j2k/crates/demo/src/bin/../../benches/support.rs",
                root,
            ),
            Ok("crates/demo/benches/support.rs".to_string())
        );
        assert!(normalize_coverage_path("/workspace/j2k/../../outside.rs", root).is_err());
    }

    #[test]
    fn coverage_preflight_matches_gits_untracked_rust_inventory() {
        let untracked =
            git_output(&["ls-files", "--others", "--exclude-standard", "--", "*.rs"]).unwrap();

        assert_eq!(
            ensure_no_untracked_rust_sources(),
            validate_no_untracked_rust_sources(&untracked)
        );
    }

    #[test]
    fn explicit_diff_base_requires_a_commit_revision() {
        assert_eq!(resolve_diff_base(Some("HEAD")).unwrap(), "HEAD");

        let error = resolve_diff_base(Some(MISSING_REVISION)).unwrap_err();
        assert!(error.contains("git rev-parse --verify"));
        assert!(error.contains(MISSING_REVISION));
    }

    #[test]
    fn revision_verification_accepts_head_and_rejects_missing_names() {
        verify_git_revision("HEAD").unwrap();

        let error = verify_git_revision(MISSING_REVISION).unwrap_err();
        assert!(error.contains("git rev-parse --verify"));
        assert!(error.contains(MISSING_REVISION));
    }

    #[test]
    fn git_output_trims_stdout_and_reports_command_failures() {
        let head = git_output(&["rev-parse", "HEAD"]).unwrap();
        assert!(!head.is_empty());
        assert!(head.bytes().all(|byte| byte.is_ascii_hexdigit()));

        let error = git_output(&["rev-parse", "--verify", MISSING_REVISION]).unwrap_err();
        assert!(error.contains("git rev-parse --verify"));
        assert!(error.contains(MISSING_REVISION));
        assert!(error.contains("exited with"));
    }
}
