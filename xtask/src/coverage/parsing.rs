// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::path::Path;

use crate::process::{self, CommandContext};

use super::model::LcovReport;

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
            current_path = Some(normalize_lcov_path(path, root)?);
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

fn normalize_lcov_path(path: &str, root: &Path) -> Result<String, String> {
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
    Ok(relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}
