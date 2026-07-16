// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use crate::source_audit::{
    auditable_rust_sources, is_production_rust_path, retain_test_only_syntax,
};

use super::stage::CloneStageSummary;

pub(super) fn stage_test_sources(
    repository_root: &Path,
    stage_root: &Path,
) -> Result<CloneStageSummary, String> {
    let sources = auditable_rust_sources(repository_root, &[repository_root.join("crates")])?;
    let mut summary = CloneStageSummary::default();
    for source_path in sources {
        let source = fs::read_to_string(&source_path.absolute).map_err(|error| {
            format!(
                "read test clone-audit source {}: {error}",
                source_path.relative.display()
            )
        })?;
        let (text, inline_nodes, mixed_lines) = if is_production_rust_path(&source_path.relative) {
            let retained =
                retain_test_only_syntax(repository_root, &source_path.relative, &source)?;
            if retained.masked_nodes == 0 {
                continue;
            }
            (
                retained.text,
                retained.masked_nodes,
                retained.mixed_lines.len(),
            )
        } else {
            (source, 0, 0)
        };
        let staged_path = stage_root.join(&source_path.relative);
        let parent = staged_path.parent().ok_or_else(|| {
            format!(
                "staged test clone-audit path has no parent: {}",
                staged_path.display()
            )
        })?;
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "create test clone-audit stage {}: {error}",
                parent.display()
            )
        })?;
        fs::write(&staged_path, text).map_err(|error| {
            format!(
                "write staged test clone-audit source {}: {error}",
                staged_path.display()
            )
        })?;
        summary.files = checked_add(summary.files, 1, "file")?;
        summary.masked_nodes = checked_add(summary.masked_nodes, inline_nodes, "inline node")?;
        summary.mixed_lines = checked_add(summary.mixed_lines, mixed_lines, "mixed line")?;
    }
    if summary.files == 0 {
        return Err("test clone audit found no eligible sources".to_string());
    }
    Ok(summary)
}

fn checked_add(current: usize, additional: usize, label: &str) -> Result<usize, String> {
    current
        .checked_add(additional)
        .ok_or_else(|| format!("test clone-audit staged {label} count overflow"))
}
