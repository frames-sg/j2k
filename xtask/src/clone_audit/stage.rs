// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use crate::source_audit::{mask_test_only_syntax, production_rust_sources};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CloneStageSummary {
    pub(super) files: usize,
    pub(super) masked_nodes: usize,
    pub(super) mixed_lines: usize,
}

pub(super) fn stage_production_sources(
    repository_root: &Path,
    stage_root: &Path,
) -> Result<CloneStageSummary, String> {
    let sources = production_rust_sources(repository_root, &[repository_root.join("crates")])?;
    let mut summary = CloneStageSummary::default();
    for source_path in sources {
        let source = fs::read_to_string(&source_path.absolute).map_err(|error| {
            format!(
                "read production clone-audit source {}: {error}",
                source_path.relative.display()
            )
        })?;
        let masked = mask_test_only_syntax(repository_root, &source_path.relative, &source)?;
        let staged_path = stage_root.join(&source_path.relative);
        let parent = staged_path.parent().ok_or_else(|| {
            format!(
                "staged clone-audit path has no parent: {}",
                staged_path.display()
            )
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create clone-audit stage {}: {error}", parent.display()))?;
        fs::write(&staged_path, masked.text).map_err(|error| {
            format!(
                "write staged clone-audit source {}: {error}",
                staged_path.display()
            )
        })?;
        summary.files = summary
            .files
            .checked_add(1)
            .ok_or_else(|| "clone-audit staged file count overflow".to_string())?;
        summary.masked_nodes = summary
            .masked_nodes
            .checked_add(masked.masked_nodes)
            .ok_or_else(|| "clone-audit masked-node count overflow".to_string())?;
        summary.mixed_lines = summary
            .mixed_lines
            .checked_add(masked.mixed_lines.len())
            .ok_or_else(|| "clone-audit mixed-line count overflow".to_string())?;
    }
    Ok(summary)
}

pub(super) fn reset_generated_directory(audit_root: &Path, directory: &Path) -> Result<(), String> {
    if directory == audit_root || !directory.starts_with(audit_root) {
        return Err(format!(
            "refusing to reset clone-audit path outside generated children: {}",
            directory.display()
        ));
    }
    if directory.exists() {
        fs::remove_dir_all(directory).map_err(|error| {
            format!(
                "clear generated clone-audit directory {}: {error}",
                directory.display()
            )
        })?;
    }
    Ok(())
}
