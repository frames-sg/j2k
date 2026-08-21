// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{ffi::OsStr, fs, path::Path};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct MetalStageSummary {
    pub(super) files: usize,
}

pub(super) fn stage_metal_sources(
    repository_root: &Path,
    stage_root: &Path,
) -> Result<MetalStageSummary, String> {
    let mut sources = Vec::new();
    collect_metal_sources(&repository_root.join("crates"), &mut sources)?;
    sources.sort();
    if sources.is_empty() {
        return Err("Metal clone audit found no shader sources".to_string());
    }

    for source in &sources {
        let relative = source.strip_prefix(repository_root).map_err(|_| {
            format!(
                "Metal clone-audit source is outside repository: {}",
                source.display()
            )
        })?;
        let staged = stage_root.join(relative);
        let parent = staged.parent().ok_or_else(|| {
            format!(
                "staged Metal clone-audit path has no parent: {}",
                staged.display()
            )
        })?;
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "create Metal clone-audit stage {}: {error}",
                parent.display()
            )
        })?;
        fs::copy(source, &staged).map_err(|error| {
            format!(
                "stage Metal clone-audit source {} as {}: {error}",
                relative.display(),
                staged.display()
            )
        })?;
    }

    Ok(MetalStageSummary {
        files: sources.len(),
    })
}

fn collect_metal_sources(
    directory: &Path,
    sources: &mut Vec<std::path::PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| {
        format!(
            "read Metal source directory {}: {error}",
            directory.display()
        )
    })? {
        let path = entry
            .map_err(|error| format!("read entry in {}: {error}", directory.display()))?
            .path();
        if path.is_dir() {
            collect_metal_sources(&path, sources)?;
        } else if path.extension().and_then(OsStr::to_str) == Some("metal") {
            sources.push(path);
        }
    }
    Ok(())
}
