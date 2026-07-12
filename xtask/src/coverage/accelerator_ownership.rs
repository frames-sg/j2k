// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

const SHARED_ACCELERATOR_MARKER: &str = "// j2k-coverage: shared-accelerator-host";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SharedAcceleratorSource {
    pub(super) path: &'static str,
    pub(super) package: &'static str,
}

const SHARED_ACCELERATOR_SOURCES: &[SharedAcceleratorSource] = &[
    source("crates/j2k-core/src/accelerator.rs", "j2k-core"),
    source("crates/j2k-core/src/backend.rs", "j2k-core"),
    source("crates/j2k-core/src/device.rs", "j2k-core"),
    source("crates/j2k-profile/src/gpu_route.rs", "j2k-profile"),
    source("crates/j2k-types/src/lib.rs", "j2k-types"),
    source("crates/j2k/src/adapter/device_plan.rs", "j2k"),
    source("crates/j2k/src/adapter/encode_stage.rs", "j2k"),
    source("crates/j2k/src/encode/routing.rs", "j2k"),
    source(
        "crates/j2k-native/src/j2c/encode/precomputed/accelerator.rs",
        "j2k-native",
    ),
    source(
        "crates/j2k-native/src/j2c/encode/single_tile/accelerator.rs",
        "j2k-native",
    ),
    source(
        "crates/j2k-transcode/src/accelerator_contracts.rs",
        "j2k-transcode",
    ),
    source(
        "crates/j2k-transcode/src/jpeg_to_htj2k/batch/accelerated_storage.rs",
        "j2k-transcode",
    ),
    source(
        "crates/j2k-transcode/src/jpeg_to_htj2k/report.rs",
        "j2k-transcode",
    ),
    source("crates/j2k-transcode/src/pipeline_map.rs", "j2k-transcode"),
];

const fn source(path: &'static str, package: &'static str) -> SharedAcceleratorSource {
    SharedAcceleratorSource { path, package }
}

pub(super) fn is_shared_accelerator_path(path: &str) -> bool {
    SHARED_ACCELERATOR_SOURCES
        .iter()
        .any(|source| source.path == path)
}

pub(super) fn shared_accelerator_packages() -> BTreeSet<&'static str> {
    SHARED_ACCELERATOR_SOURCES
        .iter()
        .map(|source| source.package)
        .collect()
}

#[cfg(test)]
pub(super) const fn shared_accelerator_sources() -> &'static [SharedAcceleratorSource] {
    SHARED_ACCELERATOR_SOURCES
}

pub(super) fn validate_shared_accelerator_registry(root: &Path) -> Result<(), String> {
    let registered = SHARED_ACCELERATOR_SOURCES
        .iter()
        .map(|source| source.path.to_string())
        .collect::<BTreeSet<_>>();
    if registered.len() != SHARED_ACCELERATOR_SOURCES.len() {
        return Err("shared accelerator coverage registry contains duplicate paths".to_string());
    }
    for source in SHARED_ACCELERATOR_SOURCES {
        let package = source
            .path
            .strip_prefix("crates/")
            .and_then(|path| path.split('/').next())
            .ok_or_else(|| {
                format!(
                    "shared accelerator source `{}` has no workspace package owner",
                    source.path
                )
            })?;
        if package != source.package {
            return Err(format!(
                "shared accelerator source `{}` is registered to `{}` instead of `{package}`",
                source.path, source.package
            ));
        }
    }

    let marked = collect_marked_sources(&root.join("crates"), root)?;
    if marked == registered {
        return Ok(());
    }
    let unregistered = marked.difference(&registered).cloned().collect::<Vec<_>>();
    let unmarked = registered.difference(&marked).cloned().collect::<Vec<_>>();
    Err(format!(
        "shared accelerator coverage ownership is incomplete; register every marked source and mark every registered source (unregistered markers: [{}]; registered paths missing markers: [{}])",
        unregistered.join(", "),
        unmarked.join(", ")
    ))
}

fn collect_marked_sources(directory: &Path, root: &Path) -> Result<BTreeSet<String>, String> {
    let mut marked = BTreeSet::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .map_err(|error| format!("failed to inspect {}: {error}", current.display()))?
        {
            let entry = entry.map_err(|error| {
                format!(
                    "failed to inspect entry under {}: {error}",
                    current.display()
                )
            })?;
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
            if file_type.is_dir() {
                pending.push(path);
            } else if file_type.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
                && source_has_marker(&path)?
            {
                marked.insert(repository_relative(root, &path)?);
            }
        }
    }
    Ok(marked)
}

fn source_has_marker(path: &Path) -> Result<bool, String> {
    let source = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    Ok(source
        .lines()
        .any(|line| line.trim() == SHARED_ACCELERATOR_MARKER))
}

fn repository_relative(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path.strip_prefix(root).map_err(|error| {
        format!(
            "shared accelerator source {} is outside {}: {error}",
            path.display(),
            root.display()
        )
    })?;
    relative
        .to_str()
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| {
            format!(
                "shared accelerator source path is not UTF-8: {}",
                path.display()
            )
        })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::validate_shared_accelerator_registry;

    #[test]
    fn source_markers_and_shared_accelerator_registry_are_complete() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        validate_shared_accelerator_registry(&root).unwrap();
    }
}
