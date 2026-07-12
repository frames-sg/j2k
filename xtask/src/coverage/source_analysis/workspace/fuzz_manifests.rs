// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::coverage::model::CoverageLane;

use super::super::graph::ReachKind;
use super::{repository_relative, SourceRoot};

#[derive(Debug)]
pub(super) struct ManifestFuzzPackage {
    pub(super) name: String,
    pub(super) enabled_features: BTreeSet<String>,
    pub(super) roots: BTreeSet<SourceRoot>,
}

pub(super) fn discover(
    root: &Path,
    lane: CoverageLane,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    workspace_manifests: &BTreeSet<String>,
) -> Result<Vec<ManifestFuzzPackage>, String> {
    let mut packages = Vec::new();
    for manifest in candidate_manifests(root, lane, changed, workspace_manifests)? {
        if let Some(package) = parse_manifest(root, &manifest)? {
            packages.push(package);
        }
    }
    Ok(packages)
}

fn candidate_manifests(
    root: &Path,
    lane: CoverageLane,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    workspace_manifests: &BTreeSet<String>,
) -> Result<BTreeSet<PathBuf>, String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("failed to canonicalize repository root: {error}"))?;
    let mut manifests = BTreeSet::new();
    for path in changed.keys().filter(|path| {
        lane.owns_path(path)
            && Path::new(path)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
    }) {
        let source = fs::canonicalize(root.join(path)).map_err(|error| {
            format!(
                "failed to resolve changed Rust source {}: {error}",
                root.join(path).display()
            )
        })?;
        let parent = source.parent().ok_or_else(|| {
            format!(
                "changed Rust source {} has no parent directory",
                source.display()
            )
        })?;
        if !source.starts_with(&canonical_root) {
            return Err(format!(
                "changed Rust source {} resolves outside repository root {}",
                source.display(),
                canonical_root.display()
            ));
        }
        for directory in parent.ancestors() {
            if !directory.starts_with(&canonical_root) {
                break;
            }
            let manifest = directory.join("Cargo.toml");
            if manifest.is_file() {
                let relative = repository_relative(root, &manifest)?;
                if !workspace_manifests.contains(&relative) {
                    manifests.insert(manifest);
                }
            }
            if directory == canonical_root {
                break;
            }
        }
    }
    Ok(manifests)
}

fn parse_manifest(root: &Path, manifest: &Path) -> Result<Option<ManifestFuzzPackage>, String> {
    let relative_manifest = repository_relative(root, manifest)?;
    let source = fs::read_to_string(manifest)
        .map_err(|error| format!("failed to read `{relative_manifest}`: {error}"))?;
    let document = toml::from_str::<toml::Value>(&source).map_err(|error| {
        format!("failed to parse cargo manifest `{relative_manifest}`: {error}")
    })?;
    let Some(package) = document.get("package").and_then(toml::Value::as_table) else {
        return Ok(None);
    };
    let cargo_fuzz = package
        .get("metadata")
        .and_then(toml::Value::as_table)
        .and_then(|metadata| metadata.get("cargo-fuzz"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false);
    if !cargo_fuzz {
        return Ok(None);
    }

    let name = package
        .get("name")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| format!("cargo-fuzz manifest `{relative_manifest}` has no package.name"))?;
    let enabled_features = document
        .get("features")
        .and_then(toml::Value::as_table)
        .map_or_else(BTreeSet::new, |features| features.keys().cloned().collect());
    let bins = document
        .get("bin")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| {
            format!(
                "cargo-fuzz manifest `{relative_manifest}` must explicitly declare [[bin]] targets"
            )
        })?;
    let manifest_dir = manifest.parent().ok_or_else(|| {
        format!("cargo-fuzz manifest `{relative_manifest}` has no parent directory")
    })?;
    let canonical_manifest_dir = fs::canonicalize(manifest_dir).map_err(|error| {
        format!(
            "failed to resolve cargo-fuzz package directory {}: {error}",
            manifest_dir.display()
        )
    })?;
    let mut roots = BTreeSet::new();
    for (index, bin) in bins.iter().enumerate() {
        let target = bin.as_table().ok_or_else(|| {
            format!("cargo-fuzz manifest `{relative_manifest}` bin[{index}] is not a table")
        })?;
        let target_path = target
            .get("path")
            .and_then(toml::Value::as_str)
            .filter(|path| !path.is_empty())
            .ok_or_else(|| {
                format!(
                    "cargo-fuzz manifest `{relative_manifest}` bin[{index}] has no explicit path"
                )
            })?;
        let source = fs::canonicalize(manifest_dir.join(target_path)).map_err(|error| {
            format!(
                "failed to resolve cargo-fuzz target `{target_path}` from `{relative_manifest}`: {error}"
            )
        })?;
        if !source.starts_with(&canonical_manifest_dir) {
            return Err(format!(
                "cargo-fuzz target `{target_path}` from `{relative_manifest}` resolves outside its package"
            ));
        }
        roots.insert(SourceRoot {
            package: name.to_string(),
            path: repository_relative(root, &source)?,
            kind: ReachKind::ExampleBenchFuzz,
        });
    }
    if roots.is_empty() {
        return Err(format!(
            "cargo-fuzz manifest `{relative_manifest}` has no explicit target roots"
        ));
    }
    Ok(Some(ManifestFuzzPackage {
        name: name.to_string(),
        enabled_features,
        roots,
    }))
}
