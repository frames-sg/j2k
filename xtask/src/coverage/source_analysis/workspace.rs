// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::coverage::build_outputs::BuildOutputEvidence;
use crate::coverage::model::CoverageLane;
use crate::process::{self, cargo, CommandContext};

use super::ast::validate_source;
use super::cfg_eval::CoverageCfgContext;
use super::graph::ReachKind;
use super::{SourceRole, GENERATED_DWT_DISPOSITION, VENDORED_BLOCK_DISPOSITION};
mod fuzz_manifests;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct SourceRoot {
    pub(super) package: String,
    pub(super) path: String,
    pub(super) kind: ReachKind,
    pub(super) crate_root: bool,
}

#[derive(Debug)]
pub(super) struct CoverageCfgContexts {
    packages: BTreeMap<String, CoverageCfgContext>,
}

impl CoverageCfgContexts {
    pub(super) fn get(&self, package: &str) -> Result<&CoverageCfgContext, String> {
        self.packages.get(package).ok_or_else(|| {
            format!("coverage cfg context for workspace package `{package}` is missing")
        })
    }

    #[cfg(test)]
    pub(super) fn synthetic(package: &str, context: CoverageCfgContext) -> Self {
        Self {
            packages: BTreeMap::from([(package.to_string(), context)]),
        }
    }

    #[cfg(test)]
    pub(super) fn synthetic_packages(packages: &BTreeSet<String>) -> Self {
        let context = CoverageCfgContext::synthetic([]);
        Self {
            packages: packages
                .iter()
                .map(|package| (package.clone(), context.clone()))
                .collect(),
        }
    }
}

#[derive(Debug)]
struct SelectedPackage {
    name: String,
    enabled_features: BTreeSet<String>,
    has_build_script: bool,
}

pub(super) fn discover_source_roots(
    root: &Path,
    lane: CoverageLane,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    build_output_evidence: &BuildOutputEvidence,
) -> Result<(Vec<SourceRoot>, CoverageCfgContexts), String> {
    let output = process::command_output(
        cargo(),
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
        CommandContext::new().current_dir(root),
    )?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed while resolving coverage source roots with {}",
            output.status
        ));
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata for coverage is malformed: {error}"))?;
    metadata_roots(root, lane, changed, &metadata, build_output_evidence)
}

fn metadata_roots(
    root: &Path,
    lane: CoverageLane,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    metadata: &serde_json::Value,
    build_output_evidence: &BuildOutputEvidence,
) -> Result<(Vec<SourceRoot>, CoverageCfgContexts), String> {
    let members = workspace_member_ids(metadata)?;
    let packages = required_array(metadata, "packages", "cargo metadata")?;
    let mut roots = BTreeSet::new();
    let mut selected = Vec::new();
    let mut workspace_manifests = BTreeSet::new();
    for (index, package) in packages.iter().enumerate() {
        let context = format!("cargo metadata packages[{index}]");
        let id = required_string(package, "id", &context)?;
        if !members.contains(id) {
            continue;
        }
        let name = required_string(package, "name", &context)?;
        let manifest = required_string(package, "manifest_path", &context)?;
        let manifest = repository_relative(root, Path::new(manifest))?;
        workspace_manifests.insert(manifest.clone());
        let package_dir = Path::new(&manifest)
            .parent()
            .ok_or_else(|| format!("workspace manifest `{manifest}` has no parent directory"))?;
        let package_dir = package_dir.to_str().ok_or_else(|| {
            format!(
                "workspace manifest parent is not UTF-8: {}",
                package_dir.display()
            )
        })?;
        if !changed
            .keys()
            .any(|path| lane.owns_path(path) && repository_path_is_within(path, package_dir))
        {
            continue;
        }
        let features = required_object(package, "features", &context)?
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let package_roots = cargo_source_roots(root, package, name, &context)?;
        selected.push(SelectedPackage {
            name: name.to_string(),
            enabled_features: features,
            has_build_script: package_roots
                .iter()
                .any(|root| root.kind == ReachKind::BuildScript),
        });

        roots.extend(package_roots);
    }

    for package in fuzz_manifests::discover(root, lane, changed, &workspace_manifests)? {
        if selected
            .iter()
            .any(|selected| selected.name == package.name)
        {
            return Err(format!(
                "coverage source discovery found duplicate package name `{}`",
                package.name
            ));
        }
        roots.extend(package.roots);
        selected.push(SelectedPackage {
            name: package.name,
            enabled_features: package.enabled_features,
            has_build_script: false,
        });
    }

    let contexts = capture_cfg_contexts(&selected, build_output_evidence)?;
    Ok((roots.into_iter().collect(), contexts))
}

fn workspace_member_ids(metadata: &serde_json::Value) -> Result<BTreeSet<String>, String> {
    required_array(metadata, "workspace_members", "cargo metadata")?
        .iter()
        .enumerate()
        .map(|(index, member)| {
            member
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| format!("cargo metadata workspace_members[{index}] is not a string"))
        })
        .collect()
}

fn cargo_source_roots(
    root: &Path,
    package: &serde_json::Value,
    package_name: &str,
    package_context: &str,
) -> Result<BTreeSet<SourceRoot>, String> {
    let mut roots = BTreeSet::new();
    for (index, target) in required_array(package, "targets", package_context)?
        .iter()
        .enumerate()
    {
        let context = format!("{package_context}.targets[{index}]");
        let Some(kind) = cargo_target_reach_kind(target, &context)? else {
            continue;
        };
        let source = required_string(target, "src_path", &context)?;
        roots.insert(SourceRoot {
            package: package_name.to_string(),
            path: repository_relative(root, Path::new(source))?,
            kind,
            crate_root: true,
        });
    }
    if roots.is_empty() {
        return Err(format!(
            "selected workspace package `{package_name}` has no recognized Cargo source root"
        ));
    }
    Ok(roots)
}

fn cargo_target_reach_kind(
    target: &serde_json::Value,
    context: &str,
) -> Result<Option<ReachKind>, String> {
    let kinds = required_array(target, "kind", context)?
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            kind.as_str()
                .ok_or_else(|| format!("{context}.kind[{index}] is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let kind = if kinds.contains(&"custom-build") {
        Some(ReachKind::BuildScript)
    } else if kinds.contains(&"test") {
        Some(ReachKind::TestTarget)
    } else if kinds
        .iter()
        .any(|kind| matches!(*kind, "example" | "bench"))
    {
        Some(ReachKind::ExampleBenchFuzz)
    } else if kinds.iter().any(|kind| {
        matches!(
            *kind,
            "lib" | "rlib" | "dylib" | "cdylib" | "staticlib" | "proc-macro" | "bin"
        )
    }) {
        Some(ReachKind::Production)
    } else {
        None
    };
    Ok(kind)
}

fn capture_cfg_contexts(
    packages: &[SelectedPackage],
    build_output_evidence: &BuildOutputEvidence,
) -> Result<CoverageCfgContexts, String> {
    let selected_packages = packages
        .iter()
        .map(|package| package.name.clone())
        .collect::<BTreeSet<_>>();
    let build_script_packages = packages
        .iter()
        .filter(|package| package.has_build_script)
        .map(|package| package.name.clone())
        .collect::<BTreeSet<_>>();
    let current_cfg_flags =
        build_output_evidence.current_cfg_flags(&selected_packages, &build_script_packages)?;
    let mut contexts = BTreeMap::new();
    for package in packages {
        let custom_flags = current_cfg_flags.get(&package.name).cloned();
        contexts.insert(
            package.name.clone(),
            CoverageCfgContext::for_current_target(package.enabled_features.clone(), custom_flags),
        );
    }
    Ok(CoverageCfgContexts { packages: contexts })
}

pub(super) fn classify_unreached_source(root: &Path, path: &str) -> Result<SourceRole, String> {
    let source = read_source(root, path)?;
    validate_source(path, &source)?;
    if path == "crates/j2k-codec-math/generated/dwt97_constants.rs" {
        return Ok(SourceRole::Generated(GENERATED_DWT_DISPOSITION));
    }
    if path == "third_party/block-0.1.6-patched/src/lib.rs" {
        return Ok(SourceRole::VendoredReviewed(VENDORED_BLOCK_DISPOSITION));
    }
    if path == "third_party/block-0.1.6-patched/src/test_utils.rs" {
        return Ok(SourceRole::TestOnly);
    }
    if is_clone_audit_fixture(path) {
        return Ok(SourceRole::TestOnly);
    }
    if path == "crates/j2k-cuda-runtime/src/cuda_oxide_simt_prelude.rs" {
        return Ok(SourceRole::Generated("cuda-shared-simt-prelude"));
    }
    if path.starts_with("crates/j2k-cuda-runtime/src/cuda_oxide_")
        && path.contains("/simt/src/")
        && has_rust_extension(path)
    {
        return Ok(SourceRole::Generated("cuda-simt-device-rust"));
    }
    if path.starts_with("crates/j2k-cuda-runtime/src/cuda_oxide_")
        && path.contains("/src/")
        && !path.contains("/simt/")
        && path.ends_with("main.rs")
    {
        return Ok(SourceRole::Generated("cuda-generated-host-scaffold"));
    }
    Err(format!(
        "changed Rust source `{path}` is unreachable from Cargo metadata roots and has no reviewed source disposition"
    ))
}

fn has_rust_extension(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
}

fn is_clone_audit_fixture(path: &str) -> bool {
    Path::new(path).parent() == Some(Path::new("xtask/tests/fixtures/clone_audit"))
        && has_rust_extension(path)
}

#[cfg(test)]
pub(super) fn discover_manifest_fuzz_source_roots(
    root: &Path,
    changed: &BTreeMap<String, BTreeSet<usize>>,
) -> Result<Vec<SourceRoot>, String> {
    Ok(
        fuzz_manifests::discover(root, CoverageLane::Host, changed, &BTreeSet::new())?
            .into_iter()
            .flat_map(|package| package.roots)
            .collect(),
    )
}

pub(super) fn read_source(root: &Path, path: &str) -> Result<String, String> {
    let source_path = root.join(path);
    fs::read_to_string(&source_path).map_err(|error| {
        format!(
            "failed to read coverage source {}: {error}",
            source_path.display()
        )
    })
}

fn required_array<'a>(
    value: &'a serde_json::Value,
    field: &str,
    context: &str,
) -> Result<&'a Vec<serde_json::Value>, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{context}.{field} is missing or is not an array"))
}

fn required_object<'a>(
    value: &'a serde_json::Value,
    field: &str,
    context: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{context}.{field} is missing or is not an object"))
}

fn required_string<'a>(
    value: &'a serde_json::Value,
    field: &str,
    context: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("{context}.{field} is missing or is not a string"))
}

fn repository_path_is_within(path: &str, directory: &str) -> bool {
    let directory = directory.replace(std::path::MAIN_SEPARATOR, "/");
    let directory = directory.as_str();
    directory == "."
        || path == directory
        || path
            .strip_prefix(directory)
            .is_some_and(|remainder| remainder.starts_with('/'))
}

fn repository_relative(root: &Path, path: &Path) -> Result<String, String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("failed to canonicalize repository root: {error}"))?;
    let canonical_path = fs::canonicalize(path)
        .map_err(|error| format!("failed to canonicalize {}: {error}", path.display()))?;
    let relative = canonical_path.strip_prefix(&canonical_root).map_err(|_| {
        format!(
            "coverage source {} is outside repository root {}",
            canonical_path.display(),
            canonical_root.display()
        )
    })?;
    relative
        .to_str()
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| format!("coverage source path is not UTF-8: {}", relative.display()))
}
