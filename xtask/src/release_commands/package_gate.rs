//! Dependency-aware construction of publishable workspace packages.

mod consumer;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::command_support::run_cargo;
use crate::process::{cargo, run_command_owned, CommandContext};

use consumer::{package_archive_path, run_j2k_ml_consumer_gate};

use super::release_manifest::{
    registry_independent_packages, release_dependencies_by_package,
    validate_release_manifest_contract, ReleaseManifestContract,
};
use super::{package_name, workspace_package_records};

#[derive(Debug)]
struct PackageGateStep {
    package: String,
    version: String,
    registry_independent: bool,
    patches: Vec<(String, String)>,
}

fn package_gate_plan(
    metadata: &serde_json::Value,
    manifest: &ReleaseManifestContract,
) -> Result<Vec<PackageGateStep>, String> {
    let packages = workspace_package_records(metadata)?;
    let mut validation_errors = Vec::new();
    validate_release_manifest_contract(manifest, &packages, &mut validation_errors)?;
    if !validation_errors.is_empty() {
        return Err(format!(
            "release manifest violations:\n- {}",
            validation_errors.join("\n- ")
        ));
    }
    let package_by_name = packages
        .iter()
        .map(|package| Ok((package_name(package)?.to_string(), package)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let dependencies_by_package = release_dependencies_by_package(manifest, &packages)?;
    let independent = registry_independent_packages(manifest, &packages)?;
    let mut processed = BTreeSet::new();
    let mut plan = Vec::with_capacity(manifest.ordered_crates().len());

    for package in manifest.ordered_crates() {
        let mut pending = dependencies_by_package
            .get(package)
            .cloned()
            .ok_or_else(|| format!("package gate dependency graph omitted `{package}`"))?;
        let mut dependency_closure = BTreeSet::new();
        while let Some(dependency_name) = pending.pop_first() {
            if !dependency_closure.insert(dependency_name.clone()) {
                continue;
            }
            let transitive = dependencies_by_package
                .get(&dependency_name)
                .ok_or_else(|| {
                    format!("package gate dependency graph omitted `{dependency_name}`")
                })?;
            pending.extend(transitive.iter().cloned());
        }

        let mut patches = Vec::with_capacity(dependency_closure.len());
        for dependency_name in dependency_closure {
            if !processed.contains(dependency_name.as_str()) {
                return Err(format!(
                    "package gate order processes `{package}` before unpublished workspace dependency `{dependency_name}`"
                ));
            }
            let dependency_record = package_by_name.get(&dependency_name).ok_or_else(|| {
                format!("workspace dependency `{dependency_name}` is absent from cargo metadata")
            })?;
            let manifest_path = dependency_record
                .get("manifest_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    format!("cargo metadata package `{dependency_name}` has no manifest path")
                })?;
            let crate_path = Path::new(manifest_path)
                .parent()
                .ok_or_else(|| format!("manifest path for `{dependency_name}` has no parent"))?;
            patches.push((dependency_name, crate_path.to_string_lossy().into_owned()));
        }

        plan.push(PackageGateStep {
            package: package.to_string(),
            version: package_by_name
                .get(package)
                .ok_or_else(|| {
                    format!("publishable package `{package}` is absent from cargo metadata")
                })?
                .get("version")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("cargo metadata package `{package}` has no version"))?
                .to_string(),
            registry_independent: independent.contains(package),
            patches,
        });
        processed.insert(package);
    }
    Ok(plan)
}

pub(super) fn run_j2k_ml_package_smoke(
    metadata: &serde_json::Value,
    manifest: &ReleaseManifestContract,
) -> Result<(), String> {
    let plan = package_gate_plan(metadata, manifest)?;
    let step = plan
        .iter()
        .find(|step| step.package == "j2k-ml")
        .ok_or_else(|| "package gate plan omitted `j2k-ml`".to_string())?;
    run_staged_package(step, true)?;
    run_j2k_ml_consumer_gate(step, &package_archive_path(metadata, step)?)
}

fn run_staged_package(step: &PackageGateStep, allow_dirty: bool) -> Result<(), String> {
    let mut args = vec![
        "package".to_string(),
        "-p".to_string(),
        step.package.clone(),
        "--no-verify".to_string(),
    ];
    if allow_dirty {
        args.push("--allow-dirty".to_string());
    }
    append_patch_config_args(&mut args, step)?;
    run_command_owned(cargo(), &args, CommandContext::new())
}

fn append_patch_config_args(args: &mut Vec<String>, step: &PackageGateStep) -> Result<(), String> {
    for (dependency, path) in &step.patches {
        args.push("--config".to_string());
        args.push(format!(
            "patch.crates-io.{dependency}.path={}",
            serde_json::to_string(path).map_err(|error| format!(
                "failed to quote patch path for `{dependency}`: {error}"
            ))?
        ));
    }
    Ok(())
}

pub(super) fn run(
    metadata: &serde_json::Value,
    manifest: &ReleaseManifestContract,
) -> Result<(), String> {
    for step in package_gate_plan(metadata, manifest)? {
        if step.registry_independent {
            run_cargo(&["publish", "-p", step.package.as_str(), "--dry-run"])?;
        } else {
            run_staged_package(&step, false)?;
        }
        if step.package == "j2k-ml" {
            run_j2k_ml_consumer_gate(&step, &package_archive_path(metadata, &step)?)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
