//! Dependency-aware construction of publishable workspace packages.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::command_support::run_cargo;
use crate::process::{cargo, run_command_owned, CommandContext};

use super::{
    package_name, str_set, workspace_package_records, PUBLISHABLE_PACKAGES,
    REGISTRY_INDEPENDENT_PACKAGES,
};

#[derive(Debug)]
struct PackageGateStep {
    package: &'static str,
    registry_independent: bool,
    patches: Vec<(String, String)>,
}

fn package_gate_plan(metadata: &serde_json::Value) -> Result<Vec<PackageGateStep>, String> {
    let packages = workspace_package_records(metadata)?;
    let package_by_name = packages
        .into_iter()
        .map(|package| Ok((package_name(package)?.to_string(), package)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let publishable = str_set(PUBLISHABLE_PACKAGES);
    let independent = str_set(REGISTRY_INDEPENDENT_PACKAGES);
    let dependencies_by_package = PUBLISHABLE_PACKAGES
        .iter()
        .map(|&package| {
            let record = package_by_name.get(package).ok_or_else(|| {
                format!("publishable package `{package}` is absent from cargo metadata")
            })?;
            let dependencies = record
                .get("dependencies")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    format!("cargo metadata package `{package}` has no dependency array")
                })?
                .iter()
                .filter(|dependency| {
                    dependency
                        .get("kind")
                        .is_none_or(serde_json::Value::is_null)
                        && dependency
                            .get("source")
                            .is_none_or(serde_json::Value::is_null)
                })
                .filter_map(|dependency| dependency.get("name")?.as_str())
                .filter(|dependency| publishable.contains(*dependency))
                .map(ToString::to_string)
                .collect::<BTreeSet<_>>();
            Ok((package.to_string(), dependencies))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut processed = BTreeSet::new();
    let mut plan = Vec::with_capacity(PUBLISHABLE_PACKAGES.len());

    for &package in PUBLISHABLE_PACKAGES {
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
            package,
            registry_independent: independent.contains(package),
            patches,
        });
        processed.insert(package);
    }
    Ok(plan)
}

fn run_staged_package(step: &PackageGateStep) -> Result<(), String> {
    let mut args = vec![
        "package".to_string(),
        "-p".to_string(),
        step.package.to_string(),
        "--no-verify".to_string(),
    ];
    for (dependency, path) in &step.patches {
        args.push("--config".to_string());
        args.push(format!(
            "patch.crates-io.{dependency}.path={}",
            serde_json::to_string(path).map_err(|error| format!(
                "failed to quote patch path for `{dependency}`: {error}"
            ))?
        ));
    }
    run_command_owned(cargo(), &args, CommandContext::new())
}

pub(super) fn run(metadata: &serde_json::Value) -> Result<(), String> {
    for step in package_gate_plan(metadata)? {
        if step.registry_independent {
            run_cargo(&["publish", "-p", step.package, "--dry-run"])?;
        } else {
            run_staged_package(&step)?;
        }
    }
    Ok(())
}
