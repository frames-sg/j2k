//! Ordered release-manifest parsing and workspace dependency validation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::{package_name, publish_false, PUBLISHABLE_PACKAGES, REGISTRY_INDEPENDENT_PACKAGES};

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ReleaseManifestContract {
    pub(super) ordered_crates: Vec<String>,
    pub(super) registry_independent: Vec<String>,
}

pub(super) fn release_manifest_contract() -> Result<ReleaseManifestContract, String> {
    let relative_path = Path::new("release-crates.json");
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .join(relative_path);
    let path = if relative_path.is_file() {
        relative_path
    } else {
        workspace_path.as_path()
    };
    let source = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    parse_release_manifest_source(&source)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

pub(super) fn parse_release_manifest_source(
    source: &str,
) -> Result<ReleaseManifestContract, String> {
    let value: serde_json::Value = serde_json::from_str(source)
        .map_err(|error| format!("release manifest is not valid JSON: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "release manifest root must be an object".to_string())?;
    if object.get("schema").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err("release manifest schema must be exactly 1".to_string());
    }

    let string_array = |field: &str| -> Result<Vec<String>, String> {
        object
            .get(field)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("release manifest {field} must be an array"))?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .map(ToString::to_string)
                    .ok_or_else(|| {
                        format!("release manifest {field}[{index}] must be a non-empty string")
                    })
            })
            .collect()
    };
    let ordered_crates = string_array("ordered_crates")?;
    let registry_independent = string_array("registry_independent")?;
    if ordered_crates.is_empty() {
        return Err("release manifest ordered_crates must not be empty".to_string());
    }
    if ordered_crates.iter().collect::<BTreeSet<_>>().len() != ordered_crates.len() {
        return Err("release manifest ordered_crates contains duplicates".to_string());
    }
    if registry_independent.iter().collect::<BTreeSet<_>>().len() != registry_independent.len() {
        return Err("release manifest registry_independent contains duplicates".to_string());
    }
    if ordered_crates.get(..registry_independent.len()) != Some(&registry_independent) {
        return Err(
            "release manifest registry_independent must be an ordered manifest prefix".to_string(),
        );
    }
    Ok(ReleaseManifestContract {
        ordered_crates,
        registry_independent,
    })
}

pub(super) fn validate_release_manifest_contract(
    manifest: &ReleaseManifestContract,
    workspace_packages: &[&serde_json::Value],
    errors: &mut Vec<String>,
) -> Result<(), String> {
    compare_declared_release_sets(manifest, errors);
    let package_by_name = workspace_packages
        .iter()
        .map(|package| Ok((package_name(package)?.to_string(), *package)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let actual_publishable = workspace_packages
        .iter()
        .filter(|package| !publish_false(package))
        .map(|package| package_name(package).map(ToString::to_string))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let manifest_publishable = manifest
        .ordered_crates
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if actual_publishable != manifest_publishable {
        errors.push(format!(
            "release-crates.json must contain every publishable workspace crate exactly once; actual={actual_publishable:?}, manifest={manifest_publishable:?}"
        ));
    }
    validate_dependency_order(manifest, &package_by_name, errors);
    Ok(())
}

fn compare_declared_release_sets(manifest: &ReleaseManifestContract, errors: &mut Vec<String>) {
    let expected = PUBLISHABLE_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if manifest.ordered_crates != expected {
        errors.push(format!(
            "release-crates.json order is {:?}, expected {:?}",
            manifest.ordered_crates, expected
        ));
    }
    let expected_independent = REGISTRY_INDEPENDENT_PACKAGES
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if manifest.registry_independent != expected_independent {
        errors.push(format!(
            "release-crates.json registry-independent prefix is {:?}, expected {:?}",
            manifest.registry_independent, expected_independent
        ));
    }
}

fn validate_dependency_order(
    manifest: &ReleaseManifestContract,
    package_by_name: &BTreeMap<String, &serde_json::Value>,
    errors: &mut Vec<String>,
) {
    let positions = manifest
        .ordered_crates
        .iter()
        .enumerate()
        .map(|(index, crate_name)| (crate_name.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    for (crate_index, crate_name) in manifest.ordered_crates.iter().enumerate() {
        let Some(package) = package_by_name.get(crate_name) else {
            continue;
        };
        let Some(dependencies) = package
            .get("dependencies")
            .and_then(serde_json::Value::as_array)
        else {
            errors.push(format!(
                "cargo metadata package `{crate_name}` has no dependencies array"
            ));
            continue;
        };
        for (dependency_index, dependency) in dependencies.iter().enumerate() {
            if dependency.get("kind").and_then(serde_json::Value::as_str) == Some("dev") {
                continue;
            }
            let Some(dependency_name) = dependency.get("name").and_then(serde_json::Value::as_str)
            else {
                errors.push(format!(
                    "cargo metadata package `{crate_name}` dependency[{dependency_index}] has no string name"
                ));
                continue;
            };
            if positions
                .get(dependency_name)
                .is_some_and(|dependency_position| *dependency_position >= crate_index)
            {
                errors.push(format!(
                    "release-crates.json places `{crate_name}` before dependency `{dependency_name}`"
                ));
            }
        }
    }
}
