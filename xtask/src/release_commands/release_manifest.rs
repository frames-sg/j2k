//! Ordered release-manifest parsing and workspace dependency validation.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::{has_lib_target, package_name};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ApiContract {
    Stable,
    Experimental,
    Implementation,
    Binary,
}

impl ApiContract {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "stable" => Ok(Self::Stable),
            "experimental" => Ok(Self::Experimental),
            "implementation" => Ok(Self::Implementation),
            "binary" => Ok(Self::Binary),
            _ => Err(format!(
                "release manifest api_contract `{value}` is not one of stable, experimental, implementation, or binary"
            )),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Experimental => "experimental",
            Self::Implementation => "implementation",
            Self::Binary => "binary",
        }
    }

    pub(super) const fn is_library(self) -> bool {
        !matches!(self, Self::Binary)
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ReleaseCrateContract {
    name: String,
    api_contract: ApiContract,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ReleaseManifestContract {
    crates: Vec<ReleaseCrateContract>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReleaseManifest {
    schema: u64,
    crates: Vec<RawReleaseCrate>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawReleaseCrate {
    name: String,
    api_contract: String,
}

impl ReleaseManifestContract {
    pub(super) fn ordered_crates(&self) -> impl ExactSizeIterator<Item = &str> {
        self.crates
            .iter()
            .map(|release_crate| release_crate.name.as_str())
    }

    pub(super) fn api_contracts(&self) -> impl Iterator<Item = (&str, ApiContract)> {
        self.crates
            .iter()
            .map(|release_crate| (release_crate.name.as_str(), release_crate.api_contract))
    }

    pub(super) fn library_packages(&self) -> impl Iterator<Item = &str> {
        self.crates
            .iter()
            .filter(|release_crate| release_crate.api_contract.is_library())
            .map(|release_crate| release_crate.name.as_str())
    }
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
    let raw: RawReleaseManifest = serde_json::from_str(source)
        .map_err(|error| format!("release manifest is not valid schema 2 JSON: {error}"))?;
    if raw.schema != 2 {
        return Err("release manifest schema must be exactly 2".to_string());
    }
    if raw.crates.is_empty() {
        return Err("release manifest crates must not be empty".to_string());
    }

    let mut crates = Vec::new();
    crates
        .try_reserve_exact(raw.crates.len())
        .map_err(|error| format!("reserve release manifest crates: {error}"))?;
    let mut names = BTreeSet::new();
    for (index, record) in raw.crates.into_iter().enumerate() {
        if record.name.is_empty() {
            return Err(format!(
                "release manifest crates[{index}].name must be a non-empty string"
            ));
        }
        if !valid_crate_name(&record.name) {
            return Err(format!(
                "release manifest crates[{index}] has malformed crate name `{}`",
                record.name
            ));
        }
        if !names.insert(record.name.clone()) {
            return Err(format!(
                "release manifest crates contains duplicate crate `{}`",
                record.name
            ));
        }
        let api_contract = ApiContract::parse(&record.api_contract)?;
        crates.push(ReleaseCrateContract {
            name: record.name,
            api_contract,
        });
    }
    Ok(ReleaseManifestContract { crates })
}

fn valid_crate_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphanumeric())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

pub(super) fn registry_independent_packages(
    manifest: &ReleaseManifestContract,
    workspace_packages: &[&serde_json::Value],
) -> Result<BTreeSet<String>, String> {
    let dependencies_by_package = release_dependencies_by_package(manifest, workspace_packages)?;
    Ok(dependencies_by_package
        .into_iter()
        .filter_map(|(name, dependencies)| dependencies.is_empty().then_some(name))
        .collect())
}

pub(super) fn release_dependencies_by_package(
    manifest: &ReleaseManifestContract,
    workspace_packages: &[&serde_json::Value],
) -> Result<BTreeMap<String, BTreeSet<String>>, String> {
    let package_by_name = package_records_by_name(workspace_packages)?;
    let release_roots = release_package_roots(manifest, &package_by_name)?;
    let mut dependencies_by_package = BTreeMap::new();
    for package_name in manifest.ordered_crates() {
        let Some(package) = package_by_name.get(package_name) else {
            dependencies_by_package.insert(package_name.to_string(), BTreeSet::new());
            continue;
        };
        let dependencies = package
            .get("dependencies")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!("cargo metadata package `{package_name}` has no dependencies array")
            })?;
        let release_dependencies = dependencies.iter().enumerate().try_fold(
            BTreeSet::new(),
            |mut found, (index, dependency)| {
                let kind = dependency_kind(package_name, index, dependency)?;
                if kind != "dev" {
                    if let Some(dependency_name) = workspace_release_dependency(
                        package_name,
                        index,
                        dependency,
                        &release_roots,
                    )? {
                        found.insert(dependency_name.to_string());
                    }
                }
                Ok::<_, String>(found)
            },
        )?;
        dependencies_by_package.insert(package_name.to_string(), release_dependencies);
    }
    Ok(dependencies_by_package)
}

pub(super) fn validate_release_manifest_contract(
    manifest: &ReleaseManifestContract,
    workspace_packages: &[&serde_json::Value],
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let package_by_name = package_records_by_name(workspace_packages)?;
    let actual_publishable = workspace_packages
        .iter()
        .map(|package| {
            let name = package_name(package)?;
            Ok((name, crates_io_publishable(package, name)?))
        })
        .filter_map(|result| match result {
            Ok((name, true)) => Some(Ok(name.to_string())),
            Ok((_, false)) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<BTreeSet<_>, String>>()?;
    let manifest_publishable = manifest
        .ordered_crates()
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();
    if actual_publishable != manifest_publishable {
        errors.push(format!(
            "release-crates.json must contain every publishable workspace crate exactly once; actual={actual_publishable:?}, manifest={manifest_publishable:?}"
        ));
    }
    validate_common_release_version(manifest, &package_by_name, errors);
    validate_contract_targets(manifest, &package_by_name, errors)?;
    let dependencies_by_package = release_dependencies_by_package(manifest, workspace_packages)?;
    validate_dependency_order(manifest, &dependencies_by_package, errors);
    validate_exact_release_dependencies(manifest, &package_by_name, errors)?;
    Ok(())
}

fn validate_common_release_version(
    manifest: &ReleaseManifestContract,
    package_by_name: &BTreeMap<String, &serde_json::Value>,
    errors: &mut Vec<String>,
) {
    let mut versions = BTreeSet::new();
    for name in manifest.ordered_crates() {
        let Some(package) = package_by_name.get(name) else {
            continue;
        };
        let Some(version) = package.get("version").and_then(serde_json::Value::as_str) else {
            errors.push(format!(
                "cargo metadata package `{name}` has no string version"
            ));
            continue;
        };
        versions.insert(version);
    }
    if versions.len() > 1 {
        errors.push(format!(
            "release manifest packages must share one release version, found {versions:?}"
        ));
    }
}

pub(super) fn crates_io_publishable(
    package: &serde_json::Value,
    name: &str,
) -> Result<bool, String> {
    match package.get("publish") {
        None | Some(serde_json::Value::Null) => Ok(true),
        Some(serde_json::Value::Array(registries)) => {
            let mut crates_io = false;
            for (index, registry) in registries.iter().enumerate() {
                let registry = registry.as_str().ok_or_else(|| {
                    format!("cargo metadata package `{name}` publish[{index}] is not a string")
                })?;
                crates_io |= registry == "crates-io";
            }
            Ok(crates_io)
        }
        Some(_) => Err(format!(
            "cargo metadata package `{name}` has invalid publish eligibility"
        )),
    }
}

fn package_records_by_name<'a>(
    workspace_packages: &[&'a serde_json::Value],
) -> Result<BTreeMap<String, &'a serde_json::Value>, String> {
    let mut package_by_name = BTreeMap::new();
    for package in workspace_packages {
        let name = package_name(package)?.to_string();
        if package_by_name.insert(name.clone(), *package).is_some() {
            return Err(format!(
                "cargo metadata contains duplicate workspace package name `{name}`"
            ));
        }
    }
    Ok(package_by_name)
}

fn validate_contract_targets(
    manifest: &ReleaseManifestContract,
    package_by_name: &BTreeMap<String, &serde_json::Value>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    for (name, contract) in manifest.api_contracts() {
        let Some(package) = package_by_name.get(name) else {
            continue;
        };
        let targets = package
            .get("targets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("cargo metadata package `{name}` has no targets array"))?;
        let mut kinds = BTreeSet::new();
        for (target_index, target) in targets.iter().enumerate() {
            let target_kinds = target
                .get("kind")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    format!(
                        "cargo metadata package `{name}` target[{target_index}] has no kind array"
                    )
                })?;
            for (kind_index, kind) in target_kinds.iter().enumerate() {
                kinds.insert(kind.as_str().ok_or_else(|| {
                    format!(
                        "cargo metadata package `{name}` target[{target_index}].kind[{kind_index}] is not a string"
                    )
                })?);
            }
        }
        let has_library = has_lib_target(package);
        let has_binary = kinds.contains("bin");
        match contract {
            ApiContract::Binary if !has_binary || has_library => errors.push(format!(
                "`{name}` has api_contract=binary but must have a binary target and no library target"
            )),
            ApiContract::Stable | ApiContract::Experimental | ApiContract::Implementation
                if !has_library =>
            {
                errors.push(format!(
                    "`{name}` has api_contract={} but has no library target",
                    contract.as_str()
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_exact_release_dependencies(
    manifest: &ReleaseManifestContract,
    package_by_name: &BTreeMap<String, &serde_json::Value>,
    errors: &mut Vec<String>,
) -> Result<(), String> {
    let release_roots = release_package_roots(manifest, package_by_name)?;
    for package_name in manifest.ordered_crates() {
        let Some(package) = package_by_name.get(package_name) else {
            continue;
        };
        let dependencies = package
            .get("dependencies")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!("cargo metadata package `{package_name}` has no dependencies array")
            })?;
        for (index, dependency) in dependencies.iter().enumerate() {
            let Some(dependency_name) =
                workspace_release_dependency(package_name, index, dependency, &release_roots)?
            else {
                continue;
            };
            let Some(dependency_version) = package_by_name
                .get(dependency_name)
                .and_then(|dependency_package| dependency_package.get("version"))
                .and_then(serde_json::Value::as_str)
            else {
                errors.push(format!(
                    "cargo metadata release dependency `{dependency_name}` has no string version"
                ));
                continue;
            };
            let requirement = dependency
                .get("req")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    format!(
                        "cargo metadata package `{package_name}` dependency `{dependency_name}` has no string requirement"
                    )
                })?;
            let exact = format!("={dependency_version}");
            if requirement != exact {
                errors.push(format!(
                    "`{package_name}` dependency `{dependency_name}` must use exact release requirement `{exact}`, found `{requirement}`"
                ));
            }
        }
    }
    Ok(())
}

fn validate_dependency_order(
    manifest: &ReleaseManifestContract,
    dependencies_by_package: &BTreeMap<String, BTreeSet<String>>,
    errors: &mut Vec<String>,
) {
    let positions = manifest
        .ordered_crates()
        .enumerate()
        .map(|(index, crate_name)| (crate_name, index))
        .collect::<BTreeMap<_, _>>();
    for (crate_index, crate_name) in manifest.ordered_crates().enumerate() {
        let Some(dependencies) = dependencies_by_package.get(crate_name) else {
            continue;
        };
        for dependency_name in dependencies {
            if positions
                .get(dependency_name.as_str())
                .is_some_and(|dependency_position| *dependency_position >= crate_index)
            {
                errors.push(format!(
                    "release-crates.json places `{crate_name}` before dependency `{dependency_name}`"
                ));
            }
        }
    }
}

fn release_package_roots(
    manifest: &ReleaseManifestContract,
    package_by_name: &BTreeMap<String, &serde_json::Value>,
) -> Result<BTreeMap<String, String>, String> {
    manifest
        .ordered_crates()
        .filter_map(|name| package_by_name.get(name).map(|package| (name, *package)))
        .map(|(name, package)| {
            let manifest_path = package
                .get("manifest_path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    format!("cargo metadata package `{name}` has no string manifest_path")
                })?;
            let root = Path::new(manifest_path).parent().ok_or_else(|| {
                format!("cargo metadata package `{name}` manifest_path has no parent")
            })?;
            Ok((name.to_string(), root.to_string_lossy().into_owned()))
        })
        .collect()
}

fn workspace_release_dependency<'a>(
    package: &str,
    index: usize,
    dependency: &'a serde_json::Value,
    release_roots: &BTreeMap<String, String>,
) -> Result<Option<&'a str>, String> {
    let dependency_name = dependency
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!("cargo metadata package `{package}` dependency[{index}] has no string name")
        })?;
    let Some(expected_root) = release_roots.get(dependency_name) else {
        return Ok(None);
    };
    if dependency
        .get("source")
        .is_some_and(|source| !source.is_null())
    {
        return Err(format!(
            "release package `{package}` dependency `{dependency_name}` must be workspace/path sourced"
        ));
    }
    let Some(path) = dependency.get("path").and_then(serde_json::Value::as_str) else {
        return Err(format!(
            "release package `{package}` dependency `{dependency_name}` must be workspace/path sourced"
        ));
    };
    if Path::new(path) != Path::new(expected_root) {
        return Err(format!(
            "release package `{package}` dependency `{dependency_name}` path `{path}` does not match workspace package root `{expected_root}`"
        ));
    }
    Ok(Some(dependency_name))
}

fn dependency_kind<'a>(
    package: &str,
    index: usize,
    dependency: &'a serde_json::Value,
) -> Result<&'a str, String> {
    match dependency.get("kind") {
        None | Some(serde_json::Value::Null) => Ok("normal"),
        Some(serde_json::Value::String(kind)) => Ok(kind),
        Some(_) => Err(format!(
            "cargo metadata package `{package}` dependency[{index}] has invalid kind"
        )),
    }
}
