// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProductionRustSource {
    pub(crate) absolute: PathBuf,
    pub(crate) relative: PathBuf,
}

pub(crate) fn production_rust_sources(
    repository_root: &Path,
    scopes: &[PathBuf],
) -> Result<Vec<ProductionRustSource>, String> {
    let mut sources = auditable_rust_sources(repository_root, scopes)?;
    sources.retain(|source| is_production_rust_path(&source.relative));
    if sources.is_empty() {
        return Err("production Rust source audit found no eligible sources".to_string());
    }
    Ok(sources)
}

pub(crate) fn auditable_rust_sources(
    repository_root: &Path,
    scopes: &[PathBuf],
) -> Result<Vec<ProductionRustSource>, String> {
    if scopes.is_empty() {
        return Err("Rust source audit requires at least one scope".to_string());
    }
    let mut sources = BTreeMap::new();
    for scope in scopes {
        scope.strip_prefix(repository_root).map_err(|_| {
            format!(
                "source-audit scope {} is outside repository root {}",
                scope.display(),
                repository_root.display()
            )
        })?;
        collect_rust_sources(repository_root, scope, &mut sources)?;
    }
    if sources.is_empty() {
        return Err("Rust source audit found no eligible sources".to_string());
    }
    Ok(sources.into_values().collect())
}

fn collect_rust_sources(
    repository_root: &Path,
    directory: &Path,
    sources: &mut BTreeMap<PathBuf, ProductionRustSource>,
) -> Result<(), String> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| {
            format!(
                "read source-audit directory {}: {error}",
                directory.display()
            )
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            format!(
                "read source-audit entry in {}: {error}",
                directory.display()
            )
        })?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let file_type = entry.file_type().map_err(|error| {
            format!(
                "inspect source-audit path {}: {error}",
                entry.path().display()
            )
        })?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(format!(
                "source-audit scope contains unsupported symlink {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            if entry.file_name().to_str().is_some_and(|name| {
                matches!(
                    name,
                    "benches" | "examples" | "fuzz" | "generated" | "target"
                )
            }) {
                continue;
            }
            collect_rust_sources(repository_root, &path, sources)?;
            continue;
        }
        if !file_type.is_file() || path.extension() != Some(OsStr::new("rs")) {
            continue;
        }
        let relative = path
            .strip_prefix(repository_root)
            .map_err(|_| {
                format!(
                    "source-audit path {} escaped repository root {}",
                    path.display(),
                    repository_root.display()
                )
            })?
            .to_path_buf();
        if !is_auditable_rust_path(&relative) {
            continue;
        }
        let source = ProductionRustSource {
            absolute: path,
            relative: relative.clone(),
        };
        if sources.insert(relative.clone(), source).is_some() {
            return Err(format!(
                "production Rust source audit discovered duplicate path {}",
                relative.display()
            ));
        }
    }
    Ok(())
}

fn is_auditable_rust_path(relative: &Path) -> bool {
    let components = normal_components(relative);
    if components.first() != Some(&"crates") || components.len() < 3 {
        return false;
    }
    if components.iter().any(|component| {
        matches!(
            *component,
            "benches" | "examples" | "fuzz" | "generated" | "target"
        )
    }) {
        return false;
    }
    relative.file_name() != Some(OsStr::new("build.rs"))
}

pub(crate) fn is_production_rust_path(relative: &Path) -> bool {
    let components = normal_components(relative);
    if !is_auditable_rust_path(relative) {
        return false;
    }
    if matches!(
        components.get(1),
        Some(&"j2k-test-support" | &"j2k-transcode-test-support")
    ) {
        return false;
    }
    if components
        .iter()
        .any(|component| matches!(*component, "tests" | "test_helpers" | "test_support"))
    {
        return false;
    }
    let Some(file_name) = relative.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    let Some(stem) = file_name.strip_suffix(".rs") else {
        return false;
    };
    stem != "tests"
        && stem != "test_helpers"
        && stem != "test_support"
        && !stem.starts_with("test_")
        && !stem.ends_with("_test")
        && !stem.ends_with("_tests")
}

fn normal_components(relative: &Path) -> Vec<&str> {
    relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
pub(super) fn production_path_for_test(relative: &str) -> bool {
    is_production_rust_path(Path::new(relative))
}
