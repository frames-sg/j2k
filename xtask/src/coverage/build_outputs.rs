// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

const MAX_BUILD_OUTPUT_DEPTH: u8 = 5;

mod target;
#[cfg(test)]
mod tests;

pub(super) use target::CurrentBuildTarget;

#[derive(Debug)]
struct ScannedOutput {
    content: Vec<u8>,
}

#[derive(Debug)]
struct BuildOutputScope {
    relative: String,
    build_directory: String,
}

#[derive(Debug)]
struct CurrentBuildOutput<'a> {
    scope: String,
    content: &'a [u8],
}

#[derive(Debug)]
pub(super) struct BuildOutputEvidence {
    target_dir: PathBuf,
    outputs: BTreeMap<PathBuf, ScannedOutput>,
}

impl BuildOutputEvidence {
    pub(super) fn capture(mut target: CurrentBuildTarget) -> Result<Self, String> {
        let target_dir = target.path()?.to_path_buf();
        let outputs = scan_outputs(&target_dir)?;
        target.cleanup()?;
        Ok(Self {
            target_dir,
            outputs,
        })
    }

    pub(super) fn current_cfg_flags(
        &self,
        selected_packages: &BTreeSet<String>,
        build_script_packages: &BTreeSet<String>,
    ) -> Result<BTreeMap<String, BTreeMap<String, bool>>, String> {
        if !build_script_packages.is_subset(selected_packages) {
            return Err(
                "coverage build-script package set is not a subset of selected packages"
                    .to_string(),
            );
        }
        let mut current_by_package = BTreeMap::<String, Vec<CurrentBuildOutput<'_>>>::new();
        for (path, output) in &self.outputs {
            let Some(scope) = build_output_scope(&self.target_dir, path)? else {
                continue;
            };
            let Some(package) = selected_package(&scope.build_directory, selected_packages)? else {
                continue;
            };
            current_by_package
                .entry(package.to_string())
                .or_default()
                .push(CurrentBuildOutput {
                    scope: scope.relative,
                    content: &output.content,
                });
        }
        reconcile_cfg_flags(current_by_package, build_script_packages)
    }
}

fn scan_outputs(target_dir: &Path) -> Result<BTreeMap<PathBuf, ScannedOutput>, String> {
    if !target_dir.exists() {
        return Ok(BTreeMap::new());
    }
    if !target_dir.is_dir() {
        return Err(format!(
            "coverage target path {} is not a directory",
            target_dir.display()
        ));
    }
    let mut outputs = BTreeMap::new();
    let mut pending = VecDeque::from([(target_dir.to_path_buf(), 0_u8)]);
    while let Some((directory, depth)) = pending.pop_front() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("failed to inspect {}: {error}", directory.display()))?
        {
            let entry = entry.map_err(|error| {
                format!(
                    "failed to inspect entry under {}: {error}",
                    directory.display()
                )
            })?;
            let file_type = entry.file_type().map_err(|error| {
                format!("failed to inspect {}: {error}", entry.path().display())
            })?;
            let path = entry.path();
            if file_type.is_dir() {
                if depth < MAX_BUILD_OUTPUT_DEPTH {
                    pending.push_back((path, depth + 1));
                }
                continue;
            }
            if !file_type.is_file()
                || path.file_name().and_then(|name| name.to_str()) != Some("output")
                || build_output_scope(target_dir, &path)?.is_none()
            {
                continue;
            }
            let content = fs::read(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            outputs.insert(path, ScannedOutput { content });
        }
    }
    Ok(outputs)
}

fn build_output_scope(
    target_dir: &Path,
    output: &Path,
) -> Result<Option<BuildOutputScope>, String> {
    let Some(build_directory) = output.parent() else {
        return Ok(None);
    };
    if build_directory
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        != Some("build")
    {
        return Ok(None);
    }
    let relative = build_directory.strip_prefix(target_dir).map_err(|_| {
        format!(
            "build output scope {} is outside coverage target directory {}",
            build_directory.display(),
            target_dir.display()
        )
    })?;
    let relative = relative
        .to_str()
        .map(|path| path.replace(std::path::MAIN_SEPARATOR, "/"))
        .ok_or_else(|| {
            format!(
                "coverage build output scope is not UTF-8: {}",
                relative.display()
            )
        })?;
    let build_directory = build_directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            format!(
                "coverage build directory is not UTF-8: {}",
                build_directory.display()
            )
        })?;
    Ok(Some(BuildOutputScope {
        relative,
        build_directory: build_directory.to_string(),
    }))
}

fn selected_package<'a>(
    build_directory: &str,
    selected_packages: &'a BTreeSet<String>,
) -> Result<Option<&'a str>, String> {
    let mut selected = None;
    for package in selected_packages {
        let Some(hash) = build_directory
            .strip_prefix(package)
            .and_then(|suffix| suffix.strip_prefix('-'))
        else {
            continue;
        };
        if hash.len() < 8 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            continue;
        }
        if let Some(existing) = selected {
            return Err(format!(
                "coverage build directory `{build_directory}` matches both selected packages `{existing}` and `{package}`"
            ));
        }
        selected = Some(package.as_str());
    }
    Ok(selected)
}

fn reconcile_cfg_flags(
    current_by_package: BTreeMap<String, Vec<CurrentBuildOutput<'_>>>,
    build_script_packages: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeMap<String, bool>>, String> {
    let mut cfg_by_package = BTreeMap::new();
    for (package, outputs) in current_by_package {
        if !build_script_packages.contains(&package) {
            return Err(format!(
                "current coverage build produced build-script output for selected package `{package}` without a custom-build target"
            ));
        }
        let mut agreed = None::<(String, BTreeMap<String, bool>)>;
        for output in outputs {
            let flags = parse_build_cfg_output(&output)?;
            if let Some((scope, expected)) = &agreed {
                if expected != &flags {
                    return Err(format!(
                        "current build-script cfg outputs for package `{package}` conflict between scopes `{scope}` and `{}`",
                        output.scope
                    ));
                }
            } else {
                agreed = Some((output.scope, flags));
            }
        }
        if let Some((_, flags)) = agreed {
            cfg_by_package.insert(package, flags);
        }
    }
    for package in build_script_packages {
        if !cfg_by_package.contains_key(package) {
            return Err(format!(
                "current coverage build produced no build-script output for selected package `{package}`"
            ));
        }
    }
    Ok(cfg_by_package)
}

fn parse_build_cfg_output(
    output: &CurrentBuildOutput<'_>,
) -> Result<BTreeMap<String, bool>, String> {
    let source = std::str::from_utf8(output.content).map_err(|error| {
        format!(
            "coverage build cfg output `{}` is not UTF-8: {error}",
            output.scope
        )
    })?;
    let mut flags = BTreeMap::new();
    for line in source.lines() {
        if let Some(value) = cargo_directive(line, "rustc-check-cfg") {
            if let Some(name) = checked_cfg_name(value) {
                flags.entry(name.to_string()).or_insert(false);
            }
        }
    }
    for line in source.lines() {
        if let Some(value) = cargo_directive(line, "rustc-cfg") {
            let name = value.split('=').next().unwrap_or(value).trim();
            if is_cfg_identifier(name) {
                flags.insert(name.to_string(), true);
            }
        }
    }
    Ok(flags)
}

fn cargo_directive<'a>(line: &'a str, directive: &str) -> Option<&'a str> {
    line.strip_prefix(&format!("cargo:{directive}="))
        .or_else(|| line.strip_prefix(&format!("cargo::{directive}=")))
}

fn checked_cfg_name(value: &str) -> Option<&str> {
    let body = value.strip_prefix("cfg(")?;
    let name = body.split([',', ')']).next()?.trim();
    is_cfg_identifier(name).then_some(name)
}

fn is_cfg_identifier(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}
