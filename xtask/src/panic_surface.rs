// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::process::Command;

use crate::process::cargo;

mod source_inventory;

use source_inventory::enforce_panic_macro_inventory;

const PANIC_SURFACE_UNWRAP_USED_BASELINE: usize = 16;
const PANIC_SURFACE_EXPECT_USED_BASELINE: usize = 50;
const PANIC_SURFACE_DEV_ONLY_FEATURES: &[(&str, &str)] = &[
    ("j2k-jpeg", "bench-internals"),
    ("j2k-jpeg", "bench-libjpeg-turbo"),
    ("j2k-transcode", "dev-support"),
    ("j2k-transcode-metal", "bench-internals"),
];
const LIBRARY_CRATE_TYPES: &[&str] = &["lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"];
#[derive(Debug, PartialEq, Eq)]
struct PanicSurfaceSelection {
    packages: Vec<String>,
    features: Vec<String>,
}

pub(super) fn panic_surface() -> Result<(), String> {
    let metadata = Command::new(cargo())
        .args(["metadata", "--no-deps", "--format-version=1"])
        .output()
        .map_err(|err| format!("failed to load Cargo metadata for panic-surface gate: {err}"))?;
    if !metadata.status.success() {
        let stdout = String::from_utf8_lossy(&metadata.stdout);
        let stderr = String::from_utf8_lossy(&metadata.stderr);
        return Err(format!(
            "cargo metadata for panic-surface gate failed with status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            metadata.status
        ));
    }
    let metadata = String::from_utf8(metadata.stdout)
        .map_err(|error| format!("cargo metadata stdout is not UTF-8: {error}"))?;
    let selection = parse_panic_surface_selection(&metadata, PANIC_SURFACE_DEV_ONLY_FEATURES)?;
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| "xtask manifest directory has no repository parent".to_string())?;
    let macro_inventory =
        enforce_panic_macro_inventory(&metadata, &selection.packages, repository_root)?;
    let args = panic_surface_clippy_args(&selection);

    let output = Command::new(cargo())
        .args(args)
        .output()
        .map_err(|err| format!("failed to run cargo clippy panic-surface gate: {err}"))?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo clippy panic-surface gate failed with status {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        ));
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|error| format!("cargo clippy panic-surface stdout is not UTF-8: {error}"))?;

    let (unwrap_used_count, expect_used_count) = parse_panic_surface_output(&stdout)?;

    if unwrap_used_count > PANIC_SURFACE_UNWRAP_USED_BASELINE {
        return Err(format!(
            "panic-surface ratchet exceeded: clippy::unwrap_used reported {unwrap_used_count}, baseline is {PANIC_SURFACE_UNWRAP_USED_BASELINE}"
        ));
    }
    if expect_used_count > PANIC_SURFACE_EXPECT_USED_BASELINE {
        return Err(format!(
            "panic-surface ratchet exceeded: clippy::expect_used reported {expect_used_count}, baseline is {PANIC_SURFACE_EXPECT_USED_BASELINE}"
        ));
    }

    println!(
        "panic-surface ratchet across {} publishable library packages: clippy::unwrap_used {unwrap_used_count}/{PANIC_SURFACE_UNWRAP_USED_BASELINE}, clippy::expect_used {expect_used_count}/{PANIC_SURFACE_EXPECT_USED_BASELINE}; explicit production macros: {macro_inventory}",
        selection.packages.len(),
    );
    Ok(())
}

fn parse_panic_surface_selection(
    metadata: &str,
    dev_only_features: &[(&str, &str)],
) -> Result<PanicSurfaceSelection, String> {
    let metadata = serde_json::from_str::<serde_json::Value>(metadata)
        .map_err(|error| format!("cargo metadata for panic-surface gate is malformed: {error}"))?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata has no `packages` array".to_string())?;
    let workspace_members = metadata
        .get("workspace_members")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "cargo metadata has no `workspace_members` array".to_string())?;
    if workspace_members.is_empty() {
        return Err("cargo metadata has no workspace members".to_string());
    }

    let dev_only = dev_only_features
        .iter()
        .map(|(package, feature)| ((*package).to_string(), (*feature).to_string()))
        .collect::<BTreeSet<_>>();
    if dev_only.len() != dev_only_features.len() {
        return Err("panic-surface dev-only feature registry contains duplicates".to_string());
    }

    let mut remaining_dev_only = dev_only.clone();
    let mut visited_members = BTreeSet::new();
    let mut selected_packages = BTreeSet::new();
    let mut selected_features = BTreeSet::new();
    for member in workspace_members {
        let member_id = member.as_str().ok_or_else(|| {
            "cargo metadata contains a non-string workspace member ID".to_string()
        })?;
        if !visited_members.insert(member_id) {
            return Err(format!(
                "cargo metadata contains duplicate workspace member `{member_id}`"
            ));
        }
        let package = packages
            .iter()
            .find(|package| {
                package.get("id").and_then(serde_json::Value::as_str) == Some(member_id)
            })
            .ok_or_else(|| {
                format!("cargo metadata omits workspace member package `{member_id}`")
            })?;
        let Some(name) = publishable_library_name(package, member_id)? else {
            continue;
        };
        if !selected_packages.insert(name.to_string()) {
            return Err(format!(
                "cargo metadata contains duplicate publishable library package name `{name}`"
            ));
        }

        let dependencies = dependency_packages(package, name)?;
        collect_production_features(
            package,
            name,
            &dependencies,
            &dev_only,
            &mut remaining_dev_only,
            &mut selected_features,
        )?;
    }

    if selected_packages.is_empty() {
        return Err("cargo metadata has no publishable library packages".to_string());
    }
    if !remaining_dev_only.is_empty() {
        return Err(format!(
            "panic-surface dev-only feature registry has stale entries: {remaining_dev_only:?}"
        ));
    }

    Ok(PanicSurfaceSelection {
        packages: selected_packages.into_iter().collect(),
        features: selected_features.into_iter().collect(),
    })
}

fn publishable_library_name<'a>(
    package: &'a serde_json::Value,
    member_id: &str,
) -> Result<Option<&'a str>, String> {
    let name = package
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("workspace package `{member_id}` has no string `name`"))?;
    let publishable = match package.get("publish") {
        Some(serde_json::Value::Null) => true,
        Some(serde_json::Value::Array(registries)) => {
            if registries.iter().any(|registry| !registry.is_string()) {
                return Err(format!(
                    "workspace package `{name}` has a non-string publish registry"
                ));
            }
            !registries.is_empty()
        }
        Some(_) => {
            return Err(format!(
                "workspace package `{name}` has an invalid `publish` value"
            ));
        }
        None => {
            return Err(format!("workspace package `{name}` has no `publish` field"));
        }
    };
    let targets = package
        .get("targets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("workspace package `{name}` has no `targets` array"))?;
    let mut has_library_target = false;
    for target in targets {
        let crate_types = target
            .get("crate_types")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                format!("workspace package `{name}` has a target without `crate_types`")
            })?;
        for crate_type in crate_types {
            let crate_type = crate_type
                .as_str()
                .ok_or_else(|| format!("workspace package `{name}` has a non-string crate type"))?;
            has_library_target |= LIBRARY_CRATE_TYPES.contains(&crate_type);
        }
    }

    Ok((publishable && has_library_target).then_some(name))
}

fn dependency_packages(
    package: &serde_json::Value,
    name: &str,
) -> Result<BTreeMap<String, String>, String> {
    let dependencies = package
        .get("dependencies")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("workspace package `{name}` has no `dependencies` array"))?;
    let mut package_by_dependency = BTreeMap::new();
    for dependency in dependencies {
        let dependency_name = dependency
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!("workspace package `{name}` has a dependency without a string `name`")
            })?;
        let dependency_key = match dependency.get("rename") {
            Some(serde_json::Value::Null) | None => dependency_name,
            Some(rename) => rename.as_str().ok_or_else(|| {
                format!(
                    "workspace package `{name}` dependency `{dependency_name}` has an invalid rename"
                )
            })?,
        };
        if package_by_dependency
            .insert(dependency_key.to_string(), dependency_name.to_string())
            .is_some_and(|previous| previous != dependency_name)
        {
            return Err(format!(
                "workspace package `{name}` contains duplicate dependency key `{dependency_key}`"
            ));
        }
    }
    Ok(package_by_dependency)
}

fn collect_production_features(
    package: &serde_json::Value,
    name: &str,
    dependencies: &BTreeMap<String, String>,
    dev_only: &BTreeSet<(String, String)>,
    remaining_dev_only: &mut BTreeSet<(String, String)>,
    selected_features: &mut BTreeSet<String>,
) -> Result<(), String> {
    let features = package
        .get("features")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("workspace package `{name}` has no `features` object"))?;
    for (feature, dependencies_enabled) in features {
        let dependencies_enabled = dependencies_enabled.as_array().ok_or_else(|| {
            format!("workspace package `{name}` feature `{feature}` has a non-array definition")
        })?;
        let activations = dependencies_enabled
            .iter()
            .map(|activation| {
                activation.as_str().ok_or_else(|| {
                    format!(
                        "workspace package `{name}` feature `{feature}` contains a non-string entry"
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let key = (name.to_string(), feature.clone());
        if remaining_dev_only.remove(&key) {
            continue;
        }

        for activation in activations {
            let activated = activated_feature(name, feature, activation, dependencies)?;
            if let Some((activated_package, activated_feature)) = activated {
                if dev_only.contains(&(activated_package.clone(), activated_feature.clone())) {
                    return Err(format!(
                        "panic-surface production feature `{name}/{feature}` enables dev-only feature `{activated_package}/{activated_feature}`"
                    ));
                }
            }
        }
        selected_features.insert(format!("{name}/{feature}"));
    }
    Ok(())
}

fn activated_feature(
    package: &str,
    feature: &str,
    activation: &str,
    dependencies: &BTreeMap<String, String>,
) -> Result<Option<(String, String)>, String> {
    if activation.starts_with("dep:") {
        return Ok(None);
    }
    if let Some((dependency, dependency_feature)) = activation.split_once('/') {
        let dependency = dependency.strip_suffix('?').unwrap_or(dependency);
        let dependency_package = dependencies.get(dependency).ok_or_else(|| {
            format!(
                "workspace package `{package}` feature `{feature}` references unknown dependency `{dependency}`"
            )
        })?;
        return Ok(Some((
            dependency_package.clone(),
            dependency_feature.to_string(),
        )));
    }
    Ok(Some((package.to_string(), activation.to_string())))
}
fn panic_surface_clippy_args(selection: &PanicSurfaceSelection) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("clippy"),
        OsString::from("--lib"),
        OsString::from("--no-default-features"),
    ];
    for package in &selection.packages {
        args.push(OsString::from("--package"));
        args.push(OsString::from(package));
    }
    if !selection.features.is_empty() {
        args.push(OsString::from("--features"));
        args.push(OsString::from(selection.features.join(",")));
    }
    args.extend([
        OsString::from("--message-format=json"),
        OsString::from("--"),
        OsString::from("-A"),
        OsString::from("clippy::all"),
        OsString::from("-W"),
        OsString::from("clippy::unwrap_used"),
        OsString::from("-W"),
        OsString::from("clippy::expect_used"),
    ]);
    args
}

fn parse_panic_surface_output(stdout: &str) -> Result<(usize, usize), String> {
    let mut unwrap_used_count = 0usize;
    let mut expect_used_count = 0usize;
    let mut build_finished = None;

    for (index, line) in stdout.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if build_finished.is_some() {
            return Err(format!(
                "cargo clippy panic-surface emitted a record after build-finished on line {}",
                index + 1
            ));
        }
        let message = serde_json::from_str::<serde_json::Value>(line).map_err(|error| {
            format!(
                "cargo clippy panic-surface emitted malformed JSON on line {}: {error}",
                index + 1
            )
        })?;
        let reason = message
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "cargo clippy panic-surface JSON line {} has no string `reason`",
                    index + 1
                )
            })?;
        if reason == "build-finished" {
            build_finished = Some(
                message
                    .get("success")
                    .and_then(serde_json::Value::as_bool)
                    .ok_or_else(|| {
                        "cargo clippy panic-surface build-finished record has no boolean `success`"
                            .to_string()
                    })?,
            );
            continue;
        }
        if reason != "compiler-message" {
            continue;
        }
        if let Some(code) = message
            .get("message")
            .and_then(|message| message.get("code"))
            .and_then(|code| code.get("code"))
            .and_then(serde_json::Value::as_str)
        {
            match code {
                "clippy::unwrap_used" => unwrap_used_count += 1,
                "clippy::expect_used" => expect_used_count += 1,
                _ => {}
            }
        }
    }

    match build_finished {
        Some(true) => Ok((unwrap_used_count, expect_used_count)),
        Some(false) => Err("cargo clippy panic-surface reported an unsuccessful build".to_string()),
        None => Err(
            "cargo clippy panic-surface output is missing the terminal build-finished record"
                .to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{
        panic_surface_clippy_args, parse_panic_surface_output, parse_panic_surface_selection,
        PanicSurfaceSelection,
    };

    #[test]
    fn panic_surface_selection_excludes_non_publishable_support_and_dev_features() {
        let metadata = serde_json::json!({
            "workspace_members": ["codec-id", "support-id", "tool-id"],
            "packages": [
                {
                    "id": "codec-id",
                    "name": "codec",
                    "publish": null,
                    "targets": [{ "crate_types": ["lib"] }],
                    "dependencies": [],
                    "features": {
                        "default": [],
                        "accelerated": [],
                        "bench-internals": []
                    }
                },
                {
                    "id": "support-id",
                    "name": "codec-test-support",
                    "publish": [],
                    "targets": [{ "crate_types": ["lib"] }],
                    "dependencies": [],
                    "features": { "fixture-panics": [] }
                },
                {
                    "id": "tool-id",
                    "name": "workspace-tool",
                    "publish": null,
                    "targets": [{ "crate_types": ["bin"] }],
                    "dependencies": [],
                    "features": {}
                }
            ]
        })
        .to_string();

        let selection =
            parse_panic_surface_selection(&metadata, &[("codec", "bench-internals")]).unwrap();

        assert_eq!(
            selection,
            PanicSurfaceSelection {
                packages: vec!["codec".to_string()],
                features: vec!["codec/accelerated".to_string(), "codec/default".to_string()],
            }
        );
    }

    #[test]
    fn panic_surface_selection_cannot_bypass_a_new_publishable_library() {
        let metadata = serde_json::json!({
            "workspace_members": ["codec-id", "future-id"],
            "packages": [
                {
                    "id": "codec-id",
                    "name": "codec",
                    "publish": null,
                    "targets": [{ "crate_types": ["lib"] }],
                    "dependencies": [],
                    "features": {}
                },
                {
                    "id": "future-id",
                    "name": "future-codec",
                    "publish": ["crates-io"],
                    "targets": [{ "crate_types": ["rlib"] }],
                    "dependencies": [],
                    "features": { "experimental": [] }
                }
            ]
        })
        .to_string();

        let selection = parse_panic_surface_selection(&metadata, &[]).unwrap();
        let args = panic_surface_clippy_args(&selection);

        assert_eq!(
            selection.packages,
            vec!["codec".to_string(), "future-codec".to_string()]
        );
        assert!(selection
            .features
            .contains(&"future-codec/experimental".to_string()));
        assert!(args
            .windows(2)
            .any(|pair| pair == [OsStr::new("--package"), OsStr::new("future-codec")]));
    }

    #[test]
    fn panic_surface_selection_rejects_stale_dev_feature_registry_entries() {
        let metadata = serde_json::json!({
            "workspace_members": ["codec-id"],
            "packages": [{
                "id": "codec-id",
                "name": "codec",
                "publish": null,
                "targets": [{ "crate_types": ["lib"] }],
                "dependencies": [],
                "features": { "default": [] }
            }]
        })
        .to_string();

        let error =
            parse_panic_surface_selection(&metadata, &[("codec", "removed-feature")]).unwrap_err();

        assert!(error.contains("dev-only feature registry has stale entries"));
        assert!(error.contains("removed-feature"));
    }

    #[test]
    fn panic_surface_parser_counts_lints_and_requires_successful_finish() {
        let output = concat!(
            r#"{"reason":"compiler-artifact"}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"code":{"code":"clippy::unwrap_used"}}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"code":{"code":"clippy::expect_used"}}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"code":{"code":"clippy::expect_used"}}}"#,
            "\n",
            r#"{"reason":"build-finished","success":true}"#,
            "\n",
        );

        assert_eq!(parse_panic_surface_output(output).unwrap(), (1, 2));
    }

    #[test]
    fn panic_surface_parser_rejects_malformed_or_incomplete_output() {
        let malformed = concat!(
            r#"{"reason":"compiler-artifact"}"#,
            "\nnot-json\n",
            r#"{"reason":"build-finished","success":true}"#,
        );
        let incomplete = r#"{"reason":"compiler-message","message":{"code":null}}"#;
        let failed = r#"{"reason":"build-finished","success":false}"#;

        assert!(parse_panic_surface_output(malformed)
            .unwrap_err()
            .contains("malformed JSON on line 2"));
        assert!(parse_panic_surface_output(incomplete)
            .unwrap_err()
            .contains("missing the terminal build-finished record"));
        assert!(parse_panic_surface_output(failed)
            .unwrap_err()
            .contains("reported an unsuccessful build"));
    }

    #[test]
    fn panic_surface_parser_rejects_records_after_finish() {
        let output = concat!(
            r#"{"reason":"build-finished","success":true}"#,
            "\n",
            r#"{"reason":"compiler-artifact"}"#,
        );

        assert!(parse_panic_surface_output(output)
            .unwrap_err()
            .contains("record after build-finished on line 2"));
    }
}
