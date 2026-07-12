// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{package_gate_plan, PUBLISHABLE_PACKAGES, REGISTRY_INDEPENDENT_PACKAGES};

fn workspace_metadata(dependencies: &[(&str, &[&str])]) -> serde_json::Value {
    let dependencies = dependencies
        .iter()
        .map(|(package, dependencies)| (*package, *dependencies))
        .collect::<BTreeMap<_, _>>();
    let packages = PUBLISHABLE_PACKAGES
        .iter()
        .map(|package| {
            let package_dependencies = dependencies
                .get(package)
                .into_iter()
                .flat_map(|dependencies| dependencies.iter())
                .map(|dependency| {
                    serde_json::json!({"name": dependency, "kind": null, "source": null})
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "id": package,
                "name": package,
                "manifest_path": format!("/workspace/{package}/Cargo.toml"),
                "dependencies": package_dependencies,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "workspace_members": PUBLISHABLE_PACKAGES,
        "packages": packages,
    })
}

#[test]
fn package_gate_plan_is_ordered_and_includes_transitive_unpublished_patches() {
    let metadata = workspace_metadata(&[
        ("j2k-native", &["j2k-core"]),
        ("j2k", &["j2k-native"]),
        ("j2k-cli", &["j2k"]),
    ]);

    let plan = package_gate_plan(&metadata).expect("dependency-aware package plan");

    assert_eq!(
        plan.iter().map(|step| step.package).collect::<Vec<_>>(),
        PUBLISHABLE_PACKAGES
    );
    for step in &plan {
        assert_eq!(
            step.registry_independent,
            REGISTRY_INDEPENDENT_PACKAGES.contains(&step.package)
        );
    }
    let cli = plan
        .iter()
        .find(|step| step.package == "j2k-cli")
        .expect("CLI plan");
    assert_eq!(
        cli.patches
            .iter()
            .map(|(name, path)| (name.as_str(), path.as_str()))
            .collect::<Vec<_>>(),
        [
            ("j2k", "/workspace/j2k"),
            ("j2k-core", "/workspace/j2k-core"),
            ("j2k-native", "/workspace/j2k-native"),
        ]
    );
}

#[test]
fn package_gate_plan_rejects_missing_or_malformed_publishable_records() {
    let mut missing = workspace_metadata(&[]);
    missing["workspace_members"]
        .as_array_mut()
        .expect("members")
        .retain(|member| member != "j2k-core");
    let error = package_gate_plan(&missing).expect_err("missing publishable package");
    assert!(error.contains("`j2k-core` is absent"));

    let mut malformed = workspace_metadata(&[]);
    malformed["packages"][0]
        .as_object_mut()
        .expect("package")
        .remove("dependencies");
    let error = package_gate_plan(&malformed).expect_err("missing dependency array");
    assert!(error.contains("has no dependency array"));
}

#[test]
fn package_gate_plan_rejects_forward_dependency_before_any_packaging() {
    let metadata = workspace_metadata(&[("j2k-core", &["j2k"])]);

    let error = package_gate_plan(&metadata).expect_err("forward dependency order");

    assert!(error.contains("processes `j2k-core` before unpublished workspace dependency `j2k`"));
}

#[test]
fn package_gate_plan_requires_manifest_paths_for_patch_dependencies() {
    let mut metadata = workspace_metadata(&[("j2k-native", &["j2k-core"])]);
    let core = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-core")
        .expect("core package");
    core.as_object_mut()
        .expect("core record")
        .remove("manifest_path");

    let error = package_gate_plan(&metadata).expect_err("missing dependency manifest path");

    assert!(error.contains("`j2k-core` has no manifest path"));
}

#[test]
fn package_gate_ignores_registry_and_non_normal_dependencies() {
    let mut metadata = workspace_metadata(&[]);
    let native = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-native")
        .expect("native package");
    native["dependencies"] = serde_json::json!([
        {"name": "j2k-core", "kind": "dev", "source": null},
        {"name": "j2k-core", "kind": null, "source": "registry+https://example.invalid"}
    ]);

    let plan = package_gate_plan(&metadata).expect("ignored non-patch dependencies");
    let native = plan
        .iter()
        .find(|step| step.package == "j2k-native")
        .expect("native step");
    assert!(native.patches.is_empty());
}
