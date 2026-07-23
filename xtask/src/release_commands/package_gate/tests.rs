// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::consumer::{extract_packaged_crate, j2k_ml_consumer_checks, j2k_ml_consumer_manifest};
use super::{package_gate_plan, PUBLISHABLE_PACKAGES, REGISTRY_INDEPENDENT_PACKAGES};

#[cfg(unix)]
use super::run;
#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

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
                "version": "0.7.5",
                "manifest_path": format!("/workspace/{package}/Cargo.toml"),
                "dependencies": package_dependencies,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "workspace_members": PUBLISHABLE_PACKAGES,
        "packages": packages,
        "target_directory": "/workspace/target",
    })
}

fn write_packaged_fixture(path: &Path) {
    let file = fs::File::create(path).expect("create package fixture");
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let contents = b"[package]\nname = \"j2k-ml\"\nversion = \"0.7.5\"\n";
    let mut header = tar::Header::new_gnu();
    header.set_size(contents.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(
            &mut header,
            "j2k-ml-0.7.5/Cargo.toml",
            Cursor::new(contents),
        )
        .expect("append package fixture");
    archive.finish().expect("finish package fixture");
}

fn test_root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("test clock")
        .as_nanos();
    std::env::temp_dir().join(format!("{label}-{}-{nonce}", std::process::id()))
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

#[test]
fn j2k_ml_consumer_matrix_matches_the_host_accelerator() {
    assert_eq!(j2k_ml_consumer_checks("linux"), ["cpu", "cuda", "cpu,cuda"]);
    assert_eq!(
        j2k_ml_consumer_checks("macos"),
        ["cpu", "metal", "cpu,metal"]
    );
    assert_eq!(j2k_ml_consumer_checks("windows"), ["cpu"]);
}

#[test]
fn j2k_ml_consumer_manifest_patches_only_workspace_crates() {
    let metadata = workspace_metadata(&[
        ("j2k", &["j2k-core"]),
        ("j2k-cuda", &["j2k", "j2k-cuda-runtime"]),
        ("j2k-metal", &["j2k", "j2k-metal-support"]),
        ("j2k-ml", &["j2k", "j2k-cuda", "j2k-metal"]),
    ]);
    let plan = package_gate_plan(&metadata).expect("package plan");
    let ml = plan
        .iter()
        .find(|step| step.package == "j2k-ml")
        .expect("j2k-ml step");

    let manifest =
        j2k_ml_consumer_manifest(ml, "/packaged/j2k-ml-0.7.5").expect("external consumer manifest");

    assert!(manifest.contains("j2k-ml = { version = \"=0.7.5\""));
    assert!(manifest.contains("j2k-ml = { path = \"/packaged/j2k-ml-0.7.5\" }"));
    assert!(!manifest.contains("j2k-ml = { path = \"/workspace/j2k-ml\""));
    assert!(manifest.contains("j2k-core = { path = \"/workspace/j2k-core\" }"));
    assert!(manifest.contains("j2k-cuda = { path = \"/workspace/j2k-cuda\" }"));
    assert!(manifest.contains("j2k-metal = { path = \"/workspace/j2k-metal\" }"));
    for third_party in [
        "cubecl-cuda",
        "cubecl-runtime",
        "wgpu",
        "wgpu-core",
        "wgpu-hal",
    ] {
        assert!(
            !manifest.contains(&format!("{third_party} = {{ path =")),
            "external consumer must resolve {third_party} from the registry"
        );
    }
}

#[test]
fn packaged_consumer_extracts_the_crate_archive_instead_of_using_workspace_source() {
    let root = test_root("j2k-ml-package-extract-test");
    fs::create_dir_all(&root).expect("create package extraction test root");
    let archive = root.join("j2k-ml-0.7.5.crate");
    write_packaged_fixture(&archive);
    let plan = package_gate_plan(&workspace_metadata(&[])).expect("package plan");
    let step = plan
        .iter()
        .find(|step| step.package == "j2k-ml")
        .expect("j2k-ml step");

    let extracted =
        extract_packaged_crate(&archive, &root.join("out"), step).expect("extract packaged crate");

    assert_eq!(
        fs::read_to_string(extracted.join("Cargo.toml")).expect("read extracted manifest"),
        "[package]\nname = \"j2k-ml\"\nversion = \"0.7.5\"\n"
    );
    fs::remove_dir_all(root).expect("remove package extraction test root");
}

#[cfg(unix)]
#[test]
fn package_gate_executes_registry_and_staged_steps_with_dependency_patches() {
    let mut metadata = workspace_metadata(&[
        ("j2k-native", &["j2k-core"]),
        ("j2k", &["j2k-native"]),
        ("j2k-cli", &["j2k"]),
    ]);
    let package_root = test_root("j2k-ml-package-gate-test");
    let package_dir = package_root.join("package");
    fs::create_dir_all(&package_dir).expect("create package gate target");
    write_packaged_fixture(&package_dir.join("j2k-ml-0.7.5.crate"));
    metadata["target_directory"] =
        serde_json::Value::String(package_root.to_string_lossy().into_owned());
    let recording = RecordingProgram::new("package-gate-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    run(&metadata).expect("hermetic package gate");
    fs::remove_dir_all(package_root).expect("remove package gate target");

    let log = recording.log();
    let lines = log.lines().collect::<Vec<_>>();
    let consumer_checks = j2k_ml_consumer_checks(std::env::consts::OS);
    assert_eq!(
        lines.len(),
        PUBLISHABLE_PACKAGES.len() + consumer_checks.len() + 2
    );
    assert!(lines[0].starts_with("publish -p j2k-core --dry-run|"));
    assert!(lines[3].starts_with("publish -p j2k-codec-math --dry-run|"));
    let native = lines
        .iter()
        .find(|line| line.starts_with("package -p j2k-native --no-verify"))
        .expect("native staged package command");
    assert!(native.contains("patch.crates-io.j2k-core.path=\"/workspace/j2k-core\""));
    let cli = lines
        .iter()
        .find(|line| line.starts_with("package -p j2k-cli --no-verify"))
        .expect("CLI staged package command");
    for dependency in ["j2k", "j2k-core", "j2k-native"] {
        assert!(
            cli.contains(&format!("patch.crates-io.{dependency}.path=")),
            "missing {dependency} patch in {cli}"
        );
    }
    for features in consumer_checks {
        assert!(lines.iter().any(|line| {
            line.contains("check --no-default-features --features") && line.contains(features)
        }));
    }
    assert!(lines
        .iter()
        .any(|line| { line.contains("doc --no-deps --no-default-features --features") }));
    assert!(lines
        .iter()
        .any(|line| { line.contains("check --examples --no-default-features --features") }));
}
