// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::consumer::{
    extract_packaged_crate, j2k_ml_consumer_checks, j2k_ml_consumer_manifest,
    package_consumer_manifest, package_consumer_source, CONSUMER_SOURCE,
};
use super::{package_gate_plan, PackageGateStep};
use crate::release_commands::release_manifest::{
    parse_release_manifest_source, ReleaseManifestContract,
};

#[cfg(unix)]
use super::run;
#[cfg(unix)]
use crate::{command_support::use_test_cargo_program, test_command::RecordingProgram};

fn workspace_metadata(dependencies: &[(&str, &[&str])]) -> serde_json::Value {
    let manifest = release_manifest();
    let publishable = manifest.ordered_crates().collect::<Vec<_>>();
    let dependencies = dependencies
        .iter()
        .map(|(package, dependencies)| (*package, *dependencies))
        .collect::<BTreeMap<_, _>>();
    let packages = publishable
        .iter()
        .map(|package| {
            let default_dependencies = if matches!(
                *package,
                "j2k-core" | "j2k-profile" | "j2k-types" | "j2k-codec-math"
            ) {
                &[][..]
            } else {
                &["j2k-core"][..]
            };
            let package_dependencies = dependencies
                .get(package)
                .copied()
                .unwrap_or(default_dependencies)
                .iter()
                .map(|dependency| {
                    serde_json::json!({
                        "name": dependency,
                        "kind": null,
                        "path": format!("/workspace/{dependency}"),
                        "req": "=0.7.5",
                        "source": null
                    })
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "id": package,
                "name": package,
                "version": "0.7.5",
                "publish": null,
                "manifest_path": format!("/workspace/{package}/Cargo.toml"),
                "dependencies": package_dependencies,
                "targets": if *package == "j2k-cli" {
                    vec![serde_json::json!({"kind": ["bin"]})]
                } else {
                    vec![serde_json::json!({"kind": ["lib"]})]
                },
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "workspace_members": publishable,
        "packages": packages,
        "target_directory": "/workspace/target",
    })
}

fn release_manifest() -> ReleaseManifestContract {
    parse_release_manifest_source(include_str!("../../../../release-crates.json"))
        .expect("checked-in release manifest")
}

fn test_package_gate_plan(metadata: &serde_json::Value) -> Result<Vec<PackageGateStep>, String> {
    package_gate_plan(metadata, &release_manifest())
}

fn write_packaged_fixture(path: &Path) {
    let archive_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("package fixture archive name");
    let package = archive_name
        .strip_suffix("-0.7.5.crate")
        .expect("package fixture version suffix");
    let file = fs::File::create(path).expect("create package fixture");
    let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    let mut archive = tar::Builder::new(encoder);
    let contents = format!("[package]\nname = \"{package}\"\nversion = \"0.7.5\"\n");
    let mut header = tar::Header::new_gnu();
    header.set_size(contents.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    archive
        .append_data(
            &mut header,
            format!("{package}-0.7.5/Cargo.toml"),
            Cursor::new(contents.as_bytes()),
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

    let plan = test_package_gate_plan(&metadata).expect("dependency-aware package plan");
    let manifest = release_manifest();

    assert_eq!(
        plan.iter()
            .map(|step| step.package.as_str())
            .collect::<Vec<_>>(),
        manifest.ordered_crates().collect::<Vec<_>>()
    );
    let registry_independent = ["j2k-core", "j2k-profile", "j2k-types", "j2k-codec-math"];
    for step in &plan {
        assert_eq!(
            step.registry_independent,
            registry_independent.contains(&step.package.as_str())
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
    let error = test_package_gate_plan(&missing).expect_err("missing publishable package");
    assert!(
        error.contains("must contain every publishable workspace crate exactly once"),
        "unexpected: {error}"
    );

    let mut malformed = workspace_metadata(&[]);
    malformed["packages"][0]
        .as_object_mut()
        .expect("package")
        .remove("dependencies");
    let error = test_package_gate_plan(&malformed).expect_err("missing dependency array");
    assert!(
        error.contains("has no dependencies array"),
        "unexpected: {error}"
    );
}

#[test]
fn package_gate_plan_rejects_forward_dependency_before_any_packaging() {
    let metadata = workspace_metadata(&[("j2k-core", &["j2k"])]);

    let error = test_package_gate_plan(&metadata).expect_err("forward dependency order");

    assert!(
        error.contains("places `j2k-core` before dependency `j2k`"),
        "unexpected: {error}"
    );
}

#[test]
fn package_gate_includes_build_optional_and_target_specific_release_edges() {
    let mut metadata = workspace_metadata(&[]);
    let runtime = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-cuda-runtime")
        .expect("CUDA runtime package");
    runtime["dependencies"] = serde_json::json!([{
        "name": "j2k-codec-math",
        "kind": "build",
        "optional": true,
        "target": "cfg(target_os = \"linux\")",
        "path": "/workspace/j2k-codec-math",
        "req": "=0.7.5",
        "source": null
    }]);

    let plan = test_package_gate_plan(&metadata).expect("build dependency plan");
    let runtime = plan
        .iter()
        .find(|step| step.package == "j2k-cuda-runtime")
        .expect("CUDA runtime plan");

    assert!(!runtime.registry_independent);
    assert_eq!(
        runtime.patches,
        [(
            "j2k-codec-math".to_string(),
            "/workspace/j2k-codec-math".to_string()
        )]
    );
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

    let error = test_package_gate_plan(&metadata).expect_err("missing dependency manifest path");

    assert!(
        error.contains("`j2k-core` has no string manifest_path"),
        "unexpected: {error}"
    );
}

#[test]
fn package_gate_ignores_dev_dependencies_and_rejects_registry_sibling_edges() {
    let mut metadata = workspace_metadata(&[]);
    let native = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-native")
        .expect("native package");
    native["dependencies"] = serde_json::json!([
        {
            "name": "j2k-core",
            "kind": "dev",
            "path": "/workspace/j2k-core",
            "req": "=0.7.5",
            "source": null
        },
        {
            "name": "external",
            "kind": null,
            "path": null,
            "req": "1",
            "source": "registry+https://example.invalid"
        }
    ]);

    let plan = test_package_gate_plan(&metadata).expect("ignored non-release dependencies");
    let native = plan
        .iter()
        .find(|step| step.package == "j2k-native")
        .expect("native step");
    assert!(native.patches.is_empty());

    let native = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-native")
        .expect("native package");
    native["dependencies"][1]["name"] = serde_json::json!("j2k-core");
    let error =
        test_package_gate_plan(&metadata).expect_err("registry sibling dependency must reject");
    assert!(
        error.contains("workspace/path sourced"),
        "unexpected: {error}"
    );
}

#[test]
fn package_gate_validates_exact_dev_dependencies_before_planning() {
    let mut metadata = workspace_metadata(&[]);
    let native = metadata["packages"]
        .as_array_mut()
        .expect("packages")
        .iter_mut()
        .find(|package| package["name"] == "j2k-native")
        .expect("native package");
    native["dependencies"] = serde_json::json!([{
        "name": "j2k-core",
        "kind": "dev",
        "path": "/workspace/j2k-core",
        "req": "^0.7.5",
        "source": null
    }]);

    let error = test_package_gate_plan(&metadata)
        .expect_err("package planning must enforce exact dev dependencies");

    assert!(
        error.contains("exact release requirement"),
        "unexpected: {error}"
    );
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
fn packaged_consumer_compiles_new_and_0_7_compatibility_decoder_names() {
    for decoder in [
        "CudaUploadBurnDecoder",
        "MetalUploadBurnDecoder",
        "CudaBurnDecoder",
        "MetalBurnDecoder",
    ] {
        assert!(
            CONSUMER_SOURCE.contains(decoder),
            "packaged consumer must compile `{decoder}`"
        );
    }
}

#[test]
fn core_gpu_consumer_manifests_patch_only_extracted_archives() {
    let metadata = workspace_metadata(&[
        ("j2k", &["j2k-core"]),
        ("j2k-cuda", &["j2k", "j2k-cuda-runtime"]),
        ("j2k-metal", &["j2k", "j2k-metal-support"]),
        ("j2k-mpsgraph-support", &[]),
        (
            "j2k-mpsgraph",
            &[
                "j2k",
                "j2k-metal",
                "j2k-metal-support",
                "j2k-mpsgraph-support",
            ],
        ),
    ]);
    let plan = test_package_gate_plan(&metadata).expect("package plan");
    let packaged = plan
        .iter()
        .map(|step| {
            (
                step.package.clone(),
                PathBuf::from(format!("/packaged/{}-{}", step.package, step.version)),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for package in ["j2k", "j2k-cuda", "j2k-metal", "j2k-mpsgraph"] {
        let step = plan
            .iter()
            .find(|step| step.package == package)
            .expect("consumer package step");
        let manifest = package_consumer_manifest(step, &packaged).expect("clean consumer manifest");

        assert!(manifest.contains(&format!(
            "{package} = {{ path = \"/packaged/{package}-0.7.5\" }}"
        )));
        assert!(!manifest.contains("/workspace/"));
        if package == "j2k-cuda" {
            assert!(manifest.contains("cuda-runtime = [\"j2k-cuda/cuda-runtime\"]"));
        }
        for (dependency, _) in &step.patches {
            assert!(
                manifest.contains(&format!(
                    "{dependency} = {{ path = \"/packaged/{dependency}-0.7.5\" }}"
                )),
                "missing packaged dependency {dependency} in {manifest}"
            );
        }
        assert!(package_consumer_source(package)
            .expect("consumer source")
            .contains(&package.replace('-', "_")));
    }
}

#[test]
fn j2k_ml_consumer_manifest_patches_only_workspace_crates() {
    let metadata = workspace_metadata(&[
        ("j2k", &["j2k-core"]),
        ("j2k-cuda", &["j2k", "j2k-cuda-runtime"]),
        ("j2k-metal", &["j2k", "j2k-metal-support"]),
        ("j2k-ml", &["j2k", "j2k-cuda", "j2k-metal"]),
    ]);
    let plan = test_package_gate_plan(&metadata).expect("package plan");
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
    let plan = test_package_gate_plan(&workspace_metadata(&[])).expect("package plan");
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
    for package in [
        "j2k-core",
        "j2k-native",
        "j2k",
        "j2k-cuda",
        "j2k-metal",
        "j2k-ml",
        "j2k-mpsgraph",
    ] {
        write_packaged_fixture(&package_dir.join(format!("{package}-0.7.5.crate")));
    }
    metadata["target_directory"] =
        serde_json::Value::String(package_root.to_string_lossy().into_owned());
    let recording = RecordingProgram::new("package-gate-command-test", "");
    let _cargo = use_test_cargo_program(recording.program().as_os_str().to_owned());

    let manifest = release_manifest();
    run(&metadata, &manifest).expect("hermetic package gate");
    fs::remove_dir_all(package_root).expect("remove package gate target");

    let log = recording.log();
    let lines = log.lines().collect::<Vec<_>>();
    let consumer_checks = j2k_ml_consumer_checks(std::env::consts::OS);
    let registry_independent = ["j2k-core", "j2k-profile", "j2k-types", "j2k-codec-math"];
    assert_eq!(
        lines.len(),
        manifest.ordered_crates().len() + registry_independent.len() + consumer_checks.len() + 6
    );
    assert!(lines[0].starts_with("publish -p j2k-core --dry-run|"));
    for package in registry_independent {
        assert!(lines
            .iter()
            .any(|line| line.starts_with(&format!("publish -p {package} --dry-run|"))));
        assert!(lines
            .iter()
            .any(|line| line.starts_with(&format!("package -p {package} --no-verify|"))));
    }
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
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.starts_with("check|"))
            .count(),
        4
    );
}
