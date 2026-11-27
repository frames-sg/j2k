// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::repo_root;

const METAL_CRATES: &[&str] = &[
    "j2k-metal-support",
    "j2k-metal",
    "j2k-jpeg-metal",
    "j2k-transcode-metal",
];

#[test]
fn metal_crates_use_the_objc2_stack_without_legacy_metal_or_block() {
    let root = repo_root();
    let workspace_source =
        fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let workspace =
        toml::from_str::<toml::Value>(&workspace_source).expect("parse workspace manifest");
    let dependencies = workspace
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(toml::Value::as_table)
        .expect("workspace dependencies");

    assert!(!dependencies.contains_key("metal"));
    // objc2 types appear in the Metal crates' public API, so a new objc2 minor
    // is a breaking change for them. Each requirement stays a caret range
    // inside the minor line listed here: an exact pin would stop downstream
    // applications from sharing one objc2 with these crates.
    for (name, minor_line) in [
        ("objc2", "0.6."),
        ("objc2-foundation", "0.3."),
        ("objc2-metal", "0.3."),
        ("objc2-metal-performance-shaders", "0.3."),
        ("objc2-metal-performance-shaders-graph", "0.3."),
        ("block2", "0.6."),
    ] {
        let dependency = dependencies
            .get(name)
            .and_then(toml::Value::as_table)
            .unwrap_or_else(|| panic!("workspace dependency `{name}`"));
        let version = dependency
            .get("version")
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("workspace dependency `{name}` version"));
        assert!(
            version.starts_with(minor_line)
                && version
                    .chars()
                    .all(|character| character.is_ascii_digit() || character == '.'),
            "workspace dependency `{name}` must be a caret requirement within {minor_line}x, \
             not `{version}`; a new minor is a breaking change for the Metal crates"
        );
    }

    let block_patch = workspace
        .get("patch")
        .and_then(|patch| patch.get("crates-io"))
        .and_then(toml::Value::as_table)
        .is_some_and(|patches| patches.contains_key("block"));
    assert!(!block_patch, "the legacy block path patch must be absent");
    assert!(
        !root.join("third_party/block-0.1.6-patched").exists(),
        "the patched block source tree must be absent"
    );

    for crate_name in METAL_CRATES {
        let source = fs::read_to_string(root.join("crates").join(crate_name).join("Cargo.toml"))
            .unwrap_or_else(|error| panic!("read {crate_name} manifest: {error}"));
        let manifest = toml::from_str::<toml::Value>(&source)
            .unwrap_or_else(|error| panic!("parse {crate_name} manifest: {error}"));
        let target_dependencies = manifest
            .get("target")
            .and_then(|target| target.get("cfg(target_os = \"macos\")"))
            .and_then(|target| target.get("dependencies"))
            .and_then(toml::Value::as_table)
            .unwrap_or_else(|| panic!("{crate_name} macOS dependencies"));
        assert!(!target_dependencies.contains_key("metal"));
        for dependency in ["objc2", "objc2-metal"] {
            assert!(
                target_dependencies.contains_key(dependency),
                "{crate_name} must depend directly on `{dependency}`"
            );
        }
    }

    let t803_source = fs::read_to_string(root.join("crates/j2k-t803/Cargo.toml"))
        .expect("read j2k-t803 manifest");
    let t803 = toml::from_str::<toml::Value>(&t803_source).expect("parse j2k-t803 manifest");
    let t803_dependencies = t803
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("j2k-t803 dependencies");
    assert!(
        !t803_dependencies.contains_key("objc2-metal"),
        "j2k-t803 must not compile objc2-metal on non-macOS targets"
    );
    let t803_macos_dependencies = t803
        .get("target")
        .and_then(|target| target.get("cfg(target_os = \"macos\")"))
        .and_then(|target| target.get("dependencies"))
        .and_then(toml::Value::as_table)
        .expect("j2k-t803 macOS dependencies");
    assert!(
        t803_macos_dependencies.contains_key("objc2-metal"),
        "j2k-t803 Metal runner must reuse the macOS-only objc2-metal dependency"
    );

    let lock_source = fs::read_to_string(root.join("Cargo.lock")).expect("read Cargo.lock");
    let lock = toml::from_str::<toml::Value>(&lock_source).expect("parse Cargo.lock");
    let packages = lock
        .get("package")
        .and_then(toml::Value::as_array)
        .expect("Cargo.lock packages");
    assert!(packages.iter().all(|package| {
        let name = package.get("name").and_then(toml::Value::as_str);
        let version = package.get("version").and_then(toml::Value::as_str);
        name != Some("metal") && !(name == Some("block") && version == Some("0.1.6"))
    }));
}
