// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use super::{repo_root, sha256_hex};

#[test]
fn patched_block_dependency_has_pinned_provenance_and_documented_abi_delta() {
    const ARCHIVE_SHA256: &str = "0d8c1fef690941d3e7788d328517591fecc684c084084702d6ff1641e993699a";
    const UPSTREAM_LIB_SHA256: &str =
        "eb31678adf63b53109d9b94eba23699fd5f9ebfdb950f6e1a57ad51bb6a146fa";
    const PATCHED_LIB_SHA256: &str =
        "bf799f4d01bb497fdcffe7a5e28d998e721ed45c1be866ed1b454df39ce876a9";

    let root = repo_root();
    let directory = root.join("third_party/block-0.1.6-patched");
    let provenance = fs::read_to_string(directory.join("PATCH_PROVENANCE.md"))
        .expect("read patched block provenance");
    for digest in [ARCHIVE_SHA256, UPSTREAM_LIB_SHA256, PATCHED_LIB_SHA256] {
        assert!(
            provenance.contains(digest),
            "patched block provenance must pin SHA-256 {digest}"
        );
    }
    assert!(provenance.contains("Documented ABI deltas from upstream"));
    assert!(provenance.contains("## Release approval"));
    assert!(provenance.contains("- Reviewer identity:"));
    assert!(provenance.contains("- Approval date:"));
    assert!(!provenance.contains("Reviewed ABI deltas from upstream"));
    assert!(provenance.contains("Replace the uninhabited opaque `enum Class {}`"));
    assert!(provenance.contains("Spell the C ABI explicitly as `extern \"C\"`"));

    let patched_source_path = directory.join("src/lib.rs");
    assert_eq!(sha256_hex(&patched_source_path), PATCHED_LIB_SHA256);
    let patched_source =
        fs::read_to_string(&patched_source_path).expect("read patched block source");
    assert!(patched_source.contains("#[repr(C)]\nstruct Class {"));
    assert!(patched_source.contains("extern \"C\" {"));
    assert!(patched_source.contains("unsafe extern \"C\" fn block_context_copy"));
    assert!(!patched_source.contains("enum Class { }"));
}

#[test]
fn block_patch_scope_and_metal_migration_debt_stay_explicit() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    assert!(workspace.contains("[patch.crates-io]"));
    assert!(workspace.contains("block = { path = \"third_party/block-0.1.6-patched\" }"));
    assert!(workspace.contains("exclude = [\"third_party/block-0.1.6-patched\"]"));

    for manifest in workspace_member_manifests(root) {
        let source = fs::read_to_string(&manifest)
            .unwrap_or_else(|error| panic!("read {}: {error}", manifest.display()));
        assert!(
            !source.contains("[patch.crates-io]"),
            "workspace member {} must not imply that the root patch is publishable metadata",
            manifest.display()
        );
    }

    let release = fs::read_to_string(root.join("docs/release.md")).expect("read release guide");
    for required in [
        "This override protects repository builds only.",
        "published Metal adapters will still resolve upstream",
        "metal 0.33.0",
        "block 0.1.6",
        "deprecated in favor of `objc2-metal`",
        "Do not describe the local patch as a downstream fix.",
    ] {
        assert!(
            release.contains(required),
            "release guide must retain Metal dependency warning `{required}`"
        );
    }
}

#[test]
fn workspace_managed_third_party_deps_are_inherited() {
    let root = repo_root();
    let workspace_deps = workspace_dependency_names(root);
    let mut offenders = Vec::new();

    for manifest in workspace_member_manifests(root) {
        let source = fs::read_to_string(&manifest)
            .unwrap_or_else(|err| panic!("read {}: {err}", manifest.display()));
        for (line_index, line) in source.lines().enumerate() {
            let trimmed = line.trim_start();
            for dep in &workspace_deps {
                if redeclares_workspace_dep_version(trimmed, dep) {
                    offenders.push(format!(
                        "{}:{} re-declares workspace dependency `{dep}` with a hardcoded version",
                        manifest.strip_prefix(root).unwrap_or(&manifest).display(),
                        line_index + 1
                    ));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "workspace-managed third-party dependencies must use `workspace = true`:\n{}",
        offenders.join("\n")
    );
}

fn workspace_dependency_names(root: &Path) -> BTreeSet<String> {
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let deps_block = manifest
        .split("[workspace.dependencies]")
        .nth(1)
        .and_then(|rest| rest.split("\n[").next())
        .expect("workspace dependencies block");
    deps_block
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            line.split('=').next().map(str::trim).map(str::to_string)
        })
        .collect()
}

fn workspace_member_manifests(root: &Path) -> Vec<PathBuf> {
    let mut manifests = fs::read_dir(root.join("crates"))
        .expect("read crates")
        .map(|entry| entry.expect("read crate entry").path().join("Cargo.toml"))
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    manifests.push(root.join("xtask/Cargo.toml"));
    manifests.sort();
    manifests
}

fn redeclares_workspace_dep_version(line: &str, dep: &str) -> bool {
    let Some(rest) = line.strip_prefix(dep) else {
        return false;
    };
    let rest = rest.trim_start();
    if !rest.starts_with('=') || rest.contains("workspace = true") {
        return false;
    }
    let value = rest.trim_start_matches('=').trim_start();
    value.starts_with('"') || value.contains("version")
}
