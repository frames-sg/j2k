// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use super::repo_root;

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
