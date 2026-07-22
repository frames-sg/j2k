// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::{Component, Path, PathBuf};

pub(super) fn workspace_path_patch_provenance_paths(
    manifest: &str,
) -> Result<Vec<PathBuf>, String> {
    let manifest = toml::from_str::<toml::Value>(manifest)
        .map_err(|error| format!("failed to parse Cargo.toml: {error}"))?;
    let Some(patches) = manifest
        .get("patch")
        .and_then(|patch| patch.get("crates-io"))
        .and_then(toml::Value::as_table)
    else {
        return Ok(Vec::new());
    };
    let mut paths = Vec::new();
    paths
        .try_reserve_exact(patches.len())
        .map_err(|error| format!("reserve path-patch provenance paths: {error}"))?;
    for (name, patch) in patches {
        let Some(path) = patch.get("path").and_then(toml::Value::as_str) else {
            continue;
        };
        let path = Path::new(path);
        let mut has_normal_component = false;
        let stays_in_repository = path.components().all(|component| match component {
            Component::Normal(_) => {
                has_normal_component = true;
                true
            }
            Component::CurDir => true,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => false,
        });
        if !stays_in_repository || !has_normal_component {
            return Err(format!(
                "workspace path patch `{name}` must use a repository-relative path"
            ));
        }
        paths.push(path.join("PATCH_PROVENANCE.md"));
    }
    Ok(paths)
}
