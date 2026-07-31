// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{normalize_path, repo_root, rust_include_paths, rust_sources};

#[test]
fn packaged_rust_sources_do_not_include_files_outside_their_crate() {
    let root = repo_root();
    let workspace_crates = root.join("crates");
    let mut escaping = Vec::new();

    for source_path in rust_sources(&workspace_crates) {
        let Ok(relative_to_crates) = source_path.strip_prefix(&workspace_crates) else {
            continue;
        };
        let Some(crate_name) = relative_to_crates.components().next() else {
            continue;
        };
        let member_root = workspace_crates.join(crate_name.as_os_str());
        let source = fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", source_path.display()));

        for include_path in rust_include_paths(&source) {
            let resolved = normalize_path(
                &source_path
                    .parent()
                    .expect("source file has parent")
                    .join(&include_path),
            );
            if !resolved.starts_with(&member_root) {
                escaping.push(format!(
                    "{} includes {} outside package root",
                    source_path
                        .strip_prefix(root)
                        .unwrap_or(&source_path)
                        .display(),
                    include_path
                ));
            }
        }
    }

    assert!(
        escaping.is_empty(),
        "package source include paths must stay inside their crate: {escaping:?}"
    );
}
