// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn path_patch_governance_has_focused_owners() {
    let root = repo_root();
    for (owner, child, module_decl, symbols, owner_limit, child_limit) in [
        (
            "xtask/tests/repo_lint_support/dependency_policy.rs",
            "xtask/tests/repo_lint_support/dependency_policy/path_patches.rs",
            "mod path_patches;",
            &[
                "fn all_workspace_path_patches_have_pinned_provenance_and_local_digests(",
                "fn patched_tree_sha256(",
            ][..],
            250usize,
            325usize,
        ),
        (
            "xtask/src/release_commands.rs",
            "xtask/src/release_commands/path_patches.rs",
            "mod path_patches;",
            &["fn workspace_path_patch_provenance_paths("][..],
            700usize,
            100usize,
        ),
    ] {
        let owner_source = fs::read_to_string(root.join(owner))
            .unwrap_or_else(|error| panic!("read {owner}: {error}"));
        let child_source = fs::read_to_string(root.join(child))
            .unwrap_or_else(|error| panic!("read {child}: {error}"));
        assert!(owner_source.contains(module_decl));
        for symbol in symbols {
            assert!(!owner_source.contains(symbol));
            assert!(child_source.contains(symbol));
        }
        assert!(owner_source.lines().count() < owner_limit);
        assert!(child_source.lines().count() < child_limit);
        assert!(!child_source
            .lines()
            .any(|line| line.trim() == "use super::*;"));
    }
}
