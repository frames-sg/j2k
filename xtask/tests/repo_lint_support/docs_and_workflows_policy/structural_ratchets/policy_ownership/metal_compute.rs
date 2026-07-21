// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn metal_compute_policy_root_is_a_focused_module_shell() {
    let root = repo_root();
    let shell_relative = "xtask/tests/repo_lint_support/metal_compute_structure_policy.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));
    assert!(shell.lines().count() < 25);
    assert!(!shell
        .lines()
        .any(|line| line.trim_start().starts_with("use ")));

    for (module, symbol, max_lines) in [
        (
            "runtime_registry",
            "fn metal_compute_runtime_registry_is_split_from_compute_god_file(",
            300,
        ),
        (
            "direct_plan_types",
            "fn metal_direct_plan_types_live_in_focused_module(",
            300,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(!shell.contains(symbol));
        let relative =
            format!("xtask/tests/repo_lint_support/metal_compute_structure_policy/{module}.rs");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(source.contains(symbol));
        assert!(source.lines().count() < max_lines);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }

    let status_relative =
        "xtask/tests/repo_lint_support/metal_compute_structure_policy/status_contracts.rs";
    let status = fs::read_to_string(root.join(status_relative))
        .unwrap_or_else(|error| panic!("read {status_relative}: {error}"));
    assert!(!status.lines().any(|line| line.trim() == "use super::*;"));
}
