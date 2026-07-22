// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn metal_region_scaled_hybrid_paths_have_explicit_responsibility_owners() {
    let root = repo_root();
    let shell_relative = "crates/j2k-metal/src/hybrid.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));

    assert!(shell.lines().count() < 125);
    for (module, symbol, max_lines) in [
        ("cache", "enum RegionScaledColorPlanCache", 225),
        ("planning", "fn build_region_scaled_direct_color_plan(", 275),
        ("execution", "fn execute_region_scaled_direct_plan(", 175),
        (
            "batch",
            "fn decode_region_scaled_color_batch_direct_to_device_routed(",
            250,
        ),
        (
            "profile",
            "fn emit_region_scaled_color_plan_build_timings(",
            150,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(!shell.contains(symbol));
        let relative = format!("crates/j2k-metal/src/hybrid/{module}.rs");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(source.contains(symbol), "{relative} must own {symbol}");
        assert!(source.lines().count() < max_lines);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }
}
