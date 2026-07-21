// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::repo_root;

#[test]
fn metal_compute_tests_are_split_by_behavior() {
    let root = repo_root();
    let shell_path = root.join("crates/j2k-metal/src/compute/tests.rs");
    let test_root = root.join("crates/j2k-metal/src/compute/tests");
    let shell = fs::read_to_string(&shell_path).expect("read Metal compute test shell");

    for (module, owned_symbol, max_lines) in [
        (
            "capacity",
            "fn classic_encode_output_capacity_keeps_conservative_default(",
            300usize,
        ),
        (
            "classic",
            "fn prepared_classic_sub_band_decodes_on_cpu_for_hybrid_upload(",
            400usize,
        ),
        (
            "grouping",
            "fn direct_sub_band_grouping_groups_adjacent_ht_runs_without_runtime(",
            400usize,
        ),
        (
            "hybrid",
            "fn hybrid_rgb8_batch_uses_stacked_component_graph(",
            450usize,
        ),
        (
            "hybrid_support",
            "fn prepared_direct_color_tier1_input_count(",
            150usize,
        ),
        (
            "reuse",
            "fn hybrid_rgb8_reused_plan_caches_cpu_tier1_inputs_across_calls(",
            350usize,
        ),
        (
            "roi",
            "fn cropped_region_scaled_ht_direct_plan_prunes_codeblocks_outside_output_roi(",
            400usize,
        ),
        (
            "runtime",
            "fn runtime_initialization_error_classifies_null_queue_as_unavailable(",
            300usize,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(
            !shell.contains(owned_symbol),
            "Metal compute test shell must not retain {owned_symbol}"
        );
        let module_path = test_root.join(format!("{module}.rs"));
        let source = fs::read_to_string(&module_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", module_path.display()));
        assert!(
            source.contains(owned_symbol),
            "{} must own {owned_symbol}",
            module_path.display()
        );
        assert!(
            source.lines().count() < max_lines,
            "{} exceeded its focused {max_lines}-line limit",
            module_path.display()
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{} must import explicit test dependencies",
            module_path.display()
        );
    }

    assert!(
        shell.lines().count() < 30,
        "Metal compute test shell must remain a module inventory"
    );
}
