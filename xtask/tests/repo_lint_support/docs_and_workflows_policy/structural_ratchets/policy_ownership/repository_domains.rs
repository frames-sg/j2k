// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn repo_lint_checks_have_domain_specific_owners() {
    let root = repo_root();
    let module_root = fs::read_to_string(root.join("xtask/tests/repo_lint_support/mod.rs"))
        .expect("read repo-lint module root");
    assert!(!module_root.contains("mod source_policy;"));
    assert!(
        !root
            .join("xtask/tests/repo_lint_support/source_policy.rs")
            .exists(),
        "repo-wide source assertions must live with their actual domains"
    );
    let gpu_root =
        fs::read_to_string(root.join("xtask/tests/repo_lint_support/gpu_adapter_policy.rs"))
            .expect("read GPU adapter policy root");
    for module in [
        "mod accelerator_test_structure_policy;",
        "mod adapter_boundary_policy;",
        "mod cuda_product_source_policy;",
    ] {
        assert!(gpu_root.contains(module));
    }
    assert!(!gpu_root.contains("mod accelerator_product_source_policy;"));
    assert!(!root
        .join(
            "xtask/tests/repo_lint_support/gpu_adapter_policy/accelerator_product_source_policy.rs"
        )
        .exists());

    for (relative, symbols) in [
        (
            "xtask/tests/repo_lint_support/native_direct_plan_structure_policy.rs",
            &["referenced_direct_plan_tests_keep_focused_owners"][..],
        ),
        (
            "xtask/tests/repo_lint_support/j2k_decode_structure_policy.rs",
            &["owned_batch_tests_are_split_by_responsibility"][..],
        ),
        (
            "xtask/tests/repo_lint_support/metal_compute_structure_policy/compute_tests.rs",
            &["referenced_plan_regression_has_focused_owner"][..],
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/metal_batch_structure_policy.rs",
            &["metal_multitile_tests_are_split_by_pixel_contract"][..],
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/accelerator_test_structure_policy.rs",
            &["accelerator_boundary_tests_keep_focused_owners"][..],
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/adapter_boundary_policy.rs",
            &[
                "adapter_crates_do_not_import_codec_private_modules",
                "cuda_adapter_crates_keep_public_libs_as_module_shells",
            ][..],
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_product_source_policy.rs",
            &[
                "production_j2k_cuda_code_does_not_reference_nvjpeg",
                "cuda_runtime_rejects_product_cuda_c_and_checked_in_ptx",
                "cuda_runtime_dispatch_does_not_read_deprecated_oxide_route_selectors",
            ][..],
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_profile_policy.rs",
            &["cuda_trace_export_is_non_clobbering_and_documented"][..],
        ),
        (
            "xtask/tests/repo_lint_support/docs_and_workflows_policy/adoption_benchmark_policy.rs",
            &["reusable_benchmark_generators_live_in_test_support"][..],
        ),
        (
            "xtask/tests/repo_lint_support/docs_and_workflows_policy/workflow_coverage_policy.rs",
            &["gpu_runtime_tests_do_not_silently_return_on_missing_hardware_gates"][..],
        ),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        for symbol in symbols {
            assert!(source.contains(symbol), "{relative} must own {symbol}");
        }
    }
}

#[test]
fn metal_policy_checks_live_in_focused_children() {
    let root = repo_root();
    let gpu_root_relative = "xtask/tests/repo_lint_support/gpu_adapter_policy.rs";
    let gpu_root = fs::read_to_string(root.join(gpu_root_relative))
        .unwrap_or_else(|error| panic!("read {gpu_root_relative}: {error}"));
    let metal_batch_relative =
        "xtask/tests/repo_lint_support/gpu_adapter_policy/metal_batch_structure_policy.rs";
    let metal_batch = fs::read_to_string(root.join(metal_batch_relative))
        .unwrap_or_else(|error| panic!("read {metal_batch_relative}: {error}"));
    assert!(gpu_root.contains("mod metal_batch_structure_policy;"));
    for symbol in [
        "fn metal_batch_heuristics_live_in_focused_module(",
        "fn metal_batch_routes_share_session_aware_implementations(",
    ] {
        assert!(!gpu_root.contains(symbol));
        assert!(metal_batch.contains(symbol));
    }
    assert!(gpu_root.lines().count() < 50);
    assert!(metal_batch.lines().count() < 300);

    let compute_root_relative = "xtask/tests/repo_lint_support/metal_compute_structure_policy.rs";
    let compute_root = fs::read_to_string(root.join(compute_root_relative))
        .unwrap_or_else(|error| panic!("read {compute_root_relative}: {error}"));
    let batch_shell_relative =
        "xtask/tests/repo_lint_support/metal_compute_structure_policy/batch_execution.rs";
    let batch_shell = fs::read_to_string(root.join(batch_shell_relative))
        .unwrap_or_else(|error| panic!("read {batch_shell_relative}: {error}"));
    assert!(compute_root.contains("mod batch_execution;"));
    assert!(compute_root.lines().count() < 25);
    assert!(batch_shell.lines().count() < 25);

    for (module, symbol, max_lines) in [
        (
            "ht_chunks",
            "fn metal_ht_chunk_tests_are_split_by_planning_cache_and_status_behavior(",
            250,
        ),
        (
            "stacked_execution",
            "fn metal_stacked_execution_is_split_by_codec_stage(",
            75,
        ),
        (
            "direct_destination",
            "fn metal_direct_destination_is_split_by_submission_and_group_encoding(",
            75,
        ),
        (
            "classic_batch",
            "fn metal_distinct_classic_batch_execution_is_split_from_cleanup_dispatch(",
            75,
        ),
    ] {
        assert!(batch_shell.contains(&format!("mod {module};")));
        let relative = format!(
            "xtask/tests/repo_lint_support/metal_compute_structure_policy/batch_execution/{module}.rs"
        );
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(source.contains(symbol), "{relative} must own {symbol}");
        assert!(source.lines().count() < max_lines);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
    }
}
