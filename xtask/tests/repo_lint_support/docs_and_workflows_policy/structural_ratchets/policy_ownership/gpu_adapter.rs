// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

#[test]
fn gpu_adapter_policy_is_a_shell_over_focused_contract_domains() {
    let root = repo_root();
    let policy_root = "xtask/tests/repo_lint_support/gpu_adapter_policy.rs";
    let shell = fs::read_to_string(root.join(policy_root))
        .unwrap_or_else(|error| panic!("read {policy_root}: {error}"));
    assert!(shell.lines().count() < 50);
    assert!(!shell.contains("#[test]"));
    assert!(!shell.lines().any(|line| line.trim() == "use super::*;"));

    for (module, symbol, max_lines) in [
        (
            "error_contracts",
            "fn gpu_adapter_error_classification_uses_shared_core_impl(",
            200,
        ),
        (
            "jpeg_encode_orchestration",
            "fn jpeg_gpu_encode_host_orchestration_uses_shared_adapter_helper(",
            200,
        ),
        (
            "metal_abi_contracts",
            "fn jpeg_metal_huffman_derivation_uses_shared_entropy_canonical_tables(",
            200,
        ),
        (
            "jpeg_metal_fast_paths",
            "fn fast444_region_scaled_batches_use_shared_region_scaled_metal_path(",
            250,
        ),
        (
            "cuda_adapter_contracts",
            "fn cuda_htj2k_compact_jobs_use_shared_planner(",
            325,
        ),
        (
            "transcode_contracts",
            "fn transcode_gpu_auto_threshold_policy_is_documented(",
            150,
        ),
        (
            "metal_public_api",
            "fn metal_public_error_lives_in_focused_module(",
            325,
        ),
        (
            "decode_request_contracts",
            "fn jpeg_metal_viewport_plane_rows_use_shared_target(",
            225,
        ),
    ] {
        assert!(shell.contains(&format!("mod {module};")));
        assert!(!shell.contains(symbol));
        let relative = format!("xtask/tests/repo_lint_support/gpu_adapter_policy/{module}.rs");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(source.contains(symbol), "{relative} must own {symbol}");
        assert!(source.lines().count() < max_lines);
        assert!(!source.lines().any(|line| line.trim() == "use super::*;"));
        assert!(!source
            .lines()
            .any(|line| line.trim_start().starts_with("include!(")));
    }
}
