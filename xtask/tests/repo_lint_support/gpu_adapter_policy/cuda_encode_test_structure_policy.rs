// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::repo_root;

const TEST_MODULES: &[(&str, usize, &[&str])] = &[
    (
        "htj2k.rs",
        300,
        &[
            "cuda_htj2k_codeblock_dispatches_when_runtime_required",
            "cuda_htj2k_codeblock_preserves_requested_refinement_passes_when_runtime_required",
            "cuda_htj2k_codeblock_batch_uses_single_dispatch_when_runtime_required",
            "cuda_resident_quantized_subband_feeds_resident_ht_batch_when_runtime_required",
            "cuda_resident_strided_codeblock_region_matches_host_gather_when_runtime_required",
        ],
    ),
    (
        "packetization.rs",
        600,
        &[
            "cuda_packetization_flatten_accepts_cleanup_only_single_block_packet",
            "cuda_packetization_flatten_accepts_cleanup_only_multi_block_packet",
            "cuda_packetization_flatten_accepts_ht_refinement_pass_packet",
            "cuda_packetization_rejects_overflowing_ht_refinement_lengths",
            "cuda_packetization_flatten_rejects_out_of_range_ht_pass_count",
            "cuda_packetization_flatten_accepts_previously_included_second_layer_packet",
            "cuda_packetization_flatten_accepts_deferred_first_inclusion_second_layer_packet",
            "cuda_packetization_flatten_accepts_deferred_first_inclusion_after_non_empty_packet",
        ],
    ),
    (
        "resident.rs",
        375,
        &[
            "cuda_lossless_encode_require_device_dispatches_cleanup_packetization_when_runtime_required",
            "cuda_lossless_buffer_encode_returns_resident_codestream_when_runtime_required",
            "cuda_lossless_encode_require_device_dispatches_multi_block_cleanup_when_runtime_required",
            "cuda_lossless_encode_require_device_dispatches_dwt53_cleanup_when_runtime_required",
            "cuda_lossless_encode_profile_reports_resident_stage_timings_when_runtime_required",
            "cuda_lossless_encode_require_device_dispatches_rgb_rct_cleanup_when_runtime_required",
        ],
    ),
    (
        "resident_session.rs",
        225,
        &[
            "resident_encode_binds_external_context_and_clones_reuse_resources_when_required",
            "cuda_lossless_buffer_batch_encode_returns_resident_codestreams_in_order_when_runtime_required",
            "resident_encode_rejects_session_context_mismatch_before_resource_upload_when_required",
        ],
    ),
    (
        "resident_tiles.rs",
        250,
        &[
            "cuda_encode_uses_resident_tile_body_when_runtime_required",
            "cuda_encode_uses_resident_dwt_tile_body_when_runtime_required",
            "cuda_encode_uses_resident_mct_dwt_tile_body_when_runtime_required",
            "cuda_encode_uses_resident_dwt97_tile_body_when_runtime_required",
        ],
    ),
    (
        "routing.rs",
        300,
        &[
            "typed_resident_input_failures_map_to_stable_cuda_rejections",
            "cuda_lossless_encode_auto_errors_for_unsupported_classic_tier1",
            "cuda_lossless_encode_profile_auto_errors_for_unsupported_classic_tier1",
            "cuda_lossless_encode_require_device_errors_for_unsupported_classic_tier1",
            "prefer_cpu_ht_subband_declines_fused_subband_but_counts_attempts",
            "cuda_lossy_htj2k_facade_require_device_dispatches_supported_stages_when_runtime_required",
            "cuda_encode_stage_accelerator_preserves_cpu_codestream_validity",
            "cuda_auto_host_output_declines_packetization_before_flattening",
            "cuda_invalid_packetization_plan_falls_back_after_classification",
            "cuda_packetization_host_allocation_is_a_hard_stage_error",
        ],
    ),
    (
        "transforms.rs",
        325,
        &[
            "cuda_deinterleave_stage_dispatches_when_runtime_required",
            "cuda_forward_rct_dispatches_when_runtime_required",
            "cuda_forward_ict_dispatches_when_runtime_required",
            "cuda_forward_dwt53_dispatches_when_runtime_required",
            "cuda_forward_dwt53_private_reshape_matches_native_reference_when_required",
            "cuda_forward_dwt97_dispatches_when_runtime_required",
            "cuda_quantize_subband_dispatches_when_runtime_required",
        ],
    ),
];

#[test]
fn cuda_encode_tests_use_focused_real_modules_with_stable_inventory() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA encode facade");
    let test_root = root.join("crates/j2k-cuda/src/encode/tests");
    let shell = fs::read_to_string(test_root.join("mod.rs")).expect("read CUDA encode test shell");

    assert!(encode.lines().count() < 550, "CUDA encode facade regrew");
    assert!(encode.contains("#[cfg(test)]\nmod tests;"));
    assert!(!encode.contains("#[cfg(test)]\nmod tests {") && !encode.contains("#[test]"));
    assert!(shell.lines().count() < 125, "CUDA encode test shell regrew");
    for module in [
        "mod htj2k;",
        "mod packetization;",
        "mod resident;",
        "mod resident_session;",
        "mod resident_tiles;",
        "mod routing;",
        "mod transforms;",
    ] {
        assert!(shell.contains(module), "test shell must contain {module}");
    }
    for helper in [
        "fn assert_strict_cuda_classic_tier1_error",
        "struct CudaTestEncodeRequest",
        "fn encode_with_cuda_test_accelerator",
    ] {
        assert!(shell.contains(helper), "test shell must own {helper}");
    }
    assert_real_module("mod.rs", &shell);

    let mut test_count = 0usize;
    for (relative, max_lines, expected_tests) in TEST_MODULES {
        let source = fs::read_to_string(test_root.join(relative))
            .unwrap_or_else(|error| panic!("read encode/tests/{relative}: {error}"));
        assert!(
            source.lines().count() < *max_lines,
            "encode/tests/{relative} exceeded its focused line-count ratchet"
        );
        assert_real_module(relative, &source);
        assert_eq!(
            source.matches("#[test]").count(),
            expected_tests.len(),
            "encode/tests/{relative} test inventory changed"
        );
        for test in *expected_tests {
            assert_eq!(
                source.matches(&format!("fn {test}(")).count(),
                1,
                "encode/tests/{relative} must own exactly one {test}"
            );
        }
        test_count += expected_tests.len();
    }
    assert_eq!(
        test_count, 43,
        "CUDA encode test inventory must remain exact"
    );
}

fn assert_real_module(relative: &str, source: &str) {
    assert!(
        !source.contains("include!(") && !source.lines().any(|line| line.contains("::*")),
        "encode/tests/{relative} must use explicit real-module boundaries"
    );
}
