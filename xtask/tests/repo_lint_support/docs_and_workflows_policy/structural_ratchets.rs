// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, read_source_files, repo_root,
    FilePatternCheck, PatternCheck,
};

mod lint_ratchets;

#[test]
fn large_test_files_stay_split_by_axis() {
    let root = repo_root();
    for (relative, max_lines) in [
        ("crates/j2k-metal/src/encode/tests.rs", 150),
        ("crates/j2k-metal/src/encode/tests/batch.rs", 450),
        ("crates/j2k-metal/src/encode/tests/dwt_parity.rs", 250),
        ("crates/j2k-metal/src/encode/tests/kernels.rs", 1_300),
        ("crates/j2k-metal/src/encode/tests/layouts.rs", 250),
        ("crates/j2k-metal/src/encode/tests/resident_batches.rs", 725),
        ("crates/j2k-metal/src/encode/tests/resident_buffers.rs", 950),
        ("crates/j2k-metal/src/encode/tests/routing.rs", 850),
        ("crates/j2k-metal/src/encode/tests/stage_validation.rs", 650),
        ("crates/j2k-metal/src/encode/tests/stats_inflight.rs", 950),
        ("crates/j2k-jpeg-metal/src/tests.rs", 2_400),
        ("crates/j2k-jpeg-metal/src/tests/reusable_output.rs", 250),
        ("crates/j2k-jpeg-metal/src/tests/textures.rs", 2_300),
        ("crates/j2k-jpeg-metal/src/tests/textures/residency.rs", 175),
        ("crates/j2k-cuda-runtime/src/tests.rs", 2_300),
        ("crates/j2k-cuda-runtime/src/tests/pipeline.rs", 2_400),
        ("crates/j2k-jpeg-cuda/tests/host_surface.rs", 25),
        ("crates/j2k-jpeg-cuda/tests/host_surface/batch.rs", 200),
        (
            "crates/j2k-jpeg-cuda/tests/host_surface/caller_buffers.rs",
            225,
        ),
        (
            "crates/j2k-jpeg-cuda/tests/host_surface/diagnostics.rs",
            110,
        ),
        (
            "crates/j2k-jpeg-cuda/tests/host_surface/owned_decode.rs",
            100,
        ),
        (
            "crates/j2k-jpeg-cuda/tests/host_surface/output_initialization.rs",
            125,
        ),
        ("crates/j2k-jpeg-cuda/tests/host_surface/regions.rs", 180),
        ("crates/j2k-jpeg-cuda/tests/host_surface/routing.rs", 200),
        (
            "crates/j2k-jpeg-cuda/tests/host_surface/submissions.rs",
            125,
        ),
        ("crates/j2k-jpeg-cuda/tests/host_surface/support.rs", 125),
        ("crates/j2k-jpeg/tests/decode_into.rs", 2_000),
        ("crates/j2k-jpeg/tests/decode_into/lossless.rs", 1_600),
        ("crates/j2k-jpeg/tests/decode_into/color.rs", 900),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below the split test-file line-count ratchet"
        );
    }
}

const REPO_LINT_POLICY_LINE_LIMITS: &[(&str, usize)] = &[
        (
            "xtask/tests/repo_lint_support/audit_integrity_policy.rs",
            200,
        ),
        (
            "xtask/tests/repo_lint_support/docs_and_workflows_policy.rs",
            2_750,
        ),
        (
            "xtask/tests/repo_lint_support/encode_compare_structure_policy.rs",
            250,
        ),
        (
            "xtask/tests/repo_lint_support/fixture_compare_structure_policy.rs",
            250,
        ),
        ("xtask/tests/repo_lint_support/gpu_adapter_policy.rs", 1_800),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_decoder_policy.rs",
            250,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_decoder_policy/architecture.rs",
            175,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_decoder_policy/color_runtime.rs",
            150,
        ),
        ("xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_decoder_policy/resident_leaf_structure.rs", 75),
        ("xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_decoder_policy/resident_leaf_structure/classic.rs", 50),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy.rs",
            200,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/allocation.rs",
            225,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/lifecycle.rs",
            175,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/lifecycle/context.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/lifecycle/context/transitions.rs",
            100,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/lifecycle/queued.rs",
            70,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/lifecycle/queued/status.rs",
            50,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/validation.rs",
            225,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/validation/htj2k_output.rs",
            175,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/cuda_runtime_safety_policy/validation/htj2k_output/planning.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy.rs",
            100,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/allocation_sources.rs",
            300,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/checks.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/checks/adapter_checks.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/checks/checkpoint_checks.rs",
            110,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/checks/packet_checks.rs",
            130,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/checks/structure_checks.rs",
            140,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/encoder_checks/contracts.rs",
            110,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_allocation_policy/gpu_capacity.rs",
            100,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_metal_compute_structure_policy.rs",
            300,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/jpeg_metal_viewport_structure_policy.rs",
            175,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_adapter_policy/resident_encode_policy.rs",
            150,
        ),
        (
            "xtask/tests/repo_lint_support/gpu_device_structure_policy.rs",
            500,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy.rs",
            400,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy/adapter_tests.rs",
            260,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy/owned_output.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_decoder_structure_policy/support_contracts.rs",
            100,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_encoder_structure_policy.rs",
            225,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_restart_policy.rs",
            125,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_restart_policy/counts.rs",
            75,
        ),
        (
            "xtask/tests/repo_lint_support/jpeg_metal_resource_safety_policy.rs",
            350,
        ),
        (
            "xtask/tests/repo_lint_support/metal_compute_structure_policy.rs",
            550,
        ),
        (
            "xtask/tests/repo_lint_support/tilecodec_error_policy.rs",
            100,
        ),
        ("xtask/tests/repo_lint_support/transcode_api_policy.rs", 125),
        (
            "xtask/tests/repo_lint_support/transcode_structure_policy.rs",
            375,
        ),
        (
            "xtask/tests/repo_lint_support/transcode_structure_policy/cpu.rs",
            250,
        ),
        (
            "xtask/tests/repo_lint_support/transcode_structure_policy/cpu/batch.rs",
            225,
        ),
        (
            "xtask/tests/repo_lint_support/transcode_structure_policy/metal.rs",
            150,
        ),
        (
            "xtask/tests/repo_lint_support/xtask_main_structure_policy.rs",
            300,
        ),
        (
            "xtask/tests/repo_lint_support/xtask_main_structure_policy/codegen.rs",
            100,
        ),
        (
            "xtask/tests/repo_lint_support/xtask_main_structure_policy/lint_policy.rs",
            50,
        ),
        (
            "xtask/tests/repo_lint_support/xtask_main_structure_policy/release_integrity.rs",
            100,
        ),
];

#[test]
fn repo_lint_policy_support_files_stay_split_by_axis() {
    let root = repo_root();
    for &(relative, max_lines) in REPO_LINT_POLICY_LINE_LIMITS {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below the split repo-lint policy line-count ratchet"
        );
    }
}

#[test]
fn decode_capability_correctness_regressions_are_guarded() {
    let root = repo_root();
    let native_codestream = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/codestream/header.rs",
            "crates/j2k-native/src/j2c/codestream/tests.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "target-resolution shrink-factor arithmetic",
        &native_codestream,
    )
    .required(&[
        ".checked_shl(u32::from(skipped_resolution_levels))",
        ".checked_mul(resolution_shrink_factor)",
        "size_data.checked_image_width()?;",
        "size_data.checked_image_height()?;",
        "checked_image_dimensions_reject_shrink_factor_overflow",
    ])]);
    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("crates/j2k-jpeg/tests/inspect.rs")
            .named("JPEG progressive inspect/decode agreement fixtures")
            .required(&[
                "fn inspect_and_decoder_info_agree_for_progressive_fixtures()",
                "progressive_8x8_jpeg()",
                "progressive_12bit_grayscale_8x8_jpeg()",
                "progressive_12bit_rgb_8x8_jpeg()",
                "assert_eq!(decoder.info(), &inspected, \"{label}\");",
            ])],
    );
}

#[test]
fn docs_and_workflows_policy_children_stay_split_by_responsibility() {
    let root = repo_root();
    let shell_relative = "xtask/tests/repo_lint_support/docs_and_workflows_policy.rs";
    let shell = fs::read_to_string(root.join(shell_relative))
        .unwrap_or_else(|error| panic!("read {shell_relative}: {error}"));
    assert!(
        shell.lines().count() < 25,
        "{shell_relative} must remain a focused child-module shell"
    );
    assert_pattern_checks(&[
        PatternCheck::new("docs/workflows policy module shell", &shell).required(&[
            "mod adoption_benchmark_policy;",
            "mod decoder_fixture_policy;",
            "mod documentation_api_evidence;",
            "mod duplication_policy;",
            "mod encoder_architecture_policy;",
            "mod stable_api_evidence;",
            "mod stable_api_governance;",
            "mod structural_ratchets;",
            "mod workflow_coverage_policy;",
        ]),
    ]);

    let policy_dir = "xtask/tests/repo_lint_support/docs_and_workflows_policy";
    for (leaf, max_lines) in [
        ("adoption_benchmark_policy.rs", 225),
        ("documentation_api_evidence.rs", 100),
        ("stable_api_evidence.rs", 175),
        ("stable_api_governance.rs", 225),
        ("workflow_coverage_policy.rs", 500),
        ("structural_ratchets.rs", 375),
        ("structural_ratchets/lint_ratchets.rs", 150),
        ("duplication_policy.rs", 625),
        ("duplication_policy/cache_identity.rs", 100),
        ("duplication_policy/classic_mq.rs", 75),
        ("encoder_architecture_policy.rs", 700),
        ("encoder_architecture_policy/native_contracts.rs", 150),
        ("decoder_fixture_policy.rs", 475),
    ] {
        let relative = format!("{policy_dir}/{leaf}");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused policy line-count ratchet of {max_lines}"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;")
                && !source
                    .lines()
                    .any(|line| line.trim_start().starts_with("include!(")),
            "{relative} must use explicit real-Rust module boundaries"
        );
    }
}
