// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_repo, read_runtime, PatternCheck};

#[test]
fn jpeg_decode_security_regressions_remain_adversarial() {
    let tests = read_runtime("jpeg/validation/tests/decode_security.rs");
    let adapter_tests = read_repo("crates/j2k-jpeg-cuda/src/owned_decode/tests.rs");
    let gpu_tests = read_repo("crates/j2k-jpeg-cuda/tests/host_surface/owned_decode.rs");
    let output_initialization =
        read_repo("crates/j2k-jpeg-cuda/tests/host_surface/output_initialization.rs");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA JPEG decode adversarial tests", &tests).required(&[
            "jpeg_decode_validation_accepts_exact_odd_edge_grids_for_every_sampling",
            "jpeg_decode_validation_accepts_complete_nonuniform_checkpoint_partition",
            "jpeg_decode_validation_rejects_grid_and_checkpoint_coverage_gaps",
            "jpeg_decode_validation_rejects_malformed_checkpoint_bit_state",
            "jpeg_decode_validation_rejects_u32_unaddressable_output_before_allocation",
            "jpeg_decode_validation_rejects_noncanonical_and_role_invalid_huffman_tables",
            "jpeg_entropy_diagnostic_validates_huffman_tables_before_launch",
            "from_jpeg_bits_values(",
            "all_ones.max_code[1] = 1;",
            "too_many.y_dc_table.values_len = 257;",
        ]),
        PatternCheck::new(
            "CUDA JPEG multi-checkpoint plan regressions",
            &adapter_tests,
        )
        .required(&[
            "multi_checkpoint_420_plans_are_ordered_and_start_from_a_clean_state",
            "generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, None)",
            "generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, Some(1))",
            "\"nonrestart\", nonrestart.as_slice()",
            "\"restart_interval_1\", restart.as_slice()",
            "pair[0].mcu_index < pair[1].mcu_index",
            "pair[0].entropy_pos <= pair[1].entropy_pos",
            "dri_equal_to_total_mcus_has_only_the_clean_initial_checkpoint",
            "assert_eq!(packet.entropy_checkpoints.len(), 1);",
        ])
        .normalized_required(&[
            "assert_eq!( plan.entropy_checkpoints.len(), 2, \"{name} must exercise one checkpoint per MCU\" )",
        ]),
        PatternCheck::new("CUDA JPEG multi-checkpoint GPU integration", &gpu_tests).required(&[
            "explicit_cuda_multi_checkpoint_420_uses_owned_decode_when_required",
            "explicit_cuda_restart_checkpoint_420_uses_owned_decode_when_required",
            "generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16)",
            "generated_rgb_jpeg_with_restart(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, Some(1))",
            "assert_full_frame_owned_cuda_decode_when_required(&input, (32, 16));",
        ]),
        PatternCheck::new("CUDA JPEG pitched output initialization", &output_initialization)
            .required(&[
                "caller_owned_pitched_decode_initializes_every_addressable_output_byte",
                "upload(&vec![0xa5; output_len])",
                "decode_tile_rgb8_into_cuda_buffer_with_session(",
                "output_start + row_bytes..output_start + pitch_bytes",
                ".all(|&byte| byte == 0)",
            ])
            .normalized_required(&[
                "assert_eq!( output.device_ptr(), sentinel_ptr, \"test must reuse sentinel\" )",
            ]),
    ]);
}
