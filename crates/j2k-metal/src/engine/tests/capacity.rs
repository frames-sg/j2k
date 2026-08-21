// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::abi::{
    J2kClassicEncodeBatchJob, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS, J2K_HT_ENCODE_BASE_OUTPUT_SIZE,
};
use super::super::{
    accumulate_classic_tier1_scan_estimates, classic_encode_code_blocks_pipeline_kind,
    classic_encode_output_capacity, classic_encode_segment_capacity,
    classic_packet_output_capacity, classic_tier1_pass_class_counts, ht_encode_output_capacity,
    J2kClassicEncodePipelineKind, J2kLosslessCodestreamAssemblyJob,
    J2kLosslessCodestreamBlockCodingMode, J2kResidentEncodeStageStats,
};

#[test]
fn classic_encode_output_capacity_keeps_conservative_default() {
    let capacity = classic_encode_output_capacity(64, 64, 11).expect("classic output capacity");

    assert_eq!(capacity, 64 * 64 * 11 * 8 + 4097);
}

#[test]
fn classic_encode_segment_capacity_uses_coding_style_bound() {
    assert_eq!(classic_encode_segment_capacity(0, 16), 1);
    assert_eq!(
        classic_encode_segment_capacity(J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS, 9),
        11
    );
    assert_eq!(
        classic_encode_segment_capacity(J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS, 16),
        25
    );
    assert_eq!(
        classic_encode_segment_capacity(J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS, 16),
        46
    );
}

#[test]
fn classic_tier1_pass_class_counts_split_bypass_pass_types() {
    let counts =
        classic_tier1_pass_class_counts(23, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS);

    assert_eq!(counts.arithmetic, 14);
    assert_eq!(counts.raw, 9);
    assert_eq!(counts.cleanup, 8);
    assert_eq!(counts.sigprop, 8);
    assert_eq!(counts.magref, 7);
    assert_eq!(counts.arithmetic_cleanup, 8);
    assert_eq!(counts.arithmetic_sigprop, 3);
    assert_eq!(counts.arithmetic_magref, 3);
    assert_eq!(counts.raw_sigprop, 5);
    assert_eq!(counts.raw_magref, 4);
}

#[test]
fn classic_tier1_pass_class_counts_style0_stays_arithmetic() {
    let counts = classic_tier1_pass_class_counts(5, 0);

    assert_eq!(counts.arithmetic, 5);
    assert_eq!(counts.raw, 0);
    assert_eq!(counts.cleanup, 2);
    assert_eq!(counts.sigprop, 2);
    assert_eq!(counts.magref, 1);
    assert_eq!(counts.arithmetic_cleanup, 2);
    assert_eq!(counts.arithmetic_sigprop, 2);
    assert_eq!(counts.arithmetic_magref, 1);
    assert_eq!(counts.raw_sigprop, 0);
    assert_eq!(counts.raw_magref, 0);
}

#[test]
fn classic_tier1_scan_estimates_multiply_passes_by_block_area() {
    let pass_counts =
        classic_tier1_pass_class_counts(23, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS);
    let mut stats = J2kResidentEncodeStageStats::default();

    accumulate_classic_tier1_scan_estimates(&mut stats, pass_counts, 32 * 32);

    assert_eq!(stats.tier1_full_scan_coeff_visit_count_total, 23 * 1024);
    assert_eq!(
        stats.tier1_arithmetic_scan_coeff_visit_count_total,
        14 * 1024
    );
    assert_eq!(stats.tier1_raw_scan_coeff_visit_count_total, 9 * 1024);
    assert_eq!(stats.tier1_cleanup_scan_coeff_visit_count_total, 8 * 1024);
    assert_eq!(stats.tier1_sigprop_scan_coeff_visit_count_total, 8 * 1024);
    assert_eq!(stats.tier1_magref_scan_coeff_visit_count_total, 7 * 1024);
    assert_eq!(stats.max_tier1_full_scan_coeff_visits_per_block, 23 * 1024);
}

#[test]
fn classic_packet_output_capacity_uses_raw_sample_bound_when_smaller() {
    let codestream = J2kLosslessCodestreamAssemblyJob {
        width: 512,
        height: 512,
        component_count: 3,
        bit_depth: 8,
        signed: false,
        num_decomposition_levels: 3,
        use_mct: true,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        progression_order: j2k_native::EncodeProgressionOrder::Lrcp,
        write_tlm: false,
        block_coding_mode: J2kLosslessCodestreamBlockCodingMode::Classic,
    };
    let header_capacity = 1024 * 256 + 4096;
    let conservative_capacity = 12 * 1024 * 1024;
    let packet_descriptor_count = 3;

    let capacity = classic_packet_output_capacity(
        conservative_capacity,
        header_capacity,
        packet_descriptor_count,
        codestream,
    )
    .expect("classic packet capacity");

    let raw_bytes = 512 * 512 * 3;
    let descriptor_slack = packet_descriptor_count * 256;
    assert_eq!(
        capacity,
        raw_bytes + header_capacity + descriptor_slack + 64 * 1024
    );

    let tiny_tier1_capacity = 4096;
    let clamped = classic_packet_output_capacity(
        tiny_tier1_capacity,
        header_capacity,
        packet_descriptor_count,
        codestream,
    )
    .expect("classic packet capacity");
    let conservative_packet_capacity =
        tiny_tier1_capacity + header_capacity * packet_descriptor_count + 1024;
    assert_eq!(clamped, conservative_packet_capacity);
}

#[test]
fn ht_encode_output_capacity_scales_with_code_block_area() {
    let max_block = ht_encode_output_capacity(128, 128).expect("max HT output capacity");
    assert_eq!(max_block, J2K_HT_ENCODE_BASE_OUTPUT_SIZE);

    let smaller_block = ht_encode_output_capacity(32, 32).expect("scaled HT output capacity");
    assert!(smaller_block < max_block / 2);
    assert!(smaller_block >= 8192);
}

#[test]
fn classic_encode_pipeline_kind_prefers_style0_32_for_resident_jobs() {
    let jobs = [J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: 0,
        ..J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        classic_encode_code_blocks_pipeline_kind(&jobs),
        J2kClassicEncodePipelineKind::Style0_32
    );
}

#[test]
fn classic_encode_pipeline_kind_prefers_bypass_32_for_resident_jobs() {
    let jobs = [J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        total_bitplanes: 31,
        ..J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        classic_encode_code_blocks_pipeline_kind(&jobs),
        J2kClassicEncodePipelineKind::Bypass32
    );
}

#[test]
fn classic_encode_pipeline_kind_prefers_bypass_u16_32_for_low_bitplane_resident_jobs() {
    let jobs = [J2kClassicEncodeBatchJob {
        width: 32,
        height: 32,
        style_flags: J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
        total_bitplanes: 16,
        ..J2kClassicEncodeBatchJob::default()
    }];

    assert_eq!(
        classic_encode_code_blocks_pipeline_kind(&jobs),
        J2kClassicEncodePipelineKind::BypassU16_32
    );
}
