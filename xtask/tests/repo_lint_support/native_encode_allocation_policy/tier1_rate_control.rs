// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier-1, rate-control, and accelerator-metadata allocation ratchets.

use super::{assert_pattern_checks, read, read_source_files, repo_root, PatternCheck};

const TIER1_RATE_CONTROL_MODULES: &[(&str, usize)] = &[
    (
        "crates/j2k-native/src/j2c/encode/code_block_metadata.rs",
        260,
    ),
    ("crates/j2k-native/src/j2c/arithmetic_encoder.rs", 320),
    ("crates/j2k-native/src/j2c/bitplane_encode.rs", 420),
    (
        "crates/j2k-native/src/j2c/bitplane_encode/allocation.rs",
        220,
    ),
    ("crates/j2k-native/src/j2c/bitplane_encode/segments.rs", 260),
    (
        "crates/j2k-native/src/j2c/bitplane_encode/segments/encoder.rs",
        320,
    ),
    ("crates/j2k-native/src/j2c/bitplane_encode/tokens.rs", 300),
    (
        "crates/j2k-native/src/j2c/bitplane_encode/tokens/reader.rs",
        80,
    ),
    ("crates/j2k-native/src/j2c/coefficient_view.rs", 320),
    (
        "crates/j2k-native/src/j2c/ht_block_encode/allocation.rs",
        220,
    ),
    (
        "crates/j2k-native/src/j2c/ht_block_encode/allocation/refinement.rs",
        140,
    ),
    (
        "crates/j2k-native/src/j2c/ht_block_encode/distribution.rs",
        400,
    ),
    ("crates/j2k-native/src/j2c/ht_block_encode/facade.rs", 150),
    (
        "crates/j2k-native/src/j2c/ht_block_encode/refinement.rs",
        340,
    ),
    (
        "crates/j2k-native/src/j2c/ht_block_encode/refinement/writers.rs",
        200,
    ),
    ("crates/j2k-native/src/j2c/ht_block_encode/writers.rs", 320),
    ("crates/j2k-native/src/j2c/encode/tier1_allocation.rs", 380),
    ("crates/j2k-native/src/j2c/encode/tier1_driver.rs", 660),
    ("crates/j2k-native/src/j2c/encode/tier1_driver/cpu.rs", 230),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/cpu/waves.rs",
        300,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/cpu/direct_i64.rs",
        120,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/layout.rs",
        80,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/output.rs",
        180,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/output/validation.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/scratch.rs",
        280,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/tests.rs",
        250,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/tests/accelerator.rs",
        220,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/tests/accelerator_metadata.rs",
        260,
    ),
    (
        "crates/j2k-native/src/j2c/encode/tier1_driver/tests/error_taxonomy.rs",
        100,
    ),
    ("crates/j2k-native/src/j2c/encode/rate_control.rs", 210),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/assignment.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted.rs",
        310,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted/classic.rs",
        280,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/assignment/legacy.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/contributions.rs",
        100,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/contributions/classic.rs",
        180,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/contributions/classic/build.rs",
        160,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/contributions/ht.rs",
        300,
    ),
    (
        "crates/j2k-native/src/j2c/encode/rate_control/contributions/ht/layout.rs",
        90,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered.rs",
        330,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/assignment.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/assignment/location.rs",
        80,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/classic.rs",
        210,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ht.rs",
        250,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ht/prepare.rs",
        110,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output.rs",
        250,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output/contributions.rs",
        120,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output/state.rs",
        180,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ownership.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/packet.rs",
        130,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/state.rs",
        80,
    ),
    (
        "crates/j2k-native/src/j2c/encode/prepared_packets/layered/tests.rs",
        180,
    ),
    (
        "crates/j2k-native/src/j2c/encode/packet_plan/accelerator_metadata.rs",
        110,
    ),
    (
        "crates/j2k-native/src/j2c/encode/packet_plan/accelerator_metadata/tests.rs",
        120,
    ),
    ("crates/j2k-native/src/j2c/encode/subband/tests.rs", 90),
];

#[test]
fn native_tier1_rate_control_keeps_focused_module_boundaries() {
    for (relative, ceiling) in TIER1_RATE_CONTROL_MODULES {
        let source = read(relative);
        let lines = source.lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; native Tier-1/rate-control ceiling is {ceiling}"
        );
    }

    let facades = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode/tier1_driver.rs",
            "crates/j2k-native/src/j2c/encode/rate_control.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new("native Tier-1 module wiring", &facades)
        .required(&[
            "mod cpu;",
            "mod layout;",
            "mod output;",
            "mod scratch;",
            "mod accounted;",
            "mod classic;",
            "mod ht;",
            "mod assignment;",
            "mod ownership;",
            "mod packet;",
            "mod state;",
            "LayeredRateControlState",
        ])
        .forbidden(&["#[path =", "clippy::too_many_arguments"])]);
}

fn tier1_allocation_production_sources() -> String {
    read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/arithmetic_encoder.rs",
            "crates/j2k-native/src/j2c/bitplane_encode.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/allocation.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/passes.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/segments.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/segments/encoder.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/tokens.rs",
            "crates/j2k-native/src/j2c/bitplane_encode/tokens/reader.rs",
            "crates/j2k-native/src/j2c/coefficient_view.rs",
            "crates/j2k-native/src/j2c/encode/code_block_metadata.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/allocation.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/allocation/refinement.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/cleanup.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/distribution.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/emit.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/facade.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/quad.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/refinement.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/refinement/writers.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/writers.rs",
            "crates/j2k-native/src/j2c/encode/tier1_allocation.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/cpu.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/cpu/waves.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/layout.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/output.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/output/validation.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/scratch.rs",
            "crates/j2k-native/src/j2c/encode/rate_control.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted/classic.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions/classic.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions/classic/build.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions/ht.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/contributions/ht/layout.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/assignment.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/assignment/location.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/classic.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ht.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ht/prepare.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output/contributions.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/output/state.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/ownership.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/packet.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/state.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/accelerator_metadata.rs",
        ],
    )
}

fn tier1_allocation_coverage_sources() -> String {
    read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/bitplane_encode/tests.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/tests.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/tests/accelerator.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/tests/accelerator_metadata.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/tests/error_taxonomy.rs",
            "crates/j2k-native/src/j2c/encode/subband/tests.rs",
            "crates/j2k-native/src/j2c/encode/prepared_packets/layered/tests.rs",
            "crates/j2k-native/src/j2c/encode/packet_plan/accelerator_metadata/tests.rs",
        ],
    )
}

fn accelerator_metadata_sources() -> String {
    read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/encode.rs",
            "crates/j2k-native/src/j2c/encode/code_block_metadata.rs",
            "crates/j2k-native/src/j2c/encode/precomputed/validation.rs",
            "crates/j2k-native/src/j2c/encode/subband.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver.rs",
            "crates/j2k-native/src/j2c/encode/tier1_driver/output/validation.rs",
        ],
    )
}

#[test]
fn native_tier1_rate_control_allocations_remain_typed_and_fallible() {
    let production = tier1_allocation_production_sources();
    let coverage = tier1_allocation_coverage_sources();
    let accelerator_metadata = accelerator_metadata_sources();

    assert_pattern_checks(&[
        PatternCheck::new("phase-bounded native Tier-1 and rate control", &production)
            .required(&[
                "PreparedCodeBlockCoefficients::I32",
                "PreparedCodeBlockCoefficients::I64",
                "Tier1PhaseTracker",
                "try_reserve_exact(",
                "try_reserve_additional(",
                "try_reserve_untracked_bounded(",
                "check_classic_wave(",
                "check_ht_wave(",
                "chunks_mut(",
                "try_encode_code_block(",
                "try_encode_code_block_with_passes(",
                "classic_layer_contributions_accounted(",
                "build_classic_layer_contribution(",
                "ClassicContributionOwners::new(",
                "classic_segment_layer_mut(",
                "ht_segment_layer_mut(",
                "ht_layer_contributions_accounted(",
                "ht_contribution_layout(",
                "try_public_packetization_resolutions(",
                "packet_descriptor_capacity",
                "resolution_packet_capacity",
            ])
            .forbidden(&[
                "Vec::with_capacity(",
                ".to_vec(",
                ".collect::<",
                ".collect()",
                "vec![",
                ".map(i64::from)",
            ]),
        PatternCheck::new(
            "accelerator Tier-1 metadata boundary",
            &accelerator_metadata,
        )
        .required(&[
            "mod code_block_metadata;",
            "fn validate_ht_code_block_metadata(",
            "fn validate_accelerated_ht_code_block(",
            "fn validate_accelerated_ht_job_output(",
            "fn validate_accelerated_classic_code_block(",
            "validate_ht_code_block(block, total_bitplanes)",
            "validate_accelerated_ht_code_block(block, plan.total_bitplanes, 1)",
            "HT Tier-1 code-block batch encode",
            "classic Tier-1 code-block batch encode",
            "fused HT subband encode",
        ]),
        PatternCheck::new("native Tier-1 boundary regressions", &coverage).required(&[
            "tier1_phase_accepts_exact_peak_and_rejects_cap_minus_one",
            "token_segments_must_follow_selective_bypass_coding_modes",
            "ordinary_i32_coefficients_are_borrowed_without_a_downcast_graph",
            "tier1_batch_failure_keeps_the_accelerator_operation_category",
            "malformed_tier1_batch_output_keeps_the_accelerator_operation_category",
            "accepted_tier1_batch_over_cap_does_not_fall_back",
            "serial_accelerator_segment_metadata_is_checked_before_conversion",
            "malformed_ht_batch_metadata_keeps_the_accelerator_operation_category",
            "divergent_ht_single_pass_count_keeps_the_accelerator_operation_category",
            "omitted_nonzero_ht_block_keeps_the_accelerator_operation_category",
            "excessive_classic_pass_count_keeps_the_accelerator_operation_category",
            "malformed_classic_segments_keep_the_accelerator_operation_category",
            "wrong_classic_zero_bitplanes_keep_the_accelerator_operation_category",
            "malformed_fused_ht_metadata_keeps_the_accelerator_operation_category",
            "layered_rate_control_accepts_exact_peak_and_rejects_cap_minus_one",
            "multilayer_contributions_preserve_classic_payload_bytes",
            "packet_accelerator_metadata_accepts_exact_peak_and_rejects_cap_minus_one",
        ]),
    ]);
}
