// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn inflight_limited_runner_starts_next_item_before_slow_peer_finishes() {
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    #[derive(Default)]
    struct Probe {
        third_item_started: bool,
    }

    let probe = Arc::new((Mutex::new(Probe::default()), Condvar::new()));
    let task_probe = Arc::clone(&probe);

    let outcomes = super::super::collect_inflight_limited_ordered(vec![0usize, 1, 2], 2, move |_, item| {
        match item {
            0 => Ok(item),
            1 => {
                let (lock, cvar) = &*task_probe;
                let state = lock.lock().expect("probe mutex");
                let (state, _timeout) = cvar
                    .wait_timeout_while(state, Duration::from_millis(250), |state| {
                        !state.third_item_started
                    })
                    .expect("probe wait");
                if !state.third_item_started {
                    return Err(crate::Error::MetalKernel {
                        message:
                            "runner waited for the whole in-flight chunk before scheduling more work"
                                .to_string(),
                    });
                }
                Ok(item)
            }
            2 => {
                let (lock, cvar) = &*task_probe;
                let mut state = lock.lock().expect("probe mutex");
                state.third_item_started = true;
                cvar.notify_all();
                Ok(item)
            }
            _ => unreachable!("unexpected test item"),
        }
    })
    .expect("in-flight runner should slide past a slow peer");

    assert_eq!(outcomes.items, vec![0, 1, 2]);
    assert!(outcomes.max_observed_inflight_items <= 2);
    assert!(outcomes.max_observed_inflight_items > 0);
}

#[test]
fn submitted_lossless_metal_encode_public_api_is_available() {
    fn assert_batch_submission<
        S: DeviceSubmission<Output = Vec<EncodedJ2k>, Error = crate::Error>,
    >() {
    }
    fn assert_submit_batch_fn(
        _submit: for<'tiles, 'tile, 'options, 'session> fn(
            super::super::MetalLosslessEncodeBatchRequest<'tiles, 'tile>,
            &'options J2kLosslessEncodeOptions,
            &'session crate::MetalBackendSession,
        ) -> Result<
            crate::SubmittedJ2kLosslessMetalEncodeBatch,
            crate::Error,
        >,
    ) {
    }

    assert_batch_submission::<crate::SubmittedJ2kLosslessMetalEncodeBatch>();
    assert_submit_batch_fn(crate::submit_lossless_batch);
}

#[test]
fn submitted_lossless_metal_buffer_encode_public_api_is_available() {
    fn assert_buffer_batch_submission<
        S: DeviceSubmission<
            Output = super::super::MetalLosslessBufferEncodeBatchOutcome,
            Error = crate::Error,
        >,
    >() {
    }
    fn assert_submit_buffer_batch_fn(
        _submit: for<'tiles, 'tile, 'options, 'session> fn(
            super::super::MetalLosslessEncodeBatchRequest<'tiles, 'tile>,
            &'options J2kLosslessEncodeOptions,
            &'session crate::MetalBackendSession,
        ) -> Result<
            crate::SubmittedJ2kLosslessMetalBufferEncodeBatch,
            crate::Error,
        >,
    ) {
    }

    assert_buffer_batch_submission::<crate::SubmittedJ2kLosslessMetalBufferEncodeBatch>();
    assert_submit_buffer_batch_fn(crate::submit_lossless_batch_to_metal);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "exhaustive default-field assertion guards the public stats contract"
)]
fn resident_lossless_stage_stats_default_to_zero() {
    let stats = super::super::MetalLosslessEncodeBatchStats::default();

    assert_eq!(
        stats.stage_stats,
        super::super::MetalLosslessEncodeStageStats::default()
    );
    assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
    assert_eq!(stats.stage_stats.deinterleave_rct_duration, Duration::ZERO);
    assert_eq!(stats.stage_stats.dwt53_duration, Duration::ZERO);
    assert_eq!(
        stats.stage_stats.coefficient_extract_duration,
        Duration::ZERO
    );
    assert_eq!(stats.stage_stats.ht_block_encode_duration, Duration::ZERO);
    assert_eq!(
        stats.stage_stats.classic_tier1_setup_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.classic_block_encode_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.classic_packet_plan_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.classic_packet_buffer_setup_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.classic_command_buffer_commit_duration,
        Duration::ZERO
    );
    assert_eq!(stats.stage_stats.packet_block_prep_duration, Duration::ZERO);
    assert_eq!(stats.stage_stats.packetization_duration, Duration::ZERO);
    assert_eq!(
        stats.stage_stats.packet_payload_copy_gpu_duration,
        Duration::ZERO
    );
    assert_eq!(stats.stage_stats.gpu_elapsed_wall_duration, Duration::ZERO);
    assert_eq!(stats.stage_stats.classic_block_gpu_duration, Duration::ZERO);
    assert_eq!(
        stats.stage_stats.classic_tier1_density_gpu_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.codestream_assembly_duration,
        Duration::ZERO
    );
    assert_eq!(
        stats.stage_stats.codestream_payload_copy_gpu_duration,
        Duration::ZERO
    );
    assert_eq!(stats.stage_stats.tier1_output_capacity_total, 0);
    assert_eq!(stats.stage_stats.max_tier1_output_capacity, 0);
    assert_eq!(stats.stage_stats.tier1_output_used_bytes_total, 0);
    assert_eq!(stats.stage_stats.max_tier1_output_used_bytes, 0);
    assert_eq!(stats.stage_stats.tier1_coding_pass_count_total, 0);
    assert_eq!(stats.stage_stats.max_tier1_coding_passes_per_block, 0);
    assert_eq!(stats.stage_stats.tier1_arithmetic_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_raw_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_cleanup_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_sigprop_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_magref_pass_count_total, 0);
    assert_eq!(
        stats.stage_stats.tier1_arithmetic_cleanup_pass_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_arithmetic_sigprop_pass_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_arithmetic_magref_pass_count_total,
        0
    );
    assert_eq!(stats.stage_stats.tier1_raw_sigprop_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_raw_magref_pass_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_full_scan_coeff_visit_count_total, 0);
    assert_eq!(
        stats
            .stage_stats
            .tier1_arithmetic_scan_coeff_visit_count_total,
        0
    );
    assert_eq!(stats.stage_stats.tier1_raw_scan_coeff_visit_count_total, 0);
    assert_eq!(
        stats.stage_stats.tier1_cleanup_scan_coeff_visit_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_sigprop_scan_coeff_visit_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_magref_scan_coeff_visit_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.max_tier1_full_scan_coeff_visits_per_block,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_sigprop_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_sigprop_new_significant_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_magref_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_arithmetic_sigprop_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_arithmetic_sigprop_new_significant_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_raw_sigprop_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_raw_sigprop_new_significant_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_arithmetic_magref_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats
            .stage_stats
            .tier1_raw_magref_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_cleanup_active_candidate_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.tier1_cleanup_new_significant_count_total,
        0
    );
    assert_eq!(stats.stage_stats.tier1_cleanup_rlc_stripe_count_total, 0);
    assert_eq!(
        stats.stage_stats.tier1_cleanup_rlc_zero_stripe_count_total,
        0
    );
    assert_eq!(stats.stage_stats.tier1_nonzero_block_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_zero_block_count_total, 0);
    assert_eq!(stats.stage_stats.tier1_missing_bitplane_count_total, 0);
    assert_eq!(stats.stage_stats.max_tier1_missing_bitplanes_per_block, 0);
    assert_eq!(stats.stage_stats.tier1_segment_count_total, 0);
    assert_eq!(stats.stage_stats.max_tier1_segments_per_block, 0);
    assert_eq!(stats.stage_stats.packet_payload_copy_job_capacity_total, 0);
    assert_eq!(stats.stage_stats.max_packet_payload_copy_jobs_per_tile, 0);
    assert_eq!(stats.stage_stats.packet_payload_copy_job_count_total, 0);
    assert_eq!(
        stats.stage_stats.max_packet_payload_copy_jobs_used_per_tile,
        0
    );
    assert_eq!(stats.stage_stats.packet_payload_copy_bytes_total, 0);
    assert_eq!(stats.stage_stats.max_packet_payload_copy_bytes_per_tile, 0);
    assert_eq!(
        stats.stage_stats.packet_payload_copy_small_job_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.packet_payload_copy_medium_job_count_total,
        0
    );
    assert_eq!(
        stats.stage_stats.packet_payload_copy_large_job_count_total,
        0
    );
    assert_eq!(stats.stage_stats.packet_output_capacity_total, 0);
    assert_eq!(stats.stage_stats.max_packet_output_capacity, 0);
    assert_eq!(stats.stage_stats.packet_output_used_bytes_total, 0);
    assert_eq!(stats.stage_stats.max_packet_output_used_bytes, 0);
    assert_eq!(stats.stage_stats.sync_wait_duration, Duration::ZERO);
    assert_eq!(stats.stage_stats.host_readback_duration, Duration::ZERO);
    assert!(!stats.stage_stats.has_timings());
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "exhaustive saturation assertion guards every stats field"
)]
fn resident_lossless_stage_stats_add_assign_saturates() {
    let mut stats = super::super::MetalLosslessEncodeStageStats {
        plan_duration: Duration::MAX,
        tile_count: usize::MAX,
        ..super::super::MetalLosslessEncodeStageStats::default()
    };

    stats.add_assign(super::super::MetalLosslessEncodeStageStats {
        plan_duration: Duration::from_micros(1),
        prepare_submit_duration: Duration::from_micros(2),
        classic_tier1_setup_duration: Duration::from_micros(4),
        classic_block_encode_duration: Duration::from_micros(5),
        classic_tier1_token_pack_duration: Duration::from_micros(9),
        classic_packet_plan_duration: Duration::from_micros(6),
        classic_packet_buffer_setup_duration: Duration::from_micros(7),
        classic_command_buffer_commit_duration: Duration::from_micros(8),
        packet_payload_copy_job_capacity_total: 11,
        max_packet_payload_copy_jobs_per_tile: 5,
        packet_payload_copy_job_count_total: 13,
        max_packet_payload_copy_jobs_used_per_tile: 6,
        packet_payload_copy_bytes_total: 23,
        max_packet_payload_copy_bytes_per_tile: 12,
        packet_payload_copy_small_job_count_total: 2,
        packet_payload_copy_medium_job_count_total: 3,
        packet_payload_copy_large_job_count_total: 4,
        tier1_output_capacity_total: 17,
        max_tier1_output_capacity: 9,
        tier1_output_used_bytes_total: 19,
        max_tier1_output_used_bytes: 10,
        tier1_segment_capacity_total: 25,
        max_tier1_segment_capacity_per_block: 11,
        tier1_coding_pass_count_total: 31,
        max_tier1_coding_passes_per_block: 8,
        tier1_arithmetic_pass_count_total: 21,
        tier1_raw_pass_count_total: 10,
        tier1_cleanup_pass_count_total: 11,
        tier1_sigprop_pass_count_total: 10,
        tier1_magref_pass_count_total: 10,
        tier1_arithmetic_cleanup_pass_count_total: 11,
        tier1_arithmetic_sigprop_pass_count_total: 6,
        tier1_arithmetic_magref_pass_count_total: 4,
        tier1_raw_sigprop_pass_count_total: 4,
        tier1_raw_magref_pass_count_total: 6,
        tier1_full_scan_coeff_visit_count_total: 31_744,
        tier1_arithmetic_scan_coeff_visit_count_total: 21_504,
        tier1_raw_scan_coeff_visit_count_total: 10_240,
        tier1_cleanup_scan_coeff_visit_count_total: 11_264,
        tier1_sigprop_scan_coeff_visit_count_total: 10_240,
        tier1_magref_scan_coeff_visit_count_total: 10_240,
        max_tier1_full_scan_coeff_visits_per_block: 8_192,
        tier1_sigprop_active_candidate_count_total: 101,
        tier1_sigprop_new_significant_count_total: 37,
        tier1_magref_active_candidate_count_total: 203,
        tier1_arithmetic_sigprop_active_candidate_count_total: 61,
        tier1_arithmetic_sigprop_new_significant_count_total: 23,
        tier1_raw_sigprop_active_candidate_count_total: 40,
        tier1_raw_sigprop_new_significant_count_total: 14,
        tier1_arithmetic_magref_active_candidate_count_total: 123,
        tier1_raw_magref_active_candidate_count_total: 80,
        tier1_cleanup_active_candidate_count_total: 307,
        tier1_cleanup_new_significant_count_total: 41,
        tier1_cleanup_rlc_stripe_count_total: 53,
        tier1_cleanup_rlc_zero_stripe_count_total: 47,
        tier1_token_pack_output_bytes_total: 29,
        max_tier1_token_pack_output_bytes_per_block: 15,
        tier1_nonzero_block_count_total: 2,
        tier1_zero_block_count_total: 1,
        tier1_missing_bitplane_count_total: 5,
        max_tier1_missing_bitplanes_per_block: 4,
        tier1_segment_count_total: 7,
        max_tier1_segments_per_block: 3,
        packet_output_capacity_total: 17,
        max_packet_output_capacity: 9,
        packet_output_used_bytes_total: 19,
        max_packet_output_used_bytes: 10,
        tile_count: 1,
        code_block_count: 3,
        ..super::super::MetalLosslessEncodeStageStats::default()
    });

    assert_eq!(stats.plan_duration, Duration::MAX);
    assert_eq!(stats.prepare_submit_duration, Duration::from_micros(2));
    assert_eq!(stats.classic_tier1_setup_duration, Duration::from_micros(4));
    assert_eq!(
        stats.classic_block_encode_duration,
        Duration::from_micros(5)
    );
    assert_eq!(
        stats.classic_tier1_token_pack_duration,
        Duration::from_micros(9)
    );
    assert_eq!(stats.classic_packet_plan_duration, Duration::from_micros(6));
    assert_eq!(
        stats.classic_packet_buffer_setup_duration,
        Duration::from_micros(7)
    );
    assert_eq!(
        stats.classic_command_buffer_commit_duration,
        Duration::from_micros(8)
    );
    assert_eq!(stats.tile_count, usize::MAX);
    assert_eq!(stats.code_block_count, 3);
    assert_eq!(stats.packet_payload_copy_job_capacity_total, 11);
    assert_eq!(stats.max_packet_payload_copy_jobs_per_tile, 5);
    assert_eq!(stats.packet_payload_copy_job_count_total, 13);
    assert_eq!(stats.max_packet_payload_copy_jobs_used_per_tile, 6);
    assert_eq!(stats.packet_payload_copy_bytes_total, 23);
    assert_eq!(stats.max_packet_payload_copy_bytes_per_tile, 12);
    assert_eq!(stats.packet_payload_copy_small_job_count_total, 2);
    assert_eq!(stats.packet_payload_copy_medium_job_count_total, 3);
    assert_eq!(stats.packet_payload_copy_large_job_count_total, 4);
    assert_eq!(stats.tier1_output_capacity_total, 17);
    assert_eq!(stats.max_tier1_output_capacity, 9);
    assert_eq!(stats.tier1_output_used_bytes_total, 19);
    assert_eq!(stats.max_tier1_output_used_bytes, 10);
    assert_eq!(stats.tier1_segment_capacity_total, 25);
    assert_eq!(stats.max_tier1_segment_capacity_per_block, 11);
    assert_eq!(stats.tier1_coding_pass_count_total, 31);
    assert_eq!(stats.max_tier1_coding_passes_per_block, 8);
    assert_eq!(stats.tier1_arithmetic_pass_count_total, 21);
    assert_eq!(stats.tier1_raw_pass_count_total, 10);
    assert_eq!(stats.tier1_cleanup_pass_count_total, 11);
    assert_eq!(stats.tier1_sigprop_pass_count_total, 10);
    assert_eq!(stats.tier1_magref_pass_count_total, 10);
    assert_eq!(stats.tier1_arithmetic_cleanup_pass_count_total, 11);
    assert_eq!(stats.tier1_arithmetic_sigprop_pass_count_total, 6);
    assert_eq!(stats.tier1_arithmetic_magref_pass_count_total, 4);
    assert_eq!(stats.tier1_raw_sigprop_pass_count_total, 4);
    assert_eq!(stats.tier1_raw_magref_pass_count_total, 6);
    assert_eq!(stats.tier1_full_scan_coeff_visit_count_total, 31_744);
    assert_eq!(stats.tier1_arithmetic_scan_coeff_visit_count_total, 21_504);
    assert_eq!(stats.tier1_raw_scan_coeff_visit_count_total, 10_240);
    assert_eq!(stats.tier1_cleanup_scan_coeff_visit_count_total, 11_264);
    assert_eq!(stats.tier1_sigprop_scan_coeff_visit_count_total, 10_240);
    assert_eq!(stats.tier1_magref_scan_coeff_visit_count_total, 10_240);
    assert_eq!(stats.max_tier1_full_scan_coeff_visits_per_block, 8_192);
    assert_eq!(stats.tier1_sigprop_active_candidate_count_total, 101);
    assert_eq!(stats.tier1_sigprop_new_significant_count_total, 37);
    assert_eq!(stats.tier1_magref_active_candidate_count_total, 203);
    assert_eq!(
        stats.tier1_arithmetic_sigprop_active_candidate_count_total,
        61
    );
    assert_eq!(
        stats.tier1_arithmetic_sigprop_new_significant_count_total,
        23
    );
    assert_eq!(stats.tier1_raw_sigprop_active_candidate_count_total, 40);
    assert_eq!(stats.tier1_raw_sigprop_new_significant_count_total, 14);
    assert_eq!(
        stats.tier1_arithmetic_magref_active_candidate_count_total,
        123
    );
    assert_eq!(stats.tier1_raw_magref_active_candidate_count_total, 80);
    assert_eq!(stats.tier1_cleanup_active_candidate_count_total, 307);
    assert_eq!(stats.tier1_cleanup_new_significant_count_total, 41);
    assert_eq!(stats.tier1_cleanup_rlc_stripe_count_total, 53);
    assert_eq!(stats.tier1_cleanup_rlc_zero_stripe_count_total, 47);
    assert_eq!(stats.tier1_token_pack_output_bytes_total, 29);
    assert_eq!(stats.max_tier1_token_pack_output_bytes_per_block, 15);
    assert_eq!(stats.tier1_nonzero_block_count_total, 2);
    assert_eq!(stats.tier1_zero_block_count_total, 1);
    assert_eq!(stats.tier1_missing_bitplane_count_total, 5);
    assert_eq!(stats.max_tier1_missing_bitplanes_per_block, 4);
    assert_eq!(stats.tier1_segment_count_total, 7);
    assert_eq!(stats.max_tier1_segments_per_block, 3);
    assert_eq!(stats.packet_output_capacity_total, 17);
    assert_eq!(stats.max_packet_output_capacity, 9);
    assert_eq!(stats.packet_output_used_bytes_total, 19);
    assert_eq!(stats.max_packet_output_used_bytes, 10);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "exhaustive stage aggregation assertion guards every timing field"
)]
fn resident_lossless_stage_stats_accumulates_split_gpu_durations() {
    let mut stats = super::super::MetalLosslessEncodeStageStats {
        ht_block_gpu_duration: Duration::from_micros(2),
        ..super::super::MetalLosslessEncodeStageStats::default()
    };

    stats.add_assign(super::super::MetalLosslessEncodeStageStats {
        coefficient_prep_gpu_duration: Duration::from_micros(23),
        coefficient_deinterleave_rct_gpu_duration: Duration::from_micros(4),
        coefficient_dwt53_gpu_duration: Duration::from_micros(6),
        coefficient_dwt53_vertical_gpu_duration: Duration::from_micros(2),
        coefficient_dwt53_horizontal_gpu_duration: Duration::from_micros(4),
        coefficient_extract_gpu_duration: Duration::from_micros(8),
        coefficient_copy_gpu_duration: Duration::from_micros(1),
        gpu_elapsed_wall_duration: Duration::from_micros(29),
        classic_block_gpu_duration: Duration::from_micros(19),
        classic_tier1_density_gpu_duration: Duration::from_micros(31),
        classic_tier1_raw_pack_gpu_duration: Duration::from_micros(37),
        classic_tier1_arithmetic_pack_gpu_duration: Duration::from_micros(39),
        classic_tier1_symbol_plan_gpu_duration: Duration::from_micros(41),
        classic_tier1_token_emit_gpu_duration: Duration::from_micros(43),
        classic_tier1_split_token_emit_gpu_duration: Duration::from_micros(45),
        classic_tier1_token_pack_gpu_duration: Duration::from_micros(47),
        ht_block_gpu_duration: Duration::from_micros(3),
        packet_block_prep_gpu_duration: Duration::from_micros(5),
        packetization_gpu_duration: Duration::from_micros(7),
        packet_payload_copy_gpu_duration: Duration::from_micros(11),
        codestream_assembly_gpu_duration: Duration::from_micros(13),
        codestream_payload_copy_gpu_duration: Duration::from_micros(17),
        ..super::super::MetalLosslessEncodeStageStats::default()
    });

    assert_eq!(
        stats.coefficient_prep_gpu_duration,
        Duration::from_micros(23)
    );
    assert_eq!(
        stats.coefficient_deinterleave_rct_gpu_duration,
        Duration::from_micros(4)
    );
    assert_eq!(
        stats.coefficient_dwt53_gpu_duration,
        Duration::from_micros(6)
    );
    assert_eq!(
        stats.coefficient_dwt53_vertical_gpu_duration,
        Duration::from_micros(2)
    );
    assert_eq!(
        stats.coefficient_dwt53_horizontal_gpu_duration,
        Duration::from_micros(4)
    );
    assert_eq!(
        stats.coefficient_extract_gpu_duration,
        Duration::from_micros(8)
    );
    assert_eq!(
        stats.coefficient_copy_gpu_duration,
        Duration::from_micros(1)
    );
    assert_eq!(stats.gpu_elapsed_wall_duration, Duration::from_micros(29));
    assert_eq!(stats.classic_block_gpu_duration, Duration::from_micros(19));
    assert_eq!(
        stats.classic_tier1_density_gpu_duration,
        Duration::from_micros(31)
    );
    assert_eq!(
        stats.classic_tier1_raw_pack_gpu_duration,
        Duration::from_micros(37)
    );
    assert_eq!(
        stats.classic_tier1_arithmetic_pack_gpu_duration,
        Duration::from_micros(39)
    );
    assert_eq!(
        stats.classic_tier1_symbol_plan_gpu_duration,
        Duration::from_micros(41)
    );
    assert_eq!(
        stats.classic_tier1_token_emit_gpu_duration,
        Duration::from_micros(43)
    );
    assert_eq!(
        stats.classic_tier1_split_token_emit_gpu_duration,
        Duration::from_micros(45)
    );
    assert_eq!(
        stats.classic_tier1_token_pack_gpu_duration,
        Duration::from_micros(47)
    );
    assert_eq!(stats.ht_block_gpu_duration, Duration::from_micros(5));
    assert_eq!(
        stats.packet_block_prep_gpu_duration,
        Duration::from_micros(5)
    );
    assert_eq!(stats.packetization_gpu_duration, Duration::from_micros(7));
    assert_eq!(
        stats.packet_payload_copy_gpu_duration,
        Duration::from_micros(11)
    );
    assert_eq!(
        stats.codestream_assembly_gpu_duration,
        Duration::from_micros(13)
    );
    assert_eq!(
        stats.codestream_payload_copy_gpu_duration,
        Duration::from_micros(17)
    );
    assert!(stats.has_timings());
}

#[test]
fn resident_lossless_prep_duration_only_records_when_profiled() {
    let mut stats = super::super::MetalLosslessEncodeBatchStats::default();

    super::super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), false);
    assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
    assert!(!stats.stage_stats.has_timings());

    super::super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), true);
    assert_eq!(
        stats.stage_stats.coefficient_prep_duration,
        Duration::from_micros(7)
    );
    assert!(stats.stage_stats.has_timings());
}

#[test]
fn resident_lossless_prep_duration_uses_wall_time_not_per_tile_sum() {
    let mut stats = super::super::MetalLosslessEncodeBatchStats::default();
    let wall_duration = Duration::from_micros(11);
    let per_tile_sum = Duration::from_micros(9).saturating_add(Duration::from_micros(10));
    assert_ne!(wall_duration, per_tile_sum);

    super::super::add_resident_prep_wall_duration(&mut stats, wall_duration, true);

    assert_eq!(stats.stage_stats.coefficient_prep_duration, wall_duration);
}

#[cfg(target_os = "macos")]
#[test]
fn resident_classic_peak_estimate_matches_tight_batch_capacity() {
    let plan = super::super::LosslessDeviceEncodePlan {
        components: 1,
        bit_depth: 8,
        block_coding_mode: J2kBlockCodingMode::Classic,
        num_decomposition_levels: 0,
        use_mct: false,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        code_blocks: vec![compute::J2kLosslessDeviceCodeBlock {
            coefficient_offset: 0,
            component: 0,
            subband_x: 0,
            subband_y: 0,
            block_x: 0,
            block_y: 0,
            width: 64,
            height: 64,
            sub_band_type: j2k_native::J2kSubBandType::LowLow,
            total_bitplanes: 11,
        }],
        resolutions: Vec::new(),
        progression_order: j2k_native::EncodeProgressionOrder::Lrcp,
        write_tlm: false,
    };

    assert_eq!(
        super::super::estimated_tier1_output_bytes(&plan),
        64 * 64 * 11 + 4097
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resident_classic_batch_retry_covers_tight_capacity_failures() {
    let tight_tier1_error = crate::Error::MetalKernelRetryable {
        message: "packetization Metal encode kernel failure (detail=7, tier1_detail=4)".to_string(),
        retry_class: crate::MetalKernelRetryClass::ResidentClassicBatch,
    };
    assert!(
        super::super::resident_classic_batch_encode_should_retry_conservative(&tight_tier1_error)
    );

    let tight_tier1_finish_error = crate::Error::MetalKernelRetryable {
        message: "classic Tier-1 Metal encode kernel failure (detail=5)".to_string(),
        retry_class: crate::MetalKernelRetryClass::ResidentClassicBatch,
    };
    assert!(
        super::super::resident_classic_batch_encode_should_retry_conservative(
            &tight_tier1_finish_error
        )
    );

    let packet_error = crate::Error::MetalKernelRetryable {
        message: "packetization Metal encode kernel failure (detail=5)".to_string(),
        retry_class: crate::MetalKernelRetryClass::ResidentClassicOrHtBatch,
    };
    assert!(super::super::resident_classic_batch_encode_should_retry_conservative(&packet_error));
    assert!(super::super::resident_ht_batch_encode_should_retry_conservative(&packet_error));

    let codestream_error = crate::Error::MetalKernelRetryable {
        message: "J2K batched codestream assembly Metal encode kernel failure (detail=2)"
            .to_string(),
        retry_class: crate::MetalKernelRetryClass::ResidentClassicBatch,
    };
    assert!(
        super::super::resident_classic_batch_encode_should_retry_conservative(&codestream_error)
    );

    let unrelated_error = crate::Error::MetalKernel {
        message: "packetization Metal encode kernel failure (detail=8)".to_string(),
    };
    assert!(
        !super::super::resident_classic_batch_encode_should_retry_conservative(&unrelated_error)
    );
}

#[test]
fn resident_lossless_ht_command_duration_matches_split_buckets() {
    let stats = super::super::MetalLosslessEncodeStageStats {
        ht_command_encode_duration: Duration::from_micros(2)
            .saturating_add(Duration::from_micros(3))
            .saturating_add(Duration::from_micros(5))
            .saturating_add(Duration::from_micros(7)),
        ht_block_encode_duration: Duration::from_micros(2),
        packet_block_prep_duration: Duration::from_micros(3),
        packetization_duration: Duration::from_micros(5),
        codestream_assembly_duration: Duration::from_micros(7),
        ..super::super::MetalLosslessEncodeStageStats::default()
    };

    assert_eq!(
        stats.ht_command_encode_duration,
        stats
            .ht_block_encode_duration
            .saturating_add(stats.packet_block_prep_duration)
            .saturating_add(stats.packetization_duration)
            .saturating_add(stats.codestream_assembly_duration)
    );
}

#[test]
fn lossless_encode_outcome_exposes_host_readback_duration() {
    let outcome = super::super::MetalLosslessEncodeOutcome {
        encoded: EncodedJ2k {
            codestream: Vec::new(),
            backend: j2k_core::BackendKind::Metal,
            dispatch_report: J2kEncodeDispatchReport::default(),
            width: 0,
            height: 0,
            components: 1,
            bit_depth: 8,
            signed: false,
        },
        input_copy_used: false,
        resident: super::super::MetalLosslessEncodeResidency {
            coefficient_prep_used: false,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration: Duration::ZERO,
        gpu_duration: None,
        validation_duration: Duration::ZERO,
        host_readback_duration: Duration::from_micros(3),
    };

    assert_eq!(outcome.host_readback_duration, Duration::from_micros(3));
}

#[test]
fn resident_lossless_chunk_ranges_respect_inflight_and_code_block_caps() {
    assert_eq!(
        super::super::resident_lossless_chunk_ranges_for_test(&[32, 32, 32, 32, 32], 3, 96),
        vec![0..3, 3..5]
    );
    assert_eq!(
        super::super::resident_lossless_chunk_ranges_for_test(&[80, 80, 10], 8, 96),
        vec![0..1, 1..3]
    );
}

#[test]
fn resident_lossless_default_code_block_cap_allows_large_wsi_chunks() {
    let code_blocks = vec![192usize; 600];
    let cap = super::super::resident_lossless_code_block_chunk_cap(&code_blocks);

    assert_eq!(
        super::super::resident_lossless_chunk_ranges_for_test(&code_blocks, 512, cap),
        vec![0..512, 512..600]
    );
}
