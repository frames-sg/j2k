// SPDX-License-Identifier: Apache-2.0

use super::MetalEncodeStageAccelerator;
#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use j2k::adapter::encode_stage::J2kForwardDwt53Job;
use j2k::adapter::encode_stage::{
    J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kForwardRctJob,
};
use j2k::{
    encode_j2k_lossless_with_accelerator, EncodeBackendPreference, EncodedJ2k,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(target_os = "macos")]
use j2k::{
    encode_j2k_lossy_with_accelerator, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLossyEncodeOptions, J2kLossySamples, J2kProgressionOrder,
};
use j2k_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_native::{deinterleave_reference, forward_dwt53_reference, J2kCodeBlockStyle};
use j2k_native::{DecodeSettings, Image};
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::Buffer;
use std::time::Duration;

#[cfg(target_os = "macos")]
macro_rules! lossless_options {
    ($($field:ident: $value:expr),+ $(,)?) => {{
        let mut options = J2kLosslessEncodeOptions::default();
        $(options.$field = $value;)+
        options
    }};
}

#[cfg(target_os = "macos")]
fn private_buffer_with_bytes(session: &crate::MetalBackendSession, bytes: &[u8]) -> Buffer {
    let upload = session.device().new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let private = session.device().new_buffer(
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModePrivate,
    );
    let queue = session.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(&upload, 0, &private, 0, bytes.len() as u64);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    private
}

#[cfg(target_os = "macos")]
fn overwrite_private_buffer_with_bytes(
    session: &crate::MetalBackendSession,
    dst: &Buffer,
    bytes: &[u8],
) {
    let upload = session.device().new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let queue = session.device().new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(&upload, 0, dst, 0, bytes.len() as u64);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

#[cfg(target_os = "macos")]
fn assert_decoded_bytes_match(actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let mismatch = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let actual_value = actual.get(mismatch).copied();
    let expected_value = expected.get(mismatch).copied();
    panic!(
        "decoded bytes mismatch at byte {mismatch}: actual={actual_value:?} expected={expected_value:?} actual_len={} expected_len={}",
        actual.len(),
        expected.len()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_encode_deinterleave_rgb8_matches_native_reference() {
    let pixels = [0, 64, 255, 128, 129, 130, 251, 7, 19, 33, 211, 91];
    let job = J2kDeinterleaveToF32Job {
        pixels: &pixels,
        num_pixels: 4,
        num_components: 3,
        bit_depth: 8,
        signed: false,
    };
    let expected = deinterleave_reference(
        job.pixels,
        job.num_pixels,
        job.num_components,
        job.bit_depth,
        job.signed,
    );
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_deinterleave(job)
        .expect("Metal deinterleave stage")
        .expect("RGB8 unsigned deinterleave should dispatch");

    assert_eq!(actual, expected);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(accelerator.dispatch_report().deinterleave, 1);
}

#[test]
fn metal_encode_deinterleave_unsupported_16_bit_returns_cpu_fallback() {
    let pixels = [
        0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
    ];
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_deinterleave(J2kDeinterleaveToF32Job {
            pixels: &pixels,
            num_pixels: 2,
            num_components: 3,
            bit_depth: 16,
            signed: false,
        })
        .expect("unsupported deinterleave should fallback cleanly");

    assert!(actual.is_none());
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 0);
    assert_eq!(accelerator.dispatch_report().deinterleave, 0);
}

#[test]
fn metal_encode_deinterleave_unsupported_16_bit_fallback_still_encodes() {
    let mut pixels = Vec::with_capacity(8 * 8 * 2);
    for idx in 0..8 * 8 {
        let sample = u16::try_from((idx * 257 + 19) & 0xffff).expect("masked sample fits u16");
        pixels.extend_from_slice(&sample.to_le_bytes());
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 8, 8, 1, 16, false).expect("valid Gray16 samples");
    let options = J2kLosslessEncodeOptions::default().with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        j2k_core::BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Gray16 encode should succeed with CPU deinterleave fallback");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.dispatch_report.deinterleave, 0);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 0);
}

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

    let outcomes = super::collect_inflight_limited_ordered(vec![0usize, 1, 2], 2, move |_, item| {
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
            super::MetalLosslessEncodeBatchRequest<'tiles, 'tile>,
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
            Output = super::MetalLosslessBufferEncodeBatchOutcome,
            Error = crate::Error,
        >,
    >() {
    }
    fn assert_submit_buffer_batch_fn(
        _submit: for<'tiles, 'tile, 'options, 'session> fn(
            super::MetalLosslessEncodeBatchRequest<'tiles, 'tile>,
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
fn resident_lossless_stage_stats_default_to_zero() {
    let stats = super::MetalLosslessEncodeBatchStats::default();

    assert_eq!(
        stats.stage_stats,
        super::MetalLosslessEncodeStageStats::default()
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
fn resident_lossless_stage_stats_add_assign_saturates() {
    let mut stats = super::MetalLosslessEncodeStageStats {
        plan_duration: Duration::MAX,
        tile_count: usize::MAX,
        ..super::MetalLosslessEncodeStageStats::default()
    };

    stats.add_assign(super::MetalLosslessEncodeStageStats {
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
        ..super::MetalLosslessEncodeStageStats::default()
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
fn resident_lossless_stage_stats_accumulates_split_gpu_durations() {
    let mut stats = super::MetalLosslessEncodeStageStats {
        ht_block_gpu_duration: Duration::from_micros(2),
        ..super::MetalLosslessEncodeStageStats::default()
    };

    stats.add_assign(super::MetalLosslessEncodeStageStats {
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
        ..super::MetalLosslessEncodeStageStats::default()
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
    let mut stats = super::MetalLosslessEncodeBatchStats::default();

    super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), false);
    assert_eq!(stats.stage_stats.coefficient_prep_duration, Duration::ZERO);
    assert!(!stats.stage_stats.has_timings());

    super::add_resident_prep_duration(&mut stats, Duration::from_micros(7), true);
    assert_eq!(
        stats.stage_stats.coefficient_prep_duration,
        Duration::from_micros(7)
    );
    assert!(stats.stage_stats.has_timings());
}

#[test]
fn resident_lossless_prep_duration_uses_wall_time_not_per_tile_sum() {
    let mut stats = super::MetalLosslessEncodeBatchStats::default();
    let wall_duration = Duration::from_micros(11);
    let per_tile_sum = Duration::from_micros(9).saturating_add(Duration::from_micros(10));
    assert_ne!(wall_duration, per_tile_sum);

    super::add_resident_prep_wall_duration(&mut stats, wall_duration, true);

    assert_eq!(stats.stage_stats.coefficient_prep_duration, wall_duration);
}

#[cfg(target_os = "macos")]
#[test]
fn resident_classic_peak_estimate_matches_tight_batch_capacity() {
    let plan = super::LosslessDeviceEncodePlan {
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
        super::estimated_tier1_output_bytes(&plan),
        64 * 64 * 11 + 4097
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resident_classic_batch_retry_covers_tight_capacity_failures() {
    let tight_tier1_error = crate::Error::MetalKernel {
        message: "packetization Metal encode kernel failure (detail=7, tier1_detail=4)".to_string(),
    };
    assert!(super::resident_classic_batch_encode_should_retry_conservative(&tight_tier1_error));

    let tight_tier1_finish_error = crate::Error::MetalKernel {
        message: "classic Tier-1 Metal encode kernel failure (detail=5)".to_string(),
    };
    assert!(
        super::resident_classic_batch_encode_should_retry_conservative(&tight_tier1_finish_error)
    );

    let packet_error = crate::Error::MetalKernel {
        message: "packetization Metal encode kernel failure (detail=5)".to_string(),
    };
    assert!(super::resident_classic_batch_encode_should_retry_conservative(&packet_error));

    let codestream_error = crate::Error::MetalKernel {
        message: "J2K batched codestream assembly Metal encode kernel failure (detail=2)"
            .to_string(),
    };
    assert!(super::resident_classic_batch_encode_should_retry_conservative(&codestream_error));

    let unrelated_error = crate::Error::MetalKernel {
        message: "packetization Metal encode kernel failure (detail=8)".to_string(),
    };
    assert!(!super::resident_classic_batch_encode_should_retry_conservative(&unrelated_error));
}

#[test]
fn resident_lossless_ht_command_duration_matches_split_buckets() {
    let stats = super::MetalLosslessEncodeStageStats {
        ht_command_encode_duration: Duration::from_micros(2)
            .saturating_add(Duration::from_micros(3))
            .saturating_add(Duration::from_micros(5))
            .saturating_add(Duration::from_micros(7)),
        ht_block_encode_duration: Duration::from_micros(2),
        packet_block_prep_duration: Duration::from_micros(3),
        packetization_duration: Duration::from_micros(5),
        codestream_assembly_duration: Duration::from_micros(7),
        ..super::MetalLosslessEncodeStageStats::default()
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
    let outcome = super::MetalLosslessEncodeOutcome {
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
        resident: super::MetalLosslessEncodeResidency {
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
        super::resident_lossless_chunk_ranges_for_test(&[32, 32, 32, 32, 32], 3, 96),
        vec![0..3, 3..5]
    );
    assert_eq!(
        super::resident_lossless_chunk_ranges_for_test(&[80, 80, 10], 8, 96),
        vec![0..1, 1..3]
    );
}

#[test]
fn resident_lossless_default_code_block_cap_allows_large_wsi_chunks() {
    let code_blocks = vec![192usize; 600];
    let cap = super::resident_lossless_code_block_chunk_cap(&code_blocks);

    assert_eq!(
        super::resident_lossless_chunk_ranges_for_test(&code_blocks, 512, cap),
        vec![0..512, 512..600]
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_dispatch_option_treats_unavailable_as_no_dispatch() {
    let result: Result<Option<u8>, &'static str> =
        super::metal_dispatch_option(Err(crate::Error::MetalUnavailable), "kernel failed");

    assert_eq!(result, Ok(None));
}

#[cfg(target_os = "macos")]
#[test]
fn metal_dispatch_option_preserves_kernel_errors() {
    let result: Result<Option<u8>, &'static str> = super::metal_dispatch_option(
        Err(crate::Error::MetalKernel {
            message: "bad status".to_string(),
        }),
        "kernel failed",
    );

    assert_eq!(result, Err("kernel failed"));
}

#[test]
fn metal_encode_stage_accelerator_preserves_cpu_codestream_validity() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 3, 8, false).expect("valid RGB samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::Auto)
        .with_max_decomposition_levels(Some(1));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        j2k_core::BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal stage accelerator");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_attempts(), 3);
    assert!(accelerator.tier1_code_block_attempts() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
}

#[test]
fn metal_encode_stage_accelerator_can_leave_forward_rct_on_cpu() {
    let mut plane0 = vec![0.0, 64.0, 128.0, 255.0];
    let mut plane1 = vec![3.0, 67.0, 131.0, 252.0];
    let mut plane2 = vec![7.0, 71.0, 135.0, 248.0];
    let original = (plane0.clone(), plane1.clone(), plane2.clone());
    let mut accelerator = MetalEncodeStageAccelerator::with_cpu_forward_rct();

    let dispatched = accelerator
        .encode_forward_rct(J2kForwardRctJob {
            plane0: &mut plane0,
            plane1: &mut plane1,
            plane2: &mut plane2,
        })
        .expect("CPU RCT fallback should be selectable");

    assert!(!dispatched);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!((plane0, plane1, plane2), original);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_rct_dispatch_round_trips_rgb8_lossless_tile() {
    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 7, 5, 3, 8, false).expect("valid RGB samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal forward RCT");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(accelerator.forward_rct_attempts(), 1);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_validation_decodes_and_compares_lossless_codestream_on_device() {
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| ((i * 29) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 16, 16, 3, 8, false).unwrap();
    let encoded = j2k::encode_j2k_lossless(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::CpuOnly,
        },
    )
    .expect("lossless encode");

    super::validate_lossless_roundtrip_on_metal(samples, &encoded.codestream)
        .expect("Metal lossless validation");
}

#[cfg(target_os = "macos")]
#[test]
fn metal_buffer_lossless_encode_pads_edge_tile_on_device() {
    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
    let device = metal::Device::system_default().expect("Metal device");
    let session = crate::MetalBackendSession::new(device);
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = super::encode_lossless_from_metal_buffer(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal buffer lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn submitted_metal_buffer_lossless_encode_wait_round_trips() {
    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 19) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let submitted = super::submit_lossless_from_metal_buffer(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("submit Metal buffer lossless encode");
    let encoded = submitted.wait().expect("wait Metal buffer lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_buffer_lossless_encode_accepts_padded_contiguous_input_without_copy() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = session.device().new_buffer_with_data(
        pixels.as_ptr().cast(),
        pixels.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert_eq!(encoded.input_copy_duration, std::time::Duration::ZERO);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_encode_uses_resident_coefficient_prep() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_host_output_encode_options_preserve_auto_for_hybrid_path() {
    let routed = super::host_output_encode_options(lossless_options! {
        backend: EncodeBackendPreference::Auto,
        validation: J2kEncodeValidation::CpuRoundTrip,
    });

    assert_eq!(routed.backend, EncodeBackendPreference::Auto);
    assert_eq!(routed.validation, J2kEncodeValidation::External);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_host_output_accelerator_uses_metal_dwt_with_cpu_block_fallback() {
    let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 17) & 0xff) as u8).collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("hybrid host-output encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    assert_eq!(accelerator.tier1_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert!(accelerator.prefer_parallel_cpu_code_block_fallback());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_small_host_output_stays_cpu_below_resident_gate() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).expect("valid RGB samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("hybrid HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_large_host_output_uses_resident_metal_rct_dwt_and_ht_with_cpu_packetization() {
    let width = 1024u32;
    let height = 1024u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("hybrid HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert!(accelerator.forward_rct_dispatches() > 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_padded_rgb8_uses_fused_metal_rct_with_cpu_packetization() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            pixels.push(((x * 19 + y * 3) & 0xff) as u8);
            pixels.push(((x * 5 + y * 23) & 0xff) as u8);
            pixels.push(((x * 11 + y * 13) & 0xff) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);
    compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 64,
            height: 64,
            pitch_bytes: 64 * 3,
            output_width: 64,
            output_height: 64,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto HTJ2K resident hybrid encode");
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
    assert!(
        compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
        "Auto HTJ2K resident hybrid should use fused deinterleave + RCT"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_auto_host_encode_routes_away_from_resident_prep() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 43) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto host-output encode should avoid resident prep and still succeed");

    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(!encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_encode_to_metal_buffer_exposes_finished_bytes() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 37) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded buffer lossless encode to Metal buffer");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    if let Some(duration) = encoded.gpu_duration {
        assert!(duration > Duration::ZERO);
    }
    assert_eq!(encoded.encoded.byte_offset, 0);
    assert!(encoded.encoded.byte_len > 0);
    assert!(encoded.encoded.capacity >= encoded.encoded.byte_len);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.starts_with(&[0xFF, 0x4F]));
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_edge_private_rgb8_encode_to_metal_buffer_pads_and_stays_resident() {
    let pixels: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 41) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_metal_buffer_to_metal_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private edge buffer lossless encode to Metal buffer");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.starts_with(&[0xFF, 0x4F]));
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
    for y in 0..8usize {
        for x in 0..8usize {
            let dst = (y * 8 + x) * 3;
            if x < 7 && y < 5 {
                let src = (y * 7 + x) * 3;
                assert_eq!(&decoded.data[dst..dst + 3], &pixels[src..src + 3]);
            } else {
                assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn submitted_private_padded_rgb8_encode_snapshots_before_wait() {
    let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let replacement = vec![0u8; pixels.len()];
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let submitted = super::submit_lossless_from_padded_metal_buffer(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("submit Metal private padded RGB8 encode");
    overwrite_private_buffer_with_bytes(&session, &buffer, &replacement);

    let encoded = submitted.wait().expect("wait submitted encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_dwt_encode_uses_resident_coefficient_prep() {
    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded DWT buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_dwt_encode_uses_resident_coefficient_prep() {
    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded RGB8 DWT buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_dwt_resident_codestream_decodes_natively() {
    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal private padded DWT buffer lossless encode");

    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_dwt_resident_codestream_decodes_natively() {
    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal private padded RGB8 DWT buffer lossless encode");

    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray8_rpcl_encode_uses_resident_coefficient_prep() {
    let mut pixels = Vec::with_capacity(128 * 128);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 5 + y * 9 + (x ^ y)) & 0xFF) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            progression: J2kProgressionOrder::Rpcl,
        },
        &session,
    )
    .expect("Metal private padded RPCL buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_gray16_encode_uses_resident_coefficient_prep() {
    let mut pixels = Vec::with_capacity(8 * 8 * 2);
    for idx in 0..64u16 {
        let value = idx.wrapping_mul(997).wrapping_add(123);
        pixels.extend_from_slice(&value.to_le_bytes());
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 2,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray16,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal private padded Gray16 buffer lossless encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Metal);
    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_ht_encode_to_metal_buffer_stays_resident() {
    let pixels: Vec<u8> = (0..8 * 8).map(|i| ((i * 31) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        &session,
    )
    .expect("Metal private padded HTJ2K buffer lossless encode");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
    let cod_marker = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_marker + 12], 0x40);
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_ht_rpcl_512_encode_preserves_three_dwt_levels_and_stays_resident() {
    let pixels: Vec<u8> = (0..512 * 512 * 3)
        .map(|idx| ((idx * 47 + idx / 17) & 0xFF) as u8)
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = private_buffer_with_bytes(&session, &pixels);

    let encoded = super::encode_lossless_from_padded_metal_buffer_to_metal_with_report(
        super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 512,
            height: 512,
            pitch_bytes: 512 * 3,
            output_width: 512,
            output_height: 512,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            progression: J2kProgressionOrder::Rpcl,
        },
        &session,
    )
    .expect("Metal private padded HTJ2K RPCL 512 buffer lossless encode");

    assert!(!encoded.input_copy_used);
    assert!(encoded.resident.coefficient_prep_used);
    assert!(encoded.resident.packetization_used);
    assert!(encoded.resident.codestream_assembly_used);
    let codestream = encoded
        .encoded
        .codestream_bytes()
        .expect("Metal codestream bytes are CPU-readable");
    let cod_marker = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(codestream[cod_marker + 5], 0x02);
    assert_eq!(codestream[cod_marker + 9], 3);
    assert_eq!(codestream[cod_marker + 12], 0x40);
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_rgb8_ht_batch_uses_fused_deinterleave_rct_kernel() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    let first: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|idx| ((idx * 29 + idx / 7) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..WIDTH * HEIGHT * 3)
        .map(|idx| 255u8.wrapping_sub(((idx * 13 + idx / 5) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH * 3,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Rgb8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH * 3,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Rgb8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::reset_lossless_deinterleave_rct_fused_dispatches_for_test();
    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles, &options, &session,
    )
    .expect("Metal RGB8 HTJ2K batch encode");

    assert_eq!(encoded.len(), 2);
    assert!(
        compute::lossless_deinterleave_rct_fused_dispatches_for_test() > 0,
        "RGB8 resident lossless encode should fuse deinterleave and RCT"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_buffer_lossless_batch_encodes_padded_contiguous_inputs() {
    let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 7) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| ((i * 13 + 5) & 0xFF) as u8)
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = session.device().new_buffer_with_data(
        first.as_ptr().cast(),
        first.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let second_buffer = session.device().new_buffer_with_data(
        second.as_ptr().cast(),
        second.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::encode_lossless_from_padded_metal_buffers_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer batch lossless encode");

    assert_eq!(encoded.len(), 2);
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert_eq!(frame.encoded.backend, BackendKind::Metal);
        assert!(!frame.input_copy_used);
        let decoded = Image::new(&frame.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_batch_encode_to_metal_buffers_exposes_per_frame_bytes() {
    let first: Vec<u8> = (0..8 * 8 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 23) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal padded buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    assert_eq!(
        encoded[0].encoded.codestream_buffer.as_ptr(),
        encoded[1].encoded.codestream_buffer.as_ptr(),
        "classic J2K resident batch encode should assemble codestreams into one shared batch buffer"
    );
    assert_eq!(encoded[0].encoded.byte_offset, 0);
    assert!(
        encoded[1].encoded.byte_offset > 0,
        "second classic J2K batch codestream should be a nonzero slice into the shared batch buffer"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_batch_dwt_encode_to_metal_buffers_round_trips() {
    let first: Vec<u8> = (0..128 * 128 * 3)
        .map(|i| ((i * 17 + i / 3) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..128 * 128 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 23 + i / 5) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 128,
            height: 128,
            pitch_bytes: 128 * 3,
            output_width: 128,
            output_height: 128,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Metal padded DWT buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_decoded_bytes_match(&decoded.data, &expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_edge_private_batch_encode_to_metal_buffers_stays_resident() {
    let first: Vec<u8> = (0..7 * 5 * 3).map(|i| ((i * 17) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..6 * 8 * 3)
        .map(|i| 255u8.wrapping_sub(((i * 19) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    compute::reset_ht_batch_coefficient_copy_blits_for_test();
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 7,
            height: 5,
            pitch_bytes: 7 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 6,
            height: 8,
            pitch_bytes: 6 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
    ];

    let encoded = super::encode_lossless_from_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
        },
        &session,
    )
    .expect("Metal edge buffer batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    for frame in &encoded {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
    }

    for (frame, (expected, width, height)) in encoded
        .iter()
        .zip([(first, 7usize, 5usize), (second, 6usize, 8usize)])
    {
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        for y in 0..8usize {
            for x in 0..8usize {
                let dst = (y * 8 + x) * 3;
                if x < width && y < height {
                    let src = (y * width + x) * 3;
                    assert_eq!(&decoded.data[dst..dst + 3], &expected[src..src + 3]);
                } else {
                    assert_eq!(&decoded.data[dst..dst + 3], &[0, 0, 0]);
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_ht_private_batch_encode_to_metal_buffers_stays_resident() {
    let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 11) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8)
        .map(|i| 255u8.wrapping_sub(((i * 13) & 0xFF) as u8))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
    ];

    compute::reset_resident_gpu_timestamp_queries_for_test();
    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        &session,
    )
    .expect("Metal HTJ2K batch lossless encode to Metal buffers");

    assert_eq!(encoded.len(), 2);
    assert_eq!(
        compute::ht_batch_coefficient_copy_blits_for_test(),
        0,
        "HTJ2K resident batch prep should write directly into the batch coefficient buffer"
    );
    assert_eq!(
        compute::resident_gpu_timestamp_queries_for_test(),
        7,
        "HTJ2K resident batch should query each unique retained command buffer timestamp once"
    );
    assert_eq!(
        encoded[0].encoded.codestream_buffer.as_ptr(),
        encoded[1].encoded.codestream_buffer.as_ptr(),
        "HTJ2K resident batch encode should assemble codestreams into one shared batch buffer"
    );
    assert_eq!(encoded[0].encoded.byte_offset, 0);
    assert!(
        encoded[1].encoded.byte_offset > 0,
        "second HTJ2K batch codestream should be a nonzero slice into the shared batch buffer"
    );
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let codestream = frame
            .encoded
            .codestream_bytes()
            .expect("Metal codestream bytes are CPU-readable");
        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_ht_private_batch_encode_reuses_private_arenas_between_batches() {
    const WIDTH: usize = 37;
    const HEIGHT: usize = 41;
    let first: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| ((i * 7 + 3) & 0xFF) as u8)
        .collect();
    let second: Vec<u8> = (0..WIDTH * HEIGHT)
        .map(|i| 255u8.wrapping_sub(((i * 5 + 11) & 0xFF) as u8))
        .collect();
    let device = metal::Device::system_default().expect("Metal device");
    let session = crate::MetalBackendSession::new(device.clone());
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Gray8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: WIDTH as u32,
            height: HEIGHT as u32,
            pitch_bytes: WIDTH,
            output_width: WIDTH as u32,
            output_height: HEIGHT as u32,
            format: PixelFormat::Gray8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::with_isolated_runtime_for_device_for_test(&device, || {
        compute::reset_private_buffer_pool_misses_for_test();
        super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;
        let first_misses = compute::private_buffer_pool_misses_for_test();
        assert!(
            first_misses > 0,
            "first unique HTJ2K batch should populate reusable private arenas"
        );

        compute::reset_private_buffer_pool_misses_for_test();
        let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_with_report(
            &tiles, &options, &session,
        )?;

        assert_eq!(
            compute::private_buffer_pool_misses_for_test(),
            0,
            "second same-shape HTJ2K batch should reuse private arenas"
        );
        assert_eq!(encoded.len(), 2);
        Ok(())
    })
    .expect("isolated HTJ2K Metal runtime");
}

#[test]
fn default_gpu_encode_memory_budget_uses_forty_percent_capped_at_ten_gib() {
    const GIB: usize = 1024 * 1024 * 1024;

    assert_eq!(
        super::default_gpu_encode_memory_budget_bytes_for_hw_mem(8 * GIB),
        8 * GIB * 40 / 100
    );
    assert_eq!(
        super::default_gpu_encode_memory_budget_bytes_for_hw_mem(16 * GIB),
        16 * GIB * 40 / 100
    );
    assert_eq!(
        super::default_gpu_encode_memory_budget_bytes_for_hw_mem(24 * GIB),
        24 * GIB * 40 / 100
    );
    assert_eq!(
        super::default_gpu_encode_memory_budget_bytes_for_hw_mem(64 * GIB),
        10 * GIB
    );
}

#[test]
fn gpu_encode_inflight_resolution_clamps_requested_tiles_by_memory_budget() {
    let stats = super::resolve_lossless_encode_config_for_test(
        100,
        1_000,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(32),
            gpu_encode_memory_budget_bytes: Some(4_500),
        },
    )
    .expect("resolved config");

    assert_eq!(stats.configured_inflight_tiles, Some(32));
    assert_eq!(stats.effective_inflight_tiles, 4);
    assert_eq!(stats.configured_memory_budget_bytes, Some(4_500));
    assert_eq!(stats.effective_memory_budget_bytes, 4_500);
    assert_eq!(stats.estimated_peak_bytes_per_tile, 1_000);
}

#[test]
fn gpu_encode_default_inflight_uses_large_wsi_batch_when_memory_allows() {
    let stats = super::resolve_lossless_encode_config_for_test(
        600,
        1_000,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
    )
    .expect("resolved config");

    assert_eq!(stats.configured_inflight_tiles, None);
    assert_eq!(stats.effective_inflight_tiles, 512);
}

#[test]
fn resident_classic_encode_default_inflight_uses_profiled_cap() {
    let config = super::resident_lossless_encode_config_for_mode(
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        16,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(16));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_classic_encode_default_inflight_uses_large_batch_cap() {
    let config = super::resident_lossless_encode_config_for_mode(
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        64,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_classic_encode_default_inflight_uses_very_large_batch_cap() {
    let config = super::resident_lossless_encode_config_for_mode(
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        true,
        128,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(128));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_htj2k_encode_medium_batch_default_inflight_uses_profiled_cap() {
    let config = super::resident_lossless_encode_config_for_mode(
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        false,
        64,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(32));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn resident_htj2k_encode_large_batch_default_inflight_uses_profiled_cap() {
    let config = super::resident_lossless_encode_config_for_mode(
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: None,
            gpu_encode_memory_budget_bytes: Some(1_000_000),
        },
        false,
        128,
    );

    assert_eq!(config.gpu_encode_inflight_tiles, Some(64));
    assert_eq!(config.gpu_encode_memory_budget_bytes, Some(1_000_000));
}

#[test]
fn gpu_encode_inflight_resolution_rejects_zero_overrides() {
    let err = super::resolve_lossless_encode_config_for_test(
        4,
        1_000,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(0),
            gpu_encode_memory_budget_bytes: Some(4_000),
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("in-flight"),
        "unexpected error: {err}"
    );

    let err = super::resolve_lossless_encode_config_for_test(
        4,
        1_000,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(0),
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("memory budget"),
        "unexpected error: {err}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_ht_batch_encode_preserves_order_and_matches_inflight_one() {
    let inputs = [
        (0..8 * 8)
            .map(|i| ((i * 11 + 3) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 13 + 5) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 17 + 7) & 0xFF) as u8)
            .collect::<Vec<_>>(),
        (0..8 * 8)
            .map(|i| ((i * 19 + 9) & 0xFF) as u8)
            .collect::<Vec<_>>(),
    ];
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffers = inputs
        .iter()
        .map(|bytes| private_buffer_with_bytes(&session, bytes))
        .collect::<Vec<_>>();
    let tiles = buffers
        .iter()
        .map(|buffer| super::MetalLosslessEncodeTile {
            buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        })
        .collect::<Vec<_>>();
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    compute::reset_resident_codestream_command_buffer_waits_for_test();
    let serial = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(1),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("serial Metal HTJ2K batch");
    assert_eq!(
        compute::resident_codestream_command_buffer_waits_for_test(),
        1,
        "multi-chunk HT batch should wait once before harvesting completed chunks"
    );

    let cpu_validated_options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::CpuRoundTrip,
    };
    compute::reset_resident_codestream_command_buffer_waits_for_test();
    let cpu_validated = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &cpu_validated_options,
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(1),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("CPU-validated Metal HTJ2K batch");
    assert_eq!(cpu_validated.outcomes.len(), inputs.len());
    assert_eq!(
        compute::resident_codestream_command_buffer_waits_for_test(),
        inputs.len(),
        "CPU roundtrip validation should keep per-chunk waits to preserve overlap"
    );

    let parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("parallel Metal HTJ2K batch");
    let repeated_parallel = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("repeated parallel Metal HTJ2K batch");

    assert_eq!(serial.outcomes.len(), inputs.len());
    assert_eq!(parallel.outcomes.len(), inputs.len());
    assert_eq!(parallel.stats.effective_inflight_tiles, 2);
    assert!(parallel.stats.max_observed_inflight_tiles <= 2);
    assert!(parallel.stats.max_observed_inflight_tiles > 0);
    for (((serial_outcome, parallel_outcome), repeated_outcome), expected) in serial
        .outcomes
        .iter()
        .zip(parallel.outcomes.iter())
        .zip(repeated_parallel.outcomes.iter())
        .zip(inputs.iter())
    {
        let serial_bytes = serial_outcome
            .encoded
            .codestream_bytes()
            .expect("serial codestream");
        let parallel_bytes = parallel_outcome
            .encoded
            .codestream_bytes()
            .expect("parallel codestream");
        let repeated_bytes = repeated_outcome
            .encoded
            .codestream_bytes()
            .expect("repeated parallel codestream");
        assert_eq!(parallel_bytes, serial_bytes);
        assert_eq!(repeated_bytes, serial_bytes);

        let decoded = Image::new(parallel_bytes, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(&decoded.data, expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_parallel_batch_returns_indexed_injected_failure() {
    let first: Vec<u8> = (0..8 * 8).map(|i| ((i * 3) & 0xFF) as u8).collect();
    let second: Vec<u8> = (0..8 * 8).map(|i| ((i * 5) & 0xFF) as u8).collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Gray8,
        },
    ];
    let options = lossless_options! {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };

    super::set_test_resident_encode_failure_index(Some(1));
    let Err(err) = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &options,
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    ) else {
        panic!("injected failure should fail the batch");
    };
    super::set_test_resident_encode_failure_index(None);

    assert!(
        err.to_string().contains("tile 1"),
        "unexpected error: {err}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_dispatch_round_trips_gray8_lossless_tile() {
    let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 5) & 0xFF) as u8).collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(1));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal forward DWT 5/3");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.forward_dwt53_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_lossless_facade_dispatches_rct_and_dwt_for_wsi_sized_rgb_tile() {
    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid RGB samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
    assert!(accelerator.tier1_code_block_attempts() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert!(accelerator.tier1_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_uses_one_batched_dispatch_for_multiple_code_blocks() {
    let pixels: Vec<u8> = (0..256 * 256)
        .map(|idx| ((idx * 17 + 3) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with batched Metal classic Tier-1");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert!(accelerator.tier1_code_block_attempts() > 1);
    assert_eq!(accelerator.tier1_code_block_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_resident_uses_mq_byte_split_gpu_token_pack_by_default() {
    let _profile_guard = compute::force_metal_profile_stages_for_test(true);
    compute::reset_classic_gpu_token_pack_dispatches_for_test();
    compute::reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test();
    let first: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x + y * 5) & 0xFF) as u8
        })
        .collect();
    let second: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x * 3 + y * 7) & 0xFF) as u8
        })
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
    ];

    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::Classic,
            validation: J2kEncodeValidation::External,
        },
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("resident batch encode with default MQ-byte GPU token-pack Classic Tier-1");
    assert_eq!(encoded.outcomes.len(), 2);
    for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
        let codestream = outcome
            .encoded
            .to_encoded_j2k()
            .expect("codestream readback");
        let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(codestream.backend, BackendKind::Metal);
        assert_eq!(&decoded.data, expected);
    }
    assert!(
        compute::classic_gpu_token_pack_dispatches_for_test() > 0,
        "default Classic GPU token-pack route was not dispatched"
    );
    assert!(
        compute::classic_split_mq_byte_gpu_token_pack_dispatches_for_test() > 0,
        "default Classic GPU token-pack route did not use MQ-byte split token emit"
    );
    assert_eq!(
        encoded
            .stats
            .stage_stats
            .tier1_token_pack_output_bytes_total,
        encoded.stats.stage_stats.tier1_output_used_bytes_total,
        "default Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_resident_gpu_token_pack_route_round_trips() {
    let _guard = compute::force_classic_gpu_token_pack_route_for_test(true);
    let _profile_guard = compute::force_metal_profile_stages_for_test(true);
    compute::reset_classic_gpu_token_pack_dispatches_for_test();
    let first: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x + y * 3) & 0xFF) as u8
        })
        .collect();
    let second: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x * 2 + y) & 0xFF) as u8
        })
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = private_buffer_with_bytes(&session, &first);
    let second_buffer = private_buffer_with_bytes(&session, &second);
    let tiles = [
        super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
        super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
    ];

    let encoded = super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::Classic,
            validation: J2kEncodeValidation::External,
        },
        &session,
        super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("resident batch encode with gated GPU token-pack Classic Tier-1");
    assert_eq!(encoded.outcomes.len(), 2);
    for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
        let codestream = outcome
            .encoded
            .to_encoded_j2k()
            .expect("codestream readback");
        let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(codestream.backend, BackendKind::Metal);
        assert_eq!(&decoded.data, expected);
    }
    assert!(
        compute::classic_gpu_token_pack_dispatches_for_test() > 0,
        "gated Classic GPU token-pack route was not dispatched"
    );
    assert!(
        encoded.stats.stage_stats.tier1_token_emit_token_bytes_total > 0,
        "gated Classic GPU token-pack route did not expose token-emitter byte counters"
    );
    assert!(
        encoded
            .stats
            .stage_stats
            .tier1_token_emit_segment_count_total
            > 0,
        "gated Classic GPU token-pack route did not expose token segment counters"
    );
    assert_eq!(
        encoded
            .stats
            .stage_stats
            .tier1_token_pack_output_bytes_total,
        encoded.stats.stage_stats.tier1_output_used_bytes_total,
        "gated Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_uses_one_batched_dispatch_for_multiple_code_blocks() {
    let pixels: Vec<u8> = (0..256 * 256)
        .map(|idx| ((idx * 23 + 9) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert!(accelerator.ht_code_block_attempts() > 1);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossless_facade_dispatches_ht_code_blocks_and_packetization() {
    let pixels: Vec<u8> = (0..64).map(|value| ((value * 13) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert!(accelerator.ht_code_block_attempts() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossy_facade_require_device_dispatches_supported_stages() {
    let pixels: Vec<u8> = (0..16 * 16)
        .map(|idx| ((idx * 17 + idx / 3) & 0xFF) as u8)
        .collect();
    let samples = J2kLossySamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossy encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert!(accelerator.ht_code_block_attempts() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_oracle() {
    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 37 + 11) & 0x1ff) - 255;
            if idx % 5 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::HighHigh,
        total_bitplanes: 9,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::HighHigh,
        9,
        style,
    )
    .expect("scalar classic encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_for_terminated_passes() {
    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 43 + 5) & 0x3ff) - 511;
            if idx % 6 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: true,
        termination_on_each_pass: true,
        vertically_causal_context: false,
        segmentation_symbols: true,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::LowHigh,
        total_bitplanes: 10,
        style,
    };

    let gpu =
        compute::encode_classic_tier1_code_block(job).expect("Metal classic terminated encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::LowHigh,
        10,
        style,
    )
    .expect("scalar classic terminated encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_for_selective_bypass() {
    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 61 + 29) & 0x7ff) - 1023;
            if idx % 4 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::HighLow,
        total_bitplanes: 11,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic bypass encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::HighLow,
        11,
        style,
    )
    .expect("scalar classic bypass encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_batched_bypass_u16_32_matches_scalar() {
    let coeffs: Vec<i32> = (0..32 * 32)
        .map(|idx| {
            let value = ((idx * 97 + idx / 3 + 19) & 0x7ff) - 1023;
            if idx % 11 == 0 || idx % 17 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 32,
        height: 32,
        sub_band_type: j2k_native::J2kSubBandType::HighHigh,
        total_bitplanes: 11,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_blocks(&[job])
        .expect("batched Metal classic bypass_u16_32 encode")
        .pop()
        .expect("one encoded codeblock");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        32,
        32,
        j2k_native::J2kSubBandType::HighHigh,
        11,
        style,
    )
    .expect("scalar classic bypass encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_token_routes_match_scalar_bytes() {
    let first_coeffs: Vec<i32> = (0..32 * 32)
        .map(|idx| {
            let value = ((idx * 37 + idx / 5 + 31) & 0xff) - 127;
            if idx % 5 == 0 || idx % 11 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let second_coeffs: Vec<i32> = (0..17 * 29)
        .map(|idx| {
            let value = ((idx * 73 + idx / 7 + 11) & 0xff) - 127;
            if idx % 7 == 0 || idx % 23 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let jobs = [
        j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &first_coeffs,
            width: 32,
            height: 32,
            sub_band_type: j2k_native::J2kSubBandType::HighHigh,
            total_bitplanes: 8,
            style,
        },
        j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &second_coeffs,
            width: 17,
            height: 29,
            sub_band_type: j2k_native::J2kSubBandType::LowLow,
            total_bitplanes: 8,
            style,
        },
    ];

    let gpu_packed = compute::encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(&jobs)
        .expect("Metal classic GPU token-pack encode");
    let cpu_packed =
        compute::encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(&jobs)
            .expect("Metal classic ordered-token CPU-pack encode");
    let split_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(&jobs)
            .expect("Metal classic split MQ/raw token CPU-pack encode");
    let split_gpu_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(&jobs)
            .expect("Metal classic split MQ/raw token GPU-pack encode");
    let mq_byte_split_gpu_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
            &jobs,
        )
        .expect("Metal classic split MQ-byte/raw-bit token GPU-pack encode");

    assert_eq!(gpu_packed.len(), jobs.len());
    assert_eq!(cpu_packed.len(), jobs.len());
    assert_eq!(split_packed.len(), jobs.len());
    assert_eq!(split_gpu_packed.len(), jobs.len());
    assert_eq!(mq_byte_split_gpu_packed.len(), jobs.len());
    for (
        (
            (((gpu_block, cpu_packed_block), split_packed_block), split_gpu_packed_block),
            mq_byte_split_gpu_packed_block,
        ),
        job,
        coeffs,
    ) in gpu_packed
        .iter()
        .zip(cpu_packed.iter())
        .zip(split_packed.iter())
        .zip(split_gpu_packed.iter())
        .zip(mq_byte_split_gpu_packed.iter())
        .zip(jobs.iter())
        .zip([&first_coeffs, &second_coeffs])
        .map(|((blocks, job), coeffs)| (blocks, job, coeffs))
    {
        let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
            coeffs,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            style,
        )
        .expect("scalar classic bypass encode");

        assert_eq!(gpu_block.data, cpu.data);
        assert_eq!(gpu_block.segments, cpu.segments);
        assert_eq!(
            gpu_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(gpu_block.missing_bit_planes, cpu.missing_bit_planes);
        assert_eq!(cpu_packed_block.data, cpu.data);
        assert_eq!(cpu_packed_block.segments, cpu.segments);
        assert_eq!(
            cpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(cpu_packed_block.missing_bit_planes, cpu.missing_bit_planes);
        assert_eq!(split_packed_block.data, cpu.data);
        assert_eq!(split_packed_block.segments, cpu.segments);
        assert_eq!(
            split_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            split_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
        assert_eq!(split_gpu_packed_block.data, cpu.data);
        assert_eq!(split_gpu_packed_block.segments, cpu.segments);
        assert_eq!(
            split_gpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            split_gpu_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
        assert_eq!(mq_byte_split_gpu_packed_block.data, cpu.data);
        assert_eq!(mq_byte_split_gpu_packed_block.segments, cpu.segments);
        assert_eq!(
            mq_byte_split_gpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            mq_byte_split_gpu_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_cleanup_kernel_matches_scalar_oracle() {
    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 19 + 7) & 0xff) - 127;
            if idx % 7 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let job = j2k_native::J2kHtCodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        total_bitplanes: 8,
        target_coding_passes: 1,
    };

    let gpu = compute::encode_ht_cleanup_code_block(job).expect("Metal HT encode");
    let cpu = j2k_native::encode_ht_code_block_scalar(&coeffs, 8, 8, 8).expect("scalar HT encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.num_coding_passes, cpu.num_coding_passes);
    assert_eq!(gpu.num_zero_bitplanes, cpu.num_zero_bitplanes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_kernel_matches_scalar_oracle() {
    let block0 = [0x12, 0x34, 0x56, 0x78];
    let block1 = [0x9a, 0xbc];
    let code_blocks = vec![
        j2k_native::J2kPacketizationCodeBlock {
            data: &block0,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
        },
        j2k_native::J2kPacketizationCodeBlock {
            data: &block1,
            ht_cleanup_length: u32::try_from(block1.len()).expect("test payload fits u32"),
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 1,
            previously_included: false,
            l_block: 3,
            block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::HighThroughput,
        },
    ];
    let subband = j2k_native::J2kPacketizationSubband {
        code_blocks,
        num_cbs_x: 2,
        num_cbs_y: 1,
    };
    let resolution = j2k_native::J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let resolutions = [resolution];
    let packet_descriptors = [j2k_native::J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    }];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_reuses_descriptor_state_across_layers() {
    let block0 = vec![0x11];
    let block1 = vec![0x22];
    let first = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let resolutions = [first, second];
    let packet_descriptors = [
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 2,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Rpcl,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_honors_explicit_descriptor_order() {
    let block0 = vec![0xA0];
    let block1 = vec![0xB0];
    let first = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let resolutions = [first, second];
    let packet_descriptors = [
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 1,
            layer: 0,
            resolution: 1,
            component: 0,
            precinct: 0,
        },
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 1,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Rpcl,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
    for (width, height) in [(1, 8), (8, 1)] {
        let samples: Vec<f32> = (0..width * height)
            .map(|i| {
                f32::from(
                    u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                        .expect("masked sample fits in u8"),
                ) - 128.0
            })
            .collect();
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let output = accelerator
            .encode_forward_dwt53(J2kForwardDwt53Job {
                samples: &samples,
                width,
                height,
                num_levels: 1,
            })
            .expect("metal DWT 5/3 stage")
            .expect("metal DWT 5/3 dispatch");

        assert_eq!(output.ll_width, width.div_ceil(2));
        assert_eq!(output.ll_height, height.div_ceil(2));
        assert_eq!(output.levels.len(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_matches_reference_for_fractional_stage_samples() {
    fn assert_slice_near(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.0001,
                "{label}[{index}] mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    let width = 8;
    let height = 8;
    let samples = (0..width * height)
        .map(|idx| f32::from(u16::try_from(idx).expect("test index fits u16")) * 0.5 - 15.25)
        .collect::<Vec<_>>();
    let expected = forward_dwt53_reference(&samples, width, height, 1);
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_forward_dwt53(J2kForwardDwt53Job {
            samples: &samples,
            width,
            height,
            num_levels: 1,
        })
        .expect("metal DWT 5/3 stage")
        .expect("metal DWT 5/3 dispatch");

    assert_eq!(actual.ll_width, expected.ll_width);
    assert_eq!(actual.ll_height, expected.ll_height);
    assert_slice_near(&actual.ll, &expected.ll, "LL");
    assert_eq!(actual.levels.len(), expected.levels.len());
    for (index, (actual, expected)) in actual.levels.iter().zip(&expected.levels).enumerate() {
        assert_eq!(actual.width, expected.width, "level {index} width");
        assert_eq!(actual.height, expected.height, "level {index} height");
        assert_eq!(
            actual.low_width, expected.low_width,
            "level {index} low width"
        );
        assert_eq!(
            actual.low_height, expected.low_height,
            "level {index} low height"
        );
        assert_eq!(
            actual.high_width, expected.high_width,
            "level {index} high width"
        );
        assert_eq!(
            actual.high_height, expected.high_height,
            "level {index} high height"
        );
        assert_slice_near(&actual.hl, &expected.hl, "HL");
        assert_slice_near(&actual.lh, &expected.lh, "LH");
        assert_slice_near(&actual.hh, &expected.hh, "HH");
    }
}
