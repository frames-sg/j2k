// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    build_cuda_rgb8_plan_data, cuda_entropy_checkpoints_with_cap,
    decode_owned_cuda_rgb8_from_decoder, fast_rgb8_packet_parts, resolve_owned_rgb8_packet,
    CudaJpegEntropyCheckpoint, CudaJpegRgb8Sampling, CudaSession, Error,
};
use j2k_jpeg::adapter::{build_fast420_packet, JpegEntropyCheckpointV1, SharedJpegFastPacket};

mod support;
use self::support::generated_rgb_jpeg;

const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
const BASELINE_422: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_444: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_444_8x8.jpg");

#[test]
fn packet_plan_helper_preserves_sampling_entropy_and_checkpoints() {
    for (input, dimensions, expected_sampling) in [
        (BASELINE_420, (16, 16), CudaJpegRgb8Sampling::Fast420),
        (BASELINE_422, (16, 8), CudaJpegRgb8Sampling::Fast422),
        (BASELINE_444, (8, 8), CudaJpegRgb8Sampling::Fast444),
    ] {
        let mut session = CudaSession::default();
        let packet = resolve_owned_rgb8_packet(input, &mut session).expect("owned packet");
        let parts = fast_rgb8_packet_parts(&packet.packet);
        let expected_checkpoint = parts.entropy_checkpoints[0];
        let expected_entropy = parts.entropy_bytes;
        let plan_data =
            build_cuda_rgb8_plan_data(&parts, dimensions, &session).expect("CUDA plan data");
        let plan = plan_data.as_plan();

        assert_eq!(plan.sampling, expected_sampling);
        assert_eq!(plan.dimensions, dimensions);
        assert_eq!(plan.entropy_bytes, expected_entropy);
        assert_eq!(
            plan.entropy_checkpoints.len(),
            fast_rgb8_packet_parts(&packet.packet)
                .entropy_checkpoints
                .len()
        );
        let actual_checkpoint = plan.entropy_checkpoints[0];
        assert_eq!(actual_checkpoint.mcu_index, expected_checkpoint.mcu_index);
        assert_eq!(
            actual_checkpoint.entropy_pos,
            expected_checkpoint.entropy_pos
        );
        assert_eq!(actual_checkpoint.bit_acc, expected_checkpoint.bit_acc);
        assert_eq!(actual_checkpoint.bit_count, expected_checkpoint.bit_count);
        assert_eq!(actual_checkpoint.y_prev_dc, expected_checkpoint.y_prev_dc);
        assert_eq!(actual_checkpoint.cb_prev_dc, expected_checkpoint.cb_prev_dc);
        assert_eq!(actual_checkpoint.cr_prev_dc, expected_checkpoint.cr_prev_dc);
        assert_eq!(actual_checkpoint.reserved, expected_checkpoint.reserved);
    }
}

#[test]
fn multi_checkpoint_420_plans_are_ordered_and_start_from_a_clean_state() {
    let nonrestart = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, None);
    let restart = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, Some(1));

    for (name, input) in [
        ("nonrestart", nonrestart.as_slice()),
        ("restart_interval_1", restart.as_slice()),
    ] {
        let packet = build_fast420_packet(input)
            .unwrap_or_else(|error| panic!("build {name} packet: {error:?}"));
        let packet = SharedJpegFastPacket::try_new(packet.into()).expect("share test packet");
        let parts = fast_rgb8_packet_parts(&packet);
        let session = CudaSession::default();
        let plan_data = build_cuda_rgb8_plan_data(&parts, (32, 16), &session)
            .unwrap_or_else(|error| panic!("build {name} CUDA plan: {error}"));
        let plan = plan_data.as_plan();

        assert_eq!(plan.sampling, CudaJpegRgb8Sampling::Fast420, "{name}");
        assert_eq!(
            plan.entropy_checkpoints.len(),
            2,
            "{name} must exercise one checkpoint per MCU"
        );
        let first = plan.entropy_checkpoints[0];
        assert_eq!(first.mcu_index, 0, "{name}");
        assert_eq!(first.entropy_pos, 0, "{name}");
        assert_eq!(first.bit_acc, 0, "{name}");
        assert_eq!(first.bit_count, 0, "{name}");
        assert_eq!(first.y_prev_dc, 0, "{name}");
        assert_eq!(first.cb_prev_dc, 0, "{name}");
        assert_eq!(first.cr_prev_dc, 0, "{name}");
        assert_eq!(first.reserved, 0, "{name}");
        assert!(
            plan.entropy_checkpoints.windows(2).all(|pair| {
                pair[0].mcu_index < pair[1].mcu_index && pair[0].entropy_pos <= pair[1].entropy_pos
            }),
            "{name} checkpoints must be strictly MCU-ordered and entropy-monotonic"
        );
    }
}

#[test]
fn dri_equal_to_total_mcus_has_only_the_clean_initial_checkpoint() {
    let packet = build_fast420_packet(j2k_test_support::JPEG_BASELINE_420_RESTART_32X16)
        .expect("build the DRI=2, two-MCU restart fixture");

    assert_eq!(packet.restart_interval_mcus, 2);
    assert_eq!(packet.mcus_per_row * packet.mcu_rows, 2);
    assert_eq!(packet.entropy_checkpoints.len(), 1);
    assert_eq!(
        packet.entropy_checkpoints[0],
        JpegEntropyCheckpointV1 {
            mcu_index: 0,
            entropy_pos: 0,
            bit_acc: 0,
            bit_count: 0,
            y_prev_dc: 0,
            cb_prev_dc: 0,
            cr_prev_dc: 0,
            reserved: 0,
        }
    );
}

#[test]
fn packet_plan_helper_rejects_decoder_dimension_mismatch() {
    let mut session = CudaSession::default();
    let packet =
        resolve_owned_rgb8_packet(BASELINE_420, &mut session).expect("owned fast420 packet");
    let packet_parts = fast_rgb8_packet_parts(&packet.packet);
    let error = build_cuda_rgb8_plan_data(&packet_parts, (15, 16), &session)
        .expect_err("metadata mismatch must fail closed");

    assert!(matches!(
        error,
        Error::UnsupportedCudaRequest {
            reason: "J2K CUDA JPEG packet dimensions do not match decoder metadata"
        }
    ));
}

#[test]
fn unaddressable_output_is_rejected_before_packet_or_context_work() {
    const DIMENSIONS: (u32, u32) = (65_500, 65_500);
    let oversized = j2k_jpeg::rewrite_sof_dimensions(
        BASELINE_420,
        (
            u16::try_from(DIMENSIONS.0).expect("test width fits SOF"),
            u16::try_from(DIMENSIONS.1).expect("test height fits SOF"),
        ),
    )
    .expect("rewrite baseline SOF dimensions");
    let decoder = j2k_jpeg::Decoder::new(&oversized).expect("oversized metadata decoder");
    let mut session = CudaSession::default();

    let error = decode_owned_cuda_rgb8_from_decoder(&decoder, &mut session)
        .expect_err("u32-unaddressable output must fail before CUDA setup");

    assert!(matches!(
        error,
        Error::UnsupportedCudaRequest {
            reason:
                "J2K-owned CUDA JPEG decode requires RGB8 output addressable by u32 byte offsets"
        }
    ));
    assert_eq!(session.owned_cuda_packet_cache_len(), 0);
    assert!(!session.is_runtime_initialized());
}

#[test]
fn checkpoint_conversion_propagates_the_host_allocation_cap() {
    let checkpoint = JpegEntropyCheckpointV1 {
        mcu_index: 0,
        entropy_pos: 0,
        bit_acc: 0,
        bit_count: 0,
        y_prev_dc: 0,
        cb_prev_dc: 0,
        cr_prev_dc: 0,
        reserved: 0,
    };
    let checkpoint_bytes = core::mem::size_of::<CudaJpegEntropyCheckpoint>() * 2;
    let cache_host_byte_limit = 13;
    let total_active_packet_bytes = 17;
    let existing_decoder_bytes = 19;
    let exact = cache_host_byte_limit
        + total_active_packet_bytes
        + existing_decoder_bytes
        + checkpoint_bytes;
    let converted = cuda_entropy_checkpoints_with_cap(
        &[checkpoint, checkpoint],
        cache_host_byte_limit,
        total_active_packet_bytes,
        existing_decoder_bytes,
        exact,
    )
    .expect("cache + active packets + decoder + converted checkpoints exact cap");
    assert_eq!(converted.len(), 2);
    let error = cuda_entropy_checkpoints_with_cap(
        &[checkpoint, checkpoint],
        cache_host_byte_limit,
        total_active_packet_bytes,
        existing_decoder_bytes,
        exact - 1,
    )
    .expect_err("cache + active packets + decoder + checkpoints one byte over cap");

    assert!(matches!(
        error,
        Error::HostAllocationTooLarge {
            requested,
            cap,
            what: "CUDA JPEG entropy checkpoint conversion",
        } if requested == exact && cap == exact - 1
    ));
}
