// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    bitplane_encode, classic_multilayer_code_block_style, BlockCodingMode, EncodeProgressionOrder,
    NativeEncodeRetainedInput, NativeEncodeSession, PreparedCodeBlockCoefficients,
    PreparedEncodeCodeBlock, PreparedEncodeSubband, PreparedResolutionPacket, SubBandType, Vec,
};
use super::{
    encode_prepared_resolution_packets_layered_accounted,
    encode_prepared_resolution_packets_layered_for_session,
};
use crate::{CpuOnlyJ2kEncodeStageAccelerator, EncodeError};

fn exact_vec<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small layered Tier-1 test allocation");
    values
}

fn classic_packet_fixture() -> Vec<PreparedResolutionPacket> {
    let mut coefficients = exact_vec(16);
    coefficients.extend([7, -3, 2, 0, -1, 5, 0, 2, 1, -2, 3, 0, -4, 1, 2, -1]);
    let mut blocks = exact_vec(1);
    blocks.push(PreparedEncodeCodeBlock {
        coefficients: PreparedCodeBlockCoefficients::I32(coefficients),
        width: 4,
        height: 4,
    });
    let mut subbands = exact_vec(1);
    subbands.push(PreparedEncodeSubband {
        code_blocks: blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x: 1,
        num_cbs_y: 1,
        code_block_width: 4,
        code_block_height: 4,
        width: 4,
        height: 4,
        sub_band_type: SubBandType::LowLow,
        total_bitplanes: 5,
        block_coding_mode: BlockCodingMode::Classic,
        ht_target_coding_passes: 1,
    });
    let mut packets = exact_vec(1);
    packets.push(PreparedResolutionPacket {
        component: 0,
        resolution: 0,
        precinct: 0,
        subbands,
    });
    packets
}

#[test]
fn zero_layer_and_target_count_errors_remain_invalid_input() {
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("layered session");
    let zero_layer = encode_prepared_resolution_packets_layered_for_session(
        Vec::new(),
        0,
        EncodeProgressionOrder::Lrcp,
        &[],
        &session,
        0,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("zero quality layers must be rejected")
    .into_encode_error();
    assert!(matches!(zero_layer, EncodeError::InvalidInput { .. }));

    let target_count = encode_prepared_resolution_packets_layered_for_session(
        classic_packet_fixture(),
        2,
        EncodeProgressionOrder::Lrcp,
        &[64],
        &session,
        0,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("quality-layer target count mismatch must be rejected")
    .into_encode_error();
    assert!(matches!(target_count, EncodeError::InvalidInput { .. }));
}

#[test]
fn layered_rate_control_accepts_exact_peak_and_rejects_cap_minus_one() {
    const RETAINED_BASE_BYTES: usize = 23;
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured = encode_prepared_resolution_packets_layered_accounted(
        classic_packet_fixture(),
        2,
        EncodeProgressionOrder::Lrcp,
        &[8, 4_096],
        &measurement_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("measure layered Tier-1 peak");
    let peak = measured.peak_phase_bytes;

    let exact_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), peak)
        .expect("exact layered session");
    let exact = encode_prepared_resolution_packets_layered_for_session(
        classic_packet_fixture(),
        2,
        EncodeProgressionOrder::Lrcp,
        &[8, 4_096],
        &exact_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("exact layered peak is accepted");
    assert_eq!(exact.0.len(), 2);
    assert_eq!(exact.1.len(), 2);

    let cap = peak - 1;
    let under_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("cap-minus-one layered session");
    let error = encode_prepared_resolution_packets_layered_for_session(
        classic_packet_fixture(),
        2,
        EncodeProgressionOrder::Lrcp,
        &[8, 4_096],
        &under_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("cap-minus-one layered peak must fail")
    .into_encode_error();
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            requested,
            cap: observed,
            ..
        } if requested == peak && observed == cap
    ));
}

#[test]
fn multilayer_contributions_preserve_classic_payload_bytes() {
    let coefficients = [7, -3, 2, 0, -1, 5, 0, 2, 1, -2, 3, 0, -4, 1, 2, -1];
    let reference = bitplane_encode::encode_code_block_segments_with_style(
        &coefficients,
        4,
        4,
        SubBandType::LowLow,
        5,
        classic_multilayer_code_block_style(),
    );
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("layered session");
    let (packets, descriptors) = encode_prepared_resolution_packets_layered_for_session(
        classic_packet_fixture(),
        2,
        EncodeProgressionOrder::Lrcp,
        &[],
        &session,
        0,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("two-layer classic encode");
    let mut combined = Vec::new();
    for packet in &packets {
        combined.extend_from_slice(&packet.subbands[0].code_blocks[0].data);
    }
    assert_eq!(combined, reference.data);
    assert_eq!(
        descriptors
            .iter()
            .map(|descriptor| descriptor.layer)
            .collect::<Vec<_>>(),
        [0, 1]
    );
}
