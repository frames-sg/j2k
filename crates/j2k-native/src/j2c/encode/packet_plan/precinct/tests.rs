// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    BlockCodingMode, NativeEncodeRetainedInput, PreparedCodeBlockCoefficients,
    PreparedEncodeCodeBlock, SubBandType,
};
use super::*;
use crate::EncodeError;

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small precinct ownership test allocation");
    values
}

fn prepared_precinct_fixture() -> Vec<Vec<PreparedResolutionPacket>> {
    let mut code_blocks = vector_with_capacity(8);
    let mut preencoded = vector_with_capacity(8);
    for marker in 0_u8..8 {
        let mut coefficients = vector_with_capacity(3);
        coefficients.push(i64::from(marker));
        code_blocks.push(PreparedEncodeCodeBlock {
            coefficients: PreparedCodeBlockCoefficients::I64(coefficients),
            width: 2,
            height: 2,
        });

        let mut data = vector_with_capacity(4);
        data.extend_from_slice(&[marker, marker.wrapping_add(1)]);
        preencoded.push(crate::EncodedHtJ2kCodeBlock {
            data,
            cleanup_length: 2,
            refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 0,
        });
    }

    let mut subbands = vector_with_capacity(2);
    subbands.push(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: Some(preencoded),
        num_cbs_x: 4,
        num_cbs_y: 2,
        code_block_width: 2,
        code_block_height: 2,
        width: 8,
        height: 4,
        sub_band_type: SubBandType::LowLow,
        total_bitplanes: 8,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    });
    let mut packets = vector_with_capacity(2);
    packets.push(PreparedResolutionPacket {
        component: 3,
        resolution: 0,
        precinct: 0,
        subbands,
    });
    let mut components = vector_with_capacity(2);
    components.push(packets);
    components
}

#[test]
fn precinct_split_moves_tier1_payloads_without_clone() {
    let source = prepared_precinct_fixture();
    let coefficient_ptrs: Vec<_> = source[0][0].subbands[0]
        .code_blocks
        .iter()
        .map(|block| match &block.coefficients {
            PreparedCodeBlockCoefficients::I64(values) => values.as_ptr() as usize,
            _ => panic!("expected i64 precinct fixture"),
        })
        .collect();
    let preencoded_ptrs: Vec<_> = source[0][0].subbands[0]
        .preencoded_ht_code_blocks
        .as_ref()
        .expect("preencoded fixture")
        .iter()
        .map(|block| block.data.as_ptr() as usize)
        .collect();
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("precinct split session");

    let split = split_component_resolution_packets_by_precinct_for_session(
        source,
        8,
        4,
        0,
        &[(2, 1)],
        &session,
        0,
    )
    .expect("move-only precinct split");

    assert_eq!(split.len(), 1);
    assert_eq!(split[0].len(), 4);
    for (precinct, packet) in split[0].iter().enumerate() {
        assert_eq!(packet.component, 3);
        assert_eq!(packet.resolution, 0);
        assert_eq!(
            packet.precinct,
            u64::try_from(precinct).expect("small precinct index")
        );
        assert_eq!(packet.subbands.len(), 1);
        assert_eq!(packet.subbands[0].num_cbs_x, 2);
        assert_eq!(packet.subbands[0].num_cbs_y, 1);
        assert_eq!(packet.subbands[0].width, 4);
        assert_eq!(packet.subbands[0].height, 2);
    }

    let moved_coefficients: Vec<_> = split[0]
        .iter()
        .flat_map(|packet| &packet.subbands[0].code_blocks)
        .map(|block| match &block.coefficients {
            PreparedCodeBlockCoefficients::I64(values) => values[0],
            _ => panic!("expected i64 precinct fixture"),
        })
        .collect();
    let moved_coefficient_ptrs: Vec<_> = split[0]
        .iter()
        .flat_map(|packet| &packet.subbands[0].code_blocks)
        .map(|block| match &block.coefficients {
            PreparedCodeBlockCoefficients::I64(values) => values.as_ptr() as usize,
            _ => panic!("expected i64 precinct fixture"),
        })
        .collect();
    let moved_preencoded: Vec<_> = split[0]
        .iter()
        .flat_map(|packet| {
            packet.subbands[0]
                .preencoded_ht_code_blocks
                .as_ref()
                .expect("split preencoded blocks")
        })
        .map(|block| block.data[0])
        .collect();
    let moved_preencoded_ptrs: Vec<_> = split[0]
        .iter()
        .flat_map(|packet| {
            packet.subbands[0]
                .preencoded_ht_code_blocks
                .as_ref()
                .expect("split preencoded blocks")
        })
        .map(|block| block.data.as_ptr() as usize)
        .collect();

    assert_eq!(moved_coefficients, (0_i64..8).collect::<Vec<_>>());
    assert_eq!(moved_preencoded, (0_u8..8).collect::<Vec<_>>());
    assert_eq!(moved_coefficient_ptrs, coefficient_ptrs);
    assert_eq!(moved_preencoded_ptrs, preencoded_ptrs);
}

#[test]
fn precinct_split_exact_peak_accepts_cap_and_rejects_one_byte_less() {
    const RETAINED_PHASE_BYTES: usize = 19;
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let (_measured, peak_phase_bytes) = split_component_resolution_packets_by_precinct_accounted(
        prepared_precinct_fixture(),
        8,
        4,
        0,
        &[(2, 1)],
        &measurement_session,
        RETAINED_PHASE_BYTES,
    )
    .expect("measure actual split peak");

    let exact_session =
        NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), peak_phase_bytes)
            .expect("exact split cap session");
    let exact = split_component_resolution_packets_by_precinct_for_session(
        prepared_precinct_fixture(),
        8,
        4,
        0,
        &[(2, 1)],
        &exact_session,
        RETAINED_PHASE_BYTES,
    )
    .expect("exact split peak is accepted");
    assert_eq!(exact[0].len(), 4);

    let cap = peak_phase_bytes - 1;
    let under_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("one-byte-under split session");
    let error = split_component_resolution_packets_by_precinct_for_session(
        prepared_precinct_fixture(),
        8,
        4,
        0,
        &[(2, 1)],
        &under_session,
        RETAINED_PHASE_BYTES,
    )
    .err()
    .expect("one-byte-under split peak must fail")
    .into_encode_error();
    match error {
        EncodeError::AllocationTooLarge {
            requested,
            cap: observed_cap,
            ..
        } => {
            assert_eq!(requested, peak_phase_bytes);
            assert_eq!(observed_cap, cap);
        }
        other => panic!("one-byte-under split must preserve the typed cap category: {other:?}"),
    }
}
