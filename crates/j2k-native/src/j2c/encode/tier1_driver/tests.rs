// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{CpuOnlyJ2kEncodeStageAccelerator, EncodeError};

fn exact_vec<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small Tier-1 test allocation");
    values
}

fn classic_fixture() -> Vec<PreparedEncodeSubband> {
    let mut coefficients = exact_vec(16);
    coefficients.extend([4, -3, 2, 0, -1, 5, 0, 2, 1, -2, 3, 0, -4, 1, 2, -1]);
    let mut blocks = exact_vec(1);
    blocks.push(super::super::PreparedEncodeCodeBlock {
        coefficients: PreparedCodeBlockCoefficients::I32(coefficients),
        width: 4,
        height: 4,
    });
    let mut subbands = exact_vec(1);
    subbands.push(PreparedEncodeSubband {
        origin: (0, 0),
        code_blocks: blocks,
        preencoded_ht_code_blocks: None,
        preencoded_ht_maximum_cleanup_magnitude: None,
        num_cbs_x: 1,
        num_cbs_y: 1,
        code_block_width: 4,
        code_block_height: 4,
        width: 4,
        height: 4,
        sub_band_type: super::super::SubBandType::LowLow,
        total_bitplanes: 5,
        block_coding_mode: BlockCodingMode::Classic,
        ht_target_coding_passes: 1,
    });
    subbands
}

fn ht_refinement_fixture() -> Vec<PreparedEncodeSubband> {
    let mut coefficients = exact_vec(16);
    coefficients.extend([7, -6, 5, 0, -4, 3, 0, 2, 1, -2, 3, 0, -4, 1, 2, -1]);
    let mut blocks = exact_vec(1);
    blocks.push(super::super::PreparedEncodeCodeBlock {
        coefficients: PreparedCodeBlockCoefficients::I32(coefficients),
        width: 4,
        height: 4,
    });
    let mut subbands = exact_vec(1);
    subbands.push(PreparedEncodeSubband {
        origin: (0, 0),
        code_blocks: blocks,
        preencoded_ht_code_blocks: None,
        preencoded_ht_maximum_cleanup_magnitude: None,
        num_cbs_x: 1,
        num_cbs_y: 1,
        code_block_width: 4,
        code_block_height: 4,
        width: 4,
        height: 4,
        sub_band_type: super::super::SubBandType::LowLow,
        total_bitplanes: 5,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 3,
    });
    subbands
}

#[test]
fn mixed_tier1_modes_remain_unsupported() {
    let mut mixed = exact_vec(2);
    mixed.extend(classic_fixture());
    mixed.extend(ht_refinement_fixture());
    let session =
        NativeEncodeSession::try_new(NativeEncodeRetainedInput::none()).expect("Tier-1 session");

    let result = encode_prepared_subbands_for_session(
        mixed,
        &session,
        0,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    );
    let Err(error) = result else {
        panic!("mixed Tier-1 coding modes must be rejected");
    };
    let error = error.into_encode_error();

    assert!(matches!(error, EncodeError::Unsupported { .. }));
}

#[test]
fn tier1_phase_accepts_exact_peak_and_rejects_cap_minus_one() {
    const RETAINED_BASE_BYTES: usize = 17;
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured = encode_prepared_subbands_accounted(
        classic_fixture(),
        &measurement_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("measure Tier-1 peak");
    let peak = measured.peak_phase_bytes;
    assert!(peak > RETAINED_BASE_BYTES);

    let exact_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), peak)
        .expect("exact Tier-1 session");
    let exact = encode_prepared_subbands_for_session(
        classic_fixture(),
        &exact_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("exact Tier-1 peak is accepted");
    assert_eq!(exact.len(), 1);
    assert_eq!(exact[0].code_blocks.len(), 1);

    let cap = peak - 1;
    let under_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("cap-minus-one Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        classic_fixture(),
        &under_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("cap-minus-one Tier-1 peak must fail")
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
fn ht_refinement_frontier_accepts_exact_peak_and_rejects_cap_minus_one() {
    const RETAINED_BASE_BYTES: usize = 19;
    let measurement_session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured = encode_prepared_subbands_accounted(
        ht_refinement_fixture(),
        &measurement_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("measure HT Tier-1 peak");
    let peak = measured.peak_phase_bytes;

    let exact_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), peak)
        .expect("exact HT Tier-1 session");
    let exact = encode_prepared_subbands_for_session(
        ht_refinement_fixture(),
        &exact_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("exact HT Tier-1 frontier is accepted");
    let block = &exact[0].code_blocks[0];
    assert_eq!(block.num_coding_passes, 3);
    assert!(block.ht_cleanup_length > 0);
    assert!(block.ht_refinement_length > 0);

    let cap = peak - 1;
    let under_session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("cap-minus-one HT Tier-1 session");
    let error = encode_prepared_subbands_for_session(
        ht_refinement_fixture(),
        &under_session,
        RETAINED_BASE_BYTES,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect_err("cap-minus-one HT Tier-1 frontier must fail")
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

mod accelerator;
mod accelerator_metadata;
mod error_taxonomy;

fn mixed_ht_fixture() -> (Vec<PreparedEncodeSubband>, Vec<Vec<u8>>) {
    let mut subbands = Vec::new();
    let mut expected = Vec::new();
    for index in 0..4 {
        let mut subband = ht_refinement_fixture().pop().expect("one fixture subband");
        subband.ht_target_coding_passes = 1;
        subband.total_bitplanes = 7;
        let PreparedCodeBlockCoefficients::I32(coefficients) =
            &mut subband.code_blocks[0].coefficients
        else {
            panic!("i32 fixture")
        };
        for coefficient in coefficients.iter_mut() {
            *coefficient *= index + 1;
        }
        let encoded =
            crate::encode_ht_code_block_scalar(coefficients, 4, 4, 7).expect("scalar source block");
        expected.push(encoded.data.clone());
        if index % 2 == 0 {
            subband.preencoded_ht_code_blocks = Some(vec![encoded]);
            subband.code_blocks[0].coefficients = PreparedCodeBlockCoefficients::Empty;
        }
        subbands.push(subband);
    }
    (subbands, expected)
}

#[test]
fn mixed_ht_subbands_preserve_payloads_and_enforce_the_retained_peak() {
    let (subbands, expected) = mixed_ht_fixture();
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("measurement session");
    let measured = encode_prepared_subbands_accounted(
        subbands,
        &session,
        0,
        &mut CpuOnlyJ2kEncodeStageAccelerator,
    )
    .expect("encode quantized bands alongside preencoded bands");
    let peak = measured.peak_phase_bytes;
    for (precinct, bytes) in measured.precincts.iter().zip(&expected) {
        assert_eq!(precinct.code_blocks.len(), 1);
        assert_eq!(&precinct.code_blocks[0].data, bytes);
    }
    for cap in [peak, peak - 1] {
        let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
            .expect("bounded session");
        let result = encode_prepared_subbands_for_session(
            mixed_ht_fixture().0,
            &session,
            0,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        )
        .map_err(NativeEncodePipelineError::into_encode_error);
        if cap == peak {
            assert!(result.is_ok(), "exact retained peak must fit");
        } else {
            assert!(matches!(
                result,
                Err(EncodeError::AllocationTooLarge { .. })
            ));
        }
    }
}
