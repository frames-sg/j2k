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
        code_blocks: blocks,
        preencoded_ht_code_blocks: None,
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
        code_blocks: blocks,
        preencoded_ht_code_blocks: None,
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
