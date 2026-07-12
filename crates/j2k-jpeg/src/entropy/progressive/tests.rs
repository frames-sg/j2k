// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation::{checked_phase_capacity, validate_coefficient_workspace};
use super::model::PreparedProgressiveComponentPlan;
use super::scan::{decode_eob_run, refine_non_zeroes};
use crate::allocation::checked_allocation_bytes;
use crate::entropy::ZIGZAG;
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;

#[test]
fn external_rows_reduce_the_remaining_progressive_phase_capacity() {
    let cap = 512;
    let internal = 400;

    assert_eq!(
        checked_phase_capacity(cap - internal, internal, cap).expect("exact phase boundary"),
        cap
    );
    assert!(matches!(
        checked_phase_capacity(cap - internal + 1, internal, cap),
        Err(JpegError::MemoryCapExceeded {
            requested: 513,
            cap: 512,
        })
    ));
}

#[test]
fn coefficient_workspace_rejects_aggregate_component_planes() {
    let cap = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    let blocks_per_component = cap / core::mem::size_of::<[i32; 64]>() * 3 / 5;
    let block_cols = u32::try_from(blocks_per_component).expect("test block count fits u32");
    let component = || PreparedProgressiveComponentPlan {
        h: 1,
        v: 1,
        output_index: 0,
        quant: [1; 64],
        block_cols,
        block_rows: 1,
        sample_width: block_cols.saturating_mul(8),
        sample_height: 8,
    };
    assert!(checked_allocation_bytes::<[i32; 64]>(blocks_per_component).is_ok());
    let components = [component(), component()];
    assert!(matches!(
        validate_coefficient_workspace(&components),
        Err(JpegError::MemoryCapExceeded { requested, cap: limit })
            if requested > limit && limit == cap
    ));
}

#[test]
fn decode_eob_run_combines_prefix_and_extra_bits() {
    let bytes = [0b1010_0000u8];
    let mut br = BitReader::new(&bytes);

    let run = decode_eob_run(&mut br, 3).unwrap();

    assert_eq!(run, 12);
}

#[test]
fn refine_non_zeroes_updates_existing_coefficients_by_sign() {
    let mut block = [0i32; 64];
    block[usize::from(ZIGZAG[1])] = 4;
    block[usize::from(ZIGZAG[2])] = -4;
    let bytes = [0b1100_0000u8];
    let mut br = BitReader::new(&bytes);

    refine_non_zeroes(&mut br, &mut block, 1, 2, 64, 2).unwrap();

    assert_eq!(block[usize::from(ZIGZAG[1])], 6);
    assert_eq!(block[usize::from(ZIGZAG[2])], -6);
}

#[test]
fn refine_non_zeroes_stops_at_requested_zero_run() {
    let mut block = [0i32; 64];
    block[usize::from(ZIGZAG[3])] = 8;
    let bytes = [0u8];
    let mut br = BitReader::new(&bytes);

    let index = refine_non_zeroes(&mut br, &mut block, 1, 4, 1, 2).unwrap();

    assert_eq!(index, 2);
}
