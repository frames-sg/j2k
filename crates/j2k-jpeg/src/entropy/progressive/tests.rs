// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation::{checked_phase_capacity, validate_coefficient_workspace};
use super::model::PreparedProgressiveComponentPlan;
use super::scan::{decode_eob_run, refine_band, refine_non_zeroes};
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

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Zigzag-order nonzero mask of a natural-order block's AC coefficients.
fn nonzero_mask(block: &[i32; 64]) -> u64 {
    (1..64)
        .filter(|&k| block[usize::from(ZIGZAG[k])] != 0)
        .fold(0, |mask, k| mask | 1 << k)
}

#[test]
fn mask_driven_refinement_matches_the_coefficient_walk() {
    let mut state = 0x2468_ace1;
    for case in 0..20_000 {
        let mut block = [0i32; 64];
        let density = xorshift(&mut state) % 5;
        for coefficient in block.iter_mut().skip(1) {
            if xorshift(&mut state) % 5 < density {
                let magnitude = i32::try_from(1 + xorshift(&mut state) % 40).expect("small");
                *coefficient = if xorshift(&mut state) & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
        }
        let start = u8::try_from(1 + xorshift(&mut state) % 63).expect("band start");
        let end = start + u8::try_from(xorshift(&mut state) % u32::from(64 - start)).expect("end");
        let zero_run_length = if case % 7 == 0 {
            64
        } else {
            usize::try_from(xorshift(&mut state) % 16).expect("run")
        };
        let bit = 1i32 << (xorshift(&mut state) % 3);
        let bytes: Vec<u8> = (0..16)
            .map(|_| u8::try_from(xorshift(&mut state) % 255).expect("byte"))
            .collect();

        let mut expected_block = block;
        let mut expected_reader = BitReader::new(&bytes);
        let expected = refine_non_zeroes(
            &mut expected_reader,
            &mut expected_block,
            start,
            end,
            zero_run_length,
            bit,
        );
        let mut actual_block = block;
        let mut actual_reader = BitReader::new(&bytes);
        let actual = refine_band(
            &mut actual_reader,
            &mut actual_block,
            nonzero_mask(&block),
            start,
            end,
            zero_run_length,
            bit,
        );
        let context = format!("case {case}: band {start}..={end}, run {zero_run_length}");
        assert_eq!(actual, expected, "{context}");
        assert_eq!(actual_block, expected_block, "{context}");
        assert_eq!(
            actual_reader.read_bits(8).ok(),
            expected_reader.read_bits(8).ok(),
            "{context}: bits consumed"
        );
    }
}
