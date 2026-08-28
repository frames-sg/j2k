// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use crate::j2c::coefficient_view::CoefficientBlockView;

use super::cleanup::{
    convert_nonzero_to_aligned_sign_magnitude_and_max, encode_cleanup_segment,
    encode_cleanup_segment_from_coefficients,
};
use super::distribution::collect_encode_distribution;
use super::facade::{
    encode_code_block, encode_code_block_view, encode_code_block_with_passes,
    select_tile_code_block_candidates, try_encode_code_block_candidate_sets_with_workspace,
    try_encode_code_block_set_with_workspace, try_encode_code_block_with_passes_in_workspace,
    HtCandidateRange,
};
use super::workspace::HtEncodeWorkspace;

#[test]
fn reusable_workspace_is_byte_exact_across_shrinking_and_growing_blocks() {
    let large = (0_i32..64 * 64)
        .map(|index| match index % 5 {
            0 => 0,
            1 => index & 255,
            2 => -(index & 127),
            3 => 31 - (index & 63),
            _ => index & 7,
        })
        .collect::<Vec<_>>();
    let small = [0, 3, -2, 1];
    let mut workspace = HtEncodeWorkspace::try_new().expect("HT workspace allocation");

    for (coefficients, width, height) in [
        (large.as_slice(), 64, 64),
        (small.as_slice(), 2, 2),
        (large.as_slice(), 64, 64),
    ] {
        let expected = encode_code_block_with_passes(coefficients, width, height, 10, 1)
            .expect("fresh-workspace encode");
        let actual = try_encode_code_block_with_passes_in_workspace(
            coefficients,
            width,
            height,
            10,
            1,
            &mut workspace,
        )
        .expect("reused-workspace encode");
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.num_coding_passes, expected.num_coding_passes);
        assert_eq!(actual.num_zero_bitplanes, expected.num_zero_bitplanes);
        assert_eq!(actual.ht_cleanup_length, expected.ht_cleanup_length);
        assert_eq!(actual.ht_refinement_length, expected.ht_refinement_length);
    }
}

#[test]
fn ht_strided_block_is_byte_exact_for_cleanup_and_refinement_passes() {
    const WIDTH: usize = 7;
    const HEIGHT: usize = 5;
    const STRIDE: usize = 12;
    const OFFSET: usize = 14;
    let contiguous = (0_i32..i32::try_from(WIDTH * HEIGHT).expect("test size fits i32"))
        .map(|index| match index % 6 {
            0 => 0,
            1 => index * 5,
            2 => -(index * 3),
            3 => 31 - index,
            4 => -17 + index,
            _ => index / 2,
        })
        .collect::<Vec<_>>();
    let mut padded = vec![i32::MIN; OFFSET + STRIDE * HEIGHT + 7];
    for y in 0..HEIGHT {
        padded[OFFSET + y * STRIDE..OFFSET + y * STRIDE + WIDTH]
            .copy_from_slice(&contiguous[y * WIDTH..(y + 1) * WIDTH]);
    }
    let view = CoefficientBlockView::try_new(&padded, OFFSET, WIDTH, HEIGHT, STRIDE)
        .expect("valid strided HT block");

    for coding_passes in [1, 3] {
        let expected = encode_code_block_with_passes(
            &contiguous,
            u32::try_from(WIDTH).expect("test width fits u32"),
            u32::try_from(HEIGHT).expect("test height fits u32"),
            10,
            coding_passes,
        )
        .expect("contiguous HT encode");
        let actual = encode_code_block_view(view, 10, coding_passes).expect("strided HT encode");

        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.num_coding_passes, expected.num_coding_passes);
        assert_eq!(actual.num_zero_bitplanes, expected.num_zero_bitplanes);
        assert_eq!(actual.ht_cleanup_length, expected.ht_cleanup_length);
        assert_eq!(actual.ht_refinement_length, expected.ht_refinement_length);
    }
}

#[test]
fn test_convert_to_aligned_sign_magnitude() {
    let (aligned, _) = convert_nonzero_to_aligned_sign_magnitude_and_max(&[0, 1, -2, 3], 2)
        .expect("non-zero block");
    assert_eq!(aligned, vec![0, 0x2000_0000, 0xC000_0000, 0x6000_0000]);
}

#[test]
fn aligned_sign_magnitude_conversion_reports_max_and_skips_all_zero_blocks() {
    assert!(convert_nonzero_to_aligned_sign_magnitude_and_max(&[0, 0, 0], 5).is_none());

    let (aligned, max_magnitude) =
        convert_nonzero_to_aligned_sign_magnitude_and_max(&[0, 1, -2, 3], 2)
            .expect("non-zero block");
    assert_eq!(max_magnitude, 3);
    assert_eq!(aligned, vec![0, 0x2000_0000, 0xC000_0000, 0x6000_0000]);
}

#[test]
fn all_zero_distribution_input_remains_a_zero_distribution() {
    let distribution =
        collect_encode_distribution(&[0], 1, 1, 1).expect("all-zero distribution input is valid");

    assert_eq!(distribution.total_quads, 0);
    assert_eq!(distribution.mag_sign_calls, 0);
    assert_eq!(distribution.mag_sign_encoded_samples, 0);
}

#[test]
fn maximum_axis_code_blocks_encode_without_marker_row_overflow() {
    for (width, height) in [(1024_u32, 4_u32), (4, 1024)] {
        let mut coefficients = vec![0_i32; width as usize * height as usize];
        coefficients[0] = 3;
        let last = coefficients.len() - 1;
        coefficients[last] = -2;
        let encoded = encode_code_block_with_passes(&coefficients, width, height, 2, 3)
            .expect("maximum-axis HT block encodes");
        assert_eq!(encoded.num_coding_passes, 3);
        assert!(encoded.ht_cleanup_length > 0);
        assert!(encoded.data.len() <= encoded.data.capacity());
    }
}

#[test]
fn cleanup_segment_from_i32_coefficients_matches_preconverted_path() {
    let coefficients: Vec<i32> = (0..64)
        .map(|index| match index % 5 {
            0 => 0,
            1 => index * 3,
            2 => -(index * 2),
            3 => 7 - index,
            _ => index / 2,
        })
        .collect();
    let total_bitplanes = 10;
    let missing_msbs = total_bitplanes - 1;
    let (aligned, _) =
        convert_nonzero_to_aligned_sign_magnitude_and_max(&coefficients, total_bitplanes)
            .expect("non-zero block");

    let expected =
        encode_cleanup_segment(&aligned, missing_msbs, 8, 8).expect("preconverted encode");
    let actual = encode_cleanup_segment_from_coefficients(
        &coefficients,
        missing_msbs,
        8,
        8,
        total_bitplanes,
    )
    .expect("i32 encode");

    assert_eq!(actual, expected);
}

#[test]
fn cleanup_encode_distribution_counts_quads_and_mag_sign_payloads() {
    let coefficients: Vec<i32> = (0..8 * 6)
        .map(|index| {
            if index % 7 == 0 {
                0
            } else {
                let value = ((index * 29) & 0x1ff) - 255;
                if index % 3 == 0 {
                    -value
                } else {
                    value
                }
            }
        })
        .collect();

    let distribution =
        collect_encode_distribution(&coefficients, 8, 6, 10).expect("collect distribution");

    assert_eq!(distribution.total_quads, 12);
    assert_eq!(distribution.initial_quads, 4);
    assert_eq!(distribution.non_initial_quads, 8);
    assert_eq!(distribution.rho_counts.iter().sum::<u64>(), 12);
    assert_eq!(distribution.initial_rho_counts.iter().sum::<u64>(), 4);
    assert_eq!(distribution.non_initial_rho_counts.iter().sum::<u64>(), 8);
    assert_eq!(distribution.non_initial_u_q_counts.iter().sum::<u64>(), 8);
    assert!(distribution.mag_sign_calls > 0);
    assert!(distribution.mag_sign_encoded_samples > 0);
}

#[cfg(feature = "std")]
#[test]
#[ignore = "prints HT cleanup encode rho/e_q/u_q distribution for manual tuning"]
fn ht_cleanup_encode_distribution_report() {
    fn nonzero_histogram<const N: usize>(counts: &[u64; N]) -> Vec<(usize, u64)> {
        counts
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, count)| count != 0)
            .collect()
    }

    let coefficients: Vec<i32> = (0usize..64 * 64)
        .map(|index| {
            let value = i32::try_from(((index * 73) ^ (index >> 2)) & 0x01ff)
                .expect("masked test coefficient fits i32")
                - 255;
            if index % 13 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let distribution =
        collect_encode_distribution(&coefficients, 64, 64, 10).expect("collect distribution");

    let mut rho_u_q = Vec::new();
    for (rho, counts) in distribution.non_initial_rho_u_q_counts.iter().enumerate() {
        for (u_q, count) in counts.iter().copied().enumerate() {
            if count != 0 {
                rho_u_q.push((rho, u_q, count));
            }
        }
    }
    rho_u_q.sort_by_key(|&(_, _, count)| core::cmp::Reverse(count));

    println!(
        "quads total={} initial={} non_initial={}",
        distribution.total_quads, distribution.initial_quads, distribution.non_initial_quads
    );
    println!("rho={:?}", nonzero_histogram(&distribution.rho_counts));
    println!(
        "non_initial_u_q={:?}",
        nonzero_histogram(&distribution.non_initial_u_q_counts)
    );
    println!(
        "non_initial_e_qmax={:?}",
        nonzero_histogram(&distribution.non_initial_e_qmax_counts)
    );
    println!(
        "non_initial_kappa={:?}",
        nonzero_histogram(&distribution.non_initial_kappa_counts)
    );
    println!(
        "mag_sign_sample_bits={:?}",
        nonzero_histogram(&distribution.mag_sign_sample_bit_counts)
    );
    println!(
        "top_non_initial_rho_u_q={:?}",
        &rho_u_q[..rho_u_q.len().min(8)]
    );
}

#[test]
fn test_encode_cleanup_only_nonzero_block() {
    let encoded = encode_code_block(&[1], 1, 1, 5).expect("encode HT block");
    assert_eq!(encoded.num_coding_passes, 1);
    assert_eq!(encoded.num_zero_bitplanes, 4);
    assert!(encoded.data.len() >= 2);
}

#[test]
fn explicit_cleanup_bitplane_selects_the_requested_ht_set() {
    let coefficients = [0, 7, -6, 3];
    let mut workspace = HtEncodeWorkspace::try_new().expect("HT workspace allocation");

    let coarse =
        try_encode_code_block_set_with_workspace(&coefficients, 2, 2, 8, 2, 3, &mut workspace)
            .expect("encode p=2 HT set");
    let fine =
        try_encode_code_block_set_with_workspace(&coefficients, 2, 2, 8, 1, 3, &mut workspace)
            .expect("encode p=1 HT set");

    assert_eq!(coarse.num_zero_bitplanes, 5);
    assert_eq!(fine.num_zero_bitplanes, 6);
    assert_eq!(coarse.num_coding_passes, 3);
    assert_eq!(fine.num_coding_passes, 3);
    assert_eq!(
        coarse.ht_sigprop_length + coarse.ht_magref_length,
        coarse.ht_refinement_length
    );
    assert_eq!(
        fine.ht_sigprop_length + fine.ht_magref_length,
        fine.ht_refinement_length
    );
    assert!(coarse.ht_distortion_deltas.iter().all(|delta| *delta > 0.0));
    assert!(fine.ht_distortion_deltas.iter().all(|delta| *delta > 0.0));
    assert_ne!(coarse.data, fine.data);
}

#[test]
fn explicit_cleanup_bitplane_rejects_impossible_refinement() {
    let mut workspace = HtEncodeWorkspace::try_new().expect("HT workspace allocation");
    let error = try_encode_code_block_set_with_workspace(&[1], 1, 1, 1, 0, 2, &mut workspace)
        .expect_err("p=0 has no lower refinement bitplane");

    assert!(matches!(error, crate::EncodeError::InvalidInput { .. }));
}

#[test]
fn bounded_candidate_generation_produces_two_consecutive_ht_sets() {
    let coefficients = [0, 7, -6, 3];
    let mut workspace = HtEncodeWorkspace::try_new().expect("HT workspace allocation");

    let candidates =
        try_encode_code_block_candidate_sets_with_workspace(&coefficients, 2, 2, 8, &mut workspace)
            .expect("encode bounded HT candidates");

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].num_zero_bitplanes, 5);
    assert_eq!(candidates[1].num_zero_bitplanes, 6);
    assert!(candidates
        .iter()
        .all(|candidate| candidate.num_coding_passes == 3));
}

#[test]
fn tile_candidate_selection_spends_the_shared_budget_on_the_best_block() {
    let candidates = [
        synthetic_candidate(2, 4, [10.0, 10.0, 0.0]),
        synthetic_candidate(2, 1, [10.0, 100.0, 0.0]),
    ];
    let ranges = [
        HtCandidateRange { start: 0, len: 1 },
        HtCandidateRange { start: 1, len: 1 },
    ];

    let selected = select_tile_code_block_candidates(&candidates, &ranges, 5)
        .expect("tile candidate selection");

    assert_eq!(selected[0].candidate_index, 0);
    assert_eq!(selected[0].num_coding_passes, 1);
    assert_eq!(selected[1].candidate_index, 1);
    assert_eq!(selected[1].num_coding_passes, 2);
}

#[test]
fn tile_candidate_hull_can_switch_to_a_better_alternate_set() {
    let candidates = [
        synthetic_candidate(3, 2, [30.0, 5.0, 0.0]),
        synthetic_candidate(6, 0, [90.0, 0.0, 0.0]),
    ];
    let ranges = [HtCandidateRange { start: 0, len: 2 }];

    let selected = select_tile_code_block_candidates(&candidates, &ranges, 6)
        .expect("tile candidate selection");

    assert_eq!(selected[0].candidate_index, 1);
    assert_eq!(selected[0].num_coding_passes, 1);
}

#[test]
fn tile_candidate_selection_keeps_the_cheapest_cleanup_when_under_budget() {
    let candidates = [
        synthetic_candidate(5, 0, [20.0, 0.0, 0.0]),
        synthetic_candidate(7, 0, [100.0, 0.0, 0.0]),
    ];
    let ranges = [HtCandidateRange { start: 0, len: 2 }];

    let selected = select_tile_code_block_candidates(&candidates, &ranges, 1)
        .expect("tile candidate selection");

    assert_eq!(selected[0].candidate_index, 0);
    assert_eq!(selected[0].num_coding_passes, 1);
}

fn synthetic_candidate(
    cleanup_length: u32,
    refinement_length: u32,
    distortion: [f64; 3],
) -> crate::j2c::bitplane_encode::EncodedCodeBlock {
    let sigprop_length = refinement_length.min(1);
    let magref_length = refinement_length - sigprop_length;
    let num_coding_passes = 1 + u8::from(sigprop_length != 0) + u8::from(magref_length != 0);
    crate::j2c::bitplane_encode::EncodedCodeBlock {
        data: vec![0; (cleanup_length + refinement_length) as usize],
        num_coding_passes,
        num_zero_bitplanes: 0,
        ht_cleanup_length: cleanup_length,
        ht_refinement_length: refinement_length,
        ht_sigprop_length: sigprop_length,
        ht_magref_length: magref_length,
        ht_distortion_deltas: distortion,
    }
}
