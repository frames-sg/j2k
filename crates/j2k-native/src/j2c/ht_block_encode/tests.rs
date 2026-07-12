// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use crate::j2c::coefficient_view::CoefficientBlockView;

use super::cleanup::{
    convert_nonzero_to_aligned_sign_magnitude_and_max, encode_cleanup_segment,
    encode_cleanup_segment_from_coefficients,
};
use super::distribution::collect_encode_distribution;
use super::facade::{encode_code_block, encode_code_block_view, encode_code_block_with_passes};

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
