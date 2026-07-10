// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::cleanup::{
    convert_nonzero_to_aligned_sign_magnitude_and_max, encode_cleanup_segment,
    encode_cleanup_segment_from_coefficients,
};
use super::distribution::collect_encode_distribution;
use super::facade::encode_code_block;

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
            let value = (((index * 73) ^ (index >> 2)) & 0x01ff) as i32 - 255;
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
