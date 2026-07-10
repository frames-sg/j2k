// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{collect_encode_distribution, encode_code_block_with_passes};
use crate::HtCleanupEncodeDistribution;

#[derive(Clone, Copy)]
struct ExpectedBlock<'a> {
    data: &'a [u8],
    coding_passes: u8,
    missing_bitplanes: u8,
    cleanup_length: u32,
    refinement_length: u32,
}

fn assert_encoded_block(
    name: &str,
    coefficients: &[i32],
    width: u32,
    height: u32,
    bitplanes: u8,
    passes: u8,
    expected: ExpectedBlock<'_>,
) {
    let encoded = encode_code_block_with_passes(coefficients, width, height, bitplanes, passes)
        .expect("encode golden block");

    assert_eq!(encoded.data, expected.data, "{name} bytes");
    assert_eq!(
        encoded.num_coding_passes, expected.coding_passes,
        "{name} coding passes"
    );
    assert_eq!(
        encoded.num_zero_bitplanes, expected.missing_bitplanes,
        "{name} missing bitplanes"
    );
    assert_eq!(
        encoded.ht_cleanup_length, expected.cleanup_length,
        "{name} cleanup length"
    );
    assert_eq!(
        encoded.ht_refinement_length, expected.refinement_length,
        "{name} refinement length"
    );
    assert_eq!(
        encoded.data.len(),
        expected.cleanup_length as usize + expected.refinement_length as usize,
        "{name} segment boundary"
    );
}

#[test]
fn cleanup_refinement_and_edge_codestream_bytes_are_exact() {
    let small = [0, 1, -2, 3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15];
    assert_encoded_block(
        "cleanup",
        &small,
        4,
        4,
        6,
        1,
        ExpectedBlock {
            data: &[
                0x86, 0x4C, 0xAC, 0x5B, 0x64, 0xD1, 0xEA, 0x05, 0x6B, 0x4F, 0x88, 0x56, 0x00,
            ],
            coding_passes: 1,
            missing_bitplanes: 5,
            cleanup_length: 13,
            refinement_length: 0,
        },
    );
    assert_encoded_block(
        "sigprop",
        &small,
        4,
        4,
        6,
        2,
        ExpectedBlock {
            data: &[
                0x0E, 0xB2, 0x3E, 0x30, 0xFD, 0x6B, 0x5C, 0x7A, 0xF7, 0x56, 0x00, 0x02,
            ],
            coding_passes: 2,
            missing_bitplanes: 4,
            cleanup_length: 11,
            refinement_length: 1,
        },
    );
    assert_encoded_block(
        "magref",
        &small,
        4,
        4,
        6,
        3,
        ExpectedBlock {
            data: &[
                0x0E, 0xB2, 0x3E, 0x30, 0xFD, 0x6B, 0x5C, 0x7A, 0xF7, 0x56, 0x00, 0x02, 0x3C, 0x38,
            ],
            coding_passes: 3,
            missing_bitplanes: 4,
            cleanup_length: 11,
            refinement_length: 3,
        },
    );

    let odd = (0..7 * 5)
        .map(|index| {
            if index % 6 == 0 {
                0
            } else {
                ((index * 13) % 127) - 63
            }
        })
        .collect::<Vec<_>>();
    assert_encoded_block(
        "odd",
        &odd,
        7,
        5,
        8,
        3,
        ExpectedBlock {
            data: &[
                0x3A, 0x36, 0x83, 0xDE, 0x9D, 0x1A, 0x98, 0xB3, 0x44, 0x87, 0x91, 0x1B, 0x2B, 0x2F,
                0x60, 0xF5, 0xD9, 0xED, 0x01, 0x56, 0x00, 0xDF, 0xCE, 0xFF, 0x17, 0x34, 0x2A, 0x4F,
                0x8E, 0x48, 0x94, 0x5F, 0x00, 0x00, 0x0A, 0xBC, 0xE0, 0xF0,
            ],
            coding_passes: 3,
            missing_bitplanes: 6,
            cleanup_length: 33,
            refinement_length: 5,
        },
    );
    assert_encoded_block(
        "one by one",
        &[31],
        1,
        1,
        5,
        1,
        ExpectedBlock {
            data: &[0xFC, 0x00, 0x07, 0x74, 0x00],
            coding_passes: 1,
            missing_bitplanes: 4,
            cleanup_length: 5,
            refinement_length: 0,
        },
    );
}

fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn distribution_hash(distribution: &HtCleanupEncodeDistribution) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash = hash_u64(hash, distribution.total_quads);
    hash = hash_u64(hash, distribution.initial_quads);
    hash = hash_u64(hash, distribution.non_initial_quads);
    for counts in [
        distribution.rho_counts.as_slice(),
        distribution.initial_rho_counts.as_slice(),
        distribution.non_initial_rho_counts.as_slice(),
        distribution.non_initial_u_q_counts.as_slice(),
        distribution.non_initial_e_qmax_counts.as_slice(),
        distribution.non_initial_kappa_counts.as_slice(),
    ] {
        for &count in counts {
            hash = hash_u64(hash, count);
        }
    }
    for counts in &distribution.non_initial_rho_u_q_counts {
        for &count in counts {
            hash = hash_u64(hash, count);
        }
    }
    hash = hash_u64(hash, distribution.mag_sign_calls);
    for &count in &distribution.mag_sign_rho_counts {
        hash = hash_u64(hash, count);
    }
    for &count in &distribution.mag_sign_sample_bit_counts {
        hash = hash_u64(hash, count);
    }
    hash_u64(hash, distribution.mag_sign_encoded_samples)
}

#[test]
fn representative_cleanup_distribution_is_exact() {
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

    assert_eq!(distribution_hash(&distribution), 0xA38F_305F_035B_5E47);
    assert_eq!(
        (
            distribution.total_quads,
            distribution.initial_quads,
            distribution.non_initial_quads,
            distribution.mag_sign_calls,
            distribution.mag_sign_encoded_samples,
        ),
        (12, 4, 8, 12, 41)
    );
    assert_eq!(
        distribution.rho_counts,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 8]
    );
    assert_eq!(
        distribution.non_initial_u_q_counts,
        [
            0, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    );
    assert_eq!(
        distribution.non_initial_kappa_counts,
        [
            0, 0, 0, 0, 0, 0, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    );
    assert_eq!(
        distribution.mag_sign_sample_bit_counts,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 32, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    );
}

#[test]
fn validation_error_text_and_empty_block_result_remain_exact() {
    for bitplanes in [0, 32] {
        let Err(error) = encode_code_block_with_passes(&[1], 1, 1, bitplanes, 1) else {
            panic!("invalid bitplane count must fail");
        };
        assert_eq!(
            error,
            "HTJ2K scalar encoder currently supports 1..=31 bitplanes"
        );
    }
    for passes in [0, 4] {
        let Err(error) = encode_code_block_with_passes(&[1], 1, 1, 5, passes) else {
            panic!("invalid pass count must fail");
        };
        assert_eq!(
            error,
            "HTJ2K scalar encoder currently supports cleanup, sigprop, and one magref refinement pass"
        );
    }
    let Err(error) = encode_code_block_with_passes(&[64], 1, 1, 5, 1) else {
        panic!("oversized magnitude must fail");
    };
    assert_eq!(
        error,
        "HTJ2K block magnitude exceeds configured bitplane count"
    );

    let encoded =
        encode_code_block_with_passes(&[0; 9], 3, 3, 8, 3).expect("all-zero block is valid");
    assert!(encoded.data.is_empty());
    assert_eq!(encoded.num_coding_passes, 0);
    assert_eq!(encoded.num_zero_bitplanes, 8);
    assert_eq!(encoded.ht_cleanup_length, 0);
    assert_eq!(encoded.ht_refinement_length, 0);
}

#[test]
fn encoder_modules_remain_focused_without_broad_suppressions() {
    const ROOT: &str = include_str!("../ht_block_encode.rs");
    const MODULES: [(&str, &str, usize); 7] = [
        ("cleanup", include_str!("cleanup.rs"), 260),
        ("distribution", include_str!("distribution.rs"), 390),
        ("emit", include_str!("emit.rs"), 270),
        ("facade", include_str!("facade.rs"), 110),
        ("quad", include_str!("quad.rs"), 500),
        ("refinement", include_str!("refinement.rs"), 420),
        ("writers", include_str!("writers.rs"), 340),
    ];

    assert!(ROOT.lines().count() <= 30, "encoder root regrew");
    for (name, source, line_cap) in MODULES {
        assert!(
            source.lines().count() <= line_cap,
            "{name}.rs exceeded its focused-module line cap"
        );
        assert!(
            !source.contains("include!("),
            "{name}.rs uses source inclusion"
        );
        assert!(
            !source.contains("use super::*"),
            "{name}.rs uses a wildcard import"
        );
        assert!(
            !source.contains("#![allow"),
            "{name}.rs has a module-wide allow"
        );
        assert!(
            !source.contains("allow(unused"),
            "{name}.rs suppresses unused diagnostics"
        );
        assert!(
            !source.contains("allow(clippy::too_many_lines"),
            "{name}.rs suppresses the god-function lint"
        );
    }
}
