// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::build::SubBandType;
use super::super::codestream::CodeBlockStyle;
use super::super::coefficient_view::CoefficientBlockView;
use super::distortion::segment_distortion_delta;
use super::passes::{
    clear_coded_in_current_pass, mark_coded_in_current_pass, CODED_IN_CURRENT_PASS, NEGATIVE,
    SIGNIFICANT,
};
use super::preparation::prepare_padded_coefficients;
use super::{
    encode_code_block, encode_code_block_i64, encode_code_block_segments_with_style,
    encode_code_block_segments_with_style_i64, encode_code_block_view,
    pack_classic_selective_bypass_tier1_tokens, ClassicTier1TokenSegment, EncodedCodeBlock,
};

fn assert_encoded_blocks_equal(actual: &EncodedCodeBlock, expected: &EncodedCodeBlock) {
    assert_eq!(actual.data, expected.data);
    assert_eq!(actual.num_coding_passes, expected.num_coding_passes);
    assert_eq!(actual.num_zero_bitplanes, expected.num_zero_bitplanes);
    assert_eq!(actual.ht_cleanup_length, expected.ht_cleanup_length);
    assert_eq!(actual.ht_refinement_length, expected.ht_refinement_length);
}

#[test]
fn classic_i32_strided_block_is_byte_exact_with_contiguous_adapter() {
    const WIDTH: usize = 7;
    const HEIGHT: usize = 5;
    const STRIDE: usize = 11;
    const OFFSET: usize = 13;
    let contiguous = (0_i32..i32::try_from(WIDTH * HEIGHT).expect("test size fits i32"))
        .map(|index| match index % 5 {
            0 => 0,
            1 => index * 3,
            2 => -(index * 2),
            3 => 17 - index,
            _ => index / 2,
        })
        .collect::<Vec<_>>();
    let mut padded = vec![i32::MIN; OFFSET + STRIDE * HEIGHT + 9];
    for y in 0..HEIGHT {
        padded[OFFSET + y * STRIDE..OFFSET + y * STRIDE + WIDTH]
            .copy_from_slice(&contiguous[y * WIDTH..(y + 1) * WIDTH]);
    }
    let view = CoefficientBlockView::try_new(&padded, OFFSET, WIDTH, HEIGHT, STRIDE)
        .expect("valid strided classic block");

    let expected = encode_code_block(
        &contiguous,
        u32::try_from(WIDTH).expect("test width fits u32"),
        u32::try_from(HEIGHT).expect("test height fits u32"),
        SubBandType::HighLow,
        12,
    );
    let actual = encode_code_block_view(view, SubBandType::HighLow, 12);
    assert_encoded_blocks_equal(&actual, &expected);
}

#[test]
fn classic_i64_strided_block_is_byte_exact_at_high_precision() {
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;
    const STRIDE: usize = 9;
    const OFFSET: usize = 7;
    let contiguous = (0_usize..WIDTH * HEIGHT)
        .map(|index| {
            let magnitude =
                (1_i64 << 36) - i64::try_from(index * 1_003).expect("test offset fits i64");
            if index.is_multiple_of(3) {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect::<Vec<_>>();
    let mut padded = vec![i64::MIN; OFFSET + STRIDE * HEIGHT + 3];
    for y in 0..HEIGHT {
        padded[OFFSET + y * STRIDE..OFFSET + y * STRIDE + WIDTH]
            .copy_from_slice(&contiguous[y * WIDTH..(y + 1) * WIDTH]);
    }
    let view = CoefficientBlockView::try_new(&padded, OFFSET, WIDTH, HEIGHT, STRIDE)
        .expect("valid strided exact classic block");

    let expected = encode_code_block_i64(
        &contiguous,
        u32::try_from(WIDTH).expect("test width fits u32"),
        u32::try_from(HEIGHT).expect("test height fits u32"),
        SubBandType::HighHigh,
        40,
    );
    let actual = encode_code_block_view(view, SubBandType::HighHigh, 40);
    assert_encoded_blocks_equal(&actual, &expected);
}

#[test]
fn test_encode_all_zeros() {
    let coeffs = vec![0i32; 16];
    let result = encode_code_block(&coeffs, 4, 4, SubBandType::LowLow, 8);
    assert_eq!(result.num_coding_passes, 0);
    assert!(result.data.is_empty());
    assert_eq!(result.num_zero_bitplanes, 8);
}

#[test]
fn test_encode_single_nonzero() {
    let mut coeffs = vec![0i32; 16];
    coeffs[0] = 128;
    let result = encode_code_block(&coeffs, 4, 4, SubBandType::LowLow, 8);
    assert!(result.num_coding_passes > 0);
    assert!(!result.data.is_empty());
    assert_eq!(result.num_zero_bitplanes, 0);
}

#[test]
fn pack_classic_selective_bypass_tokens_matches_scalar_single_cleanup_block() {
    let style = CodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
        high_throughput_block_coding: false,
    };
    let coefficients = [1i32];
    let scalar =
        encode_code_block_segments_with_style(&coefficients, 1, 1, SubBandType::LowLow, 1, style);
    let token_bytes = pack_mq_test_tokens(&[(0, 1), (9, 0)]);
    let packed = pack_classic_selective_bypass_tier1_tokens(
        &token_bytes,
        &[ClassicTier1TokenSegment {
            token_bit_offset: 0,
            token_bit_count: 12,
            start_coding_pass: 0,
            end_coding_pass: 1,
            use_arithmetic: true,
        }],
        scalar.num_coding_passes,
        scalar.num_zero_bitplanes,
    )
    .expect("tokens pack");

    assert_eq!(packed.data, scalar.data);
    assert_eq!(packed.num_coding_passes, scalar.num_coding_passes);
    assert_eq!(packed.num_zero_bitplanes, scalar.num_zero_bitplanes);
    assert_eq!(packed.segments.len(), scalar.segments.len());
    for (packed_segment, scalar_segment) in packed.segments.iter().zip(&scalar.segments) {
        assert_eq!(packed_segment.data_offset, scalar_segment.data_offset);
        assert_eq!(packed_segment.data_length, scalar_segment.data_length);
        assert_eq!(
            packed_segment.start_coding_pass,
            scalar_segment.start_coding_pass
        );
        assert_eq!(
            packed_segment.end_coding_pass,
            scalar_segment.end_coding_pass
        );
        assert_eq!(packed_segment.use_arithmetic, scalar_segment.use_arithmetic);
    }
}

#[test]
fn token_segments_must_cover_coding_passes_contiguously() {
    let segment = |start_coding_pass, end_coding_pass| ClassicTier1TokenSegment {
        token_bit_offset: 0,
        token_bit_count: 0,
        start_coding_pass,
        end_coding_pass,
        use_arithmetic: false,
    };

    for (segments, passes) in [
        (vec![segment(1, 2)], 2),
        (vec![segment(0, 1), segment(2, 3)], 3),
        (vec![segment(0, 1)], 2),
    ] {
        assert!(
            pack_classic_selective_bypass_tier1_tokens(&[], &segments, passes, 0).is_err(),
            "gapped token pass schedule must be rejected: {segments:?}"
        );
    }
}

fn pack_mq_test_tokens(tokens: &[(u8, u8)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut current = 0u8;
    let mut bits = 0u8;
    for &(ctx, bit) in tokens {
        let value = (ctx & 0x1F) | ((bit & 1) << 5);
        for shift in (0..6).rev() {
            current = (current << 1) | ((value >> shift) & 1);
            bits += 1;
            if bits == 8 {
                bytes.push(current);
                current = 0;
                bits = 0;
            }
        }
    }

    if bits != 0 {
        bytes.push(current << (8 - bits));
    }
    bytes
}

#[test]
fn token_segments_must_follow_selective_bypass_coding_modes() {
    let segment = |start_coding_pass, end_coding_pass, use_arithmetic| ClassicTier1TokenSegment {
        token_bit_offset: 0,
        token_bit_count: 0,
        start_coding_pass,
        end_coding_pass,
        use_arithmetic,
    };
    let malformed = [
        (vec![segment(0, 1, false)], 1),
        (vec![segment(0, 10, true), segment(10, 11, true)], 11),
        (vec![segment(0, 11, true), segment(11, 12, false)], 12),
        (vec![segment(0, 10, true), segment(10, 13, false)], 13),
        (vec![segment(0, 12, true), segment(12, 13, false)], 13),
    ];

    for (segments, passes) in malformed {
        assert!(
            pack_classic_selective_bypass_tier1_tokens(&[], &segments, passes, 0).is_err(),
            "invalid selective-bypass schedule must be rejected: {segments:?}"
        );
    }
}

#[test]
fn test_encode_various_magnitudes() {
    let coeffs: Vec<i32> = (0..64)
        .map(|x| if x % 3 == 0 { x * 10 } else { -x })
        .collect();
    let result = encode_code_block(&coeffs, 8, 8, SubBandType::HighHigh, 12);
    assert!(result.num_coding_passes > 0);
    assert!(!result.data.is_empty());
}

#[test]
fn test_zero_bitplanes_count() {
    // Max value is 7 (3 bits), so with Mb=8 we have 8 - 3 = 5 zero bitplanes.
    let coeffs = vec![7i32, -3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let result = encode_code_block(&coeffs, 4, 4, SubBandType::LowLow, 8);
    assert_eq!(result.num_zero_bitplanes, 5);
}

#[test]
fn padded_coefficient_preparation_stores_sign_in_state_flags() {
    let coeffs = vec![7i64, -3, 0, -9];
    let (magnitudes, states) = prepare_padded_coefficients(&coeffs, 2, 2, 4);

    assert_eq!(magnitudes[5], 7);
    assert_eq!(magnitudes[6], 3);
    assert_eq!(magnitudes[9], 0);
    assert_eq!(magnitudes[10], 9);
    assert_eq!(states[5] & NEGATIVE, 0);
    assert_ne!(states[6] & NEGATIVE, 0);
    assert_eq!(states[9] & NEGATIVE, 0);
    assert_ne!(states[10] & NEGATIVE, 0);
}

#[test]
fn clear_coded_in_current_pass_touches_only_recorded_indices() {
    let mut states = vec![0u8; 8];
    let mut coded_indices = Vec::new();

    mark_coded_in_current_pass(2, &mut states, &mut coded_indices);
    mark_coded_in_current_pass(5, &mut states, &mut coded_indices);
    states[6] = SIGNIFICANT;

    clear_coded_in_current_pass(&mut states, &mut coded_indices);

    assert_eq!(states[2] & CODED_IN_CURRENT_PASS, 0);
    assert_eq!(states[5] & CODED_IN_CURRENT_PASS, 0);
    assert_eq!(states[6], SIGNIFICANT);
    assert!(coded_indices.is_empty());
}

#[test]
fn pcrd_distortion_delta_reflects_residual_error_reduction() {
    let sparse_delta = segment_distortion_delta(&[8], 0, 1, 4);
    let dense_delta = segment_distortion_delta(&[15], 0, 1, 4);

    assert!(
            dense_delta > sparse_delta,
            "coefficients with the same MSB but larger residual error should have larger PCRD distortion reduction"
        );
}

#[test]
fn str011a_classic_tier1_byte_baseline() {
    let coefficients = (0..64)
        .map(|index| {
            let magnitude = i64::from((index * 37) % 4096);
            if index % 3 == 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect::<Vec<_>>();
    let styles = [
        CodeBlockStyle::default(),
        CodeBlockStyle {
            termination_on_each_pass: true,
            ..CodeBlockStyle::default()
        },
        CodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            ..CodeBlockStyle::default()
        },
        CodeBlockStyle {
            reset_context_probabilities: true,
            vertically_causal_context: true,
            segmentation_symbols: true,
            ..CodeBlockStyle::default()
        },
    ];
    let expected = [
        (99, 0xfcce_40ac_5a7f_501d, 34, 0, 1, 0x64c9_46a4_01c3_99fb),
        (150, 0x800f_e1ae_529c_bf1a, 34, 0, 34, 0x2cfa_e242_260a_6925),
        (113, 0xdf22_abbf_10e1_19e7, 34, 0, 17, 0x631f_df7d_4363_90a3),
        (105, 0x45c9_9a5b_bfc6_8a4f, 34, 0, 1, 0xf29c_50e4_c7a1_1511),
    ];

    for (index, style) in styles.iter().enumerate() {
        let encoded = encode_code_block_segments_with_style_i64(
            &coefficients,
            8,
            8,
            SubBandType::HighHigh,
            12,
            *style,
        );
        let digest = encoded
            .data
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                hash.wrapping_mul(0x0100_0000_01b3) ^ u64::from(*byte)
            });
        let mut segment_bytes = Vec::new();
        for segment in &encoded.segments {
            segment_bytes.extend_from_slice(&segment.data_offset.to_le_bytes());
            segment_bytes.extend_from_slice(&segment.data_length.to_le_bytes());
            segment_bytes.push(segment.start_coding_pass);
            segment_bytes.push(segment.end_coding_pass);
            segment_bytes.extend_from_slice(&segment.distortion_delta.to_bits().to_le_bytes());
            segment_bytes.push(u8::from(segment.use_arithmetic));
        }
        let segment_digest = segment_bytes
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                hash.wrapping_mul(0x0100_0000_01b3) ^ u64::from(*byte)
            });
        assert_eq!(
            (
                encoded.data.len(),
                digest,
                encoded.num_coding_passes,
                encoded.num_zero_bitplanes,
                encoded.segments.len(),
                segment_digest,
            ),
            expected[index],
            "classic Tier-1 bytes or segment accounting changed for style {index}",
        );
    }
}
