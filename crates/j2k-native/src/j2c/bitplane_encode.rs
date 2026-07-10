//! EBCOT Tier-1 encoder for JPEG 2000 (ITU-T T.800 Annex D).
//!
//! Encodes quantized wavelet coefficients into code-block bitstreams using:
//! - MQ arithmetic coding
//! - Context-dependent coding with the same 19 contexts as the decoder
//! - Three passes per bitplane: significance propagation, magnitude refinement, cleanup
//! - Column-stripe scanning order (4-row stripes)

use alloc::vec;
use alloc::vec::Vec;

use super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::build::SubBandType;
use super::codestream::CodeBlockStyle;

mod distortion;
mod passes;
mod segments;
mod tokens;

use self::passes::{
    cleanup_pass, clear_coded_in_current_pass, magnitude_refinement_pass,
    prepare_padded_coefficients, significance_propagation_pass,
};
use self::segments::{arithmetic_encoder_capacity, encode_segmentation_symbols, reset_contexts};
pub(crate) use self::tokens::{
    pack_classic_selective_bypass_tier1_tokens, ClassicTier1TokenSegment,
};

/// Result of encoding a single code-block.
#[derive(Debug)]
pub(crate) struct EncodedCodeBlock {
    /// The compressed bitstream data.
    pub(crate) data: Vec<u8>,
    /// Number of coding passes actually generated.
    pub(crate) num_coding_passes: u8,
    /// Number of leading zero bitplanes (missing MSBs).
    pub(crate) num_zero_bitplanes: u8,
    /// HTJ2K cleanup segment length in bytes when this block uses HT coding.
    pub(crate) ht_cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes when this block uses HT coding.
    pub(crate) ht_refinement_length: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EncodedCodeBlockSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u8,
    pub(crate) end_coding_pass: u8,
    pub(crate) distortion_delta: f64,
    pub(crate) use_arithmetic: bool,
}

#[derive(Debug)]
pub(crate) struct EncodedCodeBlockWithSegments {
    pub(crate) data: Vec<u8>,
    pub(crate) segments: Vec<EncodedCodeBlockSegment>,
    pub(crate) num_coding_passes: u8,
    pub(crate) num_zero_bitplanes: u8,
}

/// Encode a single code-block's quantized coefficients.
///
/// `coefficients` are quantized i32 values in row-major order.
/// `width`, `height` are the code-block dimensions.
/// `sub_band_type` determines which zero-coding context table to use.
/// `total_bitplanes` is the JPEG 2000 `Mb` value for this subband/code-block.
pub(crate) fn encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodedCodeBlock {
    let coefficients = i32_coefficients_to_i64(coefficients);
    encode_code_block_i64(&coefficients, width, height, sub_band_type, total_bitplanes)
}

pub(crate) fn encode_code_block_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodedCodeBlock {
    encode_code_block_with_style_i64(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        &CodeBlockStyle::default(),
    )
}

fn i32_coefficients_to_i64(coefficients: &[i32]) -> Vec<i64> {
    coefficients
        .iter()
        .map(|&coefficient| i64::from(coefficient))
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "the cohesive pass-order loop is kept intact to protect Tier-1 byte ordering"
)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn encode_code_block_with_style_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodedCodeBlock {
    let w = width as usize;
    let h = height as usize;

    // Determine maximum magnitude and number of bitplanes
    let max_magnitude = coefficients
        .iter()
        .map(|c| c.unsigned_abs())
        .max()
        .unwrap_or(0);

    if max_magnitude == 0 {
        return EncodedCodeBlock {
            data: Vec::new(),
            num_coding_passes: 0,
            num_zero_bitplanes: total_bitplanes,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
        };
    }

    let num_bitplanes = 64 - max_magnitude.leading_zeros();
    let num_bitplanes_u8 =
        u8::try_from(num_bitplanes).expect("a u64 magnitude has at most 64 bitplanes");
    debug_assert!(num_bitplanes_u8 <= total_bitplanes);
    let num_zero_bitplanes = total_bitplanes.saturating_sub(num_bitplanes_u8);

    // Build padded coefficient magnitude and state arrays.
    let pw = w + 2; // Padded width for neighbor access
    let (magnitudes, mut states) = prepare_padded_coefficients(coefficients, w, h, pw);
    let mut neighbors = vec![0u8; magnitudes.len()]; // Packed neighbor significances

    let mut encoder =
        ArithmeticEncoder::with_capacity(arithmetic_encoder_capacity(w, h, num_bitplanes as usize));
    let mut contexts = [ArithmeticEncoderContext::default(); 19];
    reset_contexts(&mut contexts);

    let mut num_coding_passes = 0u8;
    let mut coded_indices = Vec::new();

    // Process bitplanes from MSB to LSB
    for bp in (0..num_bitplanes).rev() {
        let bit_mask = 1u64 << bp;
        let is_first_bitplane = bp == num_bitplanes - 1;

        if is_first_bitplane {
            // First bitplane: cleanup pass only
            cleanup_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            if style.segmentation_symbols {
                encode_segmentation_symbols(&mut encoder, &mut contexts);
            }
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }
        } else {
            // Subsequent bitplanes: SPP, MRP, Cleanup
            significance_propagation_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut coded_indices,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }

            magnitude_refinement_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                style,
            );
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }

            cleanup_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            if style.segmentation_symbols {
                encode_segmentation_symbols(&mut encoder, &mut contexts);
            }
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }
        }

        clear_coded_in_current_pass(&mut states, &mut coded_indices);
    }

    let data = encoder.finish();

    EncodedCodeBlock {
        data,
        num_coding_passes,
        num_zero_bitplanes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable segmented Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn encode_code_block_segments_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodedCodeBlockWithSegments {
    let coefficients = i32_coefficients_to_i64(coefficients);
    encode_code_block_segments_with_style_i64(
        &coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        style,
    )
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable segmented Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn encode_code_block_segments_with_style_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodedCodeBlockWithSegments {
    segments::encode_segmented_code_block(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        style,
    )
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::super::build::SubBandType;
    use super::super::codestream::CodeBlockStyle;
    use super::distortion::segment_distortion_delta;
    use super::passes::{
        clear_coded_in_current_pass, mark_coded_in_current_pass, prepare_padded_coefficients,
        CODED_IN_CURRENT_PASS, NEGATIVE, SIGNIFICANT,
    };
    use super::{
        encode_code_block, encode_code_block_segments_with_style,
        encode_code_block_segments_with_style_i64, pack_classic_selective_bypass_tier1_tokens,
        ClassicTier1TokenSegment,
    };

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
        let scalar = encode_code_block_segments_with_style(
            &coefficients,
            1,
            1,
            SubBandType::LowLow,
            1,
            &style,
        );
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
                style,
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
}
