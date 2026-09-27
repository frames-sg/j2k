// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::super::bitplane_encode;
use super::bypass::{BitDecoder, BypassDecoder};
use super::context::{
    context_label_magnitude_refinement_coding_from_state_lazy, context_label_sign_coding_index,
    context_label_zero_coding_from_neighbors,
};
use super::facade::decode_code_block_segments_validated;
use super::flags::{
    set_significant, sign_coding_label, zero_coding_label, zero_coding_table, SIGMA_NEIGHBOURS,
    SIGMA_THIS,
};
use super::observer::NoJ2kDecodeStats;
use super::reconstruction::{reconstruct_irreversible_midpoint, MidpointReconstructor};
use super::scan::{cleanup_candidate_scan_mask, cleanup_run_length_candidate};
use super::schedule::decode_code_block_segments_inner;
use super::state::{
    BitPlaneDecodeContext, Coefficient, NeighborSignificances, COEFFICIENTS_PADDING,
    HAS_MAGNITUDE_REFINEMENT_MASK, HAS_ZERO_CODING_MASK, SIGNIFICANCE_MASK,
};
use crate::j2c::build::SubBandType;
use crate::j2c::codestream::CodeBlockStyle;
use crate::J2kCodeBlockSegment;

fn seed_130_cb_coefficients() -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(64 * 64);
    let mut state = 0x82u32 ^ 0x9e37_79b9;
    for _ in 0..64 * 64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let g = u8::try_from(state >> 24).expect("shifted PRNG channel fits u8");
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let b = u8::try_from(state >> 24).expect("shifted PRNG channel fits u8");
        coefficients.push(i32::from(b) - i32::from(g));
    }
    coefficients
}

fn generated_coefficients(width: u32, height: u32, seed: u32) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(width as usize * height as usize);
    let mut state = seed ^ 0x9e37_79b9;
    for idx in 0..width * height {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value =
            i32::try_from((state >> 16) & 0x01ff).expect("masked coefficient fits i32") - 255;
        coefficients.push(if (idx + seed).is_multiple_of(11) {
            0
        } else {
            value
        });
    }
    coefficients
}

#[test]
fn classic_coefficient_state_preserves_38_bit_magnitude() {
    let mut coefficient = Coefficient::default();
    coefficient.push_bit_at(1, 37);
    assert_eq!(coefficient.get_i64(), 1_i64 << 37);
    assert_eq!(coefficient.get(), i32::MAX);

    coefficient.set_sign(1);
    assert_eq!(coefficient.get_i64(), -(1_i64 << 37));
    assert_eq!(coefficient.get(), i32::MIN);
}

#[test]
fn per_block_midpoint_reconstruction_matches_the_reference_bit_for_bit() {
    // Magnitudes cover zero, low bits, every single-bit position, dense
    // patterns, and the 63-bit ceiling; the pass sweep hits every final-pass
    // kind, and nonzero ROI shifts (decode caps bitplanes plus shift at 63)
    // take the reference path.
    let mut magnitudes = vec![0_u64, 1, 2, 3, 5, 0x55, 0xAA, 0xFFFF, (1 << 63) - 1];
    magnitudes.extend((0..63).map(|bit| 1_u64 << bit));
    magnitudes.extend((1..63).map(|bits| (1_u64 << bits) - 1));
    for decoded_bitplanes in [0_u8, 1, 2, 3, 8, 13, 31, 62, 63, 64, 70] {
        for number_of_coding_passes in
            0..=u8::try_from((u32::from(decoded_bitplanes) * 3).min(200)).unwrap()
        {
            for roi_shift in [0_u8, 1, 7, 62, 63] {
                let block = MidpointReconstructor::new(
                    decoded_bitplanes,
                    number_of_coding_passes,
                    roi_shift,
                );
                for &magnitude in &magnitudes {
                    for sign in [0_u8, 1] {
                        let mut coefficient = Coefficient::default();
                        for bit in 0..63 {
                            coefficient.push_bit_at(u32::from(magnitude >> bit & 1 == 1), bit);
                        }
                        coefficient.set_sign(sign);
                        assert_eq!(
                            block.reconstruct(coefficient).to_bits(),
                            reconstruct_irreversible_midpoint(
                                coefficient,
                                decoded_bitplanes,
                                number_of_coding_passes,
                                roi_shift,
                            )
                            .to_bits(),
                            "bitplanes {decoded_bitplanes} passes {number_of_coding_passes} \
                             roi {roi_shift} magnitude {magnitude:#x} sign {sign}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn row_dequantization_matches_per_coefficient_reconstruction_bit_for_bit() {
    let steps = [1.0_f32, 0.007_812_5, 3.3, 1.0e-3, 257.5, 0.123_456_7];
    for decoded_bitplanes in 1_u8..=40 {
        let top = (1_u64 << decoded_bitplanes) - 1;
        let mut magnitudes = vec![0, 1, 2, 3, top, top >> 1, top & 0x5555_5555_5555];
        magnitudes.extend((0..decoded_bitplanes).map(|bit| 1_u64 << bit));
        let coefficients = magnitudes
            .iter()
            .flat_map(|&magnitude| {
                [0_u8, 1].map(|sign| {
                    let mut coefficient = Coefficient::default();
                    for bit in 0..63 {
                        coefficient.push_bit_at(u32::from(magnitude >> bit & 1 == 1), bit);
                    }
                    coefficient.set_sign(sign);
                    coefficient
                })
            })
            .collect::<Vec<_>>();
        let max_passes = 1 + 3 * (decoded_bitplanes - 1);
        for number_of_coding_passes in 1..=max_passes.min(96) {
            let block = MidpointReconstructor::new(decoded_bitplanes, number_of_coding_passes, 0);
            for step in steps {
                let mut row = vec![f32::NAN; coefficients.len()];
                block.dequantize_row(&coefficients, &mut row, step);
                for (value, &coefficient) in row.iter().zip(&coefficients) {
                    assert_eq!(
                        value.to_bits(),
                        (block.reconstruct(coefficient) * step).to_bits(),
                        "bitplanes {decoded_bitplanes} passes {number_of_coding_passes} \
                         step {step} coefficient {:#x}",
                        coefficient.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn irreversible_midpoint_reconstruction_tracks_the_last_decoded_pass() {
    let mut first_plane = Coefficient::default();
    first_plane.push_bit_at(1, 2);

    assert_eq!(
        reconstruct_irreversible_midpoint(first_plane, 3, 1, 0).to_bits(),
        6.0_f32.to_bits()
    );

    first_plane.set_sign(1);
    assert_eq!(
        reconstruct_irreversible_midpoint(first_plane, 3, 1, 0).to_bits(),
        (-6.0_f32).to_bits()
    );

    let zero = Coefficient::default();
    assert_eq!(
        reconstruct_irreversible_midpoint(zero, 3, 1, 0).to_bits(),
        0.0_f32.to_bits()
    );

    let mut newly_significant_in_sigprop = Coefficient::default();
    newly_significant_in_sigprop.push_bit_at(1, 1);
    assert_eq!(
        reconstruct_irreversible_midpoint(newly_significant_in_sigprop, 3, 2, 0).to_bits(),
        3.0_f32.to_bits()
    );

    let mut awaiting_refinement = Coefficient::default();
    awaiting_refinement.push_bit_at(1, 2);
    assert_eq!(
        reconstruct_irreversible_midpoint(awaiting_refinement, 3, 2, 0).to_bits(),
        6.0_f32.to_bits()
    );
    assert_eq!(
        reconstruct_irreversible_midpoint(awaiting_refinement, 3, 3, 0).to_bits(),
        5.0_f32.to_bits()
    );
}

#[test]
fn strict_bypass_decoder_extends_clean_segment_end_with_ones() {
    let mut decoder = BypassDecoder::new(&[0b1010_0101], true);
    let mut context = crate::j2c::arithmetic_decoder::ArithmeticDecoderContext::default();
    let mut bits = 0u16;

    for _ in 0..10 {
        bits = (bits << 1)
            | u16::try_from(decoder.read_bit(&mut context).expect("raw bit"))
                .expect("one bit fits u16");
    }

    assert_eq!(bits, 0b10_1001_0111);
}

#[test]
fn strict_bypass_decoder_rejects_missing_stuffed_bit() {
    let mut decoder = BypassDecoder::new(&[0xff], true);
    let mut context = crate::j2c::arithmetic_decoder::ArithmeticDecoderContext::default();

    for _ in 0..7 {
        assert_eq!(decoder.read_bit(&mut context), Some(1));
    }
    assert_eq!(decoder.read_bit(&mut context), None);
}

#[test]
fn classic_tier1_round_trips_38_bit_coefficients() {
    let coefficients = vec![
        0,
        1_i64 << 37,
        -((1_i64 << 37) - 1),
        17,
        -33,
        1_i64 << 36,
        0,
        -((1_i64 << 35) + 3),
        5,
        -7,
        0,
        1_i64 << 34,
        -1,
        9,
        -11,
        (1_i64 << 32) + 123,
    ];
    let style = CodeBlockStyle::default();
    let encoded = bitplane_encode::encode_code_block_segments_with_style_i64(
        &coefficients,
        4,
        4,
        SubBandType::LowLow,
        38,
        style,
    );
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut ctx = BitPlaneDecodeContext::default();

    decode_code_block_segments_validated(
        &encoded.data,
        &segments,
        4,
        4,
        encoded.num_zero_bitplanes,
        encoded.num_coding_passes,
        38,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect("decode 38-bit code block");

    let decoded = ctx
        .coefficient_rows()
        .flat_map(|row| row.iter().map(Coefficient::get_i64))
        .collect::<Vec<_>>();
    assert_eq!(decoded, coefficients);
}

fn assert_code_block_round_trip(
    style: CodeBlockStyle,
    sub_band_type: SubBandType,
    width: u32,
    height: u32,
    seed: u32,
) {
    let total_bitplanes = 10;
    let coefficients = generated_coefficients(width, height, seed);
    let encoded = bitplane_encode::encode_code_block_segments_with_style(
        &coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        style,
    );
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut ctx = BitPlaneDecodeContext::default();

    decode_code_block_segments_validated(
        &encoded.data,
        &segments,
        width,
        height,
        encoded.num_zero_bitplanes,
        encoded.num_coding_passes,
        total_bitplanes,
        sub_band_type,
        &style,
        true,
        &mut ctx,
    )
    .expect("decode code block");

    let decoded = ctx
        .coefficient_rows()
        .flat_map(|row| row.iter().map(Coefficient::get))
        .collect::<Vec<_>>();
    if let Some(index) = decoded
        .iter()
        .zip(coefficients.iter())
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "coefficient mismatch at {index}: expected {}, got {}",
            coefficients[index], decoded[index]
        );
    }
}

#[test]
fn classic_bitplane_round_trips_seed_130_cb_block() {
    let coefficients = seed_130_cb_coefficients();
    let style = CodeBlockStyle::default();
    let encoded = bitplane_encode::encode_code_block_segments_with_style(
        &coefficients,
        64,
        64,
        SubBandType::LowLow,
        8,
        style,
    );
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut ctx = BitPlaneDecodeContext::default();

    decode_code_block_segments_validated(
        &encoded.data,
        &segments,
        64,
        64,
        encoded.num_zero_bitplanes,
        encoded.num_coding_passes,
        8,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect("decode code block");

    let decoded = ctx
        .coefficient_rows()
        .flat_map(|row| row.iter().map(Coefficient::get))
        .collect::<Vec<_>>();
    let mismatch_count = decoded
        .iter()
        .zip(coefficients.iter())
        .filter(|(actual, expected)| actual != expected)
        .count();
    if let Some(index) = decoded
        .iter()
        .zip(coefficients.iter())
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "{mismatch_count} coefficient mismatch(es); first at {index}: expected {}, got {}",
            coefficients[index], decoded[index]
        );
    }
}

#[test]
fn vertically_causal_context_masks_only_the_next_stripe_neighbor() {
    let mut ctx = BitPlaneDecodeContext {
        width: 1,
        height: 8,
        padded_width: 3,
        style: CodeBlockStyle {
            vertically_causal_context: true,
            ..CodeBlockStyle::default()
        },
        ..BitPlaneDecodeContext::default()
    };
    ctx.neighbor_significances.resize(
        ctx.padded_width as usize * 10,
        NeighborSignificances::default(),
    );

    let y = 3;
    let idx = (y + COEFFICIENTS_PADDING as usize) * ctx.padded_width as usize
        + COEFFICIENTS_PADDING as usize;
    ctx.neighbor_significances[idx].set_top();
    ctx.neighbor_significances[idx].set_bottom();

    assert_eq!(ctx.neighborhood_significance_states_index(idx, y), 1 << 6);
    ctx.style.vertically_causal_context = false;
    assert_eq!(
        ctx.neighborhood_significance_states_index(idx, y),
        (1 << 6) | 1
    );
}

#[test]
fn packed_column_contexts_match_per_coefficient_contexts() {
    // Random significance and sign patterns, including stripe edges, a short
    // final stripe, and block borders, recorded through both bookkeeping
    // schemes; every coefficient must then see identical contexts.
    let mut seed = 0x2545_f491_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for (width, height) in [(1, 1), (3, 5), (7, 10), (9, 12), (16, 3)] {
        for density in [3_u32, 8, 14] {
            let mut ctx = BitPlaneDecodeContext::default();
            let style = CodeBlockStyle::default();
            ctx.reset_for_job(
                width,
                height,
                0,
                1,
                SubBandType::LowLow,
                &style,
                8,
                true,
                false,
            )
            .expect("per-coefficient context");
            let stride = ctx.padded_width as usize;
            let mut flags = vec![0_u32; (height.div_ceil(4) as usize + 2) * stride];
            let flag_index = |x: usize, y: usize| (y / 4 + 1) * stride + x + 1;
            let coefficient_index = |x: usize, y: usize| (y + 1) * stride + x + 1;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    if next() % 16 >= density {
                        continue;
                    }
                    let sign = next() & 1;
                    let idx = coefficient_index(x, y);
                    ctx.set_sign_index(idx, u8::from(sign != 0));
                    ctx.set_significant_index(idx, y, stride);
                    let index = flag_index(x, y);
                    let mut own = flags[index];
                    let row = u32::try_from(y % 4).expect("row below four");
                    set_significant(&mut flags, &mut own, index, stride, row, sign);
                    flags[index] = own;
                }
            }
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let (idx, index, row) = (
                        coefficient_index(x, y),
                        flag_index(x, y),
                        u32::try_from(y % 4).expect("row below four"),
                    );
                    let word = flags[index];
                    let neighbors = ctx.neighborhood_significance_states_index(idx, y);
                    for sub_band_type in [
                        SubBandType::LowLow,
                        SubBandType::LowHigh,
                        SubBandType::HighLow,
                        SubBandType::HighHigh,
                    ] {
                        assert_eq!(
                            zero_coding_label(zero_coding_table(sub_band_type), word, row),
                            context_label_zero_coding_from_neighbors(neighbors, sub_band_type),
                            "{width}x{height} ({x}, {y})"
                        );
                    }
                    assert_eq!(
                        sign_coding_label(word, flags[index - 1], flags[index + 1], row),
                        context_label_sign_coding_index(idx, y, &ctx),
                        "{width}x{height} ({x}, {y})"
                    );
                    assert_eq!((word >> (3 * row)) & SIGMA_NEIGHBOURS != 0, neighbors != 0);
                    assert_eq!(
                        (word >> (3 * row)) & SIGMA_THIS != 0,
                        ctx.coefficient_states[idx].is_significant()
                    );
                }
            }
        }
    }
}

/// Mostly small magnitudes with occasional large ones, like wavelet
/// subbands, so every pass and the run-length path occur.
fn random_subband_block(next: &mut impl FnMut() -> u32, len: u32) -> Vec<i32> {
    (0..len)
        .map(|_| {
            let draw = next();
            let magnitude = match draw % 8 {
                0..=3 => 0,
                4 | 5 => (draw >> 8) % 4,
                6 => (draw >> 8) % 64,
                _ => (draw >> 8) % 400,
            };
            let magnitude = i32::try_from(magnitude).expect("bounded magnitude");
            if draw & 1 << 30 != 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

/// Decode `passes` passes through the packed and the per-coefficient passes.
fn decode_through_both_pass_families(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    (width, height): (u32, u32),
    sub_band_type: SubBandType,
    passes: u8,
) -> (Vec<i64>, Vec<i64>) {
    let style = CodeBlockStyle::default();
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass.min(passes),
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut packed = BitPlaneDecodeContext::default();
    decode_code_block_segments_validated(
        &encoded.data,
        &segments,
        width,
        height,
        encoded.num_zero_bitplanes,
        passes,
        10,
        sub_band_type,
        &style,
        true,
        &mut packed,
    )
    .expect("packed decode");
    assert!(packed.uses_packed_columns());

    let mut per_coefficient = BitPlaneDecodeContext::default();
    per_coefficient
        .reset_for_job(
            width,
            height,
            encoded.num_zero_bitplanes,
            passes,
            sub_band_type,
            &style,
            10,
            true,
            false,
        )
        .expect("per-coefficient context");
    decode_code_block_segments_inner(
        &encoded.data,
        &segments,
        passes,
        &mut per_coefficient,
        &mut NoJ2kDecodeStats,
    )
    .expect("per-coefficient decode");

    let rows = |ctx: &BitPlaneDecodeContext| {
        ctx.coefficient_rows()
            .flat_map(|row| row.iter().map(Coefficient::get_i64))
            .collect::<Vec<_>>()
    };
    (rows(&packed), rows(&per_coefficient))
}

#[test]
fn packed_and_per_coefficient_passes_decode_identically() {
    let mut seed = 0x9e37_79b9_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for (width, height) in [(1_u32, 1_u32), (4, 4), (5, 7), (13, 9), (32, 17), (64, 64)] {
        for sub_band_type in [
            SubBandType::LowLow,
            SubBandType::HighLow,
            SubBandType::HighHigh,
        ] {
            let coefficients = random_subband_block(&mut next, width * height);
            let encoded = bitplane_encode::encode_code_block_segments_with_style(
                &coefficients,
                width,
                height,
                sub_band_type,
                10,
                CodeBlockStyle::default(),
            );
            // Every truncation point, so each pass family's state after a
            // partial bitplane is compared too.
            for passes in 1..=encoded.num_coding_passes {
                let (packed, per_coefficient) = decode_through_both_pass_families(
                    &encoded,
                    (width, height),
                    sub_band_type,
                    passes,
                );
                assert_eq!(
                    packed, per_coefficient,
                    "{width}x{height} {passes}/{} passes",
                    encoded.num_coding_passes
                );
                if passes == encoded.num_coding_passes {
                    let expected = coefficients.iter().map(|&value| i64::from(value));
                    assert!(packed.iter().copied().eq(expected));
                }
            }
        }
    }
}

#[test]
fn refined_magnitude_context_does_not_require_neighbor_state() {
    let state = SIGNIFICANCE_MASK | HAS_MAGNITUDE_REFINEMENT_MASK;

    assert_eq!(
        context_label_magnitude_refinement_coding_from_state_lazy(state, || {
            panic!("refined magnitude context should not inspect neighbors")
        }),
        16
    );
}

#[test]
fn first_magnitude_context_uses_neighbor_presence() {
    assert_eq!(
        context_label_magnitude_refinement_coding_from_state_lazy(SIGNIFICANCE_MASK, || 0),
        14
    );
    assert_eq!(
        context_label_magnitude_refinement_coding_from_state_lazy(SIGNIFICANCE_MASK, || 1),
        15
    );
}

#[test]
fn scan_unit_masks_track_significance_and_current_bitplane_zero_coding() {
    let mut ctx = BitPlaneDecodeContext::default();
    let style = CodeBlockStyle::default();
    ctx.reset_for_job(5, 6, 0, 4, SubBandType::LowLow, &style, 8, true, false)
        .expect("reset context");

    let padded_width = ctx.padded_width as usize;
    let y = 4usize;
    let x = 3usize;
    let idx =
        (y + COEFFICIENTS_PADDING as usize) * padded_width + x + COEFFICIENTS_PADDING as usize;
    let scan_unit = (y >> 2) * ctx.width as usize + x;
    let bit = 1u8 << (y & 3);

    ctx.set_significant_index(idx, y, padded_width);
    ctx.set_zero_coding_index(idx, y, padded_width);

    assert_eq!(ctx.significant_scan_masks[scan_unit], bit);
    assert_eq!(ctx.zero_coding_scan_masks[scan_unit], bit);

    ctx.reset_for_next_bitplane();

    assert_eq!(ctx.significant_scan_masks[scan_unit], bit);
    assert_eq!(ctx.zero_coding_scan_masks[scan_unit], 0);
    assert_ne!(
        ctx.coefficient_states[idx].0 & HAS_ZERO_CODING_MASK,
        0,
        "normal arithmetic mode uses the reset scan mask as its transient source of truth"
    );

    let bypass_style = CodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        ..CodeBlockStyle::default()
    };
    ctx.reset_for_job(
        5,
        6,
        0,
        4,
        SubBandType::LowLow,
        &bypass_style,
        8,
        true,
        false,
    )
    .expect("reset bypass context");
    ctx.set_zero_coding_index(idx, y, padded_width);
    ctx.reset_for_next_bitplane();
    assert_eq!(ctx.coefficient_states[idx].0 & HAS_ZERO_CODING_MASK, 0);
}

#[test]
fn classic_selective_bypass_round_trips_padded_rgb8_cb_block() {
    let pixels = (0..7 * 5 * 3)
        .map(|index| u8::try_from((index * 41) & 0xff).expect("masked pixel fits u8"))
        .collect::<Vec<_>>();
    let mut coefficients = vec![0i32; 8 * 8];
    for y in 0..5usize {
        for x in 0..7usize {
            let src = (y * 7 + x) * 3;
            coefficients[y * 8 + x] = i32::from(pixels[src + 2]) - i32::from(pixels[src + 1]);
        }
    }
    let style = CodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        ..CodeBlockStyle::default()
    };
    let encoded = bitplane_encode::encode_code_block_segments_with_style(
        &coefficients,
        8,
        8,
        SubBandType::LowLow,
        9,
        style,
    );
    let segments = encoded
        .segments
        .iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let mut ctx = BitPlaneDecodeContext::default();

    decode_code_block_segments_validated(
        &encoded.data,
        &segments,
        8,
        8,
        encoded.num_zero_bitplanes,
        encoded.num_coding_passes,
        9,
        SubBandType::LowLow,
        &style,
        true,
        &mut ctx,
    )
    .expect("decode selective-bypass Cb code block");

    let decoded = ctx
        .coefficient_rows()
        .flat_map(|row| row.iter().map(Coefficient::get))
        .collect::<Vec<_>>();
    assert_eq!(decoded, coefficients);
}

#[test]
fn cleanup_candidate_mask_excludes_significant_and_zero_coded_coefficients() {
    let mut ctx = BitPlaneDecodeContext::default();
    let style = CodeBlockStyle::default();
    ctx.reset_for_job(5, 6, 0, 4, SubBandType::LowLow, &style, 8, true, false)
        .expect("reset context");

    let padded_width = ctx.padded_width as usize;
    let significant_idx =
        (1 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 2;
    let zero_coded_idx =
        (3 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 2;
    let scan_unit = 2;

    ctx.set_significant_index(significant_idx, 1, padded_width);
    ctx.set_zero_coding_index(zero_coded_idx, 3, padded_width);

    assert_eq!(cleanup_candidate_scan_mask(&ctx, scan_unit, 4), 0b0101);
    assert_eq!(cleanup_candidate_scan_mask(&ctx, scan_unit, 2), 0b0001);
}

#[test]
fn cleanup_run_length_rejects_a_zero_coded_coefficient_without_significant_neighbors() {
    let mut ctx = BitPlaneDecodeContext::default();
    let style = CodeBlockStyle::default();
    ctx.reset_for_job(1, 4, 0, 4, SubBandType::LowLow, &style, 8, true, false)
        .expect("reset context");

    let padded_width = ctx.padded_width as usize;
    let top_idx = COEFFICIENTS_PADDING as usize * padded_width + COEFFICIENTS_PADDING as usize;
    ctx.set_zero_coding_index(top_idx + padded_width, 1, padded_width);

    assert!(!cleanup_run_length_candidate(
        &ctx,
        top_idx,
        padded_width,
        0
    ));
}

#[test]
fn classic_bitplane_round_trips_subband_and_style_matrix() {
    let styles = [
        CodeBlockStyle::default(),
        CodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            ..CodeBlockStyle::default()
        },
        CodeBlockStyle {
            termination_on_each_pass: true,
            reset_context_probabilities: true,
            ..CodeBlockStyle::default()
        },
        CodeBlockStyle {
            segmentation_symbols: true,
            ..CodeBlockStyle::default()
        },
        CodeBlockStyle {
            vertically_causal_context: true,
            ..CodeBlockStyle::default()
        },
    ];
    let subbands = [
        SubBandType::LowLow,
        SubBandType::LowHigh,
        SubBandType::HighLow,
        SubBandType::HighHigh,
    ];

    for (style_idx, style) in styles.into_iter().enumerate() {
        for (subband_idx, sub_band_type) in subbands.into_iter().enumerate() {
            assert_code_block_round_trip(
                style,
                sub_band_type,
                32,
                19,
                0x4a32_1000
                    + u32::try_from(style_idx).expect("style index fits u32") * 17
                    + u32::try_from(subband_idx).expect("subband index fits u32"),
            );
        }
    }
}
