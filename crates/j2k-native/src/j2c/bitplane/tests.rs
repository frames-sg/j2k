// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::super::bitplane_encode;
use super::arithmetic::cleanup_candidate_scan_mask;
use super::context::{
    context_label_magnitude_refinement_coding_from_state_lazy, context_label_sign_coding_index,
    context_label_sign_coding_index_normal,
};
use super::facade::decode_code_block_segments_validated;
use super::state::{
    BitPlaneDecodeContext, Coefficient, CoefficientState, NeighborSignificances,
    COEFFICIENTS_PADDING, HAS_MAGNITUDE_REFINEMENT_MASK, SIGNIFICANCE_MASK,
};
use crate::j2c::build::SubBandType;
use crate::j2c::codestream::CodeBlockStyle;
use crate::J2kCodeBlockSegment;

fn seed_130_cb_coefficients() -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(64 * 64);
    let mut state = 130u32 ^ 0x9e37_79b9;
    for _ in 0..64 * 64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let _r = (state >> 24) as u8;
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let g = (state >> 24) as u8;
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let b = (state >> 24) as u8;
        coefficients.push(i32::from(b) - i32::from(g));
    }
    coefficients
}

fn generated_coefficients(width: u32, height: u32, seed: u32) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(width as usize * height as usize);
    let mut state = seed ^ 0x9e37_79b9;
    for idx in 0..width * height {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let value = ((state >> 16) & 0x01ff) as i32 - 255;
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
        &style,
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
        &style,
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
        &style,
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
fn normal_neighborhood_significance_fast_path_returns_unmasked_neighbors() {
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
    assert_eq!(
        ctx.normal_neighborhood_significance_states_index(idx),
        (1 << 6) | 1
    );
}

#[test]
fn normal_sign_context_matches_generic_non_vertical_context() {
    let mut ctx = BitPlaneDecodeContext {
        width: 3,
        height: 3,
        padded_width: 5,
        style: CodeBlockStyle::default(),
        ..BitPlaneDecodeContext::default()
    };
    let len = ctx.padded_width as usize * (ctx.height as usize + 2);
    ctx.coefficients.resize(len, Coefficient::default());
    ctx.neighbor_significances
        .resize(len, NeighborSignificances::default());

    let y = 1;
    let idx = (y + COEFFICIENTS_PADDING as usize) * ctx.padded_width as usize
        + COEFFICIENTS_PADDING as usize
        + 1;
    let padded_width = ctx.padded_width as usize;

    ctx.neighbor_significances[idx].set_top();
    ctx.neighbor_significances[idx].set_left();
    ctx.neighbor_significances[idx].set_right();
    ctx.neighbor_significances[idx].set_bottom();
    ctx.set_sign_index(idx - padded_width, 1);
    ctx.set_sign_index(idx - 1, 0);
    ctx.set_sign_index(idx + 1, 1);
    ctx.set_sign_index(idx + padded_width, 0);

    assert_eq!(
        context_label_sign_coding_index_normal(idx, &ctx),
        context_label_sign_coding_index(idx, y, &ctx)
    );
}

#[test]
fn normal_set_significant_index_matches_generic_neighbor_updates() {
    let mut generic = BitPlaneDecodeContext {
        width: 3,
        height: 3,
        padded_width: 5,
        significant_scan_masks: vec![0; 3],
        zero_coding_scan_masks: vec![0; 3],
        style: CodeBlockStyle::default(),
        ..BitPlaneDecodeContext::default()
    };
    let len = generic.padded_width as usize * (generic.height as usize + 2);
    generic
        .coefficient_states
        .resize(len, CoefficientState::default());
    generic
        .neighbor_significances
        .resize(len, NeighborSignificances::default());
    let mut normal = BitPlaneDecodeContext {
        width: generic.width,
        height: generic.height,
        padded_width: generic.padded_width,
        style: generic.style,
        coefficient_states: generic.coefficient_states.clone(),
        significant_scan_masks: vec![0; 3],
        zero_coding_scan_masks: vec![0; 3],
        neighbor_significances: generic.neighbor_significances.clone(),
        ..BitPlaneDecodeContext::default()
    };

    let padded_width = generic.padded_width as usize;
    let idx =
        (1 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 1;

    generic.set_significant_index(idx, padded_width);
    normal.set_significant_index_normal(idx, padded_width);

    assert_eq!(
        normal.coefficient_states[idx].0,
        generic.coefficient_states[idx].0
    );
    assert_eq!(
        normal
            .neighbor_significances
            .iter()
            .map(|neighbors| neighbors.0)
            .collect::<Vec<_>>(),
        generic
            .neighbor_significances
            .iter()
            .map(|neighbors| neighbors.0)
            .collect::<Vec<_>>()
    );
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
    ctx.reset_for_job(5, 6, 0, 4, SubBandType::LowLow, &style, 8, true)
        .expect("reset context");

    let padded_width = ctx.padded_width as usize;
    let y = 4usize;
    let x = 3usize;
    let idx =
        (y + COEFFICIENTS_PADDING as usize) * padded_width + x + COEFFICIENTS_PADDING as usize;
    let scan_unit = (y >> 2) * ctx.width as usize + x;
    let bit = 1u8 << (y & 3);

    ctx.set_significant_index(idx, padded_width);
    ctx.set_zero_coding_index(idx, padded_width);

    assert_eq!(ctx.significant_scan_masks[scan_unit], bit);
    assert_eq!(ctx.zero_coding_scan_masks[scan_unit], bit);

    ctx.reset_for_next_bitplane();

    assert_eq!(ctx.significant_scan_masks[scan_unit], bit);
    assert_eq!(ctx.zero_coding_scan_masks[scan_unit], 0);
}

#[test]
fn cleanup_candidate_mask_excludes_significant_and_zero_coded_coefficients() {
    let mut ctx = BitPlaneDecodeContext::default();
    let style = CodeBlockStyle::default();
    ctx.reset_for_job(5, 6, 0, 4, SubBandType::LowLow, &style, 8, true)
        .expect("reset context");

    let padded_width = ctx.padded_width as usize;
    let significant_idx =
        (1 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 2;
    let zero_coded_idx =
        (3 + COEFFICIENTS_PADDING as usize) * padded_width + COEFFICIENTS_PADDING as usize + 2;
    let scan_unit = 2;

    ctx.set_significant_index(significant_idx, padded_width);
    ctx.set_zero_coding_index(zero_coded_idx, padded_width);

    assert_eq!(cleanup_candidate_scan_mask(&ctx, scan_unit, 4), 0b0101);
    assert_eq!(cleanup_candidate_scan_mask(&ctx, scan_unit, 2), 0b0001);
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
                0x4a32_1000 + style_idx as u32 * 17 + subband_idx as u32,
            );
        }
    }
}
