// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::arithmetic_decoder::ArithmeticDecoder;
use super::context::{
    context_label_magnitude_refinement_coding_from_state_lazy,
    context_label_sign_coding_index_with_neighbors, context_label_zero_coding_from_neighbors,
};
use super::state::{
    BitPlaneDecodeContext, COEFFICIENTS_PADDING, HAS_MAGNITUDE_REFINEMENT_MASK,
    HAS_ZERO_CODING_MASK, SIGNIFICANCE_MASK,
};

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn decode_sign_bit_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    idx: usize,
    y: usize,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let (ctx_label, xor_bit) =
        context_label_sign_coding_index_with_neighbors::<NORMAL_NEIGHBORS>(idx, y, ctx);
    let sign_bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label)) ^ u32::from(xor_bit);
    ctx.set_sign_index(idx, u8::from(sign_bit != 0));
}

pub(super) fn cleanup_pass_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for (stripe, base_y) in (0..height).step_by(4).enumerate() {
        let y_end = (base_y + 4).min(height);
        let stripe_height = y_end - base_y;
        let valid_mask = scan_unit_valid_mask(stripe_height);
        let scan_unit_row = stripe * width;

        for x in 0..width {
            let scan_unit = scan_unit_row + x;
            let candidate_mask = cleanup_candidate_scan_mask(ctx, scan_unit, stripe_height);
            if candidate_mask == 0 {
                continue;
            }

            let top_idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            if candidate_mask == valid_mask
                && stripe_height == 4
                && cleanup_run_length_candidate_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx,
                    top_idx,
                    padded_width,
                    base_y,
                )
            {
                // The four contiguous samples are all cleanup candidates
                // with zero context, so Annex D permits the RLC context.
                let bit = decoder.read_bit(ctx.arithmetic_decoder_context(17));
                if bit == 0 {
                    continue;
                }

                let first_significant = (decoder.read_bit(ctx.arithmetic_decoder_context(18)) << 1)
                    | decoder.read_bit(ctx.arithmetic_decoder_context(18));
                let first_significant = first_significant as usize;
                let significant_y = base_y + first_significant;
                let significant_idx = top_idx + first_significant * padded_width;
                ctx.push_magnitude_bit_index(significant_idx, 1);
                decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    significant_idx,
                    significant_y,
                    ctx,
                    decoder,
                );
                ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(
                    significant_idx,
                    padded_width,
                );

                let mut idx = significant_idx + padded_width;
                for y in significant_y + 1..y_end {
                    cleanup_coefficient_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                        ctx,
                        decoder,
                        idx,
                        y,
                        padded_width,
                    );
                    idx += padded_width;
                }
                continue;
            }

            let mut mask = candidate_mask;
            while mask != 0 {
                let bit_y = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let y = base_y + bit_y;
                let idx = top_idx + bit_y * padded_width;
                cleanup_coefficient_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx,
                    decoder,
                    idx,
                    y,
                    padded_width,
                );
            }
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
pub(super) fn cleanup_candidate_scan_mask(
    ctx: &BitPlaneDecodeContext,
    scan_unit: usize,
    stripe_height: usize,
) -> u8 {
    scan_unit_valid_mask(stripe_height)
        & !(ctx.significant_scan_masks[scan_unit] | ctx.zero_coding_scan_masks[scan_unit])
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn scan_unit_valid_mask(stripe_height: usize) -> u8 {
    (1u8 << stripe_height) - 1
}

pub(super) fn significance_propagation_pass_arithmetic_with_neighbors<
    const NORMAL_NEIGHBORS: bool,
>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for base_y in (0..height).step_by(4) {
        let y_end = (base_y + 4).min(height);
        for x in 0..width {
            let mut idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            for y in base_y..y_end {
                let state = ctx.coefficient_states[idx].0;
                let neighbors =
                    neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y);

                // "The significance propagation pass only includes bits of coefficients
                // that were insignificant (the significance state has yet to be set)
                // and have a non-zero context."
                if state & SIGNIFICANCE_MASK == 0 && neighbors != 0 {
                    let ctx_label =
                        context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                    ctx.push_magnitude_bit_index(idx, bit);
                    ctx.set_zero_coding_index(idx, padded_width);

                    // "If the value of this bit is 1 then the significance
                    // state is set to 1 and the immediate next bit to be decoded is
                    // the sign bit for the coefficient. Otherwise, the significance
                    // state remains 0."
                    if bit == 1 {
                        decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                            idx, y, ctx, decoder,
                        );
                        ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(idx, padded_width);
                    }
                }

                idx += padded_width;
            }
        }
    }
}

pub(super) fn magnitude_refinement_pass_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let width = ctx.width as usize;
    let height = ctx.height as usize;
    let padded_width = ctx.padded_width as usize;

    for (stripe, base_y) in (0..height).step_by(4).enumerate() {
        let stripe_height = (base_y + 4).min(height) - base_y;
        let scan_unit_row = stripe * width;

        for x in 0..width {
            let mut mask = ctx.significant_scan_masks[scan_unit_row + x]
                & !ctx.zero_coding_scan_masks[scan_unit_row + x];
            if mask == 0 {
                continue;
            }

            let top_idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                + x
                + COEFFICIENTS_PADDING as usize;

            while mask != 0 {
                let bit_y = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if bit_y >= stripe_height {
                    continue;
                }

                let y = base_y + bit_y;
                let idx = top_idx + bit_y * padded_width;
                let state = ctx.coefficient_states[idx].0;

                debug_assert!(state & SIGNIFICANCE_MASK != 0);
                debug_assert!(state & HAS_ZERO_CODING_MASK == 0);

                let ctx_label =
                    context_label_magnitude_refinement_coding_from_state_lazy(state, || {
                        neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y)
                    });
                let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                ctx.push_magnitude_bit_index(idx, bit);
                ctx.coefficient_states[idx].0 |= HAS_MAGNITUDE_REFINEMENT_MASK;
            }
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn neighborhood_significance_states_for_path<const NORMAL_NEIGHBORS: bool>(
    ctx: &BitPlaneDecodeContext,
    idx: usize,
    y: usize,
) -> u8 {
    if NORMAL_NEIGHBORS {
        ctx.normal_neighborhood_significance_states_index(idx)
    } else {
        ctx.neighborhood_significance_states_index(idx, y)
    }
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn cleanup_run_length_candidate_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &BitPlaneDecodeContext,
    top_idx: usize,
    padded_width: usize,
    base_y: usize,
) -> bool {
    let mut idx = top_idx;
    for y in base_y..base_y + 4 {
        if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) != 0
            || neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y) != 0
        {
            return false;
        }
        idx += padded_width;
    }

    true
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn cleanup_coefficient_arithmetic_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    idx: usize,
    y: usize,
    padded_width: usize,
) {
    if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) == 0 {
        let neighbors = neighborhood_significance_states_for_path::<NORMAL_NEIGHBORS>(ctx, idx, y);
        let ctx_label = context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
        ctx.push_magnitude_bit_index(idx, bit);

        if bit == 1 {
            decode_sign_bit_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(idx, y, ctx, decoder);
            ctx.set_significant_index_for_path::<NORMAL_NEIGHBORS>(idx, padded_width);
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
pub(super) fn cleanup_run_length_candidate(
    ctx: &BitPlaneDecodeContext,
    top_idx: usize,
    padded_width: usize,
    base_y: usize,
) -> bool {
    cleanup_run_length_candidate_with_neighbors::<false>(ctx, top_idx, padded_width, base_y)
}
