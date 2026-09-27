// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-coefficient arithmetic-coded passes, used when a code block cannot take
//! the packed-column passes in `flag_passes.rs`: vertically causal contexts,
//! or segments that mix in raw (bypass) coding.

use super::super::arithmetic_decoder::ArithmeticDecoder;
use super::context::{
    context_label_magnitude_refinement_coding_from_state_lazy, context_label_sign_coding_index,
    context_label_zero_coding_from_neighbors,
};
use super::scan::{
    cleanup_candidate_scan_mask, cleanup_run_length_candidate, scan_unit_valid_mask,
};
use super::state::{
    BitPlaneDecodeContext, COEFFICIENTS_PADDING, HAS_MAGNITUDE_REFINEMENT_MASK, SIGNIFICANCE_MASK,
};

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn decode_sign_bit_arithmetic(
    idx: usize,
    y: usize,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
) {
    let (ctx_label, xor_bit) = context_label_sign_coding_index(idx, y, ctx);
    let sign_bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label)) ^ u32::from(xor_bit);
    ctx.set_sign_index(idx, u8::from(sign_bit != 0));
}

pub(super) fn cleanup_pass_arithmetic(
    ctx: &mut BitPlaneDecodeContext,
    shared_decoder: &mut ArithmeticDecoder<'_>,
) {
    // Decode against a local copy: behind the `&mut` the MQ registers were
    // stored and reloaded on every symbol. Written back after the pass.
    let mut local_decoder = shared_decoder.clone();
    let decoder = &mut local_decoder;
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
                && cleanup_run_length_candidate(ctx, top_idx, padded_width, base_y)
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
                decode_sign_bit_arithmetic(significant_idx, significant_y, ctx, decoder);
                ctx.set_significant_index(significant_idx, significant_y, padded_width);

                let mut idx = significant_idx + padded_width;
                for y in significant_y + 1..y_end {
                    cleanup_coefficient_arithmetic(ctx, decoder, idx, y, padded_width);
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
                cleanup_coefficient_arithmetic(ctx, decoder, idx, y, padded_width);
            }
        }
    }
    *shared_decoder = local_decoder;
}

pub(super) fn significance_propagation_pass_arithmetic(
    ctx: &mut BitPlaneDecodeContext,
    shared_decoder: &mut ArithmeticDecoder<'_>,
) {
    // Local MQ copy, as in the cleanup pass.
    let mut local_decoder = shared_decoder.clone();
    let decoder = &mut local_decoder;
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
                let neighbors = ctx.neighborhood_significance_states_index(idx, y);

                // "The significance propagation pass only includes bits of coefficients
                // that were insignificant (the significance state has yet to be set)
                // and have a non-zero context."
                if state & SIGNIFICANCE_MASK == 0 && neighbors != 0 {
                    let ctx_label =
                        context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                    ctx.push_magnitude_bit_index(idx, bit);
                    ctx.set_zero_coding_index(idx, y, padded_width);

                    // "If the value of this bit is 1 then the significance
                    // state is set to 1 and the immediate next bit to be decoded is
                    // the sign bit for the coefficient. Otherwise, the significance
                    // state remains 0."
                    if bit == 1 {
                        decode_sign_bit_arithmetic(idx, y, ctx, decoder);
                        ctx.set_significant_index(idx, y, padded_width);
                    }
                }

                idx += padded_width;
            }
        }
    }
    *shared_decoder = local_decoder;
}

pub(super) fn magnitude_refinement_pass_arithmetic(
    ctx: &mut BitPlaneDecodeContext,
    shared_decoder: &mut ArithmeticDecoder<'_>,
) {
    // Local MQ copy, as in the cleanup pass.
    let mut local_decoder = shared_decoder.clone();
    let decoder = &mut local_decoder;
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

                let ctx_label =
                    context_label_magnitude_refinement_coding_from_state_lazy(state, || {
                        ctx.neighborhood_significance_states_index(idx, y)
                    });
                let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
                ctx.push_magnitude_bit_index(idx, bit);
                ctx.coefficient_states[idx].0 |= HAS_MAGNITUDE_REFINEMENT_MASK;
            }
        }
    }
    *shared_decoder = local_decoder;
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 coefficient helpers are measured inner-loop hot paths"
)]
#[inline(always)]
fn cleanup_coefficient_arithmetic(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    idx: usize,
    y: usize,
    padded_width: usize,
) {
    let neighbors = ctx.neighborhood_significance_states_index(idx, y);
    let ctx_label = context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label));
    ctx.push_magnitude_bit_index(idx, bit);

    if bit == 1 {
        decode_sign_bit_arithmetic(idx, y, ctx, decoder);
        ctx.set_significant_index(idx, y, padded_width);
    }
}
