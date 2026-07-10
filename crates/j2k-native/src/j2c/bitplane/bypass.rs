// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::arithmetic_decoder::ArithmeticDecoderContext;
use super::arithmetic::cleanup_run_length_candidate;
use super::context::{
    context_label_magnitude_refinement_coding_from_state, context_label_sign_coding_index,
    context_label_zero_coding_from_neighbors,
};
use super::state::{
    BitPlaneDecodeContext, COEFFICIENTS_PADDING, HAS_MAGNITUDE_REFINEMENT_MASK,
    HAS_ZERO_CODING_MASK, SIGNIFICANCE_MASK,
};
use crate::reader::BitReader;

// Bypass bit reads can fail in strict mode when the raw segment runs short.
pub(super) trait BitDecoder {
    fn read_bit(&mut self, context: &mut ArithmeticDecoderContext) -> Option<u32>;
}

pub(super) struct BypassDecoder<'a>(BitReader<'a>, bool);

impl<'a> BypassDecoder<'a> {
    pub(super) fn new(data: &'a [u8], strict: bool) -> Self {
        Self(BitReader::new(data), strict)
    }
}

impl BitDecoder for BypassDecoder<'_> {
    fn read_bit(&mut self, _: &mut ArithmeticDecoderContext) -> Option<u32> {
        self.0.read_bits_with_stuffing(1).or({
            if !self.1 {
                // If not in strict mode, just pad with ones. Not sure if
                // zeroes would be better here, but since the arithmetic decoder
                // is also padded with 0xFF maybe 1 is the better choice?
                Some(1)
            } else {
                // We have too little data, return `None`.
                None
            }
        })
    }
}

pub(super) struct SafeScalarTier1;

impl SafeScalarTier1 {
    pub(super) fn cleanup_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
        let width = ctx.width as usize;
        let height = ctx.height as usize;
        let padded_width = ctx.padded_width as usize;

        for base_y in (0..height).step_by(4) {
            let y_end = (base_y + 4).min(height);
            let stripe_height = y_end - base_y;

            for x in 0..width {
                let top_idx = (base_y + COEFFICIENTS_PADDING as usize) * padded_width
                    + x
                    + COEFFICIENTS_PADDING as usize;

                if stripe_height == 4
                    && cleanup_run_length_candidate(ctx, top_idx, padded_width, base_y)
                {
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(17))?;
                    if bit == 0 {
                        continue;
                    }

                    let first_significant =
                        (decoder.read_bit(ctx.arithmetic_decoder_context(18))? << 1)
                            | decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let first_significant = first_significant as usize;
                    let significant_y = base_y + first_significant;
                    let significant_idx = top_idx + first_significant * padded_width;
                    ctx.push_magnitude_bit_index(significant_idx, 1);
                    decode_sign_bit_bypass(significant_idx, significant_y, ctx, decoder)?;
                    ctx.set_significant_index(significant_idx, padded_width);

                    let mut idx = significant_idx + padded_width;
                    for y in significant_y + 1..y_end {
                        cleanup_coefficient_bypass(ctx, decoder, idx, y, padded_width)?;
                        idx += padded_width;
                    }
                    continue;
                }

                let mut idx = top_idx;
                for y in base_y..y_end {
                    cleanup_coefficient_bypass(ctx, decoder, idx, y, padded_width)?;
                    idx += padded_width;
                }
            }
        }

        Some(())
    }

    pub(super) fn significance_propagation_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
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

                    if state & SIGNIFICANCE_MASK == 0 && neighbors != 0 {
                        let ctx_label =
                            context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
                        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
                        ctx.push_magnitude_bit_index(idx, bit);
                        ctx.set_zero_coding_index(idx, padded_width);

                        if bit == 1 {
                            decode_sign_bit_bypass(idx, y, ctx, decoder)?;
                            ctx.set_significant_index(idx, padded_width);
                        }
                    }

                    idx += padded_width;
                }
            }
        }

        Some(())
    }

    pub(super) fn magnitude_refinement_pass_bypass(
        ctx: &mut BitPlaneDecodeContext,
        decoder: &mut BypassDecoder<'_>,
    ) -> Option<()> {
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

                    let neighbors = ctx.neighborhood_significance_states_index(idx, y);
                    let ctx_label =
                        context_label_magnitude_refinement_coding_from_state(state, neighbors);
                    let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
                    ctx.push_magnitude_bit_index(idx, bit);
                    ctx.coefficient_states[idx].0 |= HAS_MAGNITUDE_REFINEMENT_MASK;
                }
            }
        }

        Some(())
    }
}

#[inline(always)]
fn cleanup_coefficient_bypass(
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
    idx: usize,
    y: usize,
    padded_width: usize,
) -> Option<()> {
    if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) == 0 {
        let neighbors = ctx.neighborhood_significance_states_index(idx, y);
        let ctx_label = context_label_zero_coding_from_neighbors(neighbors, ctx.sub_band_type);
        let bit = decoder.read_bit(ctx.arithmetic_decoder_context(ctx_label))?;
        ctx.push_magnitude_bit_index(idx, bit);

        if bit == 1 {
            decode_sign_bit_bypass(idx, y, ctx, decoder)?;
            ctx.set_significant_index(idx, padded_width);
        }
    }

    Some(())
}

/// Decode a raw bypass sign bit (Section D.3.2).
#[inline(always)]
fn decode_sign_bit_bypass(
    idx: usize,
    y: usize,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
) -> Option<()> {
    let (ctx_label, xor_bit) = context_label_sign_coding_index(idx, y, ctx);
    let ad_ctx = ctx.arithmetic_decoder_context(ctx_label);
    let _ = xor_bit;
    let sign_bit = decoder.read_bit(ad_ctx)?;
    ctx.set_sign_index(idx, sign_bit as u8);

    Some(())
}

#[cfg(test)]
mod tests {
    use super::{BitDecoder, BypassDecoder};
    use crate::j2c::arithmetic_decoder::ArithmeticDecoderContext;

    #[test]
    fn bypass_bit_consumption_matches_pre_split_golden() {
        let data = [0xAA, 0xFF, 0x7F, 0x80];
        let mut decoder = BypassDecoder::new(&data, true);
        let mut context = ArithmeticDecoderContext::default();
        let mut bits = 0u32;

        for _ in 0..29 {
            bits = (bits << 1) | decoder.read_bit(&mut context).expect("bypass bit");
        }

        assert_eq!(bits, 0x155F_FFE0);
        assert_eq!(decoder.0.offset(), 3);
        assert_eq!(decoder.0.bit_pos(), 6);
        assert!(!decoder.0.at_end());
    }
}
