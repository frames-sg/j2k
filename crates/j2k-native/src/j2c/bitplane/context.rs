// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::build::SubBandType;
use super::state::{BitPlaneDecodeContext, HAS_MAGNITUDE_REFINEMENT_MASK};
use j2k_codec_math::classic::{
    SIGN_CONTEXT_LOOKUP, ZERO_CTX_HH_LOOKUP, ZERO_CTX_HL_LOOKUP, ZERO_CTX_LL_LH_LOOKUP,
};

/// Based on Table D.2.
#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_sign_coding_index(
    idx: usize,
    y: usize,
    ctx: &BitPlaneDecodeContext,
) -> (u8, u8) {
    // A lot of subtleties go into this path. We need the significances and
    // signs of the four cardinal neighbors and then assign a context label
    // based on the signed sum, without branching on each neighbor.
    let significances = ctx.neighborhood_significance_states_index(idx, y) & 0b0101_0101;
    let padded_width = ctx.padded_width as usize;

    let top_sign = ctx.sign_index(idx - padded_width);
    let left_sign = ctx.sign_index(idx - 1);
    let right_sign = ctx.sign_index(idx + 1);
    let bottom_sign = if ctx.style.vertically_causal_context && ctx.neighbor_in_next_stripe_y(y) {
        0
    } else {
        ctx.sign_index(idx + padded_width)
    };

    // Due to the specific layout of `NeighborSignificances`, direct neighbors
    // and diagonals are interleaved. Therefore, we create a new bit-packed
    // representation that indicates whether the top/left/right/bottom sign is
    // positive, negative, or insignificant. We need two bits for this.
    // 00 represents insignificant, 01 positive and 10 negative.
    let signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;
    let negative_significances = significances & signs;
    let positive_significances = significances & !signs;
    let merged_significances = (negative_significances << 1) | positive_significances;

    SIGN_CONTEXT_LOOKUP[merged_significances as usize]
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_sign_coding_index_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    idx: usize,
    y: usize,
    ctx: &BitPlaneDecodeContext,
) -> (u8, u8) {
    if NORMAL_NEIGHBORS {
        context_label_sign_coding_index_normal(idx, ctx)
    } else {
        context_label_sign_coding_index(idx, y, ctx)
    }
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_sign_coding_index_normal(
    idx: usize,
    ctx: &BitPlaneDecodeContext,
) -> (u8, u8) {
    let significances = ctx.normal_neighborhood_significance_states_index(idx) & 0b0101_0101;
    let padded_width = ctx.padded_width as usize;

    let top_sign = ctx.sign_index(idx - padded_width);
    let left_sign = ctx.sign_index(idx - 1);
    let right_sign = ctx.sign_index(idx + 1);
    let bottom_sign = ctx.sign_index(idx + padded_width);

    let signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;
    let negative_significances = significances & signs;
    let positive_significances = significances & !signs;
    let merged_significances = (negative_significances << 1) | positive_significances;

    SIGN_CONTEXT_LOOKUP[merged_significances as usize]
}

/// Return the context label for zero coding (Section D.3.1).
#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_zero_coding_from_neighbors(
    neighbors: u8,
    sub_band_type: SubBandType,
) -> u8 {
    // Once again, the neighbors field is bit-packed, so we can just generate
    // a table for all u8 values and assign the correct context based on the
    // exact value of that field.
    match sub_band_type {
        SubBandType::LowLow | SubBandType::LowHigh => ZERO_CTX_LL_LH_LOOKUP[neighbors as usize],
        SubBandType::HighLow => ZERO_CTX_HL_LOOKUP[neighbors as usize],
        SubBandType::HighHigh => ZERO_CTX_HH_LOOKUP[neighbors as usize],
    }
}

/// Return the context label for magnitude refinement coding (Table D.4).
#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_magnitude_refinement_coding_from_state(state: u8, neighbors: u8) -> u8 {
    context_label_magnitude_refinement_coding_from_state_lazy(state, || neighbors)
}

#[expect(
    clippy::inline_always,
    reason = "Tier-1 context lookup is a measured coefficient-loop hot path"
)]
#[inline(always)]
pub(super) fn context_label_magnitude_refinement_coding_from_state_lazy(
    state: u8,
    neighbors: impl FnOnce() -> u8,
) -> u8 {
    // If magnitude refined, then 16.
    if state & HAS_MAGNITUDE_REFINEMENT_MASK != 0 {
        16
    } else {
        // Else: If at least one neighbor is significant then 15, else 14.
        14 + neighbors().min(1)
    }
}
