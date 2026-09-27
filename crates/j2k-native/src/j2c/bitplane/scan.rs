// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stripe-column scan helpers shared by the arithmetic and bypass passes.

use super::state::{BitPlaneDecodeContext, HAS_ZERO_CODING_MASK, SIGNIFICANCE_MASK};

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
pub(super) fn scan_unit_valid_mask(stripe_height: usize) -> u8 {
    (1u8 << stripe_height) - 1
}

/// Whether a full stripe column qualifies for run-length cleanup coding:
/// no coefficient is significant, zero-coded, or has a significant neighbor.
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
    let mut idx = top_idx;
    for y in base_y..base_y + 4 {
        if ctx.coefficient_states[idx].0 & (SIGNIFICANCE_MASK | HAS_ZERO_CODING_MASK) != 0
            || ctx.neighborhood_significance_states_index(idx, y) != 0
        {
            return false;
        }
        idx += padded_width;
    }
    true
}
