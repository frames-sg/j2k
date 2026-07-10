// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};
use core::mem::size_of;

use super::super::arithmetic_decoder::ArithmeticDecoderContext;
use super::super::build::{CodeBlock, SubBandType};
use super::super::codestream::CodeBlockStyle;
use crate::error::{bail, DecodingError, Result};

// JPEG 2000 Part 1 permits up to 38 sample bits; keep additional headroom for
// guard bits and ROI-shifted code-block magnitudes while reserving one sign bit.
#[expect(clippy::cast_possible_truncation, reason = "u64 is eight bytes")]
pub(crate) const BITPLANE_BIT_SIZE: u32 = size_of::<u64>() as u32 * 8 - 1;

pub(super) const HAS_MAGNITUDE_REFINEMENT_SHIFT: u8 = 6;
pub(super) const HAS_ZERO_CODING_SHIFT: u8 = 5;
pub(super) const SIGNIFICANCE_MASK: u8 = 1 << 7;
pub(super) const HAS_MAGNITUDE_REFINEMENT_MASK: u8 = 1 << HAS_MAGNITUDE_REFINEMENT_SHIFT;
pub(super) const HAS_ZERO_CODING_MASK: u8 = 1 << HAS_ZERO_CODING_SHIFT;

/// Bit-packed coefficient state (only 3 bits used):
/// - Bit 7: significance state (set when first non-zero bit is encountered)
/// - Bit 6: has had magnitude refinement pass
/// - Bit 5: zero coded in current bitplane's significance propagation pass
#[derive(Default, Copy, Clone)]
pub(crate) struct CoefficientState(pub(super) u8);

impl CoefficientState {
    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_significant(&mut self) {
        self.0 |= SIGNIFICANCE_MASK;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn is_significant(self) -> bool {
        self.0 & SIGNIFICANCE_MASK != 0
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct Coefficient(pub(super) u64);

impl Coefficient {
    #[cfg(test)]
    #[expect(clippy::trivially_copy_pass_by_ref, reason = "stable accessor")]
    pub(crate) fn get(&self) -> i32 {
        i32::try_from(
            self.get_i64()
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .expect("coefficient is clamped to the i32 range")
    }

    #[expect(clippy::trivially_copy_pass_by_ref, reason = "stable accessor")]
    pub(crate) fn get_i64(&self) -> i64 {
        let mut magnitude = (self.0 & !(1_u64 << 63)).cast_signed();
        // Map sign (0 for positive, 1 for negative) to 1, -1.
        magnitude *= 1 - 2 * i64::from(self.sign() != 0);

        magnitude
    }

    pub(super) fn set_sign(&mut self, sign: u8) {
        self.0 |= u64::from(sign) << 63;
    }

    pub(super) fn sign(self) -> u64 {
        (self.0 >> 63) & 1
    }

    pub(super) fn push_bit_at(&mut self, bit: u32, position: u8) {
        self.0 |= u64::from(bit) << position;
    }
}

pub(super) const COEFFICIENTS_PADDING: u32 = 1;

/// Store the significances of each neighbor for a specific coefficient.
/// The order from MSB to LSB is as follows:
///
/// top-left, top, top-right, left, bottom-left, right, bottom-right, bottom.
///
/// See the `context_label_sign_coding` method for why we aren't simply using
/// row-major order.
#[derive(Default, Copy, Clone)]
pub(super) struct NeighborSignificances(pub(super) u8);

impl NeighborSignificances {
    pub(super) fn set_top_left(&mut self) {
        self.0 |= 1 << 7;
    }

    pub(super) fn set_top(&mut self) {
        self.0 |= 1 << 6;
    }

    pub(super) fn set_top_right(&mut self) {
        self.0 |= 1 << 5;
    }

    pub(super) fn set_left(&mut self) {
        self.0 |= 1 << 4;
    }

    pub(super) fn set_bottom_left(&mut self) {
        self.0 |= 1 << 3;
    }

    pub(super) fn set_right(&mut self) {
        self.0 |= 1 << 2;
    }

    pub(super) fn set_bottom_right(&mut self) {
        self.0 |= 1 << 1;
    }

    pub(super) fn set_bottom(&mut self) {
        self.0 |= 1;
    }

    pub(super) fn all(self) -> u8 {
        self.0
    }

    // Needed for vertically causal context.
    pub(super) fn all_without_bottom(self) -> u8 {
        self.0 & 0b1111_0100
    }
}

#[derive(Default)]
pub(crate) struct BitPlaneDecodeBuffers {
    pub(super) combined_layers: Vec<u8>,
    pub(super) segment_ranges: Vec<usize>,
    pub(super) segment_coding_passes: Vec<u8>,
}

impl BitPlaneDecodeBuffers {
    pub(super) fn reset(&mut self) {
        self.combined_layers.clear();
        self.segment_ranges.clear();
        self.segment_coding_passes.clear();

        // The design of these two buffers is that the ranges are stored
        // as [idx, idx + 1), so we need to store the first 0 when resetting.
        self.segment_ranges.push(0);
        self.segment_coding_passes.push(0);
    }
}

pub(crate) struct BitPlaneDecodeContext {
    /// A vector of bit-packed fields for each coefficient in the code-block.
    pub(super) coefficient_states: Vec<super::CoefficientState>,
    /// One 4-bit mask per scan stripe column for coefficients that are significant.
    pub(super) significant_scan_masks: Vec<u8>,
    /// One 4-bit mask per scan stripe column for zero-coded coefficients in this bitplane.
    pub(super) zero_coding_scan_masks: Vec<u8>,
    /// The neighbor significances for each coefficient.
    pub(super) neighbor_significances: Vec<NeighborSignificances>,
    /// The magnitude and signs of each coefficient that is successively built
    /// as we advance through the bitplanes.
    pub(super) coefficients: Vec<super::Coefficient>,
    /// The width of the code-block we are processing.
    pub(super) width: u32,
    /// The width of the code-block we are processing, with padding.
    pub(super) padded_width: u32,
    /// The height of the code-block we are processing.
    pub(super) height: u32,
    /// The code-block style for the current code-block.
    pub(super) style: CodeBlockStyle,
    /// The number of bitplanes (minus implicitly missing bitplanes) to decode.
    pub(super) bitplanes: u8,
    /// Whether strict mode is enabled.
    pub(super) strict: bool,
    /// The maximum number of coding passes to process.
    pub(super) max_coding_passes: u8,
    /// The type of sub-band the current code block belongs to.
    pub(super) sub_band_type: SubBandType,
    /// The arithmetic decoder contexts for each context label.
    pub(super) contexts: [ArithmeticDecoderContext; 19],
    /// The bit position for the current bitplane.
    pub(super) current_bit_position: u8,
}

impl Default for BitPlaneDecodeContext {
    fn default() -> Self {
        Self {
            coefficient_states: vec![],
            significant_scan_masks: vec![],
            zero_coding_scan_masks: vec![],
            coefficients: vec![],
            neighbor_significances: vec![],
            width: 0,
            padded_width: COEFFICIENTS_PADDING * 2,
            height: 0,
            style: CodeBlockStyle::default(),
            bitplanes: 0,
            max_coding_passes: 0,
            strict: false,
            sub_band_type: SubBandType::LowLow,
            contexts: [ArithmeticDecoderContext::default(); 19],
            current_bit_position: 0,
        }
    }
}

impl BitPlaneDecodeContext {
    #[expect(
        clippy::too_many_arguments,
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable reset boundary mirrors validated codestream job fields explicitly"
    )]
    pub(super) fn reset_for_job(
        &mut self,
        width: u32,
        height: u32,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        sub_band_type: SubBandType,
        code_block_style: &CodeBlockStyle,
        total_bitplanes: u8,
        strict: bool,
    ) -> Result<()> {
        let padded_width = width + COEFFICIENTS_PADDING * 2;
        let padded_height = height + COEFFICIENTS_PADDING * 2;
        let num_coefficients = padded_width as usize * padded_height as usize;

        self.coefficients.clear();
        self.coefficients
            .resize(num_coefficients, Coefficient::default());

        self.neighbor_significances.clear();
        self.neighbor_significances
            .resize(num_coefficients, NeighborSignificances::default());

        self.coefficient_states.clear();
        self.coefficient_states
            .resize(num_coefficients, CoefficientState::default());

        let scan_units = width as usize * height.div_ceil(4) as usize;
        self.significant_scan_masks.clear();
        self.significant_scan_masks.resize(scan_units, 0);
        self.zero_coding_scan_masks.clear();
        self.zero_coding_scan_masks.resize(scan_units, 0);

        self.width = width;
        self.padded_width = padded_width;
        self.height = height;
        self.sub_band_type = sub_band_type;
        self.style = *code_block_style;
        self.reset_contexts();

        self.bitplanes = if strict {
            total_bitplanes
                .checked_sub(missing_bit_planes)
                .ok_or(DecodingError::InvalidBitplaneCount)?
        } else {
            total_bitplanes.saturating_sub(missing_bit_planes)
        };

        self.max_coding_passes = if self.bitplanes == 0 {
            0
        } else {
            1 + 3 * (self.bitplanes - 1)
        };

        if self.max_coding_passes < number_of_coding_passes && strict {
            bail!(DecodingError::TooManyCodingPasses);
        }

        self.strict = strict;

        Ok(())
    }

    /// Completely reset context so that it can be reused for a new code-block.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "stable borrowed style boundary"
    )]
    pub(crate) fn reset(
        &mut self,
        code_block: &CodeBlock,
        sub_band_type: SubBandType,
        code_block_style: &CodeBlockStyle,
        total_bitplanes: u8,
        strict: bool,
    ) -> Result<()> {
        self.reset_for_job(
            code_block.rect.width(),
            code_block.rect.height(),
            code_block.missing_bit_planes,
            code_block.number_of_coding_passes,
            sub_band_type,
            code_block_style,
            total_bitplanes,
            strict,
        )
    }

    pub(crate) fn coefficient_rows(&self) -> impl Iterator<Item = &[Coefficient]> {
        self.coefficients
            .chunks_exact(self.padded_width as usize)
            // Exclude the padding that we added.
            .map(|row| &row[COEFFICIENTS_PADDING as usize..][..self.width as usize])
            .skip(COEFFICIENTS_PADDING as usize)
            .take(self.height as usize)
    }

    pub(super) fn arithmetic_decoder_context(
        &mut self,
        ctx_label: u8,
    ) -> &mut ArithmeticDecoderContext {
        &mut self.contexts[ctx_label as usize]
    }

    /// Reset each context to the initial state defined in table D.7.
    pub(super) fn reset_contexts(&mut self) {
        for context in &mut self.contexts {
            context.reset();
        }

        self.contexts[0].reset_with_index(4);
        self.contexts[17].reset_with_index(3);
        self.contexts[18].reset_with_index(46);
    }

    /// Reset state that is transient for each bitplane that is decoded.
    pub(super) fn reset_for_next_bitplane(&mut self) {
        let padded_width = self.padded_width as usize;
        let width = self.width as usize;
        let row_start = COEFFICIENTS_PADDING as usize;

        for row in self
            .coefficient_states
            .chunks_exact_mut(padded_width)
            .skip(COEFFICIENTS_PADDING as usize)
            .take(self.height as usize)
        {
            for state in &mut row[row_start..row_start + width] {
                state.0 &= !HAS_ZERO_CODING_MASK;
            }
        }
        self.zero_coding_scan_masks.fill(0);
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_sign_index(&mut self, idx: usize, sign: u8) {
        self.coefficients[idx].set_sign(sign);
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_significant_index(&mut self, idx: usize, padded_width: usize) {
        let is_significant = self.coefficient_states[idx].is_significant();

        if !is_significant {
            self.coefficient_states[idx].set_significant();
            self.set_significant_scan_mask(idx, padded_width);

            // Update all neighbors so they know this coefficient is significant
            // now.
            self.neighbor_significances[idx - padded_width - 1].set_bottom_right();
            self.neighbor_significances[idx - padded_width].set_bottom();
            self.neighbor_significances[idx - padded_width + 1].set_bottom_left();
            self.neighbor_significances[idx - 1].set_right();
            self.neighbor_significances[idx + 1].set_left();
            self.neighbor_significances[idx + padded_width - 1].set_top_right();
            self.neighbor_significances[idx + padded_width].set_top();
            self.neighbor_significances[idx + padded_width + 1].set_top_left();
        }
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_significant_index_for_path<const NORMAL_NEIGHBORS: bool>(
        &mut self,
        idx: usize,
        padded_width: usize,
    ) {
        if NORMAL_NEIGHBORS {
            self.set_significant_index_normal(idx, padded_width);
        } else {
            self.set_significant_index(idx, padded_width);
        }
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_significant_index_normal(&mut self, idx: usize, padded_width: usize) {
        if self.coefficient_states[idx].is_significant() {
            return;
        }

        self.coefficient_states[idx].set_significant();
        self.set_significant_scan_mask(idx, padded_width);

        let top_start = idx - padded_width - 1;
        let top = &mut self.neighbor_significances[top_start..top_start + 3];
        top[0].set_bottom_right();
        top[1].set_bottom();
        top[2].set_bottom_left();

        let middle_start = idx - 1;
        let middle = &mut self.neighbor_significances[middle_start..middle_start + 3];
        middle[0].set_right();
        middle[2].set_left();

        let bottom_start = idx + padded_width - 1;
        let bottom = &mut self.neighbor_significances[bottom_start..bottom_start + 3];
        bottom[0].set_top_right();
        bottom[1].set_top();
        bottom[2].set_top_left();
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn push_magnitude_bit_index(&mut self, idx: usize, bit: u32) {
        self.coefficients[idx].push_bit_at(bit, self.current_bit_position);
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_zero_coding_index(&mut self, idx: usize, padded_width: usize) {
        self.coefficient_states[idx].0 |= HAS_ZERO_CODING_MASK;
        let (scan_unit, bit) = self.scan_unit_mask_index(idx, padded_width);
        self.zero_coding_scan_masks[scan_unit] |= bit;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn set_significant_scan_mask(&mut self, idx: usize, padded_width: usize) {
        let (scan_unit, bit) = self.scan_unit_mask_index(idx, padded_width);
        self.significant_scan_masks[scan_unit] |= bit;
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn scan_unit_mask_index(&self, idx: usize, padded_width: usize) -> (usize, u8) {
        let row = idx / padded_width;
        let col = idx - row * padded_width;
        let pad = COEFFICIENTS_PADDING as usize;
        debug_assert!(row >= pad);
        debug_assert!(col >= pad);

        let y = row - pad;
        let x = col - pad;
        debug_assert!(y < self.height as usize);
        debug_assert!(x < self.width as usize);

        let scan_unit = (y >> 2) * self.width as usize + x;
        (scan_unit, 1u8 << (y & 3))
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn sign_index(&self, idx: usize) -> u8 {
        u8::from(self.coefficients[idx].sign() != 0)
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn neighbor_in_next_stripe_y(&self, y: usize) -> bool {
        let neighbor_y = y + 1;
        neighbor_y < self.height as usize && (neighbor_y >> 2) > (y >> 2)
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn neighborhood_significance_states_index(&self, idx: usize, y: usize) -> u8 {
        let neighbors = &self.neighbor_significances[idx];

        if self.style.vertically_causal_context && self.neighbor_in_next_stripe_y(y) {
            neighbors.all_without_bottom()
        } else {
            neighbors.all()
        }
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn normal_neighborhood_significance_states_index(&self, idx: usize) -> u8 {
        self.neighbor_significances[idx].all()
    }

    #[expect(clippy::inline_always, reason = "Tier-1 coefficient-loop hot path")]
    #[inline(always)]
    pub(super) fn uses_normal_arithmetic_neighbor_path(&self) -> bool {
        !self.style.selective_arithmetic_coding_bypass
            && !self.style.termination_on_each_pass
            && !self.style.vertically_causal_context
    }
}
