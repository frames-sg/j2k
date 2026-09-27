// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::arithmetic_decoder::ArithmeticDecoderContext;
use super::super::build::{CodeBlock, SubBandType};
use super::super::codestream::CodeBlockStyle;
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::try_reserve_decode_elements;

mod model;
mod scan_masks;
mod workspace;

pub(crate) use model::{Coefficient, CoefficientState, BITPLANE_BIT_SIZE};
pub(super) use model::{
    NeighborSignificances, COEFFICIENTS_PADDING, HAS_MAGNITUDE_REFINEMENT_MASK,
    HAS_ZERO_CODING_MASK, SIGNIFICANCE_MASK,
};
pub(crate) use workspace::classic_decode_workspace_bytes;

#[derive(Default)]
pub(crate) struct BitPlaneDecodeBuffers {
    pub(super) combined_layers: Vec<u8>,
    pub(super) segment_ranges: Vec<usize>,
    pub(super) segment_coding_passes: Vec<u8>,
}

impl BitPlaneDecodeBuffers {
    pub(crate) fn prepare(&mut self, data_len: usize, boundary_len: usize) -> Result<()> {
        self.combined_layers.clear();
        self.segment_ranges.clear();
        self.segment_coding_passes.clear();
        try_reserve_decode_elements(&mut self.combined_layers, data_len)?;
        try_reserve_decode_elements(&mut self.segment_ranges, boundary_len)?;
        try_reserve_decode_elements(&mut self.segment_coding_passes, boundary_len)
    }

    pub(super) fn reset(&mut self) -> Option<()> {
        self.combined_layers.clear();
        self.segment_ranges.clear();
        self.segment_coding_passes.clear();

        // The design of these two buffers is that the ranges are stored
        // as [idx, idx + 1), so we need to store the first 0 when resetting.
        push_preallocated(&mut self.segment_ranges, 0)?;
        push_preallocated(&mut self.segment_coding_passes, 0)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        include_capacity::<u8>(&mut bytes, self.combined_layers.capacity())?;
        include_capacity::<usize>(&mut bytes, self.segment_ranges.capacity())?;
        include_capacity::<u8>(&mut bytes, self.segment_coding_passes.capacity())?;
        Ok(bytes)
    }

    #[cfg(test)]
    pub(crate) fn combined_layers_owner_for_test(&self) -> (*const u8, usize) {
        (
            self.combined_layers.as_ptr(),
            self.combined_layers.capacity(),
        )
    }
}

pub(super) fn push_preallocated<T>(values: &mut Vec<T>, value: T) -> Option<()> {
    if values.len() == values.capacity() {
        return None;
    }
    values.push(value);
    Some(())
}

pub(super) fn extend_preallocated<T: Copy>(values: &mut Vec<T>, source: &[T]) -> Option<()> {
    let required_len = values.len().checked_add(source.len())?;
    if required_len > values.capacity() {
        return None;
    }
    values.extend_from_slice(source);
    Some(())
}

fn include_capacity<T>(bytes: &mut usize, capacity: usize) -> Result<()> {
    let additional = capacity
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(())
}

pub(crate) struct BitPlaneDecodeContext {
    /// A vector of bit-packed fields for each coefficient in the code-block.
    pub(super) coefficient_states: Vec<super::CoefficientState>,
    /// One 4-bit mask per scan stripe column for coefficients that are significant.
    pub(super) significant_scan_masks: Vec<u8>,
    /// One 4-bit mask per scan stripe column for zero-coded coefficients in this bitplane.
    pub(super) zero_coding_scan_masks: Vec<u8>,
    /// Packed per-stripe-column state (see `flags.rs`) used instead of the
    /// per-coefficient state, neighbor, and scan-mask arrays when
    /// `packed_columns` is set. One padding column and stripe on each side.
    pub(super) flags: Vec<u32>,
    /// Whether this block decodes through the packed-column passes: normal
    /// neighbor contexts and arithmetic coding throughout.
    pub(super) packed_columns: bool,
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
        Self::empty()
    }
}

impl BitPlaneDecodeContext {
    pub(crate) const fn empty() -> Self {
        Self {
            coefficient_states: Vec::new(),
            significant_scan_masks: Vec::new(),
            zero_coding_scan_masks: Vec::new(),
            flags: Vec::new(),
            packed_columns: false,
            coefficients: Vec::new(),
            neighbor_significances: Vec::new(),
            width: 0,
            padded_width: COEFFICIENTS_PADDING * 2,
            height: 0,
            style: CodeBlockStyle {
                selective_arithmetic_coding_bypass: false,
                reset_context_probabilities: false,
                termination_on_each_pass: false,
                vertically_causal_context: false,
                segmentation_symbols: false,
                high_throughput_block_coding: false,
                mixed_block_coding: false,
            },
            bitplanes: 0,
            max_coding_passes: 0,
            strict: false,
            sub_band_type: SubBandType::LowLow,
            contexts: [ArithmeticDecoderContext::empty(); 19],
            current_bit_position: 0,
        }
    }

    /// Allocate a workspace able to decode `width` x `height` blocks through
    /// either pass family.
    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<()> {
        workspace::reset_decode_buffers(self, width, height, workspace::StateLayout::Both)
            .map(|_| ())
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        include_capacity::<Coefficient>(&mut bytes, self.coefficients.capacity())?;
        include_capacity::<NeighborSignificances>(
            &mut bytes,
            self.neighbor_significances.capacity(),
        )?;
        include_capacity::<CoefficientState>(&mut bytes, self.coefficient_states.capacity())?;
        include_capacity::<u8>(&mut bytes, self.significant_scan_masks.capacity())?;
        include_capacity::<u8>(&mut bytes, self.zero_coding_scan_masks.capacity())?;
        include_capacity::<u32>(&mut bytes, self.flags.capacity())?;
        Ok(bytes)
    }

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
        all_segments_arithmetic: bool,
    ) -> Result<()> {
        // Packed and per-coefficient bookkeeping cannot be mixed in a block,
        // so a raw (bypass) segment sends the whole block down the
        // per-coefficient passes.
        let packed_columns = all_segments_arithmetic
            && !code_block_style.selective_arithmetic_coding_bypass
            && !code_block_style.termination_on_each_pass
            && !code_block_style.vertically_causal_context;
        let layout = if packed_columns {
            workspace::StateLayout::PackedColumns
        } else {
            workspace::StateLayout::PerCoefficient
        };
        let padded_width = workspace::reset_decode_buffers(self, width, height, layout)?;
        self.packed_columns = packed_columns;

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

    #[cfg(test)]
    pub(crate) fn reserve_coefficients_for_test(&mut self, additional: usize) {
        self.coefficients.reserve(additional);
    }

    #[cfg(test)]
    pub(crate) fn coefficient_capacity_for_test(&self) -> usize {
        self.coefficients.capacity()
    }

    #[cfg(test)]
    pub(crate) fn coefficient_ptr_for_test(&self) -> *const Coefficient {
        self.coefficients.as_ptr()
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
            // Codestream blocks take raw segments only under selective
            // bypass, which already selects the per-coefficient passes.
            true,
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

    /// Midpoint reconstruction for this code block's decoded coefficients.
    pub(crate) fn midpoint_reconstructor(
        &self,
        number_of_coding_passes: u8,
        roi_shift: u8,
    ) -> super::reconstruction::MidpointReconstructor {
        super::reconstruction::MidpointReconstructor::new(
            self.bitplanes,
            number_of_coding_passes,
            roi_shift,
        )
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
        // Arithmetic passes select cleanup and refinement coefficients from
        // `zero_coding_scan_masks`, which is reset below. Raw bypass cleanup
        // also reads the per-coefficient marker and therefore still needs it
        // cleared explicitly.
        if self.style.selective_arithmetic_coding_bypass {
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
    pub(super) fn set_significant_index(&mut self, idx: usize, y: usize, padded_width: usize) {
        let is_significant = self.coefficient_states[idx].is_significant();

        if !is_significant {
            self.coefficient_states[idx].set_significant();
            self.set_significant_scan_mask(idx, y, padded_width);

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
    pub(super) fn push_magnitude_bit_index(&mut self, idx: usize, bit: u32) {
        self.coefficients[idx].push_bit_at(bit, self.current_bit_position);
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
    pub(super) fn uses_packed_columns(&self) -> bool {
        self.packed_columns
    }
}
