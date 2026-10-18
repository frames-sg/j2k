// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::CodeBlock;
use super::pipeline::prepare_scratch;
use crate::error::{Result, ValidationError};
use crate::{checked_decode_sample_count, try_reserve_decode_elements, try_resize_decode_elements};
use core::mem::size_of;

#[expect(
    clippy::cast_possible_wrap,
    reason = "the sign bit is masked before converting the at-most 31-bit coefficient magnitude"
)]
pub(crate) fn coefficient_to_i32(value: u32, k_max: u8) -> i32 {
    let shift = 31_u32.saturating_sub(u32::from(k_max));
    let magnitude = ((value & 0x7FFF_FFFF) >> shift) as i32;

    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "HT irreversible reconstruction intentionally converts its fixed-point magnitude to the codec f32 domain"
)]
pub(crate) fn coefficient_to_f32(value: u32, k_max: u8) -> f32 {
    let binary_point = 31_i32 - i32::from(k_max);
    let magnitude = (value & 0x7FFF_FFFF) as f32 * crate::math::pow2i(-binary_point);

    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Convert one row of decoded coefficients to dequantized samples: the same
/// result as [`coefficient_to_i32`] (or [`coefficient_to_f32`] for the
/// irreversible midpoint), then the ROI inverse shift, then `* step`. Rows
/// without an ROI shift take branch-free loops that vectorize.
#[expect(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "magnitudes are at most 31 bits; the codec float domain receives them on purpose"
)]
pub(crate) fn store_coefficient_row(
    coefficients: &[u32],
    output: &mut [f32],
    k_max: u8,
    roi_shift: u8,
    irreversible_midpoint: bool,
    step: f32,
) {
    let pairs = output.iter_mut().zip(coefficients.iter().copied());
    if roi_shift != 0 {
        for (sample, coefficient) in pairs {
            *sample = if irreversible_midpoint {
                crate::roi::apply_roi_maxshift_inverse_f32(
                    coefficient_to_f32(coefficient, k_max),
                    roi_shift,
                )
            } else {
                crate::roi::apply_roi_maxshift_inverse_i32(
                    coefficient_to_i32(coefficient, k_max),
                    roi_shift,
                ) as f32
            } * step;
        }
    } else if irreversible_midpoint {
        let scale = crate::math::pow2i(-(31_i32 - i32::from(k_max)));
        for (sample, coefficient) in pairs {
            let magnitude = (coefficient & 0x7FFF_FFFF) as f32 * scale;
            // Setting the sign bit of a non-negative value negates it.
            *sample = f32::from_bits(magnitude.to_bits() | (coefficient & 0x8000_0000)) * step;
        }
    } else {
        let shift = 31_u32.saturating_sub(u32::from(k_max));
        for (sample, coefficient) in pairs {
            let magnitude = ((coefficient & 0x7FFF_FFFF) >> shift) as i32;
            let sign = (coefficient as i32) >> 31;
            *sample = ((magnitude ^ sign) - sign) as f32 * step;
        }
    }
}

mod scratch;
#[cfg(test)]
pub(super) use self::scratch::zeroed_u32_scratch;
pub(super) use self::scratch::{resized_u16_scratch, resized_u32_scratch, zeroed_u16_scratch};

#[derive(Default)]
pub(crate) struct HtBlockDecodeContext {
    pub(super) coefficients: Vec<u32>,
    pub(super) segment_data: Vec<u8>,
    pub(super) scratch: HtBlockDecodeScratch,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl HtBlockDecodeContext {
    pub(crate) fn prepare(
        &mut self,
        width: u32,
        height: u32,
        segment_data_bytes: usize,
    ) -> Result<()> {
        self.width = width;
        self.height = height;
        let coefficient_count = checked_decode_sample_count(width, height)?;
        self.coefficients.clear();
        try_resize_decode_elements(&mut self.coefficients, coefficient_count, 0)?;
        self.segment_data.clear();
        try_reserve_decode_elements(&mut self.segment_data, segment_data_bytes)?;
        prepare_scratch(&mut self.scratch, width, height)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let coefficient_bytes = self
            .coefficients
            .capacity()
            .checked_mul(size_of::<u32>())
            .ok_or(ValidationError::ImageTooLarge)?;
        let segment_bytes = self.segment_data.capacity();
        coefficient_bytes
            .checked_add(segment_bytes)
            .ok_or(ValidationError::ImageTooLarge)?
            .checked_add(self.scratch.allocated_bytes()?)
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    /// Size the buffers for `code_block` without clearing coefficients left by
    /// an earlier block. A successful decode overwrites every coefficient;
    /// callers that return early without decoding must call
    /// [`Self::zero_coefficients`].
    pub(super) fn reset_for_overwrite(&mut self, code_block: &CodeBlock) -> Result<()> {
        let width = code_block.rect.width();
        let height = code_block.rect.height();
        self.width = width;
        self.height = height;
        let coefficient_count = checked_decode_sample_count(width, height)?;
        if self.coefficients.len() > coefficient_count {
            self.coefficients.truncate(coefficient_count);
        } else {
            try_resize_decode_elements(&mut self.coefficients, coefficient_count, 0)?;
        }
        self.segment_data.clear();
        prepare_scratch(&mut self.scratch, width, height)
    }

    #[cfg(feature = "simd")]
    pub(super) fn reset_for_direct_output(&mut self, code_block: &CodeBlock) -> Result<()> {
        let width = code_block.rect.width();
        let height = code_block.rect.height();
        self.width = width;
        self.height = height;
        self.segment_data.clear();
        prepare_scratch(&mut self.scratch, width, height)
    }

    pub(super) fn zero_coefficients(&mut self) {
        self.coefficients.fill(0);
    }

    pub(crate) fn coefficient_rows(&self) -> impl Iterator<Item = &[u32]> {
        self.coefficients.chunks_exact(self.width as usize)
    }

    #[cfg(test)]
    pub(crate) fn coefficient_owner_for_test(&self) -> (*const u32, usize) {
        (self.coefficients.as_ptr(), self.coefficients.capacity())
    }
}

#[derive(Debug, Default)]
pub(crate) struct HtBlockDecodeScratch {
    pub(super) cleanup: Vec<u16>,
    pub(super) v_n: Vec<u32>,
    pub(super) sigma: Vec<u16>,
    pub(super) prev_row_sig: Vec<u16>,
    /// The `MagSgn` segment unstuffed into a flat, padded bit buffer.
    pub(super) magsgn: Vec<u8>,
}

impl HtBlockDecodeScratch {
    pub(crate) const fn empty() -> Self {
        Self {
            cleanup: Vec::new(),
            v_n: Vec::new(),
            sigma: Vec::new(),
            prev_row_sig: Vec::new(),
            magsgn: Vec::new(),
        }
    }

    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<()> {
        prepare_scratch(self, width, height)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        include_capacity::<u16>(&mut bytes, self.cleanup.capacity())?;
        include_capacity::<u32>(&mut bytes, self.v_n.capacity())?;
        include_capacity::<u16>(&mut bytes, self.sigma.capacity())?;
        include_capacity::<u16>(&mut bytes, self.prev_row_sig.capacity())?;
        include_capacity::<u8>(&mut bytes, self.magsgn.capacity())?;
        Ok(bytes)
    }
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HtBlockDecodeScratchCapacities {
    pub(super) cleanup: usize,
    pub(super) v_n: usize,
    pub(super) sigma: usize,
    pub(super) prev_row_sig: usize,
    pub(super) magsgn: usize,
}

#[cfg(test)]
impl HtBlockDecodeScratch {
    pub(super) fn capacities_for_test(&self) -> HtBlockDecodeScratchCapacities {
        HtBlockDecodeScratchCapacities {
            cleanup: self.cleanup.capacity(),
            v_n: self.v_n.capacity(),
            sigma: self.sigma.capacity(),
            prev_row_sig: self.prev_row_sig.capacity(),
            magsgn: self.magsgn.capacity(),
        }
    }

    pub(super) fn poison_for_test(&mut self) {
        self.cleanup.fill(u16::MAX);
        self.v_n.fill(u32::MAX);
        self.sigma.fill(u16::MAX);
        self.prev_row_sig.fill(u16::MAX);
        self.magsgn.fill(u8::MAX);
    }
}
