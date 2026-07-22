// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::CodeBlock;
use super::pipeline::prepare_scratch;
use crate::error::{Result, ValidationError};
use crate::{checked_decode_sample_count, try_resize_decode_elements};
use core::mem::size_of;

mod scratch;
#[cfg(test)]
pub(super) use self::scratch::zeroed_u32_scratch;
pub(super) use self::scratch::{resized_u16_scratch, resized_u32_scratch, zeroed_u16_scratch};

#[derive(Default)]
pub(crate) struct HtBlockDecodeContext {
    pub(super) coefficients: Vec<u32>,
    pub(super) scratch: HtBlockDecodeScratch,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl HtBlockDecodeContext {
    pub(crate) fn prepare(&mut self, width: u32, height: u32) -> Result<()> {
        self.width = width;
        self.height = height;
        let coefficient_count = checked_decode_sample_count(width, height)?;
        self.coefficients.clear();
        try_resize_decode_elements(&mut self.coefficients, coefficient_count, 0)?;
        prepare_scratch(&mut self.scratch, width, height)
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize> {
        let coefficient_bytes = self
            .coefficients
            .capacity()
            .checked_mul(size_of::<u32>())
            .ok_or(ValidationError::ImageTooLarge)?;
        coefficient_bytes
            .checked_add(self.scratch.allocated_bytes()?)
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    pub(super) fn reset(&mut self, code_block: &CodeBlock) -> Result<()> {
        self.prepare(code_block.rect.width(), code_block.rect.height())
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
}

impl HtBlockDecodeScratch {
    pub(crate) const fn empty() -> Self {
        Self {
            cleanup: Vec::new(),
            v_n: Vec::new(),
            sigma: Vec::new(),
            prev_row_sig: Vec::new(),
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
}

#[cfg(test)]
impl HtBlockDecodeScratch {
    pub(super) fn capacities_for_test(&self) -> HtBlockDecodeScratchCapacities {
        HtBlockDecodeScratchCapacities {
            cleanup: self.cleanup.capacity(),
            v_n: self.v_n.capacity(),
            sigma: self.sigma.capacity(),
            prev_row_sig: self.prev_row_sig.capacity(),
        }
    }

    pub(super) fn poison_for_test(&mut self) {
        self.cleanup.fill(u16::MAX);
        self.v_n.fill(u32::MAX);
        self.sigma.fill(u16::MAX);
        self.prev_row_sig.fill(u16::MAX);
    }
}
