// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Dct53GridScratch, Dct97GridScratch, JpegToHtj2kError};
use crate::allocation::{checked_add_allocation_bytes, checked_capacity_bytes};

#[derive(Debug, Default)]
pub(super) struct JpegToHtj2kScratch {
    pub(super) dct_blocks_f64: Vec<[[f64; 8]; 8]>,
    pub(super) dct53_grid: Dct53GridScratch,
    pub(super) dct97_grid: Dct97GridScratch,
    pub(super) integer_idct_blocks: Vec<Option<[i32; 64]>>,
    pub(super) integer_row: Vec<i32>,
}

impl JpegToHtj2kScratch {
    pub(super) fn retained_bytes(&self) -> Result<usize, JpegToHtj2kError> {
        let mut total = checked_capacity_bytes::<[[f64; 8]; 8]>(self.dct_blocks_f64.capacity())?;
        total = checked_add_allocation_bytes(total, self.dct53_grid.retained_bytes()?)?;
        total = checked_add_allocation_bytes(total, self.dct97_grid.retained_bytes()?)?;
        total = checked_add_allocation_bytes(
            total,
            checked_capacity_bytes::<Option<[i32; 64]>>(self.integer_idct_blocks.capacity())?,
        )?;
        Ok(checked_add_allocation_bytes(
            total,
            checked_capacity_bytes::<i32>(self.integer_row.capacity())?,
        )?)
    }
}
