// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{idct_islow_block, validate_component_block_grid, JpegDctComponent, JpegToHtj2kError};
use crate::allocation::try_vec_filled;

pub(super) fn idct_component_samples_i32(
    component: &JpegDctComponent,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    validate_component_block_grid(component)?;

    let width = component.width as usize;
    let height = component.height as usize;
    let block_cols = component.block_cols as usize;
    let block_rows = component.block_rows as usize;
    let mut samples = try_vec_filled(checked_product(width, height)?, 0i32)?;
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let block = &component.dequantized_blocks[block_y * block_cols + block_x];
            let block_samples = idct_islow_block(block);
            for local_y in 0..8 {
                let y = block_y * 8 + local_y;
                if y >= height {
                    continue;
                }
                for local_x in 0..8 {
                    let x = block_x * 8 + local_x;
                    if x >= width {
                        continue;
                    }
                    samples[y * width + x] = i32::from(block_samples[local_y * 8 + local_x]) - 128;
                }
            }
        }
    }

    Ok(samples)
}

pub(super) fn checked_product(left: usize, right: usize) -> Result<usize, JpegToHtj2kError> {
    left.checked_mul(right).ok_or_else(cap_overflow)
}

pub(super) fn checked_sum(left: usize, right: usize) -> Result<usize, JpegToHtj2kError> {
    left.checked_add(right).ok_or_else(cap_overflow)
}

pub(super) fn validate_band_len(
    band: &[i32],
    width: usize,
    height: usize,
) -> Result<(), JpegToHtj2kError> {
    if band.len() != checked_product(width, height)? {
        return Err(JpegToHtj2kError::Validation(
            "accelerated reversible 5/3 band length does not match dimensions",
        ));
    }
    Ok(())
}

fn cap_overflow() -> JpegToHtj2kError {
    JpegToHtj2kError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}
