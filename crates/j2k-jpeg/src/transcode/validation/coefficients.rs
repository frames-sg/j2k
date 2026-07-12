// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline entropy-category validation for caller-supplied coefficients.

use super::JpegDctImageError;
use crate::adapter::JpegBaselineSampling;
use crate::encoder::magnitude;
use crate::transcode::JpegDctImage;

const MAX_BASELINE_DC_CATEGORY: u8 = 11;
const MAX_BASELINE_AC_CATEGORY: u8 = 10;

pub(super) fn validate_baseline_coefficients(
    image: &JpegDctImage,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegDctImageError> {
    let mcu_cols = image.width.div_ceil(u32::from(sampling.max_h) * 8);
    let mcu_rows = image.height.div_ceil(u32::from(sampling.max_v) * 8);

    for (component_index, component) in image.components.iter().enumerate() {
        let mut previous_dc = 0i32;
        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                for block_y in 0..sampling.v[component_index] {
                    for block_x in 0..sampling.h[component_index] {
                        let source_x =
                            mcu_x * u32::from(sampling.h[component_index]) + u32::from(block_x);
                        let source_y =
                            mcu_y * u32::from(sampling.v[component_index]) + u32::from(block_y);
                        let block_index = checked_block_index(
                            component_index,
                            source_x,
                            source_y,
                            component.block_cols,
                        )?;
                        let block = &component.quantized_blocks[block_index];
                        validate_block(component_index, block_index, block, &mut previous_dc)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn checked_block_index(
    component_index: usize,
    source_x: u32,
    source_y: u32,
    block_cols: u32,
) -> Result<usize, JpegDctImageError> {
    source_y
        .checked_mul(block_cols)
        .and_then(|row| row.checked_add(source_x))
        .and_then(|index| usize::try_from(index).ok())
        .ok_or(JpegDctImageError::BlockGridArithmeticOverflow { component_index })
}

fn validate_block(
    component_index: usize,
    block_index: usize,
    block: &[i16; 64],
    previous_dc: &mut i32,
) -> Result<(), JpegDctImageError> {
    let dc = i32::from(block[0]);
    let difference = dc - *previous_dc;
    *previous_dc = dc;
    let category = magnitude(difference).0;
    if category > MAX_BASELINE_DC_CATEGORY {
        return Err(JpegDctImageError::DcMagnitudeCategoryOutOfRange {
            component_index,
            block_index,
            difference,
            category,
        });
    }

    for (coefficient_index, &value) in block.iter().enumerate().skip(1) {
        let category = magnitude(i32::from(value)).0;
        if category > MAX_BASELINE_AC_CATEGORY {
            return Err(JpegDctImageError::AcMagnitudeCategoryOutOfRange {
                component_index,
                block_index,
                coefficient_index,
                value,
                category,
            });
        }
    }
    Ok(())
}
