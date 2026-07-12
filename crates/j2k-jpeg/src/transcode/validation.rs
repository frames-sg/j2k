// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation for caller-constructed DCT images before baseline re-emission.

use super::{JpegDctCodingMode, JpegDctImage};
use crate::adapter::JpegBaselineSampling;
use crate::dct_contract::JpegDctImageError;
use crate::entropy::ZIGZAG;

mod coefficients;

use self::coefficients::validate_baseline_coefficients;

const MAX_SAMPLING_FACTOR: u8 = 4;
const MAX_BLOCKS_PER_MCU: u16 = 10;

pub(super) struct ValidatedBaselineDctImage {
    pub(super) sampling: JpegBaselineSampling,
    pub(super) luma_quant: [u8; 64],
    pub(super) chroma_quant: Option<[u8; 64]>,
}

pub(super) fn validate_baseline_dct_image(
    image: &JpegDctImage,
) -> Result<ValidatedBaselineDctImage, JpegDctImageError> {
    if image.coding_mode != JpegDctCodingMode::BaselineSequential {
        return Err(JpegDctImageError::UnsupportedCodingMode {
            actual: image.coding_mode,
        });
    }
    validate_dimensions(image.width, image.height)?;

    let component_count = match image.components.len() {
        1 => 1,
        3 => 3,
        actual => return Err(JpegDctImageError::UnsupportedComponentCount { actual }),
    };
    let sampling = validate_sampling_and_order(image, component_count)?;
    validate_component_grids(image, sampling)?;

    let luma_quant = quant_table_to_natural_u8(0, &image.components[0].quant_table)?;
    let chroma_quant = if component_count == 3 {
        if image.components[1].quant_table != image.components[2].quant_table {
            return Err(JpegDctImageError::ChromaQuantizationTableMismatch);
        }
        Some(quant_table_to_natural_u8(
            1,
            &image.components[1].quant_table,
        )?)
    } else {
        None
    };
    validate_baseline_coefficients(image, sampling)?;

    Ok(ValidatedBaselineDctImage {
        sampling,
        luma_quant,
        chroma_quant,
    })
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), JpegDctImageError> {
    if width == 0 || height == 0 {
        return Err(JpegDctImageError::EmptyDimensions { width, height });
    }
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(JpegDctImageError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

fn validate_sampling_and_order(
    image: &JpegDctImage,
    component_count: u8,
) -> Result<JpegBaselineSampling, JpegDctImageError> {
    let mut h = [0; 3];
    let mut v = [0; 3];
    let mut max_h = 0;
    let mut max_v = 0;
    let mut blocks_per_mcu = 0u16;
    for (position, component) in image.components.iter().enumerate() {
        if component.component_index != position {
            return Err(JpegDctImageError::ComponentOrderMismatch {
                position,
                component_index: component.component_index,
            });
        }
        if !(1..=MAX_SAMPLING_FACTOR).contains(&component.h_samp)
            || !(1..=MAX_SAMPLING_FACTOR).contains(&component.v_samp)
        {
            return Err(JpegDctImageError::SamplingFactorOutOfRange {
                component_index: position,
                h_samp: component.h_samp,
                v_samp: component.v_samp,
            });
        }
        h[position] = component.h_samp;
        v[position] = component.v_samp;
        max_h = max_h.max(component.h_samp);
        max_v = max_v.max(component.v_samp);
        blocks_per_mcu += u16::from(component.h_samp) * u16::from(component.v_samp);
    }
    if component_count == 1 && (h[0] != 1 || v[0] != 1) {
        return Err(JpegDctImageError::UnsupportedGrayscaleSampling {
            h_samp: h[0],
            v_samp: v[0],
        });
    }
    if blocks_per_mcu > MAX_BLOCKS_PER_MCU {
        return Err(JpegDctImageError::TooManyBlocksPerMcu { blocks_per_mcu });
    }
    Ok(JpegBaselineSampling {
        components: component_count,
        h,
        v,
        max_h,
        max_v,
    })
}

fn validate_component_grids(
    image: &JpegDctImage,
    sampling: JpegBaselineSampling,
) -> Result<(), JpegDctImageError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcu_cols = image.width.div_ceil(mcu_width);
    let mcu_rows = image.height.div_ceil(mcu_height);
    for (component_index, component) in image.components.iter().enumerate() {
        let expected_cols = mcu_cols
            .checked_mul(u32::from(sampling.h[component_index]))
            .ok_or(JpegDctImageError::BlockGridArithmeticOverflow { component_index })?;
        let expected_rows = mcu_rows
            .checked_mul(u32::from(sampling.v[component_index]))
            .ok_or(JpegDctImageError::BlockGridArithmeticOverflow { component_index })?;
        if component.block_cols != expected_cols || component.block_rows != expected_rows {
            return Err(JpegDctImageError::BlockGridMismatch {
                component_index,
                actual_cols: component.block_cols,
                actual_rows: component.block_rows,
                expected_cols,
                expected_rows,
            });
        }
        let expected = expected_cols
            .checked_mul(expected_rows)
            .and_then(|blocks| usize::try_from(blocks).ok())
            .ok_or(JpegDctImageError::BlockGridArithmeticOverflow { component_index })?;
        let actual = component.quantized_blocks.len();
        if actual != expected {
            return Err(JpegDctImageError::QuantizedBlockCountMismatch {
                component_index,
                actual,
                expected,
            });
        }
    }
    Ok(())
}

fn quant_table_to_natural_u8(
    component_index: usize,
    quant: &[u16; 64],
) -> Result<[u8; 64], JpegDctImageError> {
    let mut natural = [0; 64];
    for (zigzag_index, &natural_index) in ZIGZAG.iter().enumerate() {
        let value = quant[zigzag_index];
        let value = u8::try_from(value).ok().filter(|value| *value != 0).ok_or(
            JpegDctImageError::QuantizationValueOutOfRange {
                component_index,
                zigzag_index,
                value,
            },
        )?;
        natural[usize::from(natural_index)] = value;
    }
    Ok(natural)
}
