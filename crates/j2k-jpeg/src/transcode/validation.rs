// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation for caller-constructed DCT images before baseline re-emission.

use thiserror::Error;

use super::{JpegDctCodingMode, JpegDctImage};
use crate::adapter::JpegBaselineSampling;
use crate::entropy::ZIGZAG;

mod coefficients;

use self::coefficients::validate_baseline_coefficients;

const MAX_SAMPLING_FACTOR: u8 = 4;
const MAX_BLOCKS_PER_MCU: u16 = 10;

/// Why a [`JpegDctImage`] cannot be re-emitted as a canonical baseline JPEG.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum JpegDctImageError {
    /// The coefficient image did not originate from baseline sequential coding.
    #[error("baseline DCT re-emission requires baseline sequential coding, got {actual:?}")]
    UnsupportedCodingMode {
        /// Supplied entropy coding mode.
        actual: JpegDctCodingMode,
    },
    /// The reference-grid width or height was zero.
    #[error("DCT image dimensions must be nonzero, got {width}x{height}")]
    EmptyDimensions {
        /// Supplied reference-grid width.
        width: u32,
        /// Supplied reference-grid height.
        height: u32,
    },
    /// The reference grid does not fit baseline JPEG's 16-bit frame fields.
    #[error("baseline DCT image dimensions must fit in u16, got {width}x{height}")]
    DimensionsTooLarge {
        /// Supplied reference-grid width.
        width: u32,
        /// Supplied reference-grid height.
        height: u32,
    },
    /// Baseline re-emission supports grayscale or three-component images only.
    #[error("baseline DCT re-emission supports 1 or 3 components, got {actual}")]
    UnsupportedComponentCount {
        /// Supplied component count.
        actual: usize,
    },
    /// A component index did not match its SOF declaration position.
    #[error("DCT component at SOF position {position} declares component index {component_index}")]
    ComponentOrderMismatch {
        /// Component position in the supplied vector.
        position: usize,
        /// Component index declared by the value at that position.
        component_index: usize,
    },
    /// A horizontal or vertical sampling factor was outside JPEG's 1 through 4 range.
    #[error(
        "DCT component {component_index} sampling factors must each be in 1..=4, got {h_samp}x{v_samp}"
    )]
    SamplingFactorOutOfRange {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Supplied horizontal sampling factor.
        h_samp: u8,
        /// Supplied vertical sampling factor.
        v_samp: u8,
    },
    /// A single-component image did not use the canonical one-block MCU shape.
    #[error("baseline grayscale DCT re-emission requires 1x1 sampling, got {h_samp}x{v_samp}")]
    UnsupportedGrayscaleSampling {
        /// Supplied horizontal sampling factor.
        h_samp: u8,
        /// Supplied vertical sampling factor.
        v_samp: u8,
    },
    /// The aggregate sampling factors exceeded JPEG's ten-block MCU limit.
    #[error("DCT sampling uses {blocks_per_mcu} blocks per MCU; JPEG permits at most 10")]
    TooManyBlocksPerMcu {
        /// Sum of `H_i * V_i` across frame components.
        blocks_per_mcu: u16,
    },
    /// Checked component-grid arithmetic overflowed.
    #[error("DCT component {component_index} block-grid arithmetic overflow")]
    BlockGridArithmeticOverflow {
        /// Component in SOF declaration order.
        component_index: usize,
    },
    /// A component's declared padded block grid did not match its sampling geometry.
    #[error(
        "DCT component {component_index} block grid is {actual_cols}x{actual_rows}, expected {expected_cols}x{expected_rows}"
    )]
    BlockGridMismatch {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Supplied padded block columns.
        actual_cols: u32,
        /// Supplied padded block rows.
        actual_rows: u32,
        /// Required padded block columns.
        expected_cols: u32,
        /// Required padded block rows.
        expected_rows: u32,
    },
    /// The quantized block owner did not cover the complete padded grid.
    #[error("DCT component {component_index} has {actual} quantized blocks, expected {expected}")]
    QuantizedBlockCountMismatch {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Supplied block count.
        actual: usize,
        /// Required block count.
        expected: usize,
    },
    /// A DC difference required a magnitude category unavailable in baseline JPEG.
    #[error(
        "DCT component {component_index} block {block_index} DC difference {difference} uses category {category}; baseline JPEG permits at most 11"
    )]
    DcMagnitudeCategoryOutOfRange {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Block index in the component's padded row-major grid.
        block_index: usize,
        /// Difference from the previous DC value in entropy scan order.
        difference: i32,
        /// Magnitude category required to encode the difference.
        category: u8,
    },
    /// An AC value required a magnitude category unavailable in baseline JPEG.
    #[error(
        "DCT component {component_index} block {block_index} AC coefficient {coefficient_index} value {value} uses category {category}; baseline JPEG permits at most 10"
    )]
    AcMagnitudeCategoryOutOfRange {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Block index in the component's padded row-major grid.
        block_index: usize,
        /// Coefficient index in natural row-major order.
        coefficient_index: usize,
        /// Supplied quantized coefficient.
        value: i16,
        /// Magnitude category required to encode the coefficient.
        category: u8,
    },
    /// A baseline quantization entry was zero or required more than eight bits.
    #[error(
        "DCT component {component_index} quantization entry {zigzag_index} must be in 1..=255, got {value}"
    )]
    QuantizationValueOutOfRange {
        /// Component in SOF declaration order.
        component_index: usize,
        /// Entry index in JPEG zigzag table order.
        zigzag_index: usize,
        /// Supplied quantization value.
        value: u16,
    },
    /// The two chroma components require different quantization tables.
    #[error("baseline DCT re-emission supports one shared chroma quantization table")]
    ChromaQuantizationTableMismatch,
}

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
