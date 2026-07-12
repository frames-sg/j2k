// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public coefficient-domain contracts shared by validation and encoding.

use thiserror::Error;

/// JPEG DCT entropy coding mode represented by a coefficient-domain image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDctCodingMode {
    /// SOF0 baseline sequential Huffman DCT.
    BaselineSequential,
    /// SOF2 progressive Huffman DCT with accumulated scan coefficients.
    Progressive,
}

/// Why a coefficient-domain image cannot be re-emitted as a canonical baseline JPEG.
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
