// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral JPEG 2000 encode geometry policy.

pub use crate::packetization::sort_packet_descriptors_for_progression;
use crate::{J2kPacketizationProgressionOrder, J2kSubBandType};

const MINIMUM_LOSSLESS_DWT_DIMENSION: u32 = 64;
const MAXIMUM_STORED_CODE_BLOCK_EXPONENT: u8 = 8;
const MAXIMUM_COMBINED_ACTUAL_CODE_BLOCK_EXPONENT: u8 = 12;

/// Dimensions of one forward-DWT decomposition level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeDwtLevelDimensions {
    /// Width of the low-pass output.
    pub low_width: u32,
    /// Height of the low-pass output.
    pub low_height: u32,
    /// Width of the high-pass output.
    pub high_width: u32,
    /// Height of the high-pass output.
    pub high_height: u32,
}

/// Iterator over forward-DWT dimensions, from the full-resolution input inward.
#[derive(Debug, Clone)]
pub struct EncodeDwtLevelDimensionsIter {
    width: u32,
    height: u32,
    remaining: u8,
}

impl Iterator for EncodeDwtLevelDimensionsIter {
    type Item = EncodeDwtLevelDimensions;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let dimensions = encode_dwt_level_dimensions_for_input(self.width, self.height);
        self.width = dimensions.low_width;
        self.height = dimensions.low_height;
        self.remaining -= 1;
        Some(dimensions)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::from(self.remaining);
        (remaining, Some(remaining))
    }
}

/// Derive the low- and high-pass dimensions for one forward-DWT input.
#[must_use]
pub const fn encode_dwt_level_dimensions_for_input(
    width: u32,
    height: u32,
) -> EncodeDwtLevelDimensions {
    EncodeDwtLevelDimensions {
        low_width: width / 2 + width % 2,
        low_height: height / 2 + height % 2,
        high_width: width / 2,
        high_height: height / 2,
    }
}

impl ExactSizeIterator for EncodeDwtLevelDimensionsIter {}

/// Validated JPEG 2000 code-block dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeCodeBlockDimensions {
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
}

/// Reason that JPEG 2000 code-block geometry is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeBlockGeometryError {
    /// A dimension is smaller than the minimum legal four coefficients.
    DimensionTooSmall,
    /// A dimension is not a power of two.
    DimensionNotPowerOfTwo,
    /// A COD exponent-minus-two field exceeds its eight-bit Part 1 range.
    StoredExponentTooLarge,
    /// The code-block area exceeds 4096 coefficients.
    AreaTooLarge,
}

/// Return the legal decomposition-level ceiling for an image geometry.
///
/// The ceiling is `floor(log2(min(width, height)))`; a zero or unit-length
/// axis supports no decomposition levels.
#[must_use]
pub const fn maximum_decomposition_levels(width: u32, height: u32) -> u8 {
    let mut minimum_dimension = if width < height { width } else { height };
    let mut levels = 0_u8;
    while minimum_dimension > 1 {
        minimum_dimension >>= 1;
        levels += 1;
    }
    levels
}

/// Resolve the shared lossless decomposition policy for an image geometry.
///
/// Without an explicit maximum, LRCP and RLCP use one level for dimensions at
/// least 64, while position-sensitive progressions reduce the base resolution
/// to at most 64. An explicit value selects that many levels, capped by legal
/// geometry. Dimensions below 64 intentionally remain undecomposed even when
/// an explicit value is supplied.
#[must_use]
pub const fn lossless_decomposition_levels(
    width: u32,
    height: u32,
    progression: J2kPacketizationProgressionOrder,
    explicit_maximum: Option<u8>,
) -> u8 {
    let minimum_dimension = if width < height { width } else { height };
    let legal_maximum = maximum_decomposition_levels(width, height);
    if let Some(requested) = explicit_maximum {
        if minimum_dimension < MINIMUM_LOSSLESS_DWT_DIMENSION {
            return 0;
        }
        return if requested < legal_maximum {
            requested
        } else {
            legal_maximum
        };
    }

    match progression {
        J2kPacketizationProgressionOrder::Lrcp | J2kPacketizationProgressionOrder::Rlcp => {
            if minimum_dimension < MINIMUM_LOSSLESS_DWT_DIMENSION {
                0
            } else {
                1
            }
        }
        J2kPacketizationProgressionOrder::Rpcl
        | J2kPacketizationProgressionOrder::Pcrl
        | J2kPacketizationProgressionOrder::Cprl => {
            let mut current_width = width;
            let mut current_height = height;
            let mut levels = 0_u8;
            while (if current_width < current_height {
                current_width
            } else {
                current_height
            }) > MINIMUM_LOSSLESS_DWT_DIMENSION
                && levels < legal_maximum
            {
                current_width = current_width / 2 + current_width % 2;
                current_height = current_height / 2 + current_height % 2;
                levels += 1;
            }
            levels
        }
    }
}

/// Iterate over legal forward-DWT level dimensions.
///
/// Requests above the legal geometry ceiling are capped before iteration.
#[must_use]
pub const fn encode_dwt_level_dimensions(
    width: u32,
    height: u32,
    requested_levels: u8,
) -> EncodeDwtLevelDimensionsIter {
    let legal_levels = maximum_decomposition_levels(width, height);
    EncodeDwtLevelDimensionsIter {
        width,
        height,
        remaining: if requested_levels < legal_levels {
            requested_levels
        } else {
            legal_levels
        },
    }
}

/// Convert a power-of-two code-block dimension into COD's exponent-minus-two.
///
/// Valid dimensions range from 4 through 1024 coefficients.
pub fn code_block_exponent(dimension: u32) -> Result<u8, CodeBlockGeometryError> {
    if dimension < 4 {
        return Err(CodeBlockGeometryError::DimensionTooSmall);
    }
    if !dimension.is_power_of_two() {
        return Err(CodeBlockGeometryError::DimensionNotPowerOfTwo);
    }
    let stored = dimension.trailing_zeros() - 2;
    if stored > u32::from(MAXIMUM_STORED_CODE_BLOCK_EXPONENT) {
        return Err(CodeBlockGeometryError::StoredExponentTooLarge);
    }
    u8::try_from(stored).map_err(|_| CodeBlockGeometryError::StoredExponentTooLarge)
}

/// Validate one COD exponent-minus-two field and derive its dimension.
pub const fn code_block_dimension(stored_exponent: u8) -> Result<u32, CodeBlockGeometryError> {
    if stored_exponent > MAXIMUM_STORED_CODE_BLOCK_EXPONENT {
        return Err(CodeBlockGeometryError::StoredExponentTooLarge);
    }
    Ok(1_u32 << (stored_exponent + 2))
}

/// Validate COD exponent-minus-two fields and derive code-block dimensions.
pub const fn code_block_dimensions(
    width_exponent: u8,
    height_exponent: u8,
) -> Result<EncodeCodeBlockDimensions, CodeBlockGeometryError> {
    let width = match code_block_dimension(width_exponent) {
        Ok(width) => width,
        Err(error) => return Err(error),
    };
    let height = match code_block_dimension(height_exponent) {
        Ok(height) => height,
        Err(error) => return Err(error),
    };
    let actual_width_exponent = width_exponent + 2;
    let actual_height_exponent = height_exponent + 2;
    if actual_width_exponent + actual_height_exponent > MAXIMUM_COMBINED_ACTUAL_CODE_BLOCK_EXPONENT
    {
        return Err(CodeBlockGeometryError::AreaTooLarge);
    }
    Ok(EncodeCodeBlockDimensions { width, height })
}

/// Derive reversible, no-quantization total bitplanes for one subband.
#[must_use]
pub const fn reversible_subband_total_bitplanes(
    bit_depth: u8,
    guard_bits: u8,
    subband: J2kSubBandType,
) -> u8 {
    let base = bit_depth.saturating_add(guard_bits);
    match subband {
        J2kSubBandType::LowLow => base.saturating_sub(1),
        J2kSubBandType::HighLow | J2kSubBandType::LowHigh => base,
        J2kSubBandType::HighHigh => base.saturating_add(1),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        code_block_dimension, code_block_dimensions, code_block_exponent,
        encode_dwt_level_dimensions, lossless_decomposition_levels, maximum_decomposition_levels,
        reversible_subband_total_bitplanes, CodeBlockGeometryError,
    };
    use crate::{
        sort_packet_descriptors_for_progression, J2kPacketizationPacketDescriptor,
        J2kPacketizationProgressionOrder, J2kSubBandType,
    };

    const SQUARE_DIMENSIONS: [u32; 12] = [1, 2, 3, 31, 32, 63, 64, 65, 127, 128, 512, 1024];
    const REPRESENTATIVE_DIMENSIONS: [(u32, u32); 3] = [(640, 480), (1024, 1024), (2592, 1944)];
    const PROGRESSIONS: [J2kPacketizationProgressionOrder; 5] = [
        J2kPacketizationProgressionOrder::Lrcp,
        J2kPacketizationProgressionOrder::Rlcp,
        J2kPacketizationProgressionOrder::Rpcl,
        J2kPacketizationProgressionOrder::Pcrl,
        J2kPacketizationProgressionOrder::Cprl,
    ];
    const MAXIMUM_LEVELS: [Option<u8>; 6] = [None, Some(0), Some(1), Some(2), Some(5), Some(255)];

    fn reference_maximum_levels(width: u32, height: u32) -> u8 {
        let mut dimension = width.min(height);
        let mut levels = 0;
        while dimension > 1 {
            dimension /= 2;
            levels += 1;
        }
        levels
    }

    fn reference_default_levels(
        width: u32,
        height: u32,
        progression: J2kPacketizationProgressionOrder,
    ) -> u8 {
        if !matches!(
            progression,
            J2kPacketizationProgressionOrder::Rpcl
                | J2kPacketizationProgressionOrder::Pcrl
                | J2kPacketizationProgressionOrder::Cprl
        ) {
            return u8::from(width.min(height) >= 64);
        }

        let mut width = width;
        let mut height = height;
        let mut levels = 0;
        while width.min(height) > 64 {
            width = width.div_ceil(2);
            height = height.div_ceil(2);
            levels += 1;
        }
        levels
    }

    #[test]
    fn lossless_policy_covers_required_geometry_progression_and_override_matrix() {
        let geometries = SQUARE_DIMENSIONS
            .map(|dimension| (dimension, dimension))
            .into_iter()
            .chain(REPRESENTATIVE_DIMENSIONS);

        for (width, height) in geometries {
            let legal = reference_maximum_levels(width, height);
            assert_eq!(maximum_decomposition_levels(width, height), legal);
            for progression in PROGRESSIONS {
                let default = reference_default_levels(width, height, progression);
                for maximum in MAXIMUM_LEVELS {
                    let expected = maximum.map_or(default, |requested| {
                        if width.min(height) < 64 {
                            0
                        } else {
                            requested.min(legal)
                        }
                    });
                    assert_eq!(
                        lossless_decomposition_levels(width, height, progression, maximum),
                        expected,
                        "{width}x{height}, {progression:?}, {maximum:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_levels_do_not_force_decomposition_below_64() {
        for dimension in [1, 2, 3, 31, 32, 63] {
            for progression in PROGRESSIONS {
                assert_eq!(
                    lossless_decomposition_levels(dimension, dimension, progression, Some(u8::MAX)),
                    0
                );
            }
        }
    }

    #[test]
    fn dwt_level_dimensions_preserve_each_input_extent() {
        for (width, height) in SQUARE_DIMENSIONS
            .map(|dimension| (dimension, dimension))
            .into_iter()
            .chain(REPRESENTATIVE_DIMENSIONS)
        {
            let legal = maximum_decomposition_levels(width, height);
            let levels: Vec<_> = encode_dwt_level_dimensions(width, height, legal).collect();
            assert_eq!(levels.len(), usize::from(legal));
            let mut input_width = width;
            let mut input_height = height;
            for level in levels {
                assert_eq!(level.low_width + level.high_width, input_width);
                assert_eq!(level.low_height + level.high_height, input_height);
                assert_eq!(level.low_width, input_width.div_ceil(2));
                assert_eq!(level.low_height, input_height.div_ceil(2));
                input_width = level.low_width;
                input_height = level.low_height;
            }
        }
    }

    #[test]
    fn maximum_geometry_never_overflows() {
        assert_eq!(maximum_decomposition_levels(u32::MAX, u32::MAX), 31);
        let levels: Vec<_> = encode_dwt_level_dimensions(u32::MAX, u32::MAX, 31).collect();
        assert_eq!(levels.len(), 31);
        assert_eq!(levels[0].low_width, 1 << 31);
        assert_eq!(levels[0].high_width, (1 << 31) - 1);
        assert_eq!(levels.last().expect("final level").low_width, 2);
    }

    #[test]
    fn maximum_levels_use_the_shorter_axis_and_power_boundaries() {
        for (width, height, expected) in [
            (0, u32::MAX, 0),
            (u32::MAX, 0, 0),
            (1, u32::MAX, 0),
            (u32::MAX, 1, 0),
            (2, 8, 1),
            (8, 2, 1),
            (3, 9, 1),
            (9, 3, 1),
            (7, 9, 2),
            (9, 7, 2),
        ] {
            assert_eq!(maximum_decomposition_levels(width, height), expected);
        }

        for exponent in 1_u8..=31 {
            let power = 1_u32 << exponent;
            assert_eq!(maximum_decomposition_levels(power, power), exponent);
            assert_eq!(
                maximum_decomposition_levels(power - 1, u32::MAX),
                exponent - 1
            );
            assert_eq!(
                maximum_decomposition_levels(power.saturating_add(1), u32::MAX),
                exponent
            );
        }
    }

    #[test]
    fn code_block_geometry_validates_part1_exponents_and_area() {
        assert_eq!(code_block_exponent(4), Ok(0));
        assert_eq!(code_block_exponent(64), Ok(4));
        assert_eq!(code_block_exponent(1024), Ok(8));
        assert_eq!(
            code_block_exponent(0),
            Err(CodeBlockGeometryError::DimensionTooSmall)
        );
        assert_eq!(
            code_block_exponent(3),
            Err(CodeBlockGeometryError::DimensionTooSmall)
        );
        assert_eq!(
            code_block_exponent(12),
            Err(CodeBlockGeometryError::DimensionNotPowerOfTwo)
        );
        assert_eq!(
            code_block_exponent(2048),
            Err(CodeBlockGeometryError::StoredExponentTooLarge)
        );
        assert_eq!(code_block_dimension(0), Ok(4));
        assert_eq!(code_block_dimension(8), Ok(1024));
        assert_eq!(
            code_block_dimension(9),
            Err(CodeBlockGeometryError::StoredExponentTooLarge)
        );

        let dimensions = code_block_dimensions(4, 4).expect("64x64 is legal");
        assert_eq!((dimensions.width, dimensions.height), (64, 64));
        assert_eq!(
            code_block_dimensions(8, 8),
            Err(CodeBlockGeometryError::AreaTooLarge)
        );
        assert_eq!(
            code_block_dimensions(9, 0),
            Err(CodeBlockGeometryError::StoredExponentTooLarge)
        );
    }

    #[test]
    fn reversible_total_bitplanes_follow_subband_gain() {
        assert_eq!(
            reversible_subband_total_bitplanes(8, 2, J2kSubBandType::LowLow),
            9
        );
        assert_eq!(
            reversible_subband_total_bitplanes(8, 2, J2kSubBandType::HighLow),
            10
        );
        assert_eq!(
            reversible_subband_total_bitplanes(8, 2, J2kSubBandType::LowHigh),
            10
        );
        assert_eq!(
            reversible_subband_total_bitplanes(8, 2, J2kSubBandType::HighHigh),
            11
        );
        assert_eq!(
            reversible_subband_total_bitplanes(u8::MAX, u8::MAX, J2kSubBandType::HighHigh),
            u8::MAX
        );
    }

    #[test]
    fn packet_ordering_covers_required_component_counts_and_progressions() {
        for component_count in [1_u16, 3, 4] {
            for progression in PROGRESSIONS {
                let mut descriptors = Vec::new();
                for layer in 0..2 {
                    for resolution in 0..3 {
                        for component in 0..component_count {
                            descriptors.push(J2kPacketizationPacketDescriptor {
                                packet_index: u32::try_from(descriptors.len())
                                    .expect("small fixture"),
                                state_index: 0,
                                layer,
                                resolution,
                                component,
                                precinct: u64::from(component) + u64::from(resolution),
                            });
                        }
                    }
                }
                sort_packet_descriptors_for_progression(&mut descriptors, progression);
                assert_eq!(descriptors.len(), usize::from(component_count) * 6);
                assert!(descriptors
                    .iter()
                    .all(|descriptor| descriptor.component < component_count));
            }
        }
    }
}
