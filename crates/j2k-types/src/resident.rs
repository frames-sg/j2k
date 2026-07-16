// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    J2kPacketizationProgressionOrder, MAX_JPEG2000_PART1_COMPONENTS,
    MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH,
};

mod input_error;
pub use self::input_error::J2kResidentEncodeInputError;

/// Validated geometry and sample format for a backend-resident encode input.
///
/// The descriptor deliberately carries no host sample slice. Backends that own
/// the pixels in device memory can therefore describe the logical image
/// without manufacturing a host allocation or an invalid borrowed slice.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kResidentEncodeInput {
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
}

impl J2kResidentEncodeInput {
    /// Validate and construct a backend-resident input descriptor.
    pub fn new(
        width: u32,
        height: u32,
        num_components: u16,
        bit_depth: u8,
        signed: bool,
    ) -> Result<Self, J2kResidentEncodeInputError> {
        if width == 0 || height == 0 {
            return Err(J2kResidentEncodeInputError::EmptyGeometry { width, height });
        }
        if num_components == 0 || num_components > MAX_JPEG2000_PART1_COMPONENTS {
            return Err(J2kResidentEncodeInputError::ComponentCountOutOfRange { num_components });
        }
        if bit_depth == 0 || bit_depth > MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH {
            return Err(J2kResidentEncodeInputError::PrecisionOutOfRange { bit_depth });
        }
        let width_usize = usize::try_from(width)
            .map_err(|_| J2kResidentEncodeInputError::AddressSpaceOverflow)?;
        let height_usize = usize::try_from(height)
            .map_err(|_| J2kResidentEncodeInputError::AddressSpaceOverflow)?;
        let bytes_per_sample = usize::from(bit_depth).div_ceil(8);
        width_usize
            .checked_mul(height_usize)
            .and_then(|pixels| pixels.checked_mul(usize::from(num_components)))
            .and_then(|samples| samples.checked_mul(bytes_per_sample))
            .ok_or(J2kResidentEncodeInputError::AddressSpaceOverflow)?;
        Ok(Self {
            width,
            height,
            num_components,
            bit_depth,
            signed,
        })
    }

    /// Image width in samples.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Image height in samples.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Number of interleaved image components.
    #[must_use]
    pub const fn num_components(self) -> u16 {
        self.num_components
    }

    /// Significant bits per component sample.
    #[must_use]
    pub const fn bit_depth(self) -> u8 {
        self.bit_depth
    }

    /// Whether component samples are signed.
    #[must_use]
    pub const fn signed(self) -> bool {
        self.signed
    }
}

/// Adapter HTJ2K tile-body job whose source pixels remain backend-resident.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kResidentHtj2kTileEncodeJob<'a> {
    /// Validated logical image geometry and sample format.
    pub input: J2kResidentEncodeInput,
    /// Number of DWT decomposition levels.
    pub num_decomposition_levels: u8,
    /// Whether the codestream uses reversible coding.
    pub reversible: bool,
    /// Whether a multi-component transform should be applied.
    pub use_mct: bool,
    /// JPEG 2000 guard bits used to derive total coded bitplanes.
    pub guard_bits: u8,
    /// Code-block width in samples.
    pub code_block_width: u32,
    /// Code-block height in samples.
    pub code_block_height: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Per-component sampling factors, as `(x_rsiz, y_rsiz)`.
    pub component_sampling: &'a [(u8, u8)],
    /// Quantization step sizes, as `(exponent, mantissa)`, in codestream order.
    pub quantization_steps: &'a [(u16, u16)],
}

#[cfg(test)]
mod tests;
