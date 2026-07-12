// SPDX-License-Identifier: MIT OR Apache-2.0

/// Validation failure for a backend-resident encode input descriptor.
#[doc(hidden)]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum J2kResidentEncodeInputError {
    /// Either image dimension was zero.
    EmptyGeometry {
        /// Requested image width.
        width: u32,
        /// Requested image height.
        height: u32,
    },
    /// The component count was outside the JPEG 2000 Part 1 range.
    ComponentCountOutOfRange {
        /// Requested component count.
        num_components: u16,
    },
    /// The sample precision was outside the JPEG 2000 Part 1 range.
    PrecisionOutOfRange {
        /// Requested significant bits per sample.
        bit_depth: u8,
    },
    /// The logical sample storage size exceeded the target address space.
    AddressSpaceOverflow,
}

impl J2kResidentEncodeInputError {
    /// Stable validation reason for adapter boundaries with string-based errors.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::EmptyGeometry { .. } => "resident encode input dimensions must be non-zero",
            Self::ComponentCountOutOfRange { .. } => {
                "resident encode input component count must be in 1..=16384"
            }
            Self::PrecisionOutOfRange { .. } => "resident encode input bit depth must be in 1..=38",
            Self::AddressSpaceOverflow => "resident encode input dimensions overflow address space",
        }
    }
}

impl core::fmt::Display for J2kResidentEncodeInputError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.reason())
    }
}

impl core::error::Error for J2kResidentEncodeInputError {}
