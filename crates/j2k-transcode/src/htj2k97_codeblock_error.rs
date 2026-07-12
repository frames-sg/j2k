// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed validation failures for shared HTJ2K 9/7 code-block options.

use core::fmt;

/// Code-block dimension whose exponent failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Htj2k97CodeBlockAxis {
    /// Horizontal code-block dimension.
    Width,
    /// Vertical code-block dimension.
    Height,
}

impl fmt::Display for Htj2k97CodeBlockAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Width => formatter.write_str("width"),
            Self::Height => formatter.write_str("height"),
        }
    }
}

/// Failure returned when shared HTJ2K 9/7 code-block options are unsupported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Htj2k97CodeBlockOptionsError {
    /// Bit depth, guard bits, or the global quantization scale is unsupported.
    NumericOptionsOutOfRange,
    /// A subband scale, derived delta, or declared bitplane count is unsupported.
    QuantizationOptionsOutOfRange,
    /// A dimension exponent cannot be represented by the host index type.
    DimensionExponentUnsupported {
        /// Dimension whose exponent was rejected.
        axis: Htj2k97CodeBlockAxis,
        /// JPEG 2000 code-block exponent minus two.
        exponent_minus_two: u8,
    },
    /// Decoded code-block dimensions exceed the HTJ2K side or area limit.
    DimensionsExceedLimits {
        /// Decoded code-block width.
        width: usize,
        /// Decoded code-block height.
        height: usize,
    },
}

impl Htj2k97CodeBlockOptionsError {
    /// Returns allocation-free presentation text for legacy diagnostics.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NumericOptionsOutOfRange => {
                "9/7 code-block options are outside supported numeric range"
            }
            Self::QuantizationOptionsOutOfRange => {
                "9/7 code-block quantization options are outside supported range"
            }
            Self::DimensionExponentUnsupported { .. } => {
                "9/7 code-block dimension exponent is unsupported"
            }
            Self::DimensionsExceedLimits { .. } => "9/7 code-block dimensions exceed HTJ2K limits",
        }
    }
}

impl fmt::Display for Htj2k97CodeBlockOptionsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NumericOptionsOutOfRange | Self::QuantizationOptionsOutOfRange => {
                formatter.write_str(self.reason())
            }
            Self::DimensionExponentUnsupported {
                axis,
                exponent_minus_two,
            } => write!(
                formatter,
                "{}: {axis} exponent-minus-two {exponent_minus_two}",
                self.reason()
            ),
            Self::DimensionsExceedLimits { width, height } => {
                write!(formatter, "{}: {width}x{height}", self.reason())
            }
        }
    }
}

impl std::error::Error for Htj2k97CodeBlockOptionsError {}
