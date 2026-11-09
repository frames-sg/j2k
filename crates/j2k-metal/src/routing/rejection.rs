// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitMetalRejection {
    UnsupportedFormat { fmt: PixelFormat },
    PrecisionAboveSixteen,
}

impl ExplicitMetalRejection {
    pub(super) fn error(self) -> Error {
        let reason = self.error_reason();
        Error::capability_rejected(match self {
            Self::UnsupportedFormat { .. } => {
                j2k_core::CapabilityRejection::unsupported_operation(reason)
            }
            Self::PrecisionAboveSixteen => {
                j2k_core::CapabilityRejection::unsupported_bit_depth(reason)
            }
        })
    }

    pub(super) fn error_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat {
                fmt: PixelFormat::Rgba16,
            } => "J2K Metal does not support PixelFormat::Rgba16",
            Self::UnsupportedFormat { .. } => {
                "J2K Metal does not support the requested PixelFormat"
            }
            Self::PrecisionAboveSixteen => {
                "J2K Metal decode supports component precisions of at most 16 bits"
            }
        }
    }

    pub(super) fn profile_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat { .. } => "unsupported_format",
            Self::PrecisionAboveSixteen => "precision_above_sixteen",
        }
    }
}

pub(super) const fn unsupported_metal_format(fmt: PixelFormat) -> ExplicitMetalRejection {
    ExplicitMetalRejection::UnsupportedFormat { fmt }
}
