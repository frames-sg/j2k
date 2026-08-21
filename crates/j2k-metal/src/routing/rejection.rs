// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExplicitMetalRejection {
    UnsupportedFormat { fmt: PixelFormat },
}

impl ExplicitMetalRejection {
    pub(super) fn error_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat {
                fmt: PixelFormat::Rgba16,
            } => "J2K Metal does not support PixelFormat::Rgba16",
            Self::UnsupportedFormat { .. } => {
                "J2K Metal does not support the requested PixelFormat"
            }
        }
    }

    pub(super) fn profile_reason(self) -> &'static str {
        match self {
            Self::UnsupportedFormat { .. } => "unsupported_format",
        }
    }
}

pub(super) const fn unsupported_metal_format(fmt: PixelFormat) -> ExplicitMetalRejection {
    ExplicitMetalRejection::UnsupportedFormat { fmt }
}
