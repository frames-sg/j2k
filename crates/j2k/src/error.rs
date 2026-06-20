// SPDX-License-Identifier: Apache-2.0

use j2k_core::{BufferError, CodecError, InputError, NotImplemented, Unsupported};

/// Error returned by JPEG 2000 inspect, decode, encode, and recode APIs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum J2kError {
    /// Caller-owned buffers were too small or malformed.
    #[error(transparent)]
    Buffer(#[from] BufferError),

    /// Input was too short or truncated while parsing.
    #[error(transparent)]
    Input(#[from] InputError),

    /// Requested feature is planned but not implemented.
    #[error(transparent)]
    NotImplemented(#[from] NotImplemented),

    /// Requested input feature or option is unsupported.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),

    /// Native backend or encode/decode stage failed.
    #[error("backend failed: {0}")]
    Backend(String),

    /// Caller-provided encode samples were malformed.
    #[error("invalid JPEG 2000 samples: {what}")]
    InvalidSamples {
        /// Description of the invalid sample condition.
        what: String,
    },

    /// Lossy rate-control search could not satisfy the requested target.
    #[error("JPEG 2000 lossy rate target unreachable: {target}, best {best}")]
    RateTargetUnreachable {
        /// Requested rate target.
        target: String,
        /// Best achievable result observed by the search.
        best: String,
    },

    /// Requested region lies outside image bounds.
    #[error("region ({x},{y} {w}x{h}) is outside image bounds {image_w}x{image_h}")]
    InvalidRegion {
        /// Region left coordinate.
        x: u32,
        /// Region top coordinate.
        y: u32,
        /// Region width.
        w: u32,
        /// Region height.
        h: u32,
        /// Image width.
        image_w: u32,
        /// Image height.
        image_h: u32,
    },

    /// JP2 box structure was invalid.
    #[error("invalid JP2 box at offset {offset}: {what}")]
    InvalidBox {
        /// Byte offset of the invalid box.
        offset: usize,
        /// Description of the invalid box condition.
        what: &'static str,
    },

    /// Required JP2 box was absent.
    #[error("missing required JP2 box {box_type}")]
    MissingRequiredBox {
        /// Missing box type.
        box_type: &'static str,
    },

    /// Codestream marker was invalid.
    #[error("invalid codestream marker FF{marker:02X} at offset {offset}")]
    InvalidMarker {
        /// Byte offset of the invalid marker.
        offset: usize,
        /// Marker byte following the `0xFF` prefix.
        marker: u8,
    },

    /// Required codestream marker was absent.
    #[error("missing required codestream marker {marker}")]
    MissingRequiredMarker {
        /// Missing marker name.
        marker: &'static str,
    },

    /// SIZ segment was invalid.
    #[error("invalid SIZ segment: {what}")]
    InvalidSiz {
        /// Description of the invalid SIZ condition.
        what: &'static str,
    },

    /// COD segment was invalid.
    #[error("invalid COD segment: {what}")]
    InvalidCod {
        /// Description of the invalid COD condition.
        what: &'static str,
    },

    /// Image dimensions overflowed a size computation.
    #[error("dimension overflow: {width}x{height}")]
    DimensionOverflow {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
}

impl CodecError for J2kError {
    fn is_truncated(&self) -> bool {
        matches!(
            self,
            Self::Input(InputError::TooShort { .. } | InputError::TruncatedAt { .. })
        )
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented(_))
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }
}
