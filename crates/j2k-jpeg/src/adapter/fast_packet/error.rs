// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::JpegError;
use crate::info::{ColorSpace, SofKind};
use j2k_core::CodecError;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[doc(hidden)]
/// Error while building a backend fast-path JPEG packet.
pub enum FastPacketError {
    /// Header or entropy decode failed.
    #[error("{0}")]
    Decode(#[from] JpegError),
    /// JPEG SOF kind is not supported by the fast path.
    #[error("JPEG fast packet does not support SOF kind {0:?}")]
    UnsupportedSof(SofKind),
    /// JPEG color space is not supported by the selected fast path.
    #[error("JPEG fast packet does not support color space {0:?}")]
    UnsupportedColorSpace(ColorSpace),
    /// JPEG component sampling does not match the selected fast path.
    #[error("JPEG component sampling does not match the selected fast-packet family")]
    UnsupportedSampling,
    /// Scan component order does not match SOF component order.
    #[error("JPEG scan component order does not match the fast-packet contract")]
    UnsupportedComponentOrder,
    /// Stream does not contain a scan payload.
    #[error("JPEG fast packet input has no scan payload")]
    MissingScan,
    /// Referenced quantization table is absent.
    #[error("JPEG fast packet input is missing quantization table {slot}")]
    MissingQuantTable {
        /// Quantization table slot.
        slot: u8,
    },
    /// Referenced Huffman table is absent.
    #[error("JPEG fast packet input is missing {kind:?} Huffman table {slot}")]
    MissingHuffmanTable {
        /// Huffman table class.
        kind: TableKind,
        /// Huffman table slot.
        slot: u8,
    },
    /// Entropy payload contains a marker unsupported by the fast path.
    #[error("JPEG fast packet does not support entropy marker 0xff{marker:02x}")]
    EntropyMarkerUnsupported {
        /// Raw marker byte following `0xff`.
        marker: u8,
    },
    /// Entropy payload ended before the packet could be built.
    #[error("JPEG entropy payload ended before fast-packet construction completed")]
    TruncatedEntropy,
}

impl FastPacketError {
    /// Whether this is an ordinary fast-path capability mismatch rather than a
    /// malformed-input, resource, or internal failure.
    #[must_use]
    pub const fn is_capability_mismatch(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedSof(_)
                | Self::UnsupportedColorSpace(_)
                | Self::UnsupportedSampling
                | Self::UnsupportedComponentOrder
                | Self::EntropyMarkerUnsupported { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
/// Huffman table class used by fast-path packet builders.
pub enum TableKind {
    /// DC Huffman table.
    Dc,
    /// AC Huffman table.
    Ac,
}

#[doc(hidden)]
impl CodecError for FastPacketError {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::TruncatedEntropy)
            || matches!(self, Self::Decode(error) if error.is_truncated())
    }

    fn is_not_implemented(&self) -> bool {
        self.is_capability_mismatch()
            || matches!(self, Self::Decode(error) if error.is_not_implemented())
    }

    fn is_unsupported(&self) -> bool {
        self.is_capability_mismatch()
            || matches!(self, Self::Decode(error) if error.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Decode(error) if error.is_buffer_error())
    }
}
