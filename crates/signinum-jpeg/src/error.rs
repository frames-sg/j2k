// SPDX-License-Identifier: Apache-2.0

//! Typed error and warning taxonomy. See spec Section 6.

use crate::info::{ColorSpace, Rect, SofKind};
use signinum_core::CodecError;

/// A category of JPEG marker. Carried in [`JpegError::UnexpectedMarker`] and
/// related variants so callers can branch on marker class without parsing the
/// raw byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    /// Start of image (`FFD8`).
    Soi,
    /// Start of frame (any of `FFC0..=FFC3`).
    Sof,
    /// Define quantization table (`FFDB`).
    Dqt,
    /// Define Huffman table (`FFC4`).
    Dht,
    /// Define restart interval (`FFDD`).
    Dri,
    /// Start of scan (`FFDA`).
    Sos,
    /// End of image (`FFD9`).
    Eoi,
    /// Adobe APP14 (`FFEE`).
    App14,
    /// Any other marker, raw byte preserved.
    Other(u8),
}

/// Reason an SOF marker is permanently unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedReason {
    /// Arithmetic entropy coding is unsupported.
    ArithmeticCoding,
    /// Hierarchical JPEG is unsupported.
    Hierarchical,
    /// Arithmetic entropy coding and hierarchical mode are both present.
    ArithmeticAndHierarchical,
    /// Differential baseline coding is unsupported.
    DifferentialBaseline,
}

/// Reason Huffman entropy decode failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuffmanFailure {
    /// Huffman code construction overflowed valid code space.
    CodeOverflow,
    /// Decoded symbol was invalid for the current table.
    InvalidSymbol,
    /// Huffman table was exhausted before a valid symbol.
    TableExhausted,
}

/// Conflicting input configuration in a decode builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuilderConflictReason {
    /// No input bytes or fragments were provided.
    NoInput,
    /// Both contiguous input and scan fragments were provided.
    InputAndScanFragments,
    /// Fragmented scan input was empty.
    ScanFragmentsEmpty,
}

/// JPEG table class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    /// Quantization table.
    Quant,
    /// AC Huffman table.
    HuffmanAc,
    /// DC Huffman table.
    HuffmanDc,
}

/// Error returned by JPEG inspect and decode APIs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum JpegError {
    /// Input ended before required bytes were available.
    #[error("JPEG truncated at offset {offset}: expected {expected} more bytes")]
    Truncated {
        /// Byte offset where truncation was detected.
        offset: usize,
        /// Additional bytes required.
        expected: usize,
    },

    /// Invalid marker byte sequence.
    #[error("invalid marker FF{marker:02X} at offset {offset}")]
    InvalidMarker {
        /// Byte offset of the marker.
        offset: usize,
        /// Marker byte following `0xFF`.
        marker: u8,
    },

    /// Marker did not match the parser state.
    #[error("expected {expected:?}, found FF{found:02X} at offset {offset}")]
    UnexpectedMarker {
        /// Byte offset of the marker.
        offset: usize,
        /// Expected marker category.
        expected: MarkerKind,
        /// Found marker byte.
        found: u8,
    },

    /// Required marker was absent.
    #[error("missing required marker {marker:?}")]
    MissingMarker {
        /// Missing marker category.
        marker: MarkerKind,
    },

    /// Marker appeared more than once where only one is allowed.
    #[error("duplicate {marker:?} at offset {offset}")]
    DuplicateMarker {
        /// Byte offset of the duplicate marker.
        offset: usize,
        /// Duplicate marker category.
        marker: MarkerKind,
    },

    /// Marker segment length was invalid.
    #[error("invalid length {length} for marker FF{marker:02X} at offset {offset}")]
    InvalidSegmentLength {
        /// Byte offset of the segment.
        offset: usize,
        /// Marker byte.
        marker: u8,
        /// Segment length from the marker payload.
        length: u16,
    },

    /// Unsupported SOF variant. Carries the raw marker byte (e.g. `0xC9` for
    /// arithmetic extended-sequential) so callers routing to a fallback
    /// decoder can distinguish FFC5 from FFC9 without relying on `reason`.
    #[error("unsupported SOF marker FF{marker:02X} ({reason:?})")]
    UnsupportedSof {
        /// Raw SOF marker byte.
        marker: u8,
        /// Unsupported SOF reason.
        reason: UnsupportedReason,
    },

    /// Component count is outside the supported public decode surface.
    #[error("unsupported component count: {count}")]
    UnsupportedComponentCount {
        /// Component count from the frame header.
        count: u8,
    },

    /// Color space cannot be decoded by this release.
    #[error("unsupported color space for decode: {color_space:?}")]
    UnsupportedColorSpace {
        /// Unsupported color space.
        color_space: ColorSpace,
    },

    /// Bit depth cannot be decoded by this release.
    #[error("unsupported bit depth: {depth}")]
    UnsupportedBitDepth {
        /// Unsupported bit depth.
        depth: u8,
    },

    /// Lossless predictor is unsupported.
    #[error("unsupported lossless predictor: {predictor}")]
    UnsupportedPredictor {
        /// Predictor value.
        predictor: u8,
    },

    /// SOF declared a zero width or height.
    #[error("zero dimension in SOF: {width}×{height}")]
    ZeroDimension {
        /// Declared width.
        width: u16,
        /// Declared height.
        height: u16,
    },

    /// Dimensions exceed implementation limits.
    #[error("dimension overflow: {width}×{height} exceeds 65500")]
    DimensionOverflow {
        /// Declared width.
        width: u32,
        /// Declared height.
        height: u32,
    },

    /// Invalid component sampling factor.
    #[error("invalid sampling ({h}×{v}) for component {component}")]
    InvalidSampling {
        /// Component id.
        component: u8,
        /// Horizontal sampling factor.
        h: u8,
        /// Vertical sampling factor.
        v: u8,
    },

    /// Required quantization table is missing.
    #[error("missing quantization table {table_id} for component {component}")]
    MissingQuantTable {
        /// Component id.
        component: u8,
        /// Quantization table id.
        table_id: u8,
    },

    /// Required Huffman table is missing.
    #[error("missing Huffman table class={class} id={id} for component {component}")]
    MissingHuffmanTable {
        /// Component id.
        component: u8,
        /// Huffman table class.
        class: u8,
        /// Huffman table id.
        id: u8,
    },

    #[error(
        "invalid sequential scan parameters at offset {offset}: Ss={ss} Se={se} Ah={ah} Al={al}"
    )]
    /// Sequential scan parameters are invalid for the SOF.
    InvalidScanParameters {
        /// Byte offset of SOS.
        offset: usize,
        /// Spectral selection start.
        ss: u8,
        /// Spectral selection end.
        se: u8,
        /// Successive approximation high bit.
        ah: u8,
        /// Successive approximation low bit.
        al: u8,
    },

    /// Scan referenced an unknown component id.
    #[error("unknown scan component id {component} at offset {offset}")]
    UnknownScanComponent {
        /// Byte offset of SOS.
        offset: usize,
        /// Component id.
        component: u8,
    },

    /// Scan listed a component id more than once.
    #[error("duplicate scan component id {component} at offset {offset}")]
    DuplicateScanComponent {
        /// Byte offset of SOS.
        offset: usize,
        /// Component id.
        component: u8,
    },

    #[error(
        "invalid sequential scan component set at offset {offset}: expected {expected} components, found {found}"
    )]
    /// Sequential scan component set does not match the frame.
    InvalidSequentialComponentSet {
        /// Byte offset of SOS.
        offset: usize,
        /// Expected component count.
        expected: u8,
        /// Found component count.
        found: u8,
    },

    /// Sequential stream used an invalid number of scans.
    #[error("invalid sequential scan count for {sof:?}: expected 1, found {count}")]
    InvalidSequentialScanCount {
        /// Start-of-frame kind.
        sof: SofKind,
        /// Observed scan count.
        count: u16,
    },

    /// Huffman entropy decode failed for an MCU.
    #[error("Huffman decode failed at MCU {mcu}: {reason:?}")]
    HuffmanDecode {
        /// MCU index.
        mcu: u32,
        /// Failure reason.
        reason: HuffmanFailure,
    },

    #[error("restart mismatch at offset {offset}: expected RST{expected}, found FF{found:02X}")]
    /// Restart marker did not match the expected restart index.
    RestartMismatch {
        /// Byte offset of the restart marker.
        offset: usize,
        /// Expected restart marker index.
        expected: u8,
        /// Found marker byte.
        found: u8,
    },

    /// End-of-image marker appeared before all MCUs were decoded.
    #[error("unexpected EOI at MCU {mcu_at}/{mcu_total}")]
    UnexpectedEoi {
        /// MCU index at EOI.
        mcu_at: u32,
        /// Total MCU count expected.
        mcu_total: u32,
    },

    /// Entropy coefficient exceeded the supported numeric range.
    #[error("coefficient overflow at MCU {mcu}, component {component}")]
    CoefficientOverflow {
        /// MCU index.
        mcu: u32,
        /// Component id.
        component: u8,
    },

    /// Decode allocation would exceed the configured cap.
    #[error("decode size {requested} bytes exceeds cap {cap} bytes")]
    MemoryCapExceeded {
        /// Requested byte count.
        requested: usize,
        /// Maximum allowed byte count.
        cap: usize,
    },

    /// Caller-owned output buffer is too small.
    #[error("output buffer too small: need {required} bytes, got {provided}")]
    OutputBufferTooSmall {
        /// Required byte count.
        required: usize,
        /// Provided byte count.
        provided: usize,
    },

    /// Caller-owned row stride is too small.
    #[error("stride {stride} smaller than row width {row}")]
    InvalidStride {
        /// Provided stride.
        stride: usize,
        /// Required row byte count.
        row: usize,
    },

    /// Requested region is outside image bounds.
    #[error("rect {rect:?} out of image bounds ({width}×{height})")]
    RectOutOfBounds {
        /// Requested rectangle.
        rect: Rect,
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },

    /// Downscale is unsupported for this SOF.
    #[error("downscale not supported for {sof:?} streams")]
    DownscaleUnsupported {
        /// Start-of-frame kind.
        sof: SofKind,
    },

    /// Fragmented scan inputs overlap at an MCU boundary.
    #[error("scan fragments overlap at MCU {mcu}")]
    ScanFragmentsOverlap {
        /// First overlapping MCU index.
        mcu: u32,
    },

    /// Decode builder inputs conflict.
    #[error("builder input configuration conflict: {reason:?}")]
    BuilderConflict {
        /// Conflict reason.
        reason: BuilderConflictReason,
    },

    /// Transient pre-1.0 gap: the SOF is parseable and will eventually be
    /// supported by the decoder, but the current release does not implement
    /// it yet. M3 removes this variant by implementing Extended12,
    /// Progressive12, and Lossless. Distinct from `UnsupportedSof` because
    /// callers routing
    /// to a fallback decoder on `is_unsupported()` should NOT reroute streams
    /// that a newer version of signinum will decode natively.
    #[error("decode not yet implemented for {sof:?} — see CHANGELOG for milestone")]
    NotImplemented {
        /// Start-of-frame kind.
        sof: SofKind,
    },

    /// Caller-provided row sink aborted decode.
    #[error("row sink aborted decode")]
    RowSinkAborted,
}

impl JpegError {
    /// True if the error is recoverable by routing to a different decoder —
    /// any `Unsupported*` variant.
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedSof { .. }
                | Self::UnsupportedComponentCount { .. }
                | Self::UnsupportedColorSpace { .. }
                | Self::UnsupportedBitDepth { .. }
                | Self::UnsupportedPredictor { .. }
        )
    }

    /// True if the input was truncated — caller may retry with more bytes.
    pub fn is_truncated(&self) -> bool {
        matches!(self, Self::Truncated { .. } | Self::UnexpectedEoi { .. })
    }

    /// True if the error indicates caller misuse, not a decode failure.
    pub fn is_api_misuse(&self) -> bool {
        matches!(
            self,
            Self::OutputBufferTooSmall { .. }
                | Self::InvalidStride { .. }
                | Self::RectOutOfBounds { .. }
                | Self::DownscaleUnsupported { .. }
                | Self::ScanFragmentsOverlap { .. }
                | Self::BuilderConflict { .. }
        )
    }

    /// True if the error is a transient "not yet implemented" gap — the stream
    /// is valid and will decode on a future signinum release, so callers
    /// should *not* reroute to a different decoder permanently. See
    /// [`Self::is_unsupported`] for errors that are permanent routing decisions.
    pub fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented { .. })
    }

    /// Byte offset where the error was detected in the input stream, if any.
    pub fn offset(&self) -> Option<usize> {
        match self {
            Self::Truncated { offset, .. }
            | Self::InvalidMarker { offset, .. }
            | Self::UnexpectedMarker { offset, .. }
            | Self::DuplicateMarker { offset, .. }
            | Self::InvalidSegmentLength { offset, .. }
            | Self::InvalidScanParameters { offset, .. }
            | Self::UnknownScanComponent { offset, .. }
            | Self::DuplicateScanComponent { offset, .. }
            | Self::InvalidSequentialComponentSet { offset, .. }
            | Self::RestartMismatch { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}

impl CodecError for JpegError {
    fn is_truncated(&self) -> bool {
        Self::is_truncated(self)
    }

    fn is_not_implemented(&self) -> bool {
        Self::is_not_implemented(self)
    }

    fn is_unsupported(&self) -> bool {
        Self::is_unsupported(self)
    }

    fn is_buffer_error(&self) -> bool {
        matches!(
            self,
            Self::OutputBufferTooSmall { .. }
                | Self::InvalidStride { .. }
                | Self::RectOutOfBounds { .. }
        )
    }
}

/// Non-fatal notices emitted during decode. See spec Section 6.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Warning {
    /// Input ended without an EOI marker.
    MissingEoi,
    /// SOF dimensions were patched from container metadata.
    SofDimensionsPatched {
        /// Dimensions declared by SOF.
        from: (u16, u16),
        /// Replacement dimensions.
        to: (u16, u16),
    },
    /// Tables are valid but nonstandard.
    NonstandardTables,
    /// Adobe APP14 transform was ambiguous.
    AdobeApp14Ambiguous {
        /// Raw transform byte.
        raw_transform: u8,
    },
    /// ICC profile was ignored by the decoder.
    IccProfileIgnored {
        /// ICC payload size.
        size: usize,
    },
    /// Unknown APP marker was skipped.
    UnknownAppMarker {
        /// Marker byte.
        marker: u8,
        /// Segment size.
        size: usize,
    },
    /// Decode recovered at a restart marker.
    RestartRecovered {
        /// Recovery byte offset.
        offset: usize,
    },
    /// Sample precision was clamped.
    PrecisionClamped {
        /// Original bit precision.
        from_bits: u8,
        /// Output bit precision.
        to_bits: u8,
    },
    /// Color profile could not be interpreted.
    UnknownColorProfile,
    /// Cached table metadata did not match the stream.
    TableCacheMismatch {
        /// Table kind.
        which: TableKind,
        /// Table id.
        id: u8,
    },
}

impl core::fmt::Display for Warning {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingEoi => f.write_str("missing EOI"),
            Self::SofDimensionsPatched { from, to } => {
                write!(f, "patched SOF dimensions from {from:?} to {to:?}")
            }
            Self::NonstandardTables => f.write_str("nonstandard tables"),
            Self::AdobeApp14Ambiguous { raw_transform } => {
                write!(f, "ambiguous Adobe APP14 transform {raw_transform}")
            }
            Self::IccProfileIgnored { size } => write!(f, "ignored ICC profile of {size} bytes"),
            Self::UnknownAppMarker { marker, size } => {
                write!(f, "unknown APP marker FF{marker:02X} ({size} bytes)")
            }
            Self::RestartRecovered { offset } => {
                write!(f, "recovered at restart marker near offset {offset}")
            }
            Self::PrecisionClamped { from_bits, to_bits } => {
                write!(
                    f,
                    "precision clamped from {from_bits} bits to {to_bits} bits"
                )
            }
            Self::UnknownColorProfile => f.write_str("unknown color profile"),
            Self::TableCacheMismatch { which, id } => {
                write!(f, "table cache mismatch for {which:?} {id}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::ColorSpace;

    #[test]
    fn unsupported_predicate_matches_only_unsupported_variants() {
        assert!(JpegError::UnsupportedSof {
            marker: 0xC9,
            reason: UnsupportedReason::ArithmeticCoding,
        }
        .is_unsupported());
        assert!(JpegError::UnsupportedColorSpace {
            color_space: ColorSpace::Cmyk,
        }
        .is_unsupported());
        assert!(JpegError::UnsupportedBitDepth { depth: 16 }.is_unsupported());
        assert!(!JpegError::Truncated {
            offset: 0,
            expected: 1
        }
        .is_unsupported());
    }

    #[test]
    fn truncated_predicate_covers_truncation_and_unexpected_eoi() {
        assert!(JpegError::Truncated {
            offset: 10,
            expected: 5
        }
        .is_truncated());
        assert!(JpegError::UnexpectedEoi {
            mcu_at: 3,
            mcu_total: 10
        }
        .is_truncated());
        assert!(!JpegError::InvalidMarker {
            offset: 4,
            marker: 0xFF
        }
        .is_truncated());
    }

    #[test]
    fn api_misuse_predicate_covers_caller_bugs() {
        assert!(JpegError::OutputBufferTooSmall {
            required: 100,
            provided: 64
        }
        .is_api_misuse());
        assert!(JpegError::InvalidStride { stride: 2, row: 8 }.is_api_misuse());
        assert!(JpegError::BuilderConflict {
            reason: BuilderConflictReason::NoInput
        }
        .is_api_misuse());
        assert!(!JpegError::Truncated {
            offset: 0,
            expected: 1
        }
        .is_api_misuse());
    }

    #[test]
    fn offset_returns_some_for_byte_positioned_errors() {
        assert_eq!(
            JpegError::InvalidMarker {
                offset: 42,
                marker: 0xBA
            }
            .offset(),
            Some(42),
        );
        assert_eq!(JpegError::UnsupportedBitDepth { depth: 16 }.offset(), None,);
    }

    #[test]
    fn not_implemented_predicate_distinguishes_from_unsupported() {
        let not_impl = JpegError::NotImplemented {
            sof: SofKind::Progressive8,
        };
        assert!(not_impl.is_not_implemented());
        assert!(
            !not_impl.is_unsupported(),
            "NotImplemented is a transient M1b/M2 gap — callers routing on is_unsupported() must NOT \
             reroute these streams, because M3 adds real support"
        );
        assert!(!not_impl.is_truncated());
        assert!(!not_impl.is_api_misuse());

        let unsupported = JpegError::UnsupportedSof {
            marker: 0xC9,
            reason: UnsupportedReason::ArithmeticCoding,
        };
        assert!(!unsupported.is_not_implemented());
        assert!(unsupported.is_unsupported());
    }
}
