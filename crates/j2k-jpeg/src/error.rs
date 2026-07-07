// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed error and warning taxonomy. See spec Section 6.

use crate::info::{ColorSpace, Rect, SofKind};
use j2k_core::CodecError;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Reason an otherwise recognized SOF marker cannot be decoded.
pub enum UnsupportedReason {
    /// Stream uses arithmetic entropy coding.
    ArithmeticCoding,
    /// Stream uses hierarchical JPEG coding.
    Hierarchical,
    /// Stream combines arithmetic entropy coding and hierarchical coding.
    ArithmeticAndHierarchical,
    /// Stream uses a differential baseline SOF.
    DifferentialBaseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Huffman entropy decoder failure category.
pub enum HuffmanFailure {
    /// A Huffman code exceeded the representable code space.
    CodeOverflow,
    /// A decoded symbol is invalid for its context.
    InvalidSymbol,
    /// The entropy stream ended before a symbol could be decoded.
    TableExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Invalid decoder-builder input configuration.
pub enum BuilderConflictReason {
    /// No input source was provided.
    NoInput,
    /// Raw input bytes and scan fragments were both provided.
    InputAndScanFragments,
    /// Scan-fragment mode was selected without any fragments.
    ScanFragmentsEmpty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG table class used in diagnostics.
pub enum TableKind {
    /// Quantization table.
    Quant,
    /// AC Huffman table.
    HuffmanAc,
    /// DC Huffman table.
    HuffmanDc,
}

/// Fatal JPEG decode or API error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum JpegError {
    #[error("JPEG truncated at offset {offset}: expected {expected} more bytes")]
    /// Input ended before the requested bytes could be read.
    Truncated {
        /// Byte offset where decoding stopped.
        offset: usize,
        /// Additional byte count required.
        expected: usize,
    },

    #[error("invalid marker FF{marker:02X} at offset {offset}")]
    /// Marker byte is not legal in the current JPEG position.
    InvalidMarker {
        /// Byte offset of the marker.
        offset: usize,
        /// Raw marker byte following `0xff`.
        marker: u8,
    },

    #[error("expected {expected:?}, found FF{found:02X} at offset {offset}")]
    /// A different marker kind was required at this position.
    UnexpectedMarker {
        /// Byte offset of the unexpected marker.
        offset: usize,
        /// Marker kind expected by the parser.
        expected: MarkerKind,
        /// Raw marker byte that was found.
        found: u8,
    },

    #[error("missing required marker {marker:?}")]
    /// Required marker was absent from the stream.
    MissingMarker {
        /// Missing marker kind.
        marker: MarkerKind,
    },

    #[error("duplicate {marker:?} at offset {offset}")]
    /// A marker appeared more than once where only one is legal.
    DuplicateMarker {
        /// Byte offset of the duplicate marker.
        offset: usize,
        /// Duplicated marker kind.
        marker: MarkerKind,
    },

    #[error("invalid length {length} for marker FF{marker:02X} at offset {offset}")]
    /// Marker segment length is invalid for the marker kind.
    InvalidSegmentLength {
        /// Byte offset of the segment.
        offset: usize,
        /// Raw marker byte following `0xff`.
        marker: u8,
        /// Declared segment length.
        length: u16,
    },

    #[error("conflicting duplicate JPEG table {table:?} id={id} at offset {offset}")]
    /// Duplicate DQT/DHT table has different bytes from an earlier definition.
    ConflictingDuplicateTable {
        /// Byte offset of the conflicting table segment.
        offset: usize,
        /// Table class.
        table: TableKind,
        /// Table id.
        id: u8,
    },

    #[error("expected dimensions are required to repair zero SOF dimensions at offset {offset}")]
    /// TIFF/NDPI metadata did not provide dimensions needed to repair a zero SOF.
    ExpectedDimensionsRequired {
        /// Byte offset of the SOF marker.
        offset: usize,
    },

    #[error(
        "expected dimensions {expected:?} conflict with SOF dimensions {actual:?} at offset {offset}"
    )]
    /// Container-provided dimensions conflict with non-zero SOF dimensions.
    ConflictingExpectedDimensions {
        /// Byte offset of the SOF marker.
        offset: usize,
        /// Expected dimensions supplied by the caller.
        expected: (u16, u16),
        /// Dimensions declared by the SOF marker.
        actual: (u16, u16),
    },

    #[error("invalid TIFF JPEG assembly at offset {offset}: {reason}")]
    /// TIFF/JPEGTables assembly cannot produce a valid JPEG interchange stream.
    InvalidJpegAssembly {
        /// Byte offset of the assembly problem.
        offset: usize,
        /// Static diagnostic reason.
        reason: &'static str,
    },

    #[error("conflicting DRI values at offset {offset}: existing {existing}, new {new}")]
    /// Duplicate DRI marker conflicts with an earlier DRI value.
    ConflictingDri {
        /// Byte offset of the conflicting DRI segment.
        offset: usize,
        /// Existing non-zero restart interval.
        existing: u16,
        /// New non-zero restart interval.
        new: u16,
    },

    /// Unsupported SOF variant. Carries the raw marker byte (e.g. `0xC9` for
    /// arithmetic extended-sequential) so callers routing to a fallback
    /// decoder can distinguish FFC5 from FFC9 without relying on `reason`.
    #[error("unsupported SOF marker FF{marker:02X} ({reason:?})")]
    /// SOF marker class is outside the decoder's supported JPEG subset.
    UnsupportedSof {
        /// Raw SOF marker byte.
        marker: u8,
        /// Unsupported feature category.
        reason: UnsupportedReason,
    },

    #[error("unsupported component count: {count}")]
    /// Component count is outside the supported range.
    UnsupportedComponentCount {
        /// Declared component count.
        count: u8,
    },

    #[error("unsupported color space for decode: {color_space:?}")]
    /// Color space cannot be produced by the requested decode path.
    UnsupportedColorSpace {
        /// Header-derived color space.
        color_space: ColorSpace,
    },

    #[error("unsupported bit depth: {depth}")]
    /// Sample precision is not supported by this decoder path.
    UnsupportedBitDepth {
        /// Declared sample precision in bits.
        depth: u8,
    },

    #[error("unsupported lossless predictor: {predictor}")]
    /// Lossless predictor selection is unsupported.
    UnsupportedPredictor {
        /// Predictor value from the scan header.
        predictor: u8,
    },

    #[error("zero dimension in SOF: {width}×{height}")]
    /// SOF declares a zero width or height.
    ZeroDimension {
        /// Declared width.
        width: u16,
        /// Declared height.
        height: u16,
    },

    #[error("dimension overflow: {width}×{height} exceeds 65500")]
    /// Dimensions exceed this crate's safe decode bounds.
    DimensionOverflow {
        /// Declared width.
        width: u32,
        /// Declared height.
        height: u32,
    },

    #[error("invalid sampling ({h}×{v}) for component {component}")]
    /// Component sampling factors are outside the JPEG legal range.
    InvalidSampling {
        /// Component id.
        component: u8,
        /// Horizontal sampling factor.
        h: u8,
        /// Vertical sampling factor.
        v: u8,
    },

    #[error("missing quantization table {table_id} for component {component}")]
    /// Component references a quantization table that was not defined.
    MissingQuantTable {
        /// Component id.
        component: u8,
        /// Referenced quantization table id.
        table_id: u8,
    },

    #[error("missing Huffman table class={class} id={id} for component {component}")]
    /// Scan references a Huffman table that was not defined.
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
    /// Scan spectral selection or approximation parameters are invalid.
    InvalidScanParameters {
        /// Byte offset of the scan header.
        offset: usize,
        /// Start of spectral selection.
        ss: u8,
        /// End of spectral selection.
        se: u8,
        /// Successive approximation high bit.
        ah: u8,
        /// Successive approximation low bit.
        al: u8,
    },

    #[error("unknown scan component id {component} at offset {offset}")]
    /// Scan references a component id not declared in the SOF.
    UnknownScanComponent {
        /// Byte offset of the scan header.
        offset: usize,
        /// Unknown component id.
        component: u8,
    },

    #[error("duplicate scan component id {component} at offset {offset}")]
    /// Scan lists the same component more than once.
    DuplicateScanComponent {
        /// Byte offset of the scan header.
        offset: usize,
        /// Duplicated component id.
        component: u8,
    },

    #[error(
        "invalid sequential scan component set at offset {offset}: expected {expected} components, found {found}"
    )]
    /// Sequential scan does not contain the expected component set.
    InvalidSequentialComponentSet {
        /// Byte offset of the scan header.
        offset: usize,
        /// Expected component count.
        expected: u8,
        /// Found component count.
        found: u8,
    },

    #[error("invalid sequential scan count for {sof:?}: expected 1, found {count}")]
    /// Sequential SOF contained an invalid number of scans.
    InvalidSequentialScanCount {
        /// SOF kind being decoded.
        sof: SofKind,
        /// Observed scan count.
        count: u16,
    },

    #[error("Huffman decode failed near MCU {mcu}: {reason:?}")]
    /// Huffman entropy decoding failed. `mcu` is the current MCU when the
    /// caller has MCU progress, or `0` for table/bitstream contexts that do
    /// not track image position.
    HuffmanDecode {
        /// Current MCU index, or `0` when the decoder context has no MCU index.
        mcu: u32,
        /// Failure category.
        reason: HuffmanFailure,
    },

    #[error("restart mismatch at offset {offset}: expected RST{expected}, found FF{found:02X}")]
    /// Restart marker sequence did not match the expected RST index.
    RestartMismatch {
        /// Byte offset of the marker.
        offset: usize,
        /// Expected RST index.
        expected: u8,
        /// Found raw marker byte.
        found: u8,
    },

    #[error("unexpected EOI at MCU {mcu_at}/{mcu_total}")]
    /// EOI was reached before all MCUs were decoded.
    UnexpectedEoi {
        /// MCU index where EOI was found.
        mcu_at: u32,
        /// Total MCU count expected for the image.
        mcu_total: u32,
    },

    #[error("coefficient overflow at MCU {mcu}, component {component}")]
    /// Decoded coefficient exceeded the representable range.
    CoefficientOverflow {
        /// MCU index.
        mcu: u32,
        /// Component index.
        component: u8,
    },

    #[error("decode size {requested} bytes exceeds cap {cap} bytes")]
    /// Requested decode allocation exceeds the configured memory cap.
    MemoryCapExceeded {
        /// Requested byte count.
        requested: usize,
        /// Configured byte cap.
        cap: usize,
    },

    #[error("output buffer too small: need {required} bytes, got {provided}")]
    /// Caller-provided output buffer is too small.
    OutputBufferTooSmall {
        /// Required byte count.
        required: usize,
        /// Provided byte count.
        provided: usize,
    },

    #[error("stride {stride} smaller than row width {row}")]
    /// Output stride is smaller than the decoded row size.
    InvalidStride {
        /// Caller-provided stride.
        stride: usize,
        /// Minimum row byte count.
        row: usize,
    },

    #[error("rect {rect:?} out of image bounds ({width}×{height})")]
    /// Requested decode rectangle is outside image bounds.
    RectOutOfBounds {
        /// Requested rectangle.
        rect: Rect,
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },

    #[error("downscale not supported for {sof:?} streams")]
    /// Requested downscale is not supported for the SOF kind.
    DownscaleUnsupported {
        /// SOF kind being decoded.
        sof: SofKind,
    },

    #[error("scan fragments overlap at MCU {mcu}")]
    /// Builder-provided scan fragments overlap in MCU space.
    ScanFragmentsOverlap {
        /// First overlapping MCU index.
        mcu: u32,
    },

    #[error("builder input configuration conflict: {reason:?}")]
    /// Decoder builder inputs conflict.
    BuilderConflict {
        /// Conflict category.
        reason: BuilderConflictReason,
    },

    /// Transient pre-1.0 gap: the SOF is parseable and may eventually be
    /// supported by the decoder, but the current release does not implement
    /// the requested shape yet. Distinct from `UnsupportedSof` because callers
    /// routing to a fallback decoder on `is_unsupported()` should NOT reroute
    /// streams that a newer version of j2k will decode natively.
    #[error("decode not yet implemented for {sof:?} — see CHANGELOG for milestone")]
    NotImplemented {
        /// SOF kind awaiting implementation.
        sof: SofKind,
    },

    #[error("internal JPEG invariant failed: {reason}")]
    /// Decoder state violated an invariant that should have been enforced
    /// before reaching the current path.
    InternalInvariant {
        /// Static diagnostic reason.
        reason: &'static str,
    },

    #[error("row sink aborted decode")]
    /// Row sink returned an error and aborted row-based decoding.
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
    /// is valid and will decode on a future j2k release, so callers
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

#[doc(hidden)]
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
    /// Stream ended without an EOI marker after otherwise decodable entropy.
    MissingEoi,
    /// SOF dimensions were repaired from external context.
    SofDimensionsPatched {
        /// Original SOF dimensions.
        from: (u16, u16),
        /// Replacement dimensions.
        to: (u16, u16),
    },
    /// Stream uses nonstandard but decodable table layout.
    NonstandardTables,
    /// Adobe APP14 transform value could not unambiguously define color.
    AdobeApp14Ambiguous {
        /// Raw APP14 transform byte.
        raw_transform: u8,
    },
    /// ICC profile was present but ignored by this decoder.
    IccProfileIgnored {
        /// ICC payload size in bytes.
        size: usize,
    },
    /// Unknown APP marker was skipped.
    UnknownAppMarker {
        /// Raw APP marker byte.
        marker: u8,
        /// Segment payload size in bytes.
        size: usize,
    },
    /// Decoder recovered at a restart marker.
    RestartRecovered {
        /// Byte offset near the recovered restart marker.
        offset: usize,
    },
    /// Higher-precision samples were clamped to a lower output precision.
    PrecisionClamped {
        /// Source precision in bits.
        from_bits: u8,
        /// Output precision in bits.
        to_bits: u8,
    },
    /// Color profile metadata was present but unrecognized.
    UnknownColorProfile,
    /// Cached table metadata disagreed with the active stream tables.
    TableCacheMismatch {
        /// Table class.
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
