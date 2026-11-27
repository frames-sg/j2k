// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode and encode errors.

use thiserror::Error;

use crate::info::InterleaveMode;

/// A JPEG-LS feature this crate recognizes but does not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum UnsupportedFeature {
    /// The frame is not a JPEG-LS frame (`SOF55`), for example baseline or
    /// lossless JPEG.
    #[error("frame marker FF{marker:02X} is not a JPEG-LS frame")]
    NonJpegLsFrame {
        /// The start-of-frame marker code.
        marker: u8,
    },
    /// A mapping table (LSE ID 2 or 3, or a nonzero table index in a scan).
    #[error("mapping tables")]
    MappingTable,
    /// Image dimensions above 65535 (LSE ID 4).
    #[error("oversize image dimensions (LSE ID 4)")]
    OversizeDimensions,
    /// A component with sampling factors other than 1x1.
    #[error("component {component_id} has sampling factors {horizontal}x{vertical}")]
    SamplingFactors {
        /// Component identifier from the frame header.
        component_id: u8,
        /// Horizontal sampling factor.
        horizontal: u8,
        /// Vertical sampling factor.
        vertical: u8,
    },
    /// A scan with a nonzero point transform.
    #[error("point transform {0}")]
    PointTransform(u8),
    /// An HP color transform (APP8 `mrfx`) other than none.
    #[error("HP color transform {0}")]
    ColorTransform(u8),
    /// A frame whose height is set later by a DNL marker.
    #[error("frame height defined by a DNL marker")]
    DeferredHeight,
}

/// Errors returned while inspecting or decoding JPEG-LS data.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JpegLsError {
    /// The data does not start with an SOI marker.
    #[error("data does not start with a JPEG SOI marker")]
    MissingSoi,
    /// The data ended before a complete marker segment or scan.
    #[error("JPEG-LS data truncated at offset {offset}")]
    Truncated {
        /// Byte offset where more data was needed.
        offset: usize,
    },
    /// A required marker never appeared.
    #[error("missing {marker} marker")]
    MissingMarker {
        /// Marker name, such as `SOF55`, `SOS`, or `EOI`.
        marker: &'static str,
    },
    /// A marker appeared where it is not allowed, or bytes that are not a
    /// marker appeared where one was expected.
    #[error("unexpected marker FF{marker:02X} at offset {offset}")]
    UnexpectedMarker {
        /// The marker code, or the byte found instead of `FF`.
        marker: u8,
        /// Byte offset of the marker.
        offset: usize,
    },
    /// A marker segment has an invalid length or field.
    #[error("invalid FF{marker:02X} segment at offset {offset}: {reason}")]
    InvalidSegment {
        /// The marker code of the segment.
        marker: u8,
        /// Byte offset of the marker.
        offset: usize,
        /// What is wrong with the segment.
        reason: &'static str,
    },
    /// Preset coding parameters (LSE ID 1) are out of range for a scan.
    #[error("invalid preset coding parameters: {reason}")]
    InvalidPresetParameters {
        /// Which parameter is out of range.
        reason: &'static str,
    },
    /// A frame component is not coded by any scan.
    #[error("component {component_id} is not coded by any scan")]
    MissingComponent {
        /// Component identifier from the frame header.
        component_id: u8,
    },
    /// The entropy-coded data of a scan is invalid.
    #[error("invalid entropy-coded data in scan {scan} at line {line}")]
    InvalidEntropyData {
        /// Zero-based scan index.
        scan: usize,
        /// Zero-based image line.
        line: u32,
    },
    /// A scan has entropy-coded data left over after its last line.
    #[error("extra entropy-coded data at the end of scan {scan}")]
    TrailingEntropyData {
        /// Zero-based scan index.
        scan: usize,
    },
    /// A restart marker is missing or out of sequence.
    #[error("expected restart marker RST{expected} at offset {offset}")]
    RestartMarker {
        /// The restart marker number (0 to 7) that should appear.
        expected: u8,
        /// Byte offset where it should appear.
        offset: usize,
    },
    /// The feature is valid JPEG-LS but this crate does not implement it.
    #[error("unsupported JPEG-LS feature: {0}")]
    Unsupported(#[from] UnsupportedFeature),
    /// The decoded image would exceed the output size limit.
    #[error("decoded image needs {required} bytes, above the {limit}-byte limit")]
    OutputLimitExceeded {
        /// Bytes the decoded image needs, saturated at `u64::MAX`.
        required: u64,
        /// The limit from [`DecodeOptions`](crate::DecodeOptions).
        limit: usize,
    },
    /// The output buffer cannot hold the image at the given stride.
    #[error("output buffer holds {actual} bytes, the image needs {required}")]
    OutputBufferTooSmall {
        /// Bytes needed.
        required: usize,
        /// Bytes provided.
        actual: usize,
    },
    /// The output stride is shorter than one row of samples.
    #[error("stride {stride} is shorter than a {row_bytes}-byte row")]
    InvalidStride {
        /// The stride passed in, in bytes.
        stride: usize,
        /// Bytes in one row of samples.
        row_bytes: usize,
    },
    /// The output buffer could not be allocated.
    #[error("could not allocate {bytes} bytes for the decoded image")]
    AllocationFailed {
        /// Requested allocation size.
        bytes: usize,
    },
}

/// Errors returned by [`encode_jpegls`](crate::encode_jpegls).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum JpegLsEncodeError {
    /// Width or height is zero.
    #[error("JPEG-LS encode requires nonzero dimensions")]
    EmptyDimensions,
    /// Width or height is above 65535.
    #[error("JPEG-LS dimensions must fit in 16 bits, got {width}x{height}")]
    DimensionsTooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The component count is zero.
    #[error("JPEG-LS encode requires at least one component")]
    NoComponents,
    /// The bit depth is outside 2 to 16, or above 8 for 8-bit samples.
    #[error("bit depth {bit_depth} is not valid for {sample_type} samples")]
    InvalidBitDepth {
        /// Requested bits per sample.
        bit_depth: u8,
        /// `u8` or `u16`.
        sample_type: &'static str,
    },
    /// The sample slice length does not match the dimensions.
    #[error("sample buffer holds {actual} samples, expected {expected}")]
    SampleLength {
        /// `width * height * components`.
        expected: usize,
        /// Samples supplied.
        actual: usize,
    },
    /// A sample is above the maximum sample value.
    #[error("sample {index} has value {value}, above the maximum {max_value}")]
    SampleOutOfRange {
        /// Index of the sample in the input slice.
        index: usize,
        /// The sample value.
        value: u16,
        /// The largest value the options allow.
        max_value: u16,
    },
    /// NEAR is above `min(255, MAXVAL / 2)`.
    #[error("NEAR {near} is above the maximum {max} for this maximum sample value")]
    InvalidNear {
        /// Requested NEAR.
        near: u8,
        /// Largest valid NEAR.
        max: u16,
    },
    /// Line and sample interleaving code at most four components per scan.
    #[error("{interleave:?} interleaving supports at most 4 components, got {components}")]
    InvalidInterleave {
        /// Requested interleave mode.
        interleave: InterleaveMode,
        /// Component count.
        components: u8,
    },
    /// The preset coding parameters are out of range.
    #[error("invalid preset coding parameters: {reason}")]
    InvalidPresetParameters {
        /// Which parameter is out of range.
        reason: &'static str,
    },
    /// The restart interval is zero.
    #[error("restart interval must be nonzero when provided")]
    InvalidRestartInterval,
    /// The encoded output would be too large to allocate.
    #[error("could not allocate {bytes} bytes for the encoded output")]
    AllocationFailed {
        /// Requested allocation size.
        bytes: usize,
    },
    /// Decoding the encoder's output failed.
    #[error("decoding the encoded output failed: {0}")]
    ValidationDecode(#[source] JpegLsError),
    /// Decoding the encoder's output gave samples outside the allowed error.
    #[error("sample {index} decoded to {decoded}, expected {expected} within NEAR {near}")]
    ValidationMismatch {
        /// Index of the first sample outside the allowed error.
        index: usize,
        /// The input sample.
        expected: u16,
        /// The decoded sample.
        decoded: u16,
        /// The allowed absolute error.
        near: u8,
    },
}
