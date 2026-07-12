// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public data and error contracts for baseline JPEG encoding.

use alloc::vec::Vec;

use thiserror::Error;

use crate::dct_contract::JpegDctImageError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG encoder backend selector.
pub enum JpegBackend {
    /// Choose the best available backend for the platform.
    Auto,
    /// Use the portable CPU encoder.
    Cpu,
    /// Use a Metal encoder when called through the Metal integration.
    Metal,
    /// Use a CUDA encoder when called through the CUDA integration.
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// JPEG baseline chroma subsampling mode.
pub enum JpegSubsampling {
    /// Single-component grayscale.
    Gray,
    /// Three-component YBR/RGB 4:4:4 sampling.
    Ybr444,
    /// Three-component YBR/RGB 4:2:2 sampling.
    Ybr422,
    /// Three-component YBR/RGB 4:2:0 sampling.
    Ybr420,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Options controlling baseline JPEG encoding.
pub struct JpegEncodeOptions {
    /// JPEG quality in the conventional 1..=100 range.
    pub quality: u8,
    /// Output component sampling.
    pub subsampling: JpegSubsampling,
    /// Optional restart interval in MCUs.
    pub restart_interval: Option<u16>,
    /// Requested encoder backend.
    pub backend: JpegBackend,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            subsampling: JpegSubsampling::Ybr422,
            restart_interval: None,
            backend: JpegBackend::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// Borrowed input samples for baseline JPEG encoding.
pub enum JpegSamples<'a> {
    /// Interleaved 8-bit grayscale samples.
    Gray8 {
        /// Pixel data, one byte per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// Interleaved 8-bit RGB samples.
    Rgb8 {
        /// Pixel data, three bytes per pixel.
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
}

#[derive(Debug, PartialEq, Eq)]
/// Encoded baseline JPEG bytes and the backend that produced them.
///
/// The retained codestream can approach the shared host-allocation cap, so
/// this owner is intentionally move-only rather than exposing infallible
/// full-payload cloning.
pub struct EncodedJpeg {
    /// Complete JPEG codestream.
    pub data: Vec<u8>,
    /// Backend used to encode the codestream.
    pub backend: JpegBackend,
}

#[derive(Clone, Debug, Error)]
/// Errors produced by baseline JPEG encoding.
pub enum JpegEncodeError {
    #[error("JPEG encode requires nonzero dimensions")]
    /// Width or height was zero.
    EmptyDimensions,
    #[error("JPEG baseline dimensions must fit in u16, got {width}x{height}")]
    /// JPEG baseline SOF dimensions exceed the 16-bit marker fields.
    DimensionsTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },
    #[error("JPEG sample buffer length mismatch: expected {expected}, got {actual}")]
    /// Input sample buffer length does not match width, height, and format.
    SampleLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    #[error("JPEG host buffer requires {requested} bytes, exceeding the {cap}-byte cap")]
    /// A sample layout or encoded output exceeds the shared host allocation cap.
    MemoryCapExceeded {
        /// Required byte count, saturated when arithmetic overflows.
        requested: usize,
        /// Maximum accepted host allocation size.
        cap: usize,
    },
    #[error("JPEG host allocation failed for {bytes} bytes")]
    /// A required host buffer could not reserve its capacity.
    HostAllocationFailed {
        /// Requested allocation size in bytes.
        bytes: usize,
    },
    #[error("JPEG subsampling {subsampling:?} is incompatible with {samples}")]
    /// Requested subsampling is incompatible with the supplied sample format.
    IncompatibleSubsampling {
        /// Requested output sampling.
        subsampling: JpegSubsampling,
        /// Human-readable sample format name.
        samples: &'static str,
    },
    #[error("JPEG restart interval must be nonzero when provided")]
    /// Restart interval was explicitly set to zero.
    InvalidRestartInterval,
    #[error("JPEG encode backend {backend:?} is unavailable in j2k-jpeg CPU crate")]
    /// Requested backend is not available in this crate.
    UnsupportedBackend {
        /// Requested backend.
        backend: JpegBackend,
    },
    #[error("JPEG encoded marker segment is too large: {name}")]
    /// A marker segment would exceed the JPEG 16-bit length field.
    SegmentTooLarge {
        /// Marker segment name.
        name: &'static str,
    },
    #[error("JPEG entropy symbol has no Huffman code: {symbol}")]
    /// Encoder attempted to emit a symbol absent from the active Huffman table.
    MissingHuffmanCode {
        /// Missing entropy symbol.
        symbol: u8,
    },
    #[error("invalid JPEG DCT image: {reason}")]
    /// Caller-supplied coefficient-domain input cannot be re-emitted as baseline JPEG.
    InvalidDctImage {
        /// Typed invalid-input reason.
        #[source]
        reason: JpegDctImageError,
    },
    #[error("JPEG encode internal invariant failed: {reason}")]
    /// A heap-free diagnostic for an impossible encoder state.
    InternalInvariant {
        /// Static invariant description.
        reason: &'static str,
    },
}
