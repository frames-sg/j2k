// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public baseline JPEG adapter data, error, and host-interface types.

use alloc::vec::Vec;

use crate::encoder::{JpegBackend, JpegEncodeError, JpegSubsampling};
use crate::PixelFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Baseline JPEG component sampling parameters.
pub struct JpegBaselineSampling {
    /// Number of encoded components.
    pub components: u8,
    /// Horizontal sampling factor per component.
    pub h: [u8; 3],
    /// Vertical sampling factor per component.
    pub v: [u8; 3],
    /// Maximum horizontal sampling factor across components.
    pub max_h: u8,
    /// Maximum vertical sampling factor across components.
    pub max_v: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Canonical Huffman lookup table for encoding.
pub struct JpegBaselineHuffmanTable {
    /// Huffman code value by symbol.
    pub codes: [u16; 256],
    /// Huffman code length by symbol.
    pub lens: [u8; 256],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Tables needed to assemble and entropy-code a baseline JPEG frame.
pub struct JpegBaselineEncodeTables {
    /// Component sampling metadata.
    pub sampling: JpegBaselineSampling,
    /// Luma quantization table in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantization table in natural order.
    pub q_chroma: [u8; 64],
    /// Luma DC Huffman table.
    pub huff_dc_luma: JpegBaselineHuffmanTable,
    /// Luma AC Huffman table.
    pub huff_ac_luma: JpegBaselineHuffmanTable,
    /// Chroma DC Huffman table.
    pub huff_dc_chroma: JpegBaselineHuffmanTable,
    /// Chroma AC Huffman table.
    pub huff_ac_chroma: JpegBaselineHuffmanTable,
}

/// Backend-neutral metadata for a resident GPU baseline JPEG encode tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeTile {
    /// Byte offset of the first source pixel in the resident buffer.
    pub byte_offset: usize,
    /// Width of the valid input region in pixels.
    pub width: u32,
    /// Height of the valid input region in pixels.
    pub height: u32,
    /// Number of bytes between consecutive input rows.
    pub pitch_bytes: usize,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Pixel format of the source buffer.
    pub format: PixelFormat,
    /// Total resident buffer length in bytes.
    pub buffer_len: usize,
}

/// Backend-neutral baseline JPEG encode ABI parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeParams {
    /// First input byte for this tile inside a same-buffer batch.
    pub input_offset_bytes: u32,
    /// Width of the valid input rectangle in pixels.
    pub input_width: u32,
    /// Height of the valid input rectangle in pixels.
    pub input_height: u32,
    /// Encoded frame width in pixels.
    pub output_width: u32,
    /// Encoded frame height in pixels.
    pub output_height: u32,
    /// Number of input bytes between consecutive rows.
    pub pitch_bytes: u32,
    /// Number of MCUs per encoded frame row.
    pub mcus_per_row: u32,
    /// Number of MCU rows in the encoded frame.
    pub mcu_rows: u32,
    /// Optional restart interval in MCUs, or zero when disabled.
    pub restart_interval_mcus: u32,
    /// Stable resident-encode format ABI value.
    pub format: u32,
    /// Number of encoded components.
    pub components: u32,
    /// Maximum horizontal sampling factor.
    pub max_h: u32,
    /// Maximum vertical sampling factor.
    pub max_v: u32,
    /// Component 0 horizontal sampling factor.
    pub h0: u32,
    /// Component 0 vertical sampling factor.
    pub v0: u32,
    /// Component 1 horizontal sampling factor.
    pub h1: u32,
    /// Component 1 vertical sampling factor.
    pub v1: u32,
    /// Component 2 horizontal sampling factor.
    pub h2: u32,
    /// Component 2 vertical sampling factor.
    pub v2: u32,
    /// First entropy-output byte for this tile inside a batch output allocation.
    pub entropy_offset_bytes: u32,
    /// Entropy-output capacity for this tile.
    pub entropy_capacity: u32,
}

/// Backend-neutral resident GPU baseline JPEG encode plan for one tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeTilePlan {
    /// GPU ABI parameters for this tile.
    pub params: JpegBaselineGpuEncodeParams,
    /// Entropy-output capacity for this tile.
    pub entropy_capacity: usize,
}

/// Backend-neutral resident GPU baseline JPEG encode plan for one batch span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegBaselineGpuEncodeBatchPlan {
    /// GPU ABI parameters in input tile order.
    pub params: Vec<JpegBaselineGpuEncodeParams>,
    /// Combined entropy-output capacity for the batch allocation.
    pub total_entropy_capacity: usize,
}

/// Backend hooks used by the shared resident GPU baseline JPEG encode driver.
///
/// First-party CUDA and Metal adapters provide only resident-buffer identity,
/// tile metadata conversion, backend error mapping, and kernel submission. The
/// shared driver owns table construction, planning, batch span grouping, and
/// JPEG frame assembly.
pub trait JpegBaselineGpuEncodeHostAdapter<T: Copy> {
    /// Error returned by the backend adapter.
    type Error: From<JpegEncodeError>;
    /// Stable identity for a resident source allocation.
    type SourceKey: PartialEq;

    /// Backend represented by this adapter.
    fn backend(&self) -> JpegBackend;

    /// Return the resident source allocation key for grouping batch spans.
    fn source_key(&self, tile: &T) -> Self::SourceKey;

    /// Convert a backend tile into backend-neutral planning metadata.
    fn gpu_tile(&self, tile: T) -> Result<JpegBaselineGpuEncodeTile, Self::Error>;

    /// Map a backend-neutral planning error into the backend's public error.
    fn map_plan_error(&self, error: JpegBaselineGpuEncodeError) -> Self::Error;

    /// Submit one resident tile to the backend entropy encoder.
    fn encode_tile_entropy(
        &mut self,
        tile: T,
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeTilePlan,
    ) -> Result<Vec<u8>, Self::Error>;

    /// Submit a contiguous same-source-buffer resident tile span.
    fn encode_batch_entropy(
        &mut self,
        tiles: &[T],
        tables: &JpegBaselineEncodeTables,
        plan: JpegBaselineGpuEncodeBatchPlan,
    ) -> Result<Vec<Vec<u8>>, Self::Error>;
}

/// Error returned by backend-neutral resident GPU baseline JPEG encode planning.
#[derive(Debug)]
pub enum JpegBaselineGpuEncodeError {
    /// A baseline JPEG encode option was invalid.
    Encode(JpegEncodeError),
    /// The requested public backend does not match this adapter.
    UnsupportedBackend {
        /// Requested backend.
        requested: JpegBackend,
        /// Backend accepted by the caller.
        expected: JpegBackend,
    },
    /// The valid input rectangle exceeds the encoded output dimensions.
    InputExceedsOutputDimensions,
    /// The source pixel format is unsupported by resident baseline encode.
    UnsupportedPixelFormat {
        /// Source pixel format.
        format: PixelFormat,
    },
    /// The source pixel format is incompatible with the requested subsampling.
    IncompatibleSubsampling {
        /// Requested subsampling.
        subsampling: JpegSubsampling,
        /// Source sample description.
        samples: &'static str,
    },
    /// Row-byte arithmetic overflowed.
    RowByteCountOverflow,
    /// Source pitch is shorter than one row.
    PitchTooShort {
        /// Required row bytes.
        row_bytes: usize,
        /// Provided pitch bytes.
        pitch_bytes: usize,
    },
    /// Input byte-range arithmetic overflowed.
    InputRangeOverflow,
    /// Input byte range exceeds the resident buffer length.
    InputRangeExceedsBuffer {
        /// Required exclusive byte end.
        required_end: usize,
        /// Resident buffer length in bytes.
        buffer_len: usize,
    },
    /// Pitch does not fit the GPU ABI.
    PitchTooLarge,
    /// Input offset does not fit the GPU ABI.
    InputOffsetTooLarge,
    /// Entropy offset does not fit the GPU ABI.
    EntropyOffsetTooLarge,
    /// Entropy capacity does not fit the GPU ABI.
    EntropyCapacityTooLarge,
    /// Combined batch entropy capacity overflowed host arithmetic.
    BatchEntropyCapacityOverflow,
}

impl From<JpegEncodeError> for JpegBaselineGpuEncodeError {
    fn from(error: JpegEncodeError) -> Self {
        Self::Encode(error)
    }
}
