// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{error::CudaError, execution::CudaExecutionStats, memory::CudaDeviceBuffer};

macro_rules! define_cuda_jpeg_rgb8_decode_plan {
    (
        $(#[$meta:meta])*
        pub struct $name:ident <'a> {
            $($prefix:tt)*
        }
    ) => {
        $(#[$meta])*
        pub struct $name<'a> {
            $($prefix)*
            /// Image dimensions as `(width, height)`.
            pub dimensions: (u32, u32),
            /// Number of MCUs per row.
            pub mcus_per_row: u32,
            /// Number of MCU rows.
            pub mcu_rows: u32,
            /// Entropy-coded scan payload with byte stuffing/restart markers removed.
            pub entropy_bytes: &'a [u8],
            /// Entropy resume checkpoints.
            pub entropy_checkpoints: &'a [CudaJpegEntropyCheckpoint],
            /// Luma quantization table in JPEG zigzag order.
            pub y_quant: [u16; 64],
            /// Cb quantization table in JPEG zigzag order.
            pub cb_quant: [u16; 64],
            /// Cr quantization table in JPEG zigzag order.
            pub cr_quant: [u16; 64],
            /// Y DC Huffman table.
            pub y_dc_table: CudaJpegHuffmanTable,
            /// Y AC Huffman table.
            pub y_ac_table: CudaJpegHuffmanTable,
            /// Cb DC Huffman table.
            pub cb_dc_table: CudaJpegHuffmanTable,
            /// Cb AC Huffman table.
            pub cb_ac_table: CudaJpegHuffmanTable,
            /// Cr DC Huffman table.
            pub cr_dc_table: CudaJpegHuffmanTable,
            /// Cr AC Huffman table.
            pub cr_ac_table: CudaJpegHuffmanTable,
        }
    };
}

/// Prepared baseline JPEG Huffman table for CUDA JPEG decode kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegHuffmanTable {
    /// Largest Huffman code for each bit length; negative means no codes of that length.
    pub max_code: [i32; 17],
    /// Value-index offset for each bit length.
    pub val_offset: [i32; 17],
    /// Huffman values in canonical order.
    pub values: [u8; 256],
    /// Number of valid entries in `values`.
    pub values_len: u32,
}

impl CudaJpegHuffmanTable {
    /// Prepare a CUDA Huffman table from JPEG BITS and HUFFVAL payloads.
    #[doc(hidden)]
    pub fn from_jpeg_bits_values(
        bits: [u8; 16],
        values_len: u16,
        values: [u8; 256],
    ) -> Result<Self, CudaError> {
        let values_len_usize = usize::from(values_len);
        let canonical = j2k_codec_math::jpeg::derive_canonical_huffman(&bits, values_len_usize)
            .map_err(|error| CudaError::InvalidArgument {
                message: format!("JPEG Huffman {error}"),
            })?;

        Ok(Self {
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            values,
            values_len: u32::from(values_len),
        })
    }
}

/// Entropy resume point for CUDA baseline JPEG decode.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegEntropyCheckpoint {
    /// MCU index for this checkpoint.
    pub mcu_index: u32,
    /// Byte offset into the entropy payload.
    pub entropy_pos: u32,
    /// Left-aligned buffered entropy bits.
    pub bit_acc: u64,
    /// Number of valid buffered bits.
    pub bit_count: u32,
    /// Previous Y DC predictor.
    pub y_prev_dc: i32,
    /// Previous Cb DC predictor.
    pub cb_prev_dc: i32,
    /// Previous Cr DC predictor.
    pub cr_prev_dc: i32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: u32,
}

/// J2K-owned CUDA baseline JPEG RGB8 kernel shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum CudaJpegRgb8Sampling {
    /// Fast 4:2:0 YCbCr shape: four Y blocks, then Cb and Cr per MCU.
    Fast420,
    /// Fast 4:2:2 YCbCr shape: two Y blocks, then Cb and Cr per MCU.
    Fast422,
    /// Fast 4:4:4 YCbCr shape: one Y block, then Cb and Cr per MCU.
    Fast444,
}

#[doc(hidden)]
/// Experimental JPEG entropy chunking parameters for CUDA self-sync diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJpegChunkedEntropyConfig {
    /// Subsequence size in 32-bit words.
    pub subsequence_words: u32,
    /// Reserved synchronization-sequence length for future grouped scans.
    ///
    /// The current diagnostic records adjacent-subsequence overflow results for
    /// every neighboring pair; this value is validated and passed through the
    /// ABI for compatibility with grouped synchronization experiments.
    pub sequence_len: u32,
    /// Maximum adjacent subsequences an overflow decoder may scan.
    pub max_overflow_subsequences: u32,
}

impl Default for CudaJpegChunkedEntropyConfig {
    fn default() -> Self {
        Self {
            subsequence_words: 1024,
            sequence_len: 128,
            max_overflow_subsequences: 4,
        }
    }
}

impl CudaJpegChunkedEntropyConfig {
    /// Return one subsequence size in bits.
    pub fn subsequence_bits(self) -> u32 {
        self.subsequence_words.saturating_mul(32)
    }

    /// Validate parameters before launching diagnostic kernels.
    pub fn validate(self) -> Result<(), CudaError> {
        if self.subsequence_words == 0 {
            return Err(CudaError::InvalidArgument {
                message: "JPEG entropy subsequence_words must be nonzero".to_string(),
            });
        }
        if self.subsequence_words.checked_mul(32).is_none() {
            return Err(CudaError::InvalidArgument {
                message: "JPEG entropy subsequence_words bit size exceeds u32".to_string(),
            });
        }
        if self.sequence_len == 0 {
            return Err(CudaError::InvalidArgument {
                message: "JPEG entropy sequence_len must be nonzero".to_string(),
            });
        }
        Ok(())
    }

    /// Count fixed-size bit subsequences needed for an entropy payload.
    pub fn subsequence_count_for_entropy_bytes(
        self,
        entropy_len: usize,
    ) -> Result<usize, CudaError> {
        self.validate()?;
        let entropy_bits = entropy_len
            .checked_mul(8)
            .ok_or(CudaError::LengthTooLarge { len: entropy_len })?;
        let bits = self.subsequence_bits() as usize;
        Ok(entropy_bits.div_ceil(bits))
    }
}

#[doc(hidden)]
/// Device-written state for one entropy subsequence self-sync diagnostic.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJpegEntropySyncState {
    /// Zero means success; nonzero maps to diagnostic kernel status.
    pub code: u32,
    /// Subsequence start bit offset.
    pub start_bit: u32,
    /// Subsequence exclusive end bit offset.
    pub end_bit: u32,
    /// Decoder bit position after scanning this subsequence.
    pub bit_pos: u32,
    /// Decoded coefficient-slot count.
    pub symbol_count: u32,
    /// 4:2:0 block phase: 0..=3 for Y blocks, 4 Cb, 5 Cr.
    pub block_phase: u32,
    /// Zig-zag coefficient index inside the current block.
    pub zigzag_index: u32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: u32,
}

#[doc(hidden)]
/// Device-written overflow result for adjacent subsequence synchronization.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaJpegEntropyOverflowState {
    /// Zero means success; nonzero maps to diagnostic kernel status.
    pub code: u32,
    /// Source subsequence index.
    pub from_subsequence: u32,
    /// Target subsequence index.
    pub to_subsequence: u32,
    /// Bits scanned after the target subsequence start before synchronization.
    pub overflow_bits: u32,
    /// One when synchronization was detected.
    pub synchronized: u32,
    /// Reserved for ABI-compatible expansion.
    pub reserved: [u32; 3],
}

#[doc(hidden)]
/// Host-side report returned by experimental JPEG entropy self-sync diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaJpegChunkedEntropyReport {
    /// Diagnostic chunk configuration.
    pub config: CudaJpegChunkedEntropyConfig,
    /// Entropy payload length in bytes.
    pub entropy_bytes: usize,
    /// Per-subsequence first-pass states.
    pub states: Vec<CudaJpegEntropySyncState>,
    /// Per-adjacent-subsequence overflow states.
    pub overflows: Vec<CudaJpegEntropyOverflowState>,
    /// Runtime dispatch stats for diagnostic kernels.
    pub execution: CudaExecutionStats,
}

impl CudaJpegChunkedEntropyReport {
    /// Number of subsequences examined.
    pub fn subsequence_count(&self) -> usize {
        self.states.len()
    }

    /// Number of overflow records that synchronized.
    pub fn synchronized_overflow_count(&self) -> usize {
        self.overflows
            .iter()
            .filter(|overflow| overflow.synchronized != 0)
            .count()
    }

    /// Maximum overflow scan length in bits.
    pub fn max_overflow_bits(&self) -> Option<u32> {
        self.overflows
            .iter()
            .map(|overflow| overflow.overflow_bits)
            .max()
    }

    /// Number of first-pass states with nonzero status.
    pub fn failed_state_count(&self) -> usize {
        self.states.iter().filter(|state| state.code != 0).count()
    }
}

/// CUDA baseline JPEG encode input sample format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub enum CudaJpegBaselineEncodeFormat {
    /// One byte per pixel grayscale input.
    Gray8,
    /// Three bytes per pixel RGB input.
    Rgb8,
}

impl CudaJpegBaselineEncodeFormat {
    /// Return the stable CUDA ABI value for this format.
    #[doc(hidden)]
    pub fn abi(self) -> u32 {
        match self {
            Self::Gray8 => JPEG_BASELINE_ENCODE_FORMAT_GRAY8,
            Self::Rgb8 => JPEG_BASELINE_ENCODE_FORMAT_RGB8,
        }
    }
}

const JPEG_BASELINE_ENCODE_FORMAT_GRAY8: u32 = 0;
const JPEG_BASELINE_ENCODE_FORMAT_RGB8: u32 = 1;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(super) const JPEG_BASELINE_ENCODE_STATUS_OK: u32 = 0;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(super) const JPEG_BASELINE_ENCODE_STATUS_OVERFLOW: u32 = 1;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(super) const JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN: u32 = 2;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(super) const JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS: u32 = 3;

/// CUDA baseline JPEG entropy encode parameters for one resident tile.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegBaselineEncodeParams {
    /// First byte of this input tile relative to the bound input pointer.
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
    /// Stable ABI value from [`CudaJpegBaselineEncodeFormat::abi`].
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

// SAFETY: `CudaJpegBaselineEncodeParams` is `#[repr(C)]` and contains only CUDA
// scalar ABI fields passed by value through a kernel-parameter pointer.
unsafe impl crate::execution::CudaKernelParam for CudaJpegBaselineEncodeParams {}

/// CUDA baseline JPEG canonical Huffman table for encode kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegBaselineEncodeHuffmanTable {
    /// Huffman code value by symbol.
    pub codes: [u16; 256],
    /// Huffman code length by symbol.
    pub lens: [u8; 256],
}

impl Default for CudaJpegBaselineEncodeHuffmanTable {
    fn default() -> Self {
        Self {
            codes: [0; 256],
            lens: [0; 256],
        }
    }
}

/// CUDA baseline JPEG entropy encode status for one tile.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CudaJpegBaselineEncodeStatus {
    pub(crate) code: u32,
    pub(crate) entropy_len: u32,
    pub(crate) detail: u32,
    pub(crate) reserved: u32,
}

/// CUDA baseline JPEG entropy encode plan for one resident input tile.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJpegBaselineEntropyEncodeJob<'a> {
    /// Resident CUDA input pixels.
    pub input: &'a CudaDeviceBuffer,
    /// Byte offset applied while binding the input buffer.
    pub input_offset: usize,
    /// Encoded tile parameters.
    pub params: CudaJpegBaselineEncodeParams,
    /// Luma quantization table in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantization table in natural order.
    pub q_chroma: [u8; 64],
    /// Luma DC Huffman table.
    pub huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable,
    /// Luma AC Huffman table.
    pub huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable,
    /// Chroma DC Huffman table.
    pub huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable,
    /// Chroma AC Huffman table.
    pub huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable,
    /// Entropy output capacity in bytes.
    pub entropy_capacity: usize,
}

/// CUDA baseline JPEG entropy encode plan for same-buffer resident input tiles.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJpegBaselineEntropyEncodeBatchJob<'a> {
    /// Resident CUDA input pixels shared by every tile.
    pub input: &'a CudaDeviceBuffer,
    /// Encoded tile parameters. Each entry contains its own input and entropy offset.
    pub params: Vec<CudaJpegBaselineEncodeParams>,
    /// Luma quantization table in natural order.
    pub q_luma: [u8; 64],
    /// Chroma quantization table in natural order.
    pub q_chroma: [u8; 64],
    /// Luma DC Huffman table.
    pub huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable,
    /// Luma AC Huffman table.
    pub huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable,
    /// Chroma DC Huffman table.
    pub huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable,
    /// Chroma AC Huffman table.
    pub huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable,
    /// Combined entropy output capacity in bytes.
    pub entropy_capacity: usize,
}

#[doc(hidden)]
/// Experimental J2K-owned CUDA JPEG entropy self-sync diagnostic plan.
#[derive(Debug)]
pub struct CudaJpegChunkedEntropyPlan<'a> {
    /// Chunking configuration.
    pub config: CudaJpegChunkedEntropyConfig,
    /// Entropy-coded scan payload with byte stuffing/restart markers removed.
    pub entropy_bytes: &'a [u8],
    /// Y DC Huffman table.
    pub y_dc_table: CudaJpegHuffmanTable,
    /// Y AC Huffman table.
    pub y_ac_table: CudaJpegHuffmanTable,
    /// Cb DC Huffman table.
    pub cb_dc_table: CudaJpegHuffmanTable,
    /// Cb AC Huffman table.
    pub cb_ac_table: CudaJpegHuffmanTable,
    /// Cr DC Huffman table.
    pub cr_dc_table: CudaJpegHuffmanTable,
    /// Cr AC Huffman table.
    pub cr_ac_table: CudaJpegHuffmanTable,
}

define_cuda_jpeg_rgb8_decode_plan! {
    /// J2K-owned CUDA baseline JPEG RGB8 decode plan.
    #[derive(Debug)]
    #[doc(hidden)]
    pub struct CudaJpegRgb8DecodePlan<'a> {
        /// MCU sampling/kernel shape.
        pub sampling: CudaJpegRgb8Sampling,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
pub(crate) struct CudaJpeg420Params {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mcus_per_row: u32,
    pub(crate) mcu_rows: u32,
    pub(crate) entropy_len: u32,
    pub(crate) checkpoint_count: u32,
    pub(crate) out_stride: u32,
    pub(crate) reserved: u32,
}

// SAFETY: `CudaJpeg420Params` is `#[repr(C)]` and contains only CUDA scalar
// ABI fields passed by value through a kernel-parameter pointer.
unsafe impl crate::execution::CudaKernelParam for CudaJpeg420Params {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
pub(crate) struct CudaJpegEntropyChunkParams {
    pub(crate) entropy_len: u32,
    pub(crate) entropy_bits: u32,
    pub(crate) subsequence_bits: u32,
    pub(crate) subsequence_count: u32,
    pub(crate) sequence_len: u32,
    pub(crate) max_overflow_subsequences: u32,
    pub(crate) reserved0: u32,
    pub(crate) reserved1: u32,
}

// SAFETY: `CudaJpegEntropyChunkParams` is `#[repr(C)]` and contains only CUDA
// scalar ABI fields passed by value through a kernel-parameter pointer.
unsafe impl crate::execution::CudaKernelParam for CudaJpegEntropyChunkParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
pub(crate) struct CudaJpegDecodeStatus {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) position: u32,
    pub(crate) reserved: u32,
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaJpegRgb8ValidatedPlan {
    pub(crate) params: CudaJpeg420Params,
    pub(crate) output_len: usize,
}
