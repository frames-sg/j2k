#![allow(clippy::similar_names)]

#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::bytes::{
    cuda_jpeg_baseline_encode_huffman_table_as_bytes, cuda_jpeg_baseline_encode_params_as_bytes,
    cuda_jpeg_baseline_encode_statuses_as_bytes, cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::bytes::{
    cuda_jpeg_decode_statuses_as_bytes, cuda_jpeg_decode_statuses_as_bytes_mut,
    cuda_jpeg_entropy_checkpoints_as_bytes, cuda_jpeg_entropy_overflow_states_as_bytes,
    cuda_jpeg_entropy_overflow_states_as_bytes_mut, cuda_jpeg_entropy_sync_states_as_bytes,
    cuda_jpeg_entropy_sync_states_as_bytes_mut, cuda_jpeg_huffman_table_as_bytes,
    u16_slice_as_bytes,
};
use crate::{
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelOutput},
    memory::CudaDeviceBuffer,
};
#[cfg(any(feature = "cuda-oxide-jpeg-decode", feature = "cuda-oxide-jpeg-encode"))]
use crate::{
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
};

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
    pub fn from_jpeg_bits_values(
        bits: [u8; 16],
        values_len: u16,
        values: [u8; 256],
    ) -> Result<Self, CudaError> {
        let values_len_usize = usize::from(values_len);
        let mut huffsize = [0u8; 256];
        let mut huffsize_len = 0usize;
        for (len_minus_1, &count) in bits.iter().enumerate() {
            let len = u8::try_from(len_minus_1 + 1).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman code length exceeds u8".to_string(),
            })?;
            for _ in 0..count {
                if huffsize_len >= values_len_usize || huffsize_len >= huffsize.len() {
                    return Err(CudaError::InvalidArgument {
                        message: "JPEG Huffman BITS exceed values length".to_string(),
                    });
                }
                huffsize[huffsize_len] = len;
                huffsize_len += 1;
            }
        }
        if huffsize_len != values_len_usize {
            return Err(CudaError::InvalidArgument {
                message: "JPEG Huffman BITS do not match values length".to_string(),
            });
        }

        let mut huffcode = [0u16; 256];
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for (idx, &size) in huffsize[..huffsize_len].iter().enumerate() {
            while size != si {
                code <<= 1;
                si = si.saturating_add(1);
            }
            if si > 16 || code >= (1u32 << si) {
                return Err(CudaError::InvalidArgument {
                    message: "JPEG Huffman code overflow".to_string(),
                });
            }
            huffcode[idx] = u16::try_from(code).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman code exceeds u16".to_string(),
            })?;
            code = code
                .checked_add(1)
                .ok_or_else(|| CudaError::InvalidArgument {
                    message: "JPEG Huffman code overflow".to_string(),
                })?;
        }

        let mut max_code = [-1i32; 17];
        let mut val_offset = [0i32; 17];
        let mut cursor = 0usize;
        for (len_minus_1, &count) in bits.iter().enumerate() {
            let len = len_minus_1 + 1;
            let count = usize::from(count);
            if count == 0 {
                continue;
            }
            let min_code = i32::from(huffcode[cursor]);
            max_code[len] = i32::from(huffcode[cursor + count - 1]);
            val_offset[len] = i32::try_from(cursor).map_err(|_| CudaError::InvalidArgument {
                message: "JPEG Huffman values length exceeds i32".to_string(),
            })? - min_code;
            cursor += count;
        }

        Ok(Self {
            max_code,
            val_offset,
            values,
            values_len: u32::from(values_len),
        })
    }
}

/// Entropy resume point for CUDA baseline JPEG decode.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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
pub enum CudaJpegRgb8Sampling {
    /// Fast 4:2:0 YCbCr shape: four Y blocks, then Cb and Cr per MCU.
    Fast420,
    /// Fast 4:2:2 YCbCr shape: two Y blocks, then Cb and Cr per MCU.
    Fast422,
    /// Fast 4:4:4 YCbCr shape: one Y block, then Cb and Cr per MCU.
    Fast444,
}

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

#[cfg_attr(not(feature = "cuda-oxide-jpeg-decode"), allow(dead_code))]
pub(crate) fn jpeg_entropy_overflow_count(subsequence_count: usize) -> usize {
    subsequence_count.saturating_sub(1)
}

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
pub enum CudaJpegBaselineEncodeFormat {
    /// One byte per pixel grayscale input.
    Gray8,
    /// Three bytes per pixel RGB input.
    Rgb8,
}

impl CudaJpegBaselineEncodeFormat {
    /// Return the stable CUDA ABI value for this format.
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
const JPEG_BASELINE_ENCODE_STATUS_OK: u32 = 0;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
const JPEG_BASELINE_ENCODE_STATUS_OVERFLOW: u32 = 1;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
const JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN: u32 = 2;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
const JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS: u32 = 3;

/// CUDA baseline JPEG entropy encode parameters for one resident tile.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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
    pub struct CudaJpegRgb8DecodePlan<'a> {
        /// MCU sampling/kernel shape.
        pub sampling: CudaJpegRgb8Sampling,
    }
}

define_cuda_jpeg_rgb8_decode_plan! {
    /// J2K-owned CUDA baseline JPEG 4:2:0 decode plan.
    #[derive(Debug)]
    pub struct CudaJpeg420Rgb8DecodePlan<'a> {
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

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn validate_jpeg_rgb8_plan(
    plan: &CudaJpegRgb8DecodePlan<'_>,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, _) = plan.dimensions;
    let out_stride = width.checked_mul(3).ok_or(CudaError::ImageTooLarge {
        width,
        height: plan.dimensions.1,
        channels: 3,
    })?;
    validate_jpeg_rgb8_plan_with_pitch(plan, out_stride as usize)
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn validate_jpeg_rgb8_plan_with_pitch(
    plan: &CudaJpegRgb8DecodePlan<'_>,
    pitch_bytes: usize,
) -> Result<CudaJpegRgb8ValidatedPlan, CudaError> {
    let (width, height) = plan.dimensions;
    if width == 0 || height == 0 {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA decode dimensions must be nonzero".to_string(),
        });
    }
    if plan.entropy_checkpoints.is_empty() {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA decode requires at least one entropy checkpoint".to_string(),
        });
    }
    let entropy_len =
        u32::try_from(plan.entropy_bytes.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let checkpoint_count =
        u32::try_from(plan.entropy_checkpoints.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_checkpoints.len(),
        })?;
    let row_bytes = width.checked_mul(3).ok_or(CudaError::ImageTooLarge {
        width,
        height,
        channels: 3,
    })?;
    if pitch_bytes < row_bytes as usize {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "JPEG CUDA decode pitch {pitch_bytes} is smaller than row byte count {row_bytes}"
            ),
        });
    }
    let out_stride =
        u32::try_from(pitch_bytes).map_err(|_| CudaError::LengthTooLarge { len: pitch_bytes })?;
    let output_len = pitch_bytes
        .checked_mul(height as usize - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes as usize))
        .ok_or(CudaError::ImageTooLarge {
            width,
            height,
            channels: 3,
        })?;

    Ok(CudaJpegRgb8ValidatedPlan {
        params: CudaJpeg420Params {
            width,
            height,
            mcus_per_row: plan.mcus_per_row,
            mcu_rows: plan.mcu_rows,
            entropy_len,
            checkpoint_count,
            out_stride,
            reserved: 0,
        },
        output_len,
    })
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn validate_jpeg_entropy_chunk_plan(
    plan: &CudaJpegChunkedEntropyPlan<'_>,
    subsequences: usize,
) -> Result<CudaJpegEntropyChunkParams, CudaError> {
    let entropy_len =
        u32::try_from(plan.entropy_bytes.len()).map_err(|_| CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let entropy_bits = entropy_len
        .checked_mul(8)
        .ok_or(CudaError::LengthTooLarge {
            len: plan.entropy_bytes.len(),
        })?;
    let subsequence_count =
        u32::try_from(subsequences).map_err(|_| CudaError::LengthTooLarge { len: subsequences })?;

    Ok(CudaJpegEntropyChunkParams {
        entropy_len,
        entropy_bits,
        subsequence_bits: plan.config.subsequence_bits(),
        subsequence_count,
        sequence_len: plan.config.sequence_len,
        max_overflow_subsequences: plan.config.max_overflow_subsequences,
        reserved0: 0,
        reserved1: 0,
    })
}

pub(crate) fn cuda_jpeg_rgb8_plan_from_420<'a>(
    plan: &CudaJpeg420Rgb8DecodePlan<'a>,
) -> CudaJpegRgb8DecodePlan<'a> {
    CudaJpegRgb8DecodePlan {
        sampling: CudaJpegRgb8Sampling::Fast420,
        dimensions: plan.dimensions,
        mcus_per_row: plan.mcus_per_row,
        mcu_rows: plan.mcu_rows,
        entropy_bytes: plan.entropy_bytes,
        entropy_checkpoints: plan.entropy_checkpoints,
        y_quant: plan.y_quant,
        cb_quant: plan.cb_quant,
        cr_quant: plan.cr_quant,
        y_dc_table: plan.y_dc_table,
        y_ac_table: plan.y_ac_table,
        cb_dc_table: plan.cb_dc_table,
        cb_ac_table: plan.cb_ac_table,
        cr_dc_table: plan.cr_dc_table,
        cr_ac_table: plan.cr_ac_table,
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn jpeg_rgb8_kernel(sampling: CudaJpegRgb8Sampling) -> (CudaKernel, &'static str) {
    match sampling {
        CudaJpegRgb8Sampling::Fast420 => (
            CudaKernel::JpegDecodeFast420Rgb8,
            "j2k_jpeg_decode_fast420_rgb8",
        ),
        CudaJpegRgb8Sampling::Fast422 => (
            CudaKernel::JpegDecodeFast422Rgb8,
            "j2k_jpeg_decode_fast422_rgb8",
        ),
        CudaJpegRgb8Sampling::Fast444 => (
            CudaKernel::JpegDecodeFast444Rgb8,
            "j2k_jpeg_decode_fast444_rgb8",
        ),
    }
}

impl CudaContext {
    /// Encode one CUDA-resident tile into baseline JPEG entropy bytes.
    pub fn encode_jpeg_baseline_entropy(
        &self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
    ) -> Result<Vec<u8>, CudaError> {
        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = job;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG baseline encode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        {
            self.inner.set_current()?;
            let entropy = self.allocate(job.entropy_capacity)?;
            let mut status = [CudaJpegBaselineEncodeStatus::default()];
            let status_buffer =
                self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&status))?;
            let q_luma = self.upload(&job.q_luma)?;
            let q_chroma = self.upload(&job.q_chroma)?;
            let huff_dc_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_luma,
            ))?;
            let huff_ac_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_luma,
            ))?;
            let huff_dc_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_chroma,
            ))?;
            let huff_ac_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_chroma,
            ))?;
            self.launch_jpeg_encode_baseline_entropy(
                job.input,
                job.input_offset,
                &entropy,
                &status_buffer,
                job.params,
                &q_luma,
                &q_chroma,
                &huff_dc_luma,
                &huff_ac_luma,
                &huff_dc_chroma,
                &huff_ac_chroma,
            )?;
            status_buffer
                .copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(&mut status))?;
            validate_jpeg_encode_status(status[0], "j2k_jpeg_encode_baseline_entropy")?;
            let entropy_len =
                usize::try_from(status[0].entropy_len).map_err(|_| CudaError::LengthTooLarge {
                    len: status[0].entropy_len as usize,
                })?;
            if entropy_len > job.entropy_capacity {
                return Err(CudaError::OutputTooSmall {
                    required: entropy_len,
                    have: job.entropy_capacity,
                });
            }
            let mut out = vec![0u8; entropy_len];
            entropy.copy_range_to_host(0, &mut out)?;
            Ok(out)
        }
    }

    /// Encode same-buffer CUDA-resident tiles into baseline JPEG entropy chunks.
    #[allow(clippy::too_many_lines)]
    pub fn encode_jpeg_baseline_entropy_batch(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        if job.params.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
        {
            let _ = job;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG baseline encode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-encode")]
        {
            self.inner.set_current()?;
            let tile_count =
                u32::try_from(job.params.len()).map_err(|_| CudaError::LengthTooLarge {
                    len: job.params.len(),
                })?;
            let entropy = self.allocate(job.entropy_capacity)?;
            let mut statuses = vec![CudaJpegBaselineEncodeStatus::default(); job.params.len()];
            let status_buffer =
                self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))?;
            let params_buffer =
                self.upload(cuda_jpeg_baseline_encode_params_as_bytes(&job.params))?;
            let q_luma = self.upload(&job.q_luma)?;
            let q_chroma = self.upload(&job.q_chroma)?;
            let huff_dc_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_luma,
            ))?;
            let huff_ac_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_luma,
            ))?;
            let huff_dc_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_dc_chroma,
            ))?;
            let huff_ac_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
                &job.huff_ac_chroma,
            ))?;
            self.launch_jpeg_encode_baseline_entropy_batch(
                job.input,
                &entropy,
                &status_buffer,
                &params_buffer,
                &q_luma,
                &q_chroma,
                &huff_dc_luma,
                &huff_ac_luma,
                &huff_dc_chroma,
                &huff_ac_chroma,
                tile_count,
            )?;
            status_buffer.copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
                &mut statuses,
            ))?;
            let mut out = Vec::with_capacity(job.params.len());
            for (index, (status, params)) in statuses.iter().copied().zip(&job.params).enumerate() {
                validate_jpeg_encode_status(status, "j2k_jpeg_encode_baseline_entropy_batch")?;
                let entropy_len =
                    usize::try_from(status.entropy_len).map_err(|_| CudaError::LengthTooLarge {
                        len: status.entropy_len as usize,
                    })?;
                let offset = usize::try_from(params.entropy_offset_bytes).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: params.entropy_offset_bytes as usize,
                    }
                })?;
                let capacity = usize::try_from(params.entropy_capacity).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: params.entropy_capacity as usize,
                    }
                })?;
                if entropy_len > capacity {
                    return Err(CudaError::OutputTooSmall {
                        required: entropy_len,
                        have: capacity,
                    });
                }
                let end = offset
                    .checked_add(entropy_len)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
                if end > job.entropy_capacity {
                    return Err(CudaError::OutputTooSmall {
                        required: end,
                        have: job.entropy_capacity,
                    });
                }
                let mut chunk = vec![0u8; entropy_len];
                entropy
                    .copy_range_to_host(offset, &mut chunk)
                    .map_err(|error| {
                        if matches!(error, CudaError::OutputTooSmall { .. }) {
                            CudaError::InvalidArgument {
                                message: format!(
                                "JPEG CUDA encode batch tile {index} entropy range is out of bounds"
                            ),
                            }
                        } else {
                            error
                        }
                    })?;
                out.push(chunk);
            }
            Ok(out)
        }
    }

    /// Run experimental 4:2:0 JPEG entropy self-sync diagnostics.
    pub fn diagnose_jpeg_420_entropy_self_sync(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        plan.config.validate()?;
        let subsequences = plan
            .config
            .subsequence_count_for_entropy_bytes(plan.entropy_bytes.len())?;
        if subsequences == 0 {
            return Ok(CudaJpegChunkedEntropyReport {
                config: plan.config,
                entropy_bytes: plan.entropy_bytes.len(),
                states: Vec::new(),
                overflows: Vec::new(),
                execution: CudaExecutionStats {
                    kernel_dispatches: 0,
                    copy_kernel_dispatches: 0,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = subsequences;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG entropy diagnostic PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            self.diagnose_jpeg_420_entropy_self_sync_nonempty(plan, subsequences)
        }
    }

    /// Decode one baseline JPEG 4:2:0 image to device-resident RGB8 using J2K CUDA kernels.
    pub fn decode_jpeg_420_rgb8_owned(
        &self,
        plan: &CudaJpeg420Rgb8DecodePlan<'_>,
    ) -> Result<CudaKernelOutput, CudaError> {
        let plan = cuda_jpeg_rgb8_plan_from_420(plan);
        self.decode_jpeg_rgb8_owned(&plan)
    }

    /// Decode one baseline JPEG RGB8 image to device-resident RGB8 using J2K CUDA kernels.
    pub fn decode_jpeg_rgb8_owned(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
    ) -> Result<CudaKernelOutput, CudaError> {
        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = plan;
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG RGB8 decode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            let validated = validate_jpeg_rgb8_plan(plan)?;
            self.inner.set_current()?;
            let output = self.allocate(validated.output_len)?;
            let execution = self.decode_jpeg_rgb8_owned_validated(plan, &output, validated)?;
            Ok(CudaKernelOutput {
                buffer: output,
                execution,
            })
        }
    }

    /// Decode one baseline JPEG 4:2:0 image into caller-owned CUDA RGB8 memory.
    pub fn decode_jpeg_420_rgb8_owned_into(
        &self,
        plan: &CudaJpeg420Rgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        let plan = cuda_jpeg_rgb8_plan_from_420(plan);
        self.decode_jpeg_rgb8_owned_into(&plan, output, pitch_bytes)
    }

    /// Decode one baseline JPEG RGB8 image into caller-owned CUDA RGB8 memory.
    pub fn decode_jpeg_rgb8_owned_into(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = (plan, output, pitch_bytes);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG RGB8 decode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            let validated = validate_jpeg_rgb8_plan_with_pitch(plan, pitch_bytes)?;
            if output.byte_len() < validated.output_len {
                return Err(CudaError::OutputTooSmall {
                    required: validated.output_len,
                    have: output.byte_len(),
                });
            }
            self.inner.set_current()?;
            self.decode_jpeg_rgb8_owned_validated(plan, output, validated)
        }
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[allow(clippy::similar_names)]
    fn decode_jpeg_rgb8_owned_validated(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        validated: CudaJpegRgb8ValidatedPlan,
    ) -> Result<CudaExecutionStats, CudaError> {
        let (kernel, kernel_name) = jpeg_rgb8_kernel(plan.sampling);
        let entropy = self.upload(plan.entropy_bytes)?;
        let y_quant = self.upload(u16_slice_as_bytes(&plan.y_quant))?;
        let cb_quant = self.upload(u16_slice_as_bytes(&plan.cb_quant))?;
        let cr_quant = self.upload(u16_slice_as_bytes(&plan.cr_quant))?;
        let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
        let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
        let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
        let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
        let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
        let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;
        let checkpoints = self.upload(cuda_jpeg_entropy_checkpoints_as_bytes(
            plan.entropy_checkpoints,
        ))?;
        let mut statuses = vec![CudaJpegDecodeStatus::default(); plan.entropy_checkpoints.len()];
        let status_buffer = self.upload(cuda_jpeg_decode_statuses_as_bytes(&statuses))?;
        self.launch_jpeg_decode_rgb8(
            kernel,
            &entropy,
            output,
            validated.params,
            &y_quant,
            &cb_quant,
            &cr_quant,
            &y_dc,
            &y_ac,
            &cb_dc,
            &cb_ac,
            &cr_dc,
            &cr_ac,
            &checkpoints,
            &status_buffer,
        )?;
        status_buffer.copy_to_host(cuda_jpeg_decode_statuses_as_bytes_mut(&mut statuses))?;
        for status in statuses {
            if status.code != 0 {
                return Err(CudaError::KernelStatus {
                    kernel: kernel_name,
                    code: status.code,
                    detail: status.detail,
                });
            }
        }
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[allow(clippy::similar_names)]
    fn diagnose_jpeg_420_entropy_self_sync_nonempty(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        subsequences: usize,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        let params = validate_jpeg_entropy_chunk_plan(plan, subsequences)?;
        self.inner.set_current()?;
        let entropy = self.upload_pinned(plan.entropy_bytes)?;
        let y_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_dc_table))?;
        let y_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.y_ac_table))?;
        let cb_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_dc_table))?;
        let cb_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cb_ac_table))?;
        let cr_dc = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_dc_table))?;
        let cr_ac = self.upload(cuda_jpeg_huffman_table_as_bytes(&plan.cr_ac_table))?;

        let mut states = vec![CudaJpegEntropySyncState::default(); subsequences];
        let states_buffer = self.upload(cuda_jpeg_entropy_sync_states_as_bytes(&states))?;
        self.launch_jpeg_entropy_sync420(
            &entropy,
            params,
            &y_dc,
            &y_ac,
            &cb_dc,
            &cb_ac,
            &cr_dc,
            &cr_ac,
            &states_buffer,
        )?;
        states_buffer.copy_to_host(cuda_jpeg_entropy_sync_states_as_bytes_mut(&mut states))?;

        let mut overflows = vec![
            CudaJpegEntropyOverflowState::default();
            jpeg_entropy_overflow_count(subsequences)
        ];
        if !overflows.is_empty() {
            let overflow_buffer =
                self.upload(cuda_jpeg_entropy_overflow_states_as_bytes(&overflows))?;
            self.launch_jpeg_entropy_overflow420(
                &entropy,
                params,
                &y_dc,
                &y_ac,
                &cb_dc,
                &cb_ac,
                &cr_dc,
                &cr_ac,
                &states_buffer,
                &overflow_buffer,
            )?;
            overflow_buffer.copy_to_host(cuda_jpeg_entropy_overflow_states_as_bytes_mut(
                &mut overflows,
            ))?;
        }

        Ok(CudaJpegChunkedEntropyReport {
            config: plan.config,
            entropy_bytes: plan.entropy_bytes.len(),
            states,
            overflows,
            execution: CudaExecutionStats {
                kernel_dispatches: 1 + usize::from(subsequences > 1),
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    #[allow(clippy::too_many_arguments)]
    fn launch_jpeg_encode_baseline_entropy(
        &self,
        input: &CudaDeviceBuffer,
        input_offset: usize,
        entropy: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
        mut params: CudaJpegBaselineEncodeParams,
        q_luma: &CudaDeviceBuffer,
        q_chroma: &CudaDeviceBuffer,
        huff_dc_luma: &CudaDeviceBuffer,
        huff_ac_luma: &CudaDeviceBuffer,
        huff_dc_chroma: &CudaDeviceBuffer,
        huff_ac_chroma: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropy)?;
        let input_offset = u64::try_from(input_offset)
            .map_err(|_| CudaError::LengthTooLarge { len: input_offset })?;
        let mut input_ptr = input
            .device_ptr()
            .checked_add(input_offset)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let mut entropy_ptr = entropy.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut q_luma_ptr = q_luma.device_ptr();
        let mut q_chroma_ptr = q_chroma.device_ptr();
        let mut huff_dc_luma_ptr = huff_dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = huff_ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = huff_dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = huff_ac_chroma.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr
        );
        self.launch_kernel(
            function,
            CudaLaunchGeometry {
                grid: (1, 1, 1),
                block: (1, 1, 1),
            },
            &mut kernel_params,
        )
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    #[allow(clippy::too_many_arguments)]
    fn launch_jpeg_encode_baseline_entropy_batch(
        &self,
        input: &CudaDeviceBuffer,
        entropy: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
        params: &CudaDeviceBuffer,
        q_luma: &CudaDeviceBuffer,
        q_chroma: &CudaDeviceBuffer,
        huff_dc_luma: &CudaDeviceBuffer,
        huff_ac_luma: &CudaDeviceBuffer,
        huff_dc_chroma: &CudaDeviceBuffer,
        huff_ac_chroma: &CudaDeviceBuffer,
        mut tile_count: u32,
    ) -> Result<(), CudaError> {
        let function =
            self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropyBatch)?;
        let mut input_ptr = input.device_ptr();
        let mut entropy_ptr = entropy.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut params_ptr = params.device_ptr();
        let mut q_luma_ptr = q_luma.device_ptr();
        let mut q_chroma_ptr = q_chroma.device_ptr();
        let mut huff_dc_luma_ptr = huff_dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = huff_ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = huff_dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = huff_ac_chroma.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params_ptr,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr,
            tile_count
        );
        self.launch_kernel(
            function,
            CudaLaunchGeometry {
                grid: (tile_count, 1, 1),
                block: (1, 1, 1),
            },
            &mut kernel_params,
        )
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    fn jpeg_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_encode_kernel_function(kernel)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_jpeg_decode_rgb8(
        &self,
        kernel: CudaKernel,
        entropy: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        mut params: CudaJpeg420Params,
        y_quant: &CudaDeviceBuffer,
        cb_quant: &CudaDeviceBuffer,
        cr_quant: &CudaDeviceBuffer,
        y_dc: &CudaDeviceBuffer,
        y_ac: &CudaDeviceBuffer,
        cb_dc: &CudaDeviceBuffer,
        cb_ac: &CudaDeviceBuffer,
        cr_dc: &CudaDeviceBuffer,
        cr_ac: &CudaDeviceBuffer,
        checkpoints: &CudaDeviceBuffer,
        status: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.jpeg_rgb8_kernel_function(kernel)?;
        let mut entropy_ptr = entropy.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut y_quant_ptr = y_quant.device_ptr();
        let mut cb_quant_ptr = cb_quant.device_ptr();
        let mut cr_quant_ptr = cr_quant.device_ptr();
        let mut y_dc_ptr = y_dc.device_ptr();
        let mut y_ac_ptr = y_ac.device_ptr();
        let mut cb_dc_ptr = cb_dc.device_ptr();
        let mut cb_ac_ptr = cb_ac.device_ptr();
        let mut cr_dc_ptr = cr_dc.device_ptr();
        let mut cr_ac_ptr = cr_ac.device_ptr();
        let mut checkpoints_ptr = checkpoints.device_ptr();
        let mut status_ptr = status.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            entropy_ptr,
            output_ptr,
            params,
            y_quant_ptr,
            cb_quant_ptr,
            cr_quant_ptr,
            y_dc_ptr,
            y_ac_ptr,
            cb_dc_ptr,
            cb_ac_ptr,
            cr_dc_ptr,
            cr_ac_ptr,
            checkpoints_ptr,
            status_ptr
        );
        let geometry = CudaLaunchGeometry {
            grid: (params.checkpoint_count, 1, 1),
            block: (1, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    fn jpeg_rgb8_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_decode_kernel_function(kernel)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_jpeg_entropy_sync420(
        &self,
        entropy: &CudaDeviceBuffer,
        mut params: CudaJpegEntropyChunkParams,
        y_dc: &CudaDeviceBuffer,
        y_ac: &CudaDeviceBuffer,
        cb_dc: &CudaDeviceBuffer,
        cb_ac: &CudaDeviceBuffer,
        cr_dc: &CudaDeviceBuffer,
        cr_ac: &CudaDeviceBuffer,
        states: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropySync420)?;
        let mut entropy_ptr = entropy.device_ptr();
        let mut y_dc_ptr = y_dc.device_ptr();
        let mut y_ac_ptr = y_ac.device_ptr();
        let mut cb_dc_ptr = cb_dc.device_ptr();
        let mut cb_ac_ptr = cb_ac.device_ptr();
        let mut cr_dc_ptr = cr_dc.device_ptr();
        let mut cr_ac_ptr = cr_ac.device_ptr();
        let mut states_ptr = states.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            entropy_ptr,
            params,
            y_dc_ptr,
            y_ac_ptr,
            cb_dc_ptr,
            cb_ac_ptr,
            cr_dc_ptr,
            cr_ac_ptr,
            states_ptr
        );
        let geometry = CudaLaunchGeometry {
            grid: (params.subsequence_count.div_ceil(128), 1, 1),
            block: (128, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn launch_jpeg_entropy_overflow420(
        &self,
        entropy: &CudaDeviceBuffer,
        mut params: CudaJpegEntropyChunkParams,
        y_dc: &CudaDeviceBuffer,
        y_ac: &CudaDeviceBuffer,
        cb_dc: &CudaDeviceBuffer,
        cb_ac: &CudaDeviceBuffer,
        cr_dc: &CudaDeviceBuffer,
        cr_ac: &CudaDeviceBuffer,
        states: &CudaDeviceBuffer,
        overflows: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let function = self.jpeg_entropy_kernel_function(CudaKernel::JpegEntropyOverflow420)?;
        let mut entropy_ptr = entropy.device_ptr();
        let mut y_dc_ptr = y_dc.device_ptr();
        let mut y_ac_ptr = y_ac.device_ptr();
        let mut cb_dc_ptr = cb_dc.device_ptr();
        let mut cb_ac_ptr = cb_ac.device_ptr();
        let mut cr_dc_ptr = cr_dc.device_ptr();
        let mut cr_ac_ptr = cr_ac.device_ptr();
        let mut states_ptr = states.device_ptr();
        let mut overflows_ptr = overflows.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            entropy_ptr,
            params,
            y_dc_ptr,
            y_ac_ptr,
            cb_dc_ptr,
            cb_ac_ptr,
            cr_dc_ptr,
            cr_ac_ptr,
            states_ptr,
            overflows_ptr
        );
        let geometry = CudaLaunchGeometry {
            grid: (
                (params.subsequence_count.saturating_sub(1)).div_ceil(128),
                1,
                1,
            ),
            block: (128, 1, 1),
        };

        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    fn jpeg_entropy_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_decode_kernel_function(kernel)
    }
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
fn validate_jpeg_encode_status(
    status: CudaJpegBaselineEncodeStatus,
    kernel: &'static str,
) -> Result<(), CudaError> {
    match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OK => Ok(()),
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW
        | JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN
        | JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => Err(CudaError::KernelStatus {
            kernel,
            code: status.code,
            detail: status.detail,
        }),
        code => Err(CudaError::KernelStatus {
            kernel,
            code,
            detail: status.detail,
        }),
    }
}
