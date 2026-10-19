// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA lossless (SOF3) JPEG encode.
//!
//! The device classifies every sample, prefix-sums code lengths, and packs
//! the bit stream (`cuda_oxide_jpeg_lossless_encode`); between the two
//! submissions the caller turns the device histogram into a code table.
//! Stages and parameters follow `j2k_jpeg::adapter::JpegLosslessGpuEncodeParams`.

use crate::{error::CudaError, memory::CudaDeviceBuffer, JpegCudaEngine};

/// Difference categories 0..=16 of lossless JPEG.
pub const CUDA_JPEG_LOSSLESS_CATEGORIES: usize = 17;

/// Lossless encode kernel parameters, mirroring
/// `j2k_jpeg::adapter::JpegLosslessGpuEncodeParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegLosslessEncodeParams {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// 1 or 3 interleaved components.
    pub components: u32,
    /// Bytes per sample, 1 or 2.
    pub sample_bytes: u32,
    /// Bytes from one row to the next.
    pub pitch_bytes: u32,
    /// Byte offset of the first sample in the input buffer.
    pub input_offset_bytes: u32,
    /// Predictor selection value, 1..=7.
    pub predictor: u32,
    /// Point transform.
    pub point_transform: u32,
    /// 1-D predictor seed, `2^(P - Pt - 1)`.
    pub initial_predictor: u32,
    /// Largest valid sample, `2^P - 1`.
    pub max_sample: u32,
    /// Rows per restart interval, or the height without restarts.
    pub interval_rows: u32,
    /// Pixels of a row per pack thread.
    pub chunk_pixels: u32,
    /// Pack chunks per row.
    pub chunks_per_row: u32,
    /// Restart intervals in the scan.
    pub intervals: u32,
    /// `width * height * components`.
    pub sample_count: u32,
    /// 32-bit words of the packed stream buffer.
    pub packed_words: u32,
}

/// One lossless encode of a CUDA-resident image.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJpegLosslessEncodeJob<'a> {
    /// Resident samples; must belong to the context executing this job.
    pub input: &'a CudaDeviceBuffer,
    /// Validated kernel parameters; `input_offset_bytes` is relative to
    /// the start of `input`.
    pub params: CudaJpegLosslessEncodeParams,
}

/// Result of a device lossless encode.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub enum CudaJpegLosslessEncodeOutcome {
    /// The packed bit stream, intervals back to back in byte order, and the
    /// byte length of each restart interval.
    Packed {
        /// Packed, unstuffed stream bytes.
        stream: Vec<u8>,
        /// Byte length of each restart interval.
        interval_lengths: Vec<u32>,
    },
    /// A sample exceeds `max_sample`; no stream was packed.
    SampleOutOfRange {
        /// Index of the first such sample in scan order.
        index: u32,
        /// Its value.
        value: u32,
    },
}

impl JpegCudaEngine<'_> {
    /// Classify, pack, and read back one lossless encode. `code_table` maps
    /// the device's category histogram to `code << 8 | length` per category.
    ///
    /// # Errors
    /// Returns an error for invalid parameters, an input range outside the
    /// buffer, host allocations above the cap, a CUDA failure, or whatever
    /// `code_table` returns.
    pub fn encode_lossless_with_external_live<E: From<CudaError>>(
        self,
        job: &CudaJpegLosslessEncodeJob<'_>,
        external_live_bytes: usize,
        code_table: impl FnOnce(
            &[u32; CUDA_JPEG_LOSSLESS_CATEGORIES],
        ) -> Result<[u32; CUDA_JPEG_LOSSLESS_CATEGORIES], E>,
    ) -> Result<CudaJpegLosslessEncodeOutcome, E> {
        #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
        {
            self.execute_lossless_encode(job, external_live_bytes, code_table)
        }
        #[cfg(not(feature = "cuda-oxide-jpeg-lossless-encode"))]
        {
            let _ = (job, external_live_bytes, code_table);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG lossless encode PTX was not built".to_string(),
            }
            .into())
        }
    }
}

/// Check the parameter block against itself and the input buffer.
pub(super) fn validate_lossless_encode_job(
    job: &CudaJpegLosslessEncodeJob<'_>,
) -> Result<(), CudaError> {
    let p = job.params;
    let invalid = |message: &str| CudaError::InvalidArgument {
        message: format!("JPEG lossless CUDA encode: {message}"),
    };
    if p.width == 0 || p.height == 0 || p.interval_rows == 0 || p.chunk_pixels == 0 {
        return Err(invalid(
            "dimensions, interval rows, and chunk size must be nonzero",
        ));
    }
    if !matches!(p.components, 1 | 3) || !matches!(p.sample_bytes, 1 | 2) {
        return Err(invalid(
            "expected 1 or 3 components of 1- or 2-byte samples",
        ));
    }
    let samples = u64::from(p.width) * u64::from(p.height) * u64::from(p.components);
    if u64::from(p.sample_count) != samples
        || p.chunks_per_row != p.width.div_ceil(p.chunk_pixels)
        || p.intervals != p.height.div_ceil(p.interval_rows)
        || p.packed_words == 0
    {
        return Err(invalid(
            "derived parameters disagree with the image geometry",
        ));
    }
    if p.sample_bytes == 2
        && !(p.input_offset_bytes.is_multiple_of(2) && p.pitch_bytes.is_multiple_of(2))
    {
        return Err(invalid("16-bit samples must be 2-byte aligned"));
    }
    let row_bytes = u64::from(p.width) * u64::from(p.components) * u64::from(p.sample_bytes);
    if u64::from(p.pitch_bytes) < row_bytes {
        return Err(invalid("pitch is shorter than one row"));
    }
    let end = u64::from(p.input_offset_bytes)
        + u64::from(p.height - 1) * u64::from(p.pitch_bytes)
        + row_bytes;
    if end > job.input.byte_len() as u64 {
        return Err(invalid("input range exceeds the buffer"));
    }
    Ok(())
}

#[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
mod execution;
