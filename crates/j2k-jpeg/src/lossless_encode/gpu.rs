// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral planning and host finishing for device lossless encode.
//!
//! Device encoders run the same stages on Metal and CUDA:
//!
//! 1. classify: one thread per sample predicts, differences, and
//!    categorizes it, storing a token (category in the low 8 bits,
//!    additional bits above bit 16) and counting categories.
//! 2. The host builds the table with [`super::jpeg_lossless_huffman_table`].
//! 3. chunk bits: one thread per chunk of [`JPEG_LOSSLESS_GPU_CHUNK_PIXELS`]
//!    pixels of a row sums its tokens' code lengths.
//! 4. row offsets: one thread per row prefix-sums its chunks; interval
//!    offsets: one thread per restart interval prefix-sums its rows and
//!    rounds its length up to whole bytes; interval bases: one thread
//!    prefix-sums the interval lengths.
//! 5. pack: one thread per chunk ORs its codes into a zeroed buffer of
//!    32-bit words holding the bit stream in byte order, and the last chunk
//!    of each interval pads its final byte with 1-bits.
//! 6. The host copies each interval out with JPEG byte stuffing and restart
//!    markers between intervals ([`finish_jpeg_lossless_gpu_entropy`]), then
//!    assembles the frame with [`super::assemble_jpeg_lossless_frame`].

use alloc::vec::Vec;

use j2k_codec_math::jpeg::lossless::MAX_DIFFERENCE_BITS;
use j2k_core::PixelFormat;

use super::{plan_jpeg_lossless_encode, JpegLosslessEncodeOptions, JpegLosslessEncodePlan};
use crate::encoded_output::{CappedBytes, JPEG_BASELINE_MAX_ENTROPY_BYTES};
use crate::encoder::{JpegBackend, JpegEncodeError};

/// Pixels of one row each pack thread codes.
pub const JPEG_LOSSLESS_GPU_CHUNK_PIXELS: u32 = 32;

/// A device-resident source image for lossless encode.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegLosslessGpuEncodeTile {
    /// Byte offset of the first sample in the buffer.
    pub byte_offset: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bytes from one row to the next.
    pub pitch_bytes: usize,
    /// `Gray8`, `Gray16`, `Rgb8`, or `Rgb16` (native-endian samples).
    pub format: PixelFormat,
    /// Length of the device buffer in bytes.
    pub buffer_len: usize,
}

/// Device kernel parameters, identical on every backend.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JpegLosslessGpuEncodeParams {
    pub width: u32,
    pub height: u32,
    /// 1 or 3, interleaved.
    pub components: u32,
    /// 1 or 2.
    pub sample_bytes: u32,
    pub pitch_bytes: u32,
    pub input_offset_bytes: u32,
    pub predictor: u32,
    pub point_transform: u32,
    pub initial_predictor: u32,
    /// `2^P - 1`; larger samples are rejected.
    pub max_sample: u32,
    /// Rows per restart interval, or `height` without restart markers.
    pub interval_rows: u32,
    pub chunk_pixels: u32,
    pub chunks_per_row: u32,
    pub intervals: u32,
    /// `width * height * components`.
    pub sample_count: u32,
    /// 32-bit words of the packed bit-stream buffer.
    pub packed_words: u32,
}

impl JpegLosslessGpuEncodeParams {
    /// Chunks in the image, one pack thread each.
    #[must_use]
    pub const fn chunks(&self) -> u32 {
        self.chunks_per_row * self.height
    }
}

/// A validated device lossless encode.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegLosslessGpuEncodePlan {
    pub plan: JpegLosslessEncodePlan,
    pub params: JpegLosslessGpuEncodeParams,
}

/// Error returned by backend-neutral device lossless encode planning.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum JpegLosslessGpuEncodeError {
    /// The samples or options are invalid, as for CPU encode.
    Encode(JpegEncodeError),
    /// The requested backend does not match the device encoder.
    UnsupportedBackend {
        /// Requested backend.
        requested: JpegBackend,
        /// Backend of the device encoder.
        expected: JpegBackend,
    },
    /// The pitch is shorter than one row of samples.
    PitchTooShort {
        /// Bytes one row of samples takes.
        row_bytes: usize,
        /// Supplied pitch.
        pitch_bytes: usize,
    },
    /// 16-bit samples need an even offset and pitch.
    UnalignedSamples {
        /// Required byte alignment.
        alignment: usize,
    },
    /// The input range does not fit the buffer.
    InputRangeExceedsBuffer {
        /// Exclusive end of the sample range.
        required_end: usize,
        /// Length of the buffer.
        buffer_len: usize,
    },
    /// A size exceeds the 32-bit device addressing of the kernels.
    TooLarge {
        /// What does not fit.
        reason: &'static str,
    },
}

impl From<JpegEncodeError> for JpegLosslessGpuEncodeError {
    fn from(error: JpegEncodeError) -> Self {
        Self::Encode(error)
    }
}

/// Validate a device lossless encode of `tile` exactly as CPU encode
/// validates its samples, plus the device input range, and derive the
/// kernel parameters.
///
/// # Errors
/// Returns the CPU encoder's errors for invalid options, and range,
/// alignment, and size errors for the device tile.
#[doc(hidden)]
pub fn plan_jpeg_lossless_gpu_encode(
    tile: JpegLosslessGpuEncodeTile,
    options: JpegLosslessEncodeOptions,
    expected_backend: JpegBackend,
) -> Result<JpegLosslessGpuEncodePlan, JpegLosslessGpuEncodeError> {
    if !matches!(options.backend, JpegBackend::Auto) && options.backend != expected_backend {
        return Err(JpegLosslessGpuEncodeError::UnsupportedBackend {
            requested: options.backend,
            expected: expected_backend,
        });
    }
    let plan = plan_jpeg_lossless_encode(tile.format, (tile.width, tile.height), options)?;
    let sample_bytes = if matches!(tile.format, PixelFormat::Gray16 | PixelFormat::Rgb16) {
        2
    } else {
        1
    };
    if sample_bytes == 2
        && !(tile.byte_offset.is_multiple_of(2) && tile.pitch_bytes.is_multiple_of(2))
    {
        return Err(JpegLosslessGpuEncodeError::UnalignedSamples { alignment: 2 });
    }
    let too_large = |reason| JpegLosslessGpuEncodeError::TooLarge { reason };
    let row_bytes = tile.width as usize * usize::from(plan.components) * sample_bytes;
    if tile.pitch_bytes < row_bytes {
        return Err(JpegLosslessGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes: tile.pitch_bytes,
        });
    }
    let required_end = (tile.height as usize - 1)
        .checked_mul(tile.pitch_bytes)
        .and_then(|last_row| last_row.checked_add(row_bytes))
        .and_then(|span| span.checked_add(tile.byte_offset))
        .ok_or(too_large("lossless input range"))?;
    if required_end > tile.buffer_len {
        return Err(JpegLosslessGpuEncodeError::InputRangeExceedsBuffer {
            required_end,
            buffer_len: tile.buffer_len,
        });
    }
    let narrow = |value: usize, reason| u32::try_from(value).map_err(|_| too_large(reason));
    narrow(required_end, "lossless input range")?;
    let packed_bytes = jpeg_lossless_gpu_packed_capacity(&plan)?;
    let chunks_per_row = tile.width.div_ceil(JPEG_LOSSLESS_GPU_CHUNK_PIXELS);
    let params = JpegLosslessGpuEncodeParams {
        width: tile.width,
        height: tile.height,
        components: u32::from(plan.components),
        sample_bytes: if sample_bytes == 2 { 2 } else { 1 },
        pitch_bytes: narrow(tile.pitch_bytes, "lossless pitch")?,
        input_offset_bytes: narrow(tile.byte_offset, "lossless input offset")?,
        predictor: u32::from(plan.predictor),
        point_transform: u32::from(plan.point_transform),
        initial_predictor: u32::from(plan.initial_predictor),
        max_sample: plan.max_sample(),
        interval_rows: if plan.restart_rows == 0 {
            plan.height
        } else {
            plan.restart_rows
        },
        chunk_pixels: JPEG_LOSSLESS_GPU_CHUNK_PIXELS,
        chunks_per_row,
        intervals: narrow(plan.restart_intervals(), "lossless restart intervals")?,
        sample_count: narrow(plan.samples(), "lossless sample count")?,
        packed_words: narrow(packed_bytes / 4, "lossless packed stream")?,
    };
    narrow(
        params.chunks_per_row as usize * tile.height as usize,
        "lossless chunk count",
    )?;
    Ok(JpegLosslessGpuEncodePlan { plan, params })
}

/// Bytes of the packed bit-stream buffer: every sample at its longest code,
/// plus one padding byte per interval, rounded up to whole words.
fn jpeg_lossless_gpu_packed_capacity(
    plan: &JpegLosslessEncodePlan,
) -> Result<usize, JpegLosslessGpuEncodeError> {
    let too_large = JpegLosslessGpuEncodeError::TooLarge {
        reason: "lossless packed stream",
    };
    let bits = plan
        .samples()
        .checked_mul(MAX_DIFFERENCE_BITS as usize)
        .ok_or(too_large.clone())?;
    let bytes = bits
        .div_ceil(8)
        .checked_add(plan.restart_intervals())
        .ok_or(too_large.clone())?;
    let words_bytes = bytes.div_ceil(4).checked_add(1).ok_or(too_large.clone())? * 4;
    if words_bytes > JPEG_BASELINE_MAX_ENTROPY_BYTES {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested: words_bytes,
            cap: JPEG_BASELINE_MAX_ENTROPY_BYTES,
        }
        .into());
    }
    Ok(words_bytes)
}

/// Copy the device's packed intervals into entropy-coded data: each
/// interval's bytes with a zero after every 0xFF, and RST0..RST7 between
/// intervals.
///
/// `packed` holds the intervals back to back in byte order; `interval_lengths`
/// are their byte lengths.
///
/// # Errors
/// Returns an invariant error when the lengths exceed `packed`, or an
/// allocation error.
#[doc(hidden)]
pub fn finish_jpeg_lossless_gpu_entropy(
    packed: &[u8],
    interval_lengths: &[u32],
) -> Result<Vec<u8>, JpegEncodeError> {
    let total = interval_lengths
        .iter()
        .try_fold(0_usize, |sum, &len| sum.checked_add(len as usize))
        .filter(|&total| total <= packed.len())
        .ok_or(JpegEncodeError::InternalInvariant {
            reason: "device lossless intervals exceed the packed stream",
        })?;
    let estimate = total
        .saturating_add(total / 128)
        .saturating_add(interval_lengths.len().saturating_mul(2))
        .min(JPEG_BASELINE_MAX_ENTROPY_BYTES);
    let mut out = CappedBytes::try_with_capacity(estimate, JPEG_BASELINE_MAX_ENTROPY_BYTES)?;
    let mut start = 0;
    for (index, &len) in interval_lengths.iter().enumerate() {
        if index > 0 {
            let marker = 0xD0 + u8::try_from((index - 1) % 8).unwrap_or(0);
            out.extend_from_slice(&[0xFF, marker])?;
        }
        let end = start + len as usize;
        for run in packed[start..end].split_inclusive(|&byte| byte == 0xFF) {
            out.extend_from_slice(run)?;
            if run.last() == Some(&0xFF) {
                out.push(0)?;
            }
        }
        start = end;
    }
    Ok(out.into_vec())
}

#[cfg(test)]
mod tests;
