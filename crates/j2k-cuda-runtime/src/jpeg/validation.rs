// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::{
    CudaJpeg420Params, CudaJpegChunkedEntropyPlan, CudaJpegEntropyChunkParams,
    CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling, CudaJpegRgb8ValidatedPlan,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::{error::CudaError, kernels::CudaKernel};

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
