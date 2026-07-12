// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::{CudaJpegChunkedEntropyPlan, CudaJpegEntropyChunkParams, CudaJpegRgb8Sampling};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::kernels::{CudaKernel, CudaLaunchGeometry};
use crate::{
    context::{ensure_context_ownership, CudaContext},
    error::CudaError,
    memory::CudaDeviceBuffer,
};

#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod decode_plan;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
mod huffman;

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) use decode_plan::{validate_jpeg_rgb8_plan, validate_jpeg_rgb8_plan_with_pitch};

pub(super) const JPEG_CONTEXT_MISMATCH: &str =
    "JPEG CUDA input and output buffers must belong to the launch context";

pub(super) fn validate_jpeg_context_matches(
    matches_context: impl IntoIterator<Item = bool>,
) -> Result<(), CudaError> {
    // An empty buffer set represents an API-level no-op and is valid.
    ensure_context_ownership(matches_context, JPEG_CONTEXT_MISMATCH)
}

pub(super) fn validate_jpeg_buffer_context<'a>(
    context: &CudaContext,
    buffers: impl IntoIterator<Item = &'a CudaDeviceBuffer>,
) -> Result<(), CudaError> {
    validate_jpeg_context_matches(
        buffers
            .into_iter()
            .map(|buffer| buffer.is_owned_by(context)),
    )
}

impl CudaContext {
    /// Validate that a caller-owned JPEG output buffer belongs to this context.
    #[doc(hidden)]
    pub fn validate_jpeg_output_buffer_context(
        &self,
        output: &CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        validate_jpeg_buffer_context(self, [output])
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn validate_jpeg_entropy_chunk_plan(
    plan: &CudaJpegChunkedEntropyPlan<'_>,
    subsequences: usize,
) -> Result<CudaJpegEntropyChunkParams, CudaError> {
    huffman::validate_entropy_huffman_tables(plan)?;
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
    if subsequence_count == 0 {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA entropy diagnostic requires at least one subsequence".to_string(),
        });
    }
    CudaLaunchGeometry::new((subsequence_count.div_ceil(128), 1, 1), (128, 1, 1)).ok_or(
        CudaError::InvalidArgument {
            message: "JPEG entropy sync launch exceeds static CUDA limits".to_string(),
        },
    )?;

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

#[cfg(test)]
mod tests;
