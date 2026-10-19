// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA lossless (SOF3) JPEG encode of a CUDA-resident image.

#[cfg(feature = "cuda-runtime")]
use j2k_core::PixelFormat;
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_jpeg_engine::{
    CudaJpegLosslessEncodeJob, CudaJpegLosslessEncodeOutcome, CudaJpegLosslessEncodeParams,
    JpegCudaEngine, CUDA_JPEG_LOSSLESS_CATEGORIES,
};
#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaDeviceBuffer, CudaError};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::{
    assemble_jpeg_lossless_frame, finish_jpeg_lossless_gpu_entropy, jpeg_lossless_huffman_table,
    plan_jpeg_lossless_gpu_encode, JpegLosslessGpuEncodeError, JpegLosslessGpuEncodeParams,
    JpegLosslessGpuEncodePlan, JpegLosslessGpuEncodeTile, JpegLosslessHuffmanTable,
};
use j2k_jpeg::{EncodedJpeg, JpegLosslessEncodeOptions};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{JpegBackend, JpegEncodeError};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;

#[cfg(feature = "cuda-runtime")]
#[derive(Debug, Clone, Copy)]
/// CUDA buffer and layout metadata for one lossless JPEG encode.
pub struct JpegLosslessCudaEncodeTile<'a> {
    /// Source CUDA buffer of `Gray8`, `Gray16`, `Rgb8`, or `Rgb16` samples
    /// (16-bit samples native-endian and 2-byte aligned).
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset of the first sample in `buffer`.
    pub byte_offset: usize,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of bytes between consecutive rows.
    pub pitch_bytes: usize,
    /// Sample layout of the buffer.
    pub format: PixelFormat,
}

#[cfg(not(feature = "cuda-runtime"))]
#[derive(Debug, Clone, Copy)]
/// Placeholder lossless encode tile type for builds without `cuda-runtime`.
pub struct JpegLosslessCudaEncodeTile<'a> {
    _private: core::marker::PhantomData<&'a ()>,
}

/// Encode one CUDA-resident image as a lossless (SOF3) JPEG.
///
/// Output is byte-identical to [`j2k_jpeg::encode_jpeg_lossless`] for the
/// same samples and options. `options.backend` must be `Auto` or `Cuda`.
///
/// # Errors
///
/// Returns the CPU encoder's errors for invalid samples or options
/// (including a sample above the precision), unsupported-request errors for
/// an input range outside the buffer or misaligned 16-bit samples, and CUDA
/// errors for a failed submission or missing PTX.
#[cfg(feature = "cuda-runtime")]
pub fn encode_jpeg_lossless_from_cuda_buffer(
    tile: JpegLosslessCudaEncodeTile<'_>,
    options: JpegLosslessEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<EncodedJpeg, crate::Error> {
    let plan = plan_jpeg_lossless_gpu_encode(
        JpegLosslessGpuEncodeTile {
            byte_offset: tile.byte_offset,
            width: tile.width,
            height: tile.height,
            pitch_bytes: tile.pitch_bytes,
            format: tile.format,
            buffer_len: tile.buffer.byte_len(),
        },
        options,
        JpegBackend::Cuda,
    )
    .map_err(lossless_plan_error)?;
    let operation_gate = session.jpeg_host_operation_gate();
    let _operation = operation_gate
        .lock()
        .map_err(|_| crate::Error::JpegHostOperationPoisoned)?;
    let context = session.bind_cuda_context(&tile.buffer.context())?;
    let pinned_upload = context
        .begin_pinned_upload_operation()
        .map_err(cuda_error)?;
    let pinned_accounting = session.reserve_pinned_upload_retention(&context, &pinned_upload)?;
    let external_live_bytes = session.owned_host_live_bytes()?;
    let result = encode_on_device(tile, &plan, external_live_bytes);
    pinned_accounting.finish(result)
}

/// Return `Error::CudaUnavailable` for builds without `cuda-runtime`.
#[cfg(not(feature = "cuda-runtime"))]
pub fn encode_jpeg_lossless_from_cuda_buffer(
    tile: JpegLosslessCudaEncodeTile<'_>,
    options: JpegLosslessEncodeOptions,
    session: &mut crate::CudaSession,
) -> Result<EncodedJpeg, crate::Error> {
    let _ = (tile, options, session);
    Err(crate::Error::CudaUnavailable)
}

/// Adapter errors raised inside the engine's table callback.
#[cfg(feature = "cuda-runtime")]
struct DeviceEncodeError(crate::Error);

#[cfg(feature = "cuda-runtime")]
impl From<CudaError> for DeviceEncodeError {
    fn from(error: CudaError) -> Self {
        Self(cuda_error(error))
    }
}

#[cfg(feature = "cuda-runtime")]
fn encode_on_device(
    tile: JpegLosslessCudaEncodeTile<'_>,
    plan: &JpegLosslessGpuEncodePlan,
    external_live_bytes: usize,
) -> Result<EncodedJpeg, crate::Error> {
    let mut table: Option<JpegLosslessHuffmanTable> = None;
    let outcome = JpegCudaEngine::new(&tile.buffer.context())
        .encode_lossless_with_external_live(
            &CudaJpegLosslessEncodeJob {
                input: tile.buffer,
                params: engine_params(&plan.params),
            },
            external_live_bytes,
            |counts| {
                let mut histogram = [0_u64; CUDA_JPEG_LOSSLESS_CATEGORIES];
                for (count, &device_count) in histogram.iter_mut().zip(counts) {
                    *count = u64::from(device_count);
                }
                let built = jpeg_lossless_huffman_table(&histogram)
                    .map_err(|error| DeviceEncodeError(error.into()))?;
                let codes = built.codes();
                let entries = core::array::from_fn(|category| {
                    (u32::from(codes.codes[category]) << 8) | u32::from(codes.lens[category])
                });
                table = Some(built);
                Ok::<_, DeviceEncodeError>(entries)
            },
        )
        .map_err(|DeviceEncodeError(error)| error)?;
    match outcome {
        CudaJpegLosslessEncodeOutcome::Packed {
            stream,
            interval_lengths,
        } => {
            let table = table.ok_or(JpegEncodeError::InternalInvariant {
                reason: "CUDA lossless encode packed a stream without a table",
            })?;
            let entropy = finish_jpeg_lossless_gpu_entropy(&stream, &interval_lengths)?;
            Ok(assemble_jpeg_lossless_frame(
                &plan.plan,
                &table,
                &entropy,
                JpegBackend::Cuda,
            )?)
        }
        CudaJpegLosslessEncodeOutcome::SampleOutOfRange { index, value } => {
            Err(JpegEncodeError::SampleOutOfRange {
                index: index as usize,
                value: u16::try_from(value).unwrap_or(u16::MAX),
                precision: plan.plan.precision,
            }
            .into())
        }
    }
}

#[cfg(feature = "cuda-runtime")]
fn engine_params(params: &JpegLosslessGpuEncodeParams) -> CudaJpegLosslessEncodeParams {
    CudaJpegLosslessEncodeParams {
        width: params.width,
        height: params.height,
        components: params.components,
        sample_bytes: params.sample_bytes,
        pitch_bytes: params.pitch_bytes,
        input_offset_bytes: params.input_offset_bytes,
        predictor: params.predictor,
        point_transform: params.point_transform,
        initial_predictor: params.initial_predictor,
        max_sample: params.max_sample,
        interval_rows: params.interval_rows,
        chunk_pixels: params.chunk_pixels,
        chunks_per_row: params.chunks_per_row,
        intervals: params.intervals,
        sample_count: params.sample_count,
        packed_words: params.packed_words,
    }
}

#[cfg(feature = "cuda-runtime")]
fn lossless_plan_error(error: JpegLosslessGpuEncodeError) -> crate::Error {
    let unsupported = |reason| crate::Error::UnsupportedCudaRequest { reason };
    match error {
        JpegLosslessGpuEncodeError::Encode(error) => error.into(),
        JpegLosslessGpuEncodeError::UnsupportedBackend { .. } => {
            unsupported("JPEG lossless CUDA encode accepts only the Auto or Cuda backend")
        }
        JpegLosslessGpuEncodeError::PitchTooShort { .. } => {
            unsupported("JPEG lossless CUDA encode pitch is shorter than one row")
        }
        JpegLosslessGpuEncodeError::UnalignedSamples { .. } => {
            unsupported("JPEG lossless CUDA encode needs 2-byte aligned 16-bit samples")
        }
        JpegLosslessGpuEncodeError::InputRangeExceedsBuffer { .. } => {
            unsupported("JPEG lossless CUDA encode input range exceeds buffer length")
        }
        JpegLosslessGpuEncodeError::TooLarge { reason } => unsupported(reason),
    }
}
