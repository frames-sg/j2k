// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed mapping from backend-neutral JPEG planning failures to CUDA errors.

use j2k_jpeg::adapter::JpegBaselineGpuEncodeError;
use j2k_jpeg::{JpegBackend, JpegEncodeError};

pub(super) fn cuda_gpu_encode_error(error: JpegBaselineGpuEncodeError) -> crate::Error {
    match error {
        JpegBaselineGpuEncodeError::Encode(error) => error.into(),
        JpegBaselineGpuEncodeError::UnsupportedBackend { requested, .. } => {
            let reason = match requested {
                JpegBackend::Cpu => "JPEG Baseline CUDA encode does not accept Cpu backend",
                JpegBackend::Metal => "JPEG Baseline CUDA encode does not accept Metal backend",
                JpegBackend::Auto | JpegBackend::Cuda => {
                    "JPEG Baseline CUDA encode backend request is inconsistent"
                }
            };
            crate::Error::UnsupportedCudaRequest { reason }
        }
        JpegBaselineGpuEncodeError::InputExceedsOutputDimensions => {
            unsupported("JPEG Baseline CUDA encode input cannot exceed output dimensions")
        }
        JpegBaselineGpuEncodeError::UnsupportedPixelFormat { .. } => {
            unsupported("JPEG Baseline CUDA encode supports only Gray8 and Rgb8 input buffers")
        }
        JpegBaselineGpuEncodeError::IncompatibleSubsampling {
            subsampling,
            samples,
        } => JpegEncodeError::IncompatibleSubsampling {
            subsampling,
            samples,
        }
        .into(),
        JpegBaselineGpuEncodeError::RowByteCountOverflow => {
            unsupported("JPEG Baseline CUDA encode row byte count overflow")
        }
        JpegBaselineGpuEncodeError::PitchTooShort { .. } => {
            unsupported("JPEG Baseline CUDA encode pitch is shorter than one row")
        }
        JpegBaselineGpuEncodeError::InputRangeOverflow => {
            unsupported("JPEG Baseline CUDA encode input range overflow")
        }
        JpegBaselineGpuEncodeError::InputRangeExceedsBuffer { .. } => {
            unsupported("JPEG Baseline CUDA encode input range exceeds buffer length")
        }
        JpegBaselineGpuEncodeError::PitchTooLarge => {
            unsupported("JPEG Baseline CUDA encode pitch exceeds CUDA kernel limits")
        }
        JpegBaselineGpuEncodeError::InputOffsetTooLarge => {
            unsupported("JPEG Baseline CUDA encode input offset exceeds CUDA kernel limits")
        }
        JpegBaselineGpuEncodeError::EntropyOffsetTooLarge => {
            unsupported("JPEG Baseline CUDA encode entropy offset exceeds CUDA kernel limits")
        }
        JpegBaselineGpuEncodeError::EntropyCapacityTooLarge => {
            unsupported("JPEG Baseline CUDA encode entropy capacity exceeds CUDA kernel limits")
        }
        JpegBaselineGpuEncodeError::BatchEntropyCapacityOverflow => {
            unsupported("JPEG Baseline CUDA batch entropy capacity overflow")
        }
    }
}

fn unsupported(reason: &'static str) -> crate::Error {
    crate::Error::UnsupportedCudaRequest { reason }
}
