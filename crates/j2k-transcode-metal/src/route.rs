// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use j2k_core::{BackendKind, BackendRequest};
use j2k_transcode::{
    BatchTranscodeReport, EncodedTranscode, EncodedTranscodeBatch, JpegTileBatchInput,
    JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kTranscoder, TranscodePipelineMap,
    TranscodeStageError, TranscodeTimingReport,
};
#[cfg(target_os = "macos")]
use j2k_transcode::{ResidentBufferRef, ResidentCodestreamBuffer, ResidentHandoffError};

use crate::MetalDctToWaveletStageAccelerator;

const CUDA_REQUESTED_THROUGH_METAL_ADAPTER: &str = "CUDA transcode requested through Metal adapter";
const STRICT_METAL_TRANSCODE_NO_DISPATCH: &str =
    "strict Metal transcode produced no Metal dispatch";

/// Structured CPU fallback reason for the Metal JPEG-to-HTJ2K route facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalTranscodeFallbackReason {
    /// The caller requested CPU explicitly.
    CpuRequested,
    /// Auto found no transform stage eligible for Metal.
    AutoNoEligibleTransformStage,
    /// Auto offered transform jobs to Metal, but all jobs used CPU fallback.
    AutoAllTransformJobsFellBackToCpu,
    /// Auto used Metal for some transform jobs and CPU fallback for others.
    AutoPartialTransformFallback,
}

impl MetalTranscodeFallbackReason {
    /// Stable reason label for logs, examples, and benchmark output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuRequested => "cpu_requested",
            Self::AutoNoEligibleTransformStage => "auto_no_eligible_transform_stage",
            Self::AutoAllTransformJobsFellBackToCpu => "auto_all_transform_jobs_fell_back_to_cpu",
            Self::AutoPartialTransformFallback => "auto_partial_transform_fallback",
        }
    }
}

impl fmt::Display for MetalTranscodeFallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Public route report for Metal-adapted JPEG-to-HTJ2K transcode calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalTranscodeRouteReport {
    /// Caller backend request.
    pub request: BackendRequest,
    /// Backend that handled transform stages.
    pub selected_transform_backend: BackendKind,
    /// Backend that produced the public codestream byte vector.
    pub output_backend: BackendKind,
    /// Structured CPU fallback reason when the route did not fully use Metal.
    pub fallback_reason: Option<MetalTranscodeFallbackReason>,
    /// Stage residency map derived from the existing transcode timing counters.
    pub pipeline_map: TranscodePipelineMap,
}

/// JPEG-to-HTJ2K transcode output plus Metal route report.
pub struct MetalEncodedTranscode {
    /// Encoded HTJ2K codestream and native transcode report.
    pub encoded: EncodedTranscode,
    /// Route and residency report for this call.
    pub route: MetalTranscodeRouteReport,
}

/// Batch JPEG-to-HTJ2K transcode output plus Metal route report.
pub struct MetalEncodedTranscodeBatch {
    /// Per-tile outputs and aggregate native transcode report.
    pub batch: EncodedTranscodeBatch,
    /// Route and residency report for this batch call.
    pub route: MetalTranscodeRouteReport,
}

/// Build a backend-neutral resident codestream descriptor from a Metal encode output.
#[cfg(target_os = "macos")]
pub fn resident_codestream_buffer_from_metal_encoded_j2k(
    encoded: &j2k_metal::MetalEncodedJ2k,
) -> Result<ResidentCodestreamBuffer<'_>, ResidentHandoffError> {
    let memory = encoded
        .codestream_memory_range()
        .ok_or(ResidentHandoffError::OffsetOverflow)?;
    let allocation_len = encoded
        .codestream_allocation_len()
        .ok_or(ResidentHandoffError::RangeExceedsAllocation)?;
    let buffer = ResidentBufferRef::with_allocation_len(memory, allocation_len)?;
    ResidentCodestreamBuffer::new(buffer, encoded.byte_len(), encoded.capacity())?
        .require_backend(BackendKind::Metal)
}

/// Transcode JPEG to HTJ2K using CPU, Auto Metal, or strict Metal routing.
///
/// `BackendRequest::Metal` uses the explicit Metal accelerator and returns an
/// error if Metal is unavailable or the required transform stage is unsupported.
/// `BackendRequest::Auto` may return CPU output with a structured fallback reason.
pub fn jpeg_to_htj2k_with_metal_route(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    request: BackendRequest,
) -> Result<MetalEncodedTranscode, JpegToHtj2kError> {
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let encoded = match request {
        BackendRequest::Cpu => transcoder.transcode(bytes, options)?,
        BackendRequest::Auto => {
            let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();
            transcoder.transcode_with_accelerator(bytes, options, &mut accelerator)?
        }
        BackendRequest::Metal => {
            let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
            let encoded =
                transcoder.transcode_with_accelerator(bytes, options, &mut accelerator)?;
            ensure_strict_metal_dispatched(&encoded.report.timings)?;
            encoded
        }
        BackendRequest::Cuda => {
            return Err(JpegToHtj2kError::Unsupported(
                CUDA_REQUESTED_THROUGH_METAL_ADAPTER,
            ));
        }
    };
    let route = route_report(request, &encoded.report.timings);
    Ok(MetalEncodedTranscode { encoded, route })
}

/// Batch transcode JPEG tiles to HTJ2K using CPU, Auto Metal, or strict Metal routing.
pub fn jpeg_to_htj2k_batch_with_metal_route(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    request: BackendRequest,
) -> Result<MetalEncodedTranscodeBatch, JpegToHtj2kError> {
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let batch = match request {
        BackendRequest::Cpu => transcoder.transcode_batch(tiles, options)?,
        BackendRequest::Auto => {
            let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();
            transcoder.transcode_batch_with_accelerator(tiles, options, &mut accelerator)?
        }
        BackendRequest::Metal => {
            let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
            let batch =
                transcoder.transcode_batch_with_accelerator(tiles, options, &mut accelerator)?;
            ensure_strict_metal_batch_dispatched(&batch.report)?;
            batch
        }
        BackendRequest::Cuda => {
            return Err(JpegToHtj2kError::Unsupported(
                CUDA_REQUESTED_THROUGH_METAL_ADAPTER,
            ));
        }
    };
    let route = route_report(request, &batch.report.timings);
    Ok(MetalEncodedTranscodeBatch { batch, route })
}

pub(crate) fn route_report(
    request: BackendRequest,
    timings: &TranscodeTimingReport,
) -> MetalTranscodeRouteReport {
    let selected_transform_backend = selected_transform_backend(timings);
    MetalTranscodeRouteReport {
        request,
        selected_transform_backend,
        output_backend: BackendKind::Cpu,
        fallback_reason: fallback_reason(request, selected_transform_backend, timings),
        pipeline_map: timings.pipeline_map(),
    }
}

fn selected_transform_backend(timings: &TranscodeTimingReport) -> BackendKind {
    if timings.accelerator_work_observed() {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    }
}

fn fallback_reason(
    request: BackendRequest,
    selected_transform_backend: BackendKind,
    timings: &TranscodeTimingReport,
) -> Option<MetalTranscodeFallbackReason> {
    match request {
        BackendRequest::Cpu => Some(MetalTranscodeFallbackReason::CpuRequested),
        BackendRequest::Auto
            if selected_transform_backend == BackendKind::Cpu
                && timings.accelerator_attempts == 0 =>
        {
            Some(MetalTranscodeFallbackReason::AutoNoEligibleTransformStage)
        }
        BackendRequest::Auto if selected_transform_backend == BackendKind::Cpu => {
            Some(MetalTranscodeFallbackReason::AutoAllTransformJobsFellBackToCpu)
        }
        BackendRequest::Auto if timings.cpu_fallback_jobs > 0 => {
            Some(MetalTranscodeFallbackReason::AutoPartialTransformFallback)
        }
        BackendRequest::Metal | BackendRequest::Cuda | BackendRequest::Auto => None,
    }
}

pub(crate) fn ensure_strict_metal_dispatched(
    timings: &TranscodeTimingReport,
) -> Result<(), JpegToHtj2kError> {
    if timings.accelerator_work_observed() {
        Ok(())
    } else {
        Err(JpegToHtj2kError::Accelerator(
            TranscodeStageError::Unsupported(STRICT_METAL_TRANSCODE_NO_DISPATCH),
        ))
    }
}

pub(crate) fn ensure_strict_metal_batch_dispatched(
    report: &BatchTranscodeReport,
) -> Result<(), JpegToHtj2kError> {
    if report.successful_tiles == 0 || report.timings.accelerator_work_observed() {
        Ok(())
    } else {
        Err(JpegToHtj2kError::Accelerator(
            TranscodeStageError::Unsupported(STRICT_METAL_TRANSCODE_NO_DISPATCH),
        ))
    }
}
