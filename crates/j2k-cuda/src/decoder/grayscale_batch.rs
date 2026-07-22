// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::DeviceSubmitSession;
use j2k_cuda_runtime::{
    CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Job, CudaJ2kStoreGray8Target,
    CudaJ2kStoreGrayI16Target, CudaQueuedJ2kStoreBatch,
};

use super::color_batch::finalize_color_batch_decode_report;
use super::pending_completion::{PendingCleanup, PendingDecodeCompletion};
use super::plan::{
    build_cuda_classic_grayscale_plans_from_referenced_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_for_device_plan_with_profile,
    build_cuda_htj2k_grayscale_plan_from_bytes_with_profile,
    build_cuda_htj2k_grayscale_plans_from_referenced_with_profile,
};
use super::resident::{
    can_batch_color_idwt, decode_cuda_component_subbands_with_resources,
    enqueue_component_classic_batches, enqueue_component_cleanup_dequant_batches,
    finish_cuda_component_decode, pooled_cuda_buffer, run_color_component_idwt_batches,
    run_component_cleanup_dequant_batches, run_cuda_component_idwt_steps,
};
use super::{
    cuda_error, cuda_range_storage, profile, BackendKind, CudaComponentDecodeWork,
    CudaDecodedComponent, CudaHtj2kDecodePlan, CudaHtj2kProfileReport, CudaQueuedIdwtBatch,
    CudaSession, CudaSurfaceStats, DecodeSettings, DeviceDecodePlan, Error, NativeDecoderContext,
    PixelFormat, Surface, SurfaceResidency, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
use crate::allocation::HostPhaseBudget;

mod completion;
mod execution;
mod preparation;
mod store;

use self::completion::{
    grayscale_htj2k_job_identities, GrayscaleBatchOutput, GrayscaleHtj2kCleanup,
    GrayscalePendingCompletion, StoredGrayscaleBatch,
};
pub(crate) use self::completion::{
    GrayscaleOwnedBatch, SubmittedGrayscaleExternalBatch, SubmittedGrayscaleResidentBatch,
};
use self::execution::decode_grayscale_cuda_batch_with_profile;
#[cfg(test)]
use self::preparation::prepare_grayscale_batch;
pub(crate) use self::preparation::GrayscaleBatchInput;

pub(super) fn decode_grayscale_cuda_resident_batch_surfaces_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let inputs = inputs
        .iter()
        .map(|input| GrayscaleBatchInput::full(input))
        .collect::<Vec<_>>();
    decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile(
        &inputs,
        DecodeSettings::default(),
        session,
        fmt,
        collect_stage_timings,
    )
}

pub(crate) fn decode_grayscale_cuda_resident_prepared_batch_surfaces_with_profile(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let (output, report, pending) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        collect_stage_timings,
        None,
        false,
    )?;
    if pending.is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "synchronous CUDA grayscale decode unexpectedly retained pending work",
        });
    }
    let GrayscaleBatchOutput::Owned(output) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    Ok((output.surfaces, report))
}

pub(super) fn decode_grayscale_cuda_resident_batch_into_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    collect_stage_timings: bool,
) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
    let inputs = inputs
        .iter()
        .map(|input| GrayscaleBatchInput::full(input))
        .collect::<Vec<_>>();
    decode_grayscale_cuda_resident_prepared_batch_into_with_profile(
        &inputs,
        DecodeSettings::default(),
        session,
        fmt,
        destination,
        collect_stage_timings,
    )
}

pub(crate) fn decode_grayscale_cuda_resident_prepared_batch_into_with_profile(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    collect_stage_timings: bool,
) -> Result<(Vec<CudaDeviceBufferRange>, CudaHtj2kProfileReport), Error> {
    let (output, report, pending) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        collect_stage_timings,
        Some(destination),
        false,
    )?;
    if pending.is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "synchronous external CUDA grayscale decode unexpectedly retained pending work",
        });
    }
    let GrayscaleBatchOutput::External(ranges) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    Ok((ranges, report))
}

pub(crate) fn submit_grayscale_cuda_resident_prepared_batch_into(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
    destination: &mut CudaExternalDeviceBufferViewMut<'_>,
) -> Result<SubmittedGrayscaleExternalBatch, Error> {
    let (output, report, completion) = decode_grayscale_cuda_batch_with_profile(
        inputs,
        settings,
        session,
        fmt,
        false,
        Some(destination),
        true,
    )?;
    let GrayscaleBatchOutput::External(ranges) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA external batch submission did not retain a completion owner",
    })?;
    Ok(SubmittedGrayscaleExternalBatch {
        ranges,
        report,
        completion: Some(completion),
    })
}

pub(crate) fn submit_grayscale_cuda_resident_prepared_batch(
    inputs: &[GrayscaleBatchInput<'_>],
    settings: DecodeSettings,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<SubmittedGrayscaleResidentBatch, Error> {
    let (output, report, completion) = decode_grayscale_cuda_batch_with_profile(
        inputs, settings, session, fmt, false, None, true,
    )?;
    let GrayscaleBatchOutput::Owned(output) = output else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        });
    };
    let completion = completion.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA resident grayscale submission did not retain a completion owner",
    })?;
    Ok(SubmittedGrayscaleResidentBatch {
        output: Some(output),
        report,
        completion: Some(completion),
    })
}

#[cfg(test)]
mod tests;
