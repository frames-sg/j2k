// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use super::color_batch::CUDA_HTJ2K_BATCH_DECODE_CALLS;
#[cfg(feature = "cuda-runtime")]
use super::color_batch::{
    decode_color_cuda_resident_batch_surfaces_with_profile,
    decode_color_cuda_resident_region_scaled_surface, decode_color_cuda_resident_region_surface,
    decode_color_cuda_resident_scaled_surface, decode_color_cuda_resident_surface_with_profile,
};
#[cfg(feature = "cuda-runtime")]
use super::decode_profile::{
    cuda_idwt_trace_enabled, elapsed_host_us, format_cuda_idwt_batch_host_trace_row,
    CudaIdwtBatchHostTraceRow, CudaIdwtOutputPoolTraceTotals,
};
use super::{
    cuda_error, profile, BackendKind, CudaBufferPool, CudaCoefficientBand, CudaComponentDecodeWork,
    CudaDecodeStageTimings, CudaDecodedComponent, CudaDeviceBuffer, CudaError, CudaHtj2kBandId,
    CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob, CudaHtj2kDecodePlan, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaHtj2kDequantizeTarget, CudaHtj2kIdwtStep,
    CudaHtj2kProfileReport, CudaHtj2kStoreStep, CudaHtj2kTransform, CudaJ2kIdwtJob,
    CudaJ2kIdwtTarget, CudaJ2kRect, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job,
    CudaPendingDequantBand, CudaPooledDeviceBuffer, CudaQueuedExecution, CudaQueuedHtj2kCleanup,
    CudaQueuedIdwtBatch, CudaSession, CudaSurfaceStats, DeviceDecodePlan, DeviceDecodeRequest,
    Downscale, Error, J2kDecoder, PixelFormat, Rect, Storage, Surface, SurfaceResidency,
    CUDA_HTJ2K_KERNELS_NOT_READY, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
    CUDA_HTJ2K_PLAN_INVARIANT_FAILED, CUDA_HTJ2K_STORE_UNSUPPORTED,
};

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, false)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_to_cuda_resident_surface_with_profile_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, true)
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_with_profile_control(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_batch_to_cuda_resident_surface_with_profile_control(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    #[cfg(all(test, feature = "cuda-runtime"))]
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));

    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            CudaHtj2kProfileReport {
                residency: SurfaceResidency::CudaResidentDecode,
                ..CudaHtj2kProfileReport::default()
            },
        ));
    }
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_batch_surfaces_with_profile(
                inputs,
                session,
                fmt,
                collect_stage_timings,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_region_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let plan = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Region { roi },
    )?;
    if plan.is_full_frame() {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }
    let output_dimensions = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Scaled { scale },
    )?
    .output_dims();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_region_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_region_to_cuda_resident_surface_impl(decoder, session, fmt, roi);
    }
    let source_dimensions = decoder.inner.info().dimensions;
    let scaled_dimensions =
        DeviceDecodePlan::for_image(source_dimensions, DeviceDecodeRequest::Scaled { scale })?
            .output_dims();
    let plan = DeviceDecodePlan::for_image(
        source_dimensions,
        DeviceDecodeRequest::RegionScaled { roi, scale },
    )?;
    let scaled_roi = plan.output_rect();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_profile(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_plan_with_profile(fmt)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_region_plan_with_profile(fmt, roi)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_scaled_plan_with_profile(fmt, output_dimensions)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_plan_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    plan: &CudaHtj2kDecodePlan,
    report: &mut CudaHtj2kProfileReport,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let context = session.cuda_context()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    report.h2d_us = report.h2d_us.saturating_add(table_upload_us);
    report.detail.table_upload_us = report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let pool = session.decode_buffer_pool()?;
    let component = decode_cuda_component_plan(
        &context,
        plan,
        &table_resources,
        &pool,
        collect_stage_timings,
    )?;
    let input_width = component
        .store
        .input_rect
        .x1
        .saturating_sub(component.store.input_rect.x0);
    let component_buffer = pooled_cuda_buffer(&component.buffer)?;
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.store.gray",
            || match fmt {
                PixelFormat::Gray8 => context.j2k_store_gray8_device(
                    component_buffer,
                    CudaJ2kStoreGray8Job {
                        input_width,
                        source_x: component.store.source_x,
                        source_y: component.store.source_y,
                        copy_width: component.store.copy_width,
                        copy_height: component.store.copy_height,
                        output_width: component.store.output_width,
                        output_height: component.store.output_height,
                        output_x: component.store.output_x,
                        output_y: component.store.output_y,
                        addend: component.store.addend,
                        bit_depth: u32::from(plan.bit_depth()),
                    },
                ),
                PixelFormat::Gray16 => context.j2k_store_gray16_device(
                    component_buffer,
                    CudaJ2kStoreGray16Job {
                        input_width,
                        source_x: component.store.source_x,
                        source_y: component.store.source_y,
                        copy_width: component.store.copy_width,
                        copy_height: component.store.copy_height,
                        output_width: component.store.output_width,
                        output_height: component.store.output_height,
                        output_x: component.store.output_x,
                        output_y: component.store.output_y,
                        addend: component.store.addend,
                        bit_depth: u32::from(plan.bit_depth()),
                    },
                ),
                _ => Err(CudaError::InvalidArgument {
                    message: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED.to_string(),
                }),
            },
        )
        .map_err(cuda_error)?;
    let (surface_buffer, store_stats) = store_output.into_parts();
    let dispatches = component
        .dispatches
        .saturating_add(store_stats.kernel_dispatches());
    let decode_dispatches = component
        .decode_dispatches
        .saturating_add(store_stats.decode_kernel_dispatches());
    report.dispatch_count = dispatches;
    component.timings.add_to_report(report);
    report.store_us = report.store_us.saturating_add(store_us);
    report.detail.store_dispatch_count = report
        .detail
        .store_dispatch_count
        .saturating_add(store_stats.kernel_dispatches());
    report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(report);
    report.emit("decode");

    let dimensions = (component.store.output_width, component.store.output_height);
    let surface = Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CudaResidentDecode,
        dimensions,
        fmt,
        pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            total: dispatches,
            copy: 0,
            decode: decode_dispatches,
        },
        storage: Storage::Cuda(surface_buffer),
    };
    Ok((surface, report.clone()))
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_to_cuda_resident_surface_with_profile_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_region_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_region_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
pub(super) fn decode_batch_to_cuda_resident_surface_with_profile_control(
    _inputs: &[&[u8]],
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_cuda_component_plan(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    tables: &CudaHtj2kDecodeTableResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let resource_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(plan.payload(), tables)
        .map_err(cuda_error)?;
    let resource_upload_us = profile::elapsed_us(resource_upload_start);
    let mut component = decode_cuda_component_plan_with_resources(
        context,
        plan,
        &decode_resources,
        pool,
        collect_stage_timings,
    )?;
    component.timings.h2d = component.timings.h2d.saturating_add(resource_upload_us);
    component.timings.payload_upload = component
        .timings
        .payload_upload
        .saturating_add(resource_upload_us);
    Ok(component)
}

#[cfg(test)]
pub(super) fn split_htj2k_subband_decode_dispatches(kernel_dispatches: usize) -> (usize, usize) {
    if kernel_dispatches == 0 {
        return (0, 0);
    }

    let dequant_dispatches = usize::from(kernel_dispatches > 1);
    (
        kernel_dispatches.saturating_sub(dequant_dispatches),
        dequant_dispatches,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn htj2k_batched_cleanup_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(super) fn htj2k_batched_dequant_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn htj2k_batched_cleanup_dequant_dispatches(
    target_count: usize,
    fused_cleanup_dequant: bool,
) -> (usize, usize) {
    if target_count == 0 {
        return (0, 0);
    }
    if fused_cleanup_dequant {
        (1, 0)
    } else {
        (1, 1)
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_cuda_component_plan_with_resources(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    decode_resources: &CudaHtj2kDecodeResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let mut work =
        decode_cuda_component_subbands_with_resources(context, plan, pool, collect_stage_timings)?;
    run_component_cleanup_dequant_batches(
        context,
        decode_resources,
        std::slice::from_mut(&mut work),
        pool,
        collect_stage_timings,
    )?;
    run_cuda_component_idwt_steps(
        context,
        plan.idwt_steps(),
        &mut work,
        pool,
        collect_stage_timings,
    )?;
    finish_cuda_component_decode(work)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_cuda_component_subbands_with_resources(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaComponentDecodeWork, Error> {
    let mut bands = Vec::with_capacity(plan.subbands().len() + plan.idwt_steps().len());
    let mut pending_dequant_bands = Vec::with_capacity(plan.subbands().len());
    let dispatches = 0usize;
    let decode_dispatches = 0usize;
    let mut timings = CudaDecodeStageTimings::default();

    for subband in plan.subbands() {
        let start = subband.code_block_start as usize;
        let end = start.checked_add(subband.code_block_count as usize).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            },
        )?;
        let code_blocks =
            plan.code_blocks()
                .get(start..end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                })?;
        let jobs = code_blocks
            .iter()
            .map(|block| cuda_code_block_job_from_plan_block(block, subband.width))
            .collect::<Result<Vec<_>, Error>>()?;
        let output_words = checked_area(subband.width, subband.height)?;
        let allocate_start = profile::profile_now(collect_stage_timings);
        let output = context
            .allocate_htj2k_codeblock_coefficients_with_pool(&jobs, output_words, pool)
            .map_err(cuda_error)?;
        let allocate_wall_us = profile::elapsed_us(allocate_start);
        timings.h2d = timings.h2d.saturating_add(allocate_wall_us);
        let (buffer, _, _) = output.into_parts();
        let band_index = bands.len();
        bands.push(CudaCoefficientBand {
            band_id: subband.band_id,
            buffer,
        });
        if !jobs.is_empty() {
            pending_dequant_bands.push(CudaPendingDequantBand {
                band_index,
                jobs,
                output_words,
            });
        }
    }

    let [store] = plan.store_steps() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_STORE_UNSUPPORTED,
        });
    };

    Ok(CudaComponentDecodeWork {
        bands,
        pending_dequant_bands,
        store: *store,
        dispatches,
        decode_dispatches,
        timings,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn run_component_cleanup_dequant_batches(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<(), Error> {
    let pending_count = component_work
        .iter()
        .map(|work| work.pending_dequant_bands.len())
        .sum::<usize>();
    if pending_count == 0 {
        return Ok(());
    }
    let accounting_index = component_work
        .iter()
        .position(|work| !work.pending_dequant_bands.is_empty())
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;

    let has_refinement = component_work.iter().any(|work| {
        work.pending_dequant_bands.iter().any(|pending| {
            pending
                .jobs
                .iter()
                .any(|job| job.refinement_length > 0 || u32::from(job.number_of_coding_passes) > 1)
        })
    });
    let cleanup_targets = component_work
        .iter()
        .flat_map(|work| {
            work.pending_dequant_bands
                .iter()
                .map(move |pending| (work, pending))
        })
        .map(|(work, pending)| {
            let coefficients = pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
            Ok(CudaHtj2kCleanupTarget {
                coefficients,
                jobs: &pending.jobs,
                output_words: pending.output_words,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if !has_refinement {
        let stage_start = profile::profile_now(collect_stage_timings);
        let ((stats, runtime_timings), fused_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "j2k.htj2k.decode.cleanup_dequantize.batch",
                || {
                    context
                        .decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
                            decode_resources,
                            &cleanup_targets,
                            pool,
                            collect_stage_timings,
                        )
                },
            )
            .map_err(cuda_error)?;
        let stage_wall_us = profile::elapsed_us(stage_start);
        let (cleanup_dispatches, dequant_dispatches) =
            htj2k_batched_cleanup_dequant_dispatches(pending_count, true);
        {
            let accounting = &mut component_work[accounting_index];
            accounting.timings.h2d = accounting
                .timings
                .h2d
                .saturating_add(stage_wall_us.saturating_sub(fused_us));
            accounting.timings.ht_cleanup = accounting.timings.ht_cleanup.saturating_add(fused_us);
            accounting.timings.status_d2h = accounting
                .timings
                .status_d2h
                .saturating_add(runtime_timings.status_d2h_us);
            accounting.timings.ht_dispatch_count = accounting
                .timings
                .ht_dispatch_count
                .saturating_add(cleanup_dispatches);
            accounting.timings.dequant_dispatch_count = accounting
                .timings
                .dequant_dispatch_count
                .saturating_add(dequant_dispatches);
            accounting.dispatches = accounting
                .dispatches
                .saturating_add(stats.kernel_dispatches());
            accounting.decode_dispatches = accounting
                .decode_dispatches
                .saturating_add(stats.decode_kernel_dispatches());
        }

        for work in component_work {
            work.pending_dequant_bands.clear();
        }
        return Ok(());
    }
    let mut queued_cleanup: Option<CudaQueuedHtj2kCleanup> = None;
    let stage_start = profile::profile_now(collect_stage_timings);
    let (stats, cleanup_us, status_d2h_us) = if collect_stage_timings {
        let ((stats, runtime_timings), cleanup_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "j2k.htj2k.decode.cleanup.batch",
                || {
                    context.decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
                        decode_resources,
                        &cleanup_targets,
                        pool,
                        collect_stage_timings,
                    )
                },
            )
            .map_err(cuda_error)?;
        (stats, cleanup_us, runtime_timings.status_d2h_us)
    } else {
        let (queued, cleanup_us) = context
            .time_default_stream_named_us_if(false, "j2k.htj2k.decode.cleanup.batch", || {
                context.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
                    decode_resources,
                    &cleanup_targets,
                    pool,
                )
            })
            .map_err(cuda_error)?;
        let stats = queued.execution();
        queued_cleanup = Some(queued);
        (stats, cleanup_us, 0)
    };
    drop(cleanup_targets);
    let stage_wall_us = profile::elapsed_us(stage_start);
    {
        let accounting = &mut component_work[accounting_index];
        accounting.timings.h2d = accounting
            .timings
            .h2d
            .saturating_add(stage_wall_us.saturating_sub(cleanup_us));
        accounting.timings.ht_cleanup = accounting.timings.ht_cleanup.saturating_add(cleanup_us);
        accounting.timings.status_d2h = accounting.timings.status_d2h.saturating_add(status_d2h_us);
        if has_refinement {
            accounting.timings.ht_refine = accounting.timings.ht_refine.saturating_add(cleanup_us);
        }
        accounting.timings.ht_dispatch_count = accounting
            .timings
            .ht_dispatch_count
            .saturating_add(htj2k_batched_cleanup_dispatches(pending_count));
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
    }

    let stage_start = profile::profile_now(collect_stage_timings);
    let (stats, dequant_us, dequant_target_count) = {
        let dequant_target_count = pending_count;
        let dequant_result = if let Some(queued) = queued_cleanup.as_ref() {
            context.time_default_stream_named_us_if(
                collect_stage_timings,
                "j2k.htj2k.decode.dequantize.batch",
                || context.j2k_dequantize_queued_htj2k_cleanup_with_pool(queued),
            )
        } else {
            let dequant_targets = component_work
                .iter()
                .flat_map(|work| {
                    work.pending_dequant_bands
                        .iter()
                        .map(move |pending| (work, pending))
                })
                .map(|(work, pending)| {
                    let coefficients = pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
                    Ok(CudaHtj2kDequantizeTarget {
                        coefficients,
                        jobs: &pending.jobs,
                        output_words: pending.output_words,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            context.time_default_stream_named_us_if(
                collect_stage_timings,
                "j2k.htj2k.decode.dequantize.batch",
                || {
                    context.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
                        &dequant_targets,
                        pool,
                    )
                },
            )
        };
        let (stats, dequant_us) = match dequant_result {
            Ok(result) => result,
            Err(error) => {
                if let Some(queued) = queued_cleanup.take() {
                    queued.finish().map_err(cuda_error)?;
                }
                return Err(cuda_error(error));
            }
        };
        (stats, dequant_us, dequant_target_count)
    };
    let stage_wall_us = profile::elapsed_us(stage_start);
    {
        let accounting = &mut component_work[accounting_index];
        accounting.timings.h2d = accounting
            .timings
            .h2d
            .saturating_add(stage_wall_us.saturating_sub(dequant_us));
        accounting.timings.dequant = accounting.timings.dequant.saturating_add(dequant_us);
        accounting.timings.dequant_dispatch_count = accounting
            .timings
            .dequant_dispatch_count
            .saturating_add(htj2k_batched_dequant_dispatches(dequant_target_count));
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
    }
    if let Some(queued) = queued_cleanup.take() {
        queued.finish().map_err(cuda_error)?;
    }

    for work in component_work {
        work.pending_dequant_bands.clear();
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn run_cuda_component_idwt_steps(
    context: &j2k_cuda_runtime::CudaContext,
    steps: &[CudaHtj2kIdwtStep],
    work: &mut CudaComponentDecodeWork,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<(), Error> {
    for step in steps {
        let ll = find_cuda_band(&work.bands, step.ll_band_id)?;
        let hl = find_cuda_band(&work.bands, step.hl_band_id)?;
        let lh = find_cuda_band(&work.bands, step.lh_band_id)?;
        let hh = find_cuda_band(&work.bands, step.hh_band_id)?;
        let low_low_device = pooled_cuda_buffer(&ll.buffer)?;
        let high_low_device = pooled_cuda_buffer(&hl.buffer)?;
        let low_high_device = pooled_cuda_buffer(&lh.buffer)?;
        let high_high_device = pooled_cuda_buffer(&hh.buffer)?;
        let job = cuda_idwt_job_from_step(step);
        let (output, idwt_us) = context
            .time_default_stream_named_us_if(collect_stage_timings, "j2k.htj2k.decode.idwt", || {
                if collect_stage_timings {
                    return context.j2k_inverse_dwt_single_device_with_pool(
                        low_low_device,
                        high_low_device,
                        low_high_device,
                        high_high_device,
                        job,
                        pool,
                    );
                }
                context.j2k_inverse_dwt_single_device_untimed_with_pool(
                    low_low_device,
                    high_low_device,
                    low_high_device,
                    high_high_device,
                    job,
                    pool,
                )
            })
            .map_err(cuda_error)?;
        work.timings.idwt = work.timings.idwt.saturating_add(idwt_us);
        let (buffer, stats) = output.into_parts();
        work.dispatches = work.dispatches.saturating_add(stats.kernel_dispatches());
        work.decode_dispatches = work
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
        work.timings.idwt_dispatch_count = work
            .timings
            .idwt_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        work.bands.push(CudaCoefficientBand {
            band_id: step.output_band_id,
            buffer,
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn finish_cuda_component_decode(
    mut work: CudaComponentDecodeWork,
) -> Result<CudaDecodedComponent, Error> {
    let input_index = work
        .bands
        .iter()
        .position(|band| band.band_id == work.store.input_band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let input = work.bands.swap_remove(input_index);
    Ok(CudaDecodedComponent {
        buffer: input.buffer,
        store: work.store,
        dispatches: work.dispatches,
        decode_dispatches: work.decode_dispatches,
        timings: work.timings,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn can_batch_color_idwt(components: &[&CudaHtj2kDecodePlan]) -> bool {
    let Some(first) = components.first() else {
        return false;
    };
    components
        .iter()
        .all(|component| component.idwt_steps().len() == first.idwt_steps().len())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn run_color_component_idwt_batches(
    context: &j2k_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let (queued_batch, idwt_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.idwt.batch",
            || enqueue_color_component_idwt_batches(context, components, component_work, pool),
        )
        .map_err(cuda_error)?;

    if let Some(accounting) = component_work.first_mut() {
        accounting.timings.idwt = accounting.timings.idwt.saturating_add(idwt_us);
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(queued_batch.kernel_dispatches);
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(queued_batch.decode_dispatches);
        accounting.timings.idwt_dispatch_count = accounting
            .timings
            .idwt_dispatch_count
            .saturating_add(queued_batch.kernel_dispatches);
    }
    let _queued_resource_count = queued_batch
        .queued
        .iter()
        .map(CudaQueuedExecution::resource_count)
        .sum::<usize>();
    if collect_stage_timings {
        drop(queued_batch);
        Ok(None)
    } else {
        Ok(Some(queued_batch))
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn enqueue_color_component_idwt_batches(
    context: &j2k_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
) -> Result<CudaQueuedIdwtBatch, CudaError> {
    if components.len() != component_work.len() {
        return Err(CudaError::InvalidArgument {
            message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
        });
    }
    let Some(first) = components.first() else {
        return Ok(CudaQueuedIdwtBatch {
            queued: Vec::new(),
            kernel_dispatches: 0,
            decode_dispatches: 0,
        });
    };

    let mut queued = Vec::with_capacity(first.idwt_steps().len());
    let mut kernel_dispatches = 0usize;
    let mut decode_dispatches = 0usize;
    let step_count = first.idwt_steps().len();
    let trace_enabled = cuda_idwt_trace_enabled();
    let enqueue_result = (|| -> Result<(), CudaError> {
        let mut output_pool_trace = CudaIdwtOutputPoolTraceTotals::default();
        let output_alloc_start = trace_enabled.then(std::time::Instant::now);
        for step_index in 0..step_count {
            for (component_index, component) in components.iter().enumerate() {
                let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                    CudaError::InvalidArgument {
                        message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                    }
                })?;
                let width = step.rect.x1.saturating_sub(step.rect.x0);
                let height = step.rect.y1.saturating_sub(step.rect.y0);
                let output_words = checked_area(width, height).map_err(cuda_invalid_decode_plan)?;
                let output_bytes = output_words
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CudaError::InvalidArgument {
                        message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                    })?;
                let buffer = if trace_enabled {
                    let (buffer, trace) = pool.take_with_trace(output_bytes)?;
                    output_pool_trace.add_take(trace);
                    buffer
                } else {
                    pool.take(output_bytes)?
                };
                component_work[component_index]
                    .bands
                    .push(CudaCoefficientBand {
                        band_id: step.output_band_id,
                        buffer,
                    });
            }
        }
        let output_alloc_us = elapsed_host_us(output_alloc_start);

        let target_build_start = trace_enabled.then(std::time::Instant::now);
        let mut target_batches = Vec::with_capacity(step_count);
        for step_index in 0..step_count {
            let targets = components
                .iter()
                .enumerate()
                .map(|(component_index, component)| {
                    let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                        CudaError::InvalidArgument {
                            message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                        }
                    })?;
                    let work = &component_work[component_index];
                    let ll = find_cuda_band(&work.bands, step.ll_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hl = find_cuda_band(&work.bands, step.hl_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let lh = find_cuda_band(&work.bands, step.lh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hh = find_cuda_band(&work.bands, step.hh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let output = find_cuda_band(&work.bands, step.output_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    Ok(CudaJ2kIdwtTarget {
                        ll: pooled_cuda_buffer(&ll.buffer).map_err(cuda_invalid_decode_plan)?,
                        hl: pooled_cuda_buffer(&hl.buffer).map_err(cuda_invalid_decode_plan)?,
                        lh: pooled_cuda_buffer(&lh.buffer).map_err(cuda_invalid_decode_plan)?,
                        hh: pooled_cuda_buffer(&hh.buffer).map_err(cuda_invalid_decode_plan)?,
                        output: pooled_cuda_buffer(&output.buffer)
                            .map_err(cuda_invalid_decode_plan)?,
                        job: cuda_idwt_job_from_step(step),
                    })
                })
                .collect::<Result<Vec<_>, CudaError>>()?;
            target_batches.push(targets);
        }
        let target_build_us = elapsed_host_us(target_build_start);
        let target_slices = target_batches.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let enqueue_start = trace_enabled.then(std::time::Instant::now);
        let queued_execution =
            context.j2k_inverse_dwt_batch_sequence_enqueue_with_pool(&target_slices, pool)?;
        let enqueue_us = elapsed_host_us(enqueue_start);
        let execution = queued_execution.execution();
        kernel_dispatches = kernel_dispatches.saturating_add(execution.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(execution.decode_kernel_dispatches());
        queued.push(queued_execution);
        if trace_enabled {
            let row = CudaIdwtBatchHostTraceRow {
                component_count: components.len(),
                step_count,
                output_alloc_us,
                target_build_us,
                enqueue_us,
                output_take_count: output_pool_trace.take_count,
                output_pool_reuse_count: output_pool_trace.reuse_count,
                output_pool_alloc_count: output_pool_trace.alloc_count,
                output_pool_scanned_count: output_pool_trace.scanned_count,
                output_pool_max_free_count: output_pool_trace.max_free_count,
                output_requested_bytes: output_pool_trace.requested_bytes,
            };
            eprintln!("{}", format_cuda_idwt_batch_host_trace_row(row));
        }
        Ok(())
    })();
    if let Err(error) = enqueue_result {
        if !queued.is_empty() {
            let _ = context.synchronize();
        }
        return Err(error);
    }

    Ok(CudaQueuedIdwtBatch {
        queued,
        kernel_dispatches,
        decode_dispatches,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_code_block_job_from_plan_block(
    block: &crate::CudaHtj2kCodeBlock,
    subband_width: u32,
) -> Result<CudaHtj2kCodeBlockJob, Error> {
    let output_offset = block
        .output_y
        .checked_mul(subband_width)
        .and_then(|base| base.checked_add(block.output_x))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    Ok(CudaHtj2kCodeBlockJob {
        payload_offset: block.payload_offset,
        width: block.width,
        height: block.height,
        payload_len: block.payload_len,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        missing_bit_planes: block.missing_bit_planes,
        num_bitplanes: block.num_bitplanes,
        number_of_coding_passes: block.number_of_coding_passes,
        output_stride: block.output_stride,
        output_offset,
        dequantization_step: block.dequantization_step,
        stripe_causal: block.stripe_causal != 0,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn validate_color_stores(
    stores: [&CudaHtj2kStoreStep; 3],
    dimensions: (u32, u32),
) -> Result<(), Error> {
    let first = stores[0];
    for store in stores {
        let input_width = store.input_rect.x1.saturating_sub(store.input_rect.x0);
        let input_height = store.input_rect.y1.saturating_sub(store.input_rect.y0);
        let source_end_x =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let source_end_y =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        if store.output_x != 0
            || store.output_y != 0
            || store.copy_width != dimensions.0
            || store.copy_height != dimensions.1
            || store.output_width != dimensions.0
            || store.output_height != dimensions.1
            || source_end_x > input_width
            || source_end_y > input_height
            || store.source_x != first.source_x
            || store.source_y != first.source_y
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn bit_depth_addend(bit_depth: u8) -> f32 {
    let shift = bit_depth.saturating_sub(1).min(15);
    f32::from(1_u16 << shift)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn checked_area(width: u32, height: u32) -> Result<usize, Error> {
    width
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(height as usize))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn find_cuda_band(
    bands: &[CudaCoefficientBand],
    band_id: CudaHtj2kBandId,
) -> Result<&CudaCoefficientBand, Error> {
    bands
        .iter()
        .find(|band| band.band_id == band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn pooled_cuda_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, Error> {
    buffer
        .as_device_buffer()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::needless_pass_by_value)]
pub(super) fn cuda_invalid_decode_plan(error: Error) -> CudaError {
    CudaError::InvalidArgument {
        message: error.to_string(),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_runtime_rect(rect: crate::CudaHtj2kRect) -> CudaJ2kRect {
    CudaJ2kRect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_idwt_job_from_step(step: &CudaHtj2kIdwtStep) -> CudaJ2kIdwtJob {
    CudaJ2kIdwtJob {
        rect: cuda_runtime_rect(step.rect),
        ll_rect: cuda_runtime_rect(step.ll_rect),
        hl_rect: cuda_runtime_rect(step.hl_rect),
        lh_rect: cuda_runtime_rect(step.lh_rect),
        hh_rect: cuda_runtime_rect(step.hh_rect),
        irreversible97: u32::from(step.transform == CudaHtj2kTransform::Irreversible97),
    }
}
