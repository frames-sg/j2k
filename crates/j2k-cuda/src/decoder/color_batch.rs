// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use core::cell::Cell;

#[cfg(feature = "cuda-runtime")]
mod host_owners;
#[cfg(feature = "cuda-runtime")]
mod store;

#[cfg(feature = "cuda-runtime")]
use self::host_owners::{append_color_payload_to_shared, take_component_work};
#[cfg(feature = "cuda-runtime")]
use self::store::{
    can_fuse_mct_store_for_stores, dispatch_color_store, prepare_rgb8_mct_batch_store,
    rgb8_mct_batch_store_target, run_color_mct, validate_color_stores, ColorStoreInputs,
};
#[cfg(feature = "cuda-runtime")]
use super::decode_profile::aggregate_decode_reports;
#[cfg(feature = "cuda-runtime")]
use super::plan::build_cuda_htj2k_color_plans_from_bytes_with_profile;
#[cfg(feature = "cuda-runtime")]
use super::resident::{
    can_batch_color_idwt, decode_cuda_component_subbands_with_resources,
    finish_cuda_component_decode, pooled_cuda_buffer, run_color_component_idwt_batches,
    run_component_cleanup_dequant_batches, run_cuda_component_idwt_steps,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_error, cuda_range_storage, profile, Arc, BackendKind, CudaBufferPool,
    CudaComponentDecodeWork, CudaDecodedComponent, CudaDeviceBuffer, CudaExecutionStats,
    CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession,
    CudaSurfaceStats, Error, J2kDecoder, NativeDecoderContext, PixelFormat, Rect, Storage, Surface,
    SurfaceResidency, CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE, CUDA_HTJ2K_KERNELS_NOT_READY,
};
#[cfg(feature = "cuda-runtime")]
use crate::allocation::HostPhaseBudget;

#[cfg(all(test, feature = "cuda-runtime"))]
std::thread_local! {
    pub(super) static CUDA_HTJ2K_BATCH_DECODE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_reset_cuda_htj2k_batch_decode_calls() {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_cuda_htj2k_batch_decode_calls() -> usize {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(Cell::get)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_color_cuda_resident_surface_with_profile(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let color = decoder.build_cuda_htj2k_color_plans_with_profile(fmt)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_color_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_scaled_plans_with_profile(fmt, output_dimensions)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_color_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_region_plans_with_profile(fmt, roi)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_color_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_region_scaled_plans_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_color_cuda_resident_surface_with_plans_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    if color.components.len() != 3 {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    let context = session.cuda_context()?;
    let pool = session.decode_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = if color
        .components
        .iter()
        .all(|plan| plan.subbands().is_empty())
    {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    color.report.h2d_us = color.report.h2d_us.saturating_add(table_upload_us);
    color.report.detail.table_upload_us = color
        .report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = match table_resources.as_ref() {
        Some(tables) => context.upload_htj2k_decode_resources_with_tables(&color.payload, tables),
        None => context.upload_j2k_decode_payload(&color.payload),
    }
    .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    profile::add_payload_resource_upload_us(&mut color.report, payload_upload_us);
    let mut host_budget = HostPhaseBudget::new("j2k CUDA color decode execution graph");
    color.account_host_owners(&mut host_budget)?;
    let mut component_work = host_budget.try_vec_with_capacity(3)?;
    for plan in &color.components {
        component_work.push(decode_cuda_component_subbands_with_resources(
            &context,
            plan,
            &pool,
            collect_stage_timings,
            &mut host_budget,
        )?);
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut component_work,
        &pool,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    finish_color_cuda_resident_surface_with_component_work(FinishColorCudaResidentSurfaceRequest {
        context: &context,
        pool: &pool,
        fmt,
        color,
        component_work,
        wall_started,
        collect_stage_timings,
        run_idwt: true,
        emit_report: true,
    })
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "resident color batch orchestration keeps CUDA submission and profile ordering atomic"
)]
pub(super) fn decode_color_cuda_resident_batch_surfaces_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let batch_wall_started = profile::profile_now(collect_stage_timings);
    let mut initial_budget = HostPhaseBudget::new("j2k CUDA color batch plan owners");
    let mut colors = initial_budget.try_vec_with_capacity(inputs.len())?;
    let mut shared_payload = Vec::new();
    let mut native_context = NativeDecoderContext::default();
    for input in inputs {
        let mut color =
            build_cuda_htj2k_color_plans_from_bytes_with_profile(input, fmt, &mut native_context)?;
        if color.components.len() != 3 {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        let mut append_budget = host_owners::color_batch_budget(
            &colors,
            &shared_payload,
            Some(&color),
            "j2k CUDA color batch plan owners",
        )?;
        append_color_payload_to_shared(&mut color, &mut shared_payload, &mut append_budget)?;
        colors.push(color);
        host_owners::color_batch_budget(
            &colors,
            &shared_payload,
            None,
            "j2k CUDA color batch plan owners",
        )?;
    }

    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = if colors.iter().all(|color| {
        color
            .components
            .iter()
            .all(|plan| plan.subbands().is_empty())
    }) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = match table_resources.as_ref() {
        Some(tables) => context.upload_htj2k_decode_resources_with_tables_and_pool(
            &shared_payload,
            tables,
            &pool,
        ),
        None => context.upload_j2k_decode_payload_with_pool(&shared_payload, &pool),
    }
    .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    drop(shared_payload);

    let component_count = colors
        .iter()
        .map(|color| color.components.len())
        .sum::<usize>();
    let mut host_budget = HostPhaseBudget::new("j2k CUDA color batch execution graph");
    host_owners::account_colors(&mut host_budget, &colors)?;
    let mut all_component_work = host_budget.try_vec_with_capacity(component_count)?;
    for color in &colors {
        for plan in &color.components {
            all_component_work.push(decode_cuda_component_subbands_with_resources(
                &context,
                plan,
                &pool,
                collect_stage_timings,
                &mut host_budget,
            )?);
        }
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut all_component_work,
        &pool,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    let mut batch_components = host_budget.try_vec_with_capacity(component_count)?;
    for color in &colors {
        for component in &color.components {
            batch_components.push(component);
        }
    }
    let idwt_batched = can_batch_color_idwt(&batch_components);
    let pending_idwt_batch = if idwt_batched {
        run_color_component_idwt_batches(
            &context,
            &batch_components,
            &mut all_component_work,
            &pool,
            collect_stage_timings,
            host_budget.live_bytes(),
        )?
    } else {
        None
    };
    drop(batch_components);

    let completion_result = (|| {
        let can_use_batch_store =
            idwt_batched && can_batch_rgb8_mct_color_store(fmt, &colors, &all_component_work)?;
        let (surfaces, reports) = if can_use_batch_store {
            finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
                &context,
                fmt,
                colors,
                all_component_work,
                collect_stage_timings,
            )?
        } else {
            let mut output_budget = HostPhaseBudget::new("j2k CUDA color batch output graph");
            host_owners::account_colors(&mut output_budget, &colors)?;
            host_owners::account_component_work(&mut output_budget, &all_component_work)?;
            let mut surfaces = output_budget.try_vec_with_capacity(colors.len())?;
            let mut reports = output_budget.try_vec_with_capacity(colors.len())?;
            let mut work_iter = all_component_work.into_iter();
            for color in colors {
                let component_count = color.components.len();
                let component_work =
                    take_component_work(&mut work_iter, component_count, &mut output_budget)?;
                let (surface, report) = finish_color_cuda_resident_surface_with_component_work(
                    FinishColorCudaResidentSurfaceRequest {
                        context: &context,
                        pool: &pool,
                        fmt,
                        color,
                        component_work,
                        wall_started: None,
                        collect_stage_timings,
                        run_idwt: !idwt_batched,
                        emit_report: false,
                    },
                )?;
                surfaces.push(surface);
                reports.push(report);
            }
            (surfaces, reports)
        };
        // Runtime MCT/store launches synchronize before returning, so a
        // recorded dispatch is also a completion point for preceding IDWT.
        let completion_established = reports.iter().any(|report| {
            report.detail.mct_dispatch_count != 0 || report.detail.store_dispatch_count != 0
        });
        Ok(((surfaces, reports), completion_established))
    })();
    let (surfaces, reports) = CudaQueuedIdwtBatch::resolve_optional_after_completed_work(
        pending_idwt_batch,
        completion_result,
    )?;

    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    aggregate.emit("decode_batch");

    Ok((surfaces, aggregate))
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn finalize_color_batch_decode_report(
    reports: &[CudaHtj2kProfileReport],
    table_upload_us: u128,
    payload_upload_us: u128,
    batch_wall_started: Option<profile::ProfileInstant>,
) -> CudaHtj2kProfileReport {
    let mut aggregate = aggregate_decode_reports(reports);
    aggregate.h2d_us = aggregate
        .h2d_us
        .saturating_add(table_upload_us)
        .saturating_add(payload_upload_us);
    aggregate.detail.table_upload_us = aggregate
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    aggregate.detail.payload_upload_us = aggregate
        .detail
        .payload_upload_us
        .saturating_add(payload_upload_us);
    aggregate.detail.wall_total_us = profile::elapsed_us(batch_wall_started);
    profile::finalize_decode_total_us(&mut aggregate);
    aggregate
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn can_batch_rgb8_mct_color_store(
    fmt: PixelFormat,
    colors: &[CudaHtj2kColorDecodePlans],
    all_component_work: &[CudaComponentDecodeWork],
) -> Result<bool, Error> {
    if !matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8) {
        return Ok(false);
    }

    let mut offset = 0usize;
    for color in colors {
        let component_count = color.components.len();
        if component_count != 3 || offset.saturating_add(component_count) > all_component_work.len()
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        if !color.mct {
            return Ok(false);
        }
        let component_work = &all_component_work[offset..offset + component_count];
        let stores = [
            &component_work[0].store,
            &component_work[1].store,
            &component_work[2].store,
        ];
        validate_color_stores(stores, color.dimensions)?;
        if !can_fuse_mct_store_for_stores(stores) {
            return Ok(false);
        }
        offset = offset.saturating_add(component_count);
    }

    if offset != all_component_work.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    Ok(!colors.is_empty())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
    context: &j2k_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    all_component_work: Vec<CudaComponentDecodeWork>,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut host_budget = HostPhaseBudget::new("j2k CUDA prepared color batch store graph");
    host_owners::account_colors(&mut host_budget, &colors)?;
    host_owners::account_component_work(&mut host_budget, &all_component_work)?;
    let mut prepared = host_budget.try_vec_with_capacity(colors.len())?;
    let mut work_iter = all_component_work.into_iter();
    for color in colors {
        let component_count = color.components.len();
        let component_work =
            take_component_work(&mut work_iter, component_count, &mut host_budget)?;
        prepared.push(prepare_rgb8_mct_batch_store(fmt, color, component_work)?);
    }
    if work_iter.next().is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let targets =
        host_budget.try_collect_results_exact(prepared.iter().map(rgb8_mct_batch_store_target))?;
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.store.color.batch",
            || {
                context.j2k_store_rgb8_mct_batch_contiguous_device_with_live_host_bytes(
                    &targets,
                    host_budget.live_bytes(),
                )
            },
        )
        .map_err(cuda_error)?;
    drop(targets);
    let (surface_buffer, surface_ranges, store_stats) = store_output.into_parts();
    if surface_ranges.len() != prepared.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    let shared_surface_buffer = Arc::new(surface_buffer);

    let mut output_budget = HostPhaseBudget::new("j2k CUDA stored color batch output graph");
    output_budget.account_vec(&prepared)?;
    for item in &prepared {
        item.color.account_host_owners(&mut output_budget)?;
    }
    output_budget.account_vec(&surface_ranges)?;
    let mut surfaces = output_budget.try_vec_with_capacity(prepared.len())?;
    let mut reports = output_budget.try_vec_with_capacity(prepared.len())?;
    let store_dispatches = store_stats.kernel_dispatches();
    let store_decode_dispatches = store_stats.decode_kernel_dispatches();
    for (index, (mut prepared, surface_range)) in
        prepared.into_iter().zip(surface_ranges).enumerate()
    {
        let report_store_dispatches = if index == 0 { store_dispatches } else { 0 };
        let report_store_decode_dispatches = if index == 0 {
            store_decode_dispatches
        } else {
            0
        };
        let report_store_us = if index == 0 { store_us } else { 0 };
        let dispatches = prepared.dispatches.saturating_add(report_store_dispatches);
        let decode_dispatches = prepared
            .decode_dispatches
            .saturating_add(report_store_decode_dispatches);
        prepared.color.report.dispatch_count = dispatches;
        prepared.color.report.store_us = prepared
            .color
            .report
            .store_us
            .saturating_add(report_store_us);
        prepared.color.report.detail.store_dispatch_count = prepared
            .color
            .report
            .detail
            .store_dispatch_count
            .saturating_add(report_store_dispatches);
        profile::finalize_decode_total_us(&mut prepared.color.report);

        let dimensions = prepared.color.dimensions;
        surfaces.push(Surface {
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
            storage: cuda_range_storage(
                shared_surface_buffer.clone(),
                surface_range.offset,
                surface_range.len,
            ),
        });
        reports.push(prepared.color.report);
    }

    Ok((surfaces, reports))
}

#[cfg(feature = "cuda-runtime")]
struct FinishColorCudaResidentSurfaceRequest<'a> {
    context: &'a j2k_cuda_runtime::CudaContext,
    pool: &'a CudaBufferPool,
    fmt: PixelFormat,
    color: CudaHtj2kColorDecodePlans,
    component_work: Vec<CudaComponentDecodeWork>,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
    run_idwt: bool,
    emit_report: bool,
}

#[cfg(feature = "cuda-runtime")]
struct PreparedColorComponents {
    components: [CudaDecodedComponent; 3],
    dispatches: usize,
    decode_dispatches: usize,
}

#[cfg(feature = "cuda-runtime")]
struct FinalizeColorSurfaceRequest {
    fmt: PixelFormat,
    color: CudaHtj2kColorDecodePlans,
    surface_buffer: CudaDeviceBuffer,
    dispatches: usize,
    decode_dispatches: usize,
    store_stats: CudaExecutionStats,
    store_us: u128,
    wall_started: Option<profile::ProfileInstant>,
    emit_report: bool,
}

#[cfg(feature = "cuda-runtime")]
fn run_pending_color_idwt(
    context: &j2k_cuda_runtime::CudaContext,
    color: &CudaHtj2kColorDecodePlans,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    host_budget: &mut HostPhaseBudget,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let mut batch_components = host_budget.try_vec_with_capacity(color.components.len())?;
    for component in &color.components {
        batch_components.push(component);
    }
    if can_batch_color_idwt(&batch_components) {
        run_color_component_idwt_batches(
            context,
            &batch_components,
            component_work,
            pool,
            collect_stage_timings,
            host_budget.live_bytes(),
        )
    } else {
        for (plan, work) in color.components.iter().zip(component_work.iter_mut()) {
            run_cuda_component_idwt_steps(
                context,
                plan.idwt_steps(),
                work,
                pool,
                collect_stage_timings,
            )?;
        }
        Ok(None)
    }
}

#[cfg(feature = "cuda-runtime")]
fn finish_color_components(
    component_work: Vec<CudaComponentDecodeWork>,
    color: &mut CudaHtj2kColorDecodePlans,
) -> Result<PreparedColorComponents, Error> {
    let [work0, work1, work2]: [CudaComponentDecodeWork; 3] =
        component_work
            .try_into()
            .map_err(|_| Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            })?;
    let components = [
        finish_cuda_component_decode(work0)?,
        finish_cuda_component_decode(work1)?,
        finish_cuda_component_decode(work2)?,
    ];
    let stores = [
        &components[0].store,
        &components[1].store,
        &components[2].store,
    ];
    validate_color_stores(stores, color.dimensions)?;

    let dispatches = components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let decode_dispatches = components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &components {
        component.timings.add_to_report(&mut color.report);
    }
    Ok(PreparedColorComponents {
        components,
        dispatches,
        decode_dispatches,
    })
}

#[cfg(feature = "cuda-runtime")]
fn finalize_color_surface(
    request: FinalizeColorSurfaceRequest,
) -> (Surface, CudaHtj2kProfileReport) {
    let FinalizeColorSurfaceRequest {
        fmt,
        mut color,
        surface_buffer,
        mut dispatches,
        mut decode_dispatches,
        store_stats,
        store_us,
        wall_started,
        emit_report,
    } = request;
    dispatches = dispatches.saturating_add(store_stats.kernel_dispatches());
    decode_dispatches = decode_dispatches.saturating_add(store_stats.decode_kernel_dispatches());
    color.report.dispatch_count = dispatches;
    color.report.store_us = color.report.store_us.saturating_add(store_us);
    color.report.detail.store_dispatch_count = color
        .report
        .detail
        .store_dispatch_count
        .saturating_add(store_stats.kernel_dispatches());
    color.report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(&mut color.report);
    if emit_report {
        color.report.emit("decode");
    }
    let surface = Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CudaResidentDecode,
        dimensions: color.dimensions,
        fmt,
        pitch_bytes: color.dimensions.0 as usize * fmt.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            total: dispatches,
            copy: 0,
            decode: decode_dispatches,
        },
        storage: Storage::Cuda(surface_buffer),
    };
    (surface, color.report)
}

#[cfg(feature = "cuda-runtime")]
fn finish_color_cuda_resident_surface_with_component_work(
    request: FinishColorCudaResidentSurfaceRequest<'_>,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let FinishColorCudaResidentSurfaceRequest {
        context,
        pool,
        fmt,
        mut color,
        mut component_work,
        wall_started,
        collect_stage_timings,
        run_idwt,
        emit_report,
    } = request;
    let mut host_budget =
        host_owners::color_work_budget(&color, &component_work, "j2k CUDA color completion graph")?;
    let pending_idwt_batch = if run_idwt {
        run_pending_color_idwt(
            context,
            &color,
            &mut component_work,
            pool,
            collect_stage_timings,
            &mut host_budget,
        )?
    } else {
        None
    };
    let completion_result = (|| {
        let prepared = finish_color_components(component_work, &mut color)?;
        let inputs = ColorStoreInputs {
            context,
            buffers: [
                pooled_cuda_buffer(&prepared.components[0].buffer)?,
                pooled_cuda_buffer(&prepared.components[1].buffer)?,
                pooled_cuda_buffer(&prepared.components[2].buffer)?,
            ],
            stores: [
                &prepared.components[0].store,
                &prepared.components[1].store,
                &prepared.components[2].store,
            ],
            bit_depths: color.bit_depths,
        };
        let mct = run_color_mct(
            inputs,
            color.mct_dimensions,
            color.mct,
            color.transform,
            collect_stage_timings,
        )?;
        let dispatches = prepared.dispatches.saturating_add(mct.kernel_dispatches);
        let decode_dispatches = prepared
            .decode_dispatches
            .saturating_add(mct.decode_kernel_dispatches);
        color.report.mct_us = color.report.mct_us.saturating_add(mct.elapsed_us);
        color.report.detail.mct_dispatch_count = color
            .report
            .detail
            .mct_dispatch_count
            .saturating_add(mct.kernel_dispatches);
        let (store_output, store_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "j2k.htj2k.decode.store.color",
                || dispatch_color_store(inputs, mct, fmt),
            )
            .map_err(cuda_error)?;
        let (surface_buffer, store_stats) = store_output.into_parts();
        // Both runtime paths synchronize their kernel launch before success.
        let completion_established =
            mct.kernel_dispatches != 0 || store_stats.kernel_dispatches() != 0;
        let output = finalize_color_surface(FinalizeColorSurfaceRequest {
            fmt,
            color,
            surface_buffer,
            dispatches,
            decode_dispatches,
            store_stats,
            store_us,
            wall_started,
            emit_report,
        });
        Ok((output, completion_established))
    })();
    CudaQueuedIdwtBatch::resolve_optional_after_completed_work(
        pending_idwt_batch,
        completion_result,
    )
}
