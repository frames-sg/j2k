// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use core::cell::Cell;

#[cfg(feature = "cuda-runtime")]
use super::decode_profile::aggregate_decode_reports;
#[cfg(feature = "cuda-runtime")]
use super::plan::build_cuda_htj2k_color_plans_from_bytes_with_profile;
#[cfg(feature = "cuda-runtime")]
use super::resident::{
    bit_depth_addend, can_batch_color_idwt, checked_area,
    decode_cuda_component_subbands_with_resources, finish_cuda_component_decode,
    pooled_cuda_buffer, run_color_component_idwt_batches, run_component_cleanup_dequant_batches,
    run_cuda_component_idwt_steps, validate_color_stores,
};
#[cfg(feature = "cuda-runtime")]
use super::{
    cuda_error, cuda_range_storage, profile, Arc, BackendKind, CudaBufferPool,
    CudaComponentDecodeWork, CudaError, CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport,
    CudaHtj2kStoreStep, CudaHtj2kTransform, CudaJ2kInverseMctJob, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget, CudaPreparedRgb8MctBatchStore, CudaSession, CudaSurfaceStats, Error,
    J2kDecoder, NativeDecoderContext, PixelFormat, Rect, Storage, Surface, SurfaceResidency,
    CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE, CUDA_HTJ2K_KERNELS_NOT_READY,
    CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};

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
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    color.report.h2d_us = color.report.h2d_us.saturating_add(table_upload_us);
    color.report.detail.table_upload_us = color
        .report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(&color.payload, &table_resources)
        .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    profile::add_payload_resource_upload_us(&mut color.report, payload_upload_us);
    let mut component_work = Vec::with_capacity(3);
    for plan in &color.components {
        component_work.push(decode_cuda_component_subbands_with_resources(
            &context,
            plan,
            &pool,
            collect_stage_timings,
        )?);
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut component_work,
        &pool,
        collect_stage_timings,
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
    let mut colors = Vec::with_capacity(inputs.len());
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
        append_color_payload_to_shared(&mut color, &mut shared_payload)?;
        colors.push(color);
    }

    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables_and_pool(
            &shared_payload,
            &table_resources,
            &pool,
        )
        .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);

    let component_count = colors
        .iter()
        .map(|color| color.components.len())
        .sum::<usize>();
    let mut all_component_work = Vec::with_capacity(component_count);
    for color in &colors {
        for plan in &color.components {
            all_component_work.push(decode_cuda_component_subbands_with_resources(
                &context,
                plan,
                &pool,
                collect_stage_timings,
            )?);
        }
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut all_component_work,
        &pool,
        collect_stage_timings,
    )?;
    let batch_components = colors
        .iter()
        .flat_map(|color| color.components.iter())
        .collect::<Vec<_>>();
    let idwt_batched = can_batch_color_idwt(&batch_components);
    let pending_idwt_batch = if idwt_batched {
        run_color_component_idwt_batches(
            &context,
            &batch_components,
            &mut all_component_work,
            &pool,
            collect_stage_timings,
        )?
    } else {
        None
    };
    drop(batch_components);

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
        let mut surfaces = Vec::with_capacity(colors.len());
        let mut reports = Vec::with_capacity(colors.len());
        let mut work_iter = all_component_work.into_iter();
        for color in colors {
            let component_count = color.components.len();
            let component_work = work_iter.by_ref().take(component_count).collect::<Vec<_>>();
            if component_work.len() != component_count {
                return Err(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                });
            }
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
    drop(pending_idwt_batch);

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
    let mut prepared = Vec::with_capacity(colors.len());
    let mut work_iter = all_component_work.into_iter();
    for color in colors {
        let component_count = color.components.len();
        let component_work = work_iter.by_ref().take(component_count).collect::<Vec<_>>();
        if component_work.len() != component_count {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        prepared.push(prepare_rgb8_mct_batch_store(fmt, color, component_work)?);
    }
    if work_iter.next().is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let targets = prepared
        .iter()
        .map(rgb8_mct_batch_store_target)
        .collect::<Result<Vec<_>, Error>>()?;
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.store.color.batch",
            || context.j2k_store_rgb8_mct_batch_contiguous_device(&targets),
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

    let mut surfaces = Vec::with_capacity(prepared.len());
    let mut reports = Vec::with_capacity(prepared.len());
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
pub(super) fn prepare_rgb8_mct_batch_store(
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    component_work: Vec<CudaComponentDecodeWork>,
) -> Result<CudaPreparedRgb8MctBatchStore, Error> {
    let decoded_components = component_work
        .into_iter()
        .map(finish_cuda_component_decode)
        .collect::<Result<Vec<_>, Error>>()?;
    let [component0, component1, component2] = decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    let stores = [&component0.store, &component1.store, &component2.store];
    validate_color_stores(stores, color.dimensions)?;
    if !color.mct || !can_fuse_mct_store_for_stores(stores) {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let dispatches = decoded_components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let decode_dispatches = decoded_components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &decoded_components {
        component.timings.add_to_report(&mut color.report);
    }

    let addends = [
        bit_depth_addend(color.bit_depths[0]),
        bit_depth_addend(color.bit_depths[1]),
        bit_depth_addend(color.bit_depths[2]),
    ];
    let job = CudaJ2kStoreRgb8MctJob {
        store: CudaJ2kStoreRgb8Job {
            input_width0: color_store_input_width(&component0.store),
            input_width1: color_store_input_width(&component1.store),
            input_width2: color_store_input_width(&component2.store),
            source_x0: component0.store.source_x,
            source_y0: component0.store.source_y,
            source_x1: component1.store.source_x,
            source_y1: component1.store.source_y,
            source_x2: component2.store.source_x,
            source_y2: component2.store.source_y,
            copy_width: component0.store.copy_width,
            copy_height: component0.store.copy_height,
            output_width: component0.store.output_width,
            output_height: component0.store.output_height,
            output_x: component0.store.output_x,
            output_y: component0.store.output_y,
            addend0: addends[0],
            addend1: addends[1],
            addend2: addends[2],
            bit_depth0: u32::from(color.bit_depths[0]),
            bit_depth1: u32::from(color.bit_depths[1]),
            bit_depth2: u32::from(color.bit_depths[2]),
            rgba: u32::from(fmt == PixelFormat::Rgba8),
        },
        irreversible97: u32::from(color.transform == CudaHtj2kTransform::Irreversible97),
    };

    Ok(CudaPreparedRgb8MctBatchStore {
        color,
        decoded_components,
        dispatches,
        decode_dispatches,
        job,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn rgb8_mct_batch_store_target(
    prepared: &CudaPreparedRgb8MctBatchStore,
) -> Result<CudaJ2kStoreRgb8MctTarget<'_>, Error> {
    let [component0, component1, component2] = prepared.decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    Ok(CudaJ2kStoreRgb8MctTarget {
        plane0: pooled_cuda_buffer(&component0.buffer)?,
        plane1: pooled_cuda_buffer(&component1.buffer)?,
        plane2: pooled_cuda_buffer(&component2.buffer)?,
        job: prepared.job,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn can_fuse_mct_store_for_stores(stores: [&CudaHtj2kStoreStep; 3]) -> bool {
    let input_width0 = color_store_input_width(stores[0]);
    let input_width1 = color_store_input_width(stores[1]);
    let input_width2 = color_store_input_width(stores[2]);
    input_width0 == input_width1
        && input_width0 == input_width2
        && stores[0].source_x == stores[1].source_x
        && stores[0].source_x == stores[2].source_x
        && stores[0].source_y == stores[1].source_y
        && stores[0].source_y == stores[2].source_y
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn color_store_input_width(store: &CudaHtj2kStoreStep) -> u32 {
    store.input_rect.x1.saturating_sub(store.input_rect.x0)
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
#[expect(
    clippy::too_many_lines,
    reason = "surface harvest keeps component buffers, color conversion, and timings synchronized"
)]
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
    let pending_idwt_batch = if run_idwt {
        let batch_components = color.components.iter().collect::<Vec<_>>();
        if can_batch_color_idwt(&batch_components) {
            run_color_component_idwt_batches(
                context,
                &batch_components,
                &mut component_work,
                pool,
                collect_stage_timings,
            )?
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
            None
        }
    } else {
        None
    };
    let decoded_components = component_work
        .into_iter()
        .map(finish_cuda_component_decode)
        .collect::<Result<Vec<_>, Error>>()?;
    let [component0, component1, component2] = decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    validate_color_stores(
        [&component0.store, &component1.store, &component2.store],
        color.dimensions,
    )?;

    let mut dispatches = decoded_components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let mut decode_dispatches = decoded_components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &decoded_components {
        component.timings.add_to_report(&mut color.report);
    }
    let component0_buffer = pooled_cuda_buffer(&component0.buffer)?;
    let component1_buffer = pooled_cuda_buffer(&component1.buffer)?;
    let component2_buffer = pooled_cuda_buffer(&component2.buffer)?;
    let stores = [&component0.store, &component1.store, &component2.store];
    let input_width0 = color_store_input_width(stores[0]);
    let input_width1 = color_store_input_width(stores[1]);
    let input_width2 = color_store_input_width(stores[2]);
    let irreversible97 = u32::from(color.transform == CudaHtj2kTransform::Irreversible97);
    let mct_store_addends = [
        bit_depth_addend(color.bit_depths[0]),
        bit_depth_addend(color.bit_depths[1]),
        bit_depth_addend(color.bit_depths[2]),
    ];
    let can_fuse_mct_store = color.mct && can_fuse_mct_store_for_stores(stores);
    let addends = if color.mct && can_fuse_mct_store {
        mct_store_addends
    } else if color.mct {
        let mct_len = u32::try_from(checked_area(
            color.mct_dimensions.0,
            color.mct_dimensions.1,
        )?)
        .map_err(|_| Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
        let stats = context
            .time_default_stream_named_us_if(collect_stage_timings, "j2k.htj2k.decode.mct", || {
                context.j2k_inverse_mct_device(
                    component0_buffer,
                    component1_buffer,
                    component2_buffer,
                    CudaJ2kInverseMctJob {
                        len: mct_len,
                        irreversible97,
                        addend0: mct_store_addends[0],
                        addend1: mct_store_addends[1],
                        addend2: mct_store_addends[2],
                    },
                )
            })
            .map_err(cuda_error)?;
        let (stats, mct_us) = stats;
        dispatches = dispatches.saturating_add(stats.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(stats.decode_kernel_dispatches());
        color.report.mct_us = color.report.mct_us.saturating_add(mct_us);
        color.report.detail.mct_dispatch_count = color
            .report
            .detail
            .mct_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        [0.0, 0.0, 0.0]
    } else {
        [
            component0.store.addend,
            component1.store.addend,
            component2.store.addend,
        ]
    };
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "j2k.htj2k.decode.store.color",
            || match fmt {
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    let store_job = CudaJ2kStoreRgb8Job {
                        input_width0,
                        input_width1,
                        input_width2,
                        source_x0: component0.store.source_x,
                        source_y0: component0.store.source_y,
                        source_x1: component1.store.source_x,
                        source_y1: component1.store.source_y,
                        source_x2: component2.store.source_x,
                        source_y2: component2.store.source_y,
                        copy_width: component0.store.copy_width,
                        copy_height: component0.store.copy_height,
                        output_width: component0.store.output_width,
                        output_height: component0.store.output_height,
                        output_x: component0.store.output_x,
                        output_y: component0.store.output_y,
                        addend0: addends[0],
                        addend1: addends[1],
                        addend2: addends[2],
                        bit_depth0: u32::from(color.bit_depths[0]),
                        bit_depth1: u32::from(color.bit_depths[1]),
                        bit_depth2: u32::from(color.bit_depths[2]),
                        rgba: u32::from(fmt == PixelFormat::Rgba8),
                    };
                    if can_fuse_mct_store {
                        context.j2k_store_rgb8_mct_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            CudaJ2kStoreRgb8MctJob {
                                store: store_job,
                                irreversible97,
                            },
                        )
                    } else {
                        context.j2k_store_rgb8_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            store_job,
                        )
                    }
                }
                PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                    let store_job = CudaJ2kStoreRgb16Job {
                        input_width0,
                        input_width1,
                        input_width2,
                        source_x0: component0.store.source_x,
                        source_y0: component0.store.source_y,
                        source_x1: component1.store.source_x,
                        source_y1: component1.store.source_y,
                        source_x2: component2.store.source_x,
                        source_y2: component2.store.source_y,
                        copy_width: component0.store.copy_width,
                        copy_height: component0.store.copy_height,
                        output_width: component0.store.output_width,
                        output_height: component0.store.output_height,
                        output_x: component0.store.output_x,
                        output_y: component0.store.output_y,
                        addend0: addends[0],
                        addend1: addends[1],
                        addend2: addends[2],
                        bit_depth0: u32::from(color.bit_depths[0]),
                        bit_depth1: u32::from(color.bit_depths[1]),
                        bit_depth2: u32::from(color.bit_depths[2]),
                        rgba: u32::from(fmt == PixelFormat::Rgba16),
                    };
                    if can_fuse_mct_store {
                        context.j2k_store_rgb16_mct_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            CudaJ2kStoreRgb16MctJob {
                                store: store_job,
                                irreversible97,
                            },
                        )
                    } else {
                        context.j2k_store_rgb16_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            store_job,
                        )
                    }
                }
                _ => Err(CudaError::InvalidArgument {
                    message: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED.to_string(),
                }),
            },
        )
        .map_err(cuda_error)?;
    drop(pending_idwt_batch);
    let (surface_buffer, store_stats) = store_output.into_parts();
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
    Ok((surface, color.report))
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn append_color_payload_to_shared(
    color: &mut CudaHtj2kColorDecodePlans,
    shared_payload: &mut Vec<u8>,
) -> Result<(), Error> {
    let base = u64::try_from(shared_payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
    })?;
    shared_payload
        .try_reserve(color.payload.len())
        .map_err(|_| Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
        })?;
    for component in &mut color.components {
        component.rebase_payload_offsets(base)?;
    }
    shared_payload.append(&mut color.payload);
    Ok(())
}
