// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    aggregate_decode_reports, append_color_payload_to_shared,
    build_cuda_htj2k_color_plans_from_bytes_with_profile, can_batch_color_idwt,
    can_fuse_mct_store_for_stores, cuda_error, cuda_range_storage,
    decode_cuda_component_subbands_with_resources,
    finish_color_cuda_resident_surface_with_component_work, host_owners,
    prepare_rgb8_mct_batch_store, profile, rgb8_mct_batch_store_target,
    run_color_component_idwt_batches, run_component_cleanup_dequant_batches, take_component_work,
    validate_color_stores, Arc, BackendKind, CudaComponentDecodeWork, CudaHtj2kColorDecodePlans,
    CudaHtj2kProfileReport, CudaQueuedIdwtBatch, CudaSession, CudaSurfaceStats, Error,
    FinishColorCudaResidentSurfaceRequest, HostPhaseBudget, NativeDecoderContext, PixelFormat,
    Surface, SurfaceResidency, CUDA_HTJ2K_KERNELS_NOT_READY,
};

#[expect(
    clippy::too_many_lines,
    reason = "resident color batch orchestration keeps CUDA submission and profile ordering atomic"
)]
pub(in crate::decoder) fn decode_color_cuda_resident_batch_surfaces_with_profile(
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

pub(in crate::decoder) fn finalize_color_batch_decode_report(
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
fn can_batch_rgb8_mct_color_store(
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
fn finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
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
