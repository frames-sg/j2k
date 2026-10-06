// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{CudaDeviceBufferRange, CudaExecutionStats, CudaPooledDeviceBuffer};

use super::FinishColorBatchRequest;

use crate::surface::cuda_pooled_range_storage;

use super::super::super::store::CudaPreparedRgb8MctBatchStore;
use super::super::super::{
    can_fuse_mct_store_for_stores, cuda_error, host_owners, prepare_rgb8_mct_batch_store, profile,
    rgb8_mct_batch_store_target, take_component_work, validate_color_stores, Arc, BackendKind,
    CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport, CudaSurfaceStats,
    Error, HostPhaseBudget, PixelFormat, Surface, SurfaceResidency, CUDA_HTJ2K_KERNELS_NOT_READY,
};

pub(in crate::decoder::color_batch::batch_execution) fn can_batch_rgb8_mct_color_store(
    fmt: PixelFormat,
    colors: &[CudaHtj2kColorDecodePlans],
    component_work: &[CudaComponentDecodeWork],
) -> Result<bool, Error> {
    if !matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8) {
        return Ok(false);
    }
    let mut offset = 0usize;
    for color in colors {
        let component_count = color.components.len();
        if component_count != 3 || offset.saturating_add(component_count) > component_work.len() {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
            ));
        }
        if !color.mct {
            return Ok(false);
        }
        let work = &component_work[offset..offset + component_count];
        let stores = [&work[0].store, &work[1].store, &work[2].store];
        validate_color_stores(stores, color.dimensions)?;
        if !can_fuse_mct_store_for_stores(stores) {
            return Ok(false);
        }
        offset = offset.saturating_add(component_count);
    }
    if offset != component_work.len() {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
        ));
    }
    Ok(!colors.is_empty())
}

pub(super) fn finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
    request: FinishColorBatchRequest<'_>,
    fused_final_vertical: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>), Error> {
    let FinishColorBatchRequest {
        context,
        pool,
        fmt,
        colors,
        component_work,
        collect_stage_timings,
        external_live_host_bytes,
    } = request;
    let (mut host_budget, prepared) =
        prepare_batch_store_items(fmt, colors, component_work, external_live_host_bytes)?;
    let targets =
        host_budget.try_collect_results_exact(prepared.iter().map(rgb8_mct_batch_store_target))?;
    let mut run = || {
        j2k_cuda_j2k_engine::J2kCudaEngine::new(context)
            .j2k_store_rgb8_mct_batch_contiguous_device_with_pool(
                &targets,
                host_budget.live_bytes(),
                fused_final_vertical,
                pool,
            )
    };
    let (store_output, store_us) = if collect_stage_timings {
        context.time_default_stream_named_us("j2k.htj2k.decode.store.color.batch", &mut run)
    } else {
        // Successful work is already complete; preserve error-path completion
        // before any captured allocations can be released.
        context
            .with_nvtx_range("j2k.htj2k.decode.store.color.batch", run)
            .or_else(|error| context.synchronize_then_error(error))
            .map(|output| (output, 0))
    }
    .map_err(cuda_error)?;
    drop(targets);
    let (store_output, surface_ranges) = store_output;
    let (surface_buffer, store_stats) = store_output.into_parts();
    if surface_ranges.len() != prepared.len() {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
        ));
    }
    assemble_batch_store_surfaces(
        fmt,
        prepared,
        surface_buffer,
        surface_ranges,
        store_stats,
        store_us,
        external_live_host_bytes,
    )
}

fn prepare_batch_store_items(
    fmt: PixelFormat,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    external_live_host_bytes: usize,
) -> Result<(HostPhaseBudget, Vec<CudaPreparedRgb8MctBatchStore>), Error> {
    let mut host_budget = HostPhaseBudget::with_live_bytes(
        "j2k CUDA prepared color batch store graph",
        external_live_host_bytes,
    )?;
    host_owners::account_colors(&mut host_budget, &colors)?;
    host_owners::account_component_work(&mut host_budget, &component_work)?;
    let mut prepared = host_budget.try_vec_with_capacity(colors.len())?;
    let mut work_iter = component_work.into_iter();
    for color in colors {
        let component_count = color.components.len();
        let work = take_component_work(&mut work_iter, component_count, &mut host_budget)?;
        prepared.push(prepare_rgb8_mct_batch_store(fmt, color, work)?);
    }
    if work_iter.next().is_some() {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
        ));
    }
    Ok((host_budget, prepared))
}

fn assemble_batch_store_surfaces(
    fmt: PixelFormat,
    prepared: Vec<CudaPreparedRgb8MctBatchStore>,
    surface_buffer: CudaPooledDeviceBuffer,
    surface_ranges: Vec<CudaDeviceBufferRange>,
    store_stats: CudaExecutionStats,
    store_us: u128,
    external_live_host_bytes: usize,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut output_budget = HostPhaseBudget::with_live_bytes(
        "j2k CUDA stored color batch output graph",
        external_live_host_bytes,
    )?;
    output_budget.account_vec(&prepared)?;
    for item in &prepared {
        item.color.account_host_owners(&mut output_budget)?;
    }
    output_budget.account_vec(&surface_ranges)?;
    let mut surfaces = output_budget.try_vec_with_capacity(prepared.len())?;
    let mut reports = output_budget.try_vec_with_capacity(prepared.len())?;
    let shared_surface_buffer = Arc::new(surface_buffer);
    for (index, (mut prepared, surface_range)) in
        prepared.into_iter().zip(surface_ranges).enumerate()
    {
        let (dispatches, decode_dispatches) =
            record_batch_store_profile(&mut prepared, index, store_stats, store_us);
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
            storage: cuda_pooled_range_storage(
                shared_surface_buffer.clone(),
                surface_range.offset,
                surface_range.len,
            ),
        });
        reports.push(prepared.color.report);
    }
    Ok((surfaces, reports))
}

fn record_batch_store_profile(
    prepared: &mut CudaPreparedRgb8MctBatchStore,
    index: usize,
    store_stats: CudaExecutionStats,
    store_us: u128,
) -> (usize, usize) {
    let (store_dispatches, store_decode_dispatches, report_store_us) = if index == 0 {
        (
            store_stats.kernel_dispatches(),
            store_stats.decode_kernel_dispatches(),
            store_us,
        )
    } else {
        (0, 0, 0)
    };
    let dispatches = prepared.dispatches.saturating_add(store_dispatches);
    let decode_dispatches = prepared
        .decode_dispatches
        .saturating_add(store_decode_dispatches);
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
        .saturating_add(store_dispatches);
    profile::finalize_decode_total_us(&mut prepared.color.report);
    (dispatches, decode_dispatches)
}
