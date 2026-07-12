// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    cuda_error, profile, BackendKind, CudaError, CudaHtj2kDecodePlan, CudaHtj2kProfileReport,
    CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaSession, CudaSurfaceStats, Error, PixelFormat,
    Storage, Surface, SurfaceResidency, CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
};
use super::component::decode_cuda_component_plan;
use super::helpers::pooled_cuda_buffer;

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_grayscale_cuda_resident_surface_with_plan_profile(
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
