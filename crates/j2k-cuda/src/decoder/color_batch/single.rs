// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    cuda_error, decode_cuda_component_subbands_with_resources,
    finish_color_cuda_resident_surface_with_component_work, profile,
    run_component_cleanup_dequant_batches, CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport,
    CudaSession, Error, FinishColorCudaResidentSurfaceRequest, HostPhaseBudget, J2kDecoder,
    PixelFormat, Rect, Surface, CUDA_HTJ2K_KERNELS_NOT_READY,
};

pub(in crate::decoder) fn decode_color_cuda_resident_surface_with_profile(
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
pub(in crate::decoder) fn decode_color_cuda_resident_scaled_surface(
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
pub(in crate::decoder) fn decode_color_cuda_resident_region_surface(
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
pub(in crate::decoder) fn decode_color_cuda_resident_region_scaled_surface(
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
fn decode_color_cuda_resident_surface_with_plans_profile(
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
