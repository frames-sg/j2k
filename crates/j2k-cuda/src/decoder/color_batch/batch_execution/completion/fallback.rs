// SPDX-License-Identifier: MIT OR Apache-2.0

use super::FinishColorBatchRequest;

use super::super::super::{
    finish_color_cuda_resident_surface_with_component_work, host_owners, take_component_work,
    CudaHtj2kProfileReport, Error, FinishColorCudaResidentSurfaceRequest, HostPhaseBudget, Surface,
};

/// Finishes each image's MCT and store separately; IDWT already ran batched.
pub(super) fn finish_color_cuda_resident_batch_surfaces_individually(
    request: FinishColorBatchRequest<'_>,
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
    let mut output_budget = HostPhaseBudget::with_live_bytes(
        "j2k CUDA color batch output graph",
        external_live_host_bytes,
    )?;
    host_owners::account_colors(&mut output_budget, &colors)?;
    host_owners::account_component_work(&mut output_budget, &component_work)?;
    let mut surfaces = output_budget.try_vec_with_capacity(colors.len())?;
    let mut reports = output_budget.try_vec_with_capacity(colors.len())?;
    let mut work_iter = component_work.into_iter();
    for color in colors {
        let component_count = color.components.len();
        let component_work =
            take_component_work(&mut work_iter, component_count, &mut output_budget)?;
        let (surface, report) = finish_color_cuda_resident_surface_with_component_work(
            FinishColorCudaResidentSurfaceRequest {
                context,
                pool,
                fmt,
                color,
                component_work,
                wall_started: None,
                collect_stage_timings,
                run_idwt: false,
                emit_report: false,
                preaccounted_host_bytes: Some(output_budget.live_bytes()),
            },
        )?;
        surfaces.push(surface);
        reports.push(report);
    }
    Ok((surfaces, reports))
}
