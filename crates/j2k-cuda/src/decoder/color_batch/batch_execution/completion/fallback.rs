// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::CudaContext;

use super::super::super::{
    finish_color_cuda_resident_surface_with_component_work, host_owners, take_component_work,
    CudaBufferPool, CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, CudaHtj2kProfileReport,
    Error, FinishColorCudaResidentSurfaceRequest, HostPhaseBudget, PixelFormat, Surface,
};

pub(super) fn finish_color_cuda_resident_batch_surfaces_individually(
    context: &CudaContext,
    pool: &CudaBufferPool,
    fmt: PixelFormat,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    component_work: Vec<CudaComponentDecodeWork>,
    collect_stage_timings: bool,
    idwt_batched: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut output_budget = HostPhaseBudget::new("j2k CUDA color batch output graph");
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
                run_idwt: !idwt_batched,
                emit_report: false,
            },
        )?;
        surfaces.push(surface);
        reports.push(report);
    }
    Ok((surfaces, reports))
}
