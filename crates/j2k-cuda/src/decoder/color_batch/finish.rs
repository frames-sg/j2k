// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    cuda_error, dispatch_color_store, finalize_color_surface, finish_color_components, host_owners,
    pooled_cuda_buffer, run_color_mct, run_pending_color_idwt, ColorStoreInputs,
    CudaHtj2kProfileReport, CudaQueuedIdwtBatch, Error, FinalizeColorSurfaceRequest,
    FinishColorCudaResidentSurfaceRequest, Surface,
};

pub(super) fn finish_color_cuda_resident_surface_with_component_work(
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
