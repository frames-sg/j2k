// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    can_batch_color_idwt, finish_cuda_component_decode, run_color_component_idwt_batches,
    run_cuda_component_idwt_steps, validate_color_stores, CudaBufferPool, CudaComponentDecodeWork,
    CudaDecodedComponent, CudaHtj2kColorDecodePlans, CudaQueuedIdwtBatch, Error, HostPhaseBudget,
    CUDA_HTJ2K_KERNELS_NOT_READY,
};

pub(super) struct PreparedColorComponents {
    pub(super) components: [CudaDecodedComponent; 3],
    pub(super) dispatches: usize,
    pub(super) decode_dispatches: usize,
}

pub(super) fn run_pending_color_idwt(
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

pub(super) fn finish_color_components(
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
