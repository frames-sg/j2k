// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{CudaContext, CudaHtj2kDecodeResources};

use super::super::{
    can_batch_color_idwt, cuda_error, decode_cuda_component_subbands_with_resources, host_owners,
    profile, run_color_component_idwt_batches, run_component_cleanup_dequant_batches,
    CudaBufferPool, CudaComponentDecodeWork, CudaHtj2kColorDecodePlans, CudaQueuedIdwtBatch,
    CudaSession, Error, HostPhaseBudget,
};

pub(super) struct EnqueuedColorCudaResidentBatch {
    pub(super) context: CudaContext,
    pub(super) pool: CudaBufferPool,
    pub(super) colors: Vec<CudaHtj2kColorDecodePlans>,
    pub(super) component_work: Vec<CudaComponentDecodeWork>,
    pub(super) pending_idwt_batch: Option<CudaQueuedIdwtBatch>,
    pub(super) idwt_batched: bool,
    pub(super) table_upload_us: u128,
    pub(super) payload_upload_us: u128,
}

pub(super) fn enqueue_color_cuda_resident_batch(
    session: &mut CudaSession,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    shared_payload: &[u8],
    collect_stage_timings: bool,
) -> Result<EnqueuedColorCudaResidentBatch, Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let (decode_resources, table_upload_us, payload_upload_us) = upload_color_batch_resources(
        session,
        &context,
        &pool,
        &colors,
        shared_payload,
        collect_stage_timings,
    )?;
    let (component_work, idwt_batched, pending_idwt_batch) = enqueue_color_components(
        &context,
        &pool,
        &decode_resources,
        &colors,
        collect_stage_timings,
    )?;
    Ok(EnqueuedColorCudaResidentBatch {
        context,
        pool,
        colors,
        component_work,
        pending_idwt_batch,
        idwt_batched,
        table_upload_us,
        payload_upload_us,
    })
}

fn upload_color_batch_resources(
    session: &mut CudaSession,
    context: &CudaContext,
    pool: &CudaBufferPool,
    colors: &[CudaHtj2kColorDecodePlans],
    shared_payload: &[u8],
    collect_stage_timings: bool,
) -> Result<(CudaHtj2kDecodeResources, u128, u128), Error> {
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
    let resources = match table_resources.as_ref() {
        Some(tables) => {
            context.upload_htj2k_decode_resources_with_tables_and_pool(shared_payload, tables, pool)
        }
        None => context.upload_j2k_decode_payload_with_pool(shared_payload, pool),
    }
    .map_err(cuda_error)?;
    Ok((
        resources,
        table_upload_us,
        profile::elapsed_us(payload_upload_start),
    ))
}

fn enqueue_color_components(
    context: &CudaContext,
    pool: &CudaBufferPool,
    decode_resources: &CudaHtj2kDecodeResources,
    colors: &Vec<CudaHtj2kColorDecodePlans>,
    collect_stage_timings: bool,
) -> Result<
    (
        Vec<CudaComponentDecodeWork>,
        bool,
        Option<CudaQueuedIdwtBatch>,
    ),
    Error,
> {
    let component_count = colors
        .iter()
        .map(|color| color.components.len())
        .sum::<usize>();
    let mut host_budget = HostPhaseBudget::new("j2k CUDA color batch execution graph");
    host_owners::account_colors(&mut host_budget, colors)?;
    let mut component_work = host_budget.try_vec_with_capacity(component_count)?;
    for color in colors {
        for plan in &color.components {
            component_work.push(decode_cuda_component_subbands_with_resources(
                context,
                plan,
                pool,
                collect_stage_timings,
                &mut host_budget,
            )?);
        }
    }
    run_component_cleanup_dequant_batches(
        context,
        decode_resources,
        &mut component_work,
        pool,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    let mut batch_components = host_budget.try_vec_with_capacity(component_count)?;
    for color in colors {
        batch_components.extend(color.components.iter());
    }
    let idwt_batched = can_batch_color_idwt(&batch_components);
    let pending_idwt_batch = idwt_batched
        .then(|| {
            run_color_component_idwt_batches(
                context,
                &batch_components,
                &mut component_work,
                pool,
                collect_stage_timings,
                host_budget.live_bytes(),
            )
        })
        .transpose()?
        .flatten();
    Ok((component_work, idwt_batched, pending_idwt_batch))
}
