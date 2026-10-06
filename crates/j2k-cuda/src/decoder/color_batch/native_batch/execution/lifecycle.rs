// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_j2k_engine::{CudaHtj2kDecodeResources, CudaHtj2kDecodeTableResources, J2kCudaEngine};

use super::super::{
    build_color_component_work, cuda_error, enqueue_chunked_htj2k_cleanup_dequant,
    enqueue_component_classic_batches, host_owners, profile, run_color_component_idwt_batches,
    ChunkedHtj2kCleanup, CudaHtj2kColorDecodePlans, CudaQueuedIdwtBatch, CudaSession, Error,
};
use crate::allocation::HostPhaseBudget;
use crate::decoder::color_batch::CudaComponentDecodeWork;
use crate::decoder::resident::QueuedComponentClassicDecode;

pub(super) struct SubmittedNativeColorEntropy {
    pub(super) pending_cleanup: Option<ChunkedHtj2kCleanup>,
    pub(super) pending_classic: Option<QueuedComponentClassicDecode>,
    pub(super) classic_resources: Option<CudaHtj2kDecodeResources>,
    pub(super) payload_upload_us: u128,
}

pub(super) fn prepare_native_color_tables(
    session: &mut CudaSession,
    colors: &[CudaHtj2kColorDecodePlans],
) -> Result<(Option<CudaHtj2kDecodeTableResources>, u128), Error> {
    let table_upload_start = profile::profile_now(false);
    let tables = if colors.iter().all(|color| {
        color
            .components
            .iter()
            .all(|plan| plan.subbands().is_empty())
    }) {
        None
    } else {
        Some(session.htj2k_decode_table_resources()?)
    };
    Ok((tables, profile::elapsed_us(table_upload_start)))
}

pub(super) fn build_native_color_component_work(
    session: &mut CudaSession,
    colors: &Vec<CudaHtj2kColorDecodePlans>,
    source_indices: &[usize],
) -> Result<(HostPhaseBudget, Vec<CudaComponentDecodeWork>, Vec<usize>), Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let mut budget = HostPhaseBudget::new("j2k CUDA exact RGB execution graph");
    host_owners::account_colors(&mut budget, colors)?;
    let component_work = build_color_component_work(&context, &pool, colors, false, &mut budget)?;
    let mut component_sources = budget.try_vec_with_capacity(component_work.len())?;
    for (color, source_index) in colors.iter().zip(source_indices.iter().copied()) {
        component_sources.extend(std::iter::repeat_n(source_index, color.components.len()));
    }
    Ok((budget, component_work, component_sources))
}

pub(super) fn enqueue_native_color_entropy(
    session: &mut CudaSession,
    table_resources: Option<&CudaHtj2kDecodeTableResources>,
    shared_payload: &[u8],
    component_work: &mut [CudaComponentDecodeWork],
    component_source_indices: &[usize],
    live_host_bytes: usize,
) -> Result<SubmittedNativeColorEntropy, Error> {
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let payload_upload_start = profile::profile_now(false);
    let (pending_classic, classic_resources) = if component_work
        .iter()
        .any(|work| !work.pending_classic_bands.is_empty())
    {
        let resources = J2kCudaEngine::new(&context)
            .upload_j2k_decode_payload_with_pool(&[shared_payload], &pool)
            .map_err(cuda_error)?;
        let tables = session.classic_decode_table_resources()?;
        let pending = enqueue_component_classic_batches(
            &context,
            &resources,
            &tables,
            component_work,
            component_source_indices,
            &pool,
            live_host_bytes,
        )?;
        (pending, Some(resources))
    } else {
        (None, None)
    };
    let pending_cleanup = Some(enqueue_chunked_htj2k_cleanup_dequant(
        &context,
        table_resources,
        &[shared_payload],
        component_work,
        component_source_indices,
        &pool,
        session.htj2k_decode_chunk_limits(),
        live_host_bytes,
    )?);
    #[cfg(test)]
    session.record_htj2k_decode_chunk_count_for_test(
        pending_cleanup
            .as_ref()
            .map_or(0, ChunkedHtj2kCleanup::chunk_count),
    );
    Ok(SubmittedNativeColorEntropy {
        pending_cleanup,
        pending_classic,
        classic_resources,
        payload_upload_us: profile::elapsed_us(payload_upload_start),
    })
}

pub(super) fn enqueue_native_color_idwt(
    session: &mut CudaSession,
    colors: &[CudaHtj2kColorDecodePlans],
    component_work: &mut [CudaComponentDecodeWork],
    host_budget: &mut HostPhaseBudget,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let component_count = component_work.len();
    let mut component_plans = host_budget.try_vec_with_capacity(component_count)?;
    for color in colors {
        component_plans.extend(color.components.iter());
    }
    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    run_color_component_idwt_batches(
        &context,
        &component_plans,
        component_work,
        &pool,
        false,
        host_budget.live_bytes(),
        false,
    )
}
