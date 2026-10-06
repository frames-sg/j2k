// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    plan_ht_gpu_job_chunks_with_external_live_bytes, HtGpuJobChunkLimits, HtGpuJobPassBucket,
};
use j2k_cuda_j2k_engine::{
    CudaHtj2kDecodeResources, CudaQueuedHtj2kCleanup, CudaQueuedHtj2kCleanupGroup, J2kCudaEngine,
};

mod kernel;
mod materialize;
mod targets;

use self::{
    kernel::enqueue_chunk_kernel,
    materialize::{
        build_logical_payload, materialize_chunk_payload, select_chunk_jobs, LogicalPayload,
        MaterializedHtj2kChunk,
    },
    targets::build_chunk_targets,
};

use super::super::super::{
    cuda_error, CudaBufferPool, CudaComponentDecodeWork, CudaContext,
    CudaHtj2kDecodeTableResources, Error, CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
};
use super::super::cleanup_dequant::{
    retire_jobless_coefficient_clears, with_pending_coefficient_clears,
};
use super::planning::{chunk_requests, flatten_job_locations, Htj2kJobLocation};
use super::{ChunkedHtj2kCleanup, Htj2kChunkJobIdentity};
use crate::allocation::HostPhaseBudget;

struct SubmittedHtj2kChunk {
    cleanup: CudaQueuedHtj2kCleanup,
    resources: CudaHtj2kDecodeResources,
    identities: Vec<Htj2kChunkJobIdentity>,
}

fn chunk_plan_invariant_error() -> Error {
    Error::capability_rejected(j2k_core::CapabilityRejection::geometry_mismatch(
        CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
    ))
}

/// Flatten, pass-bucket, and asynchronously enqueue bounded HTJ2K arenas.
#[expect(
    clippy::too_many_arguments,
    reason = "the helper keeps explicit ownership of CUDA context, tables, group arena, outputs, and limits"
)]
pub(in crate::decoder) fn enqueue_chunked_htj2k_cleanup_dequant(
    context: &CudaContext,
    tables: Option<&CudaHtj2kDecodeTableResources>,
    payload_parts: &[&[u8]],
    component_work: &mut [CudaComponentDecodeWork],
    component_source_indices: &[usize],
    pool: &CudaBufferPool,
    limits: HtGpuJobChunkLimits,
    live_host_bytes: usize,
) -> Result<ChunkedHtj2kCleanup, Error> {
    with_pending_coefficient_clears(
        context,
        component_work,
        false,
        live_host_bytes,
        |component_work, clear_enqueued| {
            let mut retained_budget = HostPhaseBudget::with_live_bytes(
                "CUDA retained HTJ2K chunk planning",
                live_host_bytes,
            )?;
            let locations = flatten_job_locations(
                component_work,
                component_source_indices,
                &mut retained_budget,
            )?;
            if locations.is_empty() {
                retire_jobless_coefficient_clears(context, component_work, clear_enqueued)?;
                return Ok(ChunkedHtj2kCleanup {
                    group: None,
                    resources: Vec::new(),
                    identities: Vec::new(),
                    chunk_count: 0,
                    dequant_chunk_count: 0,
                });
            }
            let tables = tables.ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::contract_violation(
                    "CUDA HTJ2K chunks require resident cleanup lookup tables",
                ),
            ))?;
            let requests = chunk_requests(component_work, &locations, &mut retained_budget)?;
            let plan = plan_ht_gpu_job_chunks_with_external_live_bytes(
                &requests,
                limits,
                retained_budget.live_bytes(),
            )?;
            retained_budget.account_bytes(plan.retained_host_bytes())?;
            let logical_payload = build_logical_payload(payload_parts, &mut retained_budget)?;
            let (mut owner, mut accounted_group_host_bytes) = new_chunked_cleanup_owner(
                context,
                pool,
                locations.len(),
                plan.chunks().len(),
                &mut retained_budget,
            )?;

            for (chunk_index, chunk) in plan.chunks().iter().copied().enumerate() {
                let entries = plan
                    .chunk_entries(chunk_index)
                    .ok_or_else(chunk_plan_invariant_error)?;
                let submission = enqueue_one_chunk(
                    context,
                    tables,
                    &logical_payload,
                    component_work,
                    &locations,
                    entries,
                    chunk.bucket(),
                    chunk.payload_bytes(),
                    pool,
                    retained_budget.live_bytes(),
                    owner
                        .group
                        .as_ref()
                        .ok_or_else(chunk_plan_invariant_error)?,
                    owner.identities.len(),
                );
                let SubmittedHtj2kChunk {
                    cleanup,
                    resources,
                    identities,
                } = match submission {
                    Ok(submitted) => submitted,
                    Err(error) => return Err(owner.finish_after_error(error)),
                };
                if let Err(error) = retain_chunk_status(
                    owner.group.as_mut(),
                    cleanup,
                    &mut retained_budget,
                    &mut accounted_group_host_bytes,
                ) {
                    return Err(owner.finish_after_error(error));
                }
                owner.resources.push(resources);
                owner.identities.extend_from_slice(&identities);
                owner.chunk_count = owner.chunk_count.saturating_add(1);
                if chunk.bucket() != HtGpuJobPassBucket::CleanupOnly {
                    owner.dequant_chunk_count = owner.dequant_chunk_count.saturating_add(1);
                }
            }
            account_chunk_dispatches(component_work, &locations, &owner);
            for work in component_work {
                work.pending_dequant_bands.clear();
            }
            Ok(owner)
        },
    )
}

fn new_chunked_cleanup_owner(
    context: &CudaContext,
    pool: &CudaBufferPool,
    job_count: usize,
    chunk_count: usize,
    retained_budget: &mut HostPhaseBudget,
) -> Result<(ChunkedHtj2kCleanup, usize), Error> {
    let resources = retained_budget.try_vec_with_capacity(chunk_count)?;
    let identities = retained_budget.try_vec_with_capacity(job_count)?;
    let status_group = CudaQueuedHtj2kCleanupGroup::new(
        context,
        pool,
        job_count,
        chunk_count,
        retained_budget.live_bytes(),
    )
    .map_err(cuda_error)?;
    let accounted_group_host_bytes = status_group.retained_host_bytes();
    retained_budget.account_bytes(accounted_group_host_bytes)?;
    let owner = ChunkedHtj2kCleanup {
        group: Some(status_group),
        resources,
        identities,
        chunk_count: 0,
        dequant_chunk_count: 0,
    };
    Ok((owner, accounted_group_host_bytes))
}

fn retain_chunk_status(
    group: Option<&mut CudaQueuedHtj2kCleanupGroup>,
    cleanup: CudaQueuedHtj2kCleanup,
    retained_budget: &mut HostPhaseBudget,
    accounted_group_host_bytes: &mut usize,
) -> Result<(), Error> {
    let group = group.ok_or_else(chunk_plan_invariant_error)?;
    group.retain(cleanup).map_err(cuda_error)?;
    let retained_group_host_bytes = group.retained_host_bytes();
    let additional_group_host_bytes = retained_group_host_bytes
        .checked_sub(*accounted_group_host_bytes)
        .ok_or_else(chunk_plan_invariant_error)?;
    retained_budget.account_bytes(additional_group_host_bytes)?;
    *accounted_group_host_bytes = retained_group_host_bytes;
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "one chunk materializes an explicitly bounded arena against its retained output owners"
)]
fn enqueue_one_chunk(
    context: &CudaContext,
    tables: &CudaHtj2kDecodeTableResources,
    logical_payload: &LogicalPayload<'_>,
    component_work: &[CudaComponentDecodeWork],
    locations: &[Htj2kJobLocation],
    entries: &[j2k_core::HtGpuJobChunkEntry],
    bucket: HtGpuJobPassBucket,
    payload_bytes: usize,
    pool: &CudaBufferPool,
    live_host_bytes: usize,
    status_group: &CudaQueuedHtj2kCleanupGroup,
    status_offset: usize,
) -> Result<SubmittedHtj2kChunk, Error> {
    let mut budget = HostPhaseBudget::with_live_bytes(
        "CUDA bounded HTJ2K chunk materialization",
        live_host_bytes,
    )?;
    let selected = select_chunk_jobs(entries, locations, &mut budget)?;
    let MaterializedHtj2kChunk {
        payload_parts,
        jobs,
        identities,
    } = materialize_chunk_payload(
        logical_payload,
        component_work,
        &selected,
        payload_bytes,
        &mut budget,
    )?;
    let resources = J2kCudaEngine::new(context)
        .upload_htj2k_decode_resources_with_tables_and_pool(&payload_parts, tables, pool)
        .map_err(cuda_error)?;
    let targets = build_chunk_targets(component_work, &selected, &jobs, &mut budget)?;
    let cleanup = enqueue_chunk_kernel(
        context,
        &resources,
        &targets,
        bucket,
        pool,
        budget.live_bytes(),
        status_group,
        status_offset,
    )?;
    Ok(SubmittedHtj2kChunk {
        cleanup,
        resources,
        identities,
    })
}

fn account_chunk_dispatches(
    component_work: &mut [CudaComponentDecodeWork],
    locations: &[Htj2kJobLocation],
    owner: &ChunkedHtj2kCleanup,
) {
    let Some(first) = locations.first() else {
        return;
    };
    let dequant_dispatches = owner.dequant_chunk_count;
    let fused_dequant_dispatches = owner.chunk_count.saturating_sub(dequant_dispatches);
    let cleanup_dispatches = owner.chunk_count.saturating_add(fused_dequant_dispatches);
    let Some(accounting) = component_work.get_mut(first.work) else {
        return;
    };
    accounting.dispatches = accounting
        .dispatches
        .saturating_add(cleanup_dispatches)
        .saturating_add(dequant_dispatches);
    accounting.decode_dispatches = accounting
        .decode_dispatches
        .saturating_add(cleanup_dispatches);
    accounting.timings.ht_dispatch_count = accounting
        .timings
        .ht_dispatch_count
        .saturating_add(cleanup_dispatches);
    accounting.timings.ht_refinement_dispatch_count = accounting
        .timings
        .ht_refinement_dispatch_count
        .saturating_add(dequant_dispatches);
    accounting.timings.dequant_dispatch_count = accounting
        .timings
        .dequant_dispatch_count
        .saturating_add(dequant_dispatches);
    accounting.timings.fused_dequant_dispatch_count = accounting
        .timings
        .fused_dequant_dispatch_count
        .saturating_add(fused_dequant_dispatches);
}
