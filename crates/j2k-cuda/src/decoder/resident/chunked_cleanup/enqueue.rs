// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{plan_ht_gpu_job_chunks, HtGpuJobChunkLimits, HtGpuJobPassBucket};
use j2k_cuda_runtime::{
    CudaHtj2kCleanupTarget, CudaHtj2kDecodeResources, CudaQueuedHtj2kCleanup,
    CudaQueuedHtj2kCleanupGroup,
};

use super::super::super::{
    cuda_error, CudaBufferPool, CudaComponentDecodeWork, CudaContext,
    CudaHtj2kDecodeTableResources, Error, CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
};
use super::super::pooled_cuda_buffer;
use super::planning::{
    chunk_requests, flatten_job_locations, job_at, pending_at,
    sort_selected_jobs_for_coalesced_targets, Htj2kJobLocation, SelectedHtj2kChunkJob,
};
use super::{ChunkedHtj2kCleanup, Htj2kChunkJobIdentity};
use crate::allocation::HostPhaseBudget;

struct SubmittedHtj2kChunk {
    cleanup: CudaQueuedHtj2kCleanup,
    resources: CudaHtj2kDecodeResources,
    identities: Vec<Htj2kChunkJobIdentity>,
}

/// Flatten, pass-bucket, and asynchronously enqueue bounded HTJ2K arenas.
#[expect(
    clippy::too_many_arguments,
    reason = "the helper keeps explicit ownership of CUDA context, tables, group arena, outputs, and limits"
)]
pub(in crate::decoder) fn enqueue_chunked_htj2k_cleanup_dequant(
    context: &CudaContext,
    tables: Option<&CudaHtj2kDecodeTableResources>,
    shared_payload: &[u8],
    component_work: &mut [CudaComponentDecodeWork],
    component_source_indices: &[usize],
    pool: &CudaBufferPool,
    limits: HtGpuJobChunkLimits,
    live_host_bytes: usize,
) -> Result<ChunkedHtj2kCleanup, Error> {
    let locations = flatten_job_locations(component_work, component_source_indices)?;
    if locations.is_empty() {
        return Ok(ChunkedHtj2kCleanup {
            group: None,
            resources: Vec::new(),
            identities: Vec::new(),
            chunk_count: 0,
            dequant_chunk_count: 0,
        });
    }
    let tables = tables.ok_or(Error::UnsupportedCudaRequest {
        reason: "CUDA HTJ2K chunks require resident cleanup lookup tables",
    })?;
    let requests = chunk_requests(component_work, &locations)?;
    let plan = plan_ht_gpu_job_chunks(&requests, limits)?;
    let status_group =
        CudaQueuedHtj2kCleanupGroup::new(context, pool, locations.len()).map_err(cuda_error)?;
    let mut owner = ChunkedHtj2kCleanup {
        group: Some(status_group),
        resources: Vec::new(),
        identities: Vec::new(),
        chunk_count: 0,
        dequant_chunk_count: 0,
    };
    owner
        .resources
        .try_reserve_exact(plan.chunks().len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: plan
                .chunks()
                .len()
                .saturating_mul(core::mem::size_of::<CudaHtj2kDecodeResources>()),
            what: "CUDA retained HTJ2K chunks",
        })?;
    owner
        .identities
        .try_reserve_exact(locations.len())
        .map_err(|_| Error::HostAllocationFailed {
            bytes: locations
                .len()
                .saturating_mul(core::mem::size_of::<Htj2kChunkJobIdentity>()),
            what: "CUDA retained HTJ2K status identities",
        })?;

    for (chunk_index, chunk) in plan.chunks().iter().copied().enumerate() {
        let entries = plan
            .chunk_entries(chunk_index)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        let submission = enqueue_one_chunk(
            context,
            tables,
            shared_payload,
            component_work,
            &locations,
            entries,
            chunk.bucket(),
            chunk.payload_bytes(),
            pool,
            live_host_bytes,
            owner.group.as_ref().ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?,
            owner.identities.len(),
        );
        match submission {
            Ok(submitted) => {
                let SubmittedHtj2kChunk {
                    cleanup,
                    resources,
                    identities,
                } = submitted;
                if let Err(error) = owner
                    .group
                    .as_mut()
                    .ok_or(Error::UnsupportedCudaRequest {
                        reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                    })?
                    .retain(cleanup)
                    .map_err(cuda_error)
                {
                    return Err(owner.finish_after_error(error));
                }
                owner.resources.push(resources);
                owner.identities.extend_from_slice(&identities);
                owner.chunk_count = owner.chunk_count.saturating_add(1);
                if chunk.bucket() != HtGpuJobPassBucket::CleanupOnly {
                    owner.dequant_chunk_count = owner.dequant_chunk_count.saturating_add(1);
                }
            }
            Err(error) => return Err(owner.finish_after_error(error)),
        }
    }
    account_chunk_dispatches(component_work, &locations, &owner);
    for work in component_work {
        work.pending_dequant_bands.clear();
    }
    Ok(owner)
}

#[expect(
    clippy::too_many_arguments,
    reason = "one chunk materializes an explicitly bounded arena against its retained output owners"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one chunk keeps stable job identity, coalesced destination ownership, arena materialization, and queued completion atomic"
)]
fn enqueue_one_chunk(
    context: &CudaContext,
    tables: &CudaHtj2kDecodeTableResources,
    shared_payload: &[u8],
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
    let mut selected = budget.try_vec_with_capacity(entries.len())?;
    for entry in entries {
        let location =
            *locations
                .get(entry.original_job_index())
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                })?;
        if location.source != entry.source_index() {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            });
        }
        selected.push(SelectedHtj2kChunkJob {
            location,
            original_job_index: entry.original_job_index(),
            source_index: entry.source_index(),
        });
    }
    sort_selected_jobs_for_coalesced_targets(&mut selected);

    let mut payload = budget.try_vec_with_capacity(payload_bytes)?;
    let mut jobs = budget.try_vec_with_capacity(entries.len())?;
    let mut identities = budget.try_vec_with_capacity(entries.len())?;
    for selected in &selected {
        let location = selected.location;
        let mut job = *job_at(component_work, location)?;
        let start =
            usize::try_from(job.payload_offset).map_err(|_| Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        let end =
            start
                .checked_add(job.payload_len as usize)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                })?;
        let bytes = shared_payload
            .get(start..end)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        job.payload_offset =
            u64::try_from(payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        payload.extend_from_slice(bytes);
        jobs.push(job);
        identities.push(Htj2kChunkJobIdentity::new(
            selected.original_job_index,
            selected.source_index,
        ));
    }
    if payload.len() != payload_bytes {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        });
    }
    let resources = context
        .upload_htj2k_decode_resources_with_tables_and_pool(&payload, tables, pool)
        .map_err(cuda_error)?;
    let mut targets = budget.try_vec_with_capacity(entries.len())?;
    let mut start = 0usize;
    while start < selected.len() {
        let location = selected[start].location;
        let mut end = start + 1;
        while end < selected.len()
            && selected[end].location.work == location.work
            && selected[end].location.pending == location.pending
        {
            end += 1;
        }
        let pending = pending_at(component_work, location)?;
        targets.push(CudaHtj2kCleanupTarget {
            coefficients: pooled_cuda_buffer(
                &component_work[location.work].bands[pending.band_index].buffer,
            )?,
            jobs: &jobs[start..end],
            output_words: pending.output_words,
        });
        start = end;
    }
    // SAFETY: the returned guard retains uploaded descriptor owners; the
    // group owns the shared status allocation, `SubmittedHtj2kChunk` retains
    // payload/tables, and component_work owns every disjoint destination.
    let cleanup = unsafe {
        match bucket {
            HtGpuJobPassBucket::CleanupOnly => context
                .decode_htj2k_codeblocks_cleanup_dequantize_multi_enqueue_into_status_group(
                    &resources,
                    &targets,
                    pool,
                    budget.live_bytes(),
                    status_group,
                    status_offset,
                ),
            HtGpuJobPassBucket::SigProp | HtGpuJobPassBucket::MagRef => context
                .decode_htj2k_codeblocks_cleanup_multi_enqueue_into_status_group(
                    &resources,
                    &targets,
                    pool,
                    budget.live_bytes(),
                    status_group,
                    status_offset,
                ),
        }
    }
    .map_err(cuda_error)?;
    if bucket != HtGpuJobPassBucket::CleanupOnly {
        // SAFETY: the cleanup guard retains its device descriptors, and the
        // same default stream orders this dequantization before later IDWT.
        unsafe { context.j2k_dequantize_queued_htj2k_cleanup_enqueue(&cleanup) }
            .map_err(cuda_error)?;
    }
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
    let cleanup_dispatches = owner.chunk_count;
    let dequant_dispatches = owner.dequant_chunk_count;
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
    accounting.timings.dequant_dispatch_count = accounting
        .timings
        .dequant_dispatch_count
        .saturating_add(dequant_dispatches);
}
