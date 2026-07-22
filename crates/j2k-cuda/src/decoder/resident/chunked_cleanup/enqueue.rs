// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{plan_ht_gpu_job_chunks, HtGpuJobChunkLimits, HtGpuJobPassBucket};
use j2k_cuda_runtime::{
    CudaHtj2kDecodeResources, CudaQueuedHtj2kCleanup, CudaQueuedHtj2kCleanupGroup,
};

mod kernel;
mod materialize;
mod targets;

use self::{
    kernel::enqueue_chunk_kernel,
    materialize::{materialize_chunk_payload, select_chunk_jobs},
    targets::build_chunk_targets,
};

use super::super::super::{
    cuda_error, CudaBufferPool, CudaComponentDecodeWork, CudaContext,
    CudaHtj2kDecodeTableResources, Error, CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
};
use super::planning::{chunk_requests, flatten_job_locations, Htj2kJobLocation};
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
    let selected = select_chunk_jobs(entries, locations, &mut budget)?;
    let (payload, jobs, identities) = materialize_chunk_payload(
        shared_payload,
        component_work,
        &selected,
        payload_bytes,
        &mut budget,
    )?;
    if payload.len() != payload_bytes {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        });
    }
    let resources = context
        .upload_htj2k_decode_resources_with_tables_and_pool(&payload, tables, pool)
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
