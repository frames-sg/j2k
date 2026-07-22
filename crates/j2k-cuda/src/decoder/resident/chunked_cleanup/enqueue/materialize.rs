// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::HtGpuJobChunkEntry;
use j2k_cuda_runtime::CudaHtj2kCodeBlockJob;

use super::super::super::super::{
    CudaComponentDecodeWork, Error, CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
};
use super::super::planning::{
    job_at, sort_selected_jobs_for_coalesced_targets, Htj2kJobLocation, SelectedHtj2kChunkJob,
};
use super::super::Htj2kChunkJobIdentity;
use crate::allocation::HostPhaseBudget;

type MaterializedHtj2kChunk = (
    Vec<u8>,
    Vec<CudaHtj2kCodeBlockJob>,
    Vec<Htj2kChunkJobIdentity>,
);

pub(super) fn select_chunk_jobs(
    entries: &[HtGpuJobChunkEntry],
    locations: &[Htj2kJobLocation],
    budget: &mut HostPhaseBudget,
) -> Result<Vec<SelectedHtj2kChunkJob>, Error> {
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
    Ok(selected)
}

pub(super) fn materialize_chunk_payload(
    shared_payload: &[u8],
    component_work: &[CudaComponentDecodeWork],
    selected: &[SelectedHtj2kChunkJob],
    payload_bytes: usize,
    budget: &mut HostPhaseBudget,
) -> Result<MaterializedHtj2kChunk, Error> {
    let mut payload = budget.try_vec_with_capacity(payload_bytes)?;
    let mut jobs = budget.try_vec_with_capacity(selected.len())?;
    let mut identities = budget.try_vec_with_capacity(selected.len())?;
    for selected in selected {
        let mut job = *job_at(component_work, selected.location)?;
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
    Ok((payload, jobs, identities))
}
