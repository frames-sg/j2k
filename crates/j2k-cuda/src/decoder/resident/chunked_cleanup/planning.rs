// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::HtGpuJobChunkRequest;
use j2k_cuda_runtime::{htj2k_cleanup_multi_descriptor_bytes, CudaHtj2kCodeBlockJob};

use super::super::super::{CudaComponentDecodeWork, Error, CUDA_HTJ2K_PLAN_INVARIANT_FAILED};
use crate::allocation::HostPhaseBudget;

#[derive(Clone, Copy)]
pub(super) struct Htj2kJobLocation {
    pub(super) work: usize,
    pub(super) pending: usize,
    pub(super) job: usize,
    pub(super) source: usize,
}

#[derive(Clone, Copy)]
pub(super) struct SelectedHtj2kChunkJob {
    pub(super) location: Htj2kJobLocation,
    pub(super) original_job_index: usize,
    pub(super) source_index: usize,
}

pub(super) fn sort_selected_jobs_for_coalesced_targets(selected: &mut [SelectedHtj2kChunkJob]) {
    selected.sort_unstable_by_key(|selected| {
        (
            selected.location.work,
            selected.location.pending,
            selected.location.job,
        )
    });
}

pub(super) fn flatten_job_locations(
    component_work: &[CudaComponentDecodeWork],
    component_source_indices: &[usize],
) -> Result<Vec<Htj2kJobLocation>, Error> {
    if component_work.len() != component_source_indices.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        });
    }
    let job_count = component_work
        .iter()
        .flat_map(|work| work.pending_dequant_bands.iter())
        .try_fold(0usize, |count, pending| {
            count.checked_add(pending.jobs.len())
        })
        .ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "CUDA HTJ2K flattened job locations",
        })?;
    let mut budget = HostPhaseBudget::new("CUDA HTJ2K flattened job locations");
    let mut locations = budget.try_vec_with_capacity(job_count)?;
    for (work_index, (work, source_index)) in component_work
        .iter()
        .zip(component_source_indices.iter().copied())
        .enumerate()
    {
        for (pending_index, pending) in work.pending_dequant_bands.iter().enumerate() {
            for job_index in 0..pending.jobs.len() {
                locations.push(Htj2kJobLocation {
                    work: work_index,
                    pending: pending_index,
                    job: job_index,
                    source: source_index,
                });
            }
        }
    }
    Ok(locations)
}

pub(super) fn chunk_requests(
    component_work: &[CudaComponentDecodeWork],
    locations: &[Htj2kJobLocation],
) -> Result<Vec<HtGpuJobChunkRequest>, Error> {
    let mut budget = HostPhaseBudget::new("CUDA HTJ2K shared chunk requests");
    let mut requests = budget.try_vec_with_capacity(locations.len())?;
    let descriptor_bytes = htj2k_cleanup_multi_descriptor_bytes();
    for location in locations {
        let job = job_at(component_work, *location)?;
        requests.push(HtGpuJobChunkRequest::new(
            location.source,
            job.number_of_coding_passes,
            job.payload_len as usize,
            descriptor_bytes,
        ));
    }
    Ok(requests)
}

pub(super) fn pending_at(
    component_work: &[CudaComponentDecodeWork],
    location: Htj2kJobLocation,
) -> Result<&super::super::super::CudaPendingDequantBand, Error> {
    component_work
        .get(location.work)
        .and_then(|work| work.pending_dequant_bands.get(location.pending))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        })
}

pub(super) fn job_at(
    component_work: &[CudaComponentDecodeWork],
    location: Htj2kJobLocation,
) -> Result<&CudaHtj2kCodeBlockJob, Error> {
    pending_at(component_work, location)?
        .jobs
        .get(location.job)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        })
}
