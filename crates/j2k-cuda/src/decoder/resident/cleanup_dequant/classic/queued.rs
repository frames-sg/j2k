// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::super::{
    cuda_error, CudaBufferPool, CudaClassicDecodeTarget, CudaComponentDecodeWork,
    CudaHtj2kDecodeResources, Error,
};
use super::super::super::buffer_access::pooled_cuda_buffer;
use crate::allocation::HostPhaseBudget;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ClassicJobIdentity {
    source_index: usize,
    original_job_index: usize,
}

pub(in crate::decoder) struct QueuedComponentClassicDecode {
    queued: Option<j2k_cuda_runtime::CudaQueuedClassicDecode>,
    identities: Vec<ClassicJobIdentity>,
}

impl QueuedComponentClassicDecode {
    pub(in crate::decoder) fn has_status_readback(&self) -> bool {
        self.queued
            .as_ref()
            .is_some_and(|queued| queued.status_count() != 0)
    }

    pub(in crate::decoder) fn finish(mut self) -> Result<(), Error> {
        let Some(queued) = self.queued.take() else {
            return Ok(());
        };
        queued
            .finish()
            .map(|_| ())
            .map_err(|error| map_classic_status_error(error, &self.identities))
    }
}

impl Drop for QueuedComponentClassicDecode {
    fn drop(&mut self) {
        if let Some(queued) = self.queued.take() {
            let _ = queued.finish();
        }
    }
}

pub(in crate::decoder) fn enqueue_component_classic_batches(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    table_resources: &j2k_cuda_runtime::CudaClassicDecodeTableResources,
    component_work: &mut [CudaComponentDecodeWork],
    component_source_indices: &[usize],
    pool: &CudaBufferPool,
    live_host_bytes: usize,
) -> Result<Option<QueuedComponentClassicDecode>, Error> {
    if component_work.len() != component_source_indices.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: "CUDA classic source identity count does not match component work",
        });
    }
    let pending_count = component_work
        .iter()
        .map(|work| work.pending_classic_bands.len())
        .sum::<usize>();
    if pending_count == 0 {
        return Ok(None);
    }
    let job_count = component_work
        .iter()
        .flat_map(|work| &work.pending_classic_bands)
        .try_fold(0usize, |count, pending| {
            count.checked_add(pending.jobs.len())
        })
        .ok_or(Error::HostAllocationFailed {
            bytes: usize::MAX,
            what: "CUDA classic job source identities",
        })?;
    let mut budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA queued classic targets", live_host_bytes)?;
    let mut targets = budget.try_vec_with_capacity(pending_count)?;
    let mut identities = budget.try_vec_with_capacity(job_count)?;
    for (work, source_index) in component_work
        .iter()
        .zip(component_source_indices.iter().copied())
    {
        for pending in &work.pending_classic_bands {
            targets.push(CudaClassicDecodeTarget {
                coefficients: pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?,
                jobs: &pending.jobs,
                segments: &pending.segments,
                output_words: pending.output_words,
            });
            for _ in &pending.jobs {
                identities.push(ClassicJobIdentity {
                    source_index,
                    original_job_index: identities.len(),
                });
            }
        }
    }
    // SAFETY: component work retains disjoint coefficient targets; the
    // high-level pending owner retains payload/tables through this guard.
    let queued = unsafe {
        context.decode_classic_codeblocks_multi_enqueue_with_resources_and_pool(
            decode_resources,
            table_resources,
            &targets,
            pool,
            budget.live_bytes(),
        )
    }
    .map_err(cuda_error)?;
    let stats = queued.execution();
    if let Some(accounting) = component_work
        .iter_mut()
        .find(|work| !work.pending_classic_bands.is_empty())
    {
        accounting.timings.classic_dispatch_count = accounting
            .timings
            .classic_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
    }
    for work in component_work {
        work.pending_classic_bands.clear();
    }
    Ok(Some(QueuedComponentClassicDecode {
        queued: Some(queued),
        identities,
    }))
}

fn map_classic_status_error(
    error: j2k_cuda_runtime::CudaError,
    identities: &[ClassicJobIdentity],
) -> Error {
    let Some(job_index) = error.kernel_job_index() else {
        return cuda_error(error);
    };
    let Some(identity) = identities.get(job_index) else {
        return cuda_error(error);
    };
    Error::CudaTier1JobFailed {
        source_index: identity.source_index,
        original_job_index: identity.original_job_index,
        source: error,
    }
}

#[cfg(test)]
mod tests;
