// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob};

use super::super::super::super::{CudaComponentDecodeWork, Error};
use super::super::super::pooled_cuda_buffer;
use super::super::planning::{pending_at, SelectedHtj2kChunkJob};
use crate::allocation::HostPhaseBudget;

pub(super) fn build_chunk_targets<'a>(
    component_work: &'a [CudaComponentDecodeWork],
    selected: &[SelectedHtj2kChunkJob],
    jobs: &'a [CudaHtj2kCodeBlockJob],
    budget: &mut HostPhaseBudget,
) -> Result<Vec<CudaHtj2kCleanupTarget<'a>>, Error> {
    let mut targets = budget.try_vec_with_capacity(selected.len())?;
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
    Ok(targets)
}
