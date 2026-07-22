// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-thread J2K batch worker state and disjoint-slot execution.

use j2k_core::{BatchInfrastructureError, PixelFormat};
use std::sync::Arc;

use super::admission::BatchAllocationBudget;
use super::allocation::GENERIC_WORKER_CLAIM_BYTES;
use super::direct::{DirectColorRegionCache, DirectDecodeAttemptError, DirectWorkerState};
use super::{
    decode_tile_into_in_context, decode_tile_region_into_in_context,
    decode_tile_region_scaled_into_in_context, decode_tile_scaled_into_in_context,
    J2kBatchResultSlot, TileDecodeJob, TileDecodeOutput, TileRegionDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::{CpuDecodeParallelism, J2kContext, J2kError, J2kScratchPool};

mod owned;

/// One stack-owned worker. Dynamic native/direct ownership is covered by the
/// authoritative per-worker claim in `allocation`; these exact owners are also
/// counted structurally in the batch metadata plan.
pub(crate) struct BatchWorker {
    ctx: J2kContext,
    pool: J2kScratchPool,
    direct: DirectWorkerState,
    native_workspace: j2k_native::DecoderWorkspace,
    prepared_plan_scratch: j2k_native::J2kDirectCpuScratch,
    prepared_entropy_workspace: j2k_native::J2kDirectCpuEntropyWorkspace,
    preparation_calls: u64,
    preparation_worker_reuses: u64,
    prepared_plan_decode_calls: u64,
    allocation_budget: Option<Arc<BatchAllocationBudget>>,
}

impl BatchWorker {
    pub(super) fn new(
        batch_size: usize,
        allocation_budget: Option<Arc<BatchAllocationBudget>>,
    ) -> Self {
        let mut ctx = J2kContext::new();
        ctx.set_cpu_decode_parallelism(inner_parallelism_for_batch(batch_size));
        Self {
            ctx,
            pool: J2kScratchPool::new(),
            direct: DirectWorkerState::default(),
            native_workspace: j2k_native::DecoderWorkspace::default(),
            prepared_plan_scratch: j2k_native::J2kDirectCpuScratch::new(),
            prepared_entropy_workspace: j2k_native::J2kDirectCpuEntropyWorkspace::default(),
            preparation_calls: 0,
            preparation_worker_reuses: 0,
            prepared_plan_decode_calls: 0,
            allocation_budget,
        }
    }

    pub(super) fn decode_tile_jobs(
        &mut self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        results: &mut [J2kBatchResultSlot],
        fmt: PixelFormat,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_disjoint_result_slots(jobs.len(), results.len())?;
        for (job, slot) in jobs.iter_mut().zip(results) {
            *slot = Some(heap_free_decode_result(decode_tile_into_in_context(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                TileDecodeOutput {
                    out: job.out,
                    stride: job.stride,
                    fmt,
                },
            )));
        }
        Ok(())
    }

    pub(super) fn decode_tile_region_jobs(
        &mut self,
        jobs: &mut [TileRegionDecodeJob<'_, '_>],
        results: &mut [J2kBatchResultSlot],
        fmt: PixelFormat,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_disjoint_result_slots(jobs.len(), results.len())?;
        for (job, slot) in jobs.iter_mut().zip(results) {
            *slot = Some(heap_free_decode_result(decode_tile_region_into_in_context(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                TileDecodeOutput {
                    out: job.out,
                    stride: job.stride,
                    fmt,
                },
                job.roi,
            )));
        }
        Ok(())
    }

    pub(super) fn decode_tile_scaled_jobs(
        &mut self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        results: &mut [J2kBatchResultSlot],
        fmt: PixelFormat,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_disjoint_result_slots(jobs.len(), results.len())?;
        for (job, slot) in jobs.iter_mut().zip(results) {
            *slot = Some(heap_free_decode_result(decode_tile_scaled_into_in_context(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                TileDecodeOutput {
                    out: job.out,
                    stride: job.stride,
                    fmt,
                },
                job.scale,
            )));
        }
        Ok(())
    }

    pub(super) fn decode_tile_region_scaled_jobs(
        &mut self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        results: &mut [J2kBatchResultSlot],
        fmt: PixelFormat,
        shared_direct_plan: Option<&DirectColorRegionCache>,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_disjoint_result_slots(jobs.len(), results.len())?;
        let allocation_budget = self
            .allocation_budget
            .as_ref()
            .ok_or(BatchInfrastructureError::MissingResult { index: jobs.len() })?;
        for (job, slot) in jobs.iter_mut().zip(results) {
            let outcome =
                match self
                    .direct
                    .try_decode(job, fmt, shared_direct_plan, allocation_budget)
                {
                    Ok(Some(outcome)) => Ok(outcome),
                    Ok(None) => {
                        // Do not retain a direct plan/scratch owner while entering
                        // the generic native path under the same worker claim.
                        self.direct.release();
                        let claim = allocation_budget.claim(GENERIC_WORKER_CLAIM_BYTES)?;
                        let outcome = decode_tile_region_scaled_into_in_context(
                            job.input,
                            &mut self.ctx,
                            &mut self.pool,
                            TileDecodeOutput {
                                out: job.out,
                                stride: job.stride,
                                fmt,
                            },
                            job.roi,
                            job.scale,
                        );
                        // Generic row scratch can retain allocator capacity. End
                        // that ownership before releasing the full admission lease.
                        self.pool = J2kScratchPool::new();
                        drop(claim);
                        outcome
                    }
                    Err(DirectDecodeAttemptError::Tile(error)) => Err(error),
                    Err(DirectDecodeAttemptError::Infrastructure(error)) => return Err(error),
                };
            *slot = Some(heap_free_decode_result(outcome));
        }
        Ok(())
    }
}

fn heap_free_decode_result(
    result: Result<super::BatchOutcome, J2kError>,
) -> Result<super::BatchOutcome, J2kError> {
    result.map_err(J2kError::into_heap_free_batch_decode_error)
}

const fn inner_parallelism_for_batch(batch_size: usize) -> CpuDecodeParallelism {
    if batch_size > 1 {
        CpuDecodeParallelism::Serial
    } else {
        CpuDecodeParallelism::Auto
    }
}

fn ensure_disjoint_result_slots(
    jobs: usize,
    results: usize,
) -> Result<(), BatchInfrastructureError> {
    if results < jobs {
        return Err(BatchInfrastructureError::MissingResult { index: results });
    }
    if results > jobs {
        return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
            index: jobs,
            job_count: jobs,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
