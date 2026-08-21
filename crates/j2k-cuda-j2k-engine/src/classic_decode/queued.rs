// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::HostPhaseBudget;
use j2k_cuda_runtime::{CudaContext, CudaError, CudaExecutionStats, CudaQueuedExecution};

use super::{
    bytes::classic_statuses_as_bytes_mut, CudaClassicDecodeStageTimings, CudaClassicStatus,
};

const CLASSIC_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";

/// Enqueued classic Tier-1 work retained until one deferred status readback.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued classic decode must be finished or retained until Drop synchronizes it"]
pub struct CudaQueuedClassicDecode {
    pub(crate) context: CudaContext,
    pub(crate) queued: Option<CudaQueuedExecution>,
    pub(crate) status_index: usize,
    pub(crate) status_count: usize,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) timings: CudaClassicDecodeStageTimings,
    pub(crate) finish_host_live_bytes: usize,
}

impl CudaQueuedClassicDecode {
    pub(super) fn empty(context: &CudaContext) -> Self {
        Self {
            context: context.clone(),
            queued: None,
            status_index: 0,
            status_count: 0,
            execution: CudaExecutionStats::default(),
            timings: CudaClassicDecodeStageTimings::default(),
            finish_host_live_bytes: 0,
        }
    }

    /// Number of descriptor statuses downloaded by completion.
    #[doc(hidden)]
    pub const fn status_count(&self) -> usize {
        self.status_count
    }

    /// CUDA execution counters for the enqueued classic work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Finish the ordered graph with one status transfer and validate every job.
    pub fn finish(
        mut self,
    ) -> Result<(CudaExecutionStats, CudaClassicDecodeStageTimings), CudaError> {
        if self.status_count == 0 {
            if let Some(queued) = self.queued.take() {
                queued.finish()?;
            }
            return Ok((self.execution, self.timings));
        }
        let mut statuses = HostPhaseBudget::with_live_bytes(
            "CUDA queued classic Tier-1 status readback",
            self.finish_host_live_bytes,
        )?
        .try_vec_filled(self.status_count, CudaClassicStatus::default())?;
        let queued = self.queued.take().ok_or_else(|| CudaError::StatePoisoned {
            message: "queued classic execution disappeared before completion".to_string(),
        })?;
        let (resources, execution) = queued.finish_with_resources()?;
        let status_buffer =
            resources
                .get(self.status_index)
                .ok_or_else(|| CudaError::StatePoisoned {
                    message: "queued classic status buffer disappeared before readback".to_string(),
                })?;
        status_buffer.copy_to_host(classic_statuses_as_bytes_mut(&mut statuses))?;
        self.context.record_status_device_to_host_copy(
            self.status_count
                .saturating_mul(core::mem::size_of::<CudaClassicStatus>()),
        );
        if let Some((job_index, status)) = statuses
            .iter()
            .copied()
            .enumerate()
            .find(|(_, status)| status.code != 0)
        {
            return Err(CudaError::KernelJobStatus {
                kernel: CLASSIC_KERNEL_NAME,
                job_index,
                code: status.code,
                detail: status.detail,
            });
        }
        Ok((execution, self.timings))
    }
}
