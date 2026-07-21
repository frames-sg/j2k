// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{htj2k_statuses_as_bytes_mut, htj2k_statuses_byte_len},
    context::CudaContext,
    error::{select_resource_release_error, CudaError},
    execution::CudaExecutionStats,
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};

use super::{
    queued::CudaQueuedHtj2kCleanup,
    status::{first_group_status_error, CudaHtj2kStatusSpan},
    CudaHtj2kStatus,
};

/// One group-owned status allocation shared by all bounded HTJ2K launches.
#[doc(hidden)]
#[derive(Debug)]
pub struct CudaQueuedHtj2kCleanupGroup {
    context: CudaContext,
    status_buffer: Option<CudaPooledDeviceBuffer>,
    status_count: usize,
    cleanups: Vec<CudaQueuedHtj2kCleanup>,
    finished: bool,
}

impl CudaQueuedHtj2kCleanupGroup {
    /// Number of descriptor statuses downloaded by group completion.
    #[doc(hidden)]
    pub const fn status_count(&self) -> usize {
        self.status_count
    }

    /// Allocate one status arena for every HTJ2K descriptor in a homogeneous group.
    #[doc(hidden)]
    pub fn new(
        context: &CudaContext,
        pool: &CudaBufferPool,
        status_count: usize,
    ) -> Result<Self, CudaError> {
        if !pool.is_owned_by(context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K status group pool must belong to the decode context".to_string(),
            });
        }
        let status_buffer = (status_count != 0)
            .then(|| pool.take(htj2k_statuses_byte_len(status_count)?))
            .transpose()?;
        Ok(Self {
            context: context.clone(),
            status_buffer,
            status_count,
            cleanups: Vec::new(),
            finished: false,
        })
    }

    pub(crate) fn status_destination(
        &self,
        offset: usize,
        count: usize,
    ) -> Result<(&CudaDeviceBuffer, usize), CudaError> {
        let end = offset
            .checked_add(count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if end > self.status_count {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.status_count,
            });
        }
        let buffer = self
            .status_buffer
            .as_ref()
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "non-empty HTJ2K launch requires a group status allocation".to_string(),
            })?;
        let byte_offset = offset
            .checked_mul(core::mem::size_of::<CudaHtj2kStatus>())
            .ok_or(CudaError::LengthTooLarge { len: offset })?;
        Ok((pooled_device_buffer(buffer)?, byte_offset))
    }

    /// Retain one launch guard whose statuses were written into this group's arena.
    #[doc(hidden)]
    pub fn retain(&mut self, cleanup: CudaQueuedHtj2kCleanup) -> Result<(), CudaError> {
        if !self.context.is_same_context(&cleanup.context) {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K cleanup and status group must belong to the same context"
                    .to_string(),
            });
        }
        if cleanup.status_count != 0 && !cleanup.uses_external_status_group {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K status group can retain only group-status cleanup launches"
                    .to_string(),
            });
        }
        let end = cleanup
            .status_offset
            .checked_add(cleanup.status_count)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if end > self.status_count {
            return Err(CudaError::OutputTooSmall {
                required: end,
                have: self.status_count,
            });
        }
        if let Some(previous) = self.cleanups.last() {
            let previous_end = previous
                .status_offset
                .checked_add(previous.status_count)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if cleanup.status_offset < previous_end {
                return Err(CudaError::InvalidArgument {
                    message: "HTJ2K group status spans must be retained in non-overlapping order"
                        .to_string(),
                });
            }
        }
        self.cleanups
            .try_reserve(1)
            .map_err(|_| CudaError::HostAllocationFailed {
                bytes: self
                    .cleanups
                    .len()
                    .saturating_add(1)
                    .saturating_mul(core::mem::size_of::<CudaQueuedHtj2kCleanup>()),
            })?;
        self.cleanups.push(cleanup);
        Ok(())
    }

    /// Download and validate all launch statuses with one group-level transfer.
    pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError> {
        let result = self.finish_inner();
        self.finished = true;
        result
    }

    fn finish_inner(&mut self) -> Result<CudaExecutionStats, CudaError> {
        let mut execution = CudaExecutionStats::default();
        let retained_live_bytes = self
            .cleanups
            .iter()
            .map(|cleanup| cleanup.finish_host_live_bytes)
            .max()
            .unwrap_or(0);
        let statuses_result = HostPhaseBudget::with_live_bytes(
            "CUDA grouped HTJ2K cleanup status readback",
            retained_live_bytes,
        )
        .and_then(|mut budget| {
            budget.try_vec_filled(self.status_count, CudaHtj2kStatus::default())
        });
        let mut statuses = match statuses_result {
            Ok(statuses) => statuses,
            Err(error) => return Err(self.synchronize_release_all(error)),
        };
        if let Some(status_buffer) = self.status_buffer.as_ref() {
            if let Err(error) =
                status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))
            {
                return Err(self.synchronize_release_all(error));
            }
            self.context.record_status_device_to_host_copy(
                self.status_count
                    .saturating_mul(core::mem::size_of::<CudaHtj2kStatus>()),
            );
        }

        let status_error = self.cleanups.iter().find_map(|cleanup| {
            first_group_status_error(
                &statuses,
                &[CudaHtj2kStatusSpan {
                    start: cleanup.status_offset,
                    count: cleanup.status_count,
                    kernel: cleanup.kernel_name,
                }],
            )
        });
        let release_result = self.release_all_after_stream_completion(&mut execution);
        match (status_error, release_result) {
            (Some(primary), Err(release)) => Err(select_resource_release_error(primary, release)),
            (Some(error), Ok(())) | (None, Err(error)) => Err(error),
            (None, Ok(())) => Ok(execution),
        }
    }

    fn release_all_after_stream_completion(
        &mut self,
        execution: &mut CudaExecutionStats,
    ) -> Result<(), CudaError> {
        let mut release_error = None;
        for cleanup in &mut self.cleanups {
            accumulate_execution(execution, cleanup.execution);
            if let Err(error) = cleanup.release_after_stream_completion() {
                release_error = Some(match release_error {
                    None => error,
                    Some(primary) => select_resource_release_error(primary, error),
                });
            }
        }
        self.cleanups.clear();
        release_error.map_or(Ok(()), Err)
    }

    fn synchronize_release_all(&mut self, primary: CudaError) -> CudaError {
        let mut error = primary;
        for cleanup in &mut self.cleanups {
            if let Err(release) = cleanup.synchronize_and_release() {
                error = select_resource_release_error(error, release);
            }
        }
        self.cleanups.clear();
        error
    }
}

impl Drop for CudaQueuedHtj2kCleanupGroup {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish_inner();
        }
    }
}

fn accumulate_execution(total: &mut CudaExecutionStats, next: CudaExecutionStats) {
    total.kernel_dispatches = total
        .kernel_dispatches
        .saturating_add(next.kernel_dispatches);
    total.copy_kernel_dispatches = total
        .copy_kernel_dispatches
        .saturating_add(next.copy_kernel_dispatches);
    total.decode_kernel_dispatches = total
        .decode_kernel_dispatches
        .saturating_add(next.decode_kernel_dispatches);
    total.hardware_decode |= next.hardware_decode;
}
