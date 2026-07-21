// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{plan::GrayBatchPlan, CudaQueuedJ2kStoreBatch};
use crate::{
    bytes::{
        store_gray16_batch_jobs_as_bytes, store_gray8_batch_jobs_as_bytes,
        store_grayi16_batch_jobs_as_bytes,
    },
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    j2k_decode::types::{
        CudaJ2kStoreGray16BatchJob, CudaJ2kStoreGray16Target, CudaJ2kStoreGray8BatchJob,
        CudaJ2kStoreGray8Target, CudaJ2kStoreGrayI16BatchJob, CudaJ2kStoreGrayI16Target,
    },
};

impl CudaContext {
    pub(super) fn enqueue_gray8_batch(
        &self,
        targets: &[CudaJ2kStoreGray8Target<'_>],
        plan: &GrayBatchPlan,
        output_base: u64,
    ) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
        if plan.active_count == 0 {
            return Ok(CudaQueuedJ2kStoreBatch {
                completion: None,
                jobs: None,
                execution: CudaExecutionStats::default(),
            });
        }
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(plan.active_count)
            .map_err(|_| CudaError::HostAllocationFailed {
                bytes: plan
                    .active_count
                    .saturating_mul(std::mem::size_of::<CudaJ2kStoreGray8BatchJob>()),
            })?;
        for (target, item) in targets.iter().zip(&plan.items) {
            if !item.active {
                continue;
            }
            let range = plan.ranges[item.range_index];
            let offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            jobs.push(CudaJ2kStoreGray8BatchJob {
                input_ptr: target.input.device_ptr(),
                output_ptr: output_base
                    .checked_add(offset)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?,
                job: target.job,
                reserved_tail: 0,
            });
        }
        let jobs_buffer = self.upload(store_gray8_batch_jobs_as_bytes(&jobs))?;
        // SAFETY: the returned guard owns `jobs_buffer` until the recorded
        // default-stream completion event has finished.
        if let Err(error) = unsafe {
            self.launch_j2k_store_gray8_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_count,
            )
        } {
            return self.synchronize_then_error(error);
        }
        let completion = self.create_event().and_then(|completion| {
            completion.record_default_stream()?;
            Ok(completion)
        });
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => return self.synchronize_then_error(error),
        };
        Ok(CudaQueuedJ2kStoreBatch {
            completion: Some(completion),
            jobs: Some(jobs_buffer),
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    pub(super) fn enqueue_gray16_batch(
        &self,
        targets: &[CudaJ2kStoreGray16Target<'_>],
        plan: &GrayBatchPlan,
        output_base: u64,
    ) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
        if plan.active_count == 0 {
            return Ok(CudaQueuedJ2kStoreBatch {
                completion: None,
                jobs: None,
                execution: CudaExecutionStats::default(),
            });
        }
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(plan.active_count)
            .map_err(|_| CudaError::HostAllocationFailed {
                bytes: plan
                    .active_count
                    .saturating_mul(std::mem::size_of::<CudaJ2kStoreGray16BatchJob>()),
            })?;
        for (target, item) in targets.iter().zip(&plan.items) {
            if !item.active {
                continue;
            }
            let range = plan.ranges[item.range_index];
            let offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            jobs.push(CudaJ2kStoreGray16BatchJob {
                input_ptr: target.input.device_ptr(),
                output_ptr: output_base
                    .checked_add(offset)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?,
                job: target.job,
                reserved_tail: 0,
            });
        }
        let jobs_buffer = self.upload(store_gray16_batch_jobs_as_bytes(&jobs))?;
        // SAFETY: the returned guard owns `jobs_buffer` through completion.
        if let Err(error) = unsafe {
            self.launch_j2k_store_gray16_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_count,
            )
        } {
            return self.synchronize_then_error(error);
        }
        let completion = self.create_event().and_then(|completion| {
            completion.record_default_stream()?;
            Ok(completion)
        });
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => return self.synchronize_then_error(error),
        };
        Ok(CudaQueuedJ2kStoreBatch {
            completion: Some(completion),
            jobs: Some(jobs_buffer),
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }

    pub(super) fn enqueue_grayi16_batch(
        &self,
        targets: &[CudaJ2kStoreGrayI16Target<'_>],
        plan: &GrayBatchPlan,
        output_base: u64,
    ) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
        if plan.active_count == 0 {
            return Ok(CudaQueuedJ2kStoreBatch {
                completion: None,
                jobs: None,
                execution: CudaExecutionStats::default(),
            });
        }
        let mut jobs = Vec::new();
        jobs.try_reserve_exact(plan.active_count)
            .map_err(|_| CudaError::HostAllocationFailed {
                bytes: plan
                    .active_count
                    .saturating_mul(std::mem::size_of::<CudaJ2kStoreGrayI16BatchJob>()),
            })?;
        for (target, item) in targets.iter().zip(&plan.items) {
            if !item.active {
                continue;
            }
            let range = plan.ranges[item.range_index];
            let offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            jobs.push(CudaJ2kStoreGrayI16BatchJob {
                input_ptr: target.input.device_ptr(),
                output_ptr: output_base
                    .checked_add(offset)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?,
                job: target.job,
                reserved_tail: 0,
            });
        }
        let jobs_buffer = self.upload(store_grayi16_batch_jobs_as_bytes(&jobs))?;
        // SAFETY: the returned guard owns `jobs_buffer` through completion.
        if let Err(error) = unsafe {
            self.launch_j2k_store_grayi16_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_count,
            )
        } {
            return self.synchronize_then_error(error);
        }
        let completion = self.create_event().and_then(|completion| {
            completion.record_default_stream()?;
            Ok(completion)
        });
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => return self.synchronize_then_error(error),
        };
        Ok(CudaQueuedJ2kStoreBatch {
            completion: Some(completion),
            jobs: Some(jobs_buffer),
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }
}
