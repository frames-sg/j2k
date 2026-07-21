// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::store_rgb8_mct_batch_jobs_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

use super::{ensure_internal_count, validate_rgb8_mct_targets, Rgb8MctBatchPlan};
use crate::j2k_decode::{
    store::grayscale_batch::CudaQueuedJ2kStoreBatch,
    types::{CudaJ2kStoreRgb8MctBatchJob, CudaJ2kStoreRgb8MctTarget},
};

fn materialize_external_rgb_batch(
    targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
    plan: &Rgb8MctBatchPlan,
    base_ptr: u64,
) -> Result<(Vec<CudaDeviceBufferRange>, Vec<CudaJ2kStoreRgb8MctBatchJob>), CudaError> {
    let mut budget = HostPhaseBudget::new("CUDA external J2K RGB store batch metadata");
    budget.account_vec(&plan.targets)?;
    let mut ranges = budget.try_vec_with_capacity(plan.targets.len())?;
    let mut offset = 0usize;
    for target_plan in &plan.targets {
        ranges.push(CudaDeviceBufferRange {
            offset,
            len: target_plan.output_bytes,
        });
        offset = offset
            .checked_add(target_plan.output_bytes)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    }
    ensure_internal_count(
        offset,
        plan.total_bytes,
        "external J2K RGB store output range mismatch",
    )?;

    let mut kernel_jobs = budget.try_vec_with_capacity(plan.active_job_count)?;
    for ((target, range), target_plan) in targets.iter().zip(&ranges).zip(&plan.targets) {
        if !target_plan.active {
            continue;
        }
        let range_offset = u64::try_from(range.offset)
            .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
        let output_ptr = base_ptr
            .checked_add(range_offset)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        kernel_jobs.push(CudaJ2kStoreRgb8MctBatchJob {
            plane0_ptr: target.plane0.device_ptr(),
            plane1_ptr: target.plane1.device_ptr(),
            plane2_ptr: target.plane2.device_ptr(),
            output_ptr,
            job: target.job,
            reserved_tail: 0,
        });
    }
    ensure_internal_count(
        kernel_jobs.len(),
        plan.active_job_count,
        "external J2K RGB store active-job count mismatch",
    )?;
    Ok((ranges, kernel_jobs))
}

impl CudaContext {
    /// Enqueue inverse RCT/ICT and native RGB8/RGBA8 stores directly into a
    /// validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// Every component plane and the destination allocation must remain live
    /// and unavailable for mutation or reuse until the returned guard finishes
    /// or drops after confirmed CUDA completion. If completion cannot be
    /// proven, the caller must quarantine those allocations. Decoded pixels
    /// must not be exposed as valid until codec status validation succeeds.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgb8_mct_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgb8MctTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_rgb8_mct_targets(self, targets, 0)?;
        if !self.is_same_context(destination.context()) {
            return Err(CudaError::InvalidArgument {
                message: "external RGB destination belongs to a different CUDA context".to_string(),
            });
        }
        if destination.byte_len() < plan.total_bytes {
            return Err(CudaError::OutputTooSmall {
                required: plan.total_bytes,
                have: destination.byte_len(),
            });
        }
        if plan.requires_zero_fill {
            return Err(CudaError::InvalidArgument {
                message: "external RGB batch destination requires full output coverage".to_string(),
            });
        }

        let (ranges, kernel_jobs) =
            materialize_external_rgb_batch(targets, &plan, destination.device_ptr())?;
        if plan.active_job_count == 0 {
            return Ok((
                ranges,
                CudaQueuedJ2kStoreBatch {
                    completion: None,
                    jobs: None,
                    execution: CudaExecutionStats::default(),
                },
            ));
        }

        let jobs_buffer = self.upload(store_rgb8_mct_batch_jobs_as_bytes(&kernel_jobs))?;
        // SAFETY: the returned completion guard retains `jobs_buffer`; the
        // caller's unsafe contract retains every plane and destination range.
        if let Err(error) = unsafe {
            self.launch_j2k_store_rgb8_mct_batch_enqueue(
                &jobs_buffer,
                plan.max_pixels,
                plan.active_job_count,
            )
        } {
            return self.synchronize_then_error(error);
        }
        let completion = self.create_event().and_then(|event| {
            event.record_default_stream()?;
            Ok(event)
        });
        let completion = match completion {
            Ok(completion) => completion,
            Err(error) => return self.synchronize_then_error(error),
        };
        Ok((
            ranges,
            CudaQueuedJ2kStoreBatch {
                completion: Some(completion),
                jobs: Some(jobs_buffer),
                execution: CudaExecutionStats {
                    kernel_dispatches: 1,
                    copy_kernel_dispatches: 0,
                    decode_kernel_dispatches: 1,
                    hardware_decode: false,
                },
            },
        ))
    }
}
