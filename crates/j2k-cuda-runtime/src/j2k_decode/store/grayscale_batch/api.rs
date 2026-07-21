// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    plan::{
        external_destination_base, gray16_geometry, gray8_geometry, grayi16_geometry,
        validate_gray_targets,
    },
    CudaQueuedJ2kStoreBatch,
};
use crate::{
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelContiguousBatchOutput},
    j2k_decode::{
        store::destination::zero_unwritten_store_output,
        types::{CudaJ2kStoreGray16Target, CudaJ2kStoreGray8Target, CudaJ2kStoreGrayI16Target},
    },
    memory::{CudaDeviceBuffer, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

fn attach_zero_fill_completion(
    context: &CudaContext,
    mut queued: CudaQueuedJ2kStoreBatch,
    zero_fill_enqueued: bool,
) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
    if zero_fill_enqueued && queued.completion.is_none() {
        let completion = context.create_event().and_then(|completion| {
            completion.record_default_stream()?;
            Ok(completion)
        })?;
        queued.completion = Some(completion);
    }
    Ok(queued)
}

fn retire_failed_zero_fill(
    context: &CudaContext,
    error: CudaError,
    zero_fill_enqueued: bool,
) -> CudaError {
    if !zero_fill_enqueued {
        return error;
    }
    match context.synchronize_then_error::<()>(error) {
        Err(error) => error,
        Ok(()) => unreachable!("synchronize_then_error always returns the primary error"),
    }
}

fn finish_owned_store(
    output: CudaDeviceBuffer,
    queued: CudaQueuedJ2kStoreBatch,
) -> Result<(CudaDeviceBuffer, CudaExecutionStats), CudaError> {
    match queued.finish() {
        Ok(execution) => Ok((output, execution)),
        Err(error) => {
            if error.completion_is_uncertain() {
                core::mem::forget(output);
            }
            Err(error)
        }
    }
}

impl CudaContext {
    /// Store a Gray8 batch into one J2K-owned contiguous device allocation.
    #[doc(hidden)]
    pub fn j2k_store_gray8_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreGray8Target<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let (output, ranges, queued) =
            self.j2k_store_gray8_batch_contiguous_device_enqueue(targets)?;
        let (output, execution) = finish_owned_store(output, queued)?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution,
        })
    }

    /// Enqueue a Gray8 batch into one J2K-owned contiguous allocation.
    #[doc(hidden)]
    pub fn j2k_store_gray8_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGray8Target<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        let plan = validate_gray_targets(
            self,
            targets,
            1,
            |target| target.input,
            |target| target.output_index,
            gray8_geometry,
            0,
        )?;
        let output = self.allocate(plan.total_bytes)?;
        let zero_fill_enqueued = plan.requires_zero_fill
            && zero_unwritten_store_output(self, &output, plan.total_bytes, false)?;
        let queued = self
            .enqueue_gray8_batch(targets, &plan, output.device_ptr())
            .and_then(|queued| attach_zero_fill_completion(self, queued, zero_fill_enqueued))
            .map_err(|error| retire_failed_zero_fill(self, error, zero_fill_enqueued))?;
        Ok((output, plan.ranges, queued))
    }

    /// Store a Gray16 batch into one J2K-owned contiguous device allocation.
    #[doc(hidden)]
    pub fn j2k_store_gray16_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreGray16Target<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let (output, ranges, queued) =
            self.j2k_store_gray16_batch_contiguous_device_enqueue(targets)?;
        let (output, execution) = finish_owned_store(output, queued)?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution,
        })
    }

    /// Enqueue a Gray16 batch into one J2K-owned contiguous allocation.
    #[doc(hidden)]
    pub fn j2k_store_gray16_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGray16Target<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        let plan = validate_gray_targets(
            self,
            targets,
            std::mem::size_of::<u16>(),
            |target| target.input,
            |target| target.output_index,
            gray16_geometry,
            0,
        )?;
        let output = self.allocate(plan.total_bytes)?;
        let zero_fill_enqueued = plan.requires_zero_fill
            && zero_unwritten_store_output(self, &output, plan.total_bytes, false)?;
        let queued = self
            .enqueue_gray16_batch(targets, &plan, output.device_ptr())
            .and_then(|queued| attach_zero_fill_completion(self, queued, zero_fill_enqueued))
            .map_err(|error| retire_failed_zero_fill(self, error, zero_fill_enqueued))?;
        Ok((output, plan.ranges, queued))
    }

    /// Store a signed GrayI16 batch into one J2K-owned contiguous device allocation.
    #[doc(hidden)]
    pub fn j2k_store_grayi16_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreGrayI16Target<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let (output, ranges, queued) =
            self.j2k_store_grayi16_batch_contiguous_device_enqueue(targets)?;
        let (output, execution) = finish_owned_store(output, queued)?;
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution,
        })
    }

    /// Enqueue a signed GrayI16 batch into one J2K-owned contiguous allocation.
    #[doc(hidden)]
    pub fn j2k_store_grayi16_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGrayI16Target<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        let plan = validate_gray_targets(
            self,
            targets,
            std::mem::size_of::<i16>(),
            |target| target.input,
            |target| target.output_index,
            grayi16_geometry,
            0,
        )?;
        let output = self.allocate(plan.total_bytes)?;
        let zero_fill_enqueued = plan.requires_zero_fill
            && zero_unwritten_store_output(self, &output, plan.total_bytes, false)?;
        let queued = self
            .enqueue_grayi16_batch(targets, &plan, output.device_ptr())
            .and_then(|queued| attach_zero_fill_completion(self, queued, zero_fill_enqueued))
            .map_err(|error| retire_failed_zero_fill(self, error, zero_fill_enqueued))?;
        Ok((output, plan.ranges, queued))
    }

    /// Store Gray8 samples directly into a validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// The target inputs and destination must remain live until this method
    /// returns success. If CUDA completion cannot be proven and an error is
    /// returned, the caller must quarantine every referenced allocation.
    #[doc(hidden)]
    pub unsafe fn j2k_store_gray8_batch_into_external_device(
        &self,
        targets: &[CudaJ2kStoreGray8Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaExecutionStats), CudaError> {
        // SAFETY: this wrapper retains both borrows and immediately waits for
        // the returned completion guard before returning to safe Rust.
        let (ranges, queued) = unsafe {
            self.j2k_store_gray8_batch_into_external_device_enqueue(targets, destination)?
        };
        let execution = queued.finish()?;
        Ok((ranges, execution))
    }

    /// Enqueue Gray8 final store into a validated caller-owned CUDA range.
    ///
    /// The returned guard retains uploaded job metadata but cannot own the
    /// caller's coefficient buffers or external destination.
    ///
    /// # Safety
    ///
    /// Every target input and the destination allocation must remain live and
    /// unavailable for mutation or reuse until the returned guard finishes or
    /// drops after confirmed CUDA completion. If completion cannot be proven,
    /// the caller must quarantine those allocations rather than free them.
    /// Stream ordering alone does not validate codec status; decoded pixels
    /// must not be exposed as valid output until the owning codec guard has
    /// successfully validated the group.
    #[doc(hidden)]
    pub unsafe fn j2k_store_gray8_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGray8Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_gray_targets(
            self,
            targets,
            1,
            |target| target.input,
            |target| target.output_index,
            gray8_geometry,
            0,
        )?;
        let base = external_destination_base(self, destination, &plan, 1)?;
        let queued = self.enqueue_gray8_batch(targets, &plan, base)?;
        Ok((plan.ranges, queued))
    }

    /// Store Gray16 samples directly into a validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// The target inputs and destination must remain live until this method
    /// returns success. If CUDA completion cannot be proven and an error is
    /// returned, the caller must quarantine every referenced allocation.
    #[doc(hidden)]
    pub unsafe fn j2k_store_gray16_batch_into_external_device(
        &self,
        targets: &[CudaJ2kStoreGray16Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaExecutionStats), CudaError> {
        // SAFETY: this wrapper retains both borrows and immediately waits for
        // the returned completion guard before returning to safe Rust.
        let (ranges, queued) = unsafe {
            self.j2k_store_gray16_batch_into_external_device_enqueue(targets, destination)?
        };
        let execution = queued.finish()?;
        Ok((ranges, execution))
    }

    /// Enqueue Gray16 final store into a validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// Every target input and the destination allocation must remain live and
    /// unavailable for mutation or reuse until the returned guard finishes or
    /// drops after confirmed CUDA completion. If completion cannot be proven,
    /// the caller must quarantine those allocations rather than free them.
    /// Stream ordering alone does not validate codec status; decoded pixels
    /// must not be exposed as valid output until the owning codec guard has
    /// successfully validated the group.
    #[doc(hidden)]
    pub unsafe fn j2k_store_gray16_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGray16Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_gray_targets(
            self,
            targets,
            std::mem::size_of::<u16>(),
            |target| target.input,
            |target| target.output_index,
            gray16_geometry,
            0,
        )?;
        let base =
            external_destination_base(self, destination, &plan, std::mem::align_of::<u16>())?;
        let queued = self.enqueue_gray16_batch(targets, &plan, base)?;
        Ok((plan.ranges, queued))
    }

    /// Store signed GrayI16 samples directly into a validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// The target inputs and destination must remain live until this method
    /// returns success. If CUDA completion cannot be proven and an error is
    /// returned, the caller must quarantine every referenced allocation.
    #[doc(hidden)]
    pub unsafe fn j2k_store_grayi16_batch_into_external_device(
        &self,
        targets: &[CudaJ2kStoreGrayI16Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaExecutionStats), CudaError> {
        // SAFETY: this wrapper retains both borrows and immediately waits for
        // the returned completion guard before returning to safe Rust.
        let (ranges, queued) = unsafe {
            self.j2k_store_grayi16_batch_into_external_device_enqueue(targets, destination)?
        };
        let execution = queued.finish()?;
        Ok((ranges, execution))
    }

    /// Enqueue signed GrayI16 final store into a validated caller-owned CUDA range.
    ///
    /// # Safety
    ///
    /// Every target input and the destination allocation must remain live and
    /// unavailable for mutation or reuse until the returned guard finishes or
    /// drops after confirmed CUDA completion. If completion cannot be proven,
    /// the caller must quarantine those allocations rather than free them.
    /// Stream ordering alone does not validate codec status; decoded pixels
    /// must not be exposed as valid output until the owning codec guard has
    /// successfully validated the group.
    #[doc(hidden)]
    pub unsafe fn j2k_store_grayi16_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreGrayI16Target<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_gray_targets(
            self,
            targets,
            std::mem::size_of::<i16>(),
            |target| target.input,
            |target| target.output_index,
            grayi16_geometry,
            0,
        )?;
        let base =
            external_destination_base(self, destination, &plan, std::mem::align_of::<i16>())?;
        let queued = self.enqueue_grayi16_batch(targets, &plan, base)?;
        Ok((plan.ranges, queued))
    }
}
