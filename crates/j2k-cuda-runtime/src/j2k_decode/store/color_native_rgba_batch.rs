// SPDX-License-Identifier: MIT OR Apache-2.0

mod validation;

use validation::{validate_external, validate_targets, NativeRgbaBatchPlan};

use crate::{
    allocation::HostPhaseBudget,
    bytes::store_rgba_native_batch_jobs_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelContiguousBatchOutput},
    memory::{CudaDeviceBuffer, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

use super::CudaQueuedJ2kStoreBatch;
use crate::j2k_decode::types::{CudaJ2kStoreRgbaNativeBatchJob, CudaJ2kStoreRgbaNativeTarget};

const RGBA_CHANNELS: usize = 4;

#[derive(Clone, Copy)]
enum NativeRgbaStorage {
    U8,
    U16,
    I16,
}

impl NativeRgbaStorage {
    const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => std::mem::size_of::<u8>(),
            Self::U16 => std::mem::size_of::<u16>(),
            Self::I16 => std::mem::size_of::<i16>(),
        }
    }

    const fn max_precision(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 | Self::I16 => 16,
        }
    }
}

impl CudaContext {
    /// Store exact-native RGBA U8 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgba8_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgba_native_batch(targets, NativeRgbaStorage::U8)
    }

    /// Store exact-native RGBA U16 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgba16_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgba_native_batch(targets, NativeRgbaStorage::U16)
    }

    /// Store exact-native signed RGBA I16 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgbai16_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgba_native_batch(targets, NativeRgbaStorage::I16)
    }

    /// Enqueue exact-native RGBA U8 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgba8_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.enqueue_owned_rgba_native_batch(targets, NativeRgbaStorage::U8)
    }

    /// Enqueue exact-native RGBA U16 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgba16_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.enqueue_owned_rgba_native_batch(targets, NativeRgbaStorage::U16)
    }

    /// Enqueue exact-native signed RGBA I16 into one codec-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgbai16_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.enqueue_owned_rgba_native_batch(targets, NativeRgbaStorage::I16)
    }

    fn store_rgba_native_batch(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        storage: NativeRgbaStorage,
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let (output, ranges, queued) = self.enqueue_owned_rgba_native_batch(targets, storage)?;
        let execution = match queued.finish() {
            Ok(execution) => execution,
            Err(error) => {
                if error.completion_is_uncertain() {
                    core::mem::forget(output);
                }
                return Err(error);
            }
        };
        Ok(CudaKernelContiguousBatchOutput {
            output,
            ranges,
            execution,
        })
    }

    fn enqueue_owned_rgba_native_batch(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        storage: NativeRgbaStorage,
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        let plan = validate_targets(self, targets, storage)?;
        let output = self.allocate(plan.total_bytes)?;
        // SAFETY: returned owners retain output and the caller retains planes.
        let queued =
            unsafe { self.enqueue_rgba_native(targets, &plan, output.device_ptr(), storage)? };
        Ok((output, plan.ranges, queued))
    }

    /// Enqueue RGBA U8 into validated caller-owned CUDA storage.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgba8_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded lifetime and exclusivity contract.
        unsafe { self.enqueue_external_rgba_native(targets, destination, NativeRgbaStorage::U8) }
    }

    /// Enqueue RGBA U16 into validated caller-owned CUDA storage.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgba16_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded lifetime and exclusivity contract.
        unsafe { self.enqueue_external_rgba_native(targets, destination, NativeRgbaStorage::U16) }
    }

    /// Enqueue signed RGBA I16 into validated caller-owned CUDA storage.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgbai16_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded lifetime and exclusivity contract.
        unsafe { self.enqueue_external_rgba_native(targets, destination, NativeRgbaStorage::I16) }
    }

    unsafe fn enqueue_external_rgba_native(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
        storage: NativeRgbaStorage,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_targets(self, targets, storage)?;
        let output_base = validate_external(self, destination, &plan, storage)?;
        // SAFETY: caller retains and excludes all source/destination storage.
        let queued = unsafe { self.enqueue_rgba_native(targets, &plan, output_base, storage)? };
        Ok((plan.ranges, queued))
    }

    unsafe fn enqueue_rgba_native(
        &self,
        targets: &[CudaJ2kStoreRgbaNativeTarget<'_>],
        plan: &NativeRgbaBatchPlan,
        output_base: u64,
        storage: NativeRgbaStorage,
    ) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
        if plan.active_count == 0 {
            return Ok(CudaQueuedJ2kStoreBatch {
                completion: None,
                jobs: None,
                execution: CudaExecutionStats::default(),
            });
        }
        let mut budget = HostPhaseBudget::new("CUDA exact-native RGBA batch job metadata");
        let mut jobs = budget.try_vec_with_capacity(plan.active_count)?;
        for (target, item) in targets.iter().zip(&plan.items) {
            if !item.active {
                continue;
            }
            let range = plan.ranges[item.range_index];
            let offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            jobs.push(CudaJ2kStoreRgbaNativeBatchJob {
                plane0_ptr: target.plane0.device_ptr(),
                plane1_ptr: target.plane1.device_ptr(),
                plane2_ptr: target.plane2.device_ptr(),
                plane3_ptr: target.plane3.device_ptr(),
                output_ptr: output_base
                    .checked_add(offset)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?,
                job: target.job,
                reserved_tail: 0,
            });
        }
        let jobs = self.upload(store_rgba_native_batch_jobs_as_bytes(&jobs))?;
        let launch = match storage {
            // SAFETY: validated targets and the uploaded descriptor owner stay
            // live until the recorded completion event is retired below.
            NativeRgbaStorage::U8 => unsafe {
                self.launch_j2k_store_rgba8_native_batch_enqueue(
                    &jobs,
                    plan.max_pixels,
                    plan.active_count,
                )
            },
            // SAFETY: the same validated ownership and completion contract
            // applies to the two-byte unsigned store entrypoint.
            NativeRgbaStorage::U16 => unsafe {
                self.launch_j2k_store_rgba16_native_batch_enqueue(
                    &jobs,
                    plan.max_pixels,
                    plan.active_count,
                )
            },
            // SAFETY: the same validated ownership and completion contract
            // applies to the two-byte signed store entrypoint.
            NativeRgbaStorage::I16 => unsafe {
                self.launch_j2k_store_rgbai16_native_batch_enqueue(
                    &jobs,
                    plan.max_pixels,
                    plan.active_count,
                )
            },
        };
        if let Err(error) = launch {
            return self.synchronize_then_error(error);
        }
        let completion = match self.create_event().and_then(|event| {
            event.record_default_stream()?;
            Ok(event)
        }) {
            Ok(completion) => completion,
            Err(error) => return self.synchronize_then_error(error),
        };
        Ok(CudaQueuedJ2kStoreBatch {
            completion: Some(completion),
            jobs: Some(jobs),
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
        })
    }
}
