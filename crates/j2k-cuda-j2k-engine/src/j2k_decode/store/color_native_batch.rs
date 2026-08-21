// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::store_rgb_native_batch_jobs_as_bytes,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelContiguousBatchOutput},
    memory::{CudaDeviceBuffer, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut},
};

use super::CudaQueuedJ2kStoreBatch;
use crate::j2k_decode::types::{CudaJ2kStoreRgbNativeBatchJob, CudaJ2kStoreRgbNativeTarget};

mod plan;
use plan::{
    validate_external_destination, validate_native_rgb_targets, NativeRgbBatchPlan,
    NativeRgbStorage,
};

impl crate::J2kCudaEngine<'_> {
    /// Store an exact-native RGB U8 batch into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgb8_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgb_native_batch_contiguous_device(targets, NativeRgbStorage::U8)
    }

    /// Enqueue exact-native RGB U8 output into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgb8_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.store_rgb_native_batch_contiguous_device_enqueue(targets, NativeRgbStorage::U8)
    }

    /// Store an exact-native RGB U16 batch into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgb16_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgb_native_batch_contiguous_device(targets, NativeRgbStorage::U16)
    }

    /// Enqueue exact-native RGB U16 output into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgb16_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.store_rgb_native_batch_contiguous_device_enqueue(targets, NativeRgbStorage::U16)
    }

    /// Store an exact-native signed RGB I16 batch into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgbi16_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        self.store_rgb_native_batch_contiguous_device(targets, NativeRgbStorage::I16)
    }

    /// Enqueue exact-native signed RGB I16 output into one J2K-owned allocation.
    #[doc(hidden)]
    pub fn j2k_store_rgbi16_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        self.store_rgb_native_batch_contiguous_device_enqueue(targets, NativeRgbStorage::I16)
    }

    fn store_rgb_native_batch_contiguous_device(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        storage: NativeRgbStorage,
    ) -> Result<CudaKernelContiguousBatchOutput, CudaError> {
        let (output, ranges, queued) =
            self.store_rgb_native_batch_contiguous_device_enqueue(targets, storage)?;
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

    fn store_rgb_native_batch_contiguous_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        storage: NativeRgbStorage,
    ) -> Result<
        (
            CudaDeviceBuffer,
            Vec<CudaDeviceBufferRange>,
            CudaQueuedJ2kStoreBatch,
        ),
        CudaError,
    > {
        let plan = validate_native_rgb_targets(self.context, targets, storage)?;
        let output = self.allocate(plan.total_bytes)?;
        // SAFETY: the returned owners retain the destination and every caller
        // retains the source planes until the queued guard is retired.
        let queued =
            unsafe { self.enqueue_rgb_native_batch(targets, &plan, output.device_ptr(), storage)? };
        let ranges = plan.ranges;
        Ok((output, ranges, queued))
    }

    /// Enqueue exact-native RGB U8 output into caller-owned CUDA storage.
    ///
    /// # Safety
    ///
    /// All source planes and the external destination must remain live and
    /// inaccessible until the returned completion owner is retired.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgb8_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded from the caller after validating the destination.
        unsafe {
            self.store_rgb_native_batch_into_external_device_enqueue(
                targets,
                destination,
                NativeRgbStorage::U8,
            )
        }
    }

    /// Enqueue exact-native RGB U16 output into caller-owned CUDA storage.
    ///
    /// # Safety
    ///
    /// All source planes and the external destination must remain live and
    /// inaccessible until the returned completion owner is retired.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgb16_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded from the caller after validating the destination.
        unsafe {
            self.store_rgb_native_batch_into_external_device_enqueue(
                targets,
                destination,
                NativeRgbStorage::U16,
            )
        }
    }

    /// Enqueue exact-native signed RGB I16 output into caller-owned CUDA storage.
    ///
    /// # Safety
    ///
    /// All source planes and the external destination must remain live and
    /// inaccessible until the returned completion owner is retired.
    #[doc(hidden)]
    pub unsafe fn j2k_store_rgbi16_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        // SAFETY: forwarded from the caller after validating the destination.
        unsafe {
            self.store_rgb_native_batch_into_external_device_enqueue(
                targets,
                destination,
                NativeRgbStorage::I16,
            )
        }
    }

    unsafe fn store_rgb_native_batch_into_external_device_enqueue(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
        storage: NativeRgbStorage,
    ) -> Result<(Vec<CudaDeviceBufferRange>, CudaQueuedJ2kStoreBatch), CudaError> {
        let plan = validate_native_rgb_targets(self.context, targets, storage)?;
        let output_base = validate_external_destination(self.context, destination, &plan, storage)?;
        // SAFETY: caller retains every source and destination owner through
        // the returned completion guard.
        let queued =
            unsafe { self.enqueue_rgb_native_batch(targets, &plan, output_base, storage)? };
        Ok((plan.ranges, queued))
    }

    unsafe fn enqueue_rgb_native_batch(
        &self,
        targets: &[CudaJ2kStoreRgbNativeTarget<'_>],
        plan: &NativeRgbBatchPlan,
        output_base: u64,
        storage: NativeRgbStorage,
    ) -> Result<CudaQueuedJ2kStoreBatch, CudaError> {
        if plan.active_count == 0 {
            return Ok(CudaQueuedJ2kStoreBatch {
                completion: None,
                jobs: None,
                execution: CudaExecutionStats::default(),
            });
        }
        let mut budget = HostPhaseBudget::new("CUDA exact-native RGB batch job metadata");
        let mut jobs = budget.try_vec_with_capacity(plan.active_count)?;
        for (target, item) in targets.iter().zip(&plan.items) {
            if !item.active {
                continue;
            }
            let range = plan.ranges[item.range_index];
            let offset = u64::try_from(range.offset)
                .map_err(|_| CudaError::LengthTooLarge { len: range.offset })?;
            jobs.push(CudaJ2kStoreRgbNativeBatchJob {
                plane0_ptr: target.plane0.device_ptr(),
                plane1_ptr: target.plane1.device_ptr(),
                plane2_ptr: target.plane2.device_ptr(),
                output_ptr: output_base
                    .checked_add(offset)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?,
                job: target.job,
            });
        }
        let jobs_buffer = self.upload(store_rgb_native_batch_jobs_as_bytes(&jobs))?;
        let launch = match storage {
            NativeRgbStorage::U8 => {
                // SAFETY: `jobs_buffer` is retained below through the event.
                unsafe {
                    self.launch_j2k_store_rgb8_native_batch_enqueue(
                        &jobs_buffer,
                        plan.max_pixels,
                        plan.active_count,
                    )
                }
            }
            NativeRgbStorage::U16 => {
                // SAFETY: `jobs_buffer` is retained below through the event.
                unsafe {
                    self.launch_j2k_store_rgb16_native_batch_enqueue(
                        &jobs_buffer,
                        plan.max_pixels,
                        plan.active_count,
                    )
                }
            }
            NativeRgbStorage::I16 => {
                // SAFETY: `jobs_buffer` is retained below through the event.
                unsafe {
                    self.launch_j2k_store_rgbi16_native_batch_enqueue(
                        &jobs_buffer,
                        plan.max_pixels,
                        plan.active_count,
                    )
                }
            }
        };
        if let Err(error) = launch {
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
        Ok(CudaQueuedJ2kStoreBatch {
            completion: Some(completion),
            jobs: Some(jobs_buffer),
            execution: CudaExecutionStats::new(1, 0, 1, false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::plan::{NativeRgbStorage, RGB_LAYOUT_NCHW, RGB_LAYOUT_NHWC};

    #[test]
    fn exact_native_storage_and_layout_contract_is_not_display_scaled() {
        assert_eq!(NativeRgbStorage::U8.max_precision(), 8);
        assert_eq!(NativeRgbStorage::U16.max_precision(), 16);
        assert_eq!(NativeRgbStorage::I16.max_precision(), 16);
        assert_eq!((RGB_LAYOUT_NHWC, RGB_LAYOUT_NCHW), (0, 1));

        let exports = include_str!("../../cuda_oxide_j2k_decode_store/simt/src/exports.rs");
        let native_color =
            include_str!("../../cuda_oxide_j2k_decode_store/simt/src/native_color.rs");
        for entrypoint in [
            "j2k_store_rgb8_native_batch",
            "j2k_store_rgb16_native_batch",
            "j2k_store_rgbi16_native_batch",
            "j2k_store_rgba8_native_batch",
            "j2k_store_rgba16_native_batch",
            "j2k_store_rgbai16_native_batch",
        ] {
            assert!(exports.contains(entrypoint));
        }
        assert!(native_color.contains("sample_as_native_u8(samples.0"));
        assert!(native_color.contains("sample_as_native_u16(samples.0"));
        assert!(native_color.contains("sample_as_native_i16(samples.0"));
        assert!(native_color.contains("add_native_level_shift(out0, job.addend0, job.transform)"));
        assert!(native_color.contains(
            "add_native_level_shift(load_f32(plane3, src3), job.addend3, job.transform)"
        ));
        let sample = include_str!("../../cuda_oxide_j2k_decode_store/simt/src/sample.rs");
        assert!(sample.contains("pub(crate) fn round_ties_even_f32"));
        let color = include_str!("../../cuda_oxide_j2k_decode_store/simt/src/color.rs");
        assert!(color.contains("fn add_mct_level_shift"));
        assert_eq!(color.matches("add_mct_level_shift(out").count(), 6);
        let transform = include_str!("../../cuda_oxide_j2k_decode_store/simt/src/transform.rs");
        assert!(transform.contains("fma.rn.f32"));
        assert!(transform.contains("fused_mul_add_f32(src2, 1.402, src0)"));
        assert!(transform.contains("fused_mul_add_f32(src2, -0.71414"));
        let idwt = include_str!("../../cuda_oxide_j2k_idwt/simt/src/main.rs");
        assert!(idwt.contains("fused_mul_add_f32(left_sample + right_sample"));
        assert!(idwt.contains("fused_mul_add_f32(above + below"));
        assert!(idwt.contains(
            "fused_mul_add_f32(\n            line.load(left) + line.load(right),\n            coefficient,\n            line.load(index),"
        ));
        assert!(idwt.contains("IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5"));
        assert!(idwt.contains("transform_mode == IDWT_CODESTREAM_97_MODE"));
    }
}
