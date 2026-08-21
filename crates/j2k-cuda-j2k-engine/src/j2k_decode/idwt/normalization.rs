// SPDX-License-Identifier: MIT OR Apache-2.0

use super::J2kInverseDwtSinglePoolRequest;
use crate::{
    error::CudaError,
    execution::{CudaExecutionStats, CudaPooledKernelOutput, CudaQueuedExecution},
    memory::{CudaBufferPool, CudaDeviceBuffer},
    CudaJ2kIdwtJob, CudaJ2kIdwtTarget,
};
/// Irreversible 9/7 high-pass normalization used by CUDA IDWT entry points.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub enum CudaJ2kIdwtNormalization {
    /// Standard inverse-kappa normalization used by the existing low-level API.
    #[default]
    Standard,
    /// Historical `OpenJPEG` codestream normalization paired with parsed step sizes.
    OpenJpegCodestream,
}
impl crate::J2kCudaEngine<'_> {
    /// Apply one inverse JPEG 2000 DWT decomposition with explicit high-pass normalization.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_device_with_pool_normalized(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        job: CudaJ2kIdwtJob,
        normalization: CudaJ2kIdwtNormalization,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(J2kInverseDwtSinglePoolRequest {
            bands,
            job,
            normalization,
            synchronize_each_launch: true,
            pool,
        })
    }

    /// Apply one inverse DWT without per-kernel synchronization and with explicit normalization.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_device_untimed_with_pool_normalized(
        &self,
        bands: [&CudaDeviceBuffer; 4],
        job: CudaJ2kIdwtJob,
        normalization: CudaJ2kIdwtNormalization,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(J2kInverseDwtSinglePoolRequest {
            bands,
            job,
            normalization,
            synchronize_each_launch: false,
            pool,
        })
    }

    /// Apply batched inverse DWT decompositions with explicit normalization.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_batch_device_with_pool_normalized(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        normalization: CudaJ2kIdwtNormalization,
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, normalization, pool)
    }

    /// Enqueue an IDWT sequence while accounting caller-live host metadata.
    ///
    /// # Safety
    ///
    /// The targets and pool must satisfy the lifetime, aliasing, context, and stream requirements in
    /// [`Self::j2k_inverse_dwt_batch_sequence_enqueue_with_pool`].
    #[doc(hidden)]
    pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaQueuedExecution, CudaError> {
        // SAFETY: this forwards the caller's target, pool, and stream-lifetime
        // contract unchanged while retaining the existing standard normalization.
        unsafe {
            self.j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes_impl(
                target_batches,
                pool,
                live_host_bytes,
                CudaJ2kIdwtNormalization::Standard,
                false,
            )
        }
        .map(|(execution, _profile)| execution)
    }

    /// Enqueue a normalized IDWT sequence while accounting caller-live host metadata.
    ///
    /// # Safety
    ///
    /// Every target buffer must remain allocated and must not be mutated or
    /// reused until the returned execution is finished. Within each stage,
    /// output allocations must be pairwise disjoint and may not overlap any
    /// concurrently read input allocation. The pool and all targets must
    /// belong to this context and remain confined to the same stream.
    #[doc(hidden)]
    pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes_normalized(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        normalization: CudaJ2kIdwtNormalization,
    ) -> Result<CudaQueuedExecution, CudaError> {
        // SAFETY: selecting arithmetic metadata preserves the caller's lifetime and aliasing contract.
        unsafe {
            self.j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes_impl(
                target_batches,
                pool,
                live_host_bytes,
                normalization,
                false,
            )
        }
        .map(|(execution, _profile)| execution)
    }
}
