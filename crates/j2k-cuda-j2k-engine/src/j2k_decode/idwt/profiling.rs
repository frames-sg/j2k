// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaJ2kIdwtNormalization;
use crate::{
    error::CudaError, execution::CudaQueuedExecution, memory::CudaBufferPool, CudaJ2kIdwtTarget,
};

/// GPU-event timings for the final stage of a batched IDWT sequence.
///
/// These fields split the existing aggregate IDWT measurement for profiling;
/// they do not change the executed transform or its dispatch policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kIdwtBatchStageProfile {
    /// Whether this row describes the final non-empty IDWT stage.
    pub final_stage: bool,
    /// Complete GPU time for this IDWT stage.
    pub elapsed_us: u128,
    /// GPU time for interleave plus horizontal synthesis.
    pub interleave_horizontal_us: u128,
    /// GPU time for vertical synthesis.
    pub vertical_us: u128,
}

impl crate::J2kCudaEngine<'_> {
    /// Enqueue an IDWT sequence and collect GPU-event timings for its final
    /// interleave/horizontal and vertical stages.
    ///
    /// # Safety
    ///
    /// Every target buffer must remain allocated and must not be mutated or
    /// reused until the returned execution is finished. Within each stage,
    /// output allocations must be pairwise disjoint and may not overlap any
    /// concurrently read input allocation. The pool and all targets must
    /// belong to this context and remain confined to the same stream.
    #[doc(hidden)]
    pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_profiled_with_pool_and_live_host_bytes_normalized(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        normalization: CudaJ2kIdwtNormalization,
    ) -> Result<(CudaQueuedExecution, CudaJ2kIdwtBatchStageProfile), CudaError> {
        // SAFETY: profiling adds only ordered CUDA events and preserves the
        // caller's complete lifetime, context, aliasing, and stream contract.
        unsafe {
            self.j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes_impl(
                target_batches,
                pool,
                live_host_bytes,
                normalization,
                true,
            )
        }
    }
}
