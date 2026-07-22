// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::HtGpuJobPassBucket;
use j2k_cuda_runtime::{
    CudaHtj2kCleanupTarget, CudaHtj2kDecodeResources, CudaQueuedHtj2kCleanup,
    CudaQueuedHtj2kCleanupGroup,
};

use super::super::super::super::{cuda_error, CudaBufferPool, CudaContext, Error};

#[expect(
    clippy::too_many_arguments,
    reason = "the CUDA enqueue boundary keeps resources, targets, pass route, pool, host budget, and status range explicit"
)]
pub(super) fn enqueue_chunk_kernel(
    context: &CudaContext,
    resources: &CudaHtj2kDecodeResources,
    targets: &[CudaHtj2kCleanupTarget<'_>],
    bucket: HtGpuJobPassBucket,
    pool: &CudaBufferPool,
    live_host_bytes: usize,
    status_group: &CudaQueuedHtj2kCleanupGroup,
    status_offset: usize,
) -> Result<CudaQueuedHtj2kCleanup, Error> {
    // SAFETY: the caller retains uploaded descriptors and every disjoint
    // coefficient target through status-group completion.
    let cleanup = unsafe {
        match bucket {
            HtGpuJobPassBucket::CleanupOnly => context
                .decode_htj2k_codeblocks_cleanup_dequantize_multi_enqueue_into_status_group(
                    resources,
                    targets,
                    pool,
                    live_host_bytes,
                    status_group,
                    status_offset,
                ),
            HtGpuJobPassBucket::SigProp | HtGpuJobPassBucket::MagRef => context
                .decode_htj2k_codeblocks_cleanup_multi_enqueue_into_status_group(
                    resources,
                    targets,
                    pool,
                    live_host_bytes,
                    status_group,
                    status_offset,
                ),
        }
    }
    .map_err(cuda_error)?;
    if bucket != HtGpuJobPassBucket::CleanupOnly {
        // SAFETY: cleanup retains its descriptors, and the same default
        // stream orders dequantization before later IDWT.
        unsafe { context.j2k_dequantize_queued_htj2k_cleanup_enqueue(&cleanup) }
            .map_err(cuda_error)?;
    }
    Ok(cleanup)
}
