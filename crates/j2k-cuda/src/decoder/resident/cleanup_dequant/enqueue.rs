// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    cuda_error, pooled_cuda_buffer, run_component_cleanup_dequant_batches, CudaBufferPool,
    CudaComponentDecodeWork, CudaHtj2kCleanupTarget, CudaHtj2kDecodeResources,
    CudaQueuedHtj2kCleanup, Error, HostPhaseBudget, CUDA_HTJ2K_KERNELS_NOT_READY,
};

/// Enqueue the HT cleanup/dequantization portion of a batch without a host
/// completion boundary. Classic blocks retain their existing completed path.
pub(in crate::decoder) fn enqueue_component_cleanup_dequant_batches(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    live_host_bytes: usize,
) -> Result<Option<CudaQueuedHtj2kCleanup>, Error> {
    if component_work
        .iter()
        .any(|work| !work.pending_classic_bands.is_empty())
    {
        run_component_cleanup_dequant_batches(
            context,
            decode_resources,
            component_work,
            pool,
            false,
            live_host_bytes,
        )?;
        return Ok(None);
    }

    let pending_count = component_work
        .iter()
        .map(|work| work.pending_dequant_bands.len())
        .sum::<usize>();
    if pending_count == 0 {
        return Ok(None);
    }
    let accounting_index = component_work
        .iter()
        .position(|work| !work.pending_dequant_bands.is_empty())
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let has_refinement = component_work.iter().any(|work| {
        work.pending_dequant_bands.iter().any(|pending| {
            pending
                .jobs
                .iter()
                .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1)
        })
    });
    let mut budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA queued cleanup targets", live_host_bytes)?;
    let mut targets = budget.try_vec_with_capacity(pending_count)?;
    for work in component_work.iter() {
        for pending in &work.pending_dequant_bands {
            targets.push(CudaHtj2kCleanupTarget {
                coefficients: pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?,
                jobs: &pending.jobs,
                output_words: pending.output_words,
            });
        }
    }

    // SAFETY: component_work retains every coefficient target, decode_resources
    // retains payload/tables, and the returned typed guard is kept until the
    // final same-stream store has established completion.
    let queued = unsafe {
        if has_refinement {
            context
                .decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
                    decode_resources,
                    &targets,
                    pool,
                    budget.live_bytes(),
                )
                .map_err(cuda_error)?
        } else {
            context
                .decode_htj2k_codeblocks_cleanup_dequantize_multi_enqueue_with_resources_and_pool(
                    decode_resources,
                    &targets,
                    pool,
                    budget.live_bytes(),
                )
                .map_err(cuda_error)?
        }
    };
    let cleanup_stats = queued.execution();
    let dequant_stats = if has_refinement {
        // SAFETY: the queued guard owns the uploaded job metadata and remains
        // live through final store completion; coefficient owners remain in
        // component_work for the same duration.
        unsafe { context.j2k_dequantize_queued_htj2k_cleanup_enqueue(&queued) }
            .map_err(cuda_error)?
    } else {
        j2k_cuda_runtime::CudaExecutionStats::default()
    };
    {
        let accounting = &mut component_work[accounting_index];
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(cleanup_stats.kernel_dispatches())
            .saturating_add(dequant_stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(cleanup_stats.decode_kernel_dispatches())
            .saturating_add(dequant_stats.decode_kernel_dispatches());
        accounting.timings.ht_dispatch_count = accounting
            .timings
            .ht_dispatch_count
            .saturating_add(cleanup_stats.kernel_dispatches());
        accounting.timings.dequant_dispatch_count = accounting
            .timings
            .dequant_dispatch_count
            .saturating_add(dequant_stats.kernel_dispatches());
    }
    for work in component_work {
        work.pending_dequant_bands.clear();
    }
    Ok(Some(queued))
}
