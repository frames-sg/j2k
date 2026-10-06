// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    combine_cuda_cleanup_errors, cuda_error, profile, CudaBufferPool, CudaComponentDecodeWork,
    CudaHtj2kCleanupTarget, CudaHtj2kDecodeResources, CudaHtj2kDequantizeTarget,
    CudaQueuedHtj2kCleanup, Error, J2kCudaEngine, CUDA_HTJ2K_KERNELS_NOT_READY,
};
use super::buffer_access::pooled_cuda_buffer;
use crate::allocation::HostPhaseBudget;

#[cfg(feature = "cuda-runtime")]
mod classic;
#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) use classic::{
    enqueue_component_classic_batches, run_component_classic_batches, QueuedComponentClassicDecode,
};
#[cfg(feature = "cuda-runtime")]
mod enqueue;
#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) use enqueue::enqueue_component_cleanup_dequant_batches;

#[cfg(feature = "cuda-runtime")]
fn enqueue_pending_coefficient_clears(
    context: &j2k_cuda_runtime::CudaContext,
    component_work: &mut [CudaComponentDecodeWork],
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<bool, Error> {
    let target_count = component_work
        .iter()
        .flat_map(|work| &work.pending_dequant_bands)
        .filter(|pending| pending.needs_zero_fill)
        .count();
    if target_count == 0 {
        return Ok(false);
    }
    let accounting_index = component_work
        .iter()
        .position(|work| {
            work.pending_dequant_bands
                .iter()
                .any(|pending| pending.needs_zero_fill)
        })
        .ok_or(Error::capability_rejected(
            j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
        ))?;
    let mut budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA coefficient clear targets", live_host_bytes)?;
    let mut targets = budget.try_vec_with_capacity(target_count)?;
    for work in component_work.iter() {
        for pending in &work.pending_dequant_bands {
            if pending.needs_zero_fill {
                targets.push(CudaHtj2kCleanupTarget {
                    coefficients: pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?,
                    jobs: &pending.jobs,
                    output_words: pending.output_words,
                });
            }
        }
    }
    let clear_start = profile::profile_now(collect_stage_timings);
    // SAFETY: component_work retains these planes through the following
    // entropy, IDWT, and store operations on the context's default stream.
    let stats =
        unsafe { J2kCudaEngine::new(context).clear_htj2k_coefficients_multi_enqueue(&targets) }
            .map_err(cuda_error)?;
    let clear_us = profile::elapsed_us(clear_start);
    drop(targets);
    for work in component_work.iter_mut() {
        for pending in &mut work.pending_dequant_bands {
            pending.needs_zero_fill = false;
        }
    }
    let accounting = &mut component_work[accounting_index];
    accounting.timings.h2d = accounting.timings.h2d.saturating_add(clear_us);
    accounting.dispatches = accounting
        .dispatches
        .saturating_add(stats.kernel_dispatches());
    Ok(true)
}

/// Enqueues pending coefficient clears, then runs `submit`. A clear that is
/// still queued when `submit` fails completes before the error is returned.
#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn with_pending_coefficient_clears<T>(
    context: &j2k_cuda_runtime::CudaContext,
    component_work: &mut [CudaComponentDecodeWork],
    collect_stage_timings: bool,
    live_host_bytes: usize,
    submit: impl FnOnce(&mut [CudaComponentDecodeWork], &mut bool) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut clear_enqueued = enqueue_pending_coefficient_clears(
        context,
        component_work,
        collect_stage_timings,
        live_host_bytes,
    )?;
    match submit(component_work, &mut clear_enqueued) {
        Ok(value) => Ok(value),
        Err(error) if clear_enqueued => match context.synchronize() {
            Ok(()) => Err(error),
            Err(completion) => Err(combine_cuda_cleanup_errors(error, cuda_error(completion))),
        },
        Err(error) => Err(error),
    }
}

/// Completes queued clears and drops dequantization bands when no entropy job
/// is left to read them.
#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn retire_jobless_coefficient_clears(
    context: &j2k_cuda_runtime::CudaContext,
    component_work: &mut [CudaComponentDecodeWork],
    clear_enqueued: &mut bool,
) -> Result<(), Error> {
    if *clear_enqueued {
        context.synchronize().map_err(cuda_error)?;
        *clear_enqueued = false;
    }
    for work in component_work {
        work.pending_dequant_bands.clear();
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::decoder) fn split_htj2k_subband_decode_dispatches(
    kernel_dispatches: usize,
) -> (usize, usize) {
    if kernel_dispatches == 0 {
        return (0, 0);
    }

    let dequant_dispatches = usize::from(kernel_dispatches > 1);
    (
        kernel_dispatches.saturating_sub(dequant_dispatches),
        dequant_dispatches,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn htj2k_batched_cleanup_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(in crate::decoder) fn htj2k_batched_dequant_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn run_component_cleanup_dequant_batches(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<(), Error> {
    with_pending_coefficient_clears(
        context,
        component_work,
        collect_stage_timings,
        live_host_bytes,
        |component_work, clear_enqueued| {
            run_cleanup_dequant_after_clears(
                context,
                decode_resources,
                component_work,
                pool,
                collect_stage_timings,
                live_host_bytes,
                clear_enqueued,
            )
        },
    )
}

/// Runs classic and HT cleanup/dequantization once coefficient clears are
/// enqueued.
#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "cleanup/dequant batch submission preserves CUDA stream and buffer lifetime ordering"
)]
pub(in crate::decoder) fn run_cleanup_dequant_after_clears(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    live_host_bytes: usize,
    clear_enqueued: &mut bool,
) -> Result<(), Error> {
    {
        run_component_classic_batches(
            context,
            decode_resources,
            component_work,
            pool,
            collect_stage_timings,
            live_host_bytes,
        )?;
        let pending_count = component_work
            .iter()
            .flat_map(|work| &work.pending_dequant_bands)
            .filter(|pending| !pending.jobs.is_empty())
            .count();
        if pending_count == 0 {
            return retire_jobless_coefficient_clears(context, component_work, clear_enqueued);
        }
        let accounting_index = component_work
            .iter()
            .position(|work| {
                work.pending_dequant_bands
                    .iter()
                    .any(|pending| !pending.jobs.is_empty())
            })
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
            ))?;

        let has_refinement = component_work.iter().any(|work| {
            work.pending_dequant_bands.iter().any(|pending| {
                pending.jobs.iter().any(|job| {
                    job.refinement_length > 0 || u32::from(job.number_of_coding_passes) > 1
                })
            })
        });
        let mut cleanup_budget =
            HostPhaseBudget::with_live_bytes("j2k CUDA cleanup target phase", live_host_bytes)?;
        let mut cleanup_targets = cleanup_budget.try_vec_with_capacity(pending_count)?;
        for work in component_work.iter() {
            for pending in &work.pending_dequant_bands {
                if pending.jobs.is_empty() {
                    continue;
                }
                let coefficients = pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
                cleanup_targets.push(CudaHtj2kCleanupTarget {
                    coefficients,
                    jobs: &pending.jobs,
                    output_words: pending.output_words,
                });
            }
        }
        if !has_refinement {
            let stage_start = profile::profile_now(collect_stage_timings);
            let mut run = || {
                J2kCudaEngine::new(context)
                .decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed_and_live_host_bytes(
                    decode_resources,
                    &cleanup_targets,
                    pool,
                    collect_stage_timings,
                    cleanup_budget.live_bytes(),
                )
            };
            let ((stats, runtime_timings), fused_us) = if collect_stage_timings {
                context.time_default_stream_named_us(
                    "j2k.htj2k.decode.cleanup_dequantize.batch",
                    &mut run,
                )
            } else {
                // Successful work is already complete; preserve error-path completion
                // before any captured allocations can be released.
                context
                    .with_nvtx_range("j2k.htj2k.decode.cleanup_dequantize.batch", run)
                    .or_else(|error| context.synchronize_then_error(error))
                    .map(|output| (output, 0))
            }
            .map_err(cuda_error)?;
            let stage_wall_us = profile::elapsed_us(stage_start);
            // Fused cleanup/dequantization reports every kernel as cleanup.
            let (cleanup_dispatches, dequant_dispatches) = (stats.kernel_dispatches(), 0);
            {
                let accounting = &mut component_work[accounting_index];
                accounting.timings.h2d = accounting
                    .timings
                    .h2d
                    .saturating_add(stage_wall_us.saturating_sub(fused_us));
                accounting.timings.ht_cleanup =
                    accounting.timings.ht_cleanup.saturating_add(fused_us);
                accounting.timings.status_d2h = accounting
                    .timings
                    .status_d2h
                    .saturating_add(runtime_timings.status_d2h_us);
                accounting.timings.ht_dispatch_count = accounting
                    .timings
                    .ht_dispatch_count
                    .saturating_add(cleanup_dispatches);
                accounting.timings.dequant_dispatch_count = accounting
                    .timings
                    .dequant_dispatch_count
                    .saturating_add(dequant_dispatches);
                accounting.timings.fused_dequant_dispatch_count = accounting
                    .timings
                    .fused_dequant_dispatch_count
                    .saturating_add(usize::from(stats.kernel_dispatches() != 0));
                accounting.dispatches = accounting
                    .dispatches
                    .saturating_add(stats.kernel_dispatches());
                accounting.decode_dispatches = accounting
                    .decode_dispatches
                    .saturating_add(stats.decode_kernel_dispatches());
            }

            for work in component_work {
                work.pending_dequant_bands.clear();
            }
            return Ok(());
        }
        let mut queued_cleanup: Option<CudaQueuedHtj2kCleanup> = None;
        let stage_start = profile::profile_now(collect_stage_timings);
        let (stats, cleanup_us, status_d2h_us) = if collect_stage_timings {
            let ((stats, runtime_timings), cleanup_us) = context
            .time_default_stream_named_us("j2k.htj2k.decode.cleanup.batch", || {
                J2kCudaEngine::new(context)
                    .decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed_and_live_host_bytes(
                        decode_resources,
                        &cleanup_targets,
                        pool,
                        collect_stage_timings,
                        cleanup_budget.live_bytes(),
                    )
            })
            .map_err(cuda_error)?;
            (stats, cleanup_us, runtime_timings.status_d2h_us)
        } else {
            // SAFETY: `queued_cleanup` retains the typed cleanup guard and every
            // component target remains owned by `component_work` through ordered
            // dequantization and final guard completion below. The inner enqueue
            // receives the same ownership and default-stream guarantees.
            let queued = unsafe {
            context.submit_default_stream_named("j2k.htj2k.decode.cleanup.batch", || {
                J2kCudaEngine::new(context)
                    .decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
                        decode_resources,
                        &cleanup_targets,
                        pool,
                        cleanup_budget.live_bytes(),
                    )
            })
        }
        .map_err(cuda_error)?;
            let stats = queued.execution();
            queued_cleanup = Some(queued);
            (stats, 0, 0)
        };
        drop(cleanup_targets);
        let stage_wall_us = profile::elapsed_us(stage_start);
        let ht_dispatches = htj2k_batched_cleanup_dispatches(pending_count);
        {
            let accounting = &mut component_work[accounting_index];
            accounting.timings.h2d = accounting
                .timings
                .h2d
                .saturating_add(stage_wall_us.saturating_sub(cleanup_us));
            accounting.timings.ht_cleanup =
                accounting.timings.ht_cleanup.saturating_add(cleanup_us);
            accounting.timings.status_d2h =
                accounting.timings.status_d2h.saturating_add(status_d2h_us);
            if has_refinement {
                accounting.timings.ht_refine =
                    accounting.timings.ht_refine.saturating_add(cleanup_us);
            }
            accounting.timings.ht_dispatch_count = accounting
                .timings
                .ht_dispatch_count
                .saturating_add(ht_dispatches);
            accounting.timings.ht_refinement_dispatch_count = accounting
                .timings
                .ht_refinement_dispatch_count
                .saturating_add(ht_dispatches);
            accounting.dispatches = accounting
                .dispatches
                .saturating_add(stats.kernel_dispatches());
            accounting.decode_dispatches = accounting
                .decode_dispatches
                .saturating_add(stats.decode_kernel_dispatches());
        }

        let stage_start = profile::profile_now(collect_stage_timings);
        let (stats, dequant_us, dequant_target_count) = {
            let dequant_target_count = pending_count;
            let dequant_result = if let Some(queued) = queued_cleanup.as_ref() {
                context.time_default_stream_named_us_if(
                    collect_stage_timings,
                    "j2k.htj2k.decode.dequantize.batch",
                    || {
                        J2kCudaEngine::new(context)
                            .j2k_dequantize_queued_htj2k_cleanup_with_pool(queued)
                    },
                )
            } else {
                if !collect_stage_timings {
                    return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::contract_violation(
                        "normal CUDA HTJ2K refinement requires retained queued cleanup metadata",
                    ),
                ));
                }
                let mut dequant_budget = HostPhaseBudget::with_live_bytes(
                    "j2k CUDA dequantization target phase",
                    live_host_bytes,
                )?;
                let mut dequant_targets = dequant_budget.try_vec_with_capacity(pending_count)?;
                for work in component_work.iter() {
                    for pending in &work.pending_dequant_bands {
                        if pending.jobs.is_empty() {
                            continue;
                        }
                        let coefficients =
                            pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
                        dequant_targets.push(CudaHtj2kDequantizeTarget {
                            coefficients,
                            jobs: &pending.jobs,
                            output_words: pending.output_words,
                        });
                    }
                }
                context.time_default_stream_named_us_if(
                    collect_stage_timings,
                    "j2k.htj2k.decode.dequantize.batch",
                    || {
                        J2kCudaEngine::new(context)
                        .j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(
                            &dequant_targets,
                            pool,
                            dequant_budget.live_bytes(),
                        )
                    },
                )
            };
            let (stats, dequant_us) = match dequant_result {
                Ok(result) => result,
                Err(error) => {
                    let primary_error = cuda_error(error);
                    if let Some(queued) = queued_cleanup.take() {
                        if let Err(cleanup_error) = queued.finish() {
                            return Err(combine_cuda_cleanup_errors(
                                primary_error,
                                cuda_error(cleanup_error),
                            ));
                        }
                    }
                    return Err(primary_error);
                }
            };
            (stats, dequant_us, dequant_target_count)
        };
        let stage_wall_us = profile::elapsed_us(stage_start);
        {
            let accounting = &mut component_work[accounting_index];
            accounting.timings.h2d = accounting
                .timings
                .h2d
                .saturating_add(stage_wall_us.saturating_sub(dequant_us));
            accounting.timings.dequant = accounting.timings.dequant.saturating_add(dequant_us);
            accounting.timings.dequant_dispatch_count = accounting
                .timings
                .dequant_dispatch_count
                .saturating_add(htj2k_batched_dequant_dispatches(dequant_target_count));
            accounting.dispatches = accounting
                .dispatches
                .saturating_add(stats.kernel_dispatches());
            accounting.decode_dispatches = accounting
                .decode_dispatches
                .saturating_add(stats.decode_kernel_dispatches());
        }
        if let Some(queued) = queued_cleanup.take() {
            queued.finish().map_err(cuda_error)?;
        }

        for work in component_work {
            work.pending_dequant_bands.clear();
        }
        Ok(())
    }
}
