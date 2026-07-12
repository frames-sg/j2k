// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::decode_profile::{
    cuda_idwt_trace_enabled, elapsed_host_us, emit_cuda_idwt_batch_host_trace_row,
    CudaIdwtBatchHostTraceRow, CudaIdwtOutputPoolTraceTotals,
};
use super::super::{
    cuda_error, CudaBufferPool, CudaCoefficientBand, CudaComponentDecodeWork, CudaError,
    CudaHtj2kDecodePlan, CudaHtj2kIdwtStep, CudaJ2kIdwtTarget, CudaQueuedIdwtBatch, Error,
    CUDA_HTJ2K_KERNELS_NOT_READY,
};
use super::helpers::{
    cuda_idwt_job_from_step, cuda_invalid_decode_plan, find_cuda_band, pooled_cuda_buffer,
};
use crate::allocation::{try_collect_cuda_results_exact, HostPhaseBudget};

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn run_cuda_component_idwt_steps(
    context: &j2k_cuda_runtime::CudaContext,
    steps: &[CudaHtj2kIdwtStep],
    work: &mut CudaComponentDecodeWork,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<(), Error> {
    for step in steps {
        let ll = find_cuda_band(&work.bands, step.ll_band_id)?;
        let hl = find_cuda_band(&work.bands, step.hl_band_id)?;
        let lh = find_cuda_band(&work.bands, step.lh_band_id)?;
        let hh = find_cuda_band(&work.bands, step.hh_band_id)?;
        let low_low_device = pooled_cuda_buffer(&ll.buffer)?;
        let high_low_device = pooled_cuda_buffer(&hl.buffer)?;
        let low_high_device = pooled_cuda_buffer(&lh.buffer)?;
        let high_high_device = pooled_cuda_buffer(&hh.buffer)?;
        let job = cuda_idwt_job_from_step(step);
        let (output, idwt_us) = context
            .time_default_stream_named_us_if(collect_stage_timings, "j2k.htj2k.decode.idwt", || {
                if collect_stage_timings {
                    return context.j2k_inverse_dwt_single_device_with_pool(
                        low_low_device,
                        high_low_device,
                        low_high_device,
                        high_high_device,
                        job,
                        pool,
                    );
                }
                context.j2k_inverse_dwt_single_device_untimed_with_pool(
                    low_low_device,
                    high_low_device,
                    low_high_device,
                    high_high_device,
                    job,
                    pool,
                )
            })
            .map_err(cuda_error)?;
        work.timings.idwt = work.timings.idwt.saturating_add(idwt_us);
        let (buffer, stats) = output.into_parts();
        work.dispatches = work.dispatches.saturating_add(stats.kernel_dispatches());
        work.decode_dispatches = work
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
        work.timings.idwt_dispatch_count = work
            .timings
            .idwt_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        work.bands.push(CudaCoefficientBand {
            band_id: step.output_band_id,
            buffer,
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn can_batch_color_idwt(components: &[&CudaHtj2kDecodePlan]) -> bool {
    let Some(first) = components.first() else {
        return false;
    };
    components
        .iter()
        .all(|component| component.idwt_steps().len() == first.idwt_steps().len())
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn run_color_component_idwt_batches(
    context: &j2k_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let (queued_batch, idwt_us) = if collect_stage_timings {
        context
            .time_default_stream_named_us("j2k.htj2k.decode.idwt.batch", || {
                enqueue_color_component_idwt_batches(
                    context,
                    components,
                    component_work,
                    pool,
                    live_host_bytes,
                )
            })
            .map_err(cuda_error)?
    } else {
        // SAFETY: the returned `CudaQueuedIdwtBatch` owns every typed queued
        // guard. Callers retain it through ordered MCT/store completion before
        // resolving or dropping it.
        let queued = unsafe {
            context.submit_default_stream_named("j2k.htj2k.decode.idwt.batch", || {
                enqueue_color_component_idwt_batches(
                    context,
                    components,
                    component_work,
                    pool,
                    live_host_bytes,
                )
            })
        }
        .map_err(cuda_error)?;
        (queued, 0)
    };

    if let Some(accounting) = component_work.first_mut() {
        accounting.timings.idwt = accounting.timings.idwt.saturating_add(idwt_us);
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(queued_batch.kernel_dispatches);
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(queued_batch.decode_dispatches);
        accounting.timings.idwt_dispatch_count = accounting
            .timings
            .idwt_dispatch_count
            .saturating_add(queued_batch.kernel_dispatches);
    }
    if collect_stage_timings {
        queued_batch.finish()?;
        Ok(None)
    } else {
        Ok(Some(queued_batch))
    }
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "IDWT enqueue keeps per-component plan validation and stream ordering together"
)]
fn enqueue_color_component_idwt_batches(
    context: &j2k_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    live_host_bytes: usize,
) -> Result<CudaQueuedIdwtBatch, CudaError> {
    if components.len() != component_work.len() {
        return Err(CudaError::InvalidArgument {
            message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
        });
    }
    let Some(first) = components.first() else {
        return Ok(CudaQueuedIdwtBatch {
            context: context.clone(),
            queued: Vec::new(),
            kernel_dispatches: 0,
            decode_dispatches: 0,
        });
    };

    let mut host_budget =
        HostPhaseBudget::with_cuda_live_bytes("CUDA color IDWT batch metadata", live_host_bytes)?;
    let mut queued = host_budget.try_cuda_vec_with_capacity(1)?;
    let mut kernel_dispatches = 0usize;
    let mut decode_dispatches = 0usize;
    let step_count = first.idwt_steps().len();
    let trace_enabled = cuda_idwt_trace_enabled();
    let enqueue_result = (|| -> Result<(), CudaError> {
        let mut output_pool_trace = CudaIdwtOutputPoolTraceTotals::default();
        let output_alloc_start = trace_enabled.then(std::time::Instant::now);
        for step_index in 0..step_count {
            for (component_index, component) in components.iter().enumerate() {
                let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                    CudaError::InvalidArgument {
                        message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                    }
                })?;
                let output_bytes = {
                    let work = &component_work[component_index];
                    let ll = find_cuda_band(&work.bands, step.ll_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hl = find_cuda_band(&work.bands, step.hl_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let lh = find_cuda_band(&work.bands, step.lh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hh = find_cuda_band(&work.bands, step.hh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    context.j2k_inverse_dwt_single_output_bytes(
                        pooled_cuda_buffer(&ll.buffer).map_err(cuda_invalid_decode_plan)?,
                        pooled_cuda_buffer(&hl.buffer).map_err(cuda_invalid_decode_plan)?,
                        pooled_cuda_buffer(&lh.buffer).map_err(cuda_invalid_decode_plan)?,
                        pooled_cuda_buffer(&hh.buffer).map_err(cuda_invalid_decode_plan)?,
                        cuda_idwt_job_from_step(step),
                    )?
                };
                let buffer = if trace_enabled {
                    let (buffer, trace) = pool.take_with_trace(output_bytes)?;
                    output_pool_trace.add_take(trace);
                    buffer
                } else {
                    pool.take(output_bytes)?
                };
                component_work[component_index]
                    .bands
                    .push(CudaCoefficientBand {
                        band_id: step.output_band_id,
                        buffer,
                    });
            }
        }
        let output_alloc_us = elapsed_host_us(output_alloc_start);

        let target_build_start = trace_enabled.then(std::time::Instant::now);
        let mut target_batches = host_budget.try_cuda_vec_with_capacity(step_count)?;
        for step_index in 0..step_count {
            let targets = try_collect_cuda_results_exact(
                &mut host_budget,
                components
                    .iter()
                    .enumerate()
                    .map(|(component_index, component)| {
                        let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                            CudaError::InvalidArgument {
                                message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                            }
                        })?;
                        let work = &component_work[component_index];
                        let ll = find_cuda_band(&work.bands, step.ll_band_id)
                            .map_err(cuda_invalid_decode_plan)?;
                        let hl = find_cuda_band(&work.bands, step.hl_band_id)
                            .map_err(cuda_invalid_decode_plan)?;
                        let lh = find_cuda_band(&work.bands, step.lh_band_id)
                            .map_err(cuda_invalid_decode_plan)?;
                        let hh = find_cuda_band(&work.bands, step.hh_band_id)
                            .map_err(cuda_invalid_decode_plan)?;
                        let output = find_cuda_band(&work.bands, step.output_band_id)
                            .map_err(cuda_invalid_decode_plan)?;
                        Ok(CudaJ2kIdwtTarget {
                            ll: pooled_cuda_buffer(&ll.buffer).map_err(cuda_invalid_decode_plan)?,
                            hl: pooled_cuda_buffer(&hl.buffer).map_err(cuda_invalid_decode_plan)?,
                            lh: pooled_cuda_buffer(&lh.buffer).map_err(cuda_invalid_decode_plan)?,
                            hh: pooled_cuda_buffer(&hh.buffer).map_err(cuda_invalid_decode_plan)?,
                            output: pooled_cuda_buffer(&output.buffer)
                                .map_err(cuda_invalid_decode_plan)?,
                            job: cuda_idwt_job_from_step(step),
                        })
                    }),
            )?;
            target_batches.push(targets);
        }
        let target_build_us = elapsed_host_us(target_build_start);
        let mut target_slices = host_budget.try_cuda_vec_with_capacity(target_batches.len())?;
        for targets in &target_batches {
            target_slices.push(targets.as_slice());
        }
        let enqueue_start = trace_enabled.then(std::time::Instant::now);
        // SAFETY: `component_work` owns every target allocation and the
        // session-private pool is confined to this default stream. The queued
        // handle remains owned until completion or dependent MCT/store work is
        // submitted and synchronously completed on that stream.
        let queued_execution = unsafe {
            context.j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(
                &target_slices,
                pool,
                host_budget.live_bytes(),
            )?
        };
        let enqueue_us = elapsed_host_us(enqueue_start);
        let execution = queued_execution.execution();
        kernel_dispatches = kernel_dispatches.saturating_add(execution.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(execution.decode_kernel_dispatches());
        queued.push(queued_execution);
        if trace_enabled {
            let row = CudaIdwtBatchHostTraceRow {
                component_count: components.len(),
                step_count,
                output_alloc_us,
                target_build_us,
                enqueue_us,
                output_take_count: output_pool_trace.take_count,
                output_pool_reuse_count: output_pool_trace.reuse_count,
                output_pool_alloc_count: output_pool_trace.alloc_count,
                output_pool_scanned_count: output_pool_trace.scanned_count,
                output_pool_max_free_count: output_pool_trace.max_free_count,
                output_requested_bytes: output_pool_trace.requested_bytes,
            };
            emit_cuda_idwt_batch_host_trace_row(row);
        }
        Ok(())
    })();
    if let Err(error) = enqueue_result {
        if !queued.is_empty() {
            if let Err(completion) = context.synchronize() {
                return Err(CudaError::CompletionFailed {
                    primary: Box::new(error),
                    completion: Box::new(completion),
                });
            }
        }
        return Err(error);
    }

    Ok(CudaQueuedIdwtBatch {
        context: context.clone(),
        queued,
        kernel_dispatches,
        decode_dispatches,
    })
}
