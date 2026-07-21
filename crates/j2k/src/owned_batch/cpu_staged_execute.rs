// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-image staged CPU entropy and tile execution.

use super::cpu_execute::CpuTypedDecodeJob;
use super::{
    execute_staged_entropy_job, finish_staged_plan_samples, finish_staged_tile,
    prepare_staged_entropy_worker, prepare_staged_image, prepare_staged_tile, run_retained_chunks,
    staged_tile_count, BatchCodecRoute, BatchInfrastructureError, BatchWorker, CpuEntropyOutcome,
    CpuGroupFastWorkspace, CpuStagedWorkspace, J2kError, Ordering, PreparedBatchGroup, Range,
    TileBatchOptions, CPU_ENTROPY_IMAGES_PER_WINDOW,
};

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the staged scheduler keeps retained workers, image owners, flattened payloads, and dense output slots explicit"
)]
pub(super) fn run_staged_typed_group<T: Copy + Send>(
    workers: &mut [BatchWorker],
    group: &PreparedBatchGroup,
    tile_options: TileBatchOptions,
    image_jobs: &mut [CpuTypedDecodeJob<'_, '_, T>],
    image_results: &mut [Option<Result<(), J2kError>>],
    flattened: &mut CpuGroupFastWorkspace,
    staged: &mut CpuStagedWorkspace,
    convert: fn(f32, u8) -> T,
) -> Result<(), BatchInfrastructureError> {
    let route = flattened
        .route()
        .ok_or(BatchInfrastructureError::MissingResult { index: 0 })?;
    for window_start in (0..image_jobs.len()).step_by(CPU_ENTROPY_IMAGES_PER_WINDOW) {
        let window_end = window_start
            .saturating_add(CPU_ENTROPY_IMAGES_PER_WINDOW)
            .min(image_jobs.len());
        let image_range = Range {
            start: window_start,
            end: window_end,
        };
        staged.prepare_window(flattened, image_range.clone())?;

        {
            let execution = staged.execution();
            run_retained_chunks(
                workers,
                &mut image_jobs[image_range.clone()],
                &mut image_results[image_range.clone()],
                tile_options,
                |worker, jobs, results| {
                    let _ = worker.prepare_owned_decode();
                    for (job, result) in jobs.iter_mut().zip(results) {
                        let scratch_index = job.slot.checked_sub(window_start).ok_or(
                            BatchInfrastructureError::ResultIndexOutOfBounds {
                                index: job.slot,
                                job_count: image_range.len(),
                            },
                        )?;
                        let scratch = execution.image_scratch.get(scratch_index).ok_or(
                            BatchInfrastructureError::ResultIndexOutOfBounds {
                                index: scratch_index,
                                job_count: execution.image_scratch.len(),
                            },
                        )?;
                        let mut scratch = scratch
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        *result =
                            Some(prepare_staged_image(job.image, route, worker, &mut scratch));
                    }
                    Ok(())
                },
            )?;
            for (local_index, result) in image_results[image_range.clone()].iter().enumerate() {
                if result.as_ref().is_some_and(Result::is_err) {
                    execution.failed[local_index].store(true, Ordering::Release);
                }
            }
        }

        let mut tile_count = 0usize;
        {
            let execution = staged.execution();
            for (local_index, job) in image_jobs[image_range.clone()].iter().enumerate() {
                if image_results[job.slot].as_ref().is_some_and(Result::is_err) {
                    continue;
                }
                match staged_tile_count(job.image, route) {
                    Ok(count) => tile_count = tile_count.max(count),
                    Err(error) => {
                        image_results[job.slot] = Some(Err(error));
                        execution.failed[local_index].store(true, Ordering::Release);
                    }
                }
            }
        }

        for tile_index in 0..tile_count {
            prepare_staged_tile_window(
                workers,
                group,
                staged,
                route,
                tile_options,
                window_start,
                image_range.clone(),
                tile_index,
                image_jobs,
                image_results,
            )?;
            staged.prepare_tile_jobs(flattened, image_range.clone(), tile_index)?;
            let entropy_job_count = execute_staged_tile_entropy(
                workers,
                group,
                staged,
                flattened,
                route,
                tile_options,
                window_start,
                image_range.clone(),
                image_results,
            )?;
            flattened.record_entropy_dispatch(entropy_job_count, image_range.len());
            finish_staged_tile_window(
                workers,
                group,
                staged,
                route,
                tile_options,
                window_start,
                image_range.clone(),
                tile_index,
                image_jobs,
                image_results,
            )?;
        }

        {
            let execution = staged.execution();
            run_retained_chunks(
                workers,
                &mut image_jobs[image_range.clone()],
                &mut image_results[image_range],
                tile_options,
                |_worker, jobs, results| {
                    for (job, result) in jobs.iter_mut().zip(results) {
                        if result.as_ref().is_some_and(Result::is_err) {
                            continue;
                        }
                        let scratch_index = job.slot.checked_sub(window_start).ok_or(
                            BatchInfrastructureError::ResultIndexOutOfBounds {
                                index: job.slot,
                                job_count: execution.image_scratch.len(),
                            },
                        )?;
                        let scratch = execution.image_scratch.get(scratch_index).ok_or(
                            BatchInfrastructureError::ResultIndexOutOfBounds {
                                index: scratch_index,
                                job_count: execution.image_scratch.len(),
                            },
                        )?;
                        let mut scratch = scratch
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        *result = Some(finish_staged_plan_samples(
                            job.image,
                            &group.info,
                            &mut scratch,
                            job.output,
                            convert,
                        ));
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the tile-window preparation boundary keeps retained workers, indexed image outputs, and scratch owners explicit"
)]
fn prepare_staged_tile_window<T: Copy + Send>(
    workers: &mut [BatchWorker],
    _group: &PreparedBatchGroup,
    staged: &mut CpuStagedWorkspace,
    route: BatchCodecRoute,
    tile_options: TileBatchOptions,
    window_start: usize,
    image_range: Range<usize>,
    tile_index: usize,
    image_jobs: &mut [CpuTypedDecodeJob<'_, '_, T>],
    image_results: &mut [Option<Result<(), J2kError>>],
) -> Result<(), BatchInfrastructureError> {
    let execution = staged.execution();
    run_retained_chunks(
        workers,
        &mut image_jobs[image_range.clone()],
        &mut image_results[image_range.clone()],
        tile_options,
        |_worker, jobs, results| {
            for (job, result) in jobs.iter_mut().zip(results) {
                if result.as_ref().is_some_and(Result::is_err) {
                    continue;
                }
                let scratch_index = job.slot.checked_sub(window_start).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: job.slot,
                        job_count: image_range.len(),
                    },
                )?;
                let tile_count = match staged_tile_count(job.image, route) {
                    Ok(count) => count,
                    Err(error) => {
                        execution.failed[scratch_index].store(true, Ordering::Release);
                        *result = Some(Err(error));
                        continue;
                    }
                };
                if tile_index >= tile_count {
                    continue;
                }
                let scratch = execution.image_scratch.get(scratch_index).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: scratch_index,
                        job_count: execution.image_scratch.len(),
                    },
                )?;
                let mut scratch = scratch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Err(error) = prepare_staged_tile(job.image, route, tile_index, &mut scratch)
                {
                    execution.failed[scratch_index].store(true, Ordering::Release);
                    *result = Some(Err(error));
                }
            }
            Ok(())
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the cross-image entropy boundary keeps retained workers, flattened payloads, failure flags, and indexed results explicit"
)]
fn execute_staged_tile_entropy(
    workers: &mut [BatchWorker],
    group: &PreparedBatchGroup,
    staged: &mut CpuStagedWorkspace,
    flattened: &CpuGroupFastWorkspace,
    route: BatchCodecRoute,
    tile_options: TileBatchOptions,
    window_start: usize,
    image_range: Range<usize>,
    image_results: &mut [Option<Result<(), J2kError>>],
) -> Result<usize, BatchInfrastructureError> {
    let execution = staged.execution();
    let entropy_job_count = execution.jobs.len();
    run_retained_chunks(
        workers,
        execution.jobs,
        execution.results,
        tile_options,
        |worker, jobs, results| {
            let _ = worker.prepare_owned_decode();
            let mut workspace_ready = false;
            for (job, result) in jobs.iter_mut().zip(results) {
                let scratch_index = job.image_slot.checked_sub(window_start).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: job.image_slot,
                        job_count: image_range.len(),
                    },
                )?;
                let failed = execution.failed.get(scratch_index).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: scratch_index,
                        job_count: execution.failed.len(),
                    },
                )?;
                if failed.load(Ordering::Acquire) {
                    *result = Some(CpuEntropyOutcome::Complete);
                    continue;
                }
                let image = group.images.get(job.image_slot).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: job.image_slot,
                        job_count: group.images.len(),
                    },
                )?;
                if !workspace_ready {
                    if let Err(error) = prepare_staged_entropy_worker(image, route, worker) {
                        failed.store(true, Ordering::Release);
                        *result = Some(CpuEntropyOutcome::Error(error));
                        continue;
                    }
                    workspace_ready = true;
                }
                let scratch = execution.image_scratch.get(scratch_index).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: scratch_index,
                        job_count: execution.image_scratch.len(),
                    },
                )?;
                let mut scratch = scratch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let outcome =
                    execute_staged_entropy_job(image, *job, flattened, worker, &mut scratch);
                drop(scratch);
                if outcome.is_err() {
                    failed.store(true, Ordering::Release);
                }
                *result = Some(match outcome {
                    Ok(()) => CpuEntropyOutcome::Complete,
                    Err(error) => CpuEntropyOutcome::Error(error),
                });
            }
            Ok(())
        },
    )?;
    let image_result_count = image_results.len();
    for (job, outcome) in execution.jobs.iter().zip(execution.results.iter_mut()) {
        let Some(CpuEntropyOutcome::Error(error)) = outcome.take() else {
            continue;
        };
        let image_result = image_results.get_mut(job.image_slot).ok_or(
            BatchInfrastructureError::ResultIndexOutOfBounds {
                index: job.image_slot,
                job_count: image_result_count,
            },
        )?;
        if image_result.as_ref().is_some_and(Result::is_ok) {
            *image_result = Some(Err(error));
        }
    }
    Ok(entropy_job_count)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the tile Store boundary keeps retained workers, dense output metadata, indexed results, and scratch retirement explicit"
)]
fn finish_staged_tile_window<T: Copy + Send>(
    workers: &mut [BatchWorker],
    group: &PreparedBatchGroup,
    staged: &mut CpuStagedWorkspace,
    route: BatchCodecRoute,
    tile_options: TileBatchOptions,
    window_start: usize,
    image_range: Range<usize>,
    tile_index: usize,
    image_jobs: &mut [CpuTypedDecodeJob<'_, '_, T>],
    image_results: &mut [Option<Result<(), J2kError>>],
) -> Result<(), BatchInfrastructureError> {
    let execution = staged.execution();
    run_retained_chunks(
        workers,
        &mut image_jobs[image_range.clone()],
        &mut image_results[image_range.clone()],
        tile_options,
        |_worker, jobs, results| {
            for (job, result) in jobs.iter_mut().zip(results) {
                if result.as_ref().is_some_and(Result::is_err) {
                    continue;
                }
                let scratch_index = job.slot.checked_sub(window_start).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: job.slot,
                        job_count: execution.image_scratch.len(),
                    },
                )?;
                let tile_count = match staged_tile_count(job.image, route) {
                    Ok(count) => count,
                    Err(error) => {
                        execution.failed[scratch_index].store(true, Ordering::Release);
                        *result = Some(Err(error));
                        continue;
                    }
                };
                if tile_index >= tile_count {
                    continue;
                }
                let scratch = execution.image_scratch.get(scratch_index).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: scratch_index,
                        job_count: execution.image_scratch.len(),
                    },
                )?;
                let mut scratch = scratch
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Err(error) = finish_staged_tile(
                    job.image,
                    route,
                    tile_index,
                    group.info.signed,
                    &mut scratch,
                ) {
                    execution.failed[scratch_index].store(true, Ordering::Release);
                    *result = Some(Err(error));
                }
            }
            Ok(())
        },
    )
}
