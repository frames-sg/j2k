// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU homogeneous-group execution and final materialization.

use super::cpu_materialize::{
    convert_u16, convert_u8, decode_image_i16, decode_image_u16, decode_image_u8,
    decode_lone_gray_u8, ensure_batch_output_within_cap, round_signed, try_zeroed_vec,
};
use super::cpu_staged_execute::run_staged_typed_group;
use super::{
    decode_warnings_for_recovery, retained_inner_parallelism, run_retained_chunks, Arc,
    BatchCodecRoute, BatchColor, BatchDecodeOptions, BatchErrorStage, BatchGroupInfo,
    BatchInfrastructureError, BatchItemError, BatchWorker, CpuBatchGroup, CpuBatchSamples,
    CpuDecodeParallelism, CpuGroupFastWorkspace, CpuStagedWorkspace, IndexedBatchError, J2kError,
    NativeSampleType, NonZeroUsize, PreparedBatchGroup, PreparedImage, TileBatchOptions, Vec,
};
use crate::batch::admission::{hold_at_least, BatchAllocationBudget};
use crate::batch::allocation::{batch_execution_cap_bytes, try_vec_with_capacity};

#[expect(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the typed output allocation and one shared decode/compaction boundary remain together so every native sample type follows the same group-level ownership rules; retained workspaces and admission stay explicit"
)]
pub(super) fn decode_cpu_group(
    workers: &mut [BatchWorker],
    fast_workspace: &mut CpuGroupFastWorkspace,
    staged_workspace: &mut CpuStagedWorkspace,
    stream_pool: &mut CpuStreamPool,
    group: &PreparedBatchGroup,
    options: BatchDecodeOptions,
    workers_option: Option<NonZeroUsize>,
    admission: &Arc<BatchAllocationBudget>,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<Option<CpuBatchGroup>, BatchInfrastructureError> {
    let tile_options = TileBatchOptions::new(workers_option);
    // The staged route flattens entropy work across images for a few serial
    // workers. When workers may parallelize inside one image, each prepared
    // plan decodes faster on its own: its code-block rows and wavelet bands
    // spread over every core, still without reparsing.
    // Streamed images never hold whole-image scratch, so a group with one
    // enters the staged route only to keep its plans.
    let streams_rows = !keeps_staged_plans(group, workers_option)
        && group
            .images
            .iter()
            .any(|image| image.cpu_execution().streams());
    let has_flattened_payloads = !streams_rows
        && fast_workspace.prepare_group(group)?
        && retained_inner_parallelism(group.images.len(), tile_options)?
            == CpuDecodeParallelism::Serial;
    // With an explicit worker count, the images of such a group decode in a
    // pool of that many threads (see `CpuStreamPool`).
    let stream_pool = if streams_rows
        && group.images.len() > 1
        && workers_option.is_some_and(|count| count.get() > 1)
    {
        Some(stream_pool.get(workers.len())?)
    } else {
        None
    };
    let samples_per_image =
        group
            .info
            .samples_per_image()
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K owned batch output",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    let sample_count = samples_per_image.checked_mul(group.images.len()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K owned batch output",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    ensure_batch_output_within_cap(sample_count, group.info.sample_type)?;

    let (samples, successful_slots, copied_samples) = match group.info.sample_type {
        NativeSampleType::U8 => {
            let mut values = try_zeroed_vec::<u8>(sample_count, 0)?;
            let (successful_slots, copied_samples) = run_typed_group(
                workers,
                group,
                options,
                tile_options,
                samples_per_image,
                &mut values,
                has_flattened_payloads.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                admission,
                // The only image of a Gray8 group is the whole in-flight set.
                if group.images.len() == 1 && group.info.color == BatchColor::Gray {
                    decode_lone_gray_u8
                } else {
                    decode_image_u8
                },
                convert_u8,
                errors,
            )?;
            (
                CpuBatchSamples::U8(values),
                successful_slots,
                copied_samples,
            )
        }
        NativeSampleType::U16 => {
            let mut values = try_zeroed_vec::<u16>(sample_count, 0)?;
            let (successful_slots, copied_samples) = run_typed_group(
                workers,
                group,
                options,
                tile_options,
                samples_per_image,
                &mut values,
                has_flattened_payloads.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                admission,
                decode_image_u16,
                convert_u16,
                errors,
            )?;
            (
                CpuBatchSamples::U16(values),
                successful_slots,
                copied_samples,
            )
        }
        NativeSampleType::I16 => {
            let mut values = try_zeroed_vec::<i16>(sample_count, 0)?;
            let (successful_slots, copied_samples) = run_typed_group(
                workers,
                group,
                options,
                tile_options,
                samples_per_image,
                &mut values,
                has_flattened_payloads.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                admission,
                decode_image_i16,
                round_signed,
                errors,
            )?;
            (
                CpuBatchSamples::I16(values),
                successful_slots,
                copied_samples,
            )
        }
        _ => {
            return Err(BatchInfrastructureError::UnsupportedContract {
                what: "owned CPU batch sample type",
            });
        }
    };
    fast_workspace.record_output_group(copied_samples)?;
    if successful_slots.is_empty() {
        return Ok(None);
    }
    let successful_count = successful_slots.len();
    let mut source_indices =
        try_vec_with_capacity(successful_count, "J2K owned batch source indices")?;
    let mut decoded_rects =
        try_vec_with_capacity(successful_count, "J2K owned batch decoded rectangles")?;
    let mut warnings = try_vec_with_capacity(successful_count, "J2K owned batch warnings")?;
    for &slot in &successful_slots {
        source_indices.push(group.source_indices[slot]);
        decoded_rects.push(group.images[slot].plan().output_rect());
        warnings.push(decode_warnings_for_recovery(
            group.images[slot].used_lenient_metadata_recovery(),
        ));
    }
    Ok(Some(CpuBatchGroup::new(
        group.info.clone(),
        source_indices,
        decoded_rects,
        warnings,
        samples,
    )))
}

/// Most explicit workers for which a group whose images may stream keeps
/// their plans on the staged route (see [`keeps_staged_plans`]): one, or
/// four for Gray8, whose staged route keeps up to four workers busy.
const fn staged_plans_max_workers(color: BatchColor) -> usize {
    match color {
        BatchColor::Gray => 4,
        _ => 1,
    }
}

#[cfg(test)]
std::thread_local! {
    /// Test-only planned bytes above which a group streams instead of
    /// keeping its plans on the staged route, on this thread.
    pub(super) static STAGED_PLANS_CAP_OVERRIDE: core::cell::Cell<Option<usize>> =
        const { core::cell::Cell::new(None) };
}

/// Whether a group of images that may stream decodes their plans on the
/// staged route instead: the caller set one worker (up to four for Gray8),
/// the group holds several classic images, and each has a plan within the
/// per-decode bound, all together within the batch execution cap. Larger
/// groups stream.
///
/// The staged route decodes each code-block of a window's images back to
/// back, which speeds up classic Tier-1 when the images repeat, and while it
/// keeps its workers busy the stream pool has no idle thread to gain from:
/// 8 copies of one image stream in 1.13-1.32 times the staged time at one
/// worker, and 8 Gray8 copies in 1.12-1.24 times at two to four workers.
/// Distinct images stream as fast or faster, RGB8 streams faster from two
/// workers on and Gray8 from six, and HTJ2K images always stream faster.
/// The window keeps each image's plan scratch, as these settings did before
/// images streamed.
fn keeps_staged_plans(group: &PreparedBatchGroup, workers: Option<NonZeroUsize>) -> bool {
    group.info.route == BatchCodecRoute::Classic
        && group.images.len() > 1
        && workers
            .is_some_and(|workers| workers.get() <= staged_plans_max_workers(group.info.color))
        && group
            .images
            .iter()
            .try_fold(0usize, |total, image| {
                total.checked_add(image.cpu_execution().plan_bytes()?)
            })
            .is_some_and(|total| total <= staged_plans_cap())
}

fn staged_plans_cap() -> usize {
    #[cfg(test)]
    if let Some(cap) = STAGED_PLANS_CAP_OVERRIDE.with(core::cell::Cell::get) {
        return cap;
    }
    batch_execution_cap_bytes()
}

/// The threads a group's streamed images decode on when the caller set the
/// worker count, created on first use and kept by the session.
///
/// A streamed image hands its Tier-1 bands to the rayon pool it decodes in.
/// An explicit worker count used to keep each image serial on its worker
/// thread instead, so `M` images on `N` workers kept
/// `ceil(M / ceil(M / N))` threads busy (8 images on 6 workers: 4) and the
/// last images alone at the end. In a pool of exactly `N` threads idle
/// threads take bands of the images still in flight, so the group keeps all
/// `N` busy and never runs more: the worker threads only wait for their
/// images, and the images in flight stay one per worker.
#[derive(Default)]
pub(super) struct CpuStreamPool {
    pool: Option<rayon::ThreadPool>,
}

impl CpuStreamPool {
    fn get(&mut self, threads: usize) -> Result<&rayon::ThreadPool, BatchInfrastructureError> {
        let pool = match self.pool.take() {
            Some(pool) if pool.current_num_threads() == threads => pool,
            _ => rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("j2k-batch-stream-{index}"))
                .build()
                .map_err(|_| BatchInfrastructureError::WorkerSpawnFailed { worker: 0 })?,
        };
        Ok(self.pool.insert(pool))
    }

    /// Threads of the pool, once a group has used it.
    #[cfg(test)]
    pub(super) fn threads(&self) -> Option<usize> {
        self.pool
            .as_ref()
            .map(rayon::ThreadPool::current_num_threads)
    }
}

pub(super) struct CpuTypedDecodeJob<'image, 'output, T> {
    pub(super) slot: usize,
    pub(super) image: &'image PreparedImage,
    pub(super) output: &'output mut [T],
}

type CpuDecodeFn<T> = for<'image> fn(
    &'image PreparedImage,
    BatchDecodeOptions,
    &BatchGroupInfo,
    CpuDecodeParallelism,
    &mut j2k_native::DecoderContext<'image>,
    &mut BatchWorker,
    &mut [T],
) -> Result<(), J2kError>;

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the retained-worker boundary keeps output ownership, decode policy, and indexed error collection explicit, with the job slots, the worker loop, and the result collection in one place"
)]
fn run_typed_group<T: Copy + Send>(
    workers: &mut [BatchWorker],
    group: &PreparedBatchGroup,
    options: BatchDecodeOptions,
    tile_options: TileBatchOptions,
    samples_per_image: usize,
    output: &mut Vec<T>,
    flattened_payloads: Option<&mut CpuGroupFastWorkspace>,
    staged_workspace: &mut CpuStagedWorkspace,
    stream_pool: Option<&rayon::ThreadPool>,
    admission: &Arc<BatchAllocationBudget>,
    decode: CpuDecodeFn<T>,
    convert: fn(f32, u8) -> T,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<(Vec<usize>, usize), BatchInfrastructureError> {
    let mut jobs = try_vec_with_capacity(group.images.len(), "J2K owned batch decode jobs")?;
    for (slot, (image, output)) in group
        .images
        .iter()
        .zip(output.chunks_exact_mut(samples_per_image))
        .enumerate()
    {
        jobs.push(CpuTypedDecodeJob {
            slot,
            image,
            output,
        });
    }
    let mut results = try_vec_with_capacity(jobs.len(), "J2K owned batch decode result slots")?;
    results.resize_with(jobs.len(), || None);
    if let Some(flattened) = flattened_payloads {
        run_staged_typed_group(
            workers,
            group,
            tile_options,
            &mut jobs,
            &mut results,
            flattened,
            staged_workspace,
            admission,
            convert,
        )?;
    } else {
        run_retained_chunks(
            workers,
            &mut jobs,
            &mut results,
            tile_options,
            |worker, jobs, results| {
                let parallelism = worker.prepare_owned_decode();
                let workspace = worker.take_native_workspace();
                let mut context = j2k_native::DecoderContext::from_workspace(workspace);
                let mut claim = None;
                for (job, slot) in jobs.iter_mut().zip(results) {
                    hold_at_least(
                        admission,
                        &mut claim,
                        job.image.cpu_execution().claim_bytes(),
                    )?;
                    *slot = Some(match stream_pool {
                        // This thread waits while a pool thread decodes the
                        // image and idle ones take its bands.
                        Some(pool) => pool.install(|| {
                            decode(
                                job.image,
                                options,
                                &group.info,
                                CpuDecodeParallelism::Auto,
                                &mut context,
                                worker,
                                job.output,
                            )
                        }),
                        None => decode(
                            job.image,
                            options,
                            &group.info,
                            parallelism,
                            &mut context,
                            worker,
                            job.output,
                        ),
                    });
                }
                worker.restore_native_workspace(context.into_workspace());
                Ok(())
            },
        )?;
    }

    let mut successful_slots = try_vec_with_capacity(
        group.images.len(),
        "J2K owned batch successful result slots",
    )?;
    for (slot_index, (source_index, result)) in group
        .source_indices
        .iter()
        .copied()
        .zip(results)
        .enumerate()
    {
        match result {
            Some(Ok(())) => successful_slots.push(slot_index),
            Some(Err(source)) => {
                errors.push(IndexedBatchError {
                    index: source_index,
                    source: BatchItemError::Codec {
                        stage: BatchErrorStage::Decode,
                        source: Arc::new(source),
                    },
                });
            }
            None => {
                return Err(BatchInfrastructureError::MissingResult {
                    index: source_index,
                })
            }
        }
    }
    let mut copied_samples = 0usize;
    if successful_slots.len() != group.images.len() {
        for (destination_slot, &source_slot) in successful_slots.iter().enumerate() {
            if destination_slot == source_slot {
                continue;
            }
            let source_start = source_slot * samples_per_image;
            let source_end = source_start + samples_per_image;
            let destination_start = destination_slot * samples_per_image;
            output.copy_within(source_start..source_end, destination_start);
            copied_samples = copied_samples.saturating_add(samples_per_image);
        }
        output.truncate(successful_slots.len() * samples_per_image);
    }
    Ok((successful_slots, copied_samples))
}
