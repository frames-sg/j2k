// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU homogeneous-group execution and final materialization.

use super::cpu_fast::classic_staging_fits;
use super::cpu_materialize::{
    convert_u16, convert_u8, decode_image_i16, decode_image_u16, decode_image_u8,
    decode_lone_gray_u8, ensure_batch_output_within_cap, round_signed, try_zeroed_vec,
};
use super::cpu_staged_execute::run_staged_typed_group;
use super::{
    decode_warnings_for_recovery, retained_inner_parallelism, retained_worker_count,
    run_retained_chunks, Arc, BatchCodecRoute, BatchColor, BatchDecodeOptions, BatchErrorStage,
    BatchGroupInfo, BatchInfrastructureError, BatchItemError, BatchWorker, CpuBatchGroup,
    CpuBatchSamples, CpuDecodeParallelism, CpuGroupFastWorkspace, CpuPreparedImage,
    CpuStagedWorkspace, IndexedBatchError, J2kError, NativeSampleType, NonZeroUsize,
    PreparedBatchGroupRef, TileBatchOptions, Vec,
};
use crate::batch::allocation::try_vec_with_capacity;

#[expect(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "the typed output allocation and one shared decode/compaction boundary remain together so every native sample type follows the same group-level ownership rules and retained workspaces stay explicit"
)]
pub(super) fn decode_cpu_group(
    workers: &mut [BatchWorker],
    fast_workspace: &mut CpuGroupFastWorkspace,
    staged_workspace: &mut CpuStagedWorkspace,
    stream_pool: &mut CpuStreamPool,
    group: PreparedBatchGroupRef<'_>,
    options: BatchDecodeOptions,
    workers_option: Option<NonZeroUsize>,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<Option<CpuBatchGroup>, BatchInfrastructureError> {
    if group.images.iter().any(|image| image.control().is_some()) {
        for worker in workers.iter_mut() {
            worker.release_owned_allocations();
        }
        fast_workspace.release_allocations();
        *staged_workspace = CpuStagedWorkspace::default();
    }
    let tile_options = TileBatchOptions::new(workers_option);
    // The staged route flattens entropy work across images for a few serial
    // workers. When workers may parallelize inside one image, each prepared
    // plan decodes faster on its own: its code-block rows and wavelet bands
    // spread over every core, still without reparsing. A group with an image
    // whose rows stream never stages: streamed images hold a band of rows
    // instead of whole-image scratch, at every worker count.
    let streams_rows = group
        .images
        .iter()
        .any(|image| image.cpu_execution().streams());
    // Payloads are flattened only for the staged route, and only within the
    // flattening and staged-window caps (see `classic_staging_fits`): a
    // classic group beyond them decodes its plans one image at a time
    // instead of failing.
    let staged = !streams_rows
        && group.images.iter().all(|image| image.control().is_none())
        && group
            .images
            .iter()
            .all(|image| image.cpu_execution().plan_bytes().is_some())
        && retained_inner_parallelism(group.images.len(), tile_options)?
            == CpuDecodeParallelism::Serial
        && (group.info.route != BatchCodecRoute::Classic
            || classic_staging_fits(group.images.iter())?)
        && fast_workspace.prepare_group(group)?;
    let (pooled, image_parallelism) =
        image_policy(group.images.len(), streams_rows, workers_option);
    let stream_pool = if pooled {
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
    ensure_batch_output_within_cap(group.info, group.images.len())?;
    // A controlled request has its own group, so refuse its output before
    // allocating it. Actual capacity is reconciled below before decoding.
    if let [image] = group.images {
        if let Err(source) =
            image.check_owned_output(group.info.output_bytes(1).unwrap_or(usize::MAX))
        {
            errors.push(IndexedBatchError {
                index: group.source_indices[0],
                source: BatchItemError::Codec {
                    stage: BatchErrorStage::Decode,
                    source: Arc::new(source),
                },
            });
            return Ok(None);
        }
    }

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
                staged.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                image_parallelism,
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
                staged.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                image_parallelism,
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
                staged.then_some(&mut *fast_workspace),
                staged_workspace,
                stream_pool,
                image_parallelism,
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
        decoded_rects.push(group.images[slot].output_rect());
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

/// How the images of a group that decode one at a time run: in the stream
/// pool, and with which inner parallelism otherwise (`None`: the
/// scheduler's own). With an explicit worker count and several images,
/// streamed images decode in the pool of that many threads (two or more) and
/// plans serially, also on the worker itself, where offset-plan executions
/// read it; a single worker decodes serially.
pub(super) fn image_policy(
    image_count: usize,
    streamed: bool,
    workers: Option<NonZeroUsize>,
) -> (bool, Option<CpuDecodeParallelism>) {
    let explicit_group = image_count > 1 && workers.is_some();
    let pooled = explicit_group && streamed && workers.is_some_and(|count| count.get() > 1);
    (
        pooled,
        explicit_group.then_some(CpuDecodeParallelism::Serial),
    )
}

/// Whether the images of a group of `job_count` that decode on a stream pool
/// of `pool_threads` share it (see `DecoderContext::set_shared_pool`): the
/// batch plans at least as many workers at once as the pool has threads, so
/// each thread starts with an image of its own and an idle one takes the
/// bands of other images still in flight. The images that finish last keep
/// those rings with fewer images beside them. With fewer workers planned
/// than threads, each image holds the rings of a decode alone on the pool.
pub(super) fn shares_stream_pool(
    job_count: usize,
    tile_options: TileBatchOptions,
    pool_threads: usize,
) -> Result<bool, BatchInfrastructureError> {
    Ok(retained_worker_count(job_count, tile_options)? >= pool_threads)
}

/// The encoded bytes up to which an image of a group that shares its stream
/// pool holds the shared rings, from the encoded lengths of the group's
/// images: one and a half times their mean. Tier-1 work follows the coded
/// bytes, so a larger image is the likeliest to decode last with few other
/// images beside it, and keeps the rings of a decode alone on the pool.
pub(super) fn shared_ring_bytes(lengths: impl ExactSizeIterator<Item = usize>) -> usize {
    let count = lengths.len().max(1);
    let mean = lengths.fold(0_usize, usize::saturating_add) / count;
    mean.saturating_add(mean / 2)
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
    pub(super) image: &'image CpuPreparedImage,
    pub(super) output: &'output mut [T],
}

type CpuDecodeFn<T> = for<'image> fn(
    &'image CpuPreparedImage,
    BatchDecodeOptions,
    &BatchGroupInfo,
    CpuDecodeParallelism,
    &mut j2k_native::DecoderContext<'image>,
    &mut BatchWorker,
    (&mut [T], usize),
) -> Result<(), J2kError>;

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the retained-worker boundary keeps output ownership, decode policy, and indexed error collection explicit, with the job slots, the worker loop, and the result collection in one place"
)]
fn run_typed_group<T: Copy + Send>(
    workers: &mut [BatchWorker],
    group: PreparedBatchGroupRef<'_>,
    options: BatchDecodeOptions,
    tile_options: TileBatchOptions,
    samples_per_image: usize,
    output: &mut Vec<T>,
    flattened_payloads: Option<&mut CpuGroupFastWorkspace>,
    staged_workspace: &mut CpuStagedWorkspace,
    stream_pool: Option<&rayon::ThreadPool>,
    image_parallelism: Option<CpuDecodeParallelism>,
    decode: CpuDecodeFn<T>,
    convert: fn(f32, u8) -> T,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<(Vec<usize>, usize), BatchInfrastructureError> {
    let output_bytes = output.capacity().saturating_mul(core::mem::size_of::<T>());
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
            convert,
        )?;
    } else {
        let shared_pool = match stream_pool {
            Some(pool) => shares_stream_pool(jobs.len(), tile_options, pool.current_num_threads())?,
            None => false,
        };
        let shared_bytes = shared_ring_bytes(group.images.iter().map(|image| image.bytes().len()));
        run_retained_chunks(
            workers,
            &mut jobs,
            &mut results,
            tile_options,
            |worker, jobs, results| {
                // On the worker itself, where offset-plan executions read it.
                if let Some(parallelism) = image_parallelism {
                    worker.set_owned_inner_parallelism(parallelism);
                }
                let parallelism = worker.prepare_owned_decode();
                let workspace = worker.take_native_workspace();
                let mut context = j2k_native::DecoderContext::from_workspace(workspace);
                for (job, slot) in jobs.iter_mut().zip(results) {
                    context.set_shared_pool(shared_pool && job.image.bytes().len() <= shared_bytes);
                    let owned_output_bytes = if job.image.control().is_some() {
                        output_bytes
                    } else {
                        0
                    };
                    let default_control = crate::J2kDecodeControl::default();
                    let control = job.image.control().unwrap_or(&default_control);
                    *slot = Some(job.image.check_owned_output(output_bytes).and_then(|()| {
                        control.with_native_context(&mut context, |context| {
                            let native = context
                                .operation_control()
                                .clone()
                                .beside_retained_bytes(
                                    job.image
                                        .retained_bytes()
                                        .saturating_add(owned_output_bytes),
                                )
                                .map_err(|error| control.native_error(error))?;
                            context.with_operation_control(native, |context| match stream_pool {
                                // This thread waits while a pool thread decodes
                                // the image and idle ones take its bands.
                                Some(pool) => pool.install(|| {
                                    decode(
                                        job.image,
                                        options,
                                        group.info,
                                        CpuDecodeParallelism::Auto,
                                        context,
                                        worker,
                                        (job.output, owned_output_bytes),
                                    )
                                }),
                                None => decode(
                                    job.image,
                                    options,
                                    group.info,
                                    parallelism,
                                    context,
                                    worker,
                                    (job.output, owned_output_bytes),
                                ),
                            })
                        })
                    }));
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
