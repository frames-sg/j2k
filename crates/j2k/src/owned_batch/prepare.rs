// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concurrent inspection, representability validation, and homogeneous grouping.

use super::{
    run_retained_chunks, size_of, Arc, BatchDecodeOptions, BatchInfrastructureError,
    BatchItemError, BatchWorker, EncodedImage, IndexedBatchError, NonZeroUsize, PrepareImageResult,
    PrepareJob, PreparedBatch, PreparedBatchGroup, PreparedImage, TileBatchOptions, Vec,
    J2K_BATCH_METADATA_ALLOWANCE_BYTES, MAX_GENERIC_BATCH_WORKERS,
};

mod image;
use self::image::{
    batch_execution_shape, batch_group_info, prepare_image, reconcile_codec_plan_metadata,
};

/// Inspect, validate, and group owned encoded images without decoding pixels.
///
/// Per-image failures are retained in [`PreparedBatch::errors`]. The outer
/// error is reserved for batch infrastructure and allocation failures.
pub fn prepare_batch(
    inputs: Vec<EncodedImage>,
    options: BatchDecodeOptions,
) -> Result<PreparedBatch, BatchInfrastructureError> {
    let input_count = inputs.len();
    let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let worker_count = options
        .workers
        .map_or(available, NonZeroUsize::get)
        .clamp(1, MAX_GENERIC_BATCH_WORKERS)
        .min(input_count.max(1));
    let mut workers = try_prepare_vec(worker_count, "J2K preparation workers")?;
    for _ in 0..worker_count {
        workers.push(BatchWorker::new_owned(input_count.max(2)));
    }
    prepare_batch_with_workers(inputs, options, &mut workers)
}

pub(super) fn prepare_batch_with_workers(
    inputs: Vec<EncodedImage>,
    options: BatchDecodeOptions,
    workers: &mut [BatchWorker],
) -> Result<PreparedBatch, BatchInfrastructureError> {
    prepare_staging_bytes(inputs.len())?;
    let mut groups: Vec<PreparedBatchGroup> = Vec::new();
    let mut errors = Vec::new();
    let input_count = inputs.len();
    let mut jobs = try_prepare_vec(input_count, "J2K preparation jobs")?;
    for (source_index, input) in inputs.into_iter().enumerate() {
        jobs.push(PrepareJob {
            source_index,
            input: Some(input),
        });
    }
    let mut prepared = try_prepare_vec(input_count, "J2K preparation result slots")?;
    prepared.resize_with(input_count, || None);
    run_retained_chunks(
        workers,
        &mut jobs,
        &mut prepared,
        TileBatchOptions::new(options.workers),
        |worker, jobs, results| {
            worker.prepare_owned_decode();
            for (job, slot) in jobs.iter_mut().zip(results) {
                let input = job
                    .input
                    .take()
                    .ok_or(BatchInfrastructureError::MissingResult {
                        index: job.source_index,
                    })?;
                *slot = Some(prepare_image(input, job.source_index, options, worker));
            }
            Ok(())
        },
    )?;
    drop(jobs);

    for (source_index, result) in prepared.into_iter().enumerate() {
        let Some(result) = result else {
            return Err(BatchInfrastructureError::MissingResult {
                index: source_index,
            });
        };
        push_prepared_result(&mut groups, &mut errors, source_index, result, options)?;
    }

    Ok(PreparedBatch {
        groups: Arc::from(groups),
        errors: Arc::from(errors),
        options,
    })
}

/// Regroup caller-supplied prepared images without parsing or copying codestreams.
///
/// Returned source indices are positions in `images`, not
/// [`PreparedImage::source_index`], which remains the index from the image's
/// original preparation call. Group order follows first occurrence and image
/// order is stable within each group. A strict/lenient policy mismatch is an
/// indexed preflight error; output layout and CPU worker policy may change.
/// Errors from earlier batches are not carried because only the supplied
/// successful prepared images participate in this new batch.
pub fn prepare_batch_from_images(
    images: Vec<PreparedImage>,
    options: BatchDecodeOptions,
) -> Result<PreparedBatch, BatchInfrastructureError> {
    prepare_staging_bytes(images.len())?;
    let mut groups = Vec::new();
    let mut errors = Vec::new();
    for (source_index, image) in images.into_iter().enumerate() {
        let result = regroup_prepared_image(image, options);
        push_prepared_result(&mut groups, &mut errors, source_index, result, options)?;
    }
    Ok(PreparedBatch {
        groups: Arc::from(groups),
        errors: Arc::from(errors),
        options,
    })
}

fn regroup_prepared_image(image: PreparedImage, options: BatchDecodeOptions) -> PrepareImageResult {
    if image.decode_settings() != options.settings {
        return Err(BatchItemError::PreparedDecodeSettingsMismatch {
            prepared: image.decode_settings(),
            requested: options.settings,
        });
    }
    let mut info = batch_group_info(image.support(), image.plan(), options.layout)?;
    reconcile_codec_plan_metadata(&mut info, image.codec_plan())?;
    let execution_shape =
        batch_execution_shape(image.support(), image.plan(), image.preparation_depth());
    Ok((image, info, execution_shape))
}

fn push_prepared_result(
    groups: &mut Vec<PreparedBatchGroup>,
    errors: &mut Vec<IndexedBatchError>,
    source_index: usize,
    result: PrepareImageResult,
    options: BatchDecodeOptions,
) -> Result<(), BatchInfrastructureError> {
    match result {
        Ok((image, info, execution_shape)) => {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.info == info && group.execution_shape == execution_shape)
            {
                try_reserve_one(
                    &mut group.source_indices,
                    "J2K prepared group source indices",
                )?;
                try_reserve_one(&mut group.images, "J2K prepared group images")?;
                group.source_indices.push(source_index);
                group.images.push(image);
            } else {
                let mut images = try_prepare_vec(1, "J2K prepared group images")?;
                images.push(image);
                let mut source_indices = try_prepare_vec(1, "J2K prepared group source indices")?;
                source_indices.push(source_index);
                try_reserve_one(groups, "J2K prepared groups")?;
                groups.push(PreparedBatchGroup {
                    info,
                    options,
                    execution_shape,
                    images,
                    source_indices,
                });
            }
        }
        Err(source) => {
            try_reserve_one(errors, "J2K prepared indexed errors")?;
            errors.push(IndexedBatchError {
                index: source_index,
                source,
            });
        }
    }
    Ok(())
}

pub(super) fn prepare_staging_bytes(input_count: usize) -> Result<usize, BatchInfrastructureError> {
    let execution_item_bytes = size_of::<PrepareJob>()
        .checked_add(size_of::<Option<PrepareImageResult>>())
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K preparation staging",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let successful_output_bytes = size_of::<PreparedBatchGroup>()
        .checked_add(size_of::<PreparedImage>())
        .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K preparation staging",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let output_item_bytes = size_of::<Option<PrepareImageResult>>()
        .checked_add(successful_output_bytes.max(size_of::<IndexedBatchError>()))
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K preparation staging",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let item_bytes = execution_item_bytes.max(output_item_bytes);
    let bytes = input_count.checked_mul(item_bytes).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K preparation staging",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    if bytes > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K preparation staging",
            requested: bytes,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    Ok(bytes)
}

fn try_prepare_vec<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchInfrastructureError> {
    let bytes = capacity.checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    if bytes > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: bytes,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    j2k_core::try_host_vec_with_capacity(capacity).map_err(|error| {
        BatchInfrastructureError::HostAllocationFailed {
            what,
            bytes: error.requested_bytes(),
        }
    })
}

fn try_reserve_one<T>(
    values: &mut Vec<T>,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let requested_items =
        values
            .len()
            .checked_add(1)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
            })?;
    let requested = requested_items.checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    if requested > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    values
        .try_reserve_exact(1)
        .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
            what,
            bytes: requested,
        })
}
