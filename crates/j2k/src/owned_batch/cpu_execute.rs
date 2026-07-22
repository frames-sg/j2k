// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU homogeneous-group execution and final materialization.

use super::cpu_materialize::{
    convert_u16, convert_u8, decode_image_i16, decode_image_u16, decode_image_u8,
    ensure_batch_output_within_cap, round_signed, try_zeroed_vec,
};
use super::cpu_staged_execute::run_staged_typed_group;
use super::{
    decode_warnings_for_settings, run_retained_chunks, Arc, BatchDecodeOptions, BatchErrorStage,
    BatchGroupInfo, BatchInfrastructureError, BatchItemError, BatchWorker, CpuBatchGroup,
    CpuBatchSamples, CpuDecodeParallelism, CpuGroupFastWorkspace, CpuStagedWorkspace,
    IndexedBatchError, J2kError, NativeSampleType, NonZeroUsize, PreparedBatchGroup, PreparedImage,
    TileBatchOptions, Vec,
};

#[expect(
    clippy::too_many_lines,
    reason = "the typed output allocation and one shared decode/compaction boundary remain together so every native sample type follows the same group-level ownership rules"
)]
pub(super) fn decode_cpu_group(
    workers: &mut [BatchWorker],
    fast_workspace: &mut CpuGroupFastWorkspace,
    staged_workspace: &mut CpuStagedWorkspace,
    group: &PreparedBatchGroup,
    options: BatchDecodeOptions,
    workers_option: Option<NonZeroUsize>,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<Option<CpuBatchGroup>, BatchInfrastructureError> {
    let has_flattened_payloads = fast_workspace.prepare_group(group)?;
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

    let tile_options = TileBatchOptions::new(workers_option);

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
                decode_image_u8,
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
    let source_indices = successful_slots
        .iter()
        .map(|&slot| group.source_indices[slot])
        .collect();
    let decoded_rects = successful_slots
        .iter()
        .map(|&slot| group.images[slot].plan().output_rect())
        .collect();
    let warnings = successful_slots
        .iter()
        .map(|_| decode_warnings_for_settings(options.settings))
        .collect();
    Ok(Some(CpuBatchGroup::new(
        group.info.clone(),
        source_indices,
        decoded_rects,
        warnings,
        samples,
    )))
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
    reason = "the retained-worker boundary keeps output ownership, decode policy, and indexed error collection explicit"
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
    decode: CpuDecodeFn<T>,
    convert: fn(f32, u8) -> T,
    errors: &mut Vec<IndexedBatchError>,
) -> Result<(Vec<usize>, usize), BatchInfrastructureError> {
    let mut jobs = group
        .images
        .iter()
        .zip(output.chunks_exact_mut(samples_per_image))
        .enumerate()
        .map(|(slot, (image, output))| CpuTypedDecodeJob {
            slot,
            image,
            output,
        })
        .collect::<Vec<_>>();
    let mut results = Vec::with_capacity(jobs.len());
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
        run_retained_chunks(
            workers,
            &mut jobs,
            &mut results,
            tile_options,
            |worker, jobs, results| {
                let parallelism = worker.prepare_owned_decode();
                let workspace = worker.take_native_workspace();
                let mut context = j2k_native::DecoderContext::from_workspace(workspace);
                for (job, slot) in jobs.iter_mut().zip(results) {
                    *slot = Some(decode(
                        job.image,
                        options,
                        &group.info,
                        parallelism,
                        &mut context,
                        worker,
                        job.output,
                    ));
                }
                worker.restore_native_workspace(context.into_workspace());
                Ok(())
            },
        )?;
    }

    let mut successful_slots = Vec::with_capacity(group.images.len());
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
