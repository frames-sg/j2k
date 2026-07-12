// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    batch_component_groups, dct_blocks_to_8x8_f64, decompose_97_from_first_level,
    float97_batch_component_groups, float_direct_97_wavelet_from_component,
    integer_dct_job_for_component, integer_direct_wavelet_from_component,
    integer_wavelet_from_first_level, store_float97_batch_wavelet, store_integer_batch_wavelet,
    try_store_grouped_i16_preencoded_float97_batches, try_store_prequantized_float97_batch_group,
    validate_component_block_grid, BatchComponentRef, ComponentWavelet97,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt97Job, DctToWaveletStageAccelerator,
    Dwt97BatchStageTimings, Float97BatchTile, Htj2k97CodeBlockOptions, IndexedParallelIterator,
    Instant, IntegerBatchTile, IntegerWavelet, IntoParallelIterator, IntoParallelRefIterator,
    JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kScratch,
    ParallelIterator, TranscodeTimingReport,
};
use crate::allocation::try_vec_with_capacity;

pub(in super::super) fn transform_integer_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = batch_component_groups(tiles)?;
    let mut batch_count = 0usize;
    let mut job_count = 0usize;

    for group in groups {
        batch_count = batch_count.saturating_add(1);
        job_count = job_count.saturating_add(group.len());
        let wavelets =
            integer_wavelets_for_batch_group(&group, tiles, scratch, accelerator, timings)?;
        for (component_ref, wavelet) in group.into_iter().zip(wavelets) {
            store_integer_batch_wavelet(component_ref, &wavelet, tiles, options, scratch)?;
        }
    }

    Ok((batch_count, job_count))
}

pub(in super::super) fn transform_float97_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = float97_batch_component_groups(tiles)?;
    let grouped_i16_preencoded = try_store_grouped_i16_preencoded_float97_batches(
        &groups,
        tiles,
        options,
        accelerator,
        timings,
    )?;
    let mut batch_count = 0usize;
    let mut job_count = 0usize;

    for (group_index, group) in groups.into_iter().enumerate() {
        batch_count = batch_count.saturating_add(1);
        job_count = job_count.saturating_add(group.len());
        if grouped_i16_preencoded
            .get(group_index)
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        if try_store_prequantized_float97_batch_group(&group, tiles, options, accelerator, timings)?
        {
            continue;
        }
        let wavelets =
            float97_wavelets_for_batch_group(&group, tiles, scratch, accelerator, timings)?;
        for (component_ref, wavelet) in group.into_iter().zip(wavelets) {
            store_float97_batch_wavelet(component_ref, &wavelet, tiles, options, scratch)?;
        }
    }

    Ok((batch_count, job_count))
}

pub(in super::super) fn integer_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[IntegerBatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let mut jobs = try_vec_with_capacity(group.len())?;
    for component_ref in group {
        jobs.push(integer_dct_job_for_component(
            &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
        )?);
    }
    record_batch_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_reversible_dwt53_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "reversible 5/3 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        record_accelerator_dispatch(timings, group.len());
        let decompose_start = Instant::now();
        let mut wavelets = try_vec_with_capacity(first_levels.len())?;
        for (first_level, component_ref) in first_levels.into_iter().zip(group.iter().copied()) {
            wavelets.push(integer_wavelet_from_first_level(
                first_level,
                tiles[component_ref.tile_index].decomposition_levels,
            )?);
        }
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    let mut wavelets = try_vec_with_capacity(group.len())?;
    for component_ref in group {
        wavelets.push(integer_direct_wavelet_from_component(
            &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
            tiles[component_ref.tile_index].decomposition_levels,
            scratch,
            accelerator,
            timings,
        )?);
    }
    Ok(wavelets)
}

pub(in super::super) fn i16_htj2k97_jobs_for_batch_group<'a>(
    group: &[BatchComponentRef],
    tiles: &'a [Float97BatchTile],
) -> Result<Vec<DctGridI16ToHtj2k97CodeBlockJob<'a>>, JpegToHtj2kError> {
    let mut jobs = try_vec_with_capacity(group.len())?;
    for component_ref in group {
        let tile = &tiles[component_ref.tile_index];
        let component = &tile.jpeg.components[component_ref.component_index];
        let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
        validate_component_block_grid(component)?;
        jobs.push(DctGridI16ToHtj2k97CodeBlockJob {
            dequantized_blocks: &component.dequantized_blocks,
            block_cols: component.block_cols as usize,
            block_rows: component.block_rows as usize,
            width: component.width as usize,
            height: component.height as usize,
            x_rsiz,
            y_rsiz,
        });
    }
    Ok(jobs)
}

pub(in super::super) fn htj2k97_codeblock_options(
    options: &JpegToHtj2kEncodeOptions,
) -> Htj2k97CodeBlockOptions {
    Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: options.guard_bits.max(2),
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        irreversible_quantization_scale: options.irreversible_quantization_scale,
        irreversible_quantization_subband_scales: options.irreversible_quantization_subband_scales,
    }
}

pub(in super::super) fn float97_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[Float97BatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<ComponentWavelet97>, JpegToHtj2kError> {
    let repack_start = Instant::now();
    let mut block_storage = try_vec_with_capacity(group.len())?;
    for component_ref in group {
        block_storage.push(dct_blocks_to_8x8_f64(
            &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                .dequantized_blocks,
        )?);
    }
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let mut jobs = try_vec_with_capacity(group.len())?;
    for (component_ref, blocks) in group.iter().zip(block_storage.iter()) {
        let component =
            &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index];
        validate_component_block_grid(component)?;
        jobs.push(DctGridToDwt97Job {
            blocks,
            block_cols: component.block_cols as usize,
            block_rows: component.block_rows as usize,
            width: component.width as usize,
            height: component.height as usize,
        });
    }

    record_batch_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let accelerated_first_levels = accelerator
        .dct_grid_to_dwt97_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated_first_levels {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        record_accelerator_dispatch(timings, group.len());
        let decompose_start = Instant::now();
        let mut wavelet_results = try_vec_with_capacity(first_levels.len())?;
        first_levels
            .into_par_iter()
            .zip(group.par_iter().copied())
            .map(|(first_level, component_ref)| {
                decompose_97_from_first_level(
                    first_level,
                    usize::from(tiles[component_ref.tile_index].decomposition_levels),
                )
            })
            .collect_into_vec(&mut wavelet_results);
        let mut wavelets = try_vec_with_capacity(wavelet_results.len())?;
        for wavelet in wavelet_results {
            wavelets.push(wavelet?);
        }
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    let mut wavelets = try_vec_with_capacity(group.len())?;
    for component_ref in group {
        wavelets.push(float_direct_97_wavelet_from_component(
            &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
            tiles[component_ref.tile_index].decomposition_levels,
            scratch,
            accelerator,
            timings,
        )?);
    }
    Ok(wavelets)
}

pub(in super::super) fn add_dwt97_batch_stage_timings(
    timings: &mut TranscodeTimingReport,
    stage_timings: Dwt97BatchStageTimings,
) {
    timings.dwt97_batch_pack_upload_us = timings
        .dwt97_batch_pack_upload_us
        .saturating_add(stage_timings.pack_upload_us);
    timings.dwt97_batch_pack_upload_transfers = timings
        .dwt97_batch_pack_upload_transfers
        .saturating_add(stage_timings.pack_upload_transfers);
    timings.dwt97_batch_pack_upload_bytes = timings
        .dwt97_batch_pack_upload_bytes
        .saturating_add(stage_timings.pack_upload_bytes);
    timings.dwt97_batch_resident_dct_handoff_count = timings
        .dwt97_batch_resident_dct_handoff_count
        .saturating_add(stage_timings.resident_dct_handoff_count);
    timings.dwt97_batch_idct_row_lift_us = timings
        .dwt97_batch_idct_row_lift_us
        .saturating_add(stage_timings.idct_row_lift_us);
    timings.dwt97_batch_column_lift_us = timings
        .dwt97_batch_column_lift_us
        .saturating_add(stage_timings.column_lift_us);
    timings.dwt97_batch_resident_dwt_handoff_count = timings
        .dwt97_batch_resident_dwt_handoff_count
        .saturating_add(stage_timings.resident_dwt_handoff_count);
    timings.dwt97_batch_quantize_codeblock_us = timings
        .dwt97_batch_quantize_codeblock_us
        .saturating_add(stage_timings.quantize_codeblock_us);
    timings.dwt97_batch_ht_encode_us = timings
        .dwt97_batch_ht_encode_us
        .saturating_add(stage_timings.ht_encode_us);
    timings.dwt97_batch_ht_kernel_us = timings
        .dwt97_batch_ht_kernel_us
        .saturating_add(stage_timings.ht_kernel_us);
    timings.dwt97_batch_ht_status_readback_us = timings
        .dwt97_batch_ht_status_readback_us
        .saturating_add(stage_timings.ht_status_readback_us);
    timings.dwt97_batch_ht_status_readback_transfers = timings
        .dwt97_batch_ht_status_readback_transfers
        .saturating_add(stage_timings.ht_status_readback_transfers);
    timings.dwt97_batch_ht_status_readback_bytes = timings
        .dwt97_batch_ht_status_readback_bytes
        .saturating_add(stage_timings.ht_status_readback_bytes);
    timings.dwt97_batch_ht_compact_us = timings
        .dwt97_batch_ht_compact_us
        .saturating_add(stage_timings.ht_compact_us);
    timings.dwt97_batch_ht_output_readback_us = timings
        .dwt97_batch_ht_output_readback_us
        .saturating_add(stage_timings.ht_output_readback_us);
    timings.dwt97_batch_ht_output_readback_transfers = timings
        .dwt97_batch_ht_output_readback_transfers
        .saturating_add(stage_timings.ht_output_readback_transfers);
    timings.dwt97_batch_ht_output_readback_bytes = timings
        .dwt97_batch_ht_output_readback_bytes
        .saturating_add(stage_timings.ht_output_readback_bytes);
    timings.dwt97_batch_ht_codeblock_dispatches = timings
        .dwt97_batch_ht_codeblock_dispatches
        .saturating_add(stage_timings.ht_codeblock_dispatches);
    timings.dwt97_batch_readback_us = timings
        .dwt97_batch_readback_us
        .saturating_add(stage_timings.readback_us);
    timings.dwt97_batch_readback_transfers = timings
        .dwt97_batch_readback_transfers
        .saturating_add(stage_timings.readback_transfers);
    timings.dwt97_batch_readback_bytes = timings
        .dwt97_batch_readback_bytes
        .saturating_add(stage_timings.readback_bytes);
}

pub(in super::super) fn record_accelerator_attempt(
    timings: &mut TranscodeTimingReport,
    job_count: usize,
) {
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(job_count);
}

pub(in super::super) fn record_accelerator_dispatch(
    timings: &mut TranscodeTimingReport,
    job_count: usize,
) {
    timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
    timings.accelerator_dispatched_jobs = timings
        .accelerator_dispatched_jobs
        .saturating_add(job_count);
}

pub(in super::super) fn record_batch_attempt(
    timings: &mut TranscodeTimingReport,
    job_count: usize,
) {
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(job_count);
    record_accelerator_attempt(timings, job_count);
}

pub(in super::super) fn record_batch_dispatch(
    timings: &mut TranscodeTimingReport,
    job_count: usize,
) {
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(job_count);
    record_accelerator_dispatch(timings, job_count);
}

pub(in super::super) fn record_cpu_fallback(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.cpu_fallback_jobs = timings.cpu_fallback_jobs.saturating_add(job_count);
}
