// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use super::{
    add_dwt97_batch_stage_timings, dct_blocks_to_8x8_f64, htj2k97_codeblock_options,
    i16_htj2k97_jobs_for_batch_group, record_accelerator_attempt, record_batch_dispatch,
    validate_component_block_grid, BatchComponentRef, DctGridI16ToHtj2k97CodeBlockBatch,
    DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator, Float97BatchTile,
    Htj2k97CodeBlockOptions, IndexedParallelIterator, Instant, IntoParallelRefIterator,
    JpegToHtj2kError, JpegToHtj2kOptions, ParallelIterator, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97Component, TranscodeTimingReport,
};
use crate::allocation::{try_extend_from_slice, try_vec_filled, try_vec_with_capacity};

pub(in super::super) fn store_compact_preencoded_component(
    tile: &mut Float97BatchTile,
    component_index: usize,
    batch_payload: &[u8],
    mut component: PreencodedHtj2k97CompactComponent,
) -> Result<(), JpegToHtj2kError> {
    if component_index >= tile.preencoded_compact_components.len() {
        return Err(JpegToHtj2kError::Validation(
            "compact preencoded component index out of range",
        ));
    }

    for resolution in &mut component.resolutions {
        for subband in &mut resolution.subbands {
            for block in &mut subband.code_blocks {
                if block.payload_range.start > block.payload_range.end
                    || block.payload_range.end > batch_payload.len()
                {
                    return Err(JpegToHtj2kError::Validation(
                        "compact preencoded payload range out of bounds",
                    ));
                }
                let start = tile.preencoded_compact_payload.len();
                try_extend_from_slice(
                    &mut tile.preencoded_compact_payload,
                    &batch_payload[block.payload_range.clone()],
                )?;
                let end = tile.preencoded_compact_payload.len();
                block.payload_range = start..end;
            }
        }
    }

    tile.preencoded_compact_components[component_index] = Some(component);
    Ok(())
}

pub(in super::super) fn try_store_grouped_i16_preencoded_float97_batches<
    A: DctToWaveletStageAccelerator,
>(
    groups: &[Vec<BatchComponentRef>],
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<bool>, JpegToHtj2kError> {
    let mut handled = try_vec_filled(groups.len(), false)?;
    if !accelerator.supports_htj2k97_i16_preencoded_batch()
        || options.validate_against_float_reference
        || groups.len() <= 1
    {
        return Ok(handled);
    }

    let mut eligible_indices = try_vec_with_capacity(groups.len())?;
    for (index, group) in groups.iter().enumerate() {
        let eligible = group
            .iter()
            .all(|component_ref| tiles[component_ref.tile_index].decomposition_levels == 1);
        if eligible {
            eligible_indices.push(index);
        }
    }
    if eligible_indices.len() <= 1 {
        return Ok(handled);
    }

    let codeblock_options = htj2k97_codeblock_options(&options.encode_options);
    let total_jobs = eligible_indices
        .iter()
        .map(|&index| groups[index].len())
        .fold(0usize, usize::saturating_add);
    record_accelerator_attempt(timings, total_jobs);
    let accelerator_start = Instant::now();
    let mut jobs_by_group = try_vec_with_capacity(eligible_indices.len())?;
    for &index in &eligible_indices {
        jobs_by_group.push(i16_htj2k97_jobs_for_batch_group(&groups[index], tiles)?);
    }
    let mut batches = try_vec_with_capacity(jobs_by_group.len())?;
    for jobs in &jobs_by_group {
        batches.push(DctGridI16ToHtj2k97CodeBlockBatch { jobs });
    }

    let compact_grouped_components = if accelerator.supports_htj2k97_compact_preencoded_batch() {
        accelerator
            .dct_grid_i16_to_htj2k97_compact_preencoded_batch_groups(&batches, codeblock_options)
            .map_err(JpegToHtj2kError::Accelerator)?
    } else {
        None
    };
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    if let Some(compact_grouped_components) = compact_grouped_components {
        timings.dct_to_wavelet_accelerator_us = timings
            .dct_to_wavelet_accelerator_us
            .saturating_add(accelerator_start.elapsed().as_micros());
        store_grouped_compact_preencoded_components(
            groups,
            &eligible_indices,
            tiles,
            timings,
            &compact_grouped_components.payload,
            compact_grouped_components.groups,
            &mut handled,
        )?;
        return Ok(handled);
    }

    let grouped_components = accelerator
        .dct_grid_i16_to_htj2k97_preencoded_batch_groups(&batches, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(grouped_components) = grouped_components {
        store_grouped_preencoded_components(
            groups,
            &eligible_indices,
            tiles,
            timings,
            grouped_components,
            &mut handled,
        )?;
    }
    Ok(handled)
}

fn store_grouped_compact_preencoded_components(
    groups: &[Vec<BatchComponentRef>],
    eligible_indices: &[usize],
    tiles: &mut [Float97BatchTile],
    timings: &mut TranscodeTimingReport,
    compact_payload: &[u8],
    compact_groups: Vec<Vec<PreencodedHtj2k97CompactComponent>>,
    handled: &mut [bool],
) -> Result<(), JpegToHtj2kError> {
    if compact_groups.len() != eligible_indices.len() {
        return Err(JpegToHtj2kError::Validation(
            "9/7 grouped i16 compact preencoded accelerator returned wrong group count",
        ));
    }
    for (&group_index, components) in eligible_indices.iter().zip(compact_groups) {
        let group = &groups[group_index];
        if components.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 grouped i16 compact preencoded accelerator returned wrong component count",
            ));
        }

        timings.component_count = timings.component_count.saturating_add(group.len());
        record_batch_dispatch(timings, group.len());
        for (component_ref, component) in group.iter().copied().zip(components) {
            store_compact_preencoded_component(
                &mut tiles[component_ref.tile_index],
                component_ref.component_index,
                compact_payload,
                component,
            )?;
        }
        handled[group_index] = true;
    }
    Ok(())
}

fn store_grouped_preencoded_components(
    groups: &[Vec<BatchComponentRef>],
    eligible_indices: &[usize],
    tiles: &mut [Float97BatchTile],
    timings: &mut TranscodeTimingReport,
    grouped_components: Vec<Vec<PreencodedHtj2k97Component>>,
    handled: &mut [bool],
) -> Result<(), JpegToHtj2kError> {
    if grouped_components.len() != eligible_indices.len() {
        return Err(JpegToHtj2kError::Validation(
            "9/7 grouped i16 preencoded accelerator returned wrong group count",
        ));
    }
    for (&group_index, components) in eligible_indices.iter().zip(grouped_components) {
        let group = &groups[group_index];
        if components.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 grouped i16 preencoded accelerator returned wrong component count",
            ));
        }

        timings.component_count = timings.component_count.saturating_add(group.len());
        record_batch_dispatch(timings, group.len());
        for (component_ref, component) in group.iter().copied().zip(components) {
            tiles[component_ref.tile_index].preencoded_components[component_ref.component_index] =
                Some(component);
        }
        handled[group_index] = true;
    }
    Ok(())
}

pub(in super::super) fn try_store_prequantized_float97_batch_group<
    A: DctToWaveletStageAccelerator,
>(
    group: &[BatchComponentRef],
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<bool, JpegToHtj2kError> {
    if !(accelerator.supports_htj2k97_codeblock_batch()
        || accelerator.supports_htj2k97_i16_preencoded_batch())
        || options.validate_against_float_reference
        || group
            .iter()
            .any(|component_ref| tiles[component_ref.tile_index].decomposition_levels != 1)
    {
        return Ok(false);
    }

    let codeblock_options = htj2k97_codeblock_options(&options.encode_options);
    if try_store_i16_preencoded_float97_batch_group(
        group,
        tiles,
        codeblock_options,
        accelerator,
        timings,
    )? {
        return Ok(true);
    }
    try_store_f64_prequantized_float97_batch_group(
        group,
        tiles,
        codeblock_options,
        accelerator,
        timings,
    )
}

fn try_store_i16_preencoded_float97_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &mut [Float97BatchTile],
    codeblock_options: Htj2k97CodeBlockOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<bool, JpegToHtj2kError> {
    if !accelerator.supports_htj2k97_i16_preencoded_batch() {
        return Ok(false);
    }
    let jobs = i16_htj2k97_jobs_for_batch_group(group, tiles)?;
    record_accelerator_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let compact_preencoded_components = if accelerator.supports_htj2k97_compact_preencoded_batch() {
        accelerator
            .dct_grid_i16_to_htj2k97_compact_preencoded_batch(&jobs, codeblock_options)
            .map_err(JpegToHtj2kError::Accelerator)?
    } else {
        None
    };
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    if let Some(compact_batch) = compact_preencoded_components {
        timings.dct_to_wavelet_accelerator_us = timings
            .dct_to_wavelet_accelerator_us
            .saturating_add(accelerator_start.elapsed().as_micros());
        if compact_batch.components.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 i16 compact preencoded accelerator returned wrong component count",
            ));
        }

        timings.component_count = timings.component_count.saturating_add(group.len());
        record_batch_dispatch(timings, group.len());
        for (component_ref, component) in group.iter().copied().zip(compact_batch.components) {
            store_compact_preencoded_component(
                &mut tiles[component_ref.tile_index],
                component_ref.component_index,
                &compact_batch.payload,
                component,
            )?;
        }
        return Ok(true);
    }

    let preencoded_components = accelerator
        .dct_grid_i16_to_htj2k97_preencoded_batch(&jobs, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    let Some(components) = preencoded_components else {
        return Ok(false);
    };
    if components.len() != group.len() {
        return Err(JpegToHtj2kError::Validation(
            "9/7 i16 preencoded accelerator returned wrong component count",
        ));
    }

    timings.component_count = timings.component_count.saturating_add(group.len());
    record_batch_dispatch(timings, group.len());
    for (component_ref, component) in group.iter().copied().zip(components) {
        tiles[component_ref.tile_index].preencoded_components[component_ref.component_index] =
            Some(component);
    }
    Ok(true)
}

fn try_store_f64_prequantized_float97_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &mut [Float97BatchTile],
    codeblock_options: Htj2k97CodeBlockOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<bool, JpegToHtj2kError> {
    let repack_start = Instant::now();
    let mut block_results = try_vec_with_capacity(group.len())?;
    group
        .par_iter()
        .map(|component_ref| {
            dct_blocks_to_8x8_f64(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                    .dequantized_blocks,
            )
        })
        .collect_into_vec(&mut block_results);
    let mut block_storage = try_vec_with_capacity(block_results.len())?;
    for blocks in block_results {
        block_storage.push(blocks?);
    }
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let mut jobs = try_vec_with_capacity(group.len())?;
    for (component_ref, blocks) in group.iter().zip(block_storage.iter()) {
        let tile = &tiles[component_ref.tile_index];
        let component = &tile.jpeg.components[component_ref.component_index];
        let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
        validate_component_block_grid(component)?;
        jobs.push(DctGridToHtj2k97CodeBlockJob {
            blocks,
            block_cols: component.block_cols as usize,
            block_rows: component.block_rows as usize,
            width: component.width as usize,
            height: component.height as usize,
            x_rsiz,
            y_rsiz,
        });
    }

    record_accelerator_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let preencoded_components = accelerator
        .dct_grid_to_htj2k97_preencoded_batch(&jobs, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(components) = preencoded_components {
        if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
            add_dwt97_batch_stage_timings(timings, stage_timings);
        }
        timings.dct_to_wavelet_accelerator_us = timings
            .dct_to_wavelet_accelerator_us
            .saturating_add(accelerator_start.elapsed().as_micros());
        if components.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 preencoded accelerator returned wrong component count",
            ));
        }

        timings.component_count = timings.component_count.saturating_add(group.len());
        record_batch_dispatch(timings, group.len());
        for (component_ref, component) in group.iter().copied().zip(components) {
            tiles[component_ref.tile_index].preencoded_components[component_ref.component_index] =
                Some(component);
        }
        return Ok(true);
    }

    let accelerated_components = accelerator
        .dct_grid_to_htj2k97_codeblock_batch(&jobs, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    let Some(components) = accelerated_components else {
        return Ok(false);
    };
    if components.len() != group.len() {
        return Err(JpegToHtj2kError::Validation(
            "9/7 code-block accelerator returned wrong component count",
        ));
    }

    timings.component_count = timings.component_count.saturating_add(group.len());
    record_batch_dispatch(timings, group.len());
    for (component_ref, component) in group.iter().copied().zip(components) {
        tiles[component_ref.tile_index].prequantized_components[component_ref.component_index] =
            Some(component);
    }
    Ok(true)
}
