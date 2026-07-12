// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    component_sampling_for_jpeg, dct_blocks_to_8x8_f64, decompose_97_from_first_level,
    decomposition_levels_for_components,
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes,
    encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes,
    encoded_transcode_retained_bytes, error_metrics_i32_with_live_budget, extract_dct_blocks,
    flatten_integer_wavelet, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_reference_coefficients,
    integer_dct_job_for_component, integer_direct_wavelet_from_component,
    integer_reference_coefficients, integer_wavelet_from_first_level, j2k_dwt97_from_wavelet,
    j2k_dwt_from_integer_wavelet, jpeg_to_htj2k_with_scratch, map_encode_error,
    rounded_wavelet97_i32, transcode_path_name, validate_component_block_grid,
    validate_jpeg_transcode_workspace, validate_transcode_options, BatchTranscodeReport,
    ComponentWavelet97, CpuOnlyJ2kEncodeStageAccelerator, DctExtractOptions,
    DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator, Dwt97BatchStageTimings,
    EncodedTranscode, EncodedTranscodeBatch, HostLiveBudget, Htj2k97CodeBlockOptions,
    IndexedParallelIterator, Instant, IntegerWavelet, IntoParallelIterator,
    IntoParallelRefIterator, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, JpegDctImage,
    JpegTileBatchInput, JpegToHtj2kCoefficientPath, JpegToHtj2kEncodeOptions, JpegToHtj2kError,
    JpegToHtj2kOptions, JpegToHtj2kScratch, JpegToHtj2kTranscoder, ParallelIterator,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification,
};
use crate::allocation::try_vec_with_capacity;
mod group_budget;
mod workspace;
use self::workspace::{validate_batch_workspace, BatchWorkspaceKind};
mod result_slots;
use self::result_slots::BatchResultSlots;
mod prepare;
pub(super) use self::prepare::{
    batch_component_groups, float97_batch_component_groups, prepare_float97_batch_tile,
    prepare_integer_batch_tile, BatchComponentRef, Float97BatchTile, Float97PrecomputedBatchRecord,
    IntegerBatchTile,
};
mod transform;
pub(super) use self::transform::{
    add_dwt97_batch_stage_timings, htj2k97_codeblock_options, i16_htj2k97_jobs_for_batch_group,
    record_accelerator_attempt, record_accelerator_dispatch, record_batch_attempt,
    record_batch_dispatch, record_cpu_fallback, transform_float97_batch_tiles,
    transform_integer_batch_tiles,
};
mod accelerated_storage;
#[cfg(test)]
pub(super) use self::accelerated_storage::store_compact_preencoded_component;
pub(super) use self::accelerated_storage::{
    try_store_grouped_i16_preencoded_float97_batches, try_store_prequantized_float97_batch_group,
};
mod storage;
pub(super) use self::storage::{store_float97_batch_wavelet, store_integer_batch_wavelet};
mod encode;
pub(super) use self::encode::{
    add_encode_timing_counters_from_result, encode_float97_prepared_tiles,
    encode_integer_prepared_tiles, record_encode_dispatch_delta,
};
mod actual_live;
use self::actual_live::{
    validate_float97_prepared_collection, validate_integer_prepared_collection,
};
mod individual;
use self::individual::transcode_tile_batch_individually;

/// Transcode many JPEG tiles into HTJ2K codestreams.
pub fn jpeg_to_htj2k_batch(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    JpegToHtj2kTranscoder::default().transcode_batch(tiles, options)
}

enum BatchExecutionRoute {
    Integer53,
    AcceleratedFloat97,
    Individual,
}

fn validate_batch_route(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    accelerator: &impl DctToWaveletStageAccelerator,
) -> Result<BatchExecutionRoute, JpegToHtj2kError> {
    validate_transcode_options(options)?;
    let (route, workspace_kind) = match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {
            (BatchExecutionRoute::Integer53, BatchWorkspaceKind::Integer)
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
            if accelerator.supports_dwt97_batch()
                || accelerator.supports_htj2k97_codeblock_batch() =>
        {
            (
                BatchExecutionRoute::AcceleratedFloat97,
                BatchWorkspaceKind::AcceleratedFloat97,
            )
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53
        | JpegToHtj2kCoefficientPath::FloatDirectLinear97 => (
            BatchExecutionRoute::Individual,
            BatchWorkspaceKind::Individual,
        ),
    };
    validate_batch_workspace(tiles, options, workspace_kind)?;
    Ok(route)
}

pub(super) fn jpeg_tile_batch_to_htj2k_with_scratch<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    match validate_batch_route(tiles, options, accelerator)? {
        BatchExecutionRoute::Integer53 => {}
        BatchExecutionRoute::AcceleratedFloat97 => {
            return jpeg_float97_tile_batch_to_htj2k_with_scratch(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            );
        }
        BatchExecutionRoute::Individual => {
            return transcode_tile_batch_individually(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            );
        }
    }

    let extract_start = Instant::now();
    let mut prepared_results = try_vec_with_capacity(tiles.len())?;
    tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_integer_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect_into_vec(&mut prepared_results);
    validate_integer_prepared_collection(&prepared_results, prepared_results.capacity(), scratch)?;
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results = BatchResultSlots::try_new(tiles.len())?;
    let mut prepared_tiles = try_vec_with_capacity(tiles.len())?;
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results.insert(tile_index, Err(error))?,
        }
    }

    let transform_start = Instant::now();
    let mut timings = TranscodeTimingReport::default();
    let (reversible_dwt53_batches, reversible_dwt53_batch_jobs) = transform_integer_batch_tiles(
        &mut prepared_tiles,
        options,
        scratch,
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.jpeg_dct_extract_us = extract_us;
    timings.dct_to_wavelet_total_us = transform_us;
    timings.tile_count = prepared_tiles.len();

    let encode_start = Instant::now();
    let mut encode_external = HostLiveBudget::process_cap();
    encode_external.add_bytes(scratch.retained_bytes()?)?;
    encode_external.add_bytes(tile_results.retained_slot_bytes()?)?;
    let encoded_tiles = encode_integer_prepared_tiles(
        prepared_tiles,
        options,
        encode_accelerator,
        encode_external.live_bytes(),
    )?;
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results.insert(tile_index, encoded)?;
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles =
        tile_results.into_results_with_live_budget(scratch.retained_bytes()?, |result| {
            result
                .as_ref()
                .map_or(Ok(0), encoded_transcode_retained_bytes)
        })?;
    Ok(batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: reversible_dwt53_batch_jobs,
            reversible_dwt53_batches,
            reversible_dwt53_batch_jobs,
            extract_us,
            transform_us,
            encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    ))
}

pub(super) fn jpeg_float97_tile_batch_to_htj2k_with_scratch<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let mut prepared_results = try_vec_with_capacity(tiles.len())?;
    tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_float97_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect_into_vec(&mut prepared_results);
    validate_float97_prepared_collection(&prepared_results, prepared_results.capacity(), scratch)?;
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results = BatchResultSlots::try_new(tiles.len())?;
    let mut prepared_tiles = try_vec_with_capacity(tiles.len())?;
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results.insert(tile_index, Err(error))?,
        }
    }

    let transform_start = Instant::now();
    let mut timings = TranscodeTimingReport::default();
    let (_dwt97_batches, dwt97_batch_jobs) = transform_float97_batch_tiles(
        &mut prepared_tiles,
        options,
        scratch,
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.jpeg_dct_extract_us = extract_us;
    timings.dct_to_wavelet_total_us = transform_us;
    timings.tile_count = prepared_tiles.len();

    let encode_start = Instant::now();
    let mut encode_external = HostLiveBudget::process_cap();
    encode_external.add_bytes(scratch.retained_bytes()?)?;
    encode_external.add_bytes(tile_results.retained_slot_bytes()?)?;
    let encoded_tiles = encode_float97_prepared_tiles(
        prepared_tiles,
        options,
        encode_accelerator,
        encode_external.live_bytes(),
    )?;
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results.insert(tile_index, encoded)?;
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles =
        tile_results.into_results_with_live_budget(scratch.retained_bytes()?, |result| {
            result
                .as_ref()
                .map_or(Ok(0), encoded_transcode_retained_bytes)
        })?;
    Ok(batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: dwt97_batch_jobs,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us,
            transform_us,
            encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    ))
}

pub(super) fn batch_output(
    tiles: Vec<Result<EncodedTranscode, JpegToHtj2kError>>,
    mut report: BatchTranscodeReport,
) -> EncodedTranscodeBatch {
    report.successful_tiles = tiles.iter().filter(|tile| tile.is_ok()).count();
    report.failed_tiles = tiles.len().saturating_sub(report.successful_tiles);
    EncodedTranscodeBatch { tiles, report }
}
