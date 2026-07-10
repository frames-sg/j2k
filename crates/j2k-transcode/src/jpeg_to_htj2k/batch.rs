// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    component_sampling_for_jpeg, dct_blocks_to_8x8_f64, decompose_97_from_first_level,
    decomposition_levels_for_components, encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator,
    encode_prequantized_htj2k_97_with_accelerator, error_metrics_i32, extract_dct_blocks,
    flatten_integer_wavelet, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_reference_coefficients,
    integer_dct_job_for_component, integer_direct_wavelet_from_component,
    integer_reference_coefficients, integer_wavelet_from_first_level, j2k_dwt97_from_wavelet,
    j2k_dwt_from_integer_wavelet, jpeg_to_htj2k_with_scratch, rounded_wavelet97_i32,
    transcode_path_name, validate_component_block_grid, validate_transcode_options,
    BatchTranscodeReport, ComponentWavelet97, CpuOnlyJ2kEncodeStageAccelerator, DctExtractOptions,
    DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator, Dwt97BatchStageTimings,
    EncodedTranscode, EncodedTranscodeBatch, Htj2k97CodeBlockOptions, IndexedParallelIterator,
    Instant, IntegerWavelet, IntoParallelIterator, IntoParallelRefIterator,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, JpegDctImage, JpegTileBatchInput,
    JpegToHtj2kCoefficientPath, JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions,
    JpegToHtj2kScratch, JpegToHtj2kTranscoder, ParallelIterator, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image, PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage, PreencodedHtj2k97Component,
    PreencodedHtj2k97Image, PrequantizedHtj2k97Component, PrequantizedHtj2k97Image,
    TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification,
};
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

/// Transcode many JPEG tiles into HTJ2K codestreams.
pub fn jpeg_to_htj2k_batch(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    JpegToHtj2kTranscoder::default().transcode_batch(tiles, options)
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
    validate_transcode_options(options)?;
    match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {}
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
            if accelerator.supports_dwt97_batch()
                || accelerator.supports_htj2k97_codeblock_batch() =>
        {
            return jpeg_float97_tile_batch_to_htj2k_with_scratch(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            );
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53
        | JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
            return Ok(transcode_tile_batch_individually(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            ));
        }
    }

    let extract_start = Instant::now();
    let prepared_results = tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_integer_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect::<Vec<_>>();
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results: Vec<Option<Result<EncodedTranscode, JpegToHtj2kError>>> =
        (0..tiles.len()).map(|_| None).collect();
    let mut prepared_tiles = Vec::new();
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results[tile_index] = Some(Err(error)),
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
    let encoded_tiles = encode_integer_prepared_tiles(prepared_tiles, options, encode_accelerator);
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results[tile_index] = Some(encoded);
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles = tile_results
        .into_iter()
        .map(|tile| {
            tile.unwrap_or(Err(JpegToHtj2kError::Validation(
                "batch transcode did not produce a tile result",
            )))
        })
        .collect::<Vec<_>>();
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
    let prepared_results = tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_float97_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect::<Vec<_>>();
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results: Vec<Option<Result<EncodedTranscode, JpegToHtj2kError>>> =
        (0..tiles.len()).map(|_| None).collect();
    let mut prepared_tiles = Vec::new();
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results[tile_index] = Some(Err(error)),
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
    let encoded_tiles = encode_float97_prepared_tiles(prepared_tiles, options, encode_accelerator);
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results[tile_index] = Some(encoded);
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles = tile_results
        .into_iter()
        .map(|tile| {
            tile.unwrap_or(Err(JpegToHtj2kError::Validation(
                "9/7 batch transcode did not produce a tile result",
            )))
        })
        .collect::<Vec<_>>();
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

pub(super) fn transcode_tile_batch_individually<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> EncodedTranscodeBatch {
    let start = Instant::now();
    let output_tiles = tiles
        .iter()
        .map(|tile| {
            jpeg_to_htj2k_with_scratch(
                tile.bytes,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            )
        })
        .collect::<Vec<_>>();
    let mut timings = aggregate_tile_timings(&output_tiles);
    timings.tile_count = output_tiles.iter().filter(|tile| tile.is_ok()).count();
    let elapsed_us = start.elapsed().as_micros();
    if timings.dct_to_wavelet_total_us == 0 {
        timings.dct_to_wavelet_total_us = elapsed_us
            .saturating_sub(timings.jpeg_dct_extract_us)
            .saturating_sub(timings.htj2k_encode_us);
    }
    batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: timings.component_count,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: timings.dct_to_wavelet_total_us,
            encode_us: timings.htj2k_encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    )
}

pub(super) fn aggregate_tile_timings(
    tiles: &[Result<EncodedTranscode, JpegToHtj2kError>],
) -> TranscodeTimingReport {
    let mut timings = TranscodeTimingReport::default();
    for tile in tiles.iter().filter_map(|tile| tile.as_ref().ok()) {
        timings.add_assign(tile.report.timings);
    }
    timings
}

pub(super) fn batch_output(
    tiles: Vec<Result<EncodedTranscode, JpegToHtj2kError>>,
    mut report: BatchTranscodeReport,
) -> EncodedTranscodeBatch {
    report.successful_tiles = tiles.iter().filter(|tile| tile.is_ok()).count();
    report.failed_tiles = tiles.len().saturating_sub(report.successful_tiles);
    EncodedTranscodeBatch { tiles, report }
}
