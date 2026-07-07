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

pub(super) struct IntegerBatchTile {
    pub(super) tile_index: usize,
    pub(super) jpeg: JpegDctImage,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) decomposition_levels: u8,
    pub(super) all_unit_sampled: bool,
    pub(super) component_reports: Vec<TranscodeComponentReport>,
    pub(super) precomputed_components: Vec<Option<PrecomputedHtj2k53Component>>,
    pub(super) float_validation_actual: Vec<i32>,
    pub(super) float_validation_expected: Vec<i32>,
    pub(super) integer_validation_actual: Vec<i32>,
    pub(super) integer_validation_expected: Vec<i32>,
    pub(super) timings: TranscodeTimingReport,
}

pub(super) struct Float97BatchTile {
    pub(super) tile_index: usize,
    pub(super) jpeg: JpegDctImage,
    pub(super) component_sampling: Vec<(u8, u8)>,
    pub(super) decomposition_levels: u8,
    pub(super) all_unit_sampled: bool,
    pub(super) component_reports: Vec<TranscodeComponentReport>,
    pub(super) precomputed_components: Vec<Option<PrecomputedHtj2k97Component>>,
    pub(super) preencoded_compact_payload: Vec<u8>,
    pub(super) preencoded_compact_components: Vec<Option<PreencodedHtj2k97CompactComponent>>,
    pub(super) preencoded_components: Vec<Option<PreencodedHtj2k97Component>>,
    pub(super) prequantized_components: Vec<Option<PrequantizedHtj2k97Component>>,
    pub(super) float_validation_actual: Vec<i32>,
    pub(super) float_validation_expected: Vec<i32>,
    pub(super) timings: TranscodeTimingReport,
}

pub(super) struct Float97PrecomputedBatchRecord {
    pub(super) tile_index: usize,
    pub(super) jpeg: JpegDctImage,
    pub(super) decomposition_levels: u8,
    pub(super) all_unit_sampled: bool,
    pub(super) component_reports: Vec<TranscodeComponentReport>,
    pub(super) float_validation_actual: Vec<i32>,
    pub(super) float_validation_expected: Vec<i32>,
    pub(super) timings: TranscodeTimingReport,
}

#[derive(Clone, Copy)]
pub(super) struct BatchComponentRef {
    pub(super) tile_index: usize,
    pub(super) component_index: usize,
}

pub(super) fn prepare_integer_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<IntegerBatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(IntegerBatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        integer_validation_actual: Vec::new(),
        integer_validation_expected: Vec::new(),
        timings,
    })
}

pub(super) fn prepare_float97_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<Float97BatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::dequantized_only())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();
    let preencoded_compact_components = (0..jpeg.components.len()).map(|_| None).collect();
    let preencoded_components = (0..jpeg.components.len()).map(|_| None).collect();
    let prequantized_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(Float97BatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        preencoded_compact_payload: Vec::new(),
        preencoded_compact_components,
        preencoded_components,
        prequantized_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        timings,
    })
}

pub(super) fn transform_integer_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = batch_component_groups(tiles);
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

pub(super) fn transform_float97_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = float97_batch_component_groups(tiles);
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

pub(super) fn batch_component_groups(tiles: &[IntegerBatchTile]) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for (component_index, component) in tile.jpeg.components.iter().enumerate() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                let _ = component;
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

pub(super) fn float97_batch_component_groups(
    tiles: &[Float97BatchTile],
) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for component_index in 0..tile.jpeg.components.len() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_float97_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

pub(super) fn same_batch_component_key(
    left_tile: &IntegerBatchTile,
    left_component_index: usize,
    right_tile: &IntegerBatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.component_index == right.component_index
        && left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}

pub(super) fn same_float97_batch_component_key(
    left_tile: &Float97BatchTile,
    left_component_index: usize,
    right_tile: &Float97BatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}

pub(super) fn integer_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[IntegerBatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let jobs = group
        .iter()
        .map(|component_ref| {
            integer_dct_job_for_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
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
        let wavelets = first_levels
            .into_iter()
            .zip(group.iter().copied())
            .map(|(first_level, component_ref)| {
                integer_wavelet_from_first_level(
                    first_level,
                    tiles[component_ref.tile_index].decomposition_levels,
                )
            })
            .collect();
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|component_ref| {
            integer_direct_wavelet_from_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
                tiles[component_ref.tile_index].decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

pub(super) fn i16_htj2k97_jobs_for_batch_group<'a>(
    group: &[BatchComponentRef],
    tiles: &'a [Float97BatchTile],
) -> Result<Vec<DctGridI16ToHtj2k97CodeBlockJob<'a>>, JpegToHtj2kError> {
    group
        .iter()
        .map(|component_ref| {
            let tile = &tiles[component_ref.tile_index];
            let component = &tile.jpeg.components[component_ref.component_index];
            let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
            validate_component_block_grid(component)?;
            Ok(DctGridI16ToHtj2k97CodeBlockJob {
                dequantized_blocks: &component.dequantized_blocks,
                block_cols: component.block_cols as usize,
                block_rows: component.block_rows as usize,
                width: component.width as usize,
                height: component.height as usize,
                x_rsiz,
                y_rsiz,
            })
        })
        .collect()
}

pub(super) fn store_compact_preencoded_component(
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
                tile.preencoded_compact_payload
                    .extend_from_slice(&batch_payload[block.payload_range.clone()]);
                let end = tile.preencoded_compact_payload.len();
                block.payload_range = start..end;
            }
        }
    }

    tile.preencoded_compact_components[component_index] = Some(component);
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(super) fn try_store_grouped_i16_preencoded_float97_batches<A: DctToWaveletStageAccelerator>(
    groups: &[Vec<BatchComponentRef>],
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<bool>, JpegToHtj2kError> {
    let mut handled = vec![false; groups.len()];
    if !accelerator.supports_htj2k97_i16_preencoded_batch()
        || options.validate_against_float_reference
        || groups.len() <= 1
    {
        return Ok(handled);
    }

    let eligible_indices = groups
        .iter()
        .enumerate()
        .filter_map(|(index, group)| {
            let eligible = group
                .iter()
                .all(|component_ref| tiles[component_ref.tile_index].decomposition_levels == 1);
            eligible.then_some(index)
        })
        .collect::<Vec<_>>();
    if eligible_indices.len() <= 1 {
        return Ok(handled);
    }

    let codeblock_options = htj2k97_codeblock_options(&options.encode_options);
    let total_jobs = eligible_indices
        .iter()
        .map(|&index| groups[index].len())
        .sum::<usize>();
    record_accelerator_attempt(timings, total_jobs);
    let accelerator_start = Instant::now();
    let jobs_by_group = eligible_indices
        .iter()
        .map(|&index| i16_htj2k97_jobs_for_batch_group(&groups[index], tiles))
        .collect::<Result<Vec<_>, JpegToHtj2kError>>()?;
    let batches = jobs_by_group
        .iter()
        .map(|jobs| DctGridI16ToHtj2k97CodeBlockBatch { jobs })
        .collect::<Vec<_>>();
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
        let compact_payload = compact_grouped_components.payload;
        let compact_groups = compact_grouped_components.groups;
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
                    &compact_payload,
                    component,
                )?;
            }
            handled[group_index] = true;
        }
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

    let Some(grouped_components) = grouped_components else {
        return Ok(handled);
    };
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

    Ok(handled)
}

#[allow(clippy::too_many_lines)]
pub(super) fn try_store_prequantized_float97_batch_group<A: DctToWaveletStageAccelerator>(
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
    if accelerator.supports_htj2k97_i16_preencoded_batch() {
        let jobs = i16_htj2k97_jobs_for_batch_group(group, tiles)?;

        record_accelerator_attempt(timings, group.len());
        let accelerator_start = Instant::now();
        let compact_preencoded_components =
            if accelerator.supports_htj2k97_compact_preencoded_batch() {
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
        if let Some(components) = preencoded_components {
            if components.len() != group.len() {
                return Err(JpegToHtj2kError::Validation(
                    "9/7 i16 preencoded accelerator returned wrong component count",
                ));
            }

            timings.component_count = timings.component_count.saturating_add(group.len());
            record_batch_dispatch(timings, group.len());
            for (component_ref, component) in group.iter().copied().zip(components) {
                tiles[component_ref.tile_index].preencoded_components
                    [component_ref.component_index] = Some(component);
            }

            return Ok(true);
        }
    }

    let repack_start = Instant::now();
    let block_storage = group
        .par_iter()
        .map(|component_ref| {
            dct_blocks_to_8x8_f64(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                    .dequantized_blocks,
            )
        })
        .collect::<Vec<_>>();
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let jobs = group
        .iter()
        .zip(block_storage.iter())
        .map(|(component_ref, blocks)| {
            let tile = &tiles[component_ref.tile_index];
            let component = &tile.jpeg.components[component_ref.component_index];
            let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
            validate_component_block_grid(component)?;
            Ok(DctGridToHtj2k97CodeBlockJob {
                blocks,
                block_cols: component.block_cols as usize,
                block_rows: component.block_rows as usize,
                width: component.width as usize,
                height: component.height as usize,
                x_rsiz,
                y_rsiz,
            })
        })
        .collect::<Result<Vec<_>, JpegToHtj2kError>>()?;

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

pub(super) fn htj2k97_codeblock_options(
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

pub(super) fn float97_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[Float97BatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<ComponentWavelet97>, JpegToHtj2kError> {
    let repack_start = Instant::now();
    let block_storage = group
        .iter()
        .map(|component_ref| {
            dct_blocks_to_8x8_f64(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                    .dequantized_blocks,
            )
        })
        .collect::<Vec<_>>();
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let jobs = group
        .iter()
        .zip(block_storage.iter())
        .map(|(component_ref, blocks)| {
            let component =
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index];
            validate_component_block_grid(component)?;
            Ok(DctGridToDwt97Job {
                blocks,
                block_cols: component.block_cols as usize,
                block_rows: component.block_rows as usize,
                width: component.width as usize,
                height: component.height as usize,
            })
        })
        .collect::<Result<Vec<_>, JpegToHtj2kError>>()?;

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
        let wavelets = first_levels
            .into_par_iter()
            .zip(group.par_iter().copied())
            .map(|(first_level, component_ref)| {
                decompose_97_from_first_level(
                    first_level,
                    usize::from(tiles[component_ref.tile_index].decomposition_levels),
                )
            })
            .collect::<Vec<_>>();
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|component_ref| {
            float_direct_97_wavelet_from_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
                tiles[component_ref.tile_index].decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

pub(super) fn add_dwt97_batch_stage_timings(
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

pub(super) fn record_accelerator_attempt(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(job_count);
}

pub(super) fn record_accelerator_dispatch(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
    timings.accelerator_dispatched_jobs = timings
        .accelerator_dispatched_jobs
        .saturating_add(job_count);
}

pub(super) fn record_batch_attempt(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(job_count);
    record_accelerator_attempt(timings, job_count);
}

pub(super) fn record_batch_dispatch(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(job_count);
    record_accelerator_dispatch(timings, job_count);
}

pub(super) fn record_cpu_fallback(timings: &mut TranscodeTimingReport, job_count: usize) {
    timings.cpu_fallback_jobs = timings.cpu_fallback_jobs.saturating_add(job_count);
}

pub(super) fn store_integer_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &IntegerWavelet,
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    let actual_coefficients = flatten_integer_wavelet(wavelet);
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k53Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt_from_integer_wavelet(wavelet),
        });

    if options.validate_against_float_reference {
        tile.float_validation_actual
            .extend(actual_coefficients.clone());
        tile.float_validation_expected
            .extend(float_reference_coefficients(
                component,
                tile.decomposition_levels,
                scratch,
            )?);
    }
    if options.validate_against_integer_reference {
        tile.integer_validation_actual.extend(actual_coefficients);
        tile.integer_validation_expected
            .extend(integer_reference_coefficients(
                component,
                tile.decomposition_levels,
            )?);
    }

    Ok(())
}

pub(super) fn store_float97_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &ComponentWavelet97,
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k97Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt97_from_wavelet(
                wavelet,
                component.width as usize,
                component.height as usize,
            ),
        });

    if options.validate_against_float_reference {
        let actual_coefficients = rounded_wavelet97_i32(wavelet)?;
        tile.float_validation_actual.extend(actual_coefficients);
        tile.float_validation_expected
            .extend(float97_reference_coefficients(
                component,
                tile.decomposition_levels,
                scratch,
            )?);
    }

    Ok(())
}

pub(super) fn record_encode_dispatch_delta(
    timings: &mut TranscodeTimingReport,
    before: J2kEncodeDispatchReport,
    after: J2kEncodeDispatchReport,
) {
    let delta = after.saturating_delta(before);
    timings.htj2k_encode_accelerator_dispatches = timings
        .htj2k_encode_accelerator_dispatches
        .saturating_add(delta.total());
    timings.htj2k_encode_ht_code_block_dispatches = timings
        .htj2k_encode_ht_code_block_dispatches
        .saturating_add(delta.ht_code_block);
    timings.htj2k_encode_packetization_dispatches = timings
        .htj2k_encode_packetization_dispatches
        .saturating_add(delta.packetization);
}

pub(super) fn add_encode_timing_counters_from_result(
    timings: &mut TranscodeTimingReport,
    tile: &Result<EncodedTranscode, JpegToHtj2kError>,
) {
    let Ok(tile) = tile else {
        return;
    };
    timings.htj2k_encode_accelerator_dispatches = timings
        .htj2k_encode_accelerator_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_accelerator_dispatches);
    timings.htj2k_encode_ht_code_block_dispatches = timings
        .htj2k_encode_ht_code_block_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_ht_code_block_dispatches);
    timings.htj2k_encode_packetization_dispatches = timings
        .htj2k_encode_packetization_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_packetization_dispatches);
}

pub(super) fn encode_integer_prepared_tiles<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<IntegerBatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    if encode_accelerator.prefer_parallel_cpu_tile_encode() {
        return prepared_tiles
            .into_par_iter()
            .map(|prepared| {
                let tile_index = prepared.tile_index;
                let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
                (
                    tile_index,
                    encode_integer_batch_tile(prepared, options, &mut cpu_accelerator),
                )
            })
            .collect();
    }

    prepared_tiles
        .into_iter()
        .map(|prepared| {
            let tile_index = prepared.tile_index;
            (
                tile_index,
                encode_integer_batch_tile(prepared, options, encode_accelerator),
            )
        })
        .collect()
}

pub(super) fn encode_float97_prepared_tiles<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    if !encode_accelerator.prefer_parallel_cpu_tile_encode()
        && can_encode_float97_precomputed_tiles_batch(&prepared_tiles, options)
    {
        return encode_float97_precomputed_tiles_batch(prepared_tiles, options, encode_accelerator);
    }

    if encode_accelerator.prefer_parallel_cpu_tile_encode() {
        return prepared_tiles
            .into_par_iter()
            .map(|prepared| {
                let tile_index = prepared.tile_index;
                let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
                (
                    tile_index,
                    encode_float97_batch_tile(prepared, options, &mut cpu_accelerator),
                )
            })
            .collect();
    }

    prepared_tiles
        .into_iter()
        .map(|prepared| {
            let tile_index = prepared.tile_index;
            (
                tile_index,
                encode_float97_batch_tile(prepared, options, encode_accelerator),
            )
        })
        .collect()
}

pub(super) fn can_encode_float97_precomputed_tiles_batch(
    prepared_tiles: &[Float97BatchTile],
    options: &JpegToHtj2kOptions,
) -> bool {
    options.encode_options.num_layers == 1
        && prepared_tiles.iter().all(|tile| {
            tile.precomputed_components.iter().all(Option::is_some)
                && tile.preencoded_compact_payload.is_empty()
                && tile
                    .preencoded_compact_components
                    .iter()
                    .all(Option::is_none)
                && tile.preencoded_components.iter().all(Option::is_none)
                && tile.prequantized_components.iter().all(Option::is_none)
        })
}

#[allow(clippy::too_many_lines)]
pub(super) fn encode_float97_precomputed_tiles_batch<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    let mut records = Vec::with_capacity(prepared_tiles.len());
    let mut images = Vec::with_capacity(prepared_tiles.len());

    for tile in prepared_tiles {
        let Float97BatchTile {
            tile_index,
            jpeg,
            decomposition_levels,
            all_unit_sampled,
            component_reports,
            precomputed_components,
            preencoded_compact_payload: _,
            preencoded_compact_components: _,
            preencoded_components: _,
            prequantized_components: _,
            float_validation_actual,
            float_validation_expected,
            timings,
            ..
        } = tile;
        let components = match precomputed_components
            .into_iter()
            .map(|component| {
                component.ok_or(JpegToHtj2kError::Validation(
                    "9/7 precomputed batch transcode did not produce all components",
                ))
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(components) => components,
            Err(error) => return vec![(tile_index, Err(error))],
        };
        images.push(PrecomputedHtj2k97Image {
            width: jpeg.width,
            height: jpeg.height,
            bit_depth: 8,
            signed: false,
            components,
        });
        records.push(Float97PrecomputedBatchRecord {
            tile_index,
            jpeg,
            decomposition_levels,
            all_unit_sampled,
            component_reports,
            float_validation_actual,
            float_validation_expected,
            timings,
        });
    }

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_images = images;
    let codestreams = {
        let native_encode_options = options.encode_options.to_native();
        match encode_precomputed_htj2k_97_batch_with_accelerator(
            &native_images,
            &native_encode_options,
            encode_accelerator,
        ) {
            Ok(codestreams) => codestreams,
            Err(error) => {
                return records
                    .into_iter()
                    .map(|record| (record.tile_index, Err(JpegToHtj2kError::Encode(error))))
                    .collect();
            }
        }
    };
    let encode_dispatch_after = encode_accelerator.dispatch_report();
    let encode_us = encode_start.elapsed().as_micros();

    if codestreams.len() != records.len() {
        return records
            .into_iter()
            .map(|record| {
                (
                    record.tile_index,
                    Err(JpegToHtj2kError::Validation(
                        "9/7 precomputed batch encode returned the wrong tile count",
                    )),
                )
            })
            .collect();
    }

    records
        .into_iter()
        .zip(codestreams)
        .enumerate()
        .map(|(batch_index, (record, codestream))| {
            let encode_measurement = (batch_index == 0).then_some((
                encode_dispatch_before,
                encode_dispatch_after,
                encode_us,
            ));
            (
                record.tile_index,
                encoded_float97_precomputed_batch_record(
                    record,
                    codestream,
                    options,
                    encode_measurement,
                ),
            )
        })
        .collect()
}

pub(super) fn encoded_float97_precomputed_batch_record(
    record: Float97PrecomputedBatchRecord,
    codestream: Vec<u8>,
    options: &JpegToHtj2kOptions,
    encode_measurement: Option<(J2kEncodeDispatchReport, J2kEncodeDispatchReport, u128)>,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let Float97PrecomputedBatchRecord {
        jpeg,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        float_validation_actual,
        float_validation_expected,
        mut timings,
        ..
    } = record;

    if let Some((encode_dispatch_before, encode_dispatch_after, encode_us)) = encode_measurement {
        record_encode_dispatch_delta(&mut timings, encode_dispatch_before, encode_dispatch_after);
        timings.htj2k_encode_us = encode_us;
    }
    let encode_us = timings.htj2k_encode_us;
    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: None,
            integer_reference_metrics: None,
            decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(all_unit_sampled, options.coefficient_path),
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: 0,
            encode_us,
            timings,
        },
    })
}

pub(super) fn encode_integer_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: IntegerBatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let mut timings = tile.timings;
    let components = tile
        .precomputed_components
        .into_iter()
        .map(|component| {
            component.ok_or(JpegToHtj2kError::Validation(
                "integer batch transcode did not produce all components",
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let encode_start = Instant::now();
    let precomputed = PrecomputedHtj2k53Image {
        width: tile.jpeg.width,
        height: tile.jpeg.height,
        bit_depth: 8,
        signed: false,
        components,
    };
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_precomputed = precomputed;
    let codestream = {
        let native_encode_options = options.encode_options.to_native();
        encode_precomputed_htj2k_53_with_accelerator(
            &native_precomputed,
            &native_encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)?
    };
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let integer_reference_metrics = if options.validate_against_integer_reference {
        Some(error_metrics_i32(
            &tile.integer_validation_actual,
            &tile.integer_validation_expected,
        )?)
    } else {
        None
    };
    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &tile.float_validation_actual,
            &tile.float_validation_expected,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: tile.jpeg.width,
            height: tile.jpeg.height,
            component_count: tile.jpeg.components.len(),
            components: tile.component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: integer_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            integer_reference_metrics,
            decomposition_levels: tile.decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(tile.all_unit_sampled, options.coefficient_path),
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: 0,
            encode_us,
            timings,
        },
    })
}

#[allow(clippy::too_many_lines)]
pub(super) fn encode_float97_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: Float97BatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let Float97BatchTile {
        jpeg,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        preencoded_compact_payload,
        preencoded_compact_components,
        preencoded_components,
        prequantized_components,
        float_validation_actual,
        float_validation_expected,
        mut timings,
        ..
    } = tile;

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let codestream = {
        let native_encode_options = options.encode_options.to_native();
        if preencoded_compact_components.iter().any(Option::is_some) {
            let components = preencoded_compact_components
                .into_iter()
                .map(|component| {
                    component.ok_or(JpegToHtj2kError::Validation(
                        "9/7 compact preencoded batch transcode did not produce all components",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let preencoded = PreencodedHtj2k97CompactImage {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                payload: preencoded_compact_payload,
                components,
            };
            encode_preencoded_htj2k_97_compact_owned_with_accelerator(
                preencoded,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        } else if preencoded_components.iter().any(Option::is_some) {
            let components = preencoded_components
                .into_iter()
                .map(|component| {
                    component.ok_or(JpegToHtj2kError::Validation(
                        "9/7 preencoded batch transcode did not produce all components",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let preencoded = PreencodedHtj2k97Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            encode_preencoded_htj2k_97_owned_with_accelerator(
                preencoded,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        } else if prequantized_components.iter().any(Option::is_some) {
            let components = prequantized_components
                .into_iter()
                .map(|component| {
                    component.ok_or(JpegToHtj2kError::Validation(
                        "9/7 code-block batch transcode did not produce all components",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let prequantized = PrequantizedHtj2k97Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            let native_prequantized = prequantized;
            encode_prequantized_htj2k_97_with_accelerator(
                &native_prequantized,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        } else {
            let components = precomputed_components
                .into_iter()
                .map(|component| {
                    component.ok_or(JpegToHtj2kError::Validation(
                        "9/7 batch transcode did not produce all components",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let precomputed = PrecomputedHtj2k97Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            let native_precomputed = precomputed;
            encode_precomputed_htj2k_97_with_accelerator(
                &native_precomputed,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        }
    };
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: None,
            integer_reference_metrics: None,
            decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(all_unit_sampled, options.coefficient_path),
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: 0,
            encode_us,
            timings,
        },
    })
}
