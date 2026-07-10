// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator,
    encode_prequantized_htj2k_97_with_accelerator, error_metrics_i32, transcode_path_name,
    CpuOnlyJ2kEncodeStageAccelerator, EncodedTranscode, Float97BatchTile,
    Float97PrecomputedBatchRecord, Instant, IntegerBatchTile, IntoParallelIterator,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, JpegToHtj2kError, JpegToHtj2kOptions,
    ParallelIterator, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification,
};

pub(in super::super) fn record_encode_dispatch_delta(
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

pub(in super::super) fn add_encode_timing_counters_from_result(
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

pub(in super::super) fn encode_integer_prepared_tiles<E: J2kEncodeStageAccelerator>(
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

pub(in super::super) fn encode_float97_prepared_tiles<E: J2kEncodeStageAccelerator>(
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

pub(in super::super) fn can_encode_float97_precomputed_tiles_batch(
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

pub(in super::super) fn encode_float97_precomputed_tiles_batch<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    let (records, images) = match prepare_float97_precomputed_batch(prepared_tiles) {
        Ok(prepared) => prepared,
        Err((tile_index, error)) => return vec![(tile_index, Err(error))],
    };

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_encode_options = options.encode_options.to_native();
    let codestreams = match encode_precomputed_htj2k_97_batch_with_accelerator(
        &images,
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

fn prepare_float97_precomputed_batch(
    prepared_tiles: Vec<Float97BatchTile>,
) -> Result<
    (
        Vec<Float97PrecomputedBatchRecord>,
        Vec<PrecomputedHtj2k97Image>,
    ),
    (usize, JpegToHtj2kError),
> {
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
        let components = precomputed_components
            .into_iter()
            .map(|component| {
                component.ok_or(JpegToHtj2kError::Validation(
                    "9/7 precomputed batch transcode did not produce all components",
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| (tile_index, error))?;
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

    Ok((records, images))
}

pub(in super::super) fn encoded_float97_precomputed_batch_record(
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

pub(in super::super) fn encode_integer_batch_tile<E: J2kEncodeStageAccelerator>(
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

struct Float97BatchEncodeInputs {
    width: u32,
    height: u32,
    precomputed_components: Vec<Option<PrecomputedHtj2k97Component>>,
    preencoded_compact_payload: Vec<u8>,
    preencoded_compact_components: Vec<Option<PreencodedHtj2k97CompactComponent>>,
    preencoded_components: Vec<Option<PreencodedHtj2k97Component>>,
    prequantized_components: Vec<Option<PrequantizedHtj2k97Component>>,
}

fn require_all_components<T>(
    components: Vec<Option<T>>,
    missing_component: &'static str,
) -> Result<Vec<T>, JpegToHtj2kError> {
    components
        .into_iter()
        .map(|component| component.ok_or(JpegToHtj2kError::Validation(missing_component)))
        .collect()
}

fn encode_float97_batch_components<E: J2kEncodeStageAccelerator>(
    inputs: Float97BatchEncodeInputs,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Result<Vec<u8>, JpegToHtj2kError> {
    let Float97BatchEncodeInputs {
        width,
        height,
        precomputed_components,
        preencoded_compact_payload,
        preencoded_compact_components,
        preencoded_components,
        prequantized_components,
    } = inputs;
    let native_encode_options = options.encode_options.to_native();

    if preencoded_compact_components.iter().any(Option::is_some) {
        let components = require_all_components(
            preencoded_compact_components,
            "9/7 compact preencoded batch transcode did not produce all components",
        )?;
        let preencoded = PreencodedHtj2k97CompactImage {
            width,
            height,
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
        .map_err(JpegToHtj2kError::Encode)
    } else if preencoded_components.iter().any(Option::is_some) {
        let components = require_all_components(
            preencoded_components,
            "9/7 preencoded batch transcode did not produce all components",
        )?;
        let preencoded = PreencodedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_preencoded_htj2k_97_owned_with_accelerator(
            preencoded,
            &native_encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)
    } else if prequantized_components.iter().any(Option::is_some) {
        let components = require_all_components(
            prequantized_components,
            "9/7 code-block batch transcode did not produce all components",
        )?;
        let prequantized = PrequantizedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_prequantized_htj2k_97_with_accelerator(
            &prequantized,
            &native_encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)
    } else {
        let components = require_all_components(
            precomputed_components,
            "9/7 batch transcode did not produce all components",
        )?;
        let precomputed = PrecomputedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_precomputed_htj2k_97_with_accelerator(
            &precomputed,
            &native_encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)
    }
}

pub(in super::super) fn encode_float97_batch_tile<E: J2kEncodeStageAccelerator>(
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
    let codestream = encode_float97_batch_components(
        Float97BatchEncodeInputs {
            width: jpeg.width,
            height: jpeg.height,
            precomputed_components,
            preencoded_compact_payload,
            preencoded_compact_components,
            preencoded_components,
            prequantized_components,
        },
        options,
        encode_accelerator,
    )?;
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
