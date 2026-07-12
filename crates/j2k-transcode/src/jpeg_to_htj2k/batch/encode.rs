// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes,
    encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes,
    encoded_transcode_retained_bytes, error_metrics_i32_with_live_budget, map_encode_error,
    transcode_path_name, CpuOnlyJ2kEncodeStageAccelerator, EncodedTranscode, Float97BatchTile,
    HostLiveBudget, Instant, IntegerBatchTile, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    JpegToHtj2kError, JpegToHtj2kOptions, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification,
};
use crate::allocation::try_vec_with_capacity;
use crate::TranscodeValidationMetrics;

mod precomputed;
use self::precomputed::{
    can_encode_float97_precomputed_tiles_batch, encode_float97_precomputed_tiles_batch,
};
pub(super) mod live;
use self::live::{
    checked_batch_live_bytes, float97_tile_retained_bytes, float97_tiles_nested_bytes,
    integer_tile_retained_bytes, integer_tiles_nested_bytes,
};
mod float97_input;
use self::float97_input::{select_float97_batch_encoding, Float97BatchEncodingInput};

type IndexedEncodedTile = (usize, Result<EncodedTranscode, JpegToHtj2kError>);
type EncodedTileBatchResult = Result<Vec<IndexedEncodedTile>, JpegToHtj2kError>;

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
    external_live_bytes: usize,
) -> EncodedTileBatchResult {
    let mut output = try_vec_with_capacity(prepared_tiles.len())?;
    let cpu_only = encode_accelerator.prefer_parallel_cpu_tile_encode();
    let mut fixed_live = HostLiveBudget::process_cap();
    fixed_live.add_bytes(external_live_bytes)?;
    fixed_live.add_capacity::<IntegerBatchTile>(prepared_tiles.capacity())?;
    fixed_live.add_capacity::<IndexedEncodedTile>(output.capacity())?;
    let mut remaining_tiles = integer_tiles_nested_bytes(&prepared_tiles)?;
    let mut completed_outputs = 0usize;
    for prepared in prepared_tiles {
        let tile_index = prepared.tile_index;
        let tile_bytes = integer_tile_retained_bytes(&prepared)?;
        remaining_tiles =
            remaining_tiles
                .checked_sub(tile_bytes)
                .ok_or(JpegToHtj2kError::InternalInvariant {
                    what: "integer batch tile live-byte accounting underflowed",
                })?;
        let tile_external = checked_batch_live_bytes(
            fixed_live.live_bytes(),
            remaining_tiles,
            completed_outputs,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let encoded = if cpu_only {
            let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
            encode_integer_batch_tile(prepared, options, &mut cpu_accelerator, tile_external)
        } else {
            encode_integer_batch_tile(prepared, options, encode_accelerator, tile_external)
        };
        if let Ok(encoded) = encoded.as_ref() {
            completed_outputs = checked_completed_output_bytes(completed_outputs, encoded)?;
        }
        output.push((tile_index, encoded));
    }
    Ok(output)
}

pub(in super::super) fn encode_float97_prepared_tiles<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    external_live_bytes: usize,
) -> EncodedTileBatchResult {
    if !encode_accelerator.prefer_parallel_cpu_tile_encode()
        && can_encode_float97_precomputed_tiles_batch(&prepared_tiles, options)
    {
        return encode_float97_precomputed_tiles_batch(
            prepared_tiles,
            options,
            encode_accelerator,
            external_live_bytes,
        );
    }

    let mut output = try_vec_with_capacity(prepared_tiles.len())?;
    let cpu_only = encode_accelerator.prefer_parallel_cpu_tile_encode();
    let mut fixed_live = HostLiveBudget::process_cap();
    fixed_live.add_bytes(external_live_bytes)?;
    fixed_live.add_capacity::<Float97BatchTile>(prepared_tiles.capacity())?;
    fixed_live.add_capacity::<IndexedEncodedTile>(output.capacity())?;
    let mut remaining_tiles = float97_tiles_nested_bytes(&prepared_tiles)?;
    let mut completed_outputs = 0usize;
    for prepared in prepared_tiles {
        let tile_index = prepared.tile_index;
        let tile_bytes = float97_tile_retained_bytes(&prepared)?;
        remaining_tiles =
            remaining_tiles
                .checked_sub(tile_bytes)
                .ok_or(JpegToHtj2kError::InternalInvariant {
                    what: "9/7 batch tile live-byte accounting underflowed",
                })?;
        let tile_external = checked_batch_live_bytes(
            fixed_live.live_bytes(),
            remaining_tiles,
            completed_outputs,
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?;
        let encoded = if cpu_only {
            let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
            encode_float97_batch_tile(prepared, options, &mut cpu_accelerator, tile_external)
        } else {
            encode_float97_batch_tile(prepared, options, encode_accelerator, tile_external)
        };
        if let Ok(encoded) = encoded.as_ref() {
            completed_outputs = checked_completed_output_bytes(completed_outputs, encoded)?;
        }
        output.push((tile_index, encoded));
    }
    Ok(output)
}

fn checked_completed_output_bytes(
    completed: usize,
    encoded: &EncodedTranscode,
) -> Result<usize, JpegToHtj2kError> {
    completed
        .checked_add(encoded_transcode_retained_bytes(encoded)?)
        .ok_or(JpegToHtj2kError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
}

struct IntegerBatchValidationOwners {
    float_actual: Vec<i32>,
    float_expected: Vec<i32>,
    integer_actual: Vec<i32>,
    integer_expected: Vec<i32>,
}

#[derive(Clone, Copy)]
struct IntegerBatchValidationRequest<'a> {
    options: &'a JpegToHtj2kOptions,
    batch_external_bytes: usize,
    component_report_capacity: usize,
    codestream_capacity: usize,
    native_external: HostLiveBudget,
}

fn integer_batch_validation_metrics(
    request: IntegerBatchValidationRequest<'_>,
    owners: IntegerBatchValidationOwners,
) -> Result<
    (
        Option<TranscodeValidationMetrics>,
        Option<TranscodeValidationMetrics>,
    ),
    JpegToHtj2kError,
> {
    let IntegerBatchValidationOwners {
        float_actual,
        float_expected,
        integer_actual,
        integer_expected,
    } = owners;
    let float_metrics = if request.options.validate_against_float_reference {
        let mut metrics_external = request.native_external;
        metrics_external.add_capacity::<u8>(request.codestream_capacity)?;
        Some(error_metrics_i32_with_live_budget(
            &float_actual,
            &float_expected,
            metrics_external.live_bytes(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?)
    } else {
        None
    };
    drop(float_actual);
    drop(float_expected);

    let integer_metrics = if request.options.validate_against_integer_reference {
        let mut metrics_external = HostLiveBudget::process_cap();
        metrics_external.add_bytes(request.batch_external_bytes)?;
        metrics_external
            .add_capacity::<super::TranscodeComponentReport>(request.component_report_capacity)?;
        metrics_external.add_capacity::<u8>(request.codestream_capacity)?;
        metrics_external.add_capacity::<i32>(integer_actual.capacity())?;
        metrics_external.add_capacity::<i32>(integer_expected.capacity())?;
        if let Some(metrics) = float_metrics.as_ref() {
            metrics_external.add_bytes(metrics.absolute_error_histogram.retained_bytes()?)?;
        }
        Some(error_metrics_i32_with_live_budget(
            &integer_actual,
            &integer_expected,
            metrics_external.live_bytes(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?)
    } else {
        None
    };
    Ok((float_metrics, integer_metrics))
}

pub(in super::super) fn encode_integer_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: IntegerBatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    batch_external_bytes: usize,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let IntegerBatchTile {
        jpeg,
        component_sampling: _,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        float_validation_actual,
        float_validation_expected,
        integer_validation_actual,
        integer_validation_expected,
        mut timings,
        ..
    } = tile;
    let (width, height, component_count) = (jpeg.width, jpeg.height, jpeg.components.len());
    drop(jpeg);
    let components = require_all_components(
        precomputed_components,
        "integer batch transcode did not produce all components",
    )?;
    let encode_start = Instant::now();
    let precomputed = PrecomputedHtj2k53Image {
        width,
        height,
        bit_depth: 8,
        signed: false,
        components,
    };
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let mut native_external = HostLiveBudget::process_cap();
    native_external.add_bytes(batch_external_bytes)?;
    native_external
        .add_capacity::<super::TranscodeComponentReport>(component_reports.capacity())?;
    for capacity in [
        float_validation_actual.capacity(),
        float_validation_expected.capacity(),
        integer_validation_actual.capacity(),
        integer_validation_expected.capacity(),
    ] {
        native_external.add_capacity::<i32>(capacity)?;
    }
    let native_host_cap = native_external.remaining_bytes()?;
    let codestream = {
        let native_encode_options = options.encode_options.to_native()?;
        encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(
            &precomputed,
            &native_encode_options,
            encode_accelerator,
            native_host_cap,
        )
        .map_err(map_encode_error)?
    };
    drop(precomputed);
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let (float_reference_metrics, integer_reference_metrics) = integer_batch_validation_metrics(
        IntegerBatchValidationRequest {
            options,
            batch_external_bytes,
            component_report_capacity: component_reports.capacity(),
            codestream_capacity: codestream.capacity(),
            native_external,
        },
        IntegerBatchValidationOwners {
            float_actual: float_validation_actual,
            float_expected: float_validation_expected,
            integer_actual: integer_validation_actual,
            integer_expected: integer_validation_expected,
        },
    )?;

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width,
            height,
            component_count,
            components: component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: integer_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            integer_reference_metrics,
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
    let mut required = try_vec_with_capacity(components.len())?;
    for component in components {
        required.push(component.ok_or(JpegToHtj2kError::Validation(missing_component))?);
    }
    Ok(required)
}

fn encode_float97_batch_components<E: J2kEncodeStageAccelerator>(
    inputs: Float97BatchEncodeInputs,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    max_host_bytes: usize,
) -> Result<Vec<u8>, JpegToHtj2kError> {
    let native_encode_options = options.encode_options.to_native()?;
    match select_float97_batch_encoding(inputs)? {
        Float97BatchEncodingInput::Compact(preencoded) => {
            encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes(
                preencoded,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)
        }
        Float97BatchEncodingInput::Preencoded(preencoded) => {
            encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes(
                preencoded,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)
        }
        Float97BatchEncodingInput::Prequantized(prequantized) => {
            encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes(
                &prequantized,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)
        }
        Float97BatchEncodingInput::Precomputed(precomputed) => {
            encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(
                &precomputed,
                &native_encode_options,
                encode_accelerator,
                max_host_bytes,
            )
            .map_err(map_encode_error)
        }
    }
}

pub(in super::super) fn encode_float97_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: Float97BatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    batch_external_bytes: usize,
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
    let width = jpeg.width;
    let height = jpeg.height;
    let component_count = jpeg.components.len();
    drop(jpeg);

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let mut native_external = HostLiveBudget::process_cap();
    native_external.add_bytes(batch_external_bytes)?;
    native_external
        .add_capacity::<super::TranscodeComponentReport>(component_reports.capacity())?;
    native_external.add_capacity::<i32>(float_validation_actual.capacity())?;
    native_external.add_capacity::<i32>(float_validation_expected.capacity())?;
    let native_host_cap = native_external.remaining_bytes()?;
    let codestream = encode_float97_batch_components(
        Float97BatchEncodeInputs {
            width,
            height,
            precomputed_components,
            preencoded_compact_payload,
            preencoded_compact_components,
            preencoded_components,
            prequantized_components,
        },
        options,
        encode_accelerator,
        native_host_cap,
    )?;
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let float_reference_metrics = if options.validate_against_float_reference {
        let mut metrics_external = native_external;
        metrics_external.add_capacity::<u8>(codestream.capacity())?;
        Some(error_metrics_i32_with_live_budget(
            &float_validation_actual,
            &float_validation_expected,
            metrics_external.live_bytes(),
            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width,
            height,
            component_count,
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
