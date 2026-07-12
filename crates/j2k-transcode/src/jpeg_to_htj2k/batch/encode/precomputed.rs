// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes,
    encoded_transcode_retained_bytes, error_metrics_i32_with_live_budget, map_encode_error,
    transcode_path_name, EncodedTranscode, Float97BatchTile, Float97PrecomputedBatchRecord,
    HostLiveBudget, Instant, J2kEncodeDispatchReport, J2kEncodeStageAccelerator, JpegToHtj2kError,
    JpegToHtj2kOptions, PrecomputedHtj2k97Image, TranscodeComponentReport, TranscodeReport,
    TranscodeValidationClassification,
};
use super::{record_encode_dispatch_delta, EncodedTileBatchResult, IndexedEncodedTile};
use crate::allocation::try_vec_with_capacity;

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

pub(super) fn encode_float97_precomputed_tiles_batch<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
    external_live_bytes: usize,
) -> EncodedTileBatchResult {
    let (records, images) = prepare_float97_precomputed_batch(prepared_tiles)?;

    let mut native_external = HostLiveBudget::process_cap();
    native_external.add_bytes(external_live_bytes)?;
    native_external.add_bytes(float97_records_retained_bytes(
        &records,
        records.capacity(),
    )?)?;
    let native_host_cap = native_external.remaining_bytes()?;

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_encode_options = options.encode_options.to_native()?;
    let codestreams =
        match encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes(
            images,
            &native_encode_options,
            encode_accelerator,
            native_host_cap,
        ) {
            Ok(codestreams) => codestreams,
            Err(error) => return Err(map_encode_error(error)),
        };
    let encode_dispatch_after = encode_accelerator.dispatch_report();
    let encode_us = encode_start.elapsed().as_micros();

    if codestreams.len() != records.len() {
        let mut output = try_vec_with_capacity(records.len())?;
        for record in records {
            output.push((
                record.tile_index,
                Err(JpegToHtj2kError::Validation(
                    "9/7 precomputed batch encode returned the wrong tile count",
                )),
            ));
        }
        return Ok(output);
    }

    let mut output = try_vec_with_capacity(records.len())?;
    let mut fixed_live = HostLiveBudget::process_cap();
    fixed_live.add_bytes(external_live_bytes)?;
    fixed_live.add_capacity::<Float97PrecomputedBatchRecord>(records.capacity())?;
    fixed_live.add_capacity::<Vec<u8>>(codestreams.capacity())?;
    fixed_live.add_capacity::<IndexedEncodedTile>(output.capacity())?;
    let mut remaining_records = float97_record_nested_bytes(&records)?;
    let mut remaining_codestreams = codestream_nested_bytes(&codestreams)?;
    let mut completed_outputs = 0usize;
    for (batch_index, (record, codestream)) in records.into_iter().zip(codestreams).enumerate() {
        let record_bytes = float97_record_retained_bytes(&record)?;
        let codestream_bytes = codestream.capacity();
        let mut metrics_external = fixed_live;
        metrics_external.add_bytes(remaining_records)?;
        metrics_external.add_bytes(remaining_codestreams)?;
        metrics_external.add_bytes(completed_outputs)?;
        let encode_measurement = (batch_index == 0).then_some((
            encode_dispatch_before,
            encode_dispatch_after,
            encode_us,
        ));
        let tile_index = record.tile_index;
        let encoded = encoded_float97_precomputed_batch_record(
            record,
            codestream,
            options,
            encode_measurement,
            metrics_external.live_bytes(),
        );
        remaining_records = remaining_records.checked_sub(record_bytes).ok_or(
            JpegToHtj2kError::InternalInvariant {
                what: "9/7 batch record live-byte accounting underflowed",
            },
        )?;
        remaining_codestreams = remaining_codestreams.checked_sub(codestream_bytes).ok_or(
            JpegToHtj2kError::InternalInvariant {
                what: "9/7 batch codestream live-byte accounting underflowed",
            },
        )?;
        if let Ok(encoded) = encoded.as_ref() {
            completed_outputs = completed_outputs
                .checked_add(encoded_transcode_retained_bytes(encoded)?)
                .ok_or(JpegToHtj2kError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?;
        }
        output.push((tile_index, encoded));
    }
    Ok(output)
}

fn float97_records_retained_bytes(
    records: &[Float97PrecomputedBatchRecord],
    outer_capacity: usize,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<Float97PrecomputedBatchRecord>(outer_capacity)?;
    budget.add_bytes(float97_record_nested_bytes(records)?)?;
    Ok(budget.live_bytes())
}

fn float97_record_nested_bytes(
    records: &[Float97PrecomputedBatchRecord],
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    for record in records {
        budget.add_bytes(float97_record_retained_bytes(record)?)?;
    }
    Ok(budget.live_bytes())
}

fn float97_record_retained_bytes(
    record: &Float97PrecomputedBatchRecord,
) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    budget.add_capacity::<TranscodeComponentReport>(record.component_reports.capacity())?;
    budget.add_capacity::<i32>(record.float_validation_actual.capacity())?;
    budget.add_capacity::<i32>(record.float_validation_expected.capacity())?;
    Ok(budget.live_bytes())
}

fn codestream_nested_bytes(codestreams: &[Vec<u8>]) -> Result<usize, JpegToHtj2kError> {
    let mut budget = HostLiveBudget::process_cap();
    for codestream in codestreams {
        budget.add_capacity::<u8>(codestream.capacity())?;
    }
    Ok(budget.live_bytes())
}

fn prepare_float97_precomputed_batch(
    prepared_tiles: Vec<Float97BatchTile>,
) -> Result<
    (
        Vec<Float97PrecomputedBatchRecord>,
        Vec<PrecomputedHtj2k97Image>,
    ),
    JpegToHtj2kError,
> {
    let mut records = try_vec_with_capacity(prepared_tiles.len())?;
    let mut images = try_vec_with_capacity(prepared_tiles.len())?;

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
        let mut components = try_vec_with_capacity(precomputed_components.len())?;
        for component in precomputed_components {
            components.push(component.ok_or(JpegToHtj2kError::Validation(
                "9/7 precomputed batch transcode did not produce all components",
            ))?);
        }
        images.push(PrecomputedHtj2k97Image {
            width: jpeg.width,
            height: jpeg.height,
            bit_depth: 8,
            signed: false,
            components,
        });
        let width = jpeg.width;
        let height = jpeg.height;
        let component_count = jpeg.components.len();
        drop(jpeg);
        records.push(Float97PrecomputedBatchRecord {
            tile_index,
            width,
            height,
            component_count,
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

fn encoded_float97_precomputed_batch_record(
    record: Float97PrecomputedBatchRecord,
    codestream: Vec<u8>,
    options: &JpegToHtj2kOptions,
    encode_measurement: Option<(J2kEncodeDispatchReport, J2kEncodeDispatchReport, u128)>,
    external_live_bytes: usize,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let Float97PrecomputedBatchRecord {
        width,
        height,
        component_count,
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
        Some(error_metrics_i32_with_live_budget(
            &float_validation_actual,
            &float_validation_expected,
            external_live_bytes,
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
