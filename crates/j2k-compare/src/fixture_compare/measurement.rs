// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    active_decoders, batch_input_copy_count, decode_batch, decode_mixed_batch, measurement_row,
    mixed_measurement_row, mixed_skip_reason, mixed_skip_row, sample_stats, skip_reason, skip_row,
    usize_to_f64, ActiveMeasurement, ActiveMixedMeasurement, BatchInputs, BenchmarkMode,
    FixtureCase, Instant, Measurement, MixedFixtureBatch, NonZeroUsize,
};

pub(super) fn measure_case_batch_rows(
    benchmark_mode: BenchmarkMode,
    case: &FixtureCase,
    repeats: usize,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
    case_index: usize,
    batch_index: usize,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut active = Vec::new();
    for decoder in active_decoders() {
        if let Some(reason) = skip_reason(benchmark_mode, decoder, case) {
            rows.push(skip_row(
                benchmark_mode,
                decoder,
                case,
                repeats,
                batch_size,
                reason,
            ));
        } else {
            active.push(ActiveMeasurement {
                decoder,
                batch_inputs: BatchInputs::new(
                    &case.bytes,
                    batch_size,
                    batch_input_copy_count(batch_size),
                ),
                samples_us: Vec::with_capacity(repeats),
                decoded_bytes_per_repeat: None,
            });
        }
    }

    if active.is_empty() {
        return Ok(rows);
    }

    for repeat in 0..repeats {
        let offset = (case_index + batch_index + repeat) % active.len();
        for step in 0..active.len() {
            let active_index = (offset + step) % active.len();
            let active_measurement = &mut active[active_index];
            let started = Instant::now();
            let output = decode_batch(
                benchmark_mode,
                case,
                active_measurement.decoder,
                &active_measurement.batch_inputs,
                workers,
            )?;
            let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            std::hint::black_box(&output);
            let decoded_len = output.len();
            if let Some(expected_len) = active_measurement.decoded_bytes_per_repeat {
                if decoded_len != expected_len {
                    return Err(format!(
                        "{} {} decoded length changed between repeats: {} vs {} bytes",
                        case.name,
                        active_measurement.decoder.label(),
                        decoded_len,
                        expected_len
                    ));
                }
            } else {
                active_measurement.decoded_bytes_per_repeat = Some(decoded_len);
            }
            active_measurement.samples_us.push(elapsed_us);
        }
    }

    for active_measurement in active {
        let stats = sample_stats(&active_measurement.samples_us)?;
        rows.push(measurement_row(
            benchmark_mode,
            case,
            &Measurement {
                decoder: active_measurement.decoder,
                repeats,
                batch_size,
                median_us: stats.median,
                mean_us: stats.mean,
                tiles_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
                decoded_bytes_per_repeat: active_measurement
                    .decoded_bytes_per_repeat
                    .ok_or_else(|| "missing decoded length for measured decoder".to_string())?,
                samples_us: active_measurement.samples_us,
            },
        ));
    }

    Ok(rows)
}

pub(super) fn measure_mixed_batch_rows(
    benchmark_mode: BenchmarkMode,
    mixed_batch: &MixedFixtureBatch,
    repeats: usize,
    batch_size: usize,
    workers: Option<NonZeroUsize>,
    mixed_index: usize,
    batch_index: usize,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut active = Vec::new();
    for decoder in active_decoders() {
        if let Some(reason) = mixed_skip_reason(benchmark_mode, decoder, mixed_batch) {
            rows.push(mixed_skip_row(
                benchmark_mode,
                decoder,
                mixed_batch,
                repeats,
                batch_size,
                reason,
            ));
        } else {
            active.push(ActiveMixedMeasurement {
                decoder,
                samples_us: Vec::with_capacity(repeats),
                decoded_bytes_per_repeat: None,
            });
        }
    }

    if active.is_empty() {
        return Ok(rows);
    }

    for repeat in 0..repeats {
        let offset = (mixed_index + batch_index + repeat) % active.len();
        for step in 0..active.len() {
            let active_index = (offset + step) % active.len();
            let active_measurement = &mut active[active_index];
            let started = Instant::now();
            let output = decode_mixed_batch(
                benchmark_mode,
                mixed_batch,
                active_measurement.decoder,
                batch_size,
                workers,
            )?;
            let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            std::hint::black_box(&output);
            let decoded_len = output.len();
            if let Some(expected_len) = active_measurement.decoded_bytes_per_repeat {
                if decoded_len != expected_len {
                    return Err(format!(
                        "{} {} decoded length changed between repeats: {} vs {} bytes",
                        mixed_batch.name,
                        active_measurement.decoder.label(),
                        decoded_len,
                        expected_len
                    ));
                }
            } else {
                active_measurement.decoded_bytes_per_repeat = Some(decoded_len);
            }
            active_measurement.samples_us.push(elapsed_us);
        }
    }

    for active_measurement in active {
        let stats = sample_stats(&active_measurement.samples_us)?;
        rows.push(mixed_measurement_row(
            benchmark_mode,
            mixed_batch,
            &Measurement {
                decoder: active_measurement.decoder,
                repeats,
                batch_size,
                median_us: stats.median,
                mean_us: stats.mean,
                tiles_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
                decoded_bytes_per_repeat: active_measurement
                    .decoded_bytes_per_repeat
                    .ok_or_else(|| "missing decoded length for mixed decoder".to_string())?,
                samples_us: active_measurement.samples_us,
            },
        ));
    }

    Ok(rows)
}
