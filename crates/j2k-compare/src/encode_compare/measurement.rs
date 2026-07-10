// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    command_template, fs, measurement_row, mixed_case_at, mixed_measurement_row, mixed_skip_row,
    run_encoder_once, sample_stats, skip_row, usize_to_f64, validate_case_encoder,
    EncodeMeasurementState, EncoderKind, EncoderTool, ImageCase, Instant, Measurement,
    MixedImageBatch, Path,
};

pub(super) fn measure_case_rows(
    case: &ImageCase,
    tools: &[EncoderTool],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut states = Vec::new();
    for tool in tools.iter().filter(|tool| tool.available) {
        validate_case_encoder(case, tool, work_dir)?;
        states.push(EncodeMeasurementState {
            tool,
            encoded_bytes_per_repeat: None,
            samples_us: Vec::with_capacity(repeats),
        });
    }
    measure_case_states(case, &mut states, repeats, batch_size, work_dir)?;
    for tool in tools {
        if !tool.available {
            rows.push(skip_row(
                tool.kind,
                case,
                repeats,
                batch_size,
                "encoder-tool-unavailable",
                command_template(tool.kind),
            ));
            continue;
        }
        let state = states
            .iter()
            .find(|state| state.tool.kind == tool.kind)
            .ok_or_else(|| format!("missing measurement for {}", tool.kind.label()))?;
        let measurement = measurement(
            repeats,
            batch_size,
            state.samples_us.clone(),
            state.encoded_bytes_per_repeat,
        )?;
        rows.push(measurement_row(
            tool.kind,
            case,
            &measurement,
            command_template(tool.kind),
        ));
    }
    Ok(rows)
}

pub(super) fn measure_mixed_rows(
    mixed_batch: &MixedImageBatch,
    tools: &[EncoderTool],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<Vec<String>, String> {
    let mut rows = Vec::new();
    let mut states = tools
        .iter()
        .filter(|tool| tool.available)
        .map(|tool| EncodeMeasurementState {
            tool,
            encoded_bytes_per_repeat: None,
            samples_us: Vec::with_capacity(repeats),
        })
        .collect::<Vec<_>>();
    measure_mixed_states(mixed_batch, &mut states, repeats, batch_size, work_dir)?;
    for tool in tools {
        if !tool.available {
            rows.push(mixed_skip_row(
                tool.kind,
                mixed_batch,
                repeats,
                batch_size,
                "encoder-tool-unavailable",
                command_template(tool.kind),
            ));
            continue;
        }
        let state = states
            .iter()
            .find(|state| state.tool.kind == tool.kind)
            .ok_or_else(|| format!("missing measurement for {}", tool.kind.label()))?;
        let measurement = measurement(
            repeats,
            batch_size,
            state.samples_us.clone(),
            state.encoded_bytes_per_repeat,
        )?;
        rows.push(mixed_measurement_row(
            tool.kind,
            mixed_batch,
            &measurement,
            command_template(tool.kind),
        ));
    }
    Ok(rows)
}

pub(super) fn measure_case_states(
    case: &ImageCase,
    states: &mut [EncodeMeasurementState<'_>],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<(), String> {
    if states.is_empty() {
        return Ok(());
    }
    for repeat in 0..repeats {
        let offset = repeat % states.len();
        for step in 0..states.len() {
            let index = (offset + step) % states.len();
            let state = &mut states[index];
            let (sample_us, encoded_bytes) = measure_case_encoder_once(
                case,
                state.tool,
                batch_size,
                work_dir,
                &format!("r{repeat}_e{step}"),
            )?;
            state.samples_us.push(sample_us);
            record_encoded_bytes(
                &mut state.encoded_bytes_per_repeat,
                encoded_bytes,
                state.tool.kind,
                &case.name,
            )?;
        }
    }
    Ok(())
}

pub(super) fn measure_mixed_states(
    mixed_batch: &MixedImageBatch,
    states: &mut [EncodeMeasurementState<'_>],
    repeats: usize,
    batch_size: usize,
    work_dir: &Path,
) -> Result<(), String> {
    if states.is_empty() {
        return Ok(());
    }
    for repeat in 0..repeats {
        let offset = repeat % states.len();
        for step in 0..states.len() {
            let index = (offset + step) % states.len();
            let state = &mut states[index];
            let (sample_us, encoded_bytes) = measure_mixed_encoder_once(
                mixed_batch,
                state.tool,
                batch_size,
                work_dir,
                &format!("mixed_r{repeat}_e{step}"),
            )?;
            state.samples_us.push(sample_us);
            record_encoded_bytes(
                &mut state.encoded_bytes_per_repeat,
                encoded_bytes,
                state.tool.kind,
                &mixed_batch.name,
            )?;
        }
    }
    Ok(())
}

pub(super) fn measure_case_encoder_once(
    case: &ImageCase,
    tool: &EncoderTool,
    batch_size: usize,
    work_dir: &Path,
    suffix: &str,
) -> Result<(f64, usize), String> {
    let started = Instant::now();
    let mut encoded_bytes = 0_usize;
    for index in 0..batch_size {
        let output = run_encoder_once(case, tool, work_dir, &format!("{suffix}_b{index}"))?;
        encoded_bytes = encoded_bytes
            .checked_add(encoded_file_len(&output)?)
            .ok_or_else(|| "encoded byte count overflow".to_string())?;
        std::hint::black_box(&output);
    }
    Ok((started.elapsed().as_secs_f64() * 1_000_000.0, encoded_bytes))
}

pub(super) fn measure_mixed_encoder_once(
    mixed_batch: &MixedImageBatch,
    tool: &EncoderTool,
    batch_size: usize,
    work_dir: &Path,
    suffix: &str,
) -> Result<(f64, usize), String> {
    let started = Instant::now();
    let mut encoded_bytes = 0_usize;
    for index in 0..batch_size {
        let case = mixed_case_at(mixed_batch, index);
        let output = run_encoder_once(case, tool, work_dir, &format!("{suffix}_b{index}"))?;
        encoded_bytes = encoded_bytes
            .checked_add(encoded_file_len(&output)?)
            .ok_or_else(|| "encoded byte count overflow".to_string())?;
        std::hint::black_box(&output);
    }
    Ok((started.elapsed().as_secs_f64() * 1_000_000.0, encoded_bytes))
}

fn encoded_file_len(path: &Path) -> Result<usize, String> {
    let len = fs::metadata(path)
        .map_err(|error| format!("metadata {}: {error}", path.display()))?
        .len();
    usize::try_from(len)
        .map_err(|_| format!("encoded file {} exceeds platform usize", path.display()))
}

pub(super) fn record_encoded_bytes(
    expected: &mut Option<usize>,
    actual: usize,
    encoder: EncoderKind,
    case_name: &str,
) -> Result<(), String> {
    if let Some(expected) = *expected {
        if actual != expected {
            return Err(format!(
                "{} {} encoded byte count changed: {} vs {expected}",
                encoder.label(),
                case_name,
                actual
            ));
        }
    } else {
        *expected = Some(actual);
    }
    Ok(())
}

pub(super) fn measurement(
    repeats: usize,
    batch_size: usize,
    samples_us: Vec<f64>,
    encoded_bytes_per_repeat: Option<usize>,
) -> Result<Measurement, String> {
    let stats = sample_stats(&samples_us)?;
    Ok(Measurement {
        repeats,
        batch_size,
        median_us: stats.median,
        mean_us: stats.mean,
        images_per_second_median: usize_to_f64(batch_size) / (stats.median / 1_000_000.0),
        encoded_bytes_per_repeat: encoded_bytes_per_repeat
            .ok_or_else(|| "missing encoded byte count".to_string())?,
        samples_us,
    })
}
