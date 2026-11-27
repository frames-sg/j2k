// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoding the entropy-coded data of one scan (T.87 Annex A and B).

use crate::bitio::{BitReader, ReadError};
use crate::context::{ScanState, J};
use crate::decoder::sink::LineSink;
use crate::error::JpegLsError;
use crate::golomb;
use crate::info::InterleaveMode;
use crate::markers::{next_marker, Scan, RST0};

/// Why a line failed to decode; the caller adds the scan and line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineError {
    Truncated,
    Invalid,
}

impl From<ReadError> for LineError {
    fn from(error: ReadError) -> Self {
        match error {
            ReadError::Truncated => Self::Truncated,
            ReadError::InvalidCode => Self::Invalid,
        }
    }
}

/// Decode one scan into `sink`, checking its restart markers and that no
/// entropy-coded data is left over.
pub(crate) fn decode_scan(
    data: &[u8],
    scan: &Scan,
    scan_index: usize,
    width: usize,
    height: u32,
    sink: &mut dyn LineSink,
) -> Result<(), JpegLsError> {
    let mut state = ScanState::new(scan.params, width, scan.components.len())
        .map_err(|error| JpegLsError::AllocationFailed { bytes: error.bytes })?;
    let interval = if scan.restart_interval == 0 {
        height
    } else {
        scan.restart_interval
    };
    let mut pos = scan.data_offset;
    let mut line = 0_u32;
    let mut restart = 0_u8;
    loop {
        let end = next_marker(data, pos);
        let segment = data.get(pos..end).unwrap_or_default();
        let mut reader = BitReader::new(segment);
        let lines = interval.min(height - line);
        for _ in 0..lines {
            decode_line(&mut state, &mut reader, scan.interleave).map_err(|error| match error {
                LineError::Truncated => JpegLsError::Truncated { offset: end },
                LineError::Invalid => JpegLsError::InvalidEntropyData {
                    scan: scan_index,
                    line,
                },
            })?;
            for (slot, &component) in scan.components.iter().enumerate() {
                let samples = state.current.get(slot).map_or(&[][..], |samples| {
                    samples.get(1..=width).unwrap_or_default()
                });
                sink.store(line, component, samples)?;
            }
            state.end_line();
            line += 1;
        }
        if reader.remaining_bits() >= 8 {
            return Err(JpegLsError::TrailingEntropyData { scan: scan_index });
        }
        if line == height {
            if is_restart_marker(data, end) {
                return Err(JpegLsError::UnexpectedMarker {
                    marker: data.get(end + 1).copied().unwrap_or(0),
                    offset: end,
                });
            }
            return Ok(());
        }
        if end + 2 > data.len() {
            return Err(JpegLsError::Truncated { offset: end });
        }
        let expected = RST0 + restart % 8;
        if data.get(end..end + 2) != Some(&[0xFF, expected]) {
            return Err(JpegLsError::RestartMarker {
                expected: restart % 8,
                offset: end,
            });
        }
        restart = restart.wrapping_add(1);
        pos = end + 2;
        state.reset();
    }
}

fn is_restart_marker(data: &[u8], offset: usize) -> bool {
    data.get(offset) == Some(&0xFF)
        && data
            .get(offset + 1)
            .is_some_and(|code| (RST0..RST0 + 8).contains(code))
}

fn decode_line(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    interleave: InterleaveMode,
) -> Result<(), LineError> {
    state.start_line();
    let components = state.current.len();
    if components > 1 && interleave == InterleaveMode::Sample {
        return decode_sample_interleaved_line(state, reader);
    }
    for component in 0..components {
        decode_component_line(state, reader, component)?;
    }
    Ok(())
}

/// Decode one line of one component (`ILV` 0 or 1).
fn decode_component_line(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    component: usize,
) -> Result<(), LineError> {
    let width = state.width;
    let mut x = 0;
    while x < width {
        let (context, prediction) = state.sample_context(component, x);
        if let Some((index, sign)) = context {
            let sample = decode_regular(state, reader, index, sign, prediction)?;
            state.current[component][x + 1] = sample;
            x += 1;
        } else {
            x += decode_run(state, reader, component, x)?;
        }
    }
    Ok(())
}

/// Decode one line of all components, one pixel at a time (`ILV` 2).
fn decode_sample_interleaved_line(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
) -> Result<(), LineError> {
    let width = state.width;
    let components = state.current.len();
    let mut x = 0;
    while x < width {
        let contexts = state.pixel_contexts(x);
        if contexts
            .iter()
            .take(components)
            .all(|context| context.0.is_none())
        {
            x += decode_pixel_run(state, reader, x)?;
            continue;
        }
        for (slot, &(context, prediction)) in contexts.iter().enumerate().take(components) {
            // In a pixel outside run mode, a component whose gradients are
            // all zero uses context 0 with a positive sign, as CharLS does.
            let (index, sign) = context.unwrap_or((0, 1));
            let sample = decode_regular(state, reader, index, sign, prediction)?;
            state.current[slot][x + 1] = sample;
        }
        x += 1;
    }
    Ok(())
}

/// Decode a regular-mode sample (T.87 A.4 to A.6).
fn decode_regular(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    index: usize,
    sign: i32,
    prediction: i32,
) -> Result<i32, LineError> {
    let params = state.params;
    let context = state.regular.get_mut(index).ok_or(LineError::Invalid)?;
    let k = context.golomb_k();
    let prediction = params.clamp_sample(prediction + sign * context.bias());
    let mapped = golomb::read(reader, k, params.limit, params.qbpp)?;
    let inverted = params.near == 0 && k == 0 && context.inverts_mapping();
    let error = checked_error(golomb::unmap_error(mapped, inverted), params.range)?;
    context.update(error, params.near, params.reset);
    Ok(params.reconstruct(prediction, sign * error))
}

/// Reject an error value no valid encoder produces, which keeps the context
/// counters bounded on corrupt data.
fn checked_error(error: i64, range: i32) -> Result<i32, LineError> {
    i32::try_from(error)
        .ok()
        .filter(|error| error.unsigned_abs() <= range.unsigned_abs())
        .ok_or(LineError::Invalid)
}

/// Decode a run length (T.87 A.7.1). Returns the run length and whether the
/// run reached the end of the line.
fn decode_run_length(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    run_slot: usize,
    remaining: usize,
) -> Result<(usize, bool), LineError> {
    let mut length = 0;
    while reader.read_bit()? {
        let chunk = 1_usize << J[state.run_index[run_slot]];
        let step = chunk.min(remaining - length);
        length += step;
        if step == chunk {
            state.increment_run_index(run_slot);
        }
        if length == remaining {
            return Ok((length, true));
        }
    }
    let bits = J[state.run_index[run_slot]];
    length += reader.read_bits(bits)? as usize;
    if length >= remaining {
        return Err(LineError::Invalid);
    }
    Ok((length, false))
}

/// Decode a run and its interruption sample in one component. Returns the
/// number of samples decoded.
fn decode_run(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    component: usize,
    x: usize,
) -> Result<usize, LineError> {
    let width = state.width;
    let ra = state.current[component][x];
    let (length, end_of_line) = decode_run_length(state, reader, component, width - x)?;
    state.current[component][x + 1..=x + length].fill(ra);
    if end_of_line {
        return Ok(length);
    }
    let at = x + length;
    let rb = state.previous[component][at + 1];
    let params = state.params;
    let run_index = state.run_index[component];
    let (run_type, prediction, sign) = if (ra - rb).abs() <= params.near {
        (1, ra, 1)
    } else {
        (0, rb, if rb > ra { 1 } else { -1 })
    };
    let error = decode_interruption_error(state, reader, run_type, run_index)?;
    state.current[component][at + 1] = params.reconstruct(prediction, sign * error);
    state.decrement_run_index(component);
    Ok(length + 1)
}

/// Decode a run over whole pixels and its interruption pixel (`ILV` 2).
fn decode_pixel_run(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    x: usize,
) -> Result<usize, LineError> {
    let width = state.width;
    let (length, end_of_line) = decode_run_length(state, reader, 0, width - x)?;
    for line in &mut state.current {
        let ra = line[x];
        line[x + 1..=x + length].fill(ra);
    }
    if end_of_line {
        return Ok(length);
    }
    let at = x + length;
    let params = state.params;
    let run_index = state.run_index[0];
    for slot in 0..state.current.len() {
        let ra = state.current[slot][x];
        let rb = state.previous[slot][at + 1];
        let sign = if rb >= ra { 1 } else { -1 };
        let error = decode_interruption_error(state, reader, 0, run_index)?;
        state.current[slot][at + 1] = params.reconstruct(rb, sign * error);
    }
    state.decrement_run_index(0);
    Ok(length + 1)
}

/// Decode the error of a run-interruption sample (T.87 A.7.2).
fn decode_interruption_error(
    state: &mut ScanState,
    reader: &mut BitReader<'_>,
    run_type: usize,
    run_index: usize,
) -> Result<i32, LineError> {
    let params = state.params;
    let context = state.run.get_mut(run_type).ok_or(LineError::Invalid)?;
    let k = context.golomb_k();
    let limit = params.limit - J[run_index] - 1;
    let mapped = golomb::read(reader, k, limit, params.qbpp)?;
    let error = context.unmap(mapped + context.run_type(), k);
    let checked = checked_error(error, params.range)?;
    context.update(error, mapped, params.reset);
    Ok(checked)
}
