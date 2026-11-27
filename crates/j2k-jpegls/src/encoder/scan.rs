// SPDX-License-Identifier: MIT OR Apache-2.0

//! Encoding one scan (T.87 Annex A and B). Each step mirrors
//! `decoder::scan`, and the encoder keeps the reconstructed samples the
//! decoder will see.

use crate::allocation::{zeroed, AllocationError};
use crate::bitio::BitWriter;
use crate::context::{ScanState, J};
use crate::golomb;
use crate::info::InterleaveMode;
use crate::markers::RST0;
use crate::params::CodingParameters;

/// Interleaved input samples.
#[derive(Clone, Copy)]
pub(crate) enum SourceSamples<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// The input image and the scan being coded.
pub(crate) struct ScanSource<'a> {
    pub(crate) samples: SourceSamples<'a>,
    pub(crate) width: usize,
    pub(crate) height: u32,
    /// Components per pixel in `samples`.
    pub(crate) components: usize,
}

impl ScanSource<'_> {
    /// Copy one line of one component into `out`.
    fn load_line(&self, line: u32, component: usize, out: &mut [i32]) {
        let row_len = self.width * self.components;
        let start = line as usize * row_len;
        match self.samples {
            SourceSamples::U8(data) => {
                let row = data.get(start..start + row_len).unwrap_or_default();
                for (slot, &sample) in out
                    .iter_mut()
                    .zip(row.iter().skip(component).step_by(self.components))
                {
                    *slot = i32::from(sample);
                }
            }
            SourceSamples::U16(data) => {
                let row = data.get(start..start + row_len).unwrap_or_default();
                for (slot, &sample) in out
                    .iter_mut()
                    .zip(row.iter().skip(component).step_by(self.components))
                {
                    *slot = i32::from(sample);
                }
            }
        }
    }
}

/// One scan's coding choices.
pub(crate) struct ScanPlan<'c> {
    pub(crate) components: &'c [usize],
    pub(crate) interleave: InterleaveMode,
    pub(crate) params: CodingParameters,
    /// Lines per restart interval, or 0 for none.
    pub(crate) restart_interval: u32,
}

/// Append the entropy-coded data of one scan, with restart markers, to the
/// writer's output.
pub(crate) fn encode_scan(
    writer: &mut BitWriter,
    source: &ScanSource<'_>,
    plan: &ScanPlan<'_>,
) -> Result<(), AllocationError> {
    let width = source.width;
    let height = source.height;
    let mut state = ScanState::new(plan.params, width, plan.components.len())?;
    let mut lines_in = Vec::new();
    for _ in plan.components {
        lines_in.push(zeroed::<i32>(width)?);
    }
    let interval = if plan.restart_interval == 0 {
        height
    } else {
        plan.restart_interval
    };
    let mut line = 0_u32;
    let mut restart = 0_u8;
    loop {
        for _ in 0..interval.min(height - line) {
            for (input, &component) in lines_in.iter_mut().zip(plan.components) {
                source.load_line(line, component, input);
            }
            encode_line(&mut state, writer, &lines_in, plan.interleave);
            state.end_line();
            line += 1;
        }
        writer.finish_segment();
        if line == height {
            return Ok(());
        }
        writer
            .bytes_mut()
            .extend_from_slice(&[0xFF, RST0 + restart % 8]);
        restart = restart.wrapping_add(1);
        state.reset();
    }
}

fn encode_line(
    state: &mut ScanState,
    writer: &mut BitWriter,
    input: &[Vec<i32>],
    interleave: InterleaveMode,
) {
    state.start_line();
    if input.len() > 1 && interleave == InterleaveMode::Sample {
        encode_sample_interleaved_line(state, writer, input);
        return;
    }
    for (component, samples) in input.iter().enumerate() {
        encode_component_line(state, writer, component, samples);
    }
}

fn encode_component_line(
    state: &mut ScanState,
    writer: &mut BitWriter,
    component: usize,
    input: &[i32],
) {
    let mut x = 0;
    while x < state.width {
        let (context, prediction) = state.sample_context(component, x);
        if let Some((index, sign)) = context {
            let sample = encode_regular(state, writer, index, sign, prediction, input[x]);
            state.current[component][x + 1] = sample;
            x += 1;
        } else {
            x += encode_run(state, writer, component, x, input);
        }
    }
}

fn encode_sample_interleaved_line(
    state: &mut ScanState,
    writer: &mut BitWriter,
    input: &[Vec<i32>],
) {
    let mut x = 0;
    while x < state.width {
        let contexts = state.pixel_contexts(x);
        if contexts
            .iter()
            .take(input.len())
            .all(|context| context.0.is_none())
        {
            x += encode_pixel_run(state, writer, x, input);
            continue;
        }
        for (slot, (&(context, prediction), samples)) in contexts.iter().zip(input).enumerate() {
            let (index, sign) = context.unwrap_or((0, 1));
            let sample = encode_regular(state, writer, index, sign, prediction, samples[x]);
            state.current[slot][x + 1] = sample;
        }
        x += 1;
    }
}

/// Code a regular-mode sample and return its reconstruction.
fn encode_regular(
    state: &mut ScanState,
    writer: &mut BitWriter,
    index: usize,
    sign: i32,
    prediction: i32,
    sample: i32,
) -> i32 {
    let params = state.params;
    let context = &mut state.regular[index];
    let k = context.golomb_k();
    let prediction = params.clamp_sample(prediction + sign * context.bias());
    let error = params.reduce_error(params.quantize_error(sign * (sample - prediction)));
    let inverted = params.near == 0 && k == 0 && context.inverts_mapping();
    golomb::write(
        writer,
        golomb::map_error(error, inverted),
        k,
        params.limit,
        params.qbpp,
    );
    context.update(error, params.near, params.reset);
    params.reconstruct(prediction, sign * error)
}

fn encode_run_length(
    state: &mut ScanState,
    writer: &mut BitWriter,
    run_slot: usize,
    length: usize,
    end_of_line: bool,
) {
    let mut remaining = length;
    loop {
        let chunk = 1_usize << J[state.run_index[run_slot]];
        if remaining < chunk {
            break;
        }
        writer.put_bit(true);
        remaining -= chunk;
        state.increment_run_index(run_slot);
    }
    if end_of_line {
        if remaining > 0 {
            writer.put_bit(true);
        }
    } else {
        writer.put_bit(false);
        writer.put(
            u32::try_from(remaining).unwrap_or(u32::MAX),
            J[state.run_index[run_slot]],
        );
    }
}

/// Code a run and its interruption sample in one component. Returns the
/// number of samples coded.
fn encode_run(
    state: &mut ScanState,
    writer: &mut BitWriter,
    component: usize,
    x: usize,
    input: &[i32],
) -> usize {
    let params = state.params;
    let width = state.width;
    let ra = state.current[component][x];
    let mut length = 0;
    while x + length < width && (input[x + length] - ra).abs() <= params.near {
        state.current[component][x + length + 1] = ra;
        length += 1;
    }
    let end_of_line = x + length == width;
    encode_run_length(state, writer, component, length, end_of_line);
    if end_of_line {
        return length;
    }
    let at = x + length;
    let rb = state.previous[component][at + 1];
    let run_index = state.run_index[component];
    let (run_type, prediction, sign) = if (ra - rb).abs() <= params.near {
        (1, ra, 1)
    } else {
        (0, rb, if rb > ra { 1 } else { -1 })
    };
    let error = params.reduce_error(params.quantize_error(sign * (input[at] - prediction)));
    encode_interruption_error(state, writer, run_type, run_index, error);
    state.current[component][at + 1] = params.reconstruct(prediction, sign * error);
    state.decrement_run_index(component);
    length + 1
}

/// Code a run over whole pixels and its interruption pixel (`ILV` 2).
fn encode_pixel_run(
    state: &mut ScanState,
    writer: &mut BitWriter,
    x: usize,
    input: &[Vec<i32>],
) -> usize {
    let params = state.params;
    let width = state.width;
    let in_run = |state: &ScanState, at: usize| {
        input
            .iter()
            .zip(&state.current)
            .all(|(samples, line)| (samples[at] - line[x]).abs() <= params.near)
    };
    let mut length = 0;
    while x + length < width && in_run(state, x + length) {
        for line in &mut state.current {
            line[x + length + 1] = line[x];
        }
        length += 1;
    }
    let end_of_line = x + length == width;
    encode_run_length(state, writer, 0, length, end_of_line);
    if end_of_line {
        return length;
    }
    let at = x + length;
    let run_index = state.run_index[0];
    for (slot, samples) in input.iter().enumerate() {
        let ra = state.current[slot][x];
        let rb = state.previous[slot][at + 1];
        let sign = if rb >= ra { 1 } else { -1 };
        let error = params.reduce_error(params.quantize_error(sign * (samples[at] - rb)));
        encode_interruption_error(state, writer, 0, run_index, error);
        state.current[slot][at + 1] = params.reconstruct(rb, sign * error);
    }
    state.decrement_run_index(0);
    length + 1
}

/// Code the error of a run-interruption sample (T.87 A.7.2).
fn encode_interruption_error(
    state: &mut ScanState,
    writer: &mut BitWriter,
    run_type: usize,
    run_index: usize,
    error: i32,
) {
    let params = state.params;
    let context = &mut state.run[run_type];
    let k = context.golomb_k();
    let error = i64::from(error);
    let mapped = 2 * error.abs() - context.run_type() - i64::from(context.map_bit(error, k));
    let limit = params.limit - J[run_index] - 1;
    golomb::write(
        writer,
        u32::try_from(mapped).unwrap_or(0),
        k,
        limit,
        params.qbpp,
    );
    context.update(error, mapped, params.reset);
}
