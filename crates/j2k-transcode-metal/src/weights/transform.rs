// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{try_host_vec_from_slice, try_host_vec_with_capacity};

use super::{
    error::allocation_error,
    shared::{high_len, low_len, WaveletKind, ALPHA, BETA, DELTA, GAMMA, INV_KAPPA, KAPPA},
    SparseWeightRowsError,
};

pub(super) struct DwtOneDimensional {
    pub(super) low: Vec<f64>,
    pub(super) high: Vec<f64>,
}

pub(super) fn try_linearized_from_sample_slice(
    samples: &[f64],
    wavelet: WaveletKind,
) -> Result<DwtOneDimensional, SparseWeightRowsError> {
    let low = try_host_vec_with_capacity(low_len(samples.len())).map_err(allocation_error)?;
    let high = try_host_vec_with_capacity(high_len(samples.len())).map_err(allocation_error)?;
    match wavelet {
        WaveletKind::Reversible53 => Ok(linearized_53_with_buffers(samples, low, high)),
        WaveletKind::Irreversible97 => {
            let lifted = try_host_vec_from_slice(samples).map_err(allocation_error)?;
            Ok(linearized_97_with_buffers(lifted, low, high))
        }
    }
}

fn linearized_53_with_buffers(
    samples: &[f64],
    mut low: Vec<f64>,
    mut high: Vec<f64>,
) -> DwtOneDimensional {
    for odd_idx in (1..samples.len()).step_by(2) {
        let left = samples[odd_idx - 1];
        let right = samples.get(odd_idx + 1).copied().unwrap_or(left);
        high.push(samples[odd_idx] - ((left + right) * 0.5));
    }

    for even_idx in (0..samples.len()).step_by(2) {
        let current = samples[even_idx];
        let even_output_idx = even_idx / 2;
        let left_high = even_output_idx.checked_sub(1).and_then(|idx| high.get(idx));
        let right_high = high.get(even_output_idx);
        let update = match (left_high, right_high) {
            (Some(left), Some(right)) => (*left + *right) * 0.25,
            (None, Some(right)) => *right * 0.5,
            (Some(left), None) => *left * 0.5,
            (None, None) => 0.0,
        };
        low.push(current + update);
    }

    DwtOneDimensional { low, high }
}

fn linearized_97_with_buffers(
    mut lifted: Vec<f64>,
    mut low: Vec<f64>,
    mut high: Vec<f64>,
) -> DwtOneDimensional {
    forward_lift_97(&mut lifted);
    for (index, value) in lifted.into_iter().enumerate() {
        if index.is_multiple_of(2) {
            low.push(value);
        } else {
            high.push(value);
        }
    }
    DwtOneDimensional { low, high }
}

fn forward_lift_97(data: &mut [f64]) {
    let sample_count = data.len();
    if sample_count < 2 {
        return;
    }
    let last_even = if sample_count.is_multiple_of(2) {
        sample_count - 2
    } else {
        sample_count - 1
    };

    lift_odd(data, last_even, ALPHA);
    lift_even(data, BETA);
    lift_odd(data, last_even, GAMMA);
    lift_even(data, DELTA);
    for sample_idx in (0..sample_count).step_by(2) {
        data[sample_idx] *= INV_KAPPA;
    }
    for sample_idx in (1..sample_count).step_by(2) {
        data[sample_idx] *= KAPPA;
    }
}

fn lift_odd(data: &mut [f64], last_even: usize, coefficient: f64) {
    for sample_idx in (1..data.len()).step_by(2) {
        let left = data[sample_idx - 1];
        let right = if sample_idx + 1 < data.len() {
            data[sample_idx + 1]
        } else {
            data[last_even]
        };
        data[sample_idx] += coefficient * (left + right);
    }
}

fn lift_even(data: &mut [f64], coefficient: f64) {
    for sample_idx in (0..data.len()).step_by(2) {
        let left = if sample_idx > 0 {
            data[sample_idx - 1]
        } else {
            data[1]
        };
        let right = if sample_idx + 1 < data.len() {
            data[sample_idx + 1]
        } else {
            left
        };
        data[sample_idx] += coefficient * (left + right);
    }
}
