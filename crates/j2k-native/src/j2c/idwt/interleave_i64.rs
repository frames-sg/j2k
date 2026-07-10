// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::Decomposition;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::horizontal::filter_horizontal_i64;
use super::model::{IDWTInputI64, IDWTTempOutput};
use super::vertical::filter_vertical_i64;

pub(super) fn apply_level_i64(
    input: IDWTInputI64<'_>,
    target: &mut Vec<i64>,
    decomposition: &Decomposition,
    storage: &DecompositionStorage<'_>,
) -> IDWTTempOutput {
    interleave_samples_i64(input, decomposition, target, storage);

    if decomposition.rect.width() > 0 && decomposition.rect.height() > 0 {
        filter_horizontal_i64(target, decomposition.rect);
        filter_vertical_i64(target, decomposition.rect);
    }

    IDWTTempOutput {
        rect: decomposition.rect,
    }
}

fn interleave_samples_i64(
    input: IDWTInputI64<'_>,
    decomposition: &Decomposition,
    coefficients: &mut Vec<i64>,
    storage: &DecompositionStorage<'_>,
) {
    let width = decomposition.rect.width() as usize;
    let height = decomposition.rect.height() as usize;
    assert!(coefficients.capacity() >= width * height);
    coefficients.resize(width * height, 0);

    let IntRect {
        x0: u0,
        x1: u1,
        y0: v0,
        y1: v1,
    } = decomposition.rect;

    let ll = input.coefficients;
    let hl = &storage.coefficients_i64[storage.sub_bands[decomposition.sub_bands[0]]
        .coefficients
        .clone()];
    let lh = &storage.coefficients_i64[storage.sub_bands[decomposition.sub_bands[1]]
        .coefficients
        .clone()];
    let hh = &storage.coefficients_i64[storage.sub_bands[decomposition.sub_bands[2]]
        .coefficients
        .clone()];

    let num_u_low = (u1.div_ceil(2) - u0.div_ceil(2)) as usize;
    let num_u_high = (u1 / 2 - u0 / 2) as usize;
    let num_v_low = (v1.div_ceil(2) - v0.div_ceil(2)) as usize;
    let num_v_high = (v1 / 2 - v0 / 2) as usize;

    let (first_w, second_w) = if u0 % 2 == 0 {
        (num_u_low, num_u_high)
    } else {
        (num_u_high, num_u_low)
    };

    let even_row_start = if v0 % 2 == 0 { 0 } else { 1 };
    let odd_row_start = if v0 % 2 == 0 { 1 } else { 0 };

    let (first_even, second_even) = if u0 % 2 == 0 { (ll, hl) } else { (hl, ll) };
    interleave_rows_i64(
        first_even,
        second_even,
        first_w,
        second_w,
        coefficients,
        width,
        height,
        even_row_start,
        num_v_low,
    );

    let (first_odd, second_odd) = if u0 % 2 == 0 { (lh, hh) } else { (hh, lh) };
    interleave_rows_i64(
        first_odd,
        second_odd,
        first_w,
        second_w,
        coefficients,
        width,
        height,
        odd_row_start,
        num_v_high,
    );
}

fn interleave_rows_i64(
    first_band: &[i64],
    second_band: &[i64],
    first_w: usize,
    second_w: usize,
    output: &mut [i64],
    width: usize,
    height: usize,
    start_row: usize,
    num_rows: usize,
) {
    for v in 0..num_rows {
        let out_row = start_row + v * 2;
        if out_row >= height {
            break;
        }

        let first_row = &first_band[v * first_w..][..first_w];
        let second_row = &second_band[v * second_w..][..second_w];
        let out_slice = &mut output[out_row * width..][..width];

        interleave_row_i64(first_row, second_row, out_slice);
    }
}

fn interleave_row_i64(first: &[i64], second: &[i64], output: &mut [i64]) {
    let num_pairs = first.len().min(second.len());
    for i in 0..num_pairs {
        output[i * 2] = first[i];
        output[i * 2 + 1] = second[i];
    }
    if first.len() > num_pairs {
        output[num_pairs * 2] = first[num_pairs];
    }
}
