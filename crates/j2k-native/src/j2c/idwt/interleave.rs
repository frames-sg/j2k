// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::Decomposition;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::model::IDWTInput;
use crate::math::{dispatch, f32x8, Level, Simd, SIMD_WIDTH};

/// The `2D_INTERLEAVE` procedure described in F.3.3.
pub(super) fn interleave_samples(
    input: IDWTInput<'_>,
    decomposition: &Decomposition,
    coefficients: &mut Vec<f32>,
    storage: &DecompositionStorage<'_>,
) {
    let level = Level::new();
    dispatch!(level, simd => {
        interleave_samples_inner::<_>(simd, input, decomposition, coefficients, storage);
    });
}

#[inline(always)]
fn interleave_samples_inner<S: Simd>(
    simd: S,
    input: IDWTInput<'_>,
    decomposition: &Decomposition,
    coefficients: &mut Vec<f32>,
    storage: &DecompositionStorage<'_>,
) {
    let width = decomposition.rect.width() as usize;
    let height = decomposition.rect.height() as usize;

    // Just a sanity check. We should have allocated enough upfront before
    // starting the IDWT.
    assert!(coefficients.capacity() >= width * height);

    // The cleaner way would be to first clear and then resize, so that we
    // have a clean buffer with just zeroes. However, this is actually not
    // necessary, because when interleaving and generating the border values
    // we will replace all the data anyway, so we can save the cost of
    // the clear operation.
    coefficients.resize(width * height, 0.0);

    let IntRect {
        x0: u0,
        x1: u1,
        y0: v0,
        y1: v1,
    } = decomposition.rect;

    let ll = input.coefficients;
    let hl = &storage.coefficients[storage.sub_bands[decomposition.sub_bands[0]]
        .coefficients
        .clone()];
    let lh = &storage.coefficients[storage.sub_bands[decomposition.sub_bands[1]]
        .coefficients
        .clone()];
    let hh = &storage.coefficients[storage.sub_bands[decomposition.sub_bands[2]]
        .coefficients
        .clone()];

    // See Figure F.8.
    let num_u_low = (u1.div_ceil(2) - u0.div_ceil(2)) as usize;
    let num_u_high = (u1 / 2 - u0 / 2) as usize;
    let num_v_low = (v1.div_ceil(2) - v0.div_ceil(2)) as usize;
    let num_v_high = (v1 / 2 - v0 / 2) as usize;

    // Depending on whether the start row is even or odd, either LL/HL comes first
    // or HL/HH.

    let (first_w, second_w) = if u0 % 2 == 0 {
        (num_u_low, num_u_high)
    } else {
        (num_u_high, num_u_low)
    };

    let even_row_start = if v0 % 2 == 0 { 0 } else { 1 };
    let odd_row_start = if v0 % 2 == 0 { 1 } else { 0 };

    // Determine whether LL or HL is the band in the first column.
    let (first_even, second_even) = if u0 % 2 == 0 { (ll, hl) } else { (hl, ll) };
    interleave_rows(
        simd,
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

    // Determine whether LH or HH is the band in the first column.
    let (first_odd, second_odd) = if u0 % 2 == 0 { (lh, hh) } else { (hh, lh) };
    interleave_rows(
        simd,
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

#[inline(always)]
fn interleave_rows<S: Simd>(
    simd: S,
    first_band: &[f32],
    second_band: &[f32],
    first_w: usize,
    second_w: usize,
    output: &mut [f32],
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

        interleave_row(simd, first_row, second_row, out_slice);
    }
}

#[inline(always)]
fn interleave_row<S: Simd>(simd: S, first: &[f32], second: &[f32], output: &mut [f32]) {
    let num_pairs = first.len().min(second.len());
    let simd_chunks = num_pairs / SIMD_WIDTH;

    // Process as much as possible using SIMD.
    for i in 0..simd_chunks {
        let base = i * SIMD_WIDTH;
        let f = f32x8::from_slice(simd, &first[base..base + SIMD_WIDTH]);
        let s = f32x8::from_slice(simd, &second[base..base + SIMD_WIDTH]);

        f.zip_low(s).store(&mut output[base * 2..]);
        f.zip_high(s).store(&mut output[base * 2 + SIMD_WIDTH..]);
    }

    // Scalar remainder.
    for i in (simd_chunks * SIMD_WIDTH)..num_pairs {
        output[i * 2] = first[i];
        output[i * 2 + 1] = second[i];
    }

    // Handle extra element if first is longer.
    if first.len() > num_pairs {
        output[num_pairs * 2] = first[num_pairs];
    }
}
