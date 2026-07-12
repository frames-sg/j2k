// SPDX-License-Identifier: MIT OR Apache-2.0

use super::Vec;

pub(in crate::j2c::encode) fn copy_code_block_coefficients(
    quantized: &[i32],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
) -> Vec<i32> {
    let len = cbw * cbh;
    let start = y0 * width + x0;
    if cbw == width {
        return quantized[start..start + len].to_vec();
    }
    let mut coefficients = Vec::with_capacity(len);
    for y in 0..cbh {
        let row_start = (y0 + y) * width + x0;
        coefficients.extend_from_slice(&quantized[row_start..row_start + cbw]);
    }
    coefficients
}

pub(in crate::j2c::encode) fn downcast_i64_coefficients_to_i32(
    coefficients: &[i64],
) -> Result<Vec<i32>, &'static str> {
    coefficients
        .iter()
        .map(|&coefficient| {
            i32::try_from(coefficient).map_err(|_| {
                "HTJ2K/accelerated code-block encode does not support i64 coefficients"
            })
        })
        .collect()
}
