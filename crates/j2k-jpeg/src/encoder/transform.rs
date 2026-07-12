// SPDX-License-Identifier: MIT OR Apache-2.0

use core::f64::consts::PI;

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounded baseline DCT coefficients are bounded by the encoder's validated eight-bit input domain"
)]
pub(super) fn fdct_quantize(
    block: &[u8; 64],
    quant: &[u8; 64],
    cosine: &[[f64; 8]; 8],
) -> [i32; 64] {
    let mut coeffs = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    let sample = f64::from(block[y * 8 + x]) - 128.0;
                    sum += sample * cosine[u][x] * cosine[v][y];
                }
            }
            let cu = if u == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                core::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let natural = v * 8 + u;
            let transformed = 0.25 * cu * cv * sum;
            coeffs[natural] = (transformed / f64::from(quant[natural])).round() as i32;
        }
    }
    coeffs
}

pub(super) fn cosine_table() -> [[f64; 8]; 8] {
    let mut table = [[0.0; 8]; 8];
    for (u, row) in (0u32..8).zip(&mut table) {
        for (x, value) in (0u32..8).zip(row) {
            *value = ((f64::from(2 * x + 1) * f64::from(u) * PI) / 16.0).cos();
        }
    }
    table
}
