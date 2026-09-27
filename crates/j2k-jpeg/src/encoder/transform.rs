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
    let mut columns = [[0.0; 8]; 8];
    for y in 0..8 {
        let row = fdct_8(core::array::from_fn(|x| {
            f64::from(block[y * 8 + x]) - 128.0
        }));
        for u in 0..8 {
            columns[u][y] = row[u];
        }
    }
    let mut coeffs = [0i32; 64];
    for (u, column) in columns.into_iter().enumerate() {
        for (v, value) in fdct_8(column).into_iter().enumerate() {
            let natural = v * 8 + u;
            let mut quantized =
                value / (8.0 * AAN_SCALE[u] * AAN_SCALE[v] * f64::from(quant[natural]));
            // Reassociation can move an exact half-integer across the rounding
            // boundary. Preserve the original codestream on those rare ties.
            if (quantized.abs().fract() - 0.5).abs() < 1e-9 {
                let mut direct = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        direct +=
                            (f64::from(block[y * 8 + x]) - 128.0) * cosine[u][x] * cosine[v][y];
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
                quantized = 0.25 * cu * cv * direct / f64::from(quant[natural]);
            }
            coeffs[natural] = quantized.round() as i32;
        }
    }
    coeffs
}

// Arai–Agui–Nakajima scaling: the five-multiply 1-D transform defers
// normalization to quantization. F64 plus the tie fallback above preserves
// the existing encoder's rounding, including its byte-level golden fixtures.
const AAN_SCALE: [f64; 8] = [
    1.0,
    1.387_039_845_322_147_5,
    1.306_562_964_876_376_6,
    1.175_875_602_419_358_8,
    1.0,
    0.785_694_958_387_102_2,
    0.541_196_100_146_197_1,
    0.275_899_379_282_943_1,
];

#[inline]
fn fdct_8(input: [f64; 8]) -> [f64; 8] {
    let sum: [f64; 4] = core::array::from_fn(|i| input[i] + input[7 - i]);
    let diff: [f64; 4] = core::array::from_fn(|i| input[i] - input[7 - i]);
    let outer = sum[0] + sum[3];
    let inner = sum[1] + sum[2];
    let even = sum[0] - sum[3];
    let rotation = (even + sum[1] - sum[2]) * core::f64::consts::FRAC_1_SQRT_2;
    let left = diff[3] + diff[2];
    let right = diff[1] + diff[0];
    let cross = (left - right) * 0.382_683_432_365_089_8;
    let low = left * 0.541_196_100_146_197 + cross;
    let high = right * 1.306_562_964_876_376_6 + cross;
    let middle = (diff[2] + diff[1]) * core::f64::consts::FRAC_1_SQRT_2;
    [
        outer + inner,
        diff[0] + middle + high,
        even + rotation,
        diff[0] - middle - low,
        outer - inner,
        diff[0] - middle + low,
        even - rotation,
        diff[0] + middle - high,
    ]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reference transform covers bounded eight-bit samples and quantized coefficients"
    )]
    fn separable_dct_preserves_quantization_including_rounding_ties() {
        let cosine = cosine_table();
        let mut seed = 17u32;
        for case in 0..128 {
            let block = core::array::from_fn(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match case {
                    0..=3 => [0, 127, 128, 255][case],
                    4 => {
                        if i == 0 {
                            255
                        } else {
                            0
                        }
                    }
                    5 => {
                        if i % 2 == 0 {
                            255
                        } else {
                            0
                        }
                    }
                    _ => (seed >> 24) as u8,
                }
            });
            for q in [1u8, 3, 16, 255] {
                let actual = fdct_quantize(&block, &[q; 64], &cosine);
                let reference = core::array::from_fn(|k| {
                    let (v, u) = (k / 8, k % 8);
                    let mut sum = 0.0;
                    for y in 0..8 {
                        for x in 0..8 {
                            sum +=
                                (f64::from(block[y * 8 + x]) - 128.0) * cosine[u][x] * cosine[v][y];
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
                    (0.25 * cu * cv * sum / f64::from(q)).round() as i32
                });
                assert_eq!(actual, reference, "case {case}, quantizer {q}");
            }
        }
    }
}
