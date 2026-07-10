//! Forward Discrete Wavelet Transform for JPEG 2000 encoding.
//!
//! Counterpart of the inverse DWT in `idwt.rs`.
//! Supports both 5-3 reversible (lossless) and 9-7 irreversible (lossy) transforms.
//!
//! The forward DWT decomposes spatial-domain samples into wavelet coefficients
//! organized in subbands (LL, HL, LH, HH) at each decomposition level.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::floor_f32;
use j2k_codec_math::dwt;

/// 9-7 filter lifting coefficients (Table F.4 in ITU-T T.800).
const ALPHA: f32 = dwt::DWT97_ALPHA_F32;
const BETA: f32 = dwt::DWT97_BETA_F32;
const GAMMA: f32 = dwt::DWT97_GAMMA_F32;
const DELTA: f32 = dwt::DWT97_DELTA_F32;
const KAPPA: f32 = dwt::DWT97_KAPPA_F32;
const INV_KAPPA: f32 = dwt::DWT97_INV_KAPPA_F32;

fn dwt_dimension(value: usize) -> u32 {
    u32::try_from(value).expect("DWT dimensions remain bounded by their u32 inputs")
}

/// Result of the forward DWT: wavelet coefficients organized by subbands.
#[derive(Debug)]
pub(crate) struct DwtDecomposition {
    /// LL subband coefficients (from the lowest decomposition level).
    pub(crate) ll: Vec<f32>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    /// Each level contains (HL, LH, HH) subbands.
    pub(crate) levels: Vec<DwtLevel>,
}

#[derive(Debug)]
pub(crate) struct DwtLevel {
    pub(crate) hl: Vec<f32>,
    pub(crate) lh: Vec<f32>,
    pub(crate) hh: Vec<f32>,
    /// Dimensions of the low-pass subband at this level.
    pub(crate) low_width: u32,
    pub(crate) low_height: u32,
    /// Dimensions of the high-pass subband at this level.
    pub(crate) high_width: u32,
    pub(crate) high_height: u32,
}

/// Exact reversible 5-3 DWT coefficients for high-precision lossless encode.
#[derive(Debug)]
pub(crate) struct DwtDecompositionI64 {
    pub(crate) ll: Vec<i64>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) levels: Vec<DwtLevelI64>,
}

#[derive(Debug)]
pub(crate) struct DwtLevelI64 {
    pub(crate) hl: Vec<i64>,
    pub(crate) lh: Vec<i64>,
    pub(crate) hh: Vec<i64>,
    pub(crate) low_width: u32,
    pub(crate) low_height: u32,
    pub(crate) high_width: u32,
    pub(crate) high_height: u32,
}

/// Perform multi-level forward DWT on the given image samples.
///
/// `samples` are in row-major order, `width × height`.
/// `num_levels` is the number of decomposition levels (typically 5).
/// `reversible` selects 5-3 (true) or 9-7 (false) filter.
pub(crate) fn forward_dwt(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
) -> DwtDecomposition {
    let w = width as usize;
    let h = height as usize;

    // Working buffer: we transform in-place level by level
    let mut buffer = samples.to_vec();
    let mut current_width = w;
    let mut current_height = h;

    let mut levels = Vec::with_capacity(num_levels as usize);

    for _ in 0..num_levels {
        if current_width < 2 && current_height < 2 {
            break;
        }

        // The decoder applies horizontal synthesis before vertical synthesis,
        // so analysis must apply vertical first and horizontal second.
        if current_height >= 2 {
            let mut col_buf = vec![0.0f32; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col_buf[y] = buffer[y * w + x];
                }

                if reversible {
                    forward_lift_53(&mut col_buf[..current_height]);
                } else {
                    forward_lift_97(&mut col_buf[..current_height]);
                }

                // De-interleave: evens (low) then odds (high)
                let num_low = current_height.div_ceil(2);
                for i in 0..num_low {
                    buffer[i * w + x] = col_buf[i * 2];
                }
                for i in 0..(current_height / 2) {
                    buffer[(num_low + i) * w + x] = col_buf[i * 2 + 1];
                }
            }
        }

        if current_width >= 2 {
            let mut row_buf = vec![0.0f32; current_width];
            for y in 0..current_height {
                let row_start = y * w;
                row_buf[..current_width]
                    .copy_from_slice(&buffer[row_start..row_start + current_width]);

                if reversible {
                    forward_lift_53(&mut row_buf[..current_width]);
                } else {
                    forward_lift_97(&mut row_buf[..current_width]);
                }

                // De-interleave: evens (low) then odds (high)
                let num_low = current_width.div_ceil(2);
                for i in 0..num_low {
                    buffer[row_start + i] = row_buf[i * 2];
                }
                for i in 0..(current_width / 2) {
                    buffer[row_start + num_low + i] = row_buf[i * 2 + 1];
                }
            }
        }

        let low_w = current_width.div_ceil(2);
        let low_h = current_height.div_ceil(2);
        let high_w = current_width / 2;
        let high_h = current_height / 2;

        // Extract subbands: HL (top-right), LH (bottom-left), HH (bottom-right)
        let mut hl = vec![0.0f32; high_w * low_h];
        let mut lh = vec![0.0f32; low_w * high_h];
        let mut hh = vec![0.0f32; high_w * high_h];

        for y in 0..low_h {
            for x in 0..high_w {
                hl[y * high_w + x] = buffer[y * w + low_w + x];
            }
        }
        for y in 0..high_h {
            for x in 0..low_w {
                lh[y * low_w + x] = buffer[(low_h + y) * w + x];
            }
        }
        for y in 0..high_h {
            for x in 0..high_w {
                hh[y * high_w + x] = buffer[(low_h + y) * w + low_w + x];
            }
        }

        levels.push(DwtLevel {
            hl,
            lh,
            hh,
            low_width: dwt_dimension(low_w),
            low_height: dwt_dimension(low_h),
            high_width: dwt_dimension(high_w),
            high_height: dwt_dimension(high_h),
        });

        current_width = low_w;
        current_height = low_h;
    }

    // Extract final LL subband
    let mut ll = vec![0.0f32; current_width * current_height];
    for y in 0..current_height {
        for x in 0..current_width {
            ll[y * current_width + x] = buffer[y * w + x];
        }
    }

    // Levels are stored from highest resolution to lowest, but we want
    // them in the same order the decoder expects (lowest to highest).
    levels.reverse();

    DwtDecomposition {
        ll,
        ll_width: dwt_dimension(current_width),
        ll_height: dwt_dimension(current_height),
        levels,
    }
}

/// Perform exact multi-level reversible 5-3 DWT on signed integer samples.
pub(crate) fn forward_dwt_i64(
    samples: &[i64],
    width: u32,
    height: u32,
    num_levels: u8,
) -> DwtDecompositionI64 {
    let w = width as usize;
    let h = height as usize;

    let mut buffer = samples.to_vec();
    let mut current_width = w;
    let mut current_height = h;
    let mut levels = Vec::with_capacity(num_levels as usize);

    for _ in 0..num_levels {
        if current_width < 2 && current_height < 2 {
            break;
        }

        if current_height >= 2 {
            let mut col_buf = vec![0_i64; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col_buf[y] = buffer[y * w + x];
                }

                forward_lift_53_i64(&mut col_buf[..current_height]);

                let num_low = current_height.div_ceil(2);
                for i in 0..num_low {
                    buffer[i * w + x] = col_buf[i * 2];
                }
                for i in 0..(current_height / 2) {
                    buffer[(num_low + i) * w + x] = col_buf[i * 2 + 1];
                }
            }
        }

        if current_width >= 2 {
            let mut row_buf = vec![0_i64; current_width];
            for y in 0..current_height {
                let row_start = y * w;
                row_buf[..current_width]
                    .copy_from_slice(&buffer[row_start..row_start + current_width]);

                forward_lift_53_i64(&mut row_buf[..current_width]);

                let num_low = current_width.div_ceil(2);
                for i in 0..num_low {
                    buffer[row_start + i] = row_buf[i * 2];
                }
                for i in 0..(current_width / 2) {
                    buffer[row_start + num_low + i] = row_buf[i * 2 + 1];
                }
            }
        }

        let low_w = current_width.div_ceil(2);
        let low_h = current_height.div_ceil(2);
        let high_w = current_width / 2;
        let high_h = current_height / 2;

        let mut hl = vec![0_i64; high_w * low_h];
        let mut lh = vec![0_i64; low_w * high_h];
        let mut hh = vec![0_i64; high_w * high_h];

        for y in 0..low_h {
            for x in 0..high_w {
                hl[y * high_w + x] = buffer[y * w + low_w + x];
            }
        }
        for y in 0..high_h {
            for x in 0..low_w {
                lh[y * low_w + x] = buffer[(low_h + y) * w + x];
            }
        }
        for y in 0..high_h {
            for x in 0..high_w {
                hh[y * high_w + x] = buffer[(low_h + y) * w + low_w + x];
            }
        }

        levels.push(DwtLevelI64 {
            hl,
            lh,
            hh,
            low_width: dwt_dimension(low_w),
            low_height: dwt_dimension(low_h),
            high_width: dwt_dimension(high_w),
            high_height: dwt_dimension(high_h),
        });

        current_width = low_w;
        current_height = low_h;
    }

    let mut ll = vec![0_i64; current_width * current_height];
    for y in 0..current_height {
        for x in 0..current_width {
            ll[y * current_width + x] = buffer[y * w + x];
        }
    }

    levels.reverse();

    DwtDecompositionI64 {
        ll,
        ll_width: dwt_dimension(current_width),
        ll_height: dwt_dimension(current_height),
        levels,
    }
}

/// Forward 5-3 reversible lifting (integer arithmetic).
///
/// Equations F-2 and F-3 from ITU-T T.800:
///   d(n) = x(2n+1) - floor((x(2n) + x(2n+2)) / 2)
///   s(n) = x(2n)   + floor((d(n-1) + d(n)) / 4 + 0.5)
///
/// Applied in-place: even indices are low-pass, odd indices are high-pass.
fn forward_lift_53(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        forward_lift_53_even(data);
        return;
    }

    // Step 1: Predict (high-pass) — update odd samples
    // d(i) = x(2i+1) - floor((x(2i) + x(2i+2)) / 2)
    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= floor_f32((left + right) * 0.5);
    }

    // Step 2: Update (low-pass) — update even samples
    // s(i) = x(2i) + floor((d(i-1) + d(i)) / 4 + 0.5)
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += floor_f32((left + right) * 0.25 + 0.5);
    }
}

fn forward_lift_53_i64(data: &mut [i64]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        forward_lift_53_even_i64(data);
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= floor_div2_i64(left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += floor_div4_plus_half_i64(left + right);
    }
}

fn forward_lift_53_even(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n >= 2);
    debug_assert!(n.is_multiple_of(2));

    for i in (1..n - 1).step_by(2) {
        data[i] -= floor_f32((data[i - 1] + data[i + 1]) * 0.5);
    }
    data[n - 1] -= floor_f32(data[n - 2]);

    data[0] += floor_f32(data[1] * 0.5 + 0.5);
    for i in (2..n).step_by(2) {
        data[i] += floor_f32((data[i - 1] + data[i + 1]) * 0.25 + 0.5);
    }
}

fn forward_lift_53_even_i64(data: &mut [i64]) {
    let n = data.len();
    debug_assert!(n >= 2);
    debug_assert!(n.is_multiple_of(2));

    for i in (1..n - 1).step_by(2) {
        data[i] -= floor_div2_i64(data[i - 1] + data[i + 1]);
    }
    data[n - 1] -= data[n - 2];

    data[0] += floor_div2_plus_half_i64(data[1]);
    for i in (2..n).step_by(2) {
        data[i] += floor_div4_plus_half_i64(data[i - 1] + data[i + 1]);
    }
}

fn floor_div2_i64(value: i64) -> i64 {
    value.div_euclid(2)
}

fn floor_div2_plus_half_i64(value: i64) -> i64 {
    (value + 1).div_euclid(2)
}

fn floor_div4_plus_half_i64(value: i64) -> i64 {
    (value + 2).div_euclid(4)
}

/// Forward 9-7 irreversible lifting (floating-point).
///
/// The forward transform applies the lifting steps in the order that is
/// the reverse of the inverse DWT in idwt.rs.
///
/// Forward lifting steps:
///   1. d(n) += α * (s(n) + s(n+1))     (predict high from low neighbors)
///   2. s(n) += β * (d(n-1) + d(n))     (update low from high neighbors)
///   3. d(n) += γ * (s(n) + s(n+1))     (second predict)
///   4. s(n) += δ * (d(n-1) + d(n))     (second update)
///   5. s(n) *= 1/κ                       (scale low-pass)
///   6. d(n) *= κ                         (scale high-pass)
fn forward_lift_97(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };

    // Step 1: α predict on odd (high-pass) samples
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }

    // Step 2: β update on even (low-pass) samples
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }

    // Step 3: γ predict on odd samples
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }

    // Step 4: δ update on even samples
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }

    // Step 5 & 6: Scale
    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_slice(a: &[f32], b: &[f32], eps: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn test_forward_53_basic() {
        // Simple 4-element signal
        let mut data = vec![10.0, 20.0, 30.0, 40.0];
        forward_lift_53(&mut data);

        // After forward transform, reconstruct with inverse and check
        inverse_lift_53(&mut data);
        assert!(approx_eq_slice(&data, &[10.0, 20.0, 30.0, 40.0], 0.001));
    }

    #[test]
    fn forward_53_i64_round_trips_38_bit_values() {
        let original = vec![
            (1_i64 << 37) - 1,
            -(1_i64 << 37),
            (1_i64 << 36) + 17,
            -((1_i64 << 36) - 9),
            123_456_789,
            -987_654_321,
            0,
        ];
        let mut data = original.clone();

        forward_lift_53_i64(&mut data);
        inverse_lift_53_i64(&mut data);

        assert_eq!(data, original);
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "this parity test deliberately converts coefficients restricted to the exact f32 integer range"
    )]
    #[expect(
        clippy::similar_names,
        reason = "i64 and f32 suffixes distinguish the two transform paths under comparison"
    )]
    fn forward_dwt_i64_matches_f32_path_for_exact_range() {
        let samples_i64 = (0..25)
            .map(|idx| i64::from(((idx * 37 + idx / 3) & 0xffff) - 32_768))
            .collect::<Vec<_>>();
        let samples_f32 = samples_i64
            .iter()
            .map(|sample| *sample as f32)
            .collect::<Vec<_>>();

        let i64_decomp = forward_dwt_i64(&samples_i64, 5, 5, 2);
        let f32_decomp = forward_dwt(&samples_f32, 5, 5, 2, true);

        assert_eq!(
            i64_decomp
                .ll
                .iter()
                .map(|sample| *sample as f32)
                .collect::<Vec<_>>(),
            f32_decomp.ll
        );
        assert_eq!(i64_decomp.levels.len(), f32_decomp.levels.len());
        for (actual, expected) in i64_decomp.levels.iter().zip(&f32_decomp.levels) {
            assert_eq!(
                actual
                    .hl
                    .iter()
                    .map(|sample| *sample as f32)
                    .collect::<Vec<_>>(),
                expected.hl
            );
            assert_eq!(
                actual
                    .lh
                    .iter()
                    .map(|sample| *sample as f32)
                    .collect::<Vec<_>>(),
                expected.lh
            );
            assert_eq!(
                actual
                    .hh
                    .iter()
                    .map(|sample| *sample as f32)
                    .collect::<Vec<_>>(),
                expected.hh
            );
        }
    }

    #[test]
    fn forward_53_even_fast_path_matches_reference_for_common_tile_widths() {
        for len in [2usize, 4, 8, 64, 512] {
            let mut expected = (0..len)
                .map(|idx| {
                    f32::from(
                        u8::try_from((idx * 37 + idx / 3) & 0xff)
                            .expect("masked test sample fits u8"),
                    ) - 128.0
                })
                .collect::<Vec<_>>();
            let mut actual = expected.clone();

            forward_lift_53_reference(&mut expected);
            forward_lift_53_even(&mut actual);

            assert_eq!(actual, expected, "len={len}");
        }
    }

    #[test]
    fn test_forward_97_round_trip() {
        for len in [2usize, 3, 8, 9, 64, 65] {
            let original: Vec<f32> = (0..len)
                .map(|idx| {
                    f32::from(
                        u8::try_from((idx * 37 + idx / 3) & 0xff)
                            .expect("masked test sample fits u8"),
                    ) - 128.0
                })
                .collect();
            let mut data = original.clone();

            forward_lift_97(&mut data);
            crate::j2c::idwt::test_irreversible_filter_97i(&mut data, len, 0);

            assert!(
                approx_eq_slice(&data, &original, 0.01),
                "len={len} data={data:?} original={original:?}"
            );
        }
    }

    #[test]
    fn forward_lift_97_places_constant_signal_in_low_pass() {
        for len in [2usize, 3, 8, 9, 64, 65] {
            let mut data = vec![50.0; len];

            forward_lift_97(&mut data);

            for &low in data.iter().step_by(2) {
                assert!((low - 50.0).abs() < 0.001, "len={len} data={data:?}");
            }
            for &high in data.iter().skip(1).step_by(2) {
                assert!(high.abs() < 0.001, "len={len} data={data:?}");
            }
        }
    }

    #[test]
    fn test_forward_dwt_53_single_level() {
        // 4×4 image
        let samples: Vec<f32> = (0..16)
            .map(|x| f32::from(u8::try_from(x).expect("test sample fits u8")))
            .collect();
        let decomp = forward_dwt(&samples, 4, 4, 1, true);
        assert_eq!(decomp.ll_width, 2);
        assert_eq!(decomp.ll_height, 2);
        assert_eq!(decomp.levels.len(), 1);
    }

    #[test]
    fn test_forward_dwt_97_multi_level() {
        let samples: Vec<f32> = (0..64)
            .map(|x| f32::from(u8::try_from(x).expect("test sample fits u8")))
            .collect();
        let decomp = forward_dwt(&samples, 8, 8, 3, false);
        assert_eq!(decomp.levels.len(), 3);
        // After 3 levels of 8×8: 4×4 → 2×2 → 1×1
        assert_eq!(decomp.ll_width, 1);
        assert_eq!(decomp.ll_height, 1);
    }

    #[test]
    fn test_odd_dimensions() {
        let samples: Vec<f32> = (0..15)
            .map(|x| f32::from(u8::try_from(x).expect("test sample fits u8")))
            .collect();
        let decomp = forward_dwt(&samples, 5, 3, 1, true);
        assert_eq!(decomp.ll_width, 3);
        assert_eq!(decomp.ll_height, 2);
        assert_eq!(decomp.levels[0].high_width, 2);
        assert_eq!(decomp.levels[0].high_height, 1);
    }

    // Inverse lifting functions for round-trip testing
    fn inverse_lift_53(data: &mut [f32]) {
        let n = data.len();
        if n < 2 {
            return;
        }
        // Undo update
        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] -= ((left + right) * 0.25 + 0.5).floor();
        }
        // Undo predict
        let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] += ((left + right) * 0.5).floor();
        }
    }

    fn inverse_lift_53_i64(data: &mut [i64]) {
        let n = data.len();
        if n < 2 {
            return;
        }
        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] -= floor_div4_plus_half_i64(left + right);
        }
        let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] += floor_div2_i64(left + right);
        }
    }

    fn forward_lift_53_reference(data: &mut [f32]) {
        let n = data.len();
        if n < 2 {
            return;
        }

        let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
        for i in (1..n).step_by(2) {
            let left = data[i - 1];
            let right = if i + 1 < n {
                data[i + 1]
            } else {
                data[last_even]
            };
            data[i] -= ((left + right) * 0.5).floor();
        }

        for i in (0..n).step_by(2) {
            let left = if i > 0 { data[i - 1] } else { data[1] };
            let right = if i + 1 < n { data[i + 1] } else { left };
            data[i] += ((left + right) * 0.25 + 0.5).floor();
        }
    }
}
