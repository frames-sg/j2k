//! Colour transforms — RCT (reversible) and ICT (irreversible).
//!
//! Port of `ojph_colour.cpp` generic implementations.

use crate::arch::ojph_round;
use crate::mem::{LineBuf, LineBufData, LFT_32BIT};

// =========================================================================
// ICT constants — port of C++ `CT_CNST`
// =========================================================================

/// Forward ICT coefficients.
pub const ALPHA_RF: f32 = 0.299;
pub const ALPHA_GF: f32 = 0.587;
pub const ALPHA_BF: f32 = 0.114;

// BETA_CbF = 0.5 / (1 - ALPHA_BF)
pub const BETA_CBF: f32 = (0.5 / (1.0 - ALPHA_BF as f64)) as f32;
// BETA_CrF = 0.5 / (1 - ALPHA_RF)
pub const BETA_CRF: f32 = (0.5 / (1.0 - ALPHA_RF as f64)) as f32;

// Inverse ICT coefficients
// GAMMA_CR2R = 2.0 * (1 - ALPHA_RF)
pub const GAMMA_CR2R: f32 = (2.0 * (1.0 - ALPHA_RF as f64)) as f32;
// GAMMA_CB2B = 2.0 * (1 - ALPHA_BF)
pub const GAMMA_CB2B: f32 = (2.0 * (1.0 - ALPHA_BF as f64)) as f32;
// GAMMA_CR2G = 2.0 * ALPHA_RF * (1 - ALPHA_RF) / ALPHA_GF
pub const GAMMA_CR2G: f32 =
    (2.0 * ALPHA_RF as f64 * (1.0 - ALPHA_RF as f64) / ALPHA_GF as f64) as f32;
// GAMMA_CB2G = 2.0 * ALPHA_BF * (1 - ALPHA_BF) / ALPHA_GF
pub const GAMMA_CB2G: f32 =
    (2.0 * ALPHA_BF as f64 * (1.0 - ALPHA_BF as f64) / ALPHA_GF as f64) as f32;

// =========================================================================
// Unsafe pointer accessors (same pattern as wavelet.rs)
// =========================================================================

#[inline]
unsafe fn lb_i32(buf: &LineBuf) -> *const i32 {
    match buf.data {
        LineBufData::I32(p) => p as *const i32,
        _ => panic!("expected i32 LineBuf"),
    }
}

#[inline]
unsafe fn lb_i32_mut(buf: &mut LineBuf) -> *mut i32 {
    match buf.data {
        LineBufData::I32(p) => p,
        _ => panic!("expected i32 LineBuf"),
    }
}

#[inline]
unsafe fn lb_i64(buf: &LineBuf) -> *const i64 {
    match buf.data {
        LineBufData::I64(p) => p as *const i64,
        _ => panic!("expected i64 LineBuf"),
    }
}

#[inline]
unsafe fn lb_i64_mut(buf: &mut LineBuf) -> *mut i64 {
    match buf.data {
        LineBufData::I64(p) => p,
        _ => panic!("expected i64 LineBuf"),
    }
}

#[inline]
unsafe fn lb_f32(buf: &LineBuf) -> *const f32 {
    match buf.data {
        LineBufData::F32(p) => p as *const f32,
        _ => panic!("expected f32 LineBuf"),
    }
}

#[inline]
unsafe fn lb_f32_mut(buf: &mut LineBuf) -> *mut f32 {
    match buf.data {
        LineBufData::F32(p) => p,
        _ => panic!("expected f32 LineBuf"),
    }
}

// =========================================================================
// rev_convert — reversible sample conversion (integer shift)
// =========================================================================

/// Generic reversible sample conversion.
///
/// Converts samples by adding `shift` (can be positive for encoding offset,
/// or negative for decoding). Handles 32→32, 32→64, and 64→32 paths.
pub(crate) fn gen_rev_convert(
    src_line: &LineBuf,
    src_line_offset: u32,
    dst_line: &mut LineBuf,
    dst_line_offset: u32,
    shift: i64,
    width: u32,
) {
    unsafe {
        if (src_line.flags & LFT_32BIT) != 0 {
            if (dst_line.flags & LFT_32BIT) != 0 {
                // 32 → 32
                let sp = lb_i32(src_line).add(src_line_offset as usize);
                let dp = lb_i32_mut(dst_line).add(dst_line_offset as usize);
                let s = shift as i32;
                for i in 0..width as usize {
                    *dp.add(i) = *sp.add(i) + s;
                }
            } else {
                // 32 → 64
                let sp = lb_i32(src_line).add(src_line_offset as usize);
                let dp = lb_i64_mut(dst_line).add(dst_line_offset as usize);
                for i in 0..width as usize {
                    *dp.add(i) = *sp.add(i) as i64 + shift;
                }
            }
        } else {
            // 64 → 32
            let sp = lb_i64(src_line).add(src_line_offset as usize);
            let dp = lb_i32_mut(dst_line).add(dst_line_offset as usize);
            for i in 0..width as usize {
                *dp.add(i) = (*sp.add(i) + shift) as i32;
            }
        }
    }
}

// =========================================================================
// rev_convert_nlt_type3 — NLT type 3 (DC level shift w/ sign flip)
// =========================================================================

/// Generic reversible NLT type 3 conversion.
///
/// `v >= 0 → v`, `v < 0 → -v - shift`
pub(crate) fn gen_rev_convert_nlt_type3(
    src_line: &LineBuf,
    src_line_offset: u32,
    dst_line: &mut LineBuf,
    dst_line_offset: u32,
    shift: i64,
    width: u32,
) {
    unsafe {
        if (src_line.flags & LFT_32BIT) != 0 {
            if (dst_line.flags & LFT_32BIT) != 0 {
                let sp = lb_i32(src_line).add(src_line_offset as usize);
                let dp = lb_i32_mut(dst_line).add(dst_line_offset as usize);
                let s = shift as i32;
                for i in 0..width as usize {
                    let v = *sp.add(i);
                    *dp.add(i) = if v >= 0 { v } else { -v - s };
                }
            } else {
                let sp = lb_i32(src_line).add(src_line_offset as usize);
                let dp = lb_i64_mut(dst_line).add(dst_line_offset as usize);
                for i in 0..width as usize {
                    let v = *sp.add(i) as i64;
                    *dp.add(i) = if v >= 0 { v } else { -v - shift };
                }
            }
        } else {
            let sp = lb_i64(src_line).add(src_line_offset as usize);
            let dp = lb_i32_mut(dst_line).add(dst_line_offset as usize);
            for i in 0..width as usize {
                let v = *sp.add(i);
                *dp.add(i) = (if v >= 0 { v } else { -v - shift }) as i32;
            }
        }
    }
}

// =========================================================================
// irv_convert_to_integer — float → integer quantization
// =========================================================================

/// Core implementation parameterized on NLT type 3 flag.
#[inline]
fn local_gen_irv_convert_to_integer(
    src_line: &LineBuf,
    dst_line: &mut LineBuf,
    dst_line_offset: u32,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
    nlt_type3: bool,
) {
    debug_assert!(bit_depth <= 32);

    unsafe {
        let sp = lb_f32(src_line);
        let dp = lb_i32_mut(dst_line).add(dst_line_offset as usize);

        let neg_limit = (i32::MIN) >> (32 - bit_depth);
        let mul = (1u64 << bit_depth) as f32;
        let fl_up_lim = -(neg_limit as f32); // val < upper
        let fl_low_lim = neg_limit as f32; // val >= lower
        let s32_up_lim = i32::MAX >> (32 - bit_depth);
        let s32_low_lim = i32::MIN >> (32 - bit_depth);

        if is_signed {
            let bias = ((1u64 << (bit_depth - 1)) + 1) as i32;
            for i in 0..width as usize {
                let t = *sp.add(i) * mul;
                let mut v = ojph_round(t);
                v = if t >= fl_low_lim { v } else { s32_low_lim };
                v = if t < fl_up_lim { v } else { s32_up_lim };
                if nlt_type3 {
                    v = if v >= 0 { v } else { -v - bias };
                }
                *dp.add(i) = v;
            }
        } else {
            let half = (1u64 << (bit_depth - 1)) as i32;
            for i in 0..width as usize {
                let t = *sp.add(i) * mul;
                let mut v = ojph_round(t);
                v = if t >= fl_low_lim { v } else { s32_low_lim };
                v = if t < fl_up_lim { v } else { s32_up_lim };
                *dp.add(i) = v + half;
            }
        }
    }
}

/// Generic irreversible float-to-integer conversion.
pub(crate) fn gen_irv_convert_to_integer(
    src_line: &LineBuf,
    dst_line: &mut LineBuf,
    dst_line_offset: u32,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
) {
    local_gen_irv_convert_to_integer(
        src_line,
        dst_line,
        dst_line_offset,
        bit_depth,
        is_signed,
        width,
        false,
    );
}

/// Generic irreversible float-to-integer NLT type 3 conversion.
pub(crate) fn gen_irv_convert_to_integer_nlt_type3(
    src_line: &LineBuf,
    dst_line: &mut LineBuf,
    dst_line_offset: u32,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
) {
    local_gen_irv_convert_to_integer(
        src_line,
        dst_line,
        dst_line_offset,
        bit_depth,
        is_signed,
        width,
        true,
    );
}

// =========================================================================
// irv_convert_to_float — integer → float dequantization
// =========================================================================

/// Core implementation parameterized on NLT type 3 flag.
#[inline]
fn local_gen_irv_convert_to_float(
    src_line: &LineBuf,
    src_line_offset: u32,
    dst_line: &mut LineBuf,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
    nlt_type3: bool,
) {
    debug_assert!(bit_depth <= 32);

    let mul = (1.0 / (1u64 << bit_depth) as f64) as f32;

    unsafe {
        let sp = lb_i32(src_line).add(src_line_offset as usize);
        let dp = lb_f32_mut(dst_line);

        if is_signed {
            let bias = ((1u64 << (bit_depth - 1)) + 1) as i32;
            for i in 0..width as usize {
                let mut v = *sp.add(i);
                if nlt_type3 {
                    v = if v >= 0 { v } else { -v - bias };
                }
                *dp.add(i) = v as f32 * mul;
            }
        } else {
            let half = (1u64 << (bit_depth - 1)) as i32;
            for i in 0..width as usize {
                let v = *sp.add(i) - half;
                *dp.add(i) = v as f32 * mul;
            }
        }
    }
}

/// Generic irreversible integer-to-float conversion.
pub(crate) fn gen_irv_convert_to_float(
    src_line: &LineBuf,
    src_line_offset: u32,
    dst_line: &mut LineBuf,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
) {
    local_gen_irv_convert_to_float(
        src_line,
        src_line_offset,
        dst_line,
        bit_depth,
        is_signed,
        width,
        false,
    );
}

/// Generic irreversible integer-to-float NLT type 3 conversion.
pub(crate) fn gen_irv_convert_to_float_nlt_type3(
    src_line: &LineBuf,
    src_line_offset: u32,
    dst_line: &mut LineBuf,
    bit_depth: u32,
    is_signed: bool,
    width: u32,
) {
    local_gen_irv_convert_to_float(
        src_line,
        src_line_offset,
        dst_line,
        bit_depth,
        is_signed,
        width,
        true,
    );
}

// =========================================================================
// RCT — Reversible Colour Transform
// =========================================================================

/// Generic RCT forward transform (RGB → YCbCr).
///
/// `Y = (R + 2G + B) >> 2`, `Cb = B - G`, `Cr = R - G`
pub(crate) fn gen_rct_forward(
    r: &LineBuf,
    g: &LineBuf,
    b: &LineBuf,
    y: &mut LineBuf,
    cb: &mut LineBuf,
    cr: &mut LineBuf,
    repeat: u32,
) {
    if (y.flags & LFT_32BIT) != 0 {
        unsafe {
            let rp = lb_i32(r);
            let gp = lb_i32(g);
            let bp = lb_i32(b);
            let yp = lb_i32_mut(y);
            let cbp = lb_i32_mut(cb);
            let crp = lb_i32_mut(cr);

            for i in 0..repeat as usize {
                let rr = *rp.add(i);
                let gg = *gp.add(i);
                let bb = *bp.add(i);
                *yp.add(i) = (rr + (gg << 1) + bb) >> 2;
                *cbp.add(i) = bb - gg;
                *crp.add(i) = rr - gg;
            }
        }
    } else {
        // 32-bit input → 64-bit output
        unsafe {
            let rp = lb_i32(r);
            let gp = lb_i32(g);
            let bp = lb_i32(b);
            let yp = lb_i64_mut(y);
            let cbp = lb_i64_mut(cb);
            let crp = lb_i64_mut(cr);

            for i in 0..repeat as usize {
                let rr = *rp.add(i) as i64;
                let gg = *gp.add(i) as i64;
                let bb = *bp.add(i) as i64;
                *yp.add(i) = (rr + (gg << 1) + bb) >> 2;
                *cbp.add(i) = bb - gg;
                *crp.add(i) = rr - gg;
            }
        }
    }
}

/// Generic RCT backward transform (YCbCr → RGB).
///
/// `G = Y - (Cb + Cr) >> 2`, `R = Cr + G`, `B = Cb + G`
pub(crate) fn gen_rct_backward(
    y: &LineBuf,
    cb: &LineBuf,
    cr: &LineBuf,
    r: &mut LineBuf,
    g: &mut LineBuf,
    b: &mut LineBuf,
    repeat: u32,
) {
    if (y.flags & LFT_32BIT) != 0 {
        unsafe {
            let yp = lb_i32(y);
            let cbp = lb_i32(cb);
            let crp = lb_i32(cr);
            let rp = lb_i32_mut(r);
            let gp = lb_i32_mut(g);
            let bp = lb_i32_mut(b);

            for i in 0..repeat as usize {
                let yy = *yp.add(i);
                let cbb = *cbp.add(i);
                let crr = *crp.add(i);
                let gg = yy - ((cbb + crr) >> 2);
                *rp.add(i) = crr + gg;
                *gp.add(i) = gg;
                *bp.add(i) = cbb + gg;
            }
        }
    } else {
        // 64-bit input → 32-bit output
        unsafe {
            let yp = lb_i64(y);
            let cbp = lb_i64(cb);
            let crp = lb_i64(cr);
            let rp = lb_i32_mut(r);
            let gp = lb_i32_mut(g);
            let bp = lb_i32_mut(b);

            for i in 0..repeat as usize {
                let yy = *yp.add(i);
                let cbb = *cbp.add(i);
                let crr = *crp.add(i);
                let gg = yy - ((cbb + crr) >> 2);
                *rp.add(i) = (crr + gg) as i32;
                *gp.add(i) = gg as i32;
                *bp.add(i) = (cbb + gg) as i32;
            }
        }
    }
}

// =========================================================================
// ICT — Irreversible Colour Transform
// =========================================================================

/// Generic ICT forward transform (RGB → YCbCr, float).
///
/// Uses the standard matrix from ITU-T T.800:
/// - `Y  =  α_R·R + α_G·G + α_B·B`
/// - `Cb =  β_Cb · (B - Y)`
/// - `Cr =  β_Cr · (R - Y)`
pub(crate) fn gen_ict_forward(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
    repeat: u32,
) {
    for i in 0..repeat as usize {
        let yv = ALPHA_RF * r[i] + ALPHA_GF * g[i] + ALPHA_BF * b[i];
        y[i] = yv;
        cb[i] = BETA_CBF * (b[i] - yv);
        cr[i] = BETA_CRF * (r[i] - yv);
    }
}

/// Generic ICT backward transform (YCbCr → RGB, float).
pub(crate) fn gen_ict_backward(
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    repeat: u32,
) {
    for i in 0..repeat as usize {
        g[i] = y[i] - GAMMA_CR2G * cr[i] - GAMMA_CB2G * cb[i];
        r[i] = y[i] + GAMMA_CR2R * cr[i];
        b[i] = y[i] + GAMMA_CB2B * cb[i];
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::LFT_INTEGER;

    fn make_i32_linebuf(data: &mut [i32]) -> LineBuf {
        LineBuf {
            size: data.len(),
            pre_size: 0,
            flags: LFT_32BIT | LFT_INTEGER,
            data: LineBufData::I32(data.as_mut_ptr()),
        }
    }

    fn make_f32_linebuf(data: &mut [f32]) -> LineBuf {
        LineBuf {
            size: data.len(),
            pre_size: 0,
            flags: LFT_32BIT, // float
            data: LineBufData::F32(data.as_mut_ptr()),
        }
    }

    #[test]
    fn rct_forward_backward_roundtrip() {
        let mut r_data = [100i32, 150, 200, 50];
        let mut g_data = [110i32, 160, 190, 60];
        let mut b_data = [120i32, 140, 180, 70];

        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32; 4];
        let mut cb_data = [0i32; 4];
        let mut cr_data = [0i32; 4];

        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        // Check Y = (R + 2G + B) >> 2
        assert_eq!(y_data[0], (100 + 220 + 120) >> 2); // 110

        // Now invert
        let mut r2_data = [0i32; 4];
        let mut g2_data = [0i32; 4];
        let mut b2_data = [0i32; 4];

        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        gen_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 4);

        // RCT is lossless — must be exact roundtrip
        assert_eq!(r2_data, [100, 150, 200, 50]);
        assert_eq!(g2_data, [110, 160, 190, 60]);
        assert_eq!(b2_data, [120, 140, 180, 70]);
    }

    #[test]
    fn ict_forward_backward_approx() {
        let r = [0.5f32, 0.3, 0.8, 1.0];
        let g = [0.5f32, 0.6, 0.2, 0.0];
        let b = [0.5f32, 0.1, 0.6, 0.5];

        let mut y = [0.0f32; 4];
        let mut cb = [0.0f32; 4];
        let mut cr = [0.0f32; 4];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        // For equal RGB (first sample), Cb and Cr should be ~0
        assert!(cb[0].abs() < 1e-5);
        assert!(cr[0].abs() < 1e-5);

        // Roundtrip
        let mut r2 = [0.0f32; 4];
        let mut g2 = [0.0f32; 4];
        let mut b2 = [0.0f32; 4];

        gen_ict_backward(&y, &cb, &cr, &mut r2, &mut g2, &mut b2, 4);

        for i in 0..4 {
            assert!((r2[i] - r[i]).abs() < 1e-4, "R mismatch at {i}");
            assert!((g2[i] - g[i]).abs() < 1e-4, "G mismatch at {i}");
            assert!((b2[i] - b[i]).abs() < 1e-4, "B mismatch at {i}");
        }
    }

    #[test]
    fn rev_convert_basic() {
        let mut src_data = [10i32, 20, 30, 40];
        let mut dst_data = [0i32; 4];

        let src = make_i32_linebuf(&mut src_data);
        let mut dst = make_i32_linebuf(&mut dst_data);

        gen_rev_convert(&src, 0, &mut dst, 0, 128, 4);

        assert_eq!(dst_data, [138, 148, 158, 168]);
    }

    #[test]
    fn irv_convert_roundtrip() {
        let mut int_data = [128i32, 200, 0, 255];
        let mut float_data = [0.0f32; 4];

        let int_buf = LineBuf {
            size: 4,
            pre_size: 0,
            flags: LFT_32BIT | LFT_INTEGER,
            data: LineBufData::I32(int_data.as_mut_ptr()),
        };
        let mut float_buf = LineBuf {
            size: 4,
            pre_size: 0,
            flags: LFT_32BIT,
            data: LineBufData::F32(float_data.as_mut_ptr()),
        };

        gen_irv_convert_to_float(&int_buf, 0, &mut float_buf, 8, false, 4);

        // 128 unsigned 8-bit → (128 - 128) * (1/256) = 0.0
        assert!((float_data[0]).abs() < 1e-6);
    }

    // =====================================================================
    // Additional RCT tests
    // =====================================================================

    #[test]
    fn rct_forward_backward_all_zeros() {
        let mut r_data = [0i32; 4];
        let mut g_data = [0i32; 4];
        let mut b_data = [0i32; 4];
        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32; 4];
        let mut cb_data = [0i32; 4];
        let mut cr_data = [0i32; 4];
        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        assert_eq!(y_data, [0, 0, 0, 0]);
        assert_eq!(cb_data, [0, 0, 0, 0]);
        assert_eq!(cr_data, [0, 0, 0, 0]);

        let mut r2_data = [0i32; 4];
        let mut g2_data = [0i32; 4];
        let mut b2_data = [0i32; 4];
        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        gen_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 4);

        assert_eq!(r2_data, [0, 0, 0, 0]);
        assert_eq!(g2_data, [0, 0, 0, 0]);
        assert_eq!(b2_data, [0, 0, 0, 0]);
    }

    #[test]
    fn rct_forward_backward_max_values() {
        let mut r_data = [255i32; 4];
        let mut g_data = [255i32; 4];
        let mut b_data = [255i32; 4];
        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32; 4];
        let mut cb_data = [0i32; 4];
        let mut cr_data = [0i32; 4];
        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        let mut r2_data = [0i32; 4];
        let mut g2_data = [0i32; 4];
        let mut b2_data = [0i32; 4];
        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        gen_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 4);

        assert_eq!(r2_data, [255, 255, 255, 255]);
        assert_eq!(g2_data, [255, 255, 255, 255]);
        assert_eq!(b2_data, [255, 255, 255, 255]);

        // Also test with 10-bit max (1023)
        let mut r10 = [1023i32; 4];
        let mut g10 = [512i32; 4];
        let mut b10 = [768i32; 4];
        let r_buf = make_i32_linebuf(&mut r10);
        let g_buf = make_i32_linebuf(&mut g10);
        let b_buf = make_i32_linebuf(&mut b10);

        let mut y10 = [0i32; 4];
        let mut cb10 = [0i32; 4];
        let mut cr10 = [0i32; 4];
        let mut y_buf = make_i32_linebuf(&mut y10);
        let mut cb_buf = make_i32_linebuf(&mut cb10);
        let mut cr_buf = make_i32_linebuf(&mut cr10);

        gen_rct_forward(
            &r_buf,
            &g_buf,
            &b_buf,
            &mut y_buf,
            &mut cb_buf,
            &mut cr_buf,
            4,
        );

        let mut r10b = [0i32; 4];
        let mut g10b = [0i32; 4];
        let mut b10b = [0i32; 4];
        let y_buf2 = make_i32_linebuf(&mut y10);
        let cb_buf2 = make_i32_linebuf(&mut cb10);
        let cr_buf2 = make_i32_linebuf(&mut cr10);
        let mut r_buf2 = make_i32_linebuf(&mut r10b);
        let mut g_buf2 = make_i32_linebuf(&mut g10b);
        let mut b_buf2 = make_i32_linebuf(&mut b10b);

        gen_rct_backward(
            &y_buf2,
            &cb_buf2,
            &cr_buf2,
            &mut r_buf2,
            &mut g_buf2,
            &mut b_buf2,
            4,
        );

        assert_eq!(r10b, [1023, 1023, 1023, 1023]);
        assert_eq!(g10b, [512, 512, 512, 512]);
        assert_eq!(b10b, [768, 768, 768, 768]);
    }

    #[test]
    fn rct_forward_backward_single_pixel() {
        let mut r_data = [42i32];
        let mut g_data = [99i32];
        let mut b_data = [200i32];
        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32];
        let mut cb_data = [0i32];
        let mut cr_data = [0i32];
        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 1);

        let mut r2_data = [0i32];
        let mut g2_data = [0i32];
        let mut b2_data = [0i32];
        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        gen_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 1);

        assert_eq!(r2_data[0], 42);
        assert_eq!(g2_data[0], 99);
        assert_eq!(b2_data[0], 200);
    }

    #[test]
    fn rct_forward_backward_many_pixels() {
        let mut r_data = [
            0i32, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 255, 128, 64,
        ];
        let mut g_data = [
            0i32, 5, 15, 25, 35, 45, 55, 65, 75, 85, 110, 160, 190, 245, 130, 60,
        ];
        let mut b_data = [
            0i32, 20, 10, 35, 45, 55, 50, 80, 70, 95, 120, 140, 180, 250, 132, 58,
        ];
        let orig_r = r_data;
        let orig_g = g_data;
        let orig_b = b_data;

        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32; 16];
        let mut cb_data = [0i32; 16];
        let mut cr_data = [0i32; 16];
        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 16);

        let mut r2_data = [0i32; 16];
        let mut g2_data = [0i32; 16];
        let mut b2_data = [0i32; 16];
        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        gen_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 16);

        assert_eq!(r2_data, orig_r);
        assert_eq!(g2_data, orig_g);
        assert_eq!(b2_data, orig_b);
    }

    #[test]
    fn rct_forward_known_values() {
        let mut r_data = [100i32];
        let mut g_data = [110i32];
        let mut b_data = [120i32];
        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32];
        let mut cb_data = [0i32];
        let mut cr_data = [0i32];
        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        gen_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 1);

        // Y = (100 + 2*110 + 120) >> 2 = 440 >> 2 = 110
        assert_eq!(y_data[0], 110);
        // Cb = B - G = 120 - 110 = 10
        assert_eq!(cb_data[0], 10);
        // Cr = R - G = 100 - 110 = -10
        assert_eq!(cr_data[0], -10);
    }

    // =====================================================================
    // Additional ICT tests
    // =====================================================================

    #[test]
    fn ict_forward_backward_all_zeros() {
        let r = [0.0f32; 4];
        let g = [0.0f32; 4];
        let b = [0.0f32; 4];

        let mut y = [0.0f32; 4];
        let mut cb = [0.0f32; 4];
        let mut cr = [0.0f32; 4];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        for i in 0..4 {
            assert!(y[i].abs() < 1e-6, "Y[{i}] should be zero");
            assert!(cb[i].abs() < 1e-6, "Cb[{i}] should be zero");
            assert!(cr[i].abs() < 1e-6, "Cr[{i}] should be zero");
        }

        let mut r2 = [0.0f32; 4];
        let mut g2 = [0.0f32; 4];
        let mut b2 = [0.0f32; 4];

        gen_ict_backward(&y, &cb, &cr, &mut r2, &mut g2, &mut b2, 4);

        for i in 0..4 {
            assert!(r2[i].abs() < 1e-6, "R[{i}] should be zero after roundtrip");
            assert!(g2[i].abs() < 1e-6, "G[{i}] should be zero after roundtrip");
            assert!(b2[i].abs() < 1e-6, "B[{i}] should be zero after roundtrip");
        }
    }

    #[test]
    fn ict_forward_backward_single_pixel() {
        let r = [0.6f32];
        let g = [0.4f32];
        let b = [0.8f32];

        let mut y = [0.0f32];
        let mut cb = [0.0f32];
        let mut cr = [0.0f32];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 1);

        let mut r2 = [0.0f32];
        let mut g2 = [0.0f32];
        let mut b2 = [0.0f32];

        gen_ict_backward(&y, &cb, &cr, &mut r2, &mut g2, &mut b2, 1);

        assert!((r2[0] - r[0]).abs() < 1e-4, "R mismatch");
        assert!((g2[0] - g[0]).abs() < 1e-4, "G mismatch");
        assert!((b2[0] - b[0]).abs() < 1e-4, "B mismatch");
    }

    #[test]
    fn ict_forward_backward_many_pixels() {
        let r: [f32; 16] = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.25, 0.75, 0.33, 0.66, 0.5,
        ];
        let g: [f32; 16] = [
            0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9, 0.35, 0.65, 0.44, 0.55, 0.5,
        ];
        let b: [f32; 16] = [
            0.0, 0.3, 0.15, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.8, 0.45, 0.55, 0.22, 0.77,
            0.5,
        ];

        let mut y = [0.0f32; 16];
        let mut cb = [0.0f32; 16];
        let mut cr = [0.0f32; 16];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 16);

        let mut r2 = [0.0f32; 16];
        let mut g2 = [0.0f32; 16];
        let mut b2 = [0.0f32; 16];

        gen_ict_backward(&y, &cb, &cr, &mut r2, &mut g2, &mut b2, 16);

        for i in 0..16 {
            assert!((r2[i] - r[i]).abs() < 1e-4, "R mismatch at {i}");
            assert!((g2[i] - g[i]).abs() < 1e-4, "G mismatch at {i}");
            assert!((b2[i] - b[i]).abs() < 1e-4, "B mismatch at {i}");
        }
    }

    #[test]
    fn ict_forward_gray() {
        // Equal R=G=B should produce Cb≈0 and Cr≈0
        let r = [0.5f32, 0.2, 0.8, 1.0];
        let g = [0.5f32, 0.2, 0.8, 1.0];
        let b = [0.5f32, 0.2, 0.8, 1.0];

        let mut y = [0.0f32; 4];
        let mut cb = [0.0f32; 4];
        let mut cr = [0.0f32; 4];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        for i in 0..4 {
            assert!(
                (y[i] - r[i]).abs() < 1e-5,
                "Y[{i}] should equal input for gray"
            );
            assert!(cb[i].abs() < 1e-5, "Cb[{i}] should be ~0 for gray");
            assert!(cr[i].abs() < 1e-5, "Cr[{i}] should be ~0 for gray");
        }
    }

    #[test]
    fn ict_forward_backward_large_values() {
        let r = [0.9f32, 0.1, 0.95, 0.05];
        let g = [0.1f32, 0.9, 0.05, 0.95];
        let b = [0.5f32, 0.5, 0.8, 0.2];

        let mut y = [0.0f32; 4];
        let mut cb = [0.0f32; 4];
        let mut cr = [0.0f32; 4];

        gen_ict_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 4);

        let mut r2 = [0.0f32; 4];
        let mut g2 = [0.0f32; 4];
        let mut b2 = [0.0f32; 4];

        gen_ict_backward(&y, &cb, &cr, &mut r2, &mut g2, &mut b2, 4);

        for i in 0..4 {
            assert!((r2[i] - r[i]).abs() < 1e-4, "R mismatch at {i}");
            assert!((g2[i] - g[i]).abs() < 1e-4, "G mismatch at {i}");
            assert!((b2[i] - b[i]).abs() < 1e-4, "B mismatch at {i}");
        }
    }

    // =====================================================================
    // Additional rev_convert tests
    // =====================================================================

    #[test]
    fn rev_convert_negative_shift() {
        let mut src_data = [138i32, 148, 158, 168];
        let mut dst_data = [0i32; 4];

        let src = make_i32_linebuf(&mut src_data);
        let mut dst = make_i32_linebuf(&mut dst_data);

        gen_rev_convert(&src, 0, &mut dst, 0, -128, 4);

        assert_eq!(dst_data, [10, 20, 30, 40]);
    }

    #[test]
    fn rev_convert_roundtrip() {
        let mut src_data = [10i32, 20, 30, 40];
        let mut mid_data = [0i32; 4];
        let mut dst_data = [0i32; 4];

        let src = make_i32_linebuf(&mut src_data);
        let mut mid = make_i32_linebuf(&mut mid_data);

        gen_rev_convert(&src, 0, &mut mid, 0, 128, 4);

        assert_eq!(mid_data, [138, 148, 158, 168]);

        let mid_buf = make_i32_linebuf(&mut mid_data);
        let mut dst = make_i32_linebuf(&mut dst_data);

        gen_rev_convert(&mid_buf, 0, &mut dst, 0, -128, 4);

        assert_eq!(dst_data, [10, 20, 30, 40]);
    }

    #[test]
    fn rev_convert_zero_width() {
        let mut src_data = [10i32, 20, 30, 40];
        let mut dst_data = [99i32; 4];

        let src = make_i32_linebuf(&mut src_data);
        let mut dst = make_i32_linebuf(&mut dst_data);

        gen_rev_convert(&src, 0, &mut dst, 0, 128, 0);

        // Width=0 means no samples processed, dst unchanged
        assert_eq!(dst_data, [99, 99, 99, 99]);
    }

    // =====================================================================
    // Additional irv_convert tests
    // =====================================================================

    #[test]
    fn irv_convert_roundtrip_unsigned_8bit() {
        let mut int_data = [0i32, 64, 128, 192, 255];
        let orig = int_data;
        let mut float_data = [0.0f32; 5];

        let int_buf = make_i32_linebuf(&mut int_data);
        let mut float_buf = make_f32_linebuf(&mut float_data);

        gen_irv_convert_to_float(&int_buf, 0, &mut float_buf, 8, false, 5);

        let float_src = make_f32_linebuf(&mut float_data);
        let mut int_out = [0i32; 5];
        let mut int_out_buf = make_i32_linebuf(&mut int_out);

        gen_irv_convert_to_integer(&float_src, &mut int_out_buf, 0, 8, false, 5);

        assert_eq!(int_out, orig);
    }

    #[test]
    fn irv_convert_roundtrip_signed_8bit() {
        let mut int_data = [-128i32, -64, 0, 63, 127];
        let orig = int_data;
        let mut float_data = [0.0f32; 5];

        let int_buf = make_i32_linebuf(&mut int_data);
        let mut float_buf = make_f32_linebuf(&mut float_data);

        gen_irv_convert_to_float(&int_buf, 0, &mut float_buf, 8, true, 5);

        let float_src = make_f32_linebuf(&mut float_data);
        let mut int_out = [0i32; 5];
        let mut int_out_buf = make_i32_linebuf(&mut int_out);

        gen_irv_convert_to_integer(&float_src, &mut int_out_buf, 0, 8, true, 5);

        assert_eq!(int_out, orig);
    }

    #[test]
    fn irv_convert_to_float_unsigned_known() {
        let mut int_data = [128i32, 0, 255];
        let mut float_data = [0.0f32; 3];

        let int_buf = make_i32_linebuf(&mut int_data);
        let mut float_buf = make_f32_linebuf(&mut float_data);

        gen_irv_convert_to_float(&int_buf, 0, &mut float_buf, 8, false, 3);

        // sample 128: (128 - 128) / 256 = 0.0
        assert!((float_data[0] - 0.0).abs() < 1e-6, "128 → 0.0");
        // sample 0: (0 - 128) / 256 = -0.5
        assert!((float_data[1] - (-0.5)).abs() < 1e-6, "0 → -0.5");
        // sample 255: (255 - 128) / 256 = 127/256 ≈ 0.49609375
        assert!(
            (float_data[2] - (127.0 / 256.0)).abs() < 1e-6,
            "255 → 127/256"
        );
    }
}
