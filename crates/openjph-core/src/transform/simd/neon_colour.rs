//! NEON-accelerated colour transform routines for AArch64.
//!
//! Provides SIMD-accelerated RCT (reversible) and ICT (irreversible) colour
//! transforms using ARM NEON intrinsics. Processes 4 elements at a time.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::mem::{LineBuf, LineBufData, LFT_32BIT};
use crate::transform::colour;

// =========================================================================
// Pointer helpers
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

// =========================================================================
// NEON RCT forward (RGB → YCbCr, 32-bit integer)
// =========================================================================

/// NEON-accelerated RCT forward transform.
///
/// `Y = (R + 2G + B) >> 2`, `Cb = B - G`, `Cr = R - G`
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_rct_forward(
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
            neon_rct_forward_i32(r, g, b, y, cb, cr, repeat);
        }
    } else {
        // 64-bit path: fall back to scalar
        colour::gen_rct_forward(r, g, b, y, cb, cr, repeat);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rct_forward_i32(
    r: &LineBuf,
    g: &LineBuf,
    b: &LineBuf,
    y: &mut LineBuf,
    cb: &mut LineBuf,
    cr: &mut LineBuf,
    repeat: u32,
) {
    let mut rp = lb_i32(r);
    let mut gp = lb_i32(g);
    let mut bp = lb_i32(b);
    let mut yp = lb_i32_mut(y);
    let mut cbp = lb_i32_mut(cb);
    let mut crp = lb_i32_mut(cr);

    let simd_count = repeat / 4;
    let remainder = repeat % 4;

    for _ in 0..simd_count {
        let vr = vld1q_s32(rp);
        let vg = vld1q_s32(gp);
        let vb = vld1q_s32(bp);

        // Y = (R + 2G + B) >> 2
        let g2 = vshlq_n_s32::<1>(vg);
        let sum = vaddq_s32(vaddq_s32(vr, g2), vb);
        let vy = vshrq_n_s32::<2>(sum);
        vst1q_s32(yp, vy);

        // Cb = B - G
        vst1q_s32(cbp, vsubq_s32(vb, vg));

        // Cr = R - G
        vst1q_s32(crp, vsubq_s32(vr, vg));

        rp = rp.add(4);
        gp = gp.add(4);
        bp = bp.add(4);
        yp = yp.add(4);
        cbp = cbp.add(4);
        crp = crp.add(4);
    }

    // Scalar remainder
    for _ in 0..remainder {
        let rr = *rp;
        let gg = *gp;
        let bb = *bp;
        *yp = (rr + (gg << 1) + bb) >> 2;
        *cbp = bb - gg;
        *crp = rr - gg;
        rp = rp.add(1);
        gp = gp.add(1);
        bp = bp.add(1);
        yp = yp.add(1);
        cbp = cbp.add(1);
        crp = crp.add(1);
    }
}

// =========================================================================
// NEON RCT backward (YCbCr → RGB, 32-bit integer)
// =========================================================================

/// NEON-accelerated RCT backward transform.
///
/// `G = Y - (Cb + Cr) >> 2`, `R = Cr + G`, `B = Cb + G`
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_rct_backward(
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
            neon_rct_backward_i32(y, cb, cr, r, g, b, repeat);
        }
    } else {
        colour::gen_rct_backward(y, cb, cr, r, g, b, repeat);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rct_backward_i32(
    y: &LineBuf,
    cb: &LineBuf,
    cr: &LineBuf,
    r: &mut LineBuf,
    g: &mut LineBuf,
    b: &mut LineBuf,
    repeat: u32,
) {
    let mut yp = lb_i32(y);
    let mut cbp = lb_i32(cb);
    let mut crp = lb_i32(cr);
    let mut rp = lb_i32_mut(r);
    let mut gp = lb_i32_mut(g);
    let mut bp = lb_i32_mut(b);

    let simd_count = repeat / 4;
    let remainder = repeat % 4;

    for _ in 0..simd_count {
        let vy = vld1q_s32(yp);
        let vcb = vld1q_s32(cbp);
        let vcr = vld1q_s32(crp);

        // G = Y - (Cb + Cr) >> 2
        let sum = vaddq_s32(vcb, vcr);
        let shifted = vshrq_n_s32::<2>(sum);
        let vg = vsubq_s32(vy, shifted);

        // R = Cr + G
        vst1q_s32(rp, vaddq_s32(vcr, vg));
        // G
        vst1q_s32(gp, vg);
        // B = Cb + G
        vst1q_s32(bp, vaddq_s32(vcb, vg));

        yp = yp.add(4);
        cbp = cbp.add(4);
        crp = crp.add(4);
        rp = rp.add(4);
        gp = gp.add(4);
        bp = bp.add(4);
    }

    for _ in 0..remainder {
        let yy = *yp;
        let cbb = *cbp;
        let crr = *crp;
        let gg = yy - ((cbb + crr) >> 2);
        *rp = crr + gg;
        *gp = gg;
        *bp = cbb + gg;
        yp = yp.add(1);
        cbp = cbp.add(1);
        crp = crp.add(1);
        rp = rp.add(1);
        gp = gp.add(1);
        bp = bp.add(1);
    }
}

// =========================================================================
// NEON ICT forward (RGB → YCbCr, float)
// =========================================================================

/// NEON-accelerated ICT forward transform.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_ict_forward(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
    repeat: u32,
) {
    unsafe {
        neon_ict_forward_inner(r, g, b, y, cb, cr, repeat);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_ict_forward_inner(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
    repeat: u32,
) {
    let v_alpha_r = vdupq_n_f32(colour::ALPHA_RF);
    let v_alpha_g = vdupq_n_f32(colour::ALPHA_GF);
    let v_alpha_b = vdupq_n_f32(colour::ALPHA_BF);
    let v_beta_cb = vdupq_n_f32(colour::BETA_CBF);
    let v_beta_cr = vdupq_n_f32(colour::BETA_CRF);

    let simd_count = repeat / 4;
    let remainder = repeat % 4;
    let mut i = 0usize;

    for _ in 0..simd_count {
        let vr = vld1q_f32(r.as_ptr().add(i));
        let vg = vld1q_f32(g.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));

        // Y = αR*R + αG*G + αB*B
        let vy = vfmaq_f32(
            vfmaq_f32(vmulq_f32(v_alpha_r, vr), v_alpha_g, vg),
            v_alpha_b,
            vb,
        );
        vst1q_f32(y.as_mut_ptr().add(i), vy);

        // Cb = βCb * (B - Y)
        let vcb = vmulq_f32(v_beta_cb, vsubq_f32(vb, vy));
        vst1q_f32(cb.as_mut_ptr().add(i), vcb);

        // Cr = βCr * (R - Y)
        let vcr = vmulq_f32(v_beta_cr, vsubq_f32(vr, vy));
        vst1q_f32(cr.as_mut_ptr().add(i), vcr);

        i += 4;
    }

    // Scalar remainder
    for j in 0..remainder as usize {
        let idx = i + j;
        let yv = colour::ALPHA_RF * r[idx] + colour::ALPHA_GF * g[idx] + colour::ALPHA_BF * b[idx];
        y[idx] = yv;
        cb[idx] = colour::BETA_CBF * (b[idx] - yv);
        cr[idx] = colour::BETA_CRF * (r[idx] - yv);
    }
}

// =========================================================================
// NEON ICT backward (YCbCr → RGB, float)
// =========================================================================

/// NEON-accelerated ICT backward transform.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_ict_backward(
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    repeat: u32,
) {
    unsafe {
        neon_ict_backward_inner(y, cb, cr, r, g, b, repeat);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_ict_backward_inner(
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    repeat: u32,
) {
    let v_gamma_cr2r = vdupq_n_f32(colour::GAMMA_CR2R);
    let v_gamma_cb2b = vdupq_n_f32(colour::GAMMA_CB2B);
    let v_neg_gamma_cr2g = vdupq_n_f32(-colour::GAMMA_CR2G);
    let v_neg_gamma_cb2g = vdupq_n_f32(-colour::GAMMA_CB2G);

    let simd_count = repeat / 4;
    let remainder = repeat % 4;
    let mut i = 0usize;

    for _ in 0..simd_count {
        let vy = vld1q_f32(y.as_ptr().add(i));
        let vcb = vld1q_f32(cb.as_ptr().add(i));
        let vcr = vld1q_f32(cr.as_ptr().add(i));

        // G = Y - γCr2G*Cr - γCb2G*Cb
        let vg = vfmaq_f32(vfmaq_f32(vy, v_neg_gamma_cr2g, vcr), v_neg_gamma_cb2g, vcb);
        vst1q_f32(g.as_mut_ptr().add(i), vg);

        // R = Y + γCr2R*Cr
        let vr = vfmaq_f32(vy, v_gamma_cr2r, vcr);
        vst1q_f32(r.as_mut_ptr().add(i), vr);

        // B = Y + γCb2B*Cb
        let vb = vfmaq_f32(vy, v_gamma_cb2b, vcb);
        vst1q_f32(b.as_mut_ptr().add(i), vb);

        i += 4;
    }

    for j in 0..remainder as usize {
        let idx = i + j;
        g[idx] = y[idx] - colour::GAMMA_CR2G * cr[idx] - colour::GAMMA_CB2G * cb[idx];
        r[idx] = y[idx] + colour::GAMMA_CR2R * cr[idx];
        b[idx] = y[idx] + colour::GAMMA_CB2B * cb[idx];
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;
    use crate::mem::{LineBufData, LFT_INTEGER};

    fn make_i32_linebuf(data: &mut [i32]) -> LineBuf {
        LineBuf {
            size: data.len(),
            pre_size: 0,
            flags: LFT_32BIT | LFT_INTEGER,
            data: LineBufData::I32(data.as_mut_ptr()),
        }
    }

    /// Verify NEON RCT forward matches scalar for various widths.
    #[test]
    fn neon_rct_forward_matches_scalar() {
        for width in [1, 3, 4, 7, 8, 16, 33, 64] {
            let mut r_data: Vec<i32> = (0..width).map(|i| (i * 7 + 100) as i32).collect();
            let mut g_data: Vec<i32> = (0..width).map(|i| (i * 5 + 110) as i32).collect();
            let mut b_data: Vec<i32> = (0..width).map(|i| (i * 3 + 120) as i32).collect();

            let r = make_i32_linebuf(&mut r_data);
            let g = make_i32_linebuf(&mut g_data);
            let b = make_i32_linebuf(&mut b_data);

            let mut y_scalar = vec![0i32; width];
            let mut cb_scalar = vec![0i32; width];
            let mut cr_scalar = vec![0i32; width];
            let mut y_neon = vec![0i32; width];
            let mut cb_neon = vec![0i32; width];
            let mut cr_neon = vec![0i32; width];

            let mut ys = make_i32_linebuf(&mut y_scalar);
            let mut cbs = make_i32_linebuf(&mut cb_scalar);
            let mut crs = make_i32_linebuf(&mut cr_scalar);
            colour::gen_rct_forward(&r, &g, &b, &mut ys, &mut cbs, &mut crs, width as u32);

            let mut yn = make_i32_linebuf(&mut y_neon);
            let mut cbn = make_i32_linebuf(&mut cb_neon);
            let mut crn = make_i32_linebuf(&mut cr_neon);
            neon_rct_forward(&r, &g, &b, &mut yn, &mut cbn, &mut crn, width as u32);

            assert_eq!(y_scalar, y_neon, "Y mismatch at width={width}");
            assert_eq!(cb_scalar, cb_neon, "Cb mismatch at width={width}");
            assert_eq!(cr_scalar, cr_neon, "Cr mismatch at width={width}");
        }
    }

    /// Verify NEON RCT backward matches scalar.
    #[test]
    fn neon_rct_backward_matches_scalar() {
        for width in [1, 3, 4, 7, 8, 16, 33, 64] {
            let mut y_data: Vec<i32> = (0..width).map(|i| (i * 3 + 50) as i32).collect();
            let mut cb_data: Vec<i32> = (0..width).map(|i| i as i32 * 2 - 30).collect();
            let mut cr_data: Vec<i32> = (0..width).map(|i| -(i as i32) + 20).collect();

            let y = make_i32_linebuf(&mut y_data);
            let cb = make_i32_linebuf(&mut cb_data);
            let cr = make_i32_linebuf(&mut cr_data);

            let mut r_scalar = vec![0i32; width];
            let mut g_scalar = vec![0i32; width];
            let mut b_scalar = vec![0i32; width];
            let mut r_neon = vec![0i32; width];
            let mut g_neon = vec![0i32; width];
            let mut b_neon = vec![0i32; width];

            let mut rs = make_i32_linebuf(&mut r_scalar);
            let mut gs = make_i32_linebuf(&mut g_scalar);
            let mut bs = make_i32_linebuf(&mut b_scalar);
            colour::gen_rct_backward(&y, &cb, &cr, &mut rs, &mut gs, &mut bs, width as u32);

            let mut rn = make_i32_linebuf(&mut r_neon);
            let mut gn = make_i32_linebuf(&mut g_neon);
            let mut bn = make_i32_linebuf(&mut b_neon);
            neon_rct_backward(&y, &cb, &cr, &mut rn, &mut gn, &mut bn, width as u32);

            assert_eq!(r_scalar, r_neon, "R mismatch at width={width}");
            assert_eq!(g_scalar, g_neon, "G mismatch at width={width}");
            assert_eq!(b_scalar, b_neon, "B mismatch at width={width}");
        }
    }

    /// Verify NEON ICT forward matches scalar.
    #[test]
    fn neon_ict_forward_matches_scalar() {
        for width in [1, 3, 4, 7, 8, 16, 33, 64] {
            let r: Vec<f32> = (0..width).map(|i| i as f32 * 0.01 + 0.3).collect();
            let g: Vec<f32> = (0..width).map(|i| i as f32 * 0.02 + 0.4).collect();
            let b: Vec<f32> = (0..width).map(|i| i as f32 * 0.015 + 0.2).collect();

            let mut y_scalar = vec![0.0f32; width];
            let mut cb_scalar = vec![0.0f32; width];
            let mut cr_scalar = vec![0.0f32; width];
            let mut y_neon = vec![0.0f32; width];
            let mut cb_neon = vec![0.0f32; width];
            let mut cr_neon = vec![0.0f32; width];

            colour::gen_ict_forward(
                &r,
                &g,
                &b,
                &mut y_scalar,
                &mut cb_scalar,
                &mut cr_scalar,
                width as u32,
            );
            neon_ict_forward(
                &r,
                &g,
                &b,
                &mut y_neon,
                &mut cb_neon,
                &mut cr_neon,
                width as u32,
            );

            for i in 0..width {
                assert!(
                    (y_scalar[i] - y_neon[i]).abs() < 1e-5,
                    "Y mismatch at width={width} idx={i}: scalar={} neon={}",
                    y_scalar[i],
                    y_neon[i],
                );
                assert!(
                    (cb_scalar[i] - cb_neon[i]).abs() < 1e-5,
                    "Cb mismatch at width={width} idx={i}",
                );
                assert!(
                    (cr_scalar[i] - cr_neon[i]).abs() < 1e-5,
                    "Cr mismatch at width={width} idx={i}",
                );
            }
        }
    }

    /// Verify NEON ICT backward matches scalar.
    #[test]
    fn neon_ict_backward_matches_scalar() {
        for width in [1, 3, 4, 7, 8, 16, 33, 64] {
            let y: Vec<f32> = (0..width).map(|i| i as f32 * 0.01 + 0.5).collect();
            let cb: Vec<f32> = (0..width).map(|i| i as f32 * 0.005 - 0.1).collect();
            let cr: Vec<f32> = (0..width).map(|i| -(i as f32) * 0.003 + 0.05).collect();

            let mut r_scalar = vec![0.0f32; width];
            let mut g_scalar = vec![0.0f32; width];
            let mut b_scalar = vec![0.0f32; width];
            let mut r_neon = vec![0.0f32; width];
            let mut g_neon = vec![0.0f32; width];
            let mut b_neon = vec![0.0f32; width];

            colour::gen_ict_backward(
                &y,
                &cb,
                &cr,
                &mut r_scalar,
                &mut g_scalar,
                &mut b_scalar,
                width as u32,
            );
            neon_ict_backward(
                &y,
                &cb,
                &cr,
                &mut r_neon,
                &mut g_neon,
                &mut b_neon,
                width as u32,
            );

            for i in 0..width {
                assert!(
                    (r_scalar[i] - r_neon[i]).abs() < 1e-5,
                    "R mismatch at width={width} idx={i}",
                );
                assert!(
                    (g_scalar[i] - g_neon[i]).abs() < 1e-5,
                    "G mismatch at width={width} idx={i}",
                );
                assert!(
                    (b_scalar[i] - b_neon[i]).abs() < 1e-5,
                    "B mismatch at width={width} idx={i}",
                );
            }
        }
    }

    /// RCT forward→backward roundtrip through NEON.
    #[test]
    fn neon_rct_roundtrip() {
        let mut r_data = [100i32, 150, 200, 50, 75, 225, 10, 180];
        let mut g_data = [110i32, 160, 190, 60, 80, 210, 20, 170];
        let mut b_data = [120i32, 140, 180, 70, 85, 200, 30, 160];

        let r = make_i32_linebuf(&mut r_data);
        let g = make_i32_linebuf(&mut g_data);
        let b = make_i32_linebuf(&mut b_data);

        let mut y_data = [0i32; 8];
        let mut cb_data = [0i32; 8];
        let mut cr_data = [0i32; 8];

        let mut y = make_i32_linebuf(&mut y_data);
        let mut cb = make_i32_linebuf(&mut cb_data);
        let mut cr = make_i32_linebuf(&mut cr_data);

        neon_rct_forward(&r, &g, &b, &mut y, &mut cb, &mut cr, 8);

        let mut r2_data = [0i32; 8];
        let mut g2_data = [0i32; 8];
        let mut b2_data = [0i32; 8];

        let y2 = make_i32_linebuf(&mut y_data);
        let cb2 = make_i32_linebuf(&mut cb_data);
        let cr2 = make_i32_linebuf(&mut cr_data);
        let mut r2 = make_i32_linebuf(&mut r2_data);
        let mut g2 = make_i32_linebuf(&mut g2_data);
        let mut b2 = make_i32_linebuf(&mut b2_data);

        neon_rct_backward(&y2, &cb2, &cr2, &mut r2, &mut g2, &mut b2, 8);

        assert_eq!(r2_data, [100, 150, 200, 50, 75, 225, 10, 180]);
        assert_eq!(g2_data, [110, 160, 190, 60, 80, 210, 20, 170]);
        assert_eq!(b2_data, [120, 140, 180, 70, 85, 200, 30, 160]);
    }
}
