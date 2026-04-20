//! x86 SSE2/AVX2-accelerated colour transform routines.
//!
//! All code gated behind `#[cfg(target_arch = "x86_64")]`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
// SSE2 RCT forward
// =========================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_rct_forward(
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
            sse2_rct_forward_i32(r, g, b, y, cb, cr, repeat);
        }
    } else {
        colour::gen_rct_forward(r, g, b, y, cb, cr, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_rct_forward_i32(
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
        let vr = _mm_loadu_si128(rp as *const __m128i);
        let vg = _mm_loadu_si128(gp as *const __m128i);
        let vb = _mm_loadu_si128(bp as *const __m128i);
        let g2 = _mm_slli_epi32(vg, 1);
        let sum = _mm_add_epi32(_mm_add_epi32(vr, g2), vb);
        _mm_storeu_si128(yp as *mut __m128i, _mm_srai_epi32(sum, 2));
        _mm_storeu_si128(cbp as *mut __m128i, _mm_sub_epi32(vb, vg));
        _mm_storeu_si128(crp as *mut __m128i, _mm_sub_epi32(vr, vg));
        rp = rp.add(4);
        gp = gp.add(4);
        bp = bp.add(4);
        yp = yp.add(4);
        cbp = cbp.add(4);
        crp = crp.add(4);
    }
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
// SSE2 RCT backward
// =========================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_rct_backward(
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
            sse2_rct_backward_i32(y, cb, cr, r, g, b, repeat);
        }
    } else {
        colour::gen_rct_backward(y, cb, cr, r, g, b, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_rct_backward_i32(
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
        let vy = _mm_loadu_si128(yp as *const __m128i);
        let vcb = _mm_loadu_si128(cbp as *const __m128i);
        let vcr = _mm_loadu_si128(crp as *const __m128i);
        let sum = _mm_add_epi32(vcb, vcr);
        let shifted = _mm_srai_epi32(sum, 2);
        let vg = _mm_sub_epi32(vy, shifted);
        _mm_storeu_si128(rp as *mut __m128i, _mm_add_epi32(vcr, vg));
        _mm_storeu_si128(gp as *mut __m128i, vg);
        _mm_storeu_si128(bp as *mut __m128i, _mm_add_epi32(vcb, vg));
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
// SSE2 ICT forward/backward
// =========================================================================

#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_ict_forward(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
    repeat: u32,
) {
    unsafe {
        sse2_ict_forward_inner(r, g, b, y, cb, cr, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_ict_forward_inner(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
    repeat: u32,
) {
    let v_ar = _mm_set1_ps(colour::ALPHA_RF);
    let v_ag = _mm_set1_ps(colour::ALPHA_GF);
    let v_ab = _mm_set1_ps(colour::ALPHA_BF);
    let v_bcb = _mm_set1_ps(colour::BETA_CBF);
    let v_bcr = _mm_set1_ps(colour::BETA_CRF);
    let simd_count = repeat / 4;
    let remainder = repeat % 4;
    let mut i = 0usize;

    for _ in 0..simd_count {
        let vr = _mm_loadu_ps(r.as_ptr().add(i));
        let vg = _mm_loadu_ps(g.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vy = _mm_add_ps(
            _mm_add_ps(_mm_mul_ps(v_ar, vr), _mm_mul_ps(v_ag, vg)),
            _mm_mul_ps(v_ab, vb),
        );
        _mm_storeu_ps(y.as_mut_ptr().add(i), vy);
        _mm_storeu_ps(
            cb.as_mut_ptr().add(i),
            _mm_mul_ps(v_bcb, _mm_sub_ps(vb, vy)),
        );
        _mm_storeu_ps(
            cr.as_mut_ptr().add(i),
            _mm_mul_ps(v_bcr, _mm_sub_ps(vr, vy)),
        );
        i += 4;
    }
    for j in 0..remainder as usize {
        let idx = i + j;
        let yv = colour::ALPHA_RF * r[idx] + colour::ALPHA_GF * g[idx] + colour::ALPHA_BF * b[idx];
        y[idx] = yv;
        cb[idx] = colour::BETA_CBF * (b[idx] - yv);
        cr[idx] = colour::BETA_CRF * (r[idx] - yv);
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_ict_backward(
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    repeat: u32,
) {
    unsafe {
        sse2_ict_backward_inner(y, cb, cr, r, g, b, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_ict_backward_inner(
    y: &[f32],
    cb: &[f32],
    cr: &[f32],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    repeat: u32,
) {
    let v_cr2r = _mm_set1_ps(colour::GAMMA_CR2R);
    let v_cb2b = _mm_set1_ps(colour::GAMMA_CB2B);
    let v_ncr2g = _mm_set1_ps(-colour::GAMMA_CR2G);
    let v_ncb2g = _mm_set1_ps(-colour::GAMMA_CB2G);
    let simd_count = repeat / 4;
    let remainder = repeat % 4;
    let mut i = 0usize;

    for _ in 0..simd_count {
        let vy = _mm_loadu_ps(y.as_ptr().add(i));
        let vcb = _mm_loadu_ps(cb.as_ptr().add(i));
        let vcr = _mm_loadu_ps(cr.as_ptr().add(i));
        let vg = _mm_add_ps(
            _mm_add_ps(vy, _mm_mul_ps(v_ncr2g, vcr)),
            _mm_mul_ps(v_ncb2g, vcb),
        );
        _mm_storeu_ps(g.as_mut_ptr().add(i), vg);
        _mm_storeu_ps(
            r.as_mut_ptr().add(i),
            _mm_add_ps(vy, _mm_mul_ps(v_cr2r, vcr)),
        );
        _mm_storeu_ps(
            b.as_mut_ptr().add(i),
            _mm_add_ps(vy, _mm_mul_ps(v_cb2b, vcb)),
        );
        i += 4;
    }
    for j in 0..remainder as usize {
        let idx = i + j;
        g[idx] = y[idx] - colour::GAMMA_CR2G * cr[idx] - colour::GAMMA_CB2G * cb[idx];
        r[idx] = y[idx] + colour::GAMMA_CR2R * cr[idx];
        b[idx] = y[idx] + colour::GAMMA_CB2B * cb[idx];
    }
}
