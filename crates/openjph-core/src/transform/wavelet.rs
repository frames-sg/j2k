//! Discrete wavelet transform (DWT) — 5/3 reversible and 9/7 irreversible.
//!
//! Port of `ojph_transform.cpp` generic implementations.
//!
//! These functions operate on [`LineBuf`] buffers. The 32-bit vs 64-bit
//! dispatch mirrors the C++ implementation: for reversible transforms, both
//! `i32` and `i64` paths are provided; for irreversible transforms, only
//! the `f32` path is used.

#![allow(clippy::manual_swap)]

use super::{LiftingStep, ParamAtk};
use crate::mem::{LineBuf, LineBufData, LFT_32BIT};

// =========================================================================
// Helpers to get raw slices from LineBuf
// =========================================================================

/// Returns a mutable `i32` pointer and validates the flag.
#[inline]
unsafe fn linebuf_i32_mut(buf: &mut LineBuf) -> *mut i32 {
    match buf.data {
        LineBufData::I32(p) => p,
        _ => panic!("expected i32 LineBuf"),
    }
}

/// Returns a const `i32` pointer.
#[inline]
unsafe fn linebuf_i32(buf: &LineBuf) -> *const i32 {
    match buf.data {
        LineBufData::I32(p) => p as *const i32,
        _ => panic!("expected i32 LineBuf"),
    }
}

/// Returns a mutable `i64` pointer.
#[inline]
unsafe fn linebuf_i64_mut(buf: &mut LineBuf) -> *mut i64 {
    match buf.data {
        LineBufData::I64(p) => p,
        _ => panic!("expected i64 LineBuf"),
    }
}

/// Returns a const `i64` pointer.
#[inline]
unsafe fn linebuf_i64(buf: &LineBuf) -> *const i64 {
    match buf.data {
        LineBufData::I64(p) => p as *const i64,
        _ => panic!("expected i64 LineBuf"),
    }
}

/// Returns a mutable `f32` pointer.
#[inline]
unsafe fn linebuf_f32_mut(buf: &mut LineBuf) -> *mut f32 {
    match buf.data {
        LineBufData::F32(p) => p,
        _ => panic!("expected f32 LineBuf"),
    }
}

/// Returns a const `f32` pointer.
#[inline]
unsafe fn linebuf_f32(buf: &LineBuf) -> *const f32 {
    match buf.data {
        LineBufData::F32(p) => p as *const f32,
        _ => panic!("expected f32 LineBuf"),
    }
}

// =========================================================================
// Reversible vertical lifting step — 32-bit
// =========================================================================

#[inline]
pub(crate) fn gen_rev_vert_step32(
    s: &LiftingStep,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
    synthesis: bool,
) {
    let rev = s.rev();
    let a = rev.a as i32;
    let b = rev.b as i32;
    let e = rev.e;

    unsafe {
        let mut dst = linebuf_i32_mut(aug);
        let mut src1 = linebuf_i32(sig);
        let mut src2 = linebuf_i32(other);

        if a == 1 {
            // 5/3 update and any case with a == 1
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b + *src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b + *src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else if a == -1 && b == 1 && e == 1 {
            // 5/3 predict
            if synthesis {
                for _ in 0..repeat {
                    *dst += (*src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst -= (*src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else if a == -1 {
            // any case with a == -1, which is not 5/3 predict
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b - (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b - (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else {
            // general case
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b + a * (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b + a * (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        }
    }
}

// =========================================================================
// Reversible vertical lifting step — 64-bit
// =========================================================================

#[inline]
pub(crate) fn gen_rev_vert_step64(
    s: &LiftingStep,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
    synthesis: bool,
) {
    let rev = s.rev();
    let a = rev.a as i64;
    let b = rev.b as i64;
    let e = rev.e;

    unsafe {
        let mut dst = linebuf_i64_mut(aug);
        let mut src1 = linebuf_i64(sig);
        let mut src2 = linebuf_i64(other);

        if a == 1 {
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b + *src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b + *src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else if a == -1 && b == 1 && e == 1 {
            if synthesis {
                for _ in 0..repeat {
                    *dst += (*src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst -= (*src1 + *src2) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else if a == -1 {
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b - (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b - (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        } else {
            if synthesis {
                for _ in 0..repeat {
                    *dst -= (b + a * (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            } else {
                for _ in 0..repeat {
                    *dst += (b + a * (*src1 + *src2)) >> e;
                    dst = dst.add(1);
                    src1 = src1.add(1);
                    src2 = src2.add(1);
                }
            }
        }
    }
}

// =========================================================================
// Public: Reversible vertical step (dispatches 32/64)
// =========================================================================

/// Generic reversible vertical lifting step.
///
/// Dispatches to 32-bit or 64-bit based on `LineBuf` flags.
pub(crate) fn gen_rev_vert_step(
    s: &LiftingStep,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
    synthesis: bool,
) {
    if (sig.flags & LFT_32BIT) != 0
        || (aug.flags & LFT_32BIT) != 0
        || (other.flags & LFT_32BIT) != 0
    {
        gen_rev_vert_step32(s, sig, other, aug, repeat, synthesis);
    } else {
        gen_rev_vert_step64(s, sig, other, aug, repeat, synthesis);
    }
}

// =========================================================================
// Reversible horizontal analysis (forward) — 32-bit
// =========================================================================

fn gen_rev_horz_ana32(
    atk: &ParamAtk,
    ldst: &mut LineBuf,
    hdst: &mut LineBuf,
    src: &LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            // Split src into ldst (even samples) and hdst (odd samples)
            let mut dph = linebuf_i32_mut(hdst);
            let mut dpl = linebuf_i32_mut(ldst);
            let mut sp = linebuf_i32(src);
            let mut w = width;

            if !even {
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 1;
            }
            while w > 1 {
                *dpl = *sp;
                dpl = dpl.add(1);
                sp = sp.add(1);
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 2;
            }
            if w > 0 {
                *dpl = *sp;
            }

            let mut hp = linebuf_i32_mut(hdst);
            let mut lp = linebuf_i32_mut(ldst);
            let mut l_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut h_width = (width + if even { 0 } else { 1 }) >> 1;
            let mut ev = even;
            let num_steps = atk.get_num_steps();

            for j in (1..=num_steps).rev() {
                let s = atk.get_step(j - 1);
                let rev = s.rev();
                let a = rev.a as i32;
                let b = rev.b as i32;
                let e = rev.e;

                // symmetric extension
                *lp.sub(1) = *lp;
                *lp.add(l_width as usize) = *lp.add(l_width as usize - 1);

                // lifting step
                let mut sp_l = lp.add(if ev { 1 } else { 0 }) as *const i32;
                let mut dp = hp;

                if a == 1 {
                    for _ in 0..h_width {
                        *dp += (b + (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 && b == 1 && e == 1 {
                    for _ in 0..h_width {
                        *dp -= (*sp_l.sub(1) + *sp_l) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 {
                    for _ in 0..h_width {
                        *dp += (b - (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else {
                    for _ in 0..h_width {
                        *dp += (b + a * (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                }

                // swap buffers
                let t = lp;
                lp = hp;
                hp = t;
                ev = !ev;
                let tw = l_width;
                l_width = h_width;
                h_width = tw;
            }
        }
    } else if even {
        unsafe {
            *linebuf_i32_mut(ldst) = *linebuf_i32(src);
        }
    } else {
        unsafe {
            *linebuf_i32_mut(hdst) = *linebuf_i32(src) << 1;
        }
    }
}

// =========================================================================
// Reversible horizontal analysis (forward) — 64-bit
// =========================================================================

fn gen_rev_horz_ana64(
    atk: &ParamAtk,
    ldst: &mut LineBuf,
    hdst: &mut LineBuf,
    src: &LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            let mut dph = linebuf_i64_mut(hdst);
            let mut dpl = linebuf_i64_mut(ldst);
            let mut sp = linebuf_i64(src);
            let mut w = width;

            if !even {
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 1;
            }
            while w > 1 {
                *dpl = *sp;
                dpl = dpl.add(1);
                sp = sp.add(1);
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 2;
            }
            if w > 0 {
                *dpl = *sp;
            }

            let mut hp = linebuf_i64_mut(hdst);
            let mut lp = linebuf_i64_mut(ldst);
            let mut l_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut h_width = (width + if even { 0 } else { 1 }) >> 1;
            let mut ev = even;
            let num_steps = atk.get_num_steps();

            for j in (1..=num_steps).rev() {
                let s = atk.get_step(j - 1);
                let rev = s.rev();
                let a = rev.a as i64;
                let b = rev.b as i64;
                let e = rev.e;

                *lp.sub(1) = *lp;
                *lp.add(l_width as usize) = *lp.add(l_width as usize - 1);

                let mut sp_l = lp.add(if ev { 1 } else { 0 }) as *const i64;
                let mut dp = hp;

                if a == 1 {
                    for _ in 0..h_width {
                        *dp += (b + (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 && b == 1 && e == 1 {
                    for _ in 0..h_width {
                        *dp -= (*sp_l.sub(1) + *sp_l) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 {
                    for _ in 0..h_width {
                        *dp += (b - (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                } else {
                    for _ in 0..h_width {
                        *dp += (b + a * (*sp_l.sub(1) + *sp_l)) >> e;
                        sp_l = sp_l.add(1);
                        dp = dp.add(1);
                    }
                }

                let t = lp;
                lp = hp;
                hp = t;
                ev = !ev;
                let tw = l_width;
                l_width = h_width;
                h_width = tw;
            }
        }
    } else if even {
        unsafe {
            *linebuf_i64_mut(ldst) = *linebuf_i64(src);
        }
    } else {
        unsafe {
            *linebuf_i64_mut(hdst) = *linebuf_i64(src) << 1;
        }
    }
}

/// Generic reversible horizontal analysis.
pub(crate) fn gen_rev_horz_ana(
    atk: &ParamAtk,
    ldst: &mut LineBuf,
    hdst: &mut LineBuf,
    src: &LineBuf,
    width: u32,
    even: bool,
) {
    if (src.flags & LFT_32BIT) != 0 {
        gen_rev_horz_ana32(atk, ldst, hdst, src, width, even);
    } else {
        gen_rev_horz_ana64(atk, ldst, hdst, src, width, even);
    }
}

// =========================================================================
// Reversible horizontal synthesis (inverse) — 32-bit
// =========================================================================

fn gen_rev_horz_syn32(
    atk: &ParamAtk,
    dst: &mut LineBuf,
    lsrc: &mut LineBuf,
    hsrc: &mut LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            let mut ev = even;
            let mut oth = linebuf_i32_mut(hsrc);
            let mut aug = linebuf_i32_mut(lsrc);
            let mut aug_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut oth_width = (width + if even { 0 } else { 1 }) >> 1;
            let num_steps = atk.get_num_steps();

            for j in 0..num_steps {
                let s = atk.get_step(j);
                let rev = s.rev();
                let a = rev.a as i32;
                let b = rev.b as i32;
                let e = rev.e;

                // symmetric extension
                *oth.sub(1) = *oth;
                *oth.add(oth_width as usize) = *oth.add(oth_width as usize - 1);

                let mut sp = oth.add(if ev { 0 } else { 1 }) as *const i32;
                let mut dp = aug;

                if a == 1 {
                    for _ in 0..aug_width {
                        *dp -= (b + (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 && b == 1 && e == 1 {
                    for _ in 0..aug_width {
                        *dp += (*sp.sub(1) + *sp) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 {
                    for _ in 0..aug_width {
                        *dp -= (b - (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else {
                    for _ in 0..aug_width {
                        *dp -= (b + a * (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                }

                // swap buffers
                let t = aug;
                aug = oth;
                oth = t;
                ev = !ev;
                let tw = aug_width;
                aug_width = oth_width;
                oth_width = tw;
            }

            // Interleave lsrc and hsrc into dst
            let mut sph = linebuf_i32(hsrc);
            let mut spl = linebuf_i32(lsrc);
            let mut dp = linebuf_i32_mut(dst);
            let mut w = width;

            if !even {
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 1;
            }
            while w > 1 {
                *dp = *spl;
                dp = dp.add(1);
                spl = spl.add(1);
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 2;
            }
            if w > 0 {
                *dp = *spl;
            }
        }
    } else if even {
        unsafe {
            *linebuf_i32_mut(dst) = *linebuf_i32(lsrc);
        }
    } else {
        unsafe {
            *linebuf_i32_mut(dst) = *linebuf_i32(hsrc) >> 1;
        }
    }
}

// =========================================================================
// Reversible horizontal synthesis (inverse) — 64-bit
// =========================================================================

fn gen_rev_horz_syn64(
    atk: &ParamAtk,
    dst: &mut LineBuf,
    lsrc: &mut LineBuf,
    hsrc: &mut LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            let mut ev = even;
            let mut oth = linebuf_i64_mut(hsrc);
            let mut aug = linebuf_i64_mut(lsrc);
            let mut aug_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut oth_width = (width + if even { 0 } else { 1 }) >> 1;
            let num_steps = atk.get_num_steps();

            for j in 0..num_steps {
                let s = atk.get_step(j);
                let rev = s.rev();
                let a = rev.a as i64;
                let b = rev.b as i64;
                let e = rev.e;

                *oth.sub(1) = *oth;
                *oth.add(oth_width as usize) = *oth.add(oth_width as usize - 1);

                let mut sp = oth.add(if ev { 0 } else { 1 }) as *const i64;
                let mut dp = aug;

                if a == 1 {
                    for _ in 0..aug_width {
                        *dp -= (b + (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 && b == 1 && e == 1 {
                    for _ in 0..aug_width {
                        *dp += (*sp.sub(1) + *sp) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else if a == -1 {
                    for _ in 0..aug_width {
                        *dp -= (b - (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                } else {
                    for _ in 0..aug_width {
                        *dp -= (b + a * (*sp.sub(1) + *sp)) >> e;
                        sp = sp.add(1);
                        dp = dp.add(1);
                    }
                }

                let t = aug;
                aug = oth;
                oth = t;
                ev = !ev;
                let tw = aug_width;
                aug_width = oth_width;
                oth_width = tw;
            }

            let mut sph = linebuf_i64(hsrc);
            let mut spl = linebuf_i64(lsrc);
            let mut dp = linebuf_i64_mut(dst);
            let mut w = width;

            if !even {
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 1;
            }
            while w > 1 {
                *dp = *spl;
                dp = dp.add(1);
                spl = spl.add(1);
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 2;
            }
            if w > 0 {
                *dp = *spl;
            }
        }
    } else if even {
        unsafe {
            *linebuf_i64_mut(dst) = *linebuf_i64(lsrc);
        }
    } else {
        unsafe {
            *linebuf_i64_mut(dst) = *linebuf_i64(hsrc) >> 1;
        }
    }
}

/// Generic reversible horizontal synthesis.
pub(crate) fn gen_rev_horz_syn(
    atk: &ParamAtk,
    dst: &mut LineBuf,
    lsrc: &mut LineBuf,
    hsrc: &mut LineBuf,
    width: u32,
    even: bool,
) {
    if (dst.flags & LFT_32BIT) != 0 {
        gen_rev_horz_syn32(atk, dst, lsrc, hsrc, width, even);
    } else {
        gen_rev_horz_syn64(atk, dst, lsrc, hsrc, width, even);
    }
}

// =========================================================================
// Irreversible vertical lifting step
// =========================================================================

/// Generic irreversible vertical lifting step.
///
/// Applies `dst[i] += a * (src1[i] + src2[i])` for analysis,
/// or `dst[i] -= a * (…)` for synthesis (negated `a`).
pub(crate) fn gen_irv_vert_step(
    s: &LiftingStep,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
    synthesis: bool,
) {
    let mut a = s.irv().a;
    if synthesis {
        a = -a;
    }

    unsafe {
        let mut dst = linebuf_f32_mut(aug);
        let mut src1 = linebuf_f32(sig);
        let mut src2 = linebuf_f32(other);
        for _ in 0..repeat {
            *dst += a * (*src1 + *src2);
            dst = dst.add(1);
            src1 = src1.add(1);
            src2 = src2.add(1);
        }
    }
}

// =========================================================================
// Irreversible vertical times K
// =========================================================================

/// Multiply every sample in `aug` by normalization constant `k`.
pub(crate) fn gen_irv_vert_times_k(k: f32, aug: &mut LineBuf, repeat: u32) {
    unsafe {
        let mut dst = linebuf_f32_mut(aug);
        for _ in 0..repeat {
            *dst *= k;
            dst = dst.add(1);
        }
    }
}

// =========================================================================
// Irreversible horizontal analysis (forward)
// =========================================================================

/// Generic irreversible horizontal analysis.
pub(crate) fn gen_irv_horz_ana(
    atk: &ParamAtk,
    ldst: &mut LineBuf,
    hdst: &mut LineBuf,
    src: &LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            // Split src into ldst and hdst
            let mut dph = linebuf_f32_mut(hdst);
            let mut dpl = linebuf_f32_mut(ldst);
            let mut sp = linebuf_f32(src);
            let mut w = width;

            if !even {
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 1;
            }
            while w > 1 {
                *dpl = *sp;
                dpl = dpl.add(1);
                sp = sp.add(1);
                *dph = *sp;
                dph = dph.add(1);
                sp = sp.add(1);
                w -= 2;
            }
            if w > 0 {
                *dpl = *sp;
            }

            let mut hp = linebuf_f32_mut(hdst);
            let mut lp = linebuf_f32_mut(ldst);
            let mut l_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut h_width = (width + if even { 0 } else { 1 }) >> 1;
            let mut ev = even;
            let num_steps = atk.get_num_steps();

            for j in (1..=num_steps).rev() {
                let s = atk.get_step(j - 1);
                let a = s.irv().a;

                // symmetric extension
                *lp.sub(1) = *lp;
                *lp.add(l_width as usize) = *lp.add(l_width as usize - 1);

                let mut sp_l = lp.add(if ev { 1 } else { 0 }) as *const f32;
                let mut dp = hp;

                for _ in 0..h_width {
                    *dp += a * (*sp_l.sub(1) + *sp_l);
                    sp_l = sp_l.add(1);
                    dp = dp.add(1);
                }

                let t = lp;
                lp = hp;
                hp = t;
                ev = !ev;
                let tw = l_width;
                l_width = h_width;
                h_width = tw;
            }

            // K normalization
            {
                let k = atk.get_k();
                let k_inv = 1.0f32 / k;

                let mut dp = lp;
                for _ in 0..l_width {
                    *dp *= k_inv;
                    dp = dp.add(1);
                }

                dp = hp;
                for _ in 0..h_width {
                    *dp *= k;
                    dp = dp.add(1);
                }
            }
        }
    } else if even {
        unsafe {
            *linebuf_f32_mut(ldst) = *linebuf_f32(src);
        }
    } else {
        unsafe {
            *linebuf_f32_mut(hdst) = *linebuf_f32(src) * 2.0;
        }
    }
}

// =========================================================================
// Irreversible horizontal synthesis (inverse)
// =========================================================================

/// Generic irreversible horizontal synthesis.
pub(crate) fn gen_irv_horz_syn(
    atk: &ParamAtk,
    dst: &mut LineBuf,
    lsrc: &mut LineBuf,
    hsrc: &mut LineBuf,
    width: u32,
    even: bool,
) {
    if width > 1 {
        unsafe {
            let mut ev = even;
            let mut oth = linebuf_f32_mut(hsrc);
            let mut aug = linebuf_f32_mut(lsrc);
            let mut aug_width = (width + if even { 1 } else { 0 }) >> 1;
            let mut oth_width = (width + if even { 0 } else { 1 }) >> 1;

            // K denormalization
            {
                let k = atk.get_k();
                let k_inv = 1.0f32 / k;

                let mut dp = aug;
                for _ in 0..aug_width {
                    *dp *= k;
                    dp = dp.add(1);
                }

                dp = oth;
                for _ in 0..oth_width {
                    *dp *= k_inv;
                    dp = dp.add(1);
                }
            }

            let num_steps = atk.get_num_steps();
            for j in 0..num_steps {
                let s = atk.get_step(j);
                let a = s.irv().a;

                // symmetric extension
                *oth.sub(1) = *oth;
                *oth.add(oth_width as usize) = *oth.add(oth_width as usize - 1);

                let mut sp = oth.add(if ev { 0 } else { 1 }) as *const f32;
                let mut dp = aug;

                for _ in 0..aug_width {
                    *dp -= a * (*sp.sub(1) + *sp);
                    sp = sp.add(1);
                    dp = dp.add(1);
                }

                // swap buffers
                let t = aug;
                aug = oth;
                oth = t;
                ev = !ev;
                let tw = aug_width;
                aug_width = oth_width;
                oth_width = tw;
            }

            // Interleave lsrc and hsrc into dst
            let mut sph = linebuf_f32(hsrc);
            let mut spl = linebuf_f32(lsrc);
            let mut dp = linebuf_f32_mut(dst);
            let mut w = width;

            if !even {
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 1;
            }
            while w > 1 {
                *dp = *spl;
                dp = dp.add(1);
                spl = spl.add(1);
                *dp = *sph;
                dp = dp.add(1);
                sph = sph.add(1);
                w -= 2;
            }
            if w > 0 {
                *dp = *spl;
            }
        }
    } else if even {
        unsafe {
            *linebuf_f32_mut(dst) = *linebuf_f32(lsrc);
        }
    } else {
        unsafe {
            *linebuf_f32_mut(dst) = *linebuf_f32(hsrc) * 0.5;
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::{LFT_32BIT, LFT_INTEGER};

    /// Helper: create a LineBuf backed by i32 data (with 1 element of
    /// padding before the start for symmetric extension).
    fn make_i32_buf(data: &mut [i32]) -> LineBuf {
        LineBuf {
            size: data.len() - 1, // exclude padding element
            pre_size: 1,
            flags: LFT_32BIT | LFT_INTEGER,
            // Point past the padding element
            data: LineBufData::I32(data[1..].as_mut_ptr()),
        }
    }

    /// Helper: create a LineBuf backed by f32 data (with 1 element padding).
    fn make_f32_buf(data: &mut [f32]) -> LineBuf {
        LineBuf {
            size: data.len() - 1,
            pre_size: 1,
            flags: LFT_32BIT, // 32-bit float (no LFT_INTEGER)
            data: LineBufData::F32(data[1..].as_mut_ptr()),
        }
    }

    #[test]
    fn rev_vert_step_53_predict() {
        // 5/3 predict step: a=-1, b=1, e=1
        let step = super::super::LiftingStep::Reversible(super::super::RevLiftingStep {
            a: -1,
            b: 1,
            e: 1,
        });

        let mut sig_data = vec![0i32, 10, 20, 30, 40];
        let mut other_data = vec![0i32, 12, 22, 32, 42];
        let mut aug_data = vec![0i32, 100, 200, 300, 400];

        let sig = make_i32_buf(&mut sig_data);
        let other = make_i32_buf(&mut other_data);
        let mut aug = make_i32_buf(&mut aug_data);

        gen_rev_vert_step(&step, &sig, &other, &mut aug, 4, false);

        // analysis: dst -= (src1 + src2) >> 1
        assert_eq!(aug_data[1], 100 - (10 + 12) / 2); // 100 - 11 = 89
        assert_eq!(aug_data[2], 200 - (20 + 22) / 2); // 200 - 21 = 179
    }

    #[test]
    fn irv_vert_times_k() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut buf = make_f32_buf(&mut data);

        gen_irv_vert_times_k(1.230_174_1, &mut buf, 4);

        assert!((data[1] - 1.230_174_1).abs() < 1e-6);
        assert!((data[2] - 2.460_348_2).abs() < 1e-5);
    }

    // -----------------------------------------------------------------
    // Helper: run a reversible horizontal roundtrip for given width/even
    // and verify exact reconstruction.
    // -----------------------------------------------------------------
    fn rev_horz_roundtrip(input: &[i32], width: u32, even: bool) {
        use super::super::ParamAtk;

        let mut atk = ParamAtk::default();
        atk.init_rev53();

        let l_width = (width + if even { 1 } else { 0 }) / 2;
        let h_width = (width + if even { 0 } else { 1 }) / 2;

        // Source buffer (1 pad before + width + 1 pad after)
        let mut src_data = vec![0i32; width as usize + 2];
        for (i, &v) in input.iter().enumerate() {
            src_data[i + 1] = v;
        }
        let src = make_i32_buf(&mut src_data);

        // Low / high output buffers
        let mut ldst_data = vec![0i32; l_width as usize + 2];
        let mut hdst_data = vec![0i32; h_width as usize + 2];
        let mut ldst = make_i32_buf(&mut ldst_data);
        let mut hdst = make_i32_buf(&mut hdst_data);

        // Forward DWT (analysis)
        gen_rev_horz_ana(&atk, &mut ldst, &mut hdst, &src, width, even);

        // Reconstruction buffer
        let mut dst_data = vec![0i32; width as usize + 2];
        let mut dst = make_i32_buf(&mut dst_data);

        // Inverse DWT (synthesis)
        gen_rev_horz_syn(&atk, &mut dst, &mut ldst, &mut hdst, width, even);

        for i in 0..width as usize {
            assert_eq!(
                dst_data[i + 1],
                src_data[i + 1],
                "mismatch at index {i} (width={width}, even={even})"
            );
        }
    }

    // -----------------------------------------------------------------
    // Helper: run an irreversible horizontal roundtrip and verify
    // reconstruction within epsilon.
    // -----------------------------------------------------------------
    fn irv_horz_roundtrip(input: &[f32], width: u32, even: bool, eps: f32) {
        use super::super::ParamAtk;

        let mut atk = ParamAtk::default();
        atk.init_irv97();

        let l_width = (width + if even { 1 } else { 0 }) / 2;
        let h_width = (width + if even { 0 } else { 1 }) / 2;

        let mut src_data = vec![0.0f32; width as usize + 2];
        for (i, &v) in input.iter().enumerate() {
            src_data[i + 1] = v;
        }
        let src = make_f32_buf(&mut src_data);

        let mut ldst_data = vec![0.0f32; l_width as usize + 2];
        let mut hdst_data = vec![0.0f32; h_width as usize + 2];
        let mut ldst = make_f32_buf(&mut ldst_data);
        let mut hdst = make_f32_buf(&mut hdst_data);

        gen_irv_horz_ana(&atk, &mut ldst, &mut hdst, &src, width, even);

        let mut dst_data = vec![0.0f32; width as usize + 2];
        let mut dst = make_f32_buf(&mut dst_data);

        gen_irv_horz_syn(&atk, &mut dst, &mut ldst, &mut hdst, width, even);

        for i in 0..width as usize {
            let got = dst_data[i + 1];
            let expected = src_data[i + 1];
            assert!(
                (got - expected).abs() < eps,
                "mismatch at index {i}: got {got}, expected {expected} (width={width}, even={even})"
            );
        }
    }

    // =================================================================
    // Reversible horizontal roundtrip tests
    // =================================================================

    #[test]
    fn rev_horz_ana_syn_roundtrip_even_4() {
        rev_horz_roundtrip(&[10, 20, 30, 40], 4, true);
    }

    #[test]
    fn rev_horz_ana_syn_roundtrip_even_8() {
        rev_horz_roundtrip(&[5, 15, 25, 35, 45, 55, 65, 75], 8, true);
    }

    #[test]
    fn rev_horz_ana_syn_roundtrip_odd_length() {
        rev_horz_roundtrip(&[3, 7, 11, 15, 19, 23, 27], 7, true);
    }

    #[test]
    fn rev_horz_ana_syn_roundtrip_width_2() {
        rev_horz_roundtrip(&[100, 200], 2, true);
    }

    #[test]
    fn rev_horz_ana_syn_roundtrip_width_1_even() {
        rev_horz_roundtrip(&[42], 1, true);
    }

    #[test]
    fn rev_horz_ana_syn_roundtrip_width_1_odd() {
        rev_horz_roundtrip(&[42], 1, false);
    }

    // =================================================================
    // Reversible vertical step roundtrip
    // =================================================================

    #[test]
    fn rev_vert_step_roundtrip() {
        // Use the 5/3 predict step: a=-1, b=1, e=1
        let step = super::super::LiftingStep::Reversible(super::super::RevLiftingStep {
            a: -1,
            b: 1,
            e: 1,
        });

        let mut sig_data = vec![0i32, 10, 20, 30, 40];
        let mut other_data = vec![0i32, 12, 22, 32, 42];
        let mut aug_data = vec![0i32, 100, 200, 300, 400];
        let original_aug: Vec<i32> = aug_data.clone();

        let sig = make_i32_buf(&mut sig_data);
        let other = make_i32_buf(&mut other_data);
        let mut aug = make_i32_buf(&mut aug_data);

        // Analysis (forward)
        gen_rev_vert_step(&step, &sig, &other, &mut aug, 4, false);

        // Verify values actually changed
        assert_ne!(aug_data[1..5], original_aug[1..5]);

        // Synthesis (inverse) — same sig/other, synthesis=true
        gen_rev_vert_step(&step, &sig, &other, &mut aug, 4, true);

        for i in 1..5 {
            assert_eq!(
                aug_data[i], original_aug[i],
                "vert step roundtrip mismatch at {i}"
            );
        }
    }

    // =================================================================
    // Irreversible horizontal roundtrip tests
    // =================================================================

    #[test]
    fn irv_horz_ana_syn_roundtrip_4() {
        irv_horz_roundtrip(&[1.0, 2.0, 3.0, 4.0], 4, true, 1e-4);
    }

    #[test]
    fn irv_horz_ana_syn_roundtrip_8() {
        irv_horz_roundtrip(&[1.0, -2.5, 3.3, 0.7, -1.1, 4.4, 2.2, -0.8], 8, true, 1e-4);
    }

    // =================================================================
    // Irreversible vertical step roundtrip
    // =================================================================

    #[test]
    fn irv_vert_step_roundtrip() {
        let step = super::super::LiftingStep::Irreversible(super::super::IrvLiftingStep {
            a: -1.586_134_3, // 9/7 alpha
        });

        let mut sig_data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut other_data = vec![0.0f32, 1.5, 2.5, 3.5, 4.5];
        let mut aug_data = vec![0.0f32, 10.0, 20.0, 30.0, 40.0];
        let original_aug: Vec<f32> = aug_data.clone();

        let sig = make_f32_buf(&mut sig_data);
        let other = make_f32_buf(&mut other_data);
        let mut aug = make_f32_buf(&mut aug_data);

        // Analysis
        gen_irv_vert_step(&step, &sig, &other, &mut aug, 4, false);

        // Values should have changed
        assert!((aug_data[1] - original_aug[1]).abs() > 1e-6);

        // Synthesis
        gen_irv_vert_step(&step, &sig, &other, &mut aug, 4, true);

        for i in 1..5 {
            assert!(
                (aug_data[i] - original_aug[i]).abs() < 1e-4,
                "irv vert step roundtrip mismatch at {i}: got {}, expected {}",
                aug_data[i],
                original_aug[i]
            );
        }
    }

    // =================================================================
    // Reversible horizontal roundtrip — various lengths
    // =================================================================

    #[test]
    fn rev_horz_roundtrip_various_lengths() {
        for &width in &[2u32, 3, 4, 8, 16, 63, 64, 128] {
            // Generate a deterministic pattern: alternating positive/negative
            let input: Vec<i32> = (0..width as i32)
                .map(|i| if i % 2 == 0 { i * 3 + 1 } else { -(i * 2 + 5) })
                .collect();
            rev_horz_roundtrip(&input, width, true);
        }
    }

    // =================================================================
    // Reversible horizontal roundtrip — large buffer
    // =================================================================

    #[test]
    fn rev_horz_ana_syn_roundtrip_large() {
        let width = 128u32;
        // Pseudo-random-ish pattern using a simple LCG
        let mut val: i32 = 7;
        let input: Vec<i32> = (0..width)
            .map(|_| {
                val = val.wrapping_mul(1103515245).wrapping_add(12345);
                (val >> 16) & 0xFF
            })
            .collect();
        rev_horz_roundtrip(&input, width, true);
    }
}
