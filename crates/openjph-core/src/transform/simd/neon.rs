//! NEON-accelerated wavelet transform routines for AArch64.
//!
//! Provides SIMD-accelerated vertical lifting steps for the 5/3 reversible
//! and 9/7 irreversible discrete wavelet transforms using ARM NEON intrinsics.
//! Processes 4 elements at a time (128-bit vectors).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::mem::{LineBuf, LineBufData, LFT_32BIT};
use crate::transform::LiftingStep;

// =========================================================================
// Pointer helpers (same as wavelet.rs)
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
// NEON reversible vertical lifting step — 32-bit
// =========================================================================

/// NEON-accelerated 32-bit reversible vertical step.
///
/// Processes 4 i32 values at a time using `int32x4_t` vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_rev_vert_step32(
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
    let e = rev.e as i32;

    let mut dst = lb_i32_mut(aug);
    let mut src1 = lb_i32(sig);
    let mut src2 = lb_i32(other);

    let simd_width = 4u32;
    let simd_count = repeat / simd_width;
    let remainder = repeat % simd_width;

    let neg_shift = vdupq_n_s32(-e);

    if a == 1 {
        let vb = vdupq_n_s32(b);
        if synthesis {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(vaddq_s32(vb, s1), s2);
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vsubq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst -= (b + *src1 + *src2) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(vaddq_s32(vb, s1), s2);
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vaddq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst += (b + *src1 + *src2) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
    } else if a == -1 && b == 1 && e == 1 {
        // 5/3 predict step: specialized fast path
        let _vone = vdupq_n_s32(1);
        if synthesis {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(s1, s2);
                let shifted = vshrq_n_s32::<1>(sum);
                vst1q_s32(dst, vaddq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst += (*src1 + *src2) >> 1;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(s1, s2);
                let shifted = vshrq_n_s32::<1>(sum);
                vst1q_s32(dst, vsubq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst -= (*src1 + *src2) >> 1;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
    } else if a == -1 {
        let vb = vdupq_n_s32(b);
        if synthesis {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vsubq_s32(vb, vaddq_s32(s1, s2));
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vsubq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst -= (b - (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vsubq_s32(vb, vaddq_s32(s1, s2));
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vaddq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst += (b - (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
    } else {
        // General case
        let va = vdupq_n_s32(a);
        let vb = vdupq_n_s32(b);
        if synthesis {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(vb, vmulq_s32(va, vaddq_s32(s1, s2)));
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vsubq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst -= (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..simd_count {
                let d = vld1q_s32(dst);
                let s1 = vld1q_s32(src1);
                let s2 = vld1q_s32(src2);
                let sum = vaddq_s32(vb, vmulq_s32(va, vaddq_s32(s1, s2)));
                let shifted = vshlq_s32(sum, neg_shift);
                vst1q_s32(dst, vaddq_s32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
            for _ in 0..remainder {
                *dst += (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
    }
}

// =========================================================================
// Public NEON reversible vertical step (dispatches 32/64)
// =========================================================================

/// NEON-accelerated reversible vertical lifting step.
///
/// Uses NEON intrinsics for the 32-bit path (4 i32 per vector).
/// Falls back to scalar for 64-bit.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_rev_vert_step(
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
        // SAFETY: We've checked that neon is available (aarch64 always has it)
        // and the buffers contain i32 data.
        unsafe {
            neon_rev_vert_step32(s, sig, other, aug, repeat, synthesis);
        }
    } else {
        // 64-bit path: fall back to scalar
        super::super::wavelet::gen_rev_vert_step64(s, sig, other, aug, repeat, synthesis);
    }
}

// =========================================================================
// NEON irreversible vertical lifting step (f32)
// =========================================================================

/// NEON-accelerated irreversible (9/7) vertical lifting step.
///
/// Computes `aug[i] += a * (sig[i] + other[i])` using `float32x4_t`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_irv_vert_step(
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
        let mut dst = lb_f32_mut(aug);
        let mut src1 = lb_f32(sig);
        let mut src2 = lb_f32(other);

        let va = vdupq_n_f32(a);
        let simd_count = repeat / 4;
        let remainder = repeat % 4;

        for _ in 0..simd_count {
            let d = vld1q_f32(dst);
            let s1 = vld1q_f32(src1);
            let s2 = vld1q_f32(src2);
            // d += a * (s1 + s2) using fused multiply-add
            let sum = vaddq_f32(s1, s2);
            let result = vfmaq_f32(d, va, sum);
            vst1q_f32(dst, result);
            dst = dst.add(4);
            src1 = src1.add(4);
            src2 = src2.add(4);
        }

        // Scalar remainder
        for _ in 0..remainder {
            *dst += a * (*src1 + *src2);
            dst = dst.add(1);
            src1 = src1.add(1);
            src2 = src2.add(1);
        }
    }
}

// =========================================================================
// NEON irreversible vertical times K
// =========================================================================

/// NEON-accelerated multiply by normalization constant K.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_irv_vert_times_k(k: f32, aug: &mut LineBuf, repeat: u32) {
    unsafe {
        let mut dst = lb_f32_mut(aug);
        let vk = vdupq_n_f32(k);
        let simd_count = repeat / 4;
        let remainder = repeat % 4;

        for _ in 0..simd_count {
            let d = vld1q_f32(dst);
            vst1q_f32(dst, vmulq_f32(d, vk));
            dst = dst.add(4);
        }

        for _ in 0..remainder {
            *dst *= k;
            dst = dst.add(1);
        }
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
    use crate::transform::{IrvLiftingStep, LiftingStep, RevLiftingStep};

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
            flags: LFT_32BIT,
            data: LineBufData::F32(data.as_mut_ptr()),
        }
    }

    /// Verify NEON rev_vert_step matches scalar for the 5/3 predict step.
    #[test]
    fn neon_rev_vert_step_predict_matches_scalar() {
        let step = LiftingStep::Reversible(RevLiftingStep { a: -1, b: 1, e: 1 });

        for width in [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100] {
            let mut sig_data: Vec<i32> = (0..width).map(|i| i * 3 + 1).collect();
            let mut other_data: Vec<i32> = (0..width).map(|i| i * 2 + 5).collect();

            // Scalar reference
            let mut aug_scalar: Vec<i32> = (0..width).map(|i| i * 7).collect();
            let mut aug_neon: Vec<i32> = aug_scalar.clone();

            let sig = make_i32_linebuf(&mut sig_data);
            let other = make_i32_linebuf(&mut other_data);

            let mut aug_s = make_i32_linebuf(&mut aug_scalar);
            crate::transform::wavelet::gen_rev_vert_step32(
                &step,
                &sig,
                &other,
                &mut aug_s,
                width as u32,
                false,
            );

            let mut aug_n = make_i32_linebuf(&mut aug_neon);
            neon_rev_vert_step(&step, &sig, &other, &mut aug_n, width as u32, false);

            assert_eq!(aug_scalar, aug_neon, "mismatch at width={width} (analysis)");

            // Also test synthesis
            let mut aug_scalar_syn: Vec<i32> = (0..width).map(|i| i * 7).collect();
            let mut aug_neon_syn: Vec<i32> = aug_scalar_syn.clone();

            let mut aug_ss = make_i32_linebuf(&mut aug_scalar_syn);
            crate::transform::wavelet::gen_rev_vert_step32(
                &step,
                &sig,
                &other,
                &mut aug_ss,
                width as u32,
                true,
            );

            let mut aug_ns = make_i32_linebuf(&mut aug_neon_syn);
            neon_rev_vert_step(&step, &sig, &other, &mut aug_ns, width as u32, true);

            assert_eq!(
                aug_scalar_syn, aug_neon_syn,
                "synthesis mismatch at width={width}"
            );
        }
    }

    /// Verify NEON rev_vert_step matches scalar for the 5/3 update step.
    #[test]
    fn neon_rev_vert_step_update_matches_scalar() {
        let step = LiftingStep::Reversible(RevLiftingStep { a: 1, b: 2, e: 2 });

        for width in [1, 4, 7, 16, 33, 64] {
            let mut sig_data: Vec<i32> = (0..width).map(|i| i * 5 - 100).collect();
            let mut other_data: Vec<i32> = (0..width).map(|i| -i * 3 + 50).collect();
            let mut aug_scalar: Vec<i32> = (0..width).map(|i| i * 11 - 200).collect();
            let mut aug_neon: Vec<i32> = aug_scalar.clone();

            let sig = make_i32_linebuf(&mut sig_data);
            let other = make_i32_linebuf(&mut other_data);

            let mut aug_s = make_i32_linebuf(&mut aug_scalar);
            crate::transform::wavelet::gen_rev_vert_step32(
                &step,
                &sig,
                &other,
                &mut aug_s,
                width as u32,
                false,
            );

            let mut aug_n = make_i32_linebuf(&mut aug_neon);
            neon_rev_vert_step(&step, &sig, &other, &mut aug_n, width as u32, false);

            assert_eq!(aug_scalar, aug_neon, "update mismatch at width={width}");
        }
    }

    /// Verify NEON irv_vert_step matches scalar for 9/7 lifting.
    #[test]
    fn neon_irv_vert_step_matches_scalar() {
        let coefficients = [-1.586_134_3f32, -0.052_980_118, 0.882_911_1, 0.443_506_85];

        for &a_val in &coefficients {
            let step = LiftingStep::Irreversible(IrvLiftingStep { a: a_val });

            for width in [1, 3, 4, 7, 8, 16, 33, 64] {
                let mut sig_data: Vec<f32> = (0..width).map(|i| i as f32 * 0.1 - 3.0).collect();
                let mut other_data: Vec<f32> =
                    (0..width).map(|i| -(i as f32) * 0.2 + 1.5).collect();
                let mut aug_scalar: Vec<f32> = (0..width).map(|i| i as f32 * 0.3).collect();
                let mut aug_neon: Vec<f32> = aug_scalar.clone();

                let sig = make_f32_linebuf(&mut sig_data);
                let other = make_f32_linebuf(&mut other_data);

                // Scalar
                let mut aug_s = make_f32_linebuf(&mut aug_scalar);
                crate::transform::wavelet::gen_irv_vert_step(
                    &step,
                    &sig,
                    &other,
                    &mut aug_s,
                    width as u32,
                    false,
                );

                // NEON
                let mut aug_n = make_f32_linebuf(&mut aug_neon);
                neon_irv_vert_step(&step, &sig, &other, &mut aug_n, width as u32, false);

                for i in 0..width {
                    assert!(
                        (aug_scalar[i] - aug_neon[i]).abs() < 1e-5,
                        "irv step a={a_val} width={width} idx={i}: scalar={} neon={}",
                        aug_scalar[i],
                        aug_neon[i],
                    );
                }
            }
        }
    }

    /// Verify NEON irv_vert_times_k matches scalar.
    #[test]
    fn neon_irv_vert_times_k_matches_scalar() {
        let k = 1.230_174_1f32;

        for width in [1, 3, 4, 7, 8, 16, 33, 64] {
            let mut scalar_data: Vec<f32> = (0..width).map(|i| i as f32 * 0.5 - 10.0).collect();
            let mut neon_data: Vec<f32> = scalar_data.clone();

            let mut scalar_buf = make_f32_linebuf(&mut scalar_data);
            crate::transform::wavelet::gen_irv_vert_times_k(k, &mut scalar_buf, width as u32);

            let mut neon_buf = make_f32_linebuf(&mut neon_data);
            neon_irv_vert_times_k(k, &mut neon_buf, width as u32);

            for i in 0..width {
                assert!(
                    (scalar_data[i] - neon_data[i]).abs() < 1e-5,
                    "times_k width={width} idx={i}: scalar={} neon={}",
                    scalar_data[i],
                    neon_data[i],
                );
            }
        }
    }
}
