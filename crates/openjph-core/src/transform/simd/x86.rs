//! x86 SSE2/AVX2-accelerated wavelet transform routines.
//!
//! All code is gated behind `#[cfg(target_arch = "x86_64")]` and uses
//! `#[target_feature(enable = "...")]` attributes for safe dispatch.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::mem::{LineBuf, LineBufData, LFT_32BIT};
use crate::transform::LiftingStep;

// `_mm_srai_epi32` and `_mm256_srai_epi32` have `#[rustc_legacy_const_generics]`
// on the shift argument, so it must be a compile-time constant.  This macro
// dispatches a runtime `$e` (Eatk ≤ 15 covers all JPEG 2000 lifting steps)
// to the correct constant arm so the compiler is satisfied.
#[cfg(target_arch = "x86_64")]
macro_rules! srai_dyn {
    ($fn:ident, $val:expr, $e:expr) => {
        match $e {
            0 => $fn($val, 0),
            1 => $fn($val, 1),
            2 => $fn($val, 2),
            3 => $fn($val, 3),
            4 => $fn($val, 4),
            5 => $fn($val, 5),
            6 => $fn($val, 6),
            7 => $fn($val, 7),
            8 => $fn($val, 8),
            9 => $fn($val, 9),
            10 => $fn($val, 10),
            11 => $fn($val, 11),
            12 => $fn($val, 12),
            13 => $fn($val, 13),
            14 => $fn($val, 14),
            15 => $fn($val, 15),
            _ => unreachable!("Eatk > 15 is not supported in the SIMD path"),
        }
    };
}

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
// SSE2 reversible vertical lifting step — 32-bit
// =========================================================================

/// SSE2-accelerated reversible vertical step (32-bit, 4-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_rev_vert_step32(
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

    let simd_count = repeat / 4;
    let remainder = repeat % 4;

    if a == -1 && b == 1 && e == 1 {
        // 5/3 predict: aug ∓ (src1 + src2) >> 1
        if synthesis {
            for _ in 0..simd_count {
                let d = _mm_loadu_si128(dst as *const __m128i);
                let s1 = _mm_loadu_si128(src1 as *const __m128i);
                let s2 = _mm_loadu_si128(src2 as *const __m128i);
                let sum = _mm_add_epi32(s1, s2);
                let shifted = _mm_srai_epi32(sum, 1);
                _mm_storeu_si128(dst as *mut __m128i, _mm_add_epi32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
        } else {
            for _ in 0..simd_count {
                let d = _mm_loadu_si128(dst as *const __m128i);
                let s1 = _mm_loadu_si128(src1 as *const __m128i);
                let s2 = _mm_loadu_si128(src2 as *const __m128i);
                let sum = _mm_add_epi32(s1, s2);
                let shifted = _mm_srai_epi32(sum, 1);
                _mm_storeu_si128(dst as *mut __m128i, _mm_sub_epi32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
        }
    } else if a == 1 {
        let vb = _mm_set1_epi32(b);
        if synthesis {
            for _ in 0..simd_count {
                let d = _mm_loadu_si128(dst as *const __m128i);
                let s1 = _mm_loadu_si128(src1 as *const __m128i);
                let s2 = _mm_loadu_si128(src2 as *const __m128i);
                let sum = _mm_add_epi32(_mm_add_epi32(vb, s1), s2);
                let shifted = srai_dyn!(_mm_srai_epi32, sum, e);
                _mm_storeu_si128(dst as *mut __m128i, _mm_sub_epi32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
        } else {
            for _ in 0..simd_count {
                let d = _mm_loadu_si128(dst as *const __m128i);
                let s1 = _mm_loadu_si128(src1 as *const __m128i);
                let s2 = _mm_loadu_si128(src2 as *const __m128i);
                let sum = _mm_add_epi32(_mm_add_epi32(vb, s1), s2);
                let shifted = srai_dyn!(_mm_srai_epi32, sum, e);
                _mm_storeu_si128(dst as *mut __m128i, _mm_add_epi32(d, shifted));
                dst = dst.add(4);
                src1 = src1.add(4);
                src2 = src2.add(4);
            }
        }
    } else {
        // General case: fall back to scalar for simplicity
        if synthesis {
            for _ in 0..(simd_count * 4 + remainder) {
                *dst -= (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..(simd_count * 4 + remainder) {
                *dst += (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
        return;
    }

    // Scalar remainder for the SIMD paths
    if synthesis {
        for _ in 0..remainder {
            if a == -1 && b == 1 && e == 1 {
                *dst += (*src1 + *src2) >> 1;
            } else {
                *dst -= (b + *src1 + *src2) >> e;
            }
            dst = dst.add(1);
            src1 = src1.add(1);
            src2 = src2.add(1);
        }
    } else {
        for _ in 0..remainder {
            if a == -1 && b == 1 && e == 1 {
                *dst -= (*src1 + *src2) >> 1;
            } else {
                *dst += (b + *src1 + *src2) >> e;
            }
            dst = dst.add(1);
            src1 = src1.add(1);
            src2 = src2.add(1);
        }
    }
}

// =========================================================================
// AVX2 reversible vertical lifting step — 32-bit (8-wide)
// =========================================================================

/// AVX2-accelerated reversible vertical step (32-bit, 8-wide).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_rev_vert_step32(
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

    let simd_count = repeat / 8;
    let remainder = repeat % 8;

    if a == -1 && b == 1 && e == 1 {
        if synthesis {
            for _ in 0..simd_count {
                let d = _mm256_loadu_si256(dst as *const __m256i);
                let s1 = _mm256_loadu_si256(src1 as *const __m256i);
                let s2 = _mm256_loadu_si256(src2 as *const __m256i);
                let sum = _mm256_add_epi32(s1, s2);
                let shifted = _mm256_srai_epi32(sum, 1);
                _mm256_storeu_si256(dst as *mut __m256i, _mm256_add_epi32(d, shifted));
                dst = dst.add(8);
                src1 = src1.add(8);
                src2 = src2.add(8);
            }
        } else {
            for _ in 0..simd_count {
                let d = _mm256_loadu_si256(dst as *const __m256i);
                let s1 = _mm256_loadu_si256(src1 as *const __m256i);
                let s2 = _mm256_loadu_si256(src2 as *const __m256i);
                let sum = _mm256_add_epi32(s1, s2);
                let shifted = _mm256_srai_epi32(sum, 1);
                _mm256_storeu_si256(dst as *mut __m256i, _mm256_sub_epi32(d, shifted));
                dst = dst.add(8);
                src1 = src1.add(8);
                src2 = src2.add(8);
            }
        }
    } else if a == 1 {
        let vb = _mm256_set1_epi32(b);
        if synthesis {
            for _ in 0..simd_count {
                let d = _mm256_loadu_si256(dst as *const __m256i);
                let s1 = _mm256_loadu_si256(src1 as *const __m256i);
                let s2 = _mm256_loadu_si256(src2 as *const __m256i);
                let sum = _mm256_add_epi32(_mm256_add_epi32(vb, s1), s2);
                let shifted = srai_dyn!(_mm256_srai_epi32, sum, e);
                _mm256_storeu_si256(dst as *mut __m256i, _mm256_sub_epi32(d, shifted));
                dst = dst.add(8);
                src1 = src1.add(8);
                src2 = src2.add(8);
            }
        } else {
            for _ in 0..simd_count {
                let d = _mm256_loadu_si256(dst as *const __m256i);
                let s1 = _mm256_loadu_si256(src1 as *const __m256i);
                let s2 = _mm256_loadu_si256(src2 as *const __m256i);
                let sum = _mm256_add_epi32(_mm256_add_epi32(vb, s1), s2);
                let shifted = srai_dyn!(_mm256_srai_epi32, sum, e);
                _mm256_storeu_si256(dst as *mut __m256i, _mm256_add_epi32(d, shifted));
                dst = dst.add(8);
                src1 = src1.add(8);
                src2 = src2.add(8);
            }
        }
    } else {
        // General: scalar fallback
        let total = simd_count * 8 + remainder;
        if synthesis {
            for _ in 0..total {
                *dst -= (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        } else {
            for _ in 0..total {
                *dst += (b + a * (*src1 + *src2)) >> e;
                dst = dst.add(1);
                src1 = src1.add(1);
                src2 = src2.add(1);
            }
        }
        return;
    }

    // Scalar remainder
    for _ in 0..remainder {
        if a == -1 && b == 1 && e == 1 {
            if synthesis {
                *dst += (*src1 + *src2) >> 1;
            } else {
                *dst -= (*src1 + *src2) >> 1;
            }
        } else {
            if synthesis {
                *dst -= (b + *src1 + *src2) >> e;
            } else {
                *dst += (b + *src1 + *src2) >> e;
            }
        }
        dst = dst.add(1);
        src1 = src1.add(1);
        src2 = src2.add(1);
    }
}

// =========================================================================
// Public dispatch: SSE2 or AVX2 reversible vertical step
// =========================================================================

/// SSE2-accelerated reversible vertical step (public entry point).
#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_rev_vert_step(
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
        unsafe {
            sse2_rev_vert_step32(s, sig, other, aug, repeat, synthesis);
        }
    } else {
        super::super::wavelet::gen_rev_vert_step64(s, sig, other, aug, repeat, synthesis);
    }
}

/// AVX2-accelerated reversible vertical step (public entry point).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_rev_vert_step(
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
        unsafe {
            avx2_rev_vert_step32(s, sig, other, aug, repeat, synthesis);
        }
    } else {
        super::super::wavelet::gen_rev_vert_step64(s, sig, other, aug, repeat, synthesis);
    }
}

// =========================================================================
// SSE2/AVX2 irreversible vertical step (f32)
// =========================================================================

/// SSE2-accelerated irreversible vertical step.
#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_irv_vert_step(
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
        sse2_irv_vert_step_inner(a, sig, other, aug, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_irv_vert_step_inner(
    a: f32,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
) {
    let mut dst = lb_f32_mut(aug);
    let mut src1 = lb_f32(sig);
    let mut src2 = lb_f32(other);
    let va = _mm_set1_ps(a);
    let simd_count = repeat / 4;
    let remainder = repeat % 4;

    for _ in 0..simd_count {
        let d = _mm_loadu_ps(dst);
        let s1 = _mm_loadu_ps(src1);
        let s2 = _mm_loadu_ps(src2);
        let sum = _mm_add_ps(s1, s2);
        let result = _mm_add_ps(d, _mm_mul_ps(va, sum));
        _mm_storeu_ps(dst, result);
        dst = dst.add(4);
        src1 = src1.add(4);
        src2 = src2.add(4);
    }

    for _ in 0..remainder {
        *dst += a * (*src1 + *src2);
        dst = dst.add(1);
        src1 = src1.add(1);
        src2 = src2.add(1);
    }
}

/// AVX2-accelerated irreversible vertical step.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_irv_vert_step(
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
        avx2_irv_vert_step_inner(a, sig, other, aug, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn avx2_irv_vert_step_inner(
    a: f32,
    sig: &LineBuf,
    other: &LineBuf,
    aug: &mut LineBuf,
    repeat: u32,
) {
    let mut dst = lb_f32_mut(aug);
    let mut src1 = lb_f32(sig);
    let mut src2 = lb_f32(other);
    let va = _mm256_set1_ps(a);
    let simd_count = repeat / 8;
    let remainder = repeat % 8;

    for _ in 0..simd_count {
        let d = _mm256_loadu_ps(dst);
        let s1 = _mm256_loadu_ps(src1);
        let s2 = _mm256_loadu_ps(src2);
        let sum = _mm256_add_ps(s1, s2);
        let result = _mm256_fmadd_ps(va, sum, d);
        _mm256_storeu_ps(dst, result);
        dst = dst.add(8);
        src1 = src1.add(8);
        src2 = src2.add(8);
    }

    for _ in 0..remainder {
        *dst += a * (*src1 + *src2);
        dst = dst.add(1);
        src1 = src1.add(1);
        src2 = src2.add(1);
    }
}

// =========================================================================
// SSE2/AVX2 irreversible times K
// =========================================================================

/// SSE2-accelerated multiply by K.
#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_irv_vert_times_k(k: f32, aug: &mut LineBuf, repeat: u32) {
    unsafe {
        sse2_irv_vert_times_k_inner(k, aug, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_irv_vert_times_k_inner(k: f32, aug: &mut LineBuf, repeat: u32) {
    let mut dst = lb_f32_mut(aug);
    let vk = _mm_set1_ps(k);
    let simd_count = repeat / 4;
    let remainder = repeat % 4;

    for _ in 0..simd_count {
        let d = _mm_loadu_ps(dst);
        _mm_storeu_ps(dst, _mm_mul_ps(d, vk));
        dst = dst.add(4);
    }
    for _ in 0..remainder {
        *dst *= k;
        dst = dst.add(1);
    }
}

/// AVX2-accelerated multiply by K.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_irv_vert_times_k(k: f32, aug: &mut LineBuf, repeat: u32) {
    unsafe {
        avx2_irv_vert_times_k_inner(k, aug, repeat);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_irv_vert_times_k_inner(k: f32, aug: &mut LineBuf, repeat: u32) {
    let mut dst = lb_f32_mut(aug);
    let vk = _mm256_set1_ps(k);
    let simd_count = repeat / 8;
    let remainder = repeat % 8;

    for _ in 0..simd_count {
        let d = _mm256_loadu_ps(dst);
        _mm256_storeu_ps(dst, _mm256_mul_ps(d, vk));
        dst = dst.add(8);
    }
    for _ in 0..remainder {
        *dst *= k;
        dst = dst.add(1);
    }
}
