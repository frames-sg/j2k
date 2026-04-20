//! x86 AVX2-accelerated block coding helper routines.
//!
//! All code gated behind `#[cfg(target_arch = "x86_64")]`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// AVX2-accelerated OR-reduce of a u32 buffer.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_or_reduce(data: &[u32], count: u32) -> u32 {
    unsafe { avx2_or_reduce_inner(data, count) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_or_reduce_inner(data: &[u32], count: u32) -> u32 {
    let mut acc = _mm256_setzero_si256();
    let simd_count = count / 8;
    let remainder = count % 8;
    let mut ptr = data.as_ptr();

    for _ in 0..simd_count {
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        acc = _mm256_or_si256(acc, v);
        ptr = ptr.add(8);
    }

    // Reduce 256-bit → 128-bit → scalar
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let combined = _mm_or_si128(hi, lo);
    let mut result = _mm_extract_epi32(combined, 0) as u32
        | _mm_extract_epi32(combined, 1) as u32
        | _mm_extract_epi32(combined, 2) as u32
        | _mm_extract_epi32(combined, 3) as u32;

    for i in 0..remainder as usize {
        result |= *ptr.add(i);
    }

    result
}

/// SSE2-accelerated OR-reduce of a u32 buffer.
#[cfg(target_arch = "x86_64")]
pub(crate) fn sse2_or_reduce(data: &[u32], count: u32) -> u32 {
    unsafe { sse2_or_reduce_inner(data, count) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sse2_or_reduce_inner(data: &[u32], count: u32) -> u32 {
    let mut acc = _mm_setzero_si128();
    let simd_count = count / 4;
    let remainder = count % 4;
    let mut ptr = data.as_ptr();

    for _ in 0..simd_count {
        let v = _mm_loadu_si128(ptr as *const __m128i);
        acc = _mm_or_si128(acc, v);
        ptr = ptr.add(4);
    }

    let mut result = _mm_extract_epi32(acc, 0) as u32
        | _mm_extract_epi32(acc, 1) as u32
        | _mm_extract_epi32(acc, 2) as u32
        | _mm_extract_epi32(acc, 3) as u32;

    for i in 0..remainder as usize {
        result |= *ptr.add(i);
    }

    result
}
