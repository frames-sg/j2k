//! CPU architecture utilities — port of `ojph_arch.h/cpp`.
//!
//! Provides CPU feature detection, alignment constants, and bit-manipulation
//! helpers that wrap Rust intrinsics for clarity and C++ API compatibility.

// ---------------------------------------------------------------------------
// Alignment constants
// ---------------------------------------------------------------------------

/// Required byte alignment for SIMD buffers (AVX-512 = 64 bytes).
pub const BYTE_ALIGNMENT: u32 = 64;

/// log₂(BYTE_ALIGNMENT).
pub const LOG_BYTE_ALIGNMENT: u32 = BYTE_ALIGNMENT.trailing_zeros();

/// Required alignment for heap-allocated objects.
pub const OBJECT_ALIGNMENT: u32 = 8;

// ---------------------------------------------------------------------------
// CPU extension levels — x86-64
// ---------------------------------------------------------------------------

/// Supported x86-64 SIMD extension levels, ordered by capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum CpuExtLevel {
    /// No SIMD.
    Generic = 0,
    /// MMX (legacy, rarely targeted).
    Mmx = 1,
    /// SSE.
    Sse = 2,
    /// SSE2.
    Sse2 = 3,
    /// SSE3.
    Sse3 = 4,
    /// SSSE3.
    Ssse3 = 5,
    /// SSE4.1.
    Sse41 = 6,
    /// SSE4.2.
    Sse42 = 7,
    /// AVX.
    Avx = 8,
    /// AVX2.
    Avx2 = 9,
    /// AVX2 + FMA.
    Avx2Fma = 10,
    /// AVX-512 (F+BW+CD+DQ+VL at minimum).
    Avx512 = 11,
}

// ---------------------------------------------------------------------------
// CPU extension levels — ARM
// ---------------------------------------------------------------------------

/// Supported ARM SIMD extension levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum ArmCpuExtLevel {
    /// No SIMD.
    Generic = 0,
    /// ARM NEON (ASIMD).
    Neon = 1,
    /// ARM SVE.
    Sve = 2,
    /// ARM SVE2.
    Sve2 = 3,
}

// ---------------------------------------------------------------------------
// Runtime CPU feature detection
// ---------------------------------------------------------------------------

/// Detects the highest supported x86-64 SIMD level at runtime.
///
/// On non-x86 targets this always returns `CpuExtLevel::Generic as i32`.
#[cfg(target_arch = "x86_64")]
pub fn get_cpu_ext_level() -> i32 {
    // Probe from highest to lowest.
    if is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
    {
        return CpuExtLevel::Avx512 as i32;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return CpuExtLevel::Avx2Fma as i32;
    }
    if is_x86_feature_detected!("avx2") {
        return CpuExtLevel::Avx2 as i32;
    }
    if is_x86_feature_detected!("avx") {
        return CpuExtLevel::Avx as i32;
    }
    if is_x86_feature_detected!("sse4.2") {
        return CpuExtLevel::Sse42 as i32;
    }
    if is_x86_feature_detected!("sse4.1") {
        return CpuExtLevel::Sse41 as i32;
    }
    if is_x86_feature_detected!("ssse3") {
        return CpuExtLevel::Ssse3 as i32;
    }
    if is_x86_feature_detected!("sse3") {
        return CpuExtLevel::Sse3 as i32;
    }
    if is_x86_feature_detected!("sse2") {
        return CpuExtLevel::Sse2 as i32;
    }
    if is_x86_feature_detected!("sse") {
        return CpuExtLevel::Sse as i32;
    }
    CpuExtLevel::Generic as i32
}

/// Detects the highest supported ARM SIMD level at runtime.
#[cfg(target_arch = "aarch64")]
pub fn get_cpu_ext_level() -> i32 {
    // aarch64 always has NEON.
    ArmCpuExtLevel::Neon as i32
}

/// Fallback for architectures without specialised detection.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn get_cpu_ext_level() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Bit-manipulation helpers
// ---------------------------------------------------------------------------

/// Population count (number of set bits).
#[inline]
pub const fn population_count(val: u32) -> u32 {
    val.count_ones()
}

/// Count leading zeros (32-bit).
#[inline]
pub const fn count_leading_zeros(val: u32) -> u32 {
    val.leading_zeros()
}

/// Count leading zeros (64-bit).
#[inline]
pub const fn count_leading_zeros_u64(val: u64) -> u32 {
    val.leading_zeros()
}

/// Count trailing zeros (32-bit).
#[inline]
pub const fn count_trailing_zeros(val: u32) -> u32 {
    val.trailing_zeros()
}

// ---------------------------------------------------------------------------
// Rounding helpers
// ---------------------------------------------------------------------------

/// Rounds a float to the nearest integer (ties away from zero), matching the
/// C++ `ojph_round` behaviour.
#[inline]
pub fn ojph_round(val: f32) -> i32 {
    (val + if val >= 0.0 { 0.5 } else { -0.5 }) as i32
}

/// Truncates a float toward zero, matching the C++ `ojph_trunc`.
#[inline]
pub fn ojph_trunc(val: f32) -> i32 {
    val as i32
}

// ---------------------------------------------------------------------------
// Alignment helpers
// ---------------------------------------------------------------------------

/// Returns the smallest multiple of `alignment` (in bytes) that can hold
/// `count` elements of type `T`.
///
/// This is the Rust equivalent of the C++ `calc_aligned_size<T>()` template.
#[inline]
pub const fn calc_aligned_size<T>(count: usize, alignment: u32) -> usize {
    let byte_size = count * std::mem::size_of::<T>();
    let align = alignment as usize;
    (byte_size + align - 1) & !(align - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popcount() {
        assert_eq!(population_count(0b1010_1010), 4);
    }

    #[test]
    fn leading_zeros() {
        assert_eq!(count_leading_zeros(1), 31);
        assert_eq!(count_leading_zeros_u64(1), 63);
    }

    #[test]
    fn trailing_zeros() {
        assert_eq!(count_trailing_zeros(8), 3);
    }

    #[test]
    fn round_trunc() {
        assert_eq!(ojph_round(2.3), 2);
        assert_eq!(ojph_round(2.7), 3);
        assert_eq!(ojph_round(-2.3), -2);
        assert_eq!(ojph_trunc(2.9), 2);
        assert_eq!(ojph_trunc(-2.9), -2);
    }

    #[test]
    fn aligned_size() {
        // 10 i32 = 40 bytes → next multiple of 64 = 64
        assert_eq!(calc_aligned_size::<i32>(10, 64), 64);
        // 20 i32 = 80 bytes → next multiple of 64 = 128
        assert_eq!(calc_aligned_size::<i32>(20, 64), 128);
    }

    #[test]
    fn cpu_detection_runs() {
        let level = get_cpu_ext_level();
        assert!(level >= 0);
    }
}
