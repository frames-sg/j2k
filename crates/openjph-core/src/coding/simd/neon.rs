//! NEON-accelerated block coding helper routines for AArch64.
//!
//! Provides SIMD primitives used by the HTJ2K block encoder/decoder, such as
//! population count vectorization and magnitude computation.

#![allow(dead_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// NEON-accelerated OR-reduce: computes the bitwise OR of all elements in a
/// u32 buffer. Used by the encoder to compute the aggregate significance mask.
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_or_reduce(data: &[u32], count: u32) -> u32 {
    unsafe { neon_or_reduce_inner(data, count) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_or_reduce_inner(data: &[u32], count: u32) -> u32 {
    let mut acc = vdupq_n_u32(0);
    let simd_count = count / 4;
    let remainder = count % 4;
    let mut ptr = data.as_ptr();

    for _ in 0..simd_count {
        let v = vld1q_u32(ptr);
        acc = vorrq_u32(acc, v);
        ptr = ptr.add(4);
    }

    // Reduce the 4-wide accumulator to a scalar
    let mut result = vgetq_lane_u32::<0>(acc)
        | vgetq_lane_u32::<1>(acc)
        | vgetq_lane_u32::<2>(acc)
        | vgetq_lane_u32::<3>(acc);

    // Handle remainder
    for i in 0..remainder as usize {
        result |= *ptr.add(i);
    }

    result
}

/// NEON-accelerated magnitude computation: for each sample, computes
/// the number of leading zeros (used for MSB computation in the encoder).
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_count_leading_zeros_batch(data: &[u32], out: &mut [u32], count: u32) {
    unsafe { neon_clz_batch_inner(data, out, count) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_clz_batch_inner(data: &[u32], out: &mut [u32], count: u32) {
    let simd_count = count / 4;
    let remainder = count % 4;
    let mut src = data.as_ptr();
    let mut dst = out.as_mut_ptr();

    for _ in 0..simd_count {
        let v = vld1q_u32(src);
        // Reinterpret as signed for clz (same bit pattern)
        let clz = vclzq_s32(vreinterpretq_s32_u32(v));
        vst1q_u32(dst, vreinterpretq_u32_s32(clz));
        src = src.add(4);
        dst = dst.add(4);
    }

    for i in 0..remainder as usize {
        *dst.add(i) = (*src.add(i)).leading_zeros();
    }
}

/// NEON-accelerated absolute value for i32 buffers (used in sign-magnitude
/// processing during block encoding).
#[cfg(target_arch = "aarch64")]
pub(crate) fn neon_abs_i32(data: &[i32], out: &mut [u32], count: u32) {
    unsafe { neon_abs_i32_inner(data, out, count) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_abs_i32_inner(data: &[i32], out: &mut [u32], count: u32) {
    let simd_count = count / 4;
    let remainder = count % 4;
    let mut src = data.as_ptr();
    let mut dst = out.as_mut_ptr();

    for _ in 0..simd_count {
        let v = vld1q_s32(src);
        let abs_v = vabsq_s32(v);
        vst1q_u32(dst, vreinterpretq_u32_s32(abs_v));
        src = src.add(4);
        dst = dst.add(4);
    }

    for i in 0..remainder as usize {
        *dst.add(i) = (*src.add(i)).unsigned_abs();
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn neon_or_reduce_matches_scalar() {
        for count in [1, 3, 4, 7, 8, 16, 33] {
            let data: Vec<u32> = (0..count).map(|i| 1u32 << (i % 32)).collect();
            let scalar = data.iter().copied().fold(0u32, |a, b| a | b);
            let neon = neon_or_reduce(&data, count as u32);
            assert_eq!(scalar, neon, "or_reduce mismatch at count={count}");
        }
    }

    #[test]
    fn neon_clz_batch_matches_scalar() {
        for count in [1, 4, 7, 16, 33] {
            let data: Vec<u32> = (0..count).map(|i| (i as u32 + 1) * 17).collect();
            let mut scalar_out = vec![0u32; count];
            let mut neon_out = vec![0u32; count];

            for i in 0..count {
                scalar_out[i] = data[i].leading_zeros();
            }
            neon_count_leading_zeros_batch(&data, &mut neon_out, count as u32);

            assert_eq!(scalar_out, neon_out, "clz mismatch at count={count}");
        }
    }

    #[test]
    fn neon_abs_i32_matches_scalar() {
        for count in [1, 4, 7, 16, 33] {
            let data: Vec<i32> = (0..count)
                .map(|i| {
                    if i % 2 == 0 {
                        i as i32 + 1
                    } else {
                        -(i as i32) - 1
                    }
                })
                .collect();
            let mut scalar_out = vec![0u32; count];
            let mut neon_out = vec![0u32; count];

            for i in 0..count {
                scalar_out[i] = data[i].unsigned_abs();
            }
            neon_abs_i32(&data, &mut neon_out, count as u32);

            assert_eq!(scalar_out, neon_out, "abs mismatch at count={count}");
        }
    }
}
