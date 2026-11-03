// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size x86 SIMD memory operations.

use core::arch::x86_64::{__m128i, __m256i, _mm256_broadcastsi128_si256, _mm_storel_epi64};

use crate::simd::x86::ExactAvx2;

const U8_LANES: usize = 32;

pub(crate) type U8x32Triple = (__m256i, __m256i, __m256i);

/// A capability- and lifetime-carrying cursor over three equally stepped rows.
///
/// The constructor fixes the readable extent to the shortest row. Private
/// fields ensure that `offset` and `chunks_remaining` can only advance
/// together, which lets the hot load leaf avoid repeating slice bounds checks.
pub(crate) struct U8x32TripleCursor<'a> {
    _avx2: ExactAvx2,
    y: &'a [u8],
    cb: &'a [u8],
    cr: &'a [u8],
    offset: usize,
    chunks_remaining: usize,
}

impl<'a> U8x32TripleCursor<'a> {
    pub(crate) fn new(avx2: ExactAvx2, y: &'a [u8], cb: &'a [u8], cr: &'a [u8]) -> Self {
        let chunks_remaining = y.len().min(cb.len()).min(cr.len()) / U8_LANES;
        Self {
            _avx2: avx2,
            y,
            cb,
            cr,
            offset: 0,
            chunks_remaining,
        }
    }

    pub(crate) const fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn next(&mut self) -> Option<(usize, U8x32Triple)> {
        if self.chunks_remaining == 0 {
            return None;
        }
        let offset = self.offset;
        let values = self.load_current();
        self.advance();
        Some((offset, values))
    }

    #[inline]
    fn advance(&mut self) {
        self.offset += U8_LANES;
        self.chunks_remaining -= 1;
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load_current(&self) -> U8x32Triple {
        debug_assert!(self.chunks_remaining > 0);
        // SAFETY:
        // - Feature availability: construction requires an `ExactAvx2` token
        //   and this leaf is compiled for AVX2.
        // - Bounds: the constructor derives the chunk count from the shortest
        //   row, and private cursor state advances the offset by exactly 32
        //   while a chunk remains, proving 32 readable bytes in every row.
        // - Alignment: `read_unaligned` deliberately accepts any address.
        // - Aliasing: all three rows are shared references and are only read;
        //   overlap between them is therefore harmless.
        // - Initialization: every byte reachable through the input slices is
        //   initialized, and each read covers exactly 32 such bytes.
        unsafe {
            (
                core::ptr::read_unaligned(self.y.as_ptr().add(self.offset).cast::<__m256i>()),
                core::ptr::read_unaligned(self.cb.as_ptr().add(self.offset).cast::<__m256i>()),
                core::ptr::read_unaligned(self.cr.as_ptr().add(self.offset).cast::<__m256i>()),
            )
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn load_i16x8(src: &[i16; 8]) -> __m128i {
    // SAFETY:
    // - Feature availability: callers run inside an exact-AVX2 kernel.
    // - Bounds: the array reference proves sixteen readable bytes.
    // - Alignment: `read_unaligned` deliberately accepts any i16 alignment.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all coefficients behind the reference are initialized.
    unsafe { core::ptr::read_unaligned(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn load_u8x16(src: &[u8; 16]) -> __m128i {
    // SAFETY:
    // - Feature availability: callers run inside an exact-AVX2 kernel.
    // - Bounds: the array reference proves sixteen readable bytes.
    // - Alignment: `read_unaligned` deliberately accepts any byte alignment.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all bytes behind the reference are initialized.
    unsafe { core::ptr::read_unaligned(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn load_i16x16(src: &[i16; 16]) -> __m256i {
    // SAFETY:
    // - Feature availability: callers run inside an exact-AVX2 kernel.
    // - Bounds: the array reference proves thirty-two readable bytes.
    // - Alignment: `read_unaligned` deliberately accepts any i16 alignment.
    // - Aliasing: the shared reference permits reads and no writes occur.
    // - Initialization: all coefficients behind the reference are initialized.
    unsafe { core::ptr::read_unaligned(src.as_ptr().cast::<__m256i>()) }
}

/// Load 8 `i16` values into both 128-bit lanes (`vbroadcasti128`).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn broadcast_i16x8(src: &[i16; 8]) -> __m256i {
    _mm256_broadcastsi128_si256(load_i16x8(src))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn store_u8x32(dst: &mut [u8; 32], values: __m256i) {
    // SAFETY:
    // - Feature availability: callers run inside an exact-AVX2 kernel.
    // - Bounds: the array reference proves thirty-two writable bytes.
    // - Alignment: `write_unaligned` deliberately accepts any byte alignment.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes every byte in the output array.
    unsafe { core::ptr::write_unaligned(dst.as_mut_ptr().cast::<__m256i>(), values) };
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn store_u8x8(dst: &mut [u8; 8], values: __m128i) {
    // SAFETY:
    // - Feature availability: callers run inside an exact-AVX2 kernel.
    // - Bounds: the array reference proves eight writable bytes.
    // - Alignment: `_mm_storel_epi64` permits an unaligned address.
    // - Aliasing: the exclusive reference prevents overlapping live access.
    // - Initialization: the store initializes every byte in the output array.
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast(), values) };
}

#[cfg(test)]
mod tests {
    use super::U8x32TripleCursor;
    use crate::simd::x86::{exact_avx2_kernel, ExactAvx2};

    #[test]
    fn triple_cursor_loads_unaligned_rows_and_stops_at_the_shortest() {
        let Some(avx2) = ExactAvx2::detect() else {
            return;
        };
        let y_storage = (0_u8..96).collect::<Vec<_>>();
        let cb_storage = (100_u8..196).collect::<Vec<_>>();
        let cr_storage = (150_u8..246).collect::<Vec<_>>();
        let y = &y_storage[1..70];
        let cb = &cb_storage[2..66];
        let cr = &cr_storage[3..90];
        exercise_cursor(avx2, y, cb, cr);
    }

    exact_avx2_kernel! {
        fn exercise_cursor(avx2: ExactAvx2, y: &[u8], cb: &[u8], cr: &[u8]) {
            let mut cursor = U8x32TripleCursor::new(avx2, y, cb, cr);
            assert_eq!(cursor.offset(), 0);
            let (offset, first) = cursor.next().expect("first full chunk");
            assert_eq!(offset, 0);
            assert_eq!(bytes(first.0), y[..32]);
            assert_eq!(bytes(first.1), cb[..32]);
            assert_eq!(bytes(first.2), cr[..32]);
            let (offset, second) = cursor.next().expect("second full chunk");
            assert_eq!(offset, 32);
            assert_eq!(bytes(second.0), y[32..64]);
            assert_eq!(bytes(second.1), cb[32..64]);
            assert_eq!(bytes(second.2), cr[32..64]);
            // `cb` holds 64 bytes, so no third chunk is readable in all rows.
            assert_eq!(cursor.offset(), 64);
            assert!(cursor.next().is_none());
        }
    }

    #[target_feature(enable = "avx2")]
    fn bytes(values: core::arch::x86_64::__m256i) -> [u8; 32] {
        let mut out = [0u8; 32];
        super::store_u8x32(&mut out, values);
        out
    }
}
