// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed-size x86 SIMD memory operations.

use core::arch::x86_64::{__m128i, _mm_loadl_epi64, _mm_storel_epi64};

use crate::simd::x86::ExactAvx2;

const U8_LANES: usize = 8;

pub(crate) type U8x8Triple = (__m128i, __m128i, __m128i);

/// A capability- and lifetime-carrying cursor over three equally stepped rows.
///
/// The constructor fixes the readable extent to the shortest row. Private
/// fields ensure that `offset` and `chunks_remaining` can only advance
/// together, which lets the hot load leaf avoid repeating slice bounds checks.
pub(crate) struct U8x8TripleCursor<'a> {
    _avx2: ExactAvx2,
    y: &'a [u8],
    cb: &'a [u8],
    cr: &'a [u8],
    offset: usize,
    chunks_remaining: usize,
}

impl<'a> U8x8TripleCursor<'a> {
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
    pub(crate) fn next_pair(&mut self) -> Option<(usize, U8x8Triple, U8x8Triple)> {
        if self.chunks_remaining < 2 {
            return None;
        }
        let offset = self.offset;
        let first = self.load_current();
        self.advance();
        let second = self.load_current();
        self.advance();
        Some((offset, first, second))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn next(&mut self) -> Option<(usize, U8x8Triple)> {
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
    fn load_current(&self) -> U8x8Triple {
        debug_assert!(self.chunks_remaining > 0);
        // SAFETY:
        // - Feature availability: construction requires an `ExactAvx2` token
        //   and this leaf is compiled for AVX2.
        // - Bounds: the constructor derives the chunk count from the shortest
        //   row, and private cursor state advances the offset by exactly eight
        //   while a chunk remains, proving eight readable bytes in every row.
        // - Alignment: `_mm_loadl_epi64` permits unaligned addresses.
        // - Aliasing: all three rows are shared references and are only read;
        //   overlap between them is therefore harmless.
        // - Initialization: every byte reachable through the input slices is
        //   initialized, and each intrinsic reads exactly eight such bytes.
        unsafe {
            (
                _mm_loadl_epi64(self.y.as_ptr().add(self.offset).cast()),
                _mm_loadl_epi64(self.cb.as_ptr().add(self.offset).cast()),
                _mm_loadl_epi64(self.cr.as_ptr().add(self.offset).cast()),
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
    use core::arch::x86_64::_mm_cvtsi128_si64;

    use super::U8x8TripleCursor;
    use crate::simd::x86::{exact_avx2_kernel, ExactAvx2};

    #[test]
    fn triple_cursor_loads_unaligned_rows_and_stops_at_the_shortest() {
        let Some(avx2) = ExactAvx2::detect() else {
            return;
        };
        let y_storage = (0_u8..24).collect::<Vec<_>>();
        let cb_storage = (40_u8..64).collect::<Vec<_>>();
        let cr_storage = (80_u8..104).collect::<Vec<_>>();
        let y = &y_storage[1..18];
        let cb = &cb_storage[2..18];
        let cr = &cr_storage[3..19];
        exercise_cursor(avx2, y, cb, cr);
    }

    exact_avx2_kernel! {
        fn exercise_cursor(avx2: ExactAvx2, y: &[u8], cb: &[u8], cr: &[u8]) {
            let mut cursor = U8x8TripleCursor::new(avx2, y, cb, cr);
            assert_eq!(cursor.offset(), 0);
            let (offset, first, second) = cursor.next_pair().expect("two full chunks");
            assert_eq!(offset, 0);
            assert_eq!(bytes(first.0), y[..8]);
            assert_eq!(bytes(first.1), cb[..8]);
            assert_eq!(bytes(first.2), cr[..8]);
            assert_eq!(bytes(second.0), y[8..16]);
            assert_eq!(bytes(second.1), cb[8..16]);
            assert_eq!(bytes(second.2), cr[8..16]);
            assert_eq!(cursor.offset(), 16);
            assert!(cursor.next().is_none());
        }
    }

    #[target_feature(enable = "avx2")]
    fn bytes(values: core::arch::x86_64::__m128i) -> [u8; 8] {
        _mm_cvtsi128_si64(values).to_ne_bytes()
    }
}
