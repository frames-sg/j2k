// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_cvtepu8_epi32, _mm256_extracti128_si256,
    _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_srai_epi32, _mm256_sub_epi32, _mm_cvtsi128_si64,
    _mm_loadl_epi64, _mm_packs_epi32, _mm_packus_epi16,
};
use core::cell::RefCell;

use crate::color::upsample::{
    h2v2_fancy_sample, upsample_h2v2_fancy_row, upsample_h2v2_fancy_rows,
};
use crate::color::ycbcr::{FIX_0_34414, FIX_0_71414, FIX_1_40200, FIX_1_77200, ROUND};

use super::row_pair::{normalize_simd_row_pair, normalize_ycbcr_row};
use super::{scalar, Rgb420ChromaRows, Rgb420CroppedRowPair, Rgb420RowPair};

const LANES: usize = 8;
const RGB_UNROLL: usize = 8;

#[derive(Default)]
struct RowPairScratch {
    cb_top: Vec<u8>,
    cb_bottom: Vec<u8>,
    cr_top: Vec<u8>,
    cr_bottom: Vec<u8>,
}

impl RowPairScratch {
    fn ensure_width(&mut self, width: usize) {
        self.cb_top.resize(width, 0);
        self.cb_bottom.resize(width, 0);
        self.cr_top.resize(width, 0);
        self.cr_bottom.resize(width, 0);
    }
}

std::thread_local! {
    static ROW_PAIR_SCRATCH: RefCell<RowPairScratch> = RefCell::new(RowPairScratch::default());
}

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    let width = gray_row.len().min(dst.len() / 3);
    let gray_row = &gray_row[..width];
    let dst = &mut dst[..width * 3];
    debug_assert_eq!(dst.len(), gray_row.len() * 3);
    let mut offset = 0;
    while offset + RGB_UNROLL <= gray_row.len() {
        let chunk = &gray_row[offset..offset + RGB_UNROLL];
        let dst_chunk = &mut dst[offset * 3..(offset + RGB_UNROLL) * 3];
        for (gray, pixel) in chunk.iter().zip(dst_chunk.chunks_exact_mut(3)) {
            pixel[0] = *gray;
            pixel[1] = *gray;
            pixel[2] = *gray;
        }
        offset += RGB_UNROLL;
    }
    if offset < gray_row.len() {
        scalar::fill_rgb_row_from_gray(&gray_row[offset..], &mut dst[offset * 3..]);
    }
}

pub(crate) fn fill_rgb_row_from_rgb(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    let width = r_row
        .len()
        .min(g_row.len())
        .min(b_row.len())
        .min(dst.len() / 3);
    let r_row = &r_row[..width];
    let g_row = &g_row[..width];
    let b_row = &b_row[..width];
    let dst = &mut dst[..width * 3];
    debug_assert_eq!(r_row.len(), g_row.len());
    debug_assert_eq!(r_row.len(), b_row.len());
    debug_assert_eq!(dst.len(), r_row.len() * 3);
    let mut offset = 0;
    while offset + RGB_UNROLL <= r_row.len() {
        let r_chunk = &r_row[offset..offset + RGB_UNROLL];
        let g_chunk = &g_row[offset..offset + RGB_UNROLL];
        let b_chunk = &b_row[offset..offset + RGB_UNROLL];
        let dst_chunk = &mut dst[offset * 3..(offset + RGB_UNROLL) * 3];
        for (((&r, &g), &b), pixel) in r_chunk
            .iter()
            .zip(g_chunk.iter())
            .zip(b_chunk.iter())
            .zip(dst_chunk.chunks_exact_mut(3))
        {
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
        offset += RGB_UNROLL;
    }
    if offset < r_row.len() {
        scalar::fill_rgb_row_from_rgb(
            &r_row[offset..],
            &g_row[offset..],
            &b_row[offset..],
            &mut dst[offset * 3..],
        );
    }
}

pub(crate) fn fill_rgb_row_from_ycbcr(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    let (y_row, cb_row, cr_row, dst) = normalize_ycbcr_row(y_row, cb_row, cr_row, dst);
    debug_assert_eq!(y_row.len(), cb_row.len());
    debug_assert_eq!(y_row.len(), cr_row.len());
    debug_assert_eq!(dst.len(), y_row.len() * 3);
    // SAFETY: Backend dispatch selects this path only when AVX2 is available.
    // All source rows and the destination are narrowed to the same pixel count.
    unsafe {
        fill_rgb_row_from_ycbcr_avx2(y_row, cb_row, cr_row, dst);
    }
}

pub(crate) fn fill_rgb_row_pair_from_420(request: Rgb420RowPair<'_>) {
    let Some(request) = normalize_simd_row_pair(request) else {
        return;
    };
    let width = request.y_top.len();

    ROW_PAIR_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_width(width);
        // SAFETY: Backend dispatch selects this path only when AVX2 is
        // available. The wrapper clamps luma, chroma, and destination rows so
        // all upsampled reads and RGB writes fit the passed slices.
        unsafe {
            fill_rgb_row_pair_from_420_avx2(request, &mut scratch);
        }
    });
}

pub(crate) fn fill_rgb_row_pair_from_420_cropped(request: Rgb420CroppedRowPair<'_>) {
    let Rgb420CroppedRowPair { rows, crop } = request;
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = rows;
    let crop_start = crop.start;
    let crop_width = crop.width;
    let chroma_width = chroma.min_width();
    let available_chroma = chroma_width.saturating_mul(2).saturating_sub(crop_start);
    let available_top = y_top.len().saturating_sub(crop_start);
    let bottom_available = match (y_bottom.as_ref(), dst_bottom.as_ref()) {
        (Some(row), Some(dst)) => row.len().saturating_sub(crop_start).min(dst.len() / 3),
        _ => usize::MAX,
    };
    let width = crop_width
        .min(available_top)
        .min(dst_top.len() / 3)
        .min(bottom_available)
        .min(available_chroma);
    if width == 0 {
        return;
    }
    let Some(crop_end) = crop_start.checked_add(width) else {
        return;
    };
    let Some(y_top_crop) = y_top.get(crop_start..crop_end) else {
        return;
    };
    let y_bottom = y_bottom.and_then(|row| row.get(crop_start..crop_end));
    let prev_cb = &chroma.prev_cb[..chroma_width];
    let curr_cb = &chroma.curr_cb[..chroma_width];
    let next_cb = &chroma.next_cb[..chroma_width];
    let prev_cr = &chroma.prev_cr[..chroma_width];
    let curr_cr = &chroma.curr_cr[..chroma_width];
    let next_cr = &chroma.next_cr[..chroma_width];
    let dst_top = &mut dst_top[..width * 3];
    let dst_bottom = dst_bottom.and_then(|row| row.get_mut(..width * 3));
    debug_assert_eq!(dst_top.len(), width * 3);
    debug_assert!(y_bottom.is_none_or(|row| row.len() == width));
    debug_assert!(dst_bottom.as_ref().is_none_or(|row| row.len() == width * 3));
    debug_assert_eq!(prev_cb.len(), curr_cb.len());
    debug_assert_eq!(prev_cb.len(), next_cb.len());
    debug_assert_eq!(prev_cr.len(), curr_cr.len());
    debug_assert_eq!(prev_cr.len(), next_cr.len());

    ROW_PAIR_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_width(width);
        let RowPairScratch {
            cb_top,
            cb_bottom,
            cr_top,
            cr_bottom,
        } = &mut *scratch;
        let cb_top = &mut cb_top[..width];
        let cr_top = &mut cr_top[..width];
        fill_cropped_h2v2_row(prev_cb, curr_cb, crop_start, cb_top);
        fill_cropped_h2v2_row(prev_cr, curr_cr, crop_start, cr_top);
        // SAFETY: Backend dispatch selects this path only when AVX2 is
        // available. `y_top_crop`, scratch chroma rows, and destination slices
        // all have the same bounded pixel width.
        unsafe {
            fill_rgb_row_from_ycbcr_avx2(y_top_crop, cb_top, cr_top, dst_top);
        }

        if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
            let cb_bottom = &mut cb_bottom[..width];
            let cr_bottom = &mut cr_bottom[..width];
            fill_cropped_h2v2_row(next_cb, curr_cb, crop_start, cb_bottom);
            fill_cropped_h2v2_row(next_cr, curr_cr, crop_start, cr_bottom);
            // SAFETY: Backend dispatch selects this path only when AVX2 is
            // available. The bottom luma, scratch chroma, and destination
            // slices were clamped to the same bounded pixel width.
            unsafe {
                fill_rgb_row_from_ycbcr_avx2(y_bottom, cb_bottom, cr_bottom, dst_bottom);
            }
        }
    });
}

fn fill_cropped_h2v2_row(near: &[u8], curr: &[u8], crop_start: usize, out: &mut [u8]) {
    for (local_x, slot) in out.iter_mut().enumerate() {
        *slot = h2v2_fancy_sample(near, curr, crop_start + local_x);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fill_rgb_row_pair_from_420_avx2(
    request: Rgb420RowPair<'_>,
    scratch: &mut RowPairScratch,
) {
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = request;
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    let width = y_top.len();
    let RowPairScratch {
        cb_top,
        cb_bottom,
        cr_top,
        cr_bottom,
    } = scratch;
    let cb_top = &mut cb_top[..width];
    let cr_top = &mut cr_top[..width];
    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        let cb_bottom = &mut cb_bottom[..width];
        let cr_bottom = &mut cr_bottom[..width];
        upsample_h2v2_fancy_rows(prev_cb, curr_cb, next_cb, width, cb_top, cb_bottom);
        upsample_h2v2_fancy_rows(prev_cr, curr_cr, next_cr, width, cr_top, cr_bottom);
        // SAFETY: This AVX2 helper is reached through the safe wrapper, which
        // clamps both luma rows, both scratch chroma rows, and both RGB
        // destination rows to the same pixel width.
        unsafe {
            fill_rgb_row_from_ycbcr_avx2(y_top, cb_top, cr_top, dst_top);
            fill_rgb_row_from_ycbcr_avx2(y_bottom, cb_bottom, cr_bottom, dst_bottom);
        }
    } else {
        upsample_h2v2_fancy_row(prev_cb, curr_cb, next_cb, width, false, cb_top);
        upsample_h2v2_fancy_row(prev_cr, curr_cr, next_cr, width, false, cr_top);
        // SAFETY: This AVX2 helper is reached through the safe wrapper, which
        // clamps the luma row, scratch chroma rows, and RGB destination row to
        // the same pixel width.
        unsafe {
            fill_rgb_row_from_ycbcr_avx2(y_top, cb_top, cr_top, dst_top);
        }
    }
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_ycbcr_for_test(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    fill_rgb_row_from_ycbcr(y_row, cb_row, cr_row, dst);
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_gray_for_test(gray_row: &[u8], dst: &mut [u8]) {
    fill_rgb_row_from_gray(gray_row, dst);
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_rgb_for_test(
    r_row: &[u8],
    g_row: &[u8],
    b_row: &[u8],
    dst: &mut [u8],
) {
    fill_rgb_row_from_rgb(r_row, g_row, b_row, dst);
}

#[target_feature(enable = "avx2")]
unsafe fn fill_rgb_row_from_ycbcr_avx2(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    let width = y_row.len();
    let mut offset = 0;

    while offset + (LANES * 2) <= width {
        // SAFETY: The safe wrapper slices all input rows and `dst` to the same
        // pixel count, and this loop only passes full eight-pixel chunks.
        unsafe {
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[offset * 3..(offset + LANES) * 3],
                offset,
            );
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[(offset + LANES) * 3..(offset + LANES * 2) * 3],
                offset + LANES,
            );
        }
        offset += LANES * 2;
    }

    while offset + LANES <= width {
        // SAFETY: The safe wrapper slices all input rows and `dst` to the same
        // pixel count, and this loop only passes a full eight-pixel chunk.
        unsafe {
            fill_chunk(
                y_row,
                cb_row,
                cr_row,
                &mut dst[offset * 3..(offset + LANES) * 3],
                offset,
            );
        }
        offset += LANES;
    }

    if offset < width {
        scalar::fill_rgb_row_from_ycbcr(
            &y_row[offset..],
            &cb_row[offset..],
            &cr_row[offset..],
            &mut dst[offset * 3..],
        );
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fill_chunk(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst_chunk: &mut [u8],
    offset: usize,
) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);

    // SAFETY: callers prove `offset + LANES <= row.len()` for each source row.
    let y = unsafe { load_eight(y_row, offset) };
    // SAFETY: callers prove `offset + LANES <= row.len()` for each source row.
    let cb = unsafe { load_eight(cb_row, offset) };
    // SAFETY: callers prove `offset + LANES <= row.len()` for each source row.
    let cr = unsafe { load_eight(cr_row, offset) };

    let bias = _mm256_set1_epi32(128);
    let y32 = _mm256_cvtepu8_epi32(y);
    let cb32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(cb), bias);
    let cr32 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(cr), bias);

    let r = _mm256_add_epi32(y32, fixed_mul_shift(cr32, FIX_1_40200));
    let g = _mm256_sub_epi32(
        y32,
        _mm256_srai_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(cb32, _mm256_set1_epi32(FIX_0_34414)),
                    _mm256_mullo_epi32(cr32, _mm256_set1_epi32(FIX_0_71414)),
                ),
                _mm256_set1_epi32(ROUND),
            ),
            16,
        ),
    );
    let b = _mm256_add_epi32(y32, fixed_mul_shift(cb32, FIX_1_77200));

    // SAFETY: `dst_chunk` is narrowed by the caller to exactly one RGB chunk.
    unsafe {
        store_rgb_chunk(dst_chunk, r, g, b);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn load_eight(src: &[u8], offset: usize) -> __m128i {
    debug_assert!(offset <= src.len().saturating_sub(LANES));
    // SAFETY: the caller guarantees there are at least eight readable bytes at
    // `offset`; `_mm_loadl_epi64` accepts unaligned loads.
    unsafe { _mm_loadl_epi64(src.as_ptr().add(offset).cast()) }
}

#[target_feature(enable = "avx2")]
fn fixed_mul_shift(values: __m256i, coefficient: i32) -> __m256i {
    _mm256_srai_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(values, _mm256_set1_epi32(coefficient)),
            _mm256_set1_epi32(ROUND),
        ),
        16,
    )
}

#[target_feature(enable = "avx2")]
unsafe fn store_rgb_chunk(dst_chunk: &mut [u8], r: __m256i, g: __m256i, b: __m256i) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);
    // SAFETY: packing only rearranges register values and does not dereference.
    let r_bytes = unsafe { pack_eight_u8(r) };
    // SAFETY: packing only rearranges register values and does not dereference.
    let g_bytes = unsafe { pack_eight_u8(g) };
    // SAFETY: packing only rearranges register values and does not dereference.
    let b_bytes = unsafe { pack_eight_u8(b) };

    for ((((r, g), b), pixel), _) in r_bytes
        .iter()
        .zip(g_bytes.iter())
        .zip(b_bytes.iter())
        .zip(dst_chunk.chunks_exact_mut(3))
        .zip(0..LANES)
    {
        pixel[0] = *r;
        pixel[1] = *g;
        pixel[2] = *b;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn pack_eight_u8(values: __m256i) -> [u8; LANES] {
    let words = _mm_packs_epi32(
        _mm256_extracti128_si256(values, 0),
        _mm256_extracti128_si256(values, 1),
    );
    let bytes = _mm_packus_epi16(words, words);
    _mm_cvtsi128_si64(bytes).to_ne_bytes()
}
