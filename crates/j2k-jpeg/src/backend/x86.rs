// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_broadcastsi128_si256, _mm256_cvtepu8_epi16,
    _mm256_madd_epi16, _mm256_mulhi_epi16, _mm256_or_si256, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srai_epi16,
    _mm256_srai_epi32, _mm256_srli_epi16, _mm256_sub_epi16, _mm256_unpackhi_epi16,
    _mm256_unpackhi_epi8, _mm256_unpacklo_epi16, _mm256_unpacklo_epi8, _mm_setr_epi8,
};
use core::cell::RefCell;

use crate::color::upsample::{emit_h2v2_row_with, h2v2_fancy_sample};
use crate::color::ycbcr::{FIX_0_34414, FIX_0_71414, FIX_1_40200, FIX_1_77200, ROUND};
use crate::simd::x86::{exact_avx2_kernel, madd_pair, ExactAvx2};
use crate::simd::x86_memory;

use super::row_pair::{normalize_simd_row_pair, normalize_ycbcr_row};
use super::{scalar, Rgb420ChromaRows, Rgb420CroppedRowPair, Rgb420RowPair};

/// Pixels per colour-conversion step: one 256-bit register of bytes.
const LANES: usize = 32;
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

#[inline]
pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    let width = gray_row.len().min(dst.len() / 3);
    let gray_row = &gray_row[..width];
    let dst = &mut dst[..width * 3];
    debug_assert_eq!(dst.len(), gray_row.len() * 3);
    let mut offset = 0;
    while offset + RGB_UNROLL <= gray_row.len() {
        let chunk = &gray_row[offset..offset + RGB_UNROLL];
        let dst_chunk = &mut dst[offset * 3..(offset + RGB_UNROLL) * 3];
        for (gray, pixel) in chunk.iter().zip(dst_chunk.as_chunks_mut::<3>().0) {
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
            .zip(dst_chunk.as_chunks_mut::<3>().0)
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

pub(crate) fn fill_rgb_row_from_ycbcr(
    avx2: ExactAvx2,
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    let (y_row, cb_row, cr_row, dst) = normalize_ycbcr_row(y_row, cb_row, cr_row, dst);
    debug_assert_eq!(y_row.len(), cb_row.len());
    debug_assert_eq!(y_row.len(), cr_row.len());
    debug_assert_eq!(dst.len(), y_row.len() * 3);
    fill_rgb_row_from_ycbcr_kernel(avx2, y_row, cb_row, cr_row, dst);
}

exact_avx2_kernel! {
    fn fill_rgb_row_from_ycbcr_kernel(
        avx2: ExactAvx2,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        dst: &mut [u8],
    ) {
        let width = y_row.len();
        let mut sources = x86_memory::U8x32TripleCursor::new(avx2, y_row, cb_row, cr_row);
        while let Some((offset, (y, cb, cr))) = sources.next() {
            let Some((chunk, _)) = dst[offset * 3..].split_first_chunk_mut::<{ LANES * 3 }>()
            else {
                break;
            };
            store_rgb_chunk(chunk, ycbcr_chunk_to_rgb(y, cb, cr));
        }

        // Convert the last partial chunk through zero-padded copies.
        let offset = sources.offset();
        if offset < width {
            let tail = width - offset;
            let mut padded = [[0u8; LANES]; 3];
            for (buffer, row) in padded.iter_mut().zip([y_row, cb_row, cr_row]) {
                buffer[..tail].copy_from_slice(&row[offset..width]);
            }
            let mut pixels = [0u8; LANES * 3];
            let mut tail_source =
                x86_memory::U8x32TripleCursor::new(avx2, &padded[0], &padded[1], &padded[2]);
            if let Some((_, (y, cb, cr))) = tail_source.next() {
                store_rgb_chunk(&mut pixels, ycbcr_chunk_to_rgb(y, cb, cr));
            }
            dst[offset * 3..width * 3].copy_from_slice(&pixels[..tail * 3]);
        }
    }
}

pub(crate) fn fill_rgb_row_pair_from_420(avx2: ExactAvx2, request: Rgb420RowPair<'_>) {
    let Some(request) = normalize_simd_row_pair(request) else {
        return;
    };
    let width = request.y_top.len();

    ROW_PAIR_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_width(width);
        fill_rgb_row_pair_from_420_kernel(avx2, request, &mut scratch);
    });
}

exact_avx2_kernel! {
    fn fill_rgb_row_pair_from_420_kernel(
        avx2: ExactAvx2,
        request: Rgb420RowPair<'_>,
        scratch: &mut RowPairScratch,
    ) {
        fill_rgb_row_pair_from_420_avx2(avx2, request, scratch);
    }
}

pub(crate) fn fill_rgb_row_pair_from_420_cropped(
    avx2: ExactAvx2,
    request: Rgb420CroppedRowPair<'_>,
) {
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
        fill_rgb_row_from_ycbcr_kernel(avx2, y_top_crop, cb_top, cr_top, dst_top);

        if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
            let cb_bottom = &mut cb_bottom[..width];
            let cr_bottom = &mut cr_bottom[..width];
            fill_cropped_h2v2_row(next_cb, curr_cb, crop_start, cb_bottom);
            fill_cropped_h2v2_row(next_cr, curr_cr, crop_start, cr_bottom);
            fill_rgb_row_from_ycbcr_kernel(avx2, y_bottom, cb_bottom, cr_bottom, dst_bottom);
        }
    });
}

fn fill_cropped_h2v2_row(near: &[u8], curr: &[u8], crop_start: usize, out: &mut [u8]) {
    for (local_x, slot) in out.iter_mut().enumerate() {
        *slot = h2v2_fancy_sample(near, curr, crop_start + local_x);
    }
}

#[target_feature(enable = "avx2")]
fn fill_rgb_row_pair_from_420_avx2(
    avx2: ExactAvx2,
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
        upsample_h2v2_row(avx2, prev_cb, curr_cb, width, cb_top);
        upsample_h2v2_row(avx2, next_cb, curr_cb, width, cb_bottom);
        upsample_h2v2_row(avx2, prev_cr, curr_cr, width, cr_top);
        upsample_h2v2_row(avx2, next_cr, curr_cr, width, cr_bottom);
        fill_rgb_row_from_ycbcr_kernel(avx2, y_top, cb_top, cr_top, dst_top);
        fill_rgb_row_from_ycbcr_kernel(avx2, y_bottom, cb_bottom, cr_bottom, dst_bottom);
    } else {
        upsample_h2v2_row(avx2, prev_cb, curr_cb, width, cb_top);
        upsample_h2v2_row(avx2, prev_cr, curr_cr, width, cr_top);
        fill_rgb_row_from_ycbcr_kernel(avx2, y_top, cb_top, cr_top, dst_top);
    }
}

/// One visible row of fancy 4:2:0 chroma upsampling, bit-exact with
/// [`crate::color::upsample::upsample_h2v2_fancy_row`]: the shared row
/// driver handles the edges and [`h2v2_fancy_pairs`] the interior.
pub(super) fn upsample_h2v2_row(
    avx2: ExactAvx2,
    near: &[u8],
    curr: &[u8],
    output_width: usize,
    out: &mut [u8],
) {
    emit_h2v2_row_with(near, curr, output_width, out, |curr, near, interior| {
        h2v2_fancy_pairs(avx2, curr, near, interior)
    });
}

exact_avx2_kernel! {
    /// Fill the leading interior pairs of a fancy 4:2:0 chroma row, sixteen
    /// chroma samples at a time; the AVX2 twin of
    /// `neon_upsample::h2v2_fancy_pairs`. With column sums
    /// `cs(i) = 3 * curr[i] + near[i]`, pair `i` is
    /// `interior[2i] = (3 * cs(i) + cs(i + 1) + 7) >> 4` and
    /// `interior[2i + 1] = (3 * cs(i + 1) + cs(i) + 8) >> 4`. Returns how many
    /// pairs were filled; the caller finishes the rest.
    fn h2v2_fancy_pairs(
        avx2: ExactAvx2,
        curr: &[u8],
        near: &[u8],
        interior: &mut [u8],
    ) -> usize {
        let pairs = interior.len() / 2;
        let mut done = 0;
        // Each step reads samples `done..done + 17` and writes 32 outputs.
        while done + 16 <= pairs {
            let (Some(c0), Some(c1), Some(n0), Some(n1)) = (
                curr.get(done..).and_then(<[u8]>::first_chunk::<16>),
                curr.get(done + 1..).and_then(<[u8]>::first_chunk::<16>),
                near.get(done..).and_then(<[u8]>::first_chunk::<16>),
                near.get(done + 1..).and_then(<[u8]>::first_chunk::<16>),
            ) else {
                break;
            };
            let Some(out) = interior[2 * done..].first_chunk_mut::<32>() else {
                break;
            };
            // Column sums of samples `i` (left) and `i + 1` (right), in u16.
            let left = column_sums(c0, n0);
            let right = column_sums(c1, n1);
            // At most 3 * 1020 + 1020 + 8 = 4088, so the u16 lanes cannot wrap.
            let odd = _mm256_srli_epi16::<4>(_mm256_add_epi16(
                _mm256_add_epi16(triple(left), right),
                _mm256_set1_epi16(7),
            ));
            let even = _mm256_srli_epi16::<4>(_mm256_add_epi16(
                _mm256_add_epi16(triple(right), left),
                _mm256_set1_epi16(8),
            ));
            // Both fit a byte: lane `i` becomes the byte pair `(odd, even)`.
            x86_memory::store_u8x32(out, _mm256_or_si256(odd, _mm256_slli_epi16::<8>(even)));
            done += 16;
        }
        done
    }
}

/// `3 * curr + near` for sixteen samples, widened to u16 lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn column_sums(curr: &[u8; 16], near: &[u8; 16]) -> __m256i {
    let curr = _mm256_cvtepu8_epi16(x86_memory::load_u8x16(curr));
    let near = _mm256_cvtepu8_epi16(x86_memory::load_u8x16(near));
    _mm256_add_epi16(triple(curr), near)
}

#[inline]
#[target_feature(enable = "avx2")]
fn triple(values: __m256i) -> __m256i {
    _mm256_add_epi16(_mm256_add_epi16(values, values), values)
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_ycbcr_for_test(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    let avx2 = ExactAvx2::detect().expect("x86 test requested AVX2 on an unsupported host");
    fill_rgb_row_from_ycbcr(avx2, y_row, cb_row, cr_row, dst);
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

/// `R`, `G` and `B` bytes of 32 pixels, bit-exact with
/// [`crate::color::ycbcr::ycbcr_to_rgb`]. The 16-bit halves hold pixels
/// `[0..8 | 16..24]` and `[8..16 | 24..32]`; packing restores the order.
#[inline]
#[target_feature(enable = "avx2")]
fn ycbcr_chunk_to_rgb(y: __m256i, cb: __m256i, cr: __m256i) -> [__m256i; 3] {
    let zero = _mm256_setzero_si256();
    let [r_lo, g_lo, b_lo] = ycbcr16_to_rgb(
        _mm256_unpacklo_epi8(y, zero),
        _mm256_unpacklo_epi8(cb, zero),
        _mm256_unpacklo_epi8(cr, zero),
    );
    let [r_hi, g_hi, b_hi] = ycbcr16_to_rgb(
        _mm256_unpackhi_epi8(y, zero),
        _mm256_unpackhi_epi8(cb, zero),
        _mm256_unpackhi_epi8(cr, zero),
    );
    // Unsigned saturation is the scalar clamp to [0, 255].
    [
        _mm256_packus_epi16(r_lo, r_hi),
        _mm256_packus_epi16(g_lo, g_hi),
        _mm256_packus_epi16(b_lo, b_hi),
    ]
}

/// Fractional `Cr` gain of red: `FIX_1_40200 = 2^16 + 26,345`.
const R_CR_FRACTION: i32 = FIX_1_40200 - (1 << 16);
/// Fractional `Cb` gain of blue: `FIX_1_77200 = 2 * 2^16 - 14,942`.
const B_CB_FRACTION: i32 = FIX_1_77200 - (2 << 16);
/// `vpmaddwd` gains for `(Cb, Cr)`: `FIX_0_34414` and `FIX_0_71414 - 2^16`.
const G_GAIN_PAIR: i32 = madd_pair(FIX_0_34414, FIX_0_71414 - (1 << 16));

const _: () = {
    // The halving trick below rounds by exactly 2^15.
    assert!(ROUND == 1 << 15);
    assert!(-32_768 < R_CR_FRACTION && R_CR_FRACTION < 32_768);
    assert!(-32_768 < B_CB_FRACTION && B_CB_FRACTION < 32_768);
};

/// `(2 * x * fraction) >> 16`, plus one, halved: `vpmulhw` floors the
/// product of the doubled value, and `floor((floor(2v) + 1) / 2)` equals
/// `floor(v + 1/2)`, so this is `(x * fraction + 2^15) >> 16` exactly.
#[inline]
#[target_feature(enable = "avx2")]
fn rounded_fraction(doubled: __m256i, fraction: i32) -> __m256i {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the fractional gains are asserted above to fit in i16"
    )]
    let fraction = fraction as i16;
    _mm256_srai_epi16::<1>(_mm256_add_epi16(
        _mm256_mulhi_epi16(doubled, _mm256_set1_epi16(fraction)),
        _mm256_set1_epi16(1),
    ))
}

/// 16 pixels of `ycbcr_to_rgb` in i16 lanes, before the clamp. Each
/// coefficient splits into a whole multiple of 2^16 (added exactly) and a
/// fraction, so every result equals the scalar 32-bit formula:
/// `R = Y + Cr + ((26,345 Cr + 2^15) >> 16)`,
/// `B = Y + 2 Cb + ((-14,942 Cb + 2^15) >> 16)` and
/// `G = Y - Cr - ((22,554 Cb - 18,734 Cr + 2^15) >> 16)`, with
/// `Cb`, `Cr` centred. All intermediates stay within [-600, 600].
#[inline]
#[target_feature(enable = "avx2")]
fn ycbcr16_to_rgb(y: __m256i, cb: __m256i, cr: __m256i) -> [__m256i; 3] {
    let center = _mm256_set1_epi16(128);
    let cb = _mm256_sub_epi16(cb, center);
    let cr = _mm256_sub_epi16(cr, center);
    let cb2 = _mm256_add_epi16(cb, cb);
    let cr2 = _mm256_add_epi16(cr, cr);
    let r = _mm256_add_epi16(
        _mm256_add_epi16(y, cr),
        rounded_fraction(cr2, R_CR_FRACTION),
    );
    let b = _mm256_add_epi16(
        _mm256_add_epi16(y, cb2),
        rounded_fraction(cb2, B_CB_FRACTION),
    );
    let gains = _mm256_set1_epi32(G_GAIN_PAIR);
    let round = _mm256_set1_epi32(ROUND);
    let g_lo = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpacklo_epi16(cb, cr), gains),
        round,
    ));
    let g_hi = _mm256_srai_epi32::<16>(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpackhi_epi16(cb, cr), gains),
        round,
    ));
    let g = _mm256_sub_epi16(_mm256_sub_epi16(y, cr), _mm256_packs_epi32(g_lo, g_hi));
    [r, g, b]
}

/// `vpshufb` mask placing channel `channel` of sixteen pixels into bytes
/// `16 * part .. 16 * part + 16` of their interleaved RGB.
const fn interleave_mask(part: usize, channel: usize) -> [i8; 16] {
    let mut mask = [-128i8; 16];
    let mut byte = 0;
    while byte < 16 {
        let index = 16 * part + byte;
        if index % 3 == channel {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "pixel indices within a 16-pixel lane are below 16"
            )]
            let pixel = (index / 3) as i8;
            mask[byte] = pixel;
        }
        byte += 1;
    }
    mask
}

const INTERLEAVE_MASKS: [[[i8; 16]; 3]; 3] = {
    let mut masks = [[[0; 16]; 3]; 3];
    let mut part = 0;
    while part < 3 {
        let mut channel = 0;
        while channel < 3 {
            masks[part][channel] = interleave_mask(part, channel);
            channel += 1;
        }
        part += 1;
    }
    masks
};

#[inline]
#[target_feature(enable = "avx2")]
fn lane_mask(mask: [i8; 16]) -> __m256i {
    _mm256_broadcastsi128_si256(_mm_setr_epi8(
        mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], mask[6], mask[7], mask[8], mask[9],
        mask[10], mask[11], mask[12], mask[13], mask[14], mask[15],
    ))
}

/// Interleave 32 pixels of `R`, `G` and `B` bytes into 96 bytes of RGB.
#[inline]
#[target_feature(enable = "avx2")]
fn store_rgb_chunk(dst: &mut [u8; LANES * 3], [r, g, b]: [__m256i; 3]) {
    // Part `k` of each 128-bit lane is bytes 16k..16k+16 of that lane's
    // sixteen pixels.
    let mut parts = [_mm256_setzero_si256(); 3];
    for (part, masks) in parts.iter_mut().zip(INTERLEAVE_MASKS) {
        *part = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_shuffle_epi8(r, lane_mask(masks[0])),
                _mm256_shuffle_epi8(g, lane_mask(masks[1])),
            ),
            _mm256_shuffle_epi8(b, lane_mask(masks[2])),
        );
    }
    let [first, second, third] = parts;
    let (out, _) = dst.as_chunks_mut::<32>();
    x86_memory::store_u8x32(
        &mut out[0],
        _mm256_permute2x128_si256::<0x20>(first, second),
    );
    x86_memory::store_u8x32(&mut out[1], _mm256_permute2x128_si256::<0x30>(third, first));
    x86_memory::store_u8x32(
        &mut out[2],
        _mm256_permute2x128_si256::<0x31>(second, third),
    );
}
