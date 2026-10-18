// SPDX-License-Identifier: MIT OR Apache-2.0

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, uint8x16_t, uint8x16x2_t, uint8x8_t, vaddq_s16, vaddq_u16, vcombine_s16,
    vcombine_u8, vdup_n_u8, vdupq_n_s16, vdupq_n_u16, vdupq_n_u8, vget_low_s16, vget_low_u8,
    vmlal_high_n_s16, vmlal_n_s16, vmovl_high_u8, vmovl_u8, vmull_high_n_s16, vmull_n_s16,
    vqmovn_u16, vqmovun_high_s16, vqmovun_s16, vqrdmulhq_n_s16, vreinterpretq_s16_u16,
    vrshrn_n_s32, vshlq_n_s16, vshrq_n_u16, vsubl_high_u8, vsubl_u8, vsubq_s16, vzip_u8, vzipq_u16,
    vzipq_u8,
};

use super::row_pair::{normalize_simd_row_pair, normalize_ycbcr_row};
use super::{scalar, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair, Rgb422Row};
use crate::color::upsample::{h2v2_fancy_sample, h2v2_fancy_sample_for_width};
use crate::color::ycbcr::ycbcr_to_rgb;
use crate::simd::neon_memory::{self, load_head_window, load_tail_window};

pub(crate) fn fill_rgb_row_from_gray(neon: fearless_simd::Neon, gray_row: &[u8], dst: &mut [u8]) {
    let width = gray_row.len().min(dst.len() / 3);
    let gray_row = &gray_row[..width];
    let dst = &mut dst[..width * 3];
    debug_assert_eq!(dst.len(), gray_row.len() * 3);
    fill_rgb_row_from_gray_kernel(neon, gray_row, dst);
}

fearless_simd::kernel! {
    fn fill_rgb_row_from_gray_kernel(neon: Neon, gray_row: &[u8], dst: &mut [u8]) {
        fill_rgb_row_from_gray_neon(gray_row, dst);
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_gray_neon(gray_row: &[u8], dst: &mut [u8]) {
    let (gray_chunks, gray_tail) = gray_row.as_chunks::<LANES>();
    let (rgb_chunks, rgb_tail) = dst.as_chunks_mut::<{ LANES * 3 }>();
    for (gray, rgb) in gray_chunks.iter().zip(rgb_chunks) {
        let g = neon_memory::load_u8x8(gray);
        neon_memory::store_rgb8(rgb, g, g, g);
    }
    if !gray_tail.is_empty() {
        scalar::fill_rgb_row_from_gray(gray_tail, rgb_tail);
    }
}

pub(crate) fn fill_rgb_row_from_rgb(
    neon: fearless_simd::Neon,
    r_row: &[u8],
    g_row: &[u8],
    b_row: &[u8],
    dst: &mut [u8],
) {
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
    fill_rgb_row_from_rgb_kernel(neon, r_row, g_row, b_row, dst);
}

fearless_simd::kernel! {
    fn fill_rgb_row_from_rgb_kernel(
        neon: Neon,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
        dst: &mut [u8],
    ) {
        fill_rgb_row_from_rgb_neon(r_row, g_row, b_row, dst);
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_rgb_neon(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    let (r_chunks, r_tail) = r_row.as_chunks::<LANES>();
    let (g_chunks, g_tail) = g_row.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b_row.as_chunks::<LANES>();
    let (rgb_chunks, rgb_tail) = dst.as_chunks_mut::<{ LANES * 3 }>();
    for (((r, g), b), rgb) in r_chunks.iter().zip(g_chunks).zip(b_chunks).zip(rgb_chunks) {
        let r = neon_memory::load_u8x8(r);
        let g = neon_memory::load_u8x8(g);
        let b = neon_memory::load_u8x8(b);
        neon_memory::store_rgb8(rgb, r, g, b);
    }
    if !r_tail.is_empty() {
        scalar::fill_rgb_row_from_rgb(r_tail, g_tail, b_tail, rgb_tail);
    }
}

const LANES: usize = 8;
const UPSAMPLED_LANES: usize = LANES * 2;
const FUSED_422_PIXELS: usize = UPSAMPLED_LANES * 2;

#[derive(Clone, Copy)]
struct Neon420PartialChunk {
    aligned_x: usize,
    src_skip: usize,
    copy_width: usize,
}

#[derive(Clone, Copy)]
struct Neon420TailChunk {
    sample_offset: usize,
    x: usize,
    chunk_width: usize,
    row_width: usize,
}

fn top_only_chroma(chroma: Rgb420ChromaRows<'_>) -> Rgb420ChromaRows<'_> {
    Rgb420ChromaRows::new(
        chroma.prev_cb,
        chroma.curr_cb,
        chroma.curr_cb,
        chroma.prev_cr,
        chroma.curr_cr,
        chroma.curr_cr,
    )
}

pub(crate) fn fill_rgb_row_from_ycbcr(
    neon: fearless_simd::Neon,
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    let (y_row, cb_row, cr_row, dst) = normalize_ycbcr_row(y_row, cb_row, cr_row, dst);
    debug_assert_eq!(y_row.len(), cb_row.len());
    debug_assert_eq!(y_row.len(), cr_row.len());
    debug_assert_eq!(dst.len(), y_row.len() * 3);
    fill_rgb_row_from_ycbcr_kernel(neon, y_row, cb_row, cr_row, dst);
}

fearless_simd::kernel! {
    fn fill_rgb_row_from_ycbcr_kernel(
        neon: Neon,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        dst: &mut [u8],
    ) {
        fill_rgb_row_from_ycbcr_neon(y_row, cb_row, cr_row, dst);
    }
}

#[cfg(test)]
pub(super) fn fill_rgb_row_from_ycbcr_for_test(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
) {
    let neon = fearless_simd::Level::new()
        .as_neon()
        .expect("AArch64 test host must provide NEON");
    fill_rgb_row_from_ycbcr(neon, y_row, cb_row, cr_row, dst);
}

pub(crate) fn fill_rgb_row_from_422(neon: fearless_simd::Neon, request: Rgb422Row<'_>) {
    let Rgb422Row { y, cb, cr, dst } = request;
    let width = y
        .len()
        .min(dst.len() / 3)
        .min(cb.len().min(cr.len()).saturating_mul(2));
    if width == 0 {
        return;
    }
    let chroma_width = width.div_ceil(2);
    fill_rgb_row_from_422_kernel(
        neon,
        &y[..width],
        &cb[..chroma_width],
        &cr[..chroma_width],
        &mut dst[..width * 3],
    );
}

fearless_simd::kernel! {
    fn fill_rgb_row_from_422_kernel(
        neon: Neon,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        dst: &mut [u8],
    ) {
        fill_rgb_row_from_422_neon(y, cb, cr, dst);
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::many_single_char_names,
    reason = "the hot color kernel uses standard y/cb/cr and r/g/b channel names"
)]
fn fill_rgb_row_from_422_neon(y: &[u8], cb: &[u8], cr: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(cb.len(), cr.len());
    debug_assert_eq!(y.len(), dst.len() / 3);
    debug_assert_eq!(cb.len(), y.len().div_ceil(2));
    if y.len() < FUSED_422_PIXELS + 1 {
        scalar::fill_rgb_row_from_422(Rgb422Row { y, cb, cr, dst });
        return;
    }

    let (r, g, b) = ycbcr_to_rgb(y[0], cb[0], cr[0]);
    dst[..3].copy_from_slice(&[r, g, b]);

    let mut sample = 0usize;
    let mut x = 1usize;
    while sample + UPSAMPLED_LANES < cb.len() && x + FUSED_422_PIXELS <= y.len() {
        let Some(y_chunk) = y[x..].first_chunk::<FUSED_422_PIXELS>() else {
            break;
        };
        let Some(cb_left) = cb[sample..].first_chunk::<UPSAMPLED_LANES>() else {
            break;
        };
        let Some(cb_right) = cb[sample + 1..].first_chunk::<UPSAMPLED_LANES>() else {
            break;
        };
        let Some(cr_left) = cr[sample..].first_chunk::<UPSAMPLED_LANES>() else {
            break;
        };
        let Some(cr_right) = cr[sample + 1..].first_chunk::<UPSAMPLED_LANES>() else {
            break;
        };
        let Some(dst_chunk) = dst[x * 3..].first_chunk_mut::<{ FUSED_422_PIXELS * 3 }>() else {
            break;
        };
        fill_rgb_row_from_422_chunk32_neon(
            y_chunk, cb_left, cb_right, cr_left, cr_right, dst_chunk,
        );
        sample += UPSAMPLED_LANES;
        x += FUSED_422_PIXELS;
    }

    // A regular chunk starts at output pixel 1, so power-of-two row widths
    // leave 31 pixels after the final non-overlapping vector. Re-run the last
    // 16 adjacent chroma pairs at the right edge when that is cheaper than a
    // long scalar tail. The overlap is bit-identical and leaves at most the
    // final replicated pixel of an even-width row for scalar handling.
    if y.len() - x >= UPSAMPLED_LANES {
        let tail_sample = cb.len() - (UPSAMPLED_LANES + 1);
        let tail_x = tail_sample * 2 + 1;
        let y_chunk = y[tail_x..]
            .first_chunk::<FUSED_422_PIXELS>()
            .expect("tail chunk fits the normalized visible row");
        let cb_left = cb[tail_sample..]
            .first_chunk::<UPSAMPLED_LANES>()
            .expect("tail chunk has 16 left chroma samples");
        let cb_right = cb[tail_sample + 1..]
            .first_chunk::<UPSAMPLED_LANES>()
            .expect("tail chunk has 16 right chroma samples");
        let cr_left = cr[tail_sample..]
            .first_chunk::<UPSAMPLED_LANES>()
            .expect("tail chunk has 16 left chroma samples");
        let cr_right = cr[tail_sample + 1..]
            .first_chunk::<UPSAMPLED_LANES>()
            .expect("tail chunk has 16 right chroma samples");
        let dst_chunk = dst[tail_x * 3..]
            .first_chunk_mut::<{ FUSED_422_PIXELS * 3 }>()
            .expect("tail RGB chunk fits the normalized destination");
        fill_rgb_row_from_422_chunk32_neon(
            y_chunk, cb_left, cb_right, cr_left, cr_right, dst_chunk,
        );
        x = tail_x + FUSED_422_PIXELS;
    }

    for (output_x, pixel) in dst[x * 3..].as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let output_x = output_x + x;
        let cb = scalar::h2v1_fancy_sample(cb, output_x);
        let cr = scalar::h2v1_fancy_sample(cr, output_x);
        let (r, g, b) = ycbcr_to_rgb(y[output_x], cb, cr);
        pixel.copy_from_slice(&[r, g, b]);
    }
}

pub(crate) fn fill_rgb_row_pair_from_420(neon: fearless_simd::Neon, request: Rgb420RowPair<'_>) {
    let Some(request) = normalize_simd_row_pair(request) else {
        return;
    };
    fill_rgb_row_pair_from_420_kernel(neon, request);
}

fearless_simd::kernel! {
    fn fill_rgb_row_pair_from_420_kernel(neon: Neon, request: Rgb420RowPair<'_>) {
        fill_rgb_row_pair_from_420_neon(request);
    }
}

pub(crate) fn fill_rgb_row_pair_from_420_cropped(
    neon: fearless_simd::Neon,
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
    if y_top.get(crop_start..crop_end).is_none() {
        return;
    }
    let y_bottom = y_bottom.and_then(|row| row.get(..));
    let prev_cb = &chroma.prev_cb[..chroma_width];
    let curr_cb = &chroma.curr_cb[..chroma_width];
    let next_cb = &chroma.next_cb[..chroma_width];
    let prev_cr = &chroma.prev_cr[..chroma_width];
    let curr_cr = &chroma.curr_cr[..chroma_width];
    let next_cr = &chroma.next_cr[..chroma_width];
    let dst_top = &mut dst_top[..width * 3];
    let dst_bottom = dst_bottom.and_then(|row| row.get_mut(..width * 3));
    debug_assert!(crop_end <= y_top.len());
    debug_assert_eq!(dst_top.len(), width * 3);
    debug_assert!(y_bottom.is_none_or(|row| crop_end <= row.len()));
    debug_assert!(dst_bottom.as_ref().is_none_or(|row| row.len() == width * 3));
    debug_assert_eq!(prev_cb.len(), curr_cb.len());
    debug_assert_eq!(prev_cb.len(), next_cb.len());
    debug_assert_eq!(prev_cr.len(), curr_cr.len());
    debug_assert_eq!(prev_cr.len(), next_cr.len());
    fill_rgb_row_pair_from_420_cropped_kernel(
        neon,
        Rgb420CroppedRowPair::new(
            Rgb420RowPair::new(
                y_top,
                y_bottom,
                Rgb420ChromaRows::new(prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr),
                dst_top,
                dst_bottom,
            ),
            Rgb420Crop::new(crop_start, crop_width),
        ),
    );
}

fearless_simd::kernel! {
    fn fill_rgb_row_pair_from_420_cropped_kernel(
        neon: Neon,
        request: Rgb420CroppedRowPair<'_>,
    ) {
        fill_rgb_row_pair_from_420_cropped_neon(request);
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_neon(request: Rgb420RowPair<'_>) {
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = request;
    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        {
            fill_rgb_row_pair_from_420_neon_dual(y_top, y_bottom, chroma, dst_top, dst_bottom);
        }
    } else {
        {
            fill_rgb_row_pair_from_420_neon_top_only(y_top, chroma, dst_top);
        }
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_cropped_neon(request: Rgb420CroppedRowPair<'_>) {
    let Rgb420CroppedRowPair { rows, crop } = request;
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = rows;
    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        {
            fill_rgb_row_pair_from_420_cropped_neon_dual(
                y_top, y_bottom, chroma, crop, dst_top, dst_bottom,
            );
        }
    } else {
        {
            fill_rgb_row_pair_from_420_cropped_neon_top_only(y_top, chroma, crop, dst_top);
        }
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_lines,
    reason = "the SIMD kernel mirrors one scalar 4:2:0 row-pair operation with ordered lane and edge repair"
)]
fn fill_rgb_row_pair_from_420_cropped_neon_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    crop: Rgb420Crop,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    let crop_start = crop.start;
    let crop_width = crop.width;
    let mut out_x = 0usize;
    if crop_width == 0 {
        return;
    }

    if crop_start == 0 {
        let prefix = crop_width.min(2);
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            Rgb420RowPair::new(
                y_top,
                Some(y_bottom),
                Rgb420ChromaRows::new(prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr),
                &mut dst_top[..prefix * 3],
                Some(&mut dst_bottom[..prefix * 3]),
            ),
            Rgb420Crop::new(crop_start, prefix),
        ));
        out_x = prefix;
    } else if !crop_start.is_multiple_of(2) {
        let aligned_x = crop_start - 1;
        let copy_width = crop_width.min(UPSAMPLED_LANES - 1);
        if copy_width >= LANES
            && can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), aligned_x)
        {
            {
                fill_rgb_row_pair_from_420_cropped_partial_chunk16_dual(
                    y_top,
                    y_bottom,
                    chroma,
                    Neon420PartialChunk {
                        aligned_x,
                        src_skip: 1,
                        copy_width,
                    },
                    &mut dst_top[..copy_width * 3],
                    &mut dst_bottom[..copy_width * 3],
                );
            }
            out_x = copy_width;
        } else {
            scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
                Rgb420RowPair::new(
                    y_top,
                    Some(y_bottom),
                    Rgb420ChromaRows::new(prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr),
                    &mut dst_top[..3],
                    Some(&mut dst_bottom[..3]),
                ),
                Rgb420Crop::new(crop_start, 1),
            ));
            out_x = 1;
        }
    }

    while out_x + UPSAMPLED_LANES <= crop_width {
        let x = crop_start + out_x;
        if !can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), x) {
            break;
        }

        {
            fill_rgb_row_pair_from_420_chunk16_interior_neon(
                &y_top[x..x + UPSAMPLED_LANES],
                &y_bottom[x..x + UPSAMPLED_LANES],
                chroma,
                x / 2,
                &mut dst_top[out_x * 3..(out_x + UPSAMPLED_LANES) * 3],
                &mut dst_bottom[out_x * 3..(out_x + UPSAMPLED_LANES) * 3],
            );
        }
        out_x += UPSAMPLED_LANES;
    }

    let remaining = crop_width - out_x;
    if remaining >= LANES {
        let x = crop_start + out_x;
        if can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), x) {
            {
                fill_rgb_row_pair_from_420_cropped_partial_chunk16_dual(
                    y_top,
                    y_bottom,
                    chroma,
                    Neon420PartialChunk {
                        aligned_x: x,
                        src_skip: 0,
                        copy_width: remaining,
                    },
                    &mut dst_top[out_x * 3..],
                    &mut dst_bottom[out_x * 3..],
                );
            }
            out_x = crop_width;
        }
    }

    if out_x < crop_width {
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            Rgb420RowPair::new(
                y_top,
                Some(y_bottom),
                Rgb420ChromaRows::new(prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr),
                &mut dst_top[out_x * 3..],
                Some(&mut dst_bottom[out_x * 3..]),
            ),
            Rgb420Crop::new(crop_start + out_x, crop_width - out_x),
        ));
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_cropped_neon_top_only(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    crop: Rgb420Crop,
    dst_top: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        prev_cr,
        curr_cr,
        ..
    } = chroma;
    let crop_start = crop.start;
    let crop_width = crop.width;
    let scalar_chroma = top_only_chroma(chroma);
    let mut out_x = 0usize;
    if crop_width == 0 {
        return;
    }

    if crop_start == 0 {
        let prefix = crop_width.min(2);
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            Rgb420RowPair::new(y_top, None, scalar_chroma, &mut dst_top[..prefix * 3], None),
            Rgb420Crop::new(crop_start, prefix),
        ));
        out_x = prefix;
    } else if !crop_start.is_multiple_of(2) {
        let aligned_x = crop_start - 1;
        let copy_width = crop_width.min(UPSAMPLED_LANES - 1);
        if copy_width >= LANES
            && can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), aligned_x)
        {
            {
                fill_rgb_row_pair_from_420_cropped_partial_chunk16_top_only(
                    y_top,
                    chroma,
                    Neon420PartialChunk {
                        aligned_x,
                        src_skip: 1,
                        copy_width,
                    },
                    &mut dst_top[..copy_width * 3],
                );
            }
            out_x = copy_width;
        } else {
            scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
                Rgb420RowPair::new(y_top, None, scalar_chroma, &mut dst_top[..3], None),
                Rgb420Crop::new(crop_start, 1),
            ));
            out_x = 1;
        }
    }

    while out_x + UPSAMPLED_LANES <= crop_width {
        let x = crop_start + out_x;
        if !can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), x) {
            break;
        }

        {
            fill_rgb_row_from_420_chunk16_interior_neon(
                &y_top[x..x + UPSAMPLED_LANES],
                prev_cb,
                curr_cb,
                prev_cr,
                curr_cr,
                x / 2,
                &mut dst_top[out_x * 3..(out_x + UPSAMPLED_LANES) * 3],
            );
        }
        out_x += UPSAMPLED_LANES;
    }

    let remaining = crop_width - out_x;
    if remaining >= LANES {
        let x = crop_start + out_x;
        if can_vectorize_cropped_420_chunk(y_top.len(), curr_cb.len(), x) {
            {
                fill_rgb_row_pair_from_420_cropped_partial_chunk16_top_only(
                    y_top,
                    chroma,
                    Neon420PartialChunk {
                        aligned_x: x,
                        src_skip: 0,
                        copy_width: remaining,
                    },
                    &mut dst_top[out_x * 3..],
                );
            }
            out_x = crop_width;
        }
    }

    if out_x < crop_width {
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            Rgb420RowPair::new(y_top, None, scalar_chroma, &mut dst_top[out_x * 3..], None),
            Rgb420Crop::new(crop_start + out_x, crop_width - out_x),
        ));
    }
}

fn can_vectorize_cropped_420_chunk(row_width: usize, chroma_width: usize, x: usize) -> bool {
    x.is_multiple_of(2)
        && x + UPSAMPLED_LANES <= row_width
        && can_vectorize_420_chunk(chroma_width, x / 2, UPSAMPLED_LANES)
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_cropped_partial_chunk16_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk: Neon420PartialChunk,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let Neon420PartialChunk {
        aligned_x,
        src_skip,
        copy_width,
    } = chunk;
    debug_assert!(src_skip + copy_width <= UPSAMPLED_LANES);
    debug_assert!(copy_width <= UPSAMPLED_LANES);
    let mut tmp_top = [0u8; UPSAMPLED_LANES * 3];
    let mut tmp_bottom = [0u8; UPSAMPLED_LANES * 3];
    {
        fill_rgb_row_pair_from_420_chunk16_interior_neon(
            &y_top[aligned_x..aligned_x + UPSAMPLED_LANES],
            &y_bottom[aligned_x..aligned_x + UPSAMPLED_LANES],
            chroma,
            aligned_x / 2,
            &mut tmp_top,
            &mut tmp_bottom,
        );
    }
    let src_start = src_skip * 3;
    let copy_len = copy_width * 3;
    dst_top[..copy_len].copy_from_slice(&tmp_top[src_start..src_start + copy_len]);
    dst_bottom[..copy_len].copy_from_slice(&tmp_bottom[src_start..src_start + copy_len]);
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_cropped_partial_chunk16_top_only(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk: Neon420PartialChunk,
    dst_top: &mut [u8],
) {
    let Neon420PartialChunk {
        aligned_x,
        src_skip,
        copy_width,
    } = chunk;
    debug_assert!(src_skip + copy_width <= UPSAMPLED_LANES);
    debug_assert!(copy_width <= UPSAMPLED_LANES);
    let mut tmp_top = [0u8; UPSAMPLED_LANES * 3];
    {
        fill_rgb_row_from_420_chunk16_interior_neon(
            &y_top[aligned_x..aligned_x + UPSAMPLED_LANES],
            chroma.prev_cb,
            chroma.curr_cb,
            chroma.prev_cr,
            chroma.curr_cr,
            aligned_x / 2,
            &mut tmp_top,
        );
    }
    let src_start = src_skip * 3;
    let copy_len = copy_width * 3;
    dst_top[..copy_len].copy_from_slice(&tmp_top[src_start..src_start + copy_len]);
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_lines,
    reason = "the SIMD kernel mirrors one scalar 4:2:0 row-pair operation with ordered lane and edge repair"
)]
fn fill_rgb_row_pair_from_420_neon_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let width = y_top.len();
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    let chroma_width = curr_cb.len();
    let mut sample = 0usize;

    while sample < chroma_width {
        let chunk_samples = (chroma_width - sample).min(LANES);
        let x = sample * 2;
        if x >= width {
            break;
        }
        let chunk_width = (width - x).min(chunk_samples * 2);

        if can_vectorize_420_chunk(chroma_width, sample, chunk_width) {
            {
                fill_rgb_row_pair_from_420_chunk16_interior_neon(
                    &y_top[x..x + UPSAMPLED_LANES],
                    &y_bottom[x..x + UPSAMPLED_LANES],
                    chroma,
                    sample,
                    &mut dst_top[x * 3..(x + UPSAMPLED_LANES) * 3],
                    &mut dst_bottom[x * 3..(x + UPSAMPLED_LANES) * 3],
                );
            }
            sample += chunk_samples;
            continue;
        }

        if sample == 0 {
            {
                fill_rgb_row_pair_from_420_edge_neon_dual(
                    y_top,
                    y_bottom,
                    chroma,
                    chunk_width,
                    dst_top,
                    dst_bottom,
                );
            }
        } else if can_use_tail_420_chunk(chroma_width, sample, chunk_width) {
            if chunk_width == 1 {
                record_420_dispatch_scalar_chunk();
                fill_last_420_pixel_dual(y_top, y_bottom, chroma, width, dst_top, dst_bottom);
                sample += chunk_samples;
                continue;
            }
            record_420_dispatch_neon_tail_chunk();
            // A 16k-1 row would otherwise convert 15 pixels through temporary
            // RGB arrays. Overlap the last full vector and repair one pixel.
            let overlap = chunk_width == UPSAMPLED_LANES - 1;
            let (tail_sample, tail_x, tail_width) = if overlap {
                let tail_x = width - (UPSAMPLED_LANES + 1);
                (tail_x / 2, tail_x, UPSAMPLED_LANES)
            } else {
                (sample, x, chunk_width)
            };
            {
                fill_rgb_row_pair_from_420_tail_neon_dual(
                    y_top,
                    y_bottom,
                    chroma,
                    Neon420TailChunk {
                        sample_offset: tail_sample,
                        x: tail_x,
                        chunk_width: tail_width,
                        row_width: width,
                    },
                    dst_top,
                    dst_bottom,
                );
            }
            if overlap {
                fill_last_420_pixel_dual(y_top, y_bottom, chroma, width, dst_top, dst_bottom);
            }
        } else {
            record_420_dispatch_scalar_chunk();
            let mut cb_top = [0u8; UPSAMPLED_LANES];
            let mut cb_bot = [0u8; UPSAMPLED_LANES];
            let mut cr_top = [0u8; UPSAMPLED_LANES];
            let mut cr_bot = [0u8; UPSAMPLED_LANES];

            {
                fill_upsampled_420_chunk(
                    prev_cb,
                    curr_cb,
                    sample,
                    width,
                    &mut cb_top[..chunk_width],
                );
                fill_upsampled_420_chunk(
                    next_cb,
                    curr_cb,
                    sample,
                    width,
                    &mut cb_bot[..chunk_width],
                );
                fill_upsampled_420_chunk(
                    prev_cr,
                    curr_cr,
                    sample,
                    width,
                    &mut cr_top[..chunk_width],
                );
                fill_upsampled_420_chunk(
                    next_cr,
                    curr_cr,
                    sample,
                    width,
                    &mut cr_bot[..chunk_width],
                );
                fill_rgb_row_from_ycbcr_neon(
                    &y_top[x..x + chunk_width],
                    &cb_top[..chunk_width],
                    &cr_top[..chunk_width],
                    &mut dst_top[x * 3..(x + chunk_width) * 3],
                );
                fill_rgb_row_from_ycbcr_neon(
                    &y_bottom[x..x + chunk_width],
                    &cb_bot[..chunk_width],
                    &cr_bot[..chunk_width],
                    &mut dst_bottom[x * 3..(x + chunk_width) * 3],
                );
            }
        }

        sample += chunk_samples;
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_neon_top_only(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    dst_top: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        prev_cr,
        curr_cr,
        ..
    } = chroma;
    let width = y_top.len();
    let chroma_width = curr_cb.len();
    let mut sample = 0usize;

    while sample < chroma_width {
        let chunk_samples = (chroma_width - sample).min(LANES);
        let x = sample * 2;
        if x >= width {
            break;
        }
        let chunk_width = (width - x).min(chunk_samples * 2);

        if can_vectorize_420_chunk(chroma_width, sample, chunk_width) {
            {
                fill_rgb_row_from_420_chunk16_interior_neon(
                    &y_top[x..x + UPSAMPLED_LANES],
                    prev_cb,
                    curr_cb,
                    prev_cr,
                    curr_cr,
                    sample,
                    &mut dst_top[x * 3..(x + UPSAMPLED_LANES) * 3],
                );
            }
            sample += chunk_samples;
            continue;
        }

        if sample == 0 {
            {
                fill_rgb_row_pair_from_420_edge_neon_top_only(y_top, chroma, chunk_width, dst_top);
            }
        } else if can_use_tail_420_chunk(chroma_width, sample, chunk_width) {
            if chunk_width == 1 {
                record_420_dispatch_scalar_chunk();
                fill_last_420_pixel_top(y_top, chroma, width, dst_top);
                sample += chunk_samples;
                continue;
            }
            record_420_dispatch_neon_tail_chunk();
            // A 16k-1 row would otherwise convert 15 pixels through a temporary
            // RGB array. Overlap the last full vector and repair one pixel.
            let overlap = chunk_width == UPSAMPLED_LANES - 1;
            let (tail_sample, tail_x, tail_width) = if overlap {
                let tail_x = width - (UPSAMPLED_LANES + 1);
                (tail_x / 2, tail_x, UPSAMPLED_LANES)
            } else {
                (sample, x, chunk_width)
            };
            {
                fill_rgb_row_pair_from_420_tail_neon_top_only(
                    y_top,
                    chroma,
                    Neon420TailChunk {
                        sample_offset: tail_sample,
                        x: tail_x,
                        chunk_width: tail_width,
                        row_width: width,
                    },
                    dst_top,
                );
            }
            if overlap {
                fill_last_420_pixel_top(y_top, chroma, width, dst_top);
            }
        } else {
            record_420_dispatch_scalar_chunk();
            let mut cb_top = [0u8; UPSAMPLED_LANES];
            let mut cr_top = [0u8; UPSAMPLED_LANES];
            {
                fill_upsampled_420_chunk(
                    prev_cb,
                    curr_cb,
                    sample,
                    width,
                    &mut cb_top[..chunk_width],
                );
                fill_upsampled_420_chunk(
                    prev_cr,
                    curr_cr,
                    sample,
                    width,
                    &mut cr_top[..chunk_width],
                );
                fill_rgb_row_from_ycbcr_neon(
                    &y_top[x..x + chunk_width],
                    &cb_top[..chunk_width],
                    &cr_top[..chunk_width],
                    &mut dst_top[x * 3..(x + chunk_width) * 3],
                );
            }
        }

        sample += chunk_samples;
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_edge_neon_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk_width: usize,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    let y_top_tail = load_tail_window(y_top, 0, chunk_width);
    let y_bottom_tail = load_tail_window(y_bottom, 0, chunk_width);
    let prev_cb_head = load_head_window(prev_cb, TAIL_WINDOW);
    let curr_cb_head = load_head_window(curr_cb, TAIL_WINDOW);
    let next_cb_head = load_head_window(next_cb, TAIL_WINDOW);
    let prev_cr_head = load_head_window(prev_cr, TAIL_WINDOW);
    let curr_cr_head = load_head_window(curr_cr, TAIL_WINDOW);
    let next_cr_head = load_head_window(next_cr, TAIL_WINDOW);

    let (cb_top, cb_bottom) =
        { upsampled_420_chunk16_pair_u16(&prev_cb_head, &curr_cb_head, &next_cb_head, 1) };
    let (cr_top, cr_bottom) =
        { upsampled_420_chunk16_pair_u16(&prev_cr_head, &curr_cr_head, &next_cr_head, 1) };

    let y_top_lo = { load_eight(&y_top_tail, 0) };
    let y_top_hi = { load_eight(&y_top_tail, LANES) };
    let y_bottom_lo = { load_eight(&y_bottom_tail, 0) };
    let y_bottom_hi = { load_eight(&y_bottom_tail, LANES) };

    if chunk_width == UPSAMPLED_LANES {
        {
            fill_chunk_from_vectors_u16(y_top_lo, cb_top.0, cr_top.0, &mut dst_top[..LANES * 3]);
            fill_chunk_from_vectors_u16(
                y_top_hi,
                cb_top.1,
                cr_top.1,
                &mut dst_top[LANES * 3..UPSAMPLED_LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_lo,
                cb_bottom.0,
                cr_bottom.0,
                &mut dst_bottom[..LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_hi,
                cb_bottom.1,
                cr_bottom.1,
                &mut dst_bottom[LANES * 3..UPSAMPLED_LANES * 3],
            );
        }
    } else {
        let mut rgb_top = [0u8; UPSAMPLED_LANES * 3];
        let mut rgb_bottom = [0u8; UPSAMPLED_LANES * 3];
        {
            fill_chunk_from_vectors_u16(y_top_lo, cb_top.0, cr_top.0, &mut rgb_top[..LANES * 3]);
            fill_chunk_from_vectors_u16(y_top_hi, cb_top.1, cr_top.1, &mut rgb_top[LANES * 3..]);
            fill_chunk_from_vectors_u16(
                y_bottom_lo,
                cb_bottom.0,
                cr_bottom.0,
                &mut rgb_bottom[..LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_hi,
                cb_bottom.1,
                cr_bottom.1,
                &mut rgb_bottom[LANES * 3..],
            );
        }
        dst_top[..chunk_width * 3].copy_from_slice(&rgb_top[..chunk_width * 3]);
        dst_bottom[..chunk_width * 3].copy_from_slice(&rgb_bottom[..chunk_width * 3]);
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_edge_neon_top_only(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk_width: usize,
    dst_top: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        prev_cr,
        curr_cr,
        ..
    } = chroma;
    let y_top_tail = load_tail_window(y_top, 0, chunk_width);
    let prev_cb_head = load_head_window(prev_cb, TAIL_WINDOW);
    let curr_cb_head = load_head_window(curr_cb, TAIL_WINDOW);
    let prev_cr_head = load_head_window(prev_cr, TAIL_WINDOW);
    let curr_cr_head = load_head_window(curr_cr, TAIL_WINDOW);

    let cb = { upsampled_420_chunk16_u16(&prev_cb_head, &curr_cb_head, 1) };
    let cr = { upsampled_420_chunk16_u16(&prev_cr_head, &curr_cr_head, 1) };
    let y_lo = { load_eight(&y_top_tail, 0) };
    let y_hi = { load_eight(&y_top_tail, LANES) };

    if chunk_width == UPSAMPLED_LANES {
        {
            fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut dst_top[..LANES * 3]);
            fill_chunk_from_vectors_u16(
                y_hi,
                cb.1,
                cr.1,
                &mut dst_top[LANES * 3..UPSAMPLED_LANES * 3],
            );
        }
    } else {
        let mut rgb = [0u8; UPSAMPLED_LANES * 3];
        {
            fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut rgb[..LANES * 3]);
            fill_chunk_from_vectors_u16(y_hi, cb.1, cr.1, &mut rgb[LANES * 3..]);
        }
        dst_top[..chunk_width * 3].copy_from_slice(&rgb[..chunk_width * 3]);
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_lines,
    reason = "the SIMD tail kernel keeps lane extraction and scalar edge repair in codegen order"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "NEON tail chroma sums are shifted into the u8 sample range before scalar edge repair"
)]
fn fill_rgb_row_pair_from_420_tail_neon_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk: Neon420TailChunk,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    let Neon420TailChunk {
        sample_offset,
        x,
        chunk_width,
        row_width: width,
    } = chunk;
    let y_top_tail = load_tail_window(y_top, x, chunk_width);
    let y_bottom_tail = load_tail_window(y_bottom, x, chunk_width);
    let prev_cb_tail = load_tail_window(prev_cb, sample_offset - 1, TAIL_WINDOW);
    let curr_cb_tail = load_tail_window(curr_cb, sample_offset - 1, TAIL_WINDOW);
    let next_cb_tail = load_tail_window(next_cb, sample_offset - 1, TAIL_WINDOW);
    let prev_cr_tail = load_tail_window(prev_cr, sample_offset - 1, TAIL_WINDOW);
    let curr_cr_tail = load_tail_window(curr_cr, sample_offset - 1, TAIL_WINDOW);
    let next_cr_tail = load_tail_window(next_cr, sample_offset - 1, TAIL_WINDOW);

    let (cb_top, cb_bottom) =
        { upsampled_420_chunk16_pair_u16(&prev_cb_tail, &curr_cb_tail, &next_cb_tail, 1) };
    let (cr_top, cr_bottom) =
        { upsampled_420_chunk16_pair_u16(&prev_cr_tail, &curr_cr_tail, &next_cr_tail, 1) };

    let y_top_lo = { load_eight(&y_top_tail, 0) };
    let y_top_hi = { load_eight(&y_top_tail, LANES) };
    let y_bottom_lo = { load_eight(&y_bottom_tail, 0) };
    let y_bottom_hi = { load_eight(&y_bottom_tail, LANES) };

    if chunk_width == UPSAMPLED_LANES {
        {
            fill_chunk_from_vectors_u16(
                y_top_lo,
                cb_top.0,
                cr_top.0,
                &mut dst_top[x * 3..x * 3 + LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_top_hi,
                cb_top.1,
                cr_top.1,
                &mut dst_top[x * 3 + LANES * 3..x * 3 + UPSAMPLED_LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_lo,
                cb_bottom.0,
                cr_bottom.0,
                &mut dst_bottom[x * 3..x * 3 + LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_hi,
                cb_bottom.1,
                cr_bottom.1,
                &mut dst_bottom[x * 3 + LANES * 3..x * 3 + UPSAMPLED_LANES * 3],
            );
        }

        if width.is_multiple_of(2) {
            let last = width - 1;
            let sample = curr_cb.len() - 1;
            let top_cb =
                ((u32::from(prev_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let top_cr =
                ((u32::from(prev_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let bottom_cb =
                ((u32::from(next_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let bottom_cr =
                ((u32::from(next_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let (r_top, g_top, b_top) = ycbcr_to_rgb(y_top[last], top_cb as u8, top_cr as u8);
            let (r_bottom, g_bottom, b_bottom) =
                ycbcr_to_rgb(y_bottom[last], bottom_cb as u8, bottom_cr as u8);
            dst_top[last * 3..last * 3 + 3].copy_from_slice(&[r_top, g_top, b_top]);
            dst_bottom[last * 3..last * 3 + 3].copy_from_slice(&[r_bottom, g_bottom, b_bottom]);
        }
    } else {
        let mut rgb_top = [0u8; UPSAMPLED_LANES * 3];
        let mut rgb_bottom = [0u8; UPSAMPLED_LANES * 3];
        {
            fill_chunk_from_vectors_u16(y_top_lo, cb_top.0, cr_top.0, &mut rgb_top[..LANES * 3]);
            fill_chunk_from_vectors_u16(y_top_hi, cb_top.1, cr_top.1, &mut rgb_top[LANES * 3..]);
            fill_chunk_from_vectors_u16(
                y_bottom_lo,
                cb_bottom.0,
                cr_bottom.0,
                &mut rgb_bottom[..LANES * 3],
            );
            fill_chunk_from_vectors_u16(
                y_bottom_hi,
                cb_bottom.1,
                cr_bottom.1,
                &mut rgb_bottom[LANES * 3..],
            );
        }

        if width.is_multiple_of(2) {
            let last = width - 1;
            let sample = curr_cb.len() - 1;
            let top_cb =
                ((u32::from(prev_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let top_cr =
                ((u32::from(prev_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let bottom_cb =
                ((u32::from(next_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let bottom_cr =
                ((u32::from(next_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let (r_top, g_top, b_top) = ycbcr_to_rgb(y_top[last], top_cb as u8, top_cr as u8);
            let (r_bottom, g_bottom, b_bottom) =
                ycbcr_to_rgb(y_bottom[last], bottom_cb as u8, bottom_cr as u8);
            rgb_top[(chunk_width - 1) * 3..chunk_width * 3].copy_from_slice(&[r_top, g_top, b_top]);
            rgb_bottom[(chunk_width - 1) * 3..chunk_width * 3]
                .copy_from_slice(&[r_bottom, g_bottom, b_bottom]);
        }

        dst_top[x * 3..x * 3 + chunk_width * 3].copy_from_slice(&rgb_top[..chunk_width * 3]);
        dst_bottom[x * 3..x * 3 + chunk_width * 3].copy_from_slice(&rgb_bottom[..chunk_width * 3]);
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "NEON tail chroma sums are shifted into the u8 sample range before scalar edge repair"
)]
fn fill_rgb_row_pair_from_420_tail_neon_top_only(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    chunk: Neon420TailChunk,
    dst_top: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        prev_cr,
        curr_cr,
        ..
    } = chroma;
    let Neon420TailChunk {
        sample_offset,
        x,
        chunk_width,
        row_width: width,
    } = chunk;
    let y_top_tail = load_tail_window(y_top, x, chunk_width);
    let prev_cb_tail = load_tail_window(prev_cb, sample_offset - 1, TAIL_WINDOW);
    let curr_cb_tail = load_tail_window(curr_cb, sample_offset - 1, TAIL_WINDOW);
    let prev_cr_tail = load_tail_window(prev_cr, sample_offset - 1, TAIL_WINDOW);
    let curr_cr_tail = load_tail_window(curr_cr, sample_offset - 1, TAIL_WINDOW);

    let cb = { upsampled_420_chunk16_u16(&prev_cb_tail, &curr_cb_tail, 1) };
    let cr = { upsampled_420_chunk16_u16(&prev_cr_tail, &curr_cr_tail, 1) };
    let y_lo = { load_eight(&y_top_tail, 0) };
    let y_hi = { load_eight(&y_top_tail, LANES) };

    if chunk_width == UPSAMPLED_LANES {
        {
            fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut dst_top[x * 3..x * 3 + LANES * 3]);
            fill_chunk_from_vectors_u16(
                y_hi,
                cb.1,
                cr.1,
                &mut dst_top[x * 3 + LANES * 3..x * 3 + UPSAMPLED_LANES * 3],
            );
        }

        if width.is_multiple_of(2) {
            let last = width - 1;
            let sample = curr_cb.len() - 1;
            let cb_last =
                ((u32::from(prev_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let cr_last =
                ((u32::from(prev_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let (r, g, b) = ycbcr_to_rgb(y_top[last], cb_last as u8, cr_last as u8);
            dst_top[last * 3..last * 3 + 3].copy_from_slice(&[r, g, b]);
        }
    } else {
        let mut rgb = [0u8; UPSAMPLED_LANES * 3];
        {
            fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut rgb[..LANES * 3]);
            fill_chunk_from_vectors_u16(y_hi, cb.1, cr.1, &mut rgb[LANES * 3..]);
        }

        if width.is_multiple_of(2) {
            let last = width - 1;
            let sample = curr_cb.len() - 1;
            let cb_last =
                ((u32::from(prev_cb[sample]) + 3 * u32::from(curr_cb[sample])) * 4 + 7) >> 4;
            let cr_last =
                ((u32::from(prev_cr[sample]) + 3 * u32::from(curr_cr[sample])) * 4 + 7) >> 4;
            let (r, g, b) = ycbcr_to_rgb(y_top[last], cb_last as u8, cr_last as u8);
            rgb[(chunk_width - 1) * 3..chunk_width * 3].copy_from_slice(&[r, g, b]);
        }

        dst_top[x * 3..x * 3 + chunk_width * 3].copy_from_slice(&rgb[..chunk_width * 3]);
    }
}

const TAIL_WINDOW: usize = LANES + 2;

fn can_use_tail_420_chunk(chroma_width: usize, sample_offset: usize, out_len: usize) -> bool {
    out_len <= UPSAMPLED_LANES && sample_offset > 0 && sample_offset + LANES >= chroma_width
}

#[expect(
    clippy::many_single_char_names,
    reason = "the color helper uses standard y/cb/cr and r/g/b channel names"
)]
fn last_420_pixel(
    y: &[u8],
    near_cb: &[u8],
    curr_cb: &[u8],
    near_cr: &[u8],
    curr_cr: &[u8],
    width: usize,
) -> [u8; 3] {
    let x = width - 1;
    let cb = h2v2_fancy_sample(near_cb, curr_cb, x);
    let cr = h2v2_fancy_sample(near_cr, curr_cr, x);
    let (r, g, b) = ycbcr_to_rgb(y[x], cb, cr);
    [r, g, b]
}

fn fill_last_420_pixel_dual(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    width: usize,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let x = width - 1;
    let dst = x * 3..x * 3 + 3;
    dst_top[dst.clone()].copy_from_slice(&last_420_pixel(
        y_top,
        chroma.prev_cb,
        chroma.curr_cb,
        chroma.prev_cr,
        chroma.curr_cr,
        width,
    ));
    dst_bottom[dst].copy_from_slice(&last_420_pixel(
        y_bottom,
        chroma.next_cb,
        chroma.curr_cb,
        chroma.next_cr,
        chroma.curr_cr,
        width,
    ));
}

fn fill_last_420_pixel_top(
    y_top: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    width: usize,
    dst_top: &mut [u8],
) {
    let x = width - 1;
    dst_top[x * 3..x * 3 + 3].copy_from_slice(&last_420_pixel(
        y_top,
        chroma.prev_cb,
        chroma.curr_cb,
        chroma.prev_cr,
        chroma.curr_cr,
        width,
    ));
}

fn record_420_dispatch_neon_tail_chunk() {
    #[cfg(feature = "bench-internals")]
    crate::bench_support::record_420_dispatch_neon_tail_chunk();
}

fn record_420_dispatch_scalar_chunk() {
    #[cfg(feature = "bench-internals")]
    crate::bench_support::record_420_dispatch_scalar_chunk();
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_420_chunk16_interior_neon(
    y_row: &[u8],
    near_cb: &[u8],
    curr_cb: &[u8],
    near_cr: &[u8],
    curr_cr: &[u8],
    sample_offset: usize,
    dst: &mut [u8],
) {
    debug_assert_eq!(y_row.len(), UPSAMPLED_LANES);
    debug_assert_eq!(dst.len(), UPSAMPLED_LANES * 3);

    let cb = { upsampled_420_chunk16_u16(near_cb, curr_cb, sample_offset) };
    let cr = { upsampled_420_chunk16_u16(near_cr, curr_cr, sample_offset) };
    let y_lo = { load_eight(y_row, 0) };
    let y_hi = { load_eight(y_row, LANES) };
    {
        fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut dst[..LANES * 3]);
        fill_chunk_from_vectors_u16(y_hi, cb.1, cr.1, &mut dst[LANES * 3..]);
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_pair_from_420_chunk16_interior_neon(
    y_top: &[u8],
    y_bottom: &[u8],
    chroma: Rgb420ChromaRows<'_>,
    sample_offset: usize,
    dst_top: &mut [u8],
    dst_bottom: &mut [u8],
) {
    let Rgb420ChromaRows {
        prev_cb,
        curr_cb,
        next_cb,
        prev_cr,
        curr_cr,
        next_cr,
    } = chroma;
    debug_assert_eq!(y_top.len(), UPSAMPLED_LANES);
    debug_assert_eq!(y_bottom.len(), UPSAMPLED_LANES);
    debug_assert_eq!(dst_top.len(), UPSAMPLED_LANES * 3);
    debug_assert_eq!(dst_bottom.len(), UPSAMPLED_LANES * 3);

    let (cb_top, cb_bottom) =
        { upsampled_420_chunk16_pair_u16(prev_cb, curr_cb, next_cb, sample_offset) };
    let (cr_top, cr_bottom) =
        { upsampled_420_chunk16_pair_u16(prev_cr, curr_cr, next_cr, sample_offset) };
    let y_top_lo = { load_eight(y_top, 0) };
    let y_top_hi = { load_eight(y_top, LANES) };
    let y_bottom_lo = { load_eight(y_bottom, 0) };
    let y_bottom_hi = { load_eight(y_bottom, LANES) };

    {
        fill_chunk_from_vectors_u16(y_top_lo, cb_top.0, cr_top.0, &mut dst_top[..LANES * 3]);
        fill_chunk_from_vectors_u16(y_top_hi, cb_top.1, cr_top.1, &mut dst_top[LANES * 3..]);
        fill_chunk_from_vectors_u16(
            y_bottom_lo,
            cb_bottom.0,
            cr_bottom.0,
            &mut dst_bottom[..LANES * 3],
        );
        fill_chunk_from_vectors_u16(
            y_bottom_hi,
            cb_bottom.1,
            cr_bottom.1,
            &mut dst_bottom[LANES * 3..],
        );
    }
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_ycbcr_neon(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    let (y_chunks, y_tail) = y_row.as_chunks::<UPSAMPLED_LANES>();
    let (cb_chunks, cb_tail) = cb_row.as_chunks::<UPSAMPLED_LANES>();
    let (cr_chunks, cr_tail) = cr_row.as_chunks::<UPSAMPLED_LANES>();
    let (dst_chunks, dst_tail) = dst.as_chunks_mut::<{ UPSAMPLED_LANES * 3 }>();

    for (((y, cb), cr), dst) in y_chunks
        .iter()
        .zip(cb_chunks)
        .zip(cr_chunks)
        .zip(dst_chunks)
    {
        fill_rgb_row_from_ycbcr_chunk16_neon(y, cb, cr, dst);
    }

    let (y_chunks, y_tail) = y_tail.as_chunks::<LANES>();
    let (cb_chunks, cb_tail) = cb_tail.as_chunks::<LANES>();
    let (cr_chunks, cr_tail) = cr_tail.as_chunks::<LANES>();
    let (dst_chunks, dst_tail) = dst_tail.as_chunks_mut::<{ LANES * 3 }>();

    for (((y, cb), cr), dst) in y_chunks
        .iter()
        .zip(cb_chunks)
        .zip(cr_chunks)
        .zip(dst_chunks)
    {
        fill_chunk(y, cb, cr, dst);
    }

    if !y_tail.is_empty() {
        scalar::fill_rgb_row_from_ycbcr(y_tail, cb_tail, cr_tail, dst_tail);
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "array references preserve the zero-copy fixed-size SIMD memory boundary"
)]
fn fill_chunk(y: &[u8; LANES], cb: &[u8; LANES], cr: &[u8; LANES], dst: &mut [u8; LANES * 3]) {
    fill_chunk_from_vectors(
        neon_memory::load_u8x8(y),
        neon_memory::load_u8x8(cb),
        neon_memory::load_u8x8(cr),
        dst,
    );
}

#[target_feature(enable = "neon")]
fn fill_chunk_from_vectors(y: uint8x8_t, cb: uint8x8_t, cr: uint8x8_t, dst_chunk: &mut [u8]) {
    debug_assert_eq!(dst_chunk.len(), LANES * 3);
    let cb16 = vmovl_u8(cb);
    let cr16 = vmovl_u8(cr);
    fill_chunk_from_vectors_u16(y, cb16, cr16, dst_chunk);
}

// Q14/Q15 constants from libjpeg-turbo's NEON converter. Across the complete
// centered 8-bit chroma domain (-128..=127), these produce the same rounded
// integer offsets as the scalar Q16 conversion.
const R_CR_Q14: i16 = 22_971;
const G_CB_Q15: i16 = -11_277;
const G_CR_Q15: i16 = -23_401;
const B_CB_Q14: i16 = 29_033;

/// Convert eight pixels whose chroma lanes hold 8-bit sample values.
#[target_feature(enable = "neon")]
fn fill_chunk_from_vectors_u16(y: uint8x8_t, cb: uint16x8_t, cr: uint16x8_t, dst_chunk: &mut [u8]) {
    let Some(dst) = dst_chunk.first_chunk_mut::<{ LANES * 3 }>() else {
        return;
    };
    let bias = vdupq_n_s16(128);
    let (r, g, b) = ycbcr_lanes_to_rgb(
        vreinterpretq_s16_u16(vmovl_u8(y)),
        vsubq_s16(vreinterpretq_s16_u16(cb), bias),
        vsubq_s16(vreinterpretq_s16_u16(cr), bias),
    );
    neon_memory::store_rgb8(dst, vqmovun_s16(r), vqmovun_s16(g), vqmovun_s16(b));
}

#[target_feature(enable = "neon")]
fn upsample_422_pairs16(left: &[u8; 16], right: &[u8; 16]) -> uint8x16x2_t {
    let left = neon_memory::load_u8x16(left);
    let right = neon_memory::load_u8x16(right);

    let left_lo = vmovl_u8(vget_low_u8(left));
    let left_hi = vmovl_high_u8(left);
    let right_lo = vmovl_u8(vget_low_u8(right));
    let right_hi = vmovl_high_u8(right);
    let three_left_lo = vaddq_u16(left_lo, vaddq_u16(left_lo, left_lo));
    let three_left_hi = vaddq_u16(left_hi, vaddq_u16(left_hi, left_hi));
    let three_right_lo = vaddq_u16(right_lo, vaddq_u16(right_lo, right_lo));
    let three_right_hi = vaddq_u16(right_hi, vaddq_u16(right_hi, right_hi));

    let odd_lo = vshrq_n_u16(
        vaddq_u16(vaddq_u16(three_left_lo, right_lo), vdupq_n_u16(2)),
        2,
    );
    let odd_hi = vshrq_n_u16(
        vaddq_u16(vaddq_u16(three_left_hi, right_hi), vdupq_n_u16(2)),
        2,
    );
    let even_lo = vshrq_n_u16(
        vaddq_u16(vaddq_u16(three_right_lo, left_lo), vdupq_n_u16(1)),
        2,
    );
    let even_hi = vshrq_n_u16(
        vaddq_u16(vaddq_u16(three_right_hi, left_hi), vdupq_n_u16(1)),
        2,
    );

    let odd = vcombine_u8(vqmovn_u16(odd_lo), vqmovn_u16(odd_hi));
    let even = vcombine_u8(vqmovn_u16(even_lo), vqmovn_u16(even_hi));
    vzipq_u8(odd, even)
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_422_chunk32_neon(
    y: &[u8; FUSED_422_PIXELS],
    cb_left: &[u8; UPSAMPLED_LANES],
    cb_right: &[u8; UPSAMPLED_LANES],
    cr_left: &[u8; UPSAMPLED_LANES],
    cr_right: &[u8; UPSAMPLED_LANES],
    dst: &mut [u8; FUSED_422_PIXELS * 3],
) {
    let cb = upsample_422_pairs16(cb_left, cb_right);
    let cr = upsample_422_pairs16(cr_left, cr_right);
    fill_rgb_chunk16_vectors(
        neon_memory::load_u8x16(
            y.first_chunk::<UPSAMPLED_LANES>()
                .expect("fixed first half"),
        ),
        cb.0,
        cr.0,
        dst.first_chunk_mut::<{ UPSAMPLED_LANES * 3 }>()
            .expect("fixed first RGB half"),
    );
    fill_rgb_chunk16_vectors(
        neon_memory::load_u8x16(
            y[UPSAMPLED_LANES..]
                .first_chunk::<UPSAMPLED_LANES>()
                .expect("fixed second half"),
        ),
        cb.1,
        cr.1,
        dst[UPSAMPLED_LANES * 3..]
            .first_chunk_mut::<{ UPSAMPLED_LANES * 3 }>()
            .expect("fixed second RGB half"),
    );
}

/// `ycbcr_to_rgb` on eight lanes of luma and centered chroma, before the
/// final unsigned saturation.
#[target_feature(enable = "neon")]
fn ycbcr_lanes_to_rgb(
    y: int16x8_t,
    cb: int16x8_t,
    cr: int16x8_t,
) -> (int16x8_t, int16x8_t, int16x8_t) {
    let r = vaddq_s16(y, vqrdmulhq_n_s16(vshlq_n_s16::<1>(cr), R_CR_Q14));

    let g_low = vmlal_n_s16(
        vmull_n_s16(vget_low_s16(cb), G_CB_Q15),
        vget_low_s16(cr),
        G_CR_Q15,
    );
    let g_high = vmlal_high_n_s16(vmull_high_n_s16(cb, G_CB_Q15), cr, G_CR_Q15);
    let g = vaddq_s16(
        y,
        vcombine_s16(vrshrn_n_s32::<15>(g_low), vrshrn_n_s32::<15>(g_high)),
    );

    let b = vaddq_s16(y, vqrdmulhq_n_s16(vshlq_n_s16::<1>(cb), B_CB_Q14));
    (r, g, b)
}
/// Sixteen pixels per iteration so loads and the interleaving `st3` store use
/// full 128-bit registers; two 8-pixel halves would issue twice the
/// multi-structure store µops.
#[target_feature(enable = "neon")]
fn fill_rgb_row_from_ycbcr_chunk16_neon(
    y: &[u8; UPSAMPLED_LANES],
    cb: &[u8; UPSAMPLED_LANES],
    cr: &[u8; UPSAMPLED_LANES],
    dst: &mut [u8; UPSAMPLED_LANES * 3],
) {
    let y = neon_memory::load_u8x16(y);
    let cb = neon_memory::load_u8x16(cb);
    let cr = neon_memory::load_u8x16(cr);
    fill_rgb_chunk16_vectors(y, cb, cr, dst);
}

#[target_feature(enable = "neon")]
fn fill_rgb_chunk16_vectors(
    y: uint8x16_t,
    cb: uint8x16_t,
    cr: uint8x16_t,
    dst: &mut [u8; UPSAMPLED_LANES * 3],
) {
    let bias = vdup_n_u8(128);
    // Widening subtraction wraps in `u16`; reinterpreting as `i16` yields the
    // exact centered chroma for 8-bit samples.
    let (r_lo, g_lo, b_lo) = ycbcr_lanes_to_rgb(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y))),
        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(cb), bias)),
        vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(cr), bias)),
    );
    let (r_hi, g_hi, b_hi) = ycbcr_lanes_to_rgb(
        vreinterpretq_s16_u16(vmovl_high_u8(y)),
        vreinterpretq_s16_u16(vsubl_high_u8(cb, vdupq_n_u8(128))),
        vreinterpretq_s16_u16(vsubl_high_u8(cr, vdupq_n_u8(128))),
    );
    neon_memory::store_rgb8x16(
        dst,
        vqmovun_high_s16(vqmovun_s16(r_lo), r_hi),
        vqmovun_high_s16(vqmovun_s16(g_lo), g_hi),
        vqmovun_high_s16(vqmovun_s16(b_lo), b_hi),
    );
}

#[target_feature(enable = "neon")]
fn load_eight(src: &[u8], offset: usize) -> uint8x8_t {
    debug_assert!(offset <= src.len().saturating_sub(LANES));
    let Some(chunk) = src.get(offset..).and_then(<[u8]>::first_chunk::<LANES>) else {
        return neon_memory::load_u8x8(&[0; LANES]);
    };
    neon_memory::load_u8x8(chunk)
}

#[target_feature(enable = "neon")]
fn fill_upsampled_420_chunk(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    output_width: usize,
    out: &mut [u8],
) {
    if can_vectorize_420_chunk(curr.len(), sample_offset, out.len()) {
        if let Some(out) = out.first_chunk_mut::<UPSAMPLED_LANES>() {
            fill_upsampled_420_chunk_neon(near, curr, sample_offset, out);
            return;
        }
    }
    fill_upsampled_420_chunk_scalar(near, curr, sample_offset, output_width, out);
}

fn fill_upsampled_420_chunk_scalar(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    output_width: usize,
    out: &mut [u8],
) {
    debug_assert_eq!(near.len(), curr.len());
    let n = curr.len();
    if out.is_empty() || n == 0 {
        return;
    }

    let output_x = sample_offset * 2;
    for (local_x, slot) in out.iter_mut().enumerate() {
        *slot = h2v2_fancy_sample_for_width(near, curr, output_width, output_x + local_x);
    }
}

fn can_vectorize_420_chunk(chroma_width: usize, sample_offset: usize, out_len: usize) -> bool {
    out_len == UPSAMPLED_LANES && sample_offset > 0 && sample_offset + LANES < chroma_width
}

#[target_feature(enable = "neon")]
fn fill_upsampled_420_chunk_neon(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
    out: &mut [u8; UPSAMPLED_LANES],
) {
    debug_assert!(can_vectorize_420_chunk(
        curr.len(),
        sample_offset,
        out.len()
    ));
    neon_memory::store_u8x16(out, upsampled_420_chunk16(near, curr, sample_offset));
}

#[target_feature(enable = "neon")]
fn upsampled_420_chunk16(near: &[u8], curr: &[u8], sample_offset: usize) -> uint8x16_t {
    let lanes = { upsampled_420_chunk16_u16(near, curr, sample_offset) };
    let even8 = vqmovn_u16(lanes.0);
    let odd8 = vqmovn_u16(lanes.1);
    let zipped = vzip_u8(even8, odd8);
    vcombine_u8(zipped.0, zipped.1)
}

#[target_feature(enable = "neon")]
fn upsampled_420_chunk16_u16(
    near: &[u8],
    curr: &[u8],
    sample_offset: usize,
) -> core::arch::aarch64::uint16x8x2_t {
    let (near_prev, near_this, near_next) = load_eight_triplet(near, sample_offset);
    let (curr_prev, curr_this, curr_next) = load_eight_triplet(curr, sample_offset);
    let prev = weighted_colsum(near_prev, curr_prev);
    let this = weighted_colsum(near_this, curr_this);
    let next = weighted_colsum(near_next, curr_next);
    let three_this = vaddq_u16(this, vaddq_u16(this, this));

    let even = vshrq_n_u16(vaddq_u16(vaddq_u16(three_this, prev), vdupq_n_u16(8)), 4);
    let odd = vshrq_n_u16(vaddq_u16(vaddq_u16(three_this, next), vdupq_n_u16(7)), 4);
    vzipq_u16(even, odd)
}

#[target_feature(enable = "neon")]
#[inline]
fn upsampled_420_chunk16_pair_u16(
    top_near: &[u8],
    curr: &[u8],
    bottom_near: &[u8],
    sample_offset: usize,
) -> (
    core::arch::aarch64::uint16x8x2_t,
    core::arch::aarch64::uint16x8x2_t,
) {
    let (curr_prev, curr_this, curr_next) = load_eight_triplet(curr, sample_offset);
    let curr_prev = vmovl_u8(curr_prev);
    let curr_this = vmovl_u8(curr_this);
    let curr_next = vmovl_u8(curr_next);

    let three_prev = vaddq_u16(curr_prev, vaddq_u16(curr_prev, curr_prev));
    let three_this = vaddq_u16(curr_this, vaddq_u16(curr_this, curr_this));
    let three_next = vaddq_u16(curr_next, vaddq_u16(curr_next, curr_next));

    let (top_prev, top_this, top_next) = load_eight_triplet(top_near, sample_offset);
    let top_prev = vaddq_u16(three_prev, vmovl_u8(top_prev));
    let top_this = vaddq_u16(three_this, vmovl_u8(top_this));
    let top_next = vaddq_u16(three_next, vmovl_u8(top_next));

    let (bottom_prev, bottom_this, bottom_next) = load_eight_triplet(bottom_near, sample_offset);
    let bottom_prev = vaddq_u16(three_prev, vmovl_u8(bottom_prev));
    let bottom_this = vaddq_u16(three_this, vmovl_u8(bottom_this));
    let bottom_next = vaddq_u16(three_next, vmovl_u8(bottom_next));

    let top_three_this = vaddq_u16(top_this, vaddq_u16(top_this, top_this));
    let top_even = vshrq_n_u16(
        vaddq_u16(vaddq_u16(top_three_this, top_prev), vdupq_n_u16(8)),
        4,
    );
    let top_odd = vshrq_n_u16(
        vaddq_u16(vaddq_u16(top_three_this, top_next), vdupq_n_u16(7)),
        4,
    );

    let bottom_three_this = vaddq_u16(bottom_this, vaddq_u16(bottom_this, bottom_this));
    let bottom_even = vshrq_n_u16(
        vaddq_u16(vaddq_u16(bottom_three_this, bottom_prev), vdupq_n_u16(8)),
        4,
    );
    let bottom_odd = vshrq_n_u16(
        vaddq_u16(vaddq_u16(bottom_three_this, bottom_next), vdupq_n_u16(7)),
        4,
    );

    (
        vzipq_u16(top_even, top_odd),
        vzipq_u16(bottom_even, bottom_odd),
    )
}

#[target_feature(enable = "neon")]
fn load_eight_triplet(src: &[u8], sample_offset: usize) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
    // Interior callers prove a left neighbor and eight samples plus a
    // right neighbor. Edge callers pass a replicated 16-byte window at
    // offset one. Keep violations explicit instead of synthesizing zeros.
    let start = sample_offset.checked_sub(1).expect("chroma left neighbor");
    let window = src
        .get(start..)
        .and_then(<[u8]>::first_chunk::<{ LANES + 2 }>)
        .expect("validated chroma triplet window");
    neon_memory::load_u8x8_triplet(window)
}

#[target_feature(enable = "neon")]
fn weighted_colsum(near: uint8x8_t, curr: uint8x8_t) -> uint16x8_t {
    let near16 = vmovl_u8(near);
    let curr16 = vmovl_u8(curr);
    vaddq_u16(vaddq_u16(curr16, curr16), vaddq_u16(curr16, near16))
}

#[cfg(test)]
mod top_only_tests;
