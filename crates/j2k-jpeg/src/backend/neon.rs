// SPDX-License-Identifier: MIT OR Apache-2.0

use core::arch::aarch64::{
    int32x4_t, uint16x8_t, uint8x16_t, uint8x8_t, vaddq_s32, vaddq_u16, vcombine_u16, vcombine_u8,
    vdupq_n_s32, vdupq_n_u16, vget_high_u16, vget_low_u16, vmovl_u16, vmovl_u8, vmulq_n_s32,
    vqmovn_u16, vqmovun_s32, vreinterpretq_s32_u32, vshrq_n_s32, vshrq_n_u16, vsubq_s32, vzip_u8,
    vzipq_u16,
};

use super::row_pair::{normalize_simd_row_pair, normalize_ycbcr_row};
use super::{scalar, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair};
use crate::color::upsample::h2v2_fancy_sample_for_width;
use crate::color::ycbcr::{
    ycbcr_to_rgb, FIX_0_34414, FIX_0_71414, FIX_1_40200, FIX_1_77200, ROUND,
};
use crate::simd::neon_memory;

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
            record_420_dispatch_neon_tail_chunk();
            {
                fill_rgb_row_pair_from_420_tail_neon_dual(
                    y_top,
                    y_bottom,
                    chroma,
                    Neon420TailChunk {
                        sample_offset: sample,
                        x,
                        chunk_width,
                        row_width: width,
                    },
                    dst_top,
                    dst_bottom,
                );
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
            record_420_dispatch_neon_tail_chunk();
            {
                fill_rgb_row_pair_from_420_tail_neon_top_only(
                    y_top,
                    chroma,
                    Neon420TailChunk {
                        sample_offset: sample,
                        x,
                        chunk_width,
                        row_width: width,
                    },
                    dst_top,
                );
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
#[expect(
    clippy::cast_possible_truncation,
    reason = "NEON weighted chroma sums are shifted into the u8 sample range before lane-edge repair"
)]
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

    let top_cb = ((u32::from(prev_cb[0]) + 3 * u32::from(curr_cb[0])) * 4 + 8) >> 4;
    let top_cr = ((u32::from(prev_cr[0]) + 3 * u32::from(curr_cr[0])) * 4 + 8) >> 4;
    let bottom_cb = ((u32::from(next_cb[0]) + 3 * u32::from(curr_cb[0])) * 4 + 8) >> 4;
    let bottom_cr = ((u32::from(next_cr[0]) + 3 * u32::from(curr_cr[0])) * 4 + 8) >> 4;
    let (r_top, g_top, b_top) = ycbcr_to_rgb(y_top[0], top_cb as u8, top_cr as u8);
    let (r_bottom, g_bottom, b_bottom) =
        ycbcr_to_rgb(y_bottom[0], bottom_cb as u8, bottom_cr as u8);

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
        dst_top[..3].copy_from_slice(&[r_top, g_top, b_top]);
        dst_bottom[..3].copy_from_slice(&[r_bottom, g_bottom, b_bottom]);
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
        rgb_top[..3].copy_from_slice(&[r_top, g_top, b_top]);
        rgb_bottom[..3].copy_from_slice(&[r_bottom, g_bottom, b_bottom]);
        dst_top[..chunk_width * 3].copy_from_slice(&rgb_top[..chunk_width * 3]);
        dst_bottom[..chunk_width * 3].copy_from_slice(&rgb_bottom[..chunk_width * 3]);
    }
}

#[target_feature(enable = "neon")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "NEON weighted chroma sums are shifted into the u8 sample range before lane-edge repair"
)]
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

    let cb0 = ((u32::from(prev_cb[0]) + 3 * u32::from(curr_cb[0])) * 4 + 8) >> 4;
    let cr0 = ((u32::from(prev_cr[0]) + 3 * u32::from(curr_cr[0])) * 4 + 8) >> 4;
    let (r, g, b) = ycbcr_to_rgb(y_top[0], cb0 as u8, cr0 as u8);

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
        dst_top[..3].copy_from_slice(&[r, g, b]);
    } else {
        let mut rgb = [0u8; UPSAMPLED_LANES * 3];
        {
            fill_chunk_from_vectors_u16(y_lo, cb.0, cr.0, &mut rgb[..LANES * 3]);
            fill_chunk_from_vectors_u16(y_hi, cb.1, cr.1, &mut rgb[LANES * 3..]);
        }
        rgb[..3].copy_from_slice(&[r, g, b]);
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

fn load_tail_window(src: &[u8], start: usize, len: usize) -> [u8; UPSAMPLED_LANES] {
    debug_assert!(start < src.len());
    debug_assert!(len > 0);
    debug_assert!(len <= UPSAMPLED_LANES);
    let mut out = [0u8; UPSAMPLED_LANES];
    let available = src.len() - start;
    let copy_len = available.min(len);
    out[..copy_len].copy_from_slice(&src[start..start + copy_len]);
    if copy_len < len {
        let pad = out[copy_len - 1];
        for value in &mut out[copy_len..len] {
            *value = pad;
        }
    }
    if len < UPSAMPLED_LANES {
        let pad = out[len - 1];
        for value in &mut out[len..UPSAMPLED_LANES] {
            *value = pad;
        }
    }
    out
}

fn load_head_window(src: &[u8], len: usize) -> [u8; UPSAMPLED_LANES] {
    debug_assert!(!src.is_empty());
    debug_assert!(len > 0);
    debug_assert!(len <= UPSAMPLED_LANES);
    let mut out = [0u8; UPSAMPLED_LANES];
    let copy_len = src.len().min(len);
    out[0] = src[0];
    out[1..=copy_len].copy_from_slice(&src[..copy_len]);
    if copy_len < len {
        let pad = out[copy_len];
        for value in &mut out[copy_len + 1..=len] {
            *value = pad;
        }
    }
    if len + 1 < UPSAMPLED_LANES {
        let pad = out[len];
        for value in &mut out[len + 1..UPSAMPLED_LANES] {
            *value = pad;
        }
    }
    out
}

fn can_use_tail_420_chunk(chroma_width: usize, sample_offset: usize, out_len: usize) -> bool {
    out_len <= UPSAMPLED_LANES && sample_offset > 0 && sample_offset + LANES >= chroma_width
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

#[target_feature(enable = "neon")]
fn fill_chunk_from_vectors_u16(y: uint8x8_t, cb: uint16x8_t, cr: uint16x8_t, dst_chunk: &mut [u8]) {
    let Some(dst) = dst_chunk.first_chunk_mut::<{ LANES * 3 }>() else {
        return;
    };
    let y16 = vmovl_u8(y);

    let y_lo = widen_low(y16);
    let y_hi = widen_high(y16);
    let cb_lo = subtract_bias(widen_low(cb));
    let cb_hi = subtract_bias(widen_high(cb));
    let cr_lo = subtract_bias(widen_low(cr));
    let cr_hi = subtract_bias(widen_high(cr));

    let (r_lo, g_lo, b_lo) = convert_half(y_lo, cb_lo, cr_lo);
    let (r_hi, g_hi, b_hi) = convert_half(y_hi, cb_hi, cr_hi);

    let r_bytes = pack_eight_u8(r_lo, r_hi);
    let g_bytes = pack_eight_u8(g_lo, g_hi);
    let b_bytes = pack_eight_u8(b_lo, b_hi);

    neon_memory::store_rgb8(dst, r_bytes, g_bytes, b_bytes);
}

#[target_feature(enable = "neon")]
fn fill_rgb_row_from_ycbcr_chunk16_neon(
    y: &[u8; UPSAMPLED_LANES],
    cb: &[u8; UPSAMPLED_LANES],
    cr: &[u8; UPSAMPLED_LANES],
    dst: &mut [u8; UPSAMPLED_LANES * 3],
) {
    let (y, y_remainder) = y.as_chunks::<LANES>();
    let (cb, cb_remainder) = cb.as_chunks::<LANES>();
    let (cr, cr_remainder) = cr.as_chunks::<LANES>();
    let (dst, dst_remainder) = dst.as_chunks_mut::<{ LANES * 3 }>();
    let _ = (y_remainder, cb_remainder, cr_remainder, dst_remainder);

    fill_chunk(&y[0], &cb[0], &cr[0], &mut dst[0]);
    fill_chunk(&y[1], &cb[1], &cr[1], &mut dst[1]);
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
fn widen_low(values: core::arch::aarch64::uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(values)))
}

#[target_feature(enable = "neon")]
fn widen_high(values: core::arch::aarch64::uint16x8_t) -> int32x4_t {
    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(values)))
}

#[target_feature(enable = "neon")]
fn subtract_bias(values: int32x4_t) -> int32x4_t {
    vsubq_s32(values, vdupq_n_s32(128))
}

#[target_feature(enable = "neon")]
fn fixed_mul_shift(values: int32x4_t, coefficient: i32) -> int32x4_t {
    vshrq_n_s32(
        vaddq_s32(vmulq_n_s32(values, coefficient), vdupq_n_s32(ROUND)),
        16,
    )
}

#[target_feature(enable = "neon")]
fn convert_half(y: int32x4_t, cb: int32x4_t, cr: int32x4_t) -> (int32x4_t, int32x4_t, int32x4_t) {
    let r = vaddq_s32(y, fixed_mul_shift(cr, FIX_1_40200));
    let g = vsubq_s32(
        y,
        vshrq_n_s32(
            vaddq_s32(
                vaddq_s32(vmulq_n_s32(cb, FIX_0_34414), vmulq_n_s32(cr, FIX_0_71414)),
                vdupq_n_s32(ROUND),
            ),
            16,
        ),
    );
    let b = vaddq_s32(y, fixed_mul_shift(cb, FIX_1_77200));
    (r, g, b)
}

#[target_feature(enable = "neon")]
fn pack_eight_u8(low: int32x4_t, high: int32x4_t) -> uint8x8_t {
    let words = vcombine_u16(vqmovun_s32(low), vqmovun_s32(high));
    vqmovn_u16(words)
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
    let Some(start) = sample_offset.checked_sub(1) else {
        let zero = neon_memory::load_u8x8(&[0; LANES]);
        return (zero, zero, zero);
    };
    let Some(window) = src
        .get(start..)
        .and_then(<[u8]>::first_chunk::<{ LANES + 2 }>)
    else {
        let zero = neon_memory::load_u8x8(&[0; LANES]);
        return (zero, zero, zero);
    };
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
