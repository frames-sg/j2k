// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::color::upsample::h2v2_fancy_sample;
use crate::color::ycbcr::ycbcr_to_rgb;

use super::{Rgb420CroppedRowPair, Rgb420RowPair};

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    for (&gray, pixel) in gray_row.iter().zip(dst.chunks_exact_mut(3)) {
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    }
}

pub(crate) fn fill_rgb_row_from_rgb(r_row: &[u8], g_row: &[u8], b_row: &[u8], dst: &mut [u8]) {
    for (((&r, &g), &b), pixel) in r_row
        .iter()
        .zip(g_row.iter())
        .zip(b_row.iter())
        .zip(dst.chunks_exact_mut(3))
    {
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
}

pub(crate) fn fill_rgb_row_from_ycbcr(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], dst: &mut [u8]) {
    for (((&y_sample, &cb_sample), &cr_sample), pixel) in y_row
        .iter()
        .zip(cb_row.iter())
        .zip(cr_row.iter())
        .zip(dst.chunks_exact_mut(3))
    {
        let (r, g, b) = ycbcr_to_rgb(y_sample, cb_sample, cr_sample);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
}

pub(crate) fn fill_rgba_row_from_gray(gray_row: &[u8], dst: &mut [u8], alpha: u8) {
    for (&gray, pixel) in gray_row.iter().zip(dst.chunks_exact_mut(4)) {
        write_rgba_pixel(pixel, gray, gray, gray, alpha);
    }
}

pub(crate) fn fill_rgba_row_from_rgb(
    r_row: &[u8],
    g_row: &[u8],
    b_row: &[u8],
    dst: &mut [u8],
    alpha: u8,
) {
    for (((&r, &g), &b), pixel) in r_row
        .iter()
        .zip(g_row.iter())
        .zip(b_row.iter())
        .zip(dst.chunks_exact_mut(4))
    {
        write_rgba_pixel(pixel, r, g, b, alpha);
    }
}

pub(crate) fn fill_rgba_row_from_ycbcr(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    dst: &mut [u8],
    alpha: u8,
) {
    for (((&y_sample, &cb_sample), &cr_sample), pixel) in y_row
        .iter()
        .zip(cb_row.iter())
        .zip(cr_row.iter())
        .zip(dst.chunks_exact_mut(4))
    {
        let (r, g, b) = ycbcr_to_rgb(y_sample, cb_sample, cr_sample);
        write_rgba_pixel(pixel, r, g, b, alpha);
    }
}

fn write_rgba_pixel(pixel: &mut [u8], r: u8, g: u8, b: u8, alpha: u8) {
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
    pixel[3] = alpha;
}

pub(crate) fn fill_rgb_row_pair_from_420(request: Rgb420RowPair<'_>) {
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = request;
    let prev_cb = chroma.prev_cb;
    let curr_cb = chroma.curr_cb;
    let next_cb = chroma.next_cb;
    let prev_cr = chroma.prev_cr;
    let curr_cr = chroma.curr_cr;
    let next_cr = chroma.next_cr;
    let width = y_top.len();
    debug_assert_eq!(width * 3, dst_top.len());
    debug_assert!(y_bottom.is_none_or(|row| row.len() == width));
    debug_assert!(dst_bottom.as_ref().is_none_or(|row| row.len() == width * 3));

    for (x, pixel) in dst_top.chunks_exact_mut(3).enumerate() {
        let cb = h2v2_fancy_sample(prev_cb, curr_cb, x);
        let cr = h2v2_fancy_sample(prev_cr, curr_cr, x);
        let (r, g, b) = ycbcr_to_rgb(y_top[x], cb, cr);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }

    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        for (x, pixel) in dst_bottom.chunks_exact_mut(3).enumerate() {
            let cb = h2v2_fancy_sample(next_cb, curr_cb, x);
            let cr = h2v2_fancy_sample(next_cr, curr_cr, x);
            let (r, g, b) = ycbcr_to_rgb(y_bottom[x], cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
    }
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
    let prev_cb = chroma.prev_cb;
    let curr_cb = chroma.curr_cb;
    let next_cb = chroma.next_cb;
    let prev_cr = chroma.prev_cr;
    let curr_cr = chroma.curr_cr;
    let next_cr = chroma.next_cr;
    let crop_start = crop.start;
    let crop_width = crop.width;
    let crop_end = crop_start + crop_width;
    debug_assert!(crop_end <= y_top.len());
    debug_assert_eq!(crop_width * 3, dst_top.len());
    debug_assert!(y_bottom.is_none_or(|row| row.len() == y_top.len()));
    debug_assert!(dst_bottom
        .as_ref()
        .is_none_or(|row| row.len() == crop_width * 3));
    debug_assert_eq!(prev_cb.len(), curr_cb.len());
    debug_assert_eq!(prev_cb.len(), next_cb.len());
    debug_assert_eq!(prev_cr.len(), curr_cr.len());
    debug_assert_eq!(prev_cr.len(), next_cr.len());

    for (local_x, pixel) in dst_top.chunks_exact_mut(3).enumerate() {
        let x = crop_start + local_x;
        let cb = h2v2_fancy_sample(prev_cb, curr_cb, x);
        let cr = h2v2_fancy_sample(prev_cr, curr_cr, x);
        let (r, g, b) = ycbcr_to_rgb(y_top[x], cb, cr);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }

    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        for (local_x, pixel) in dst_bottom.chunks_exact_mut(3).enumerate() {
            let x = crop_start + local_x;
            let cb = h2v2_fancy_sample(next_cb, curr_cb, x);
            let cr = h2v2_fancy_sample(next_cr, curr_cr, x);
            let (r, g, b) = ycbcr_to_rgb(y_bottom[x], cb, cr);
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
        }
    }
}
