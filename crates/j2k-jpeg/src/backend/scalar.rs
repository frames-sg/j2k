// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::color::upsample::h2v2_fancy_sample;
use crate::color::ycbcr::ycbcr_to_rgb;

use super::{Rgb420CroppedRowPair, Rgb420RowPair, Rgb422Row};

pub(crate) fn fill_rgb_row_from_gray(gray_row: &[u8], dst: &mut [u8]) {
    for (&gray, pixel) in gray_row.iter().zip(dst.as_chunks_mut::<3>().0.iter_mut()) {
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
        .zip(dst.as_chunks_mut::<3>().0.iter_mut())
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
        .zip(dst.as_chunks_mut::<3>().0.iter_mut())
    {
        let (r, g, b) = ycbcr_to_rgb(y_sample, cb_sample, cr_sample);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
}

pub(crate) fn fill_rgb_row_from_422(request: Rgb422Row<'_>) {
    let Rgb422Row { y, cb, cr, dst } = request;
    let width = y
        .len()
        .min(dst.len() / 3)
        .min(cb.len().min(cr.len()).saturating_mul(2));
    let chroma_width = width.div_ceil(2);
    let cb = &cb[..chroma_width];
    let cr = &cr[..chroma_width];
    for (x, pixel) in dst[..width * 3]
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (r, g, b) = ycbcr_to_rgb(y[x], h2v1_fancy_sample(cb, x), h2v1_fancy_sample(cr, x));
        pixel.copy_from_slice(&[r, g, b]);
    }
}

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the weighted mean of u8 chroma samples remains in the u8 range"
)]
pub(super) fn h2v1_fancy_sample(input: &[u8], x: usize) -> u8 {
    let sample = x / 2;
    match x {
        0 => input[0],
        _ if x == input.len() * 2 - 1 => input[input.len() - 1],
        _ if x.is_multiple_of(2) => {
            ((3 * u16::from(input[sample]) + u16::from(input[sample - 1]) + 1) >> 2) as u8
        }
        _ => ((3 * u16::from(input[sample]) + u16::from(input[sample + 1]) + 2) >> 2) as u8,
    }
}

pub(crate) fn fill_rgba_row_from_gray(gray_row: &[u8], dst: &mut [u8], alpha: u8) {
    for (&gray, pixel) in gray_row.iter().zip(dst.as_chunks_mut::<4>().0.iter_mut()) {
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
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
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
        .zip(dst.as_chunks_mut::<4>().0.iter_mut())
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

    for (x, pixel) in dst_top.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let cb = h2v2_fancy_sample(prev_cb, curr_cb, x);
        let cr = h2v2_fancy_sample(prev_cr, curr_cr, x);
        let (r, g, b) = ycbcr_to_rgb(y_top[x], cb, cr);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }

    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        for (x, pixel) in dst_bottom.as_chunks_mut::<3>().0.iter_mut().enumerate() {
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

    for (local_x, pixel) in dst_top.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let x = crop_start + local_x;
        let cb = h2v2_fancy_sample(prev_cb, curr_cb, x);
        let cr = h2v2_fancy_sample(prev_cr, curr_cr, x);
        let (r, g, b) = ycbcr_to_rgb(y_top[x], cb, cr);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }

    if let (Some(y_bottom), Some(dst_bottom)) = (y_bottom, dst_bottom) {
        for (local_x, pixel) in dst_bottom.as_chunks_mut::<3>().0.iter_mut().enumerate() {
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
