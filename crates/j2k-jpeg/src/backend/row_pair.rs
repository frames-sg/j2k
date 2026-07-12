// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared safe slice normalization before architecture-specific 4:2:0 kernels.

use super::{Rgb420ChromaRows, Rgb420RowPair};

pub(super) fn normalize_ycbcr_row<'a>(
    y_row: &'a [u8],
    cb_row: &'a [u8],
    cr_row: &'a [u8],
    dst: &'a mut [u8],
) -> (&'a [u8], &'a [u8], &'a [u8], &'a mut [u8]) {
    let width = y_row
        .len()
        .min(cb_row.len())
        .min(cr_row.len())
        .min(dst.len() / 3);
    (
        &y_row[..width],
        &cb_row[..width],
        &cr_row[..width],
        &mut dst[..width * 3],
    )
}

pub(super) fn normalize_simd_row_pair(request: Rgb420RowPair<'_>) -> Option<Rgb420RowPair<'_>> {
    let Rgb420RowPair {
        y_top,
        y_bottom,
        chroma,
        dst_top,
        dst_bottom,
    } = request;
    let chroma_width = chroma.min_width();
    let bottom_width = match (y_bottom.as_ref(), dst_bottom.as_ref()) {
        (Some(row), Some(dst)) => row.len().min(dst.len() / 3),
        _ => usize::MAX,
    };
    let width = y_top
        .len()
        .min(dst_top.len() / 3)
        .min(bottom_width)
        .min(chroma_width.saturating_mul(2));
    if width == 0 {
        return None;
    }

    Some(Rgb420RowPair::new(
        &y_top[..width],
        y_bottom.and_then(|row| row.get(..width)),
        Rgb420ChromaRows::new(
            &chroma.prev_cb[..chroma_width],
            &chroma.curr_cb[..chroma_width],
            &chroma.next_cb[..chroma_width],
            &chroma.prev_cr[..chroma_width],
            &chroma.curr_cr[..chroma_width],
            &chroma.next_cr[..chroma_width],
        ),
        &mut dst_top[..width * 3],
        dst_bottom.and_then(|row| row.get_mut(..width * 3)),
    ))
}

#[cfg(test)]
mod tests {
    use super::{normalize_simd_row_pair, normalize_ycbcr_row};
    use crate::backend::{Rgb420ChromaRows, Rgb420RowPair};

    #[test]
    fn ycbcr_rows_clamp_to_the_shortest_complete_pixel_extent() {
        let y = [1_u8; 4];
        let cb = [2_u8; 3];
        let cr = [3_u8; 2];
        let mut dst = [0_u8; 9];
        let (y, cb, cr, dst) = normalize_ycbcr_row(&y, &cb, &cr, &mut dst);
        assert_eq!((y.len(), cb.len(), cr.len(), dst.len()), (2, 2, 2, 6));
    }

    #[test]
    fn row_pair_clamps_luma_chroma_and_both_destinations_together() {
        let y_top = [1_u8; 8];
        let y_bottom = [2_u8; 7];
        let chroma = [3_u8; 3];
        let mut dst_top = [0_u8; 24];
        let mut dst_bottom = [0_u8; 18];
        let normalized = normalize_simd_row_pair(Rgb420RowPair::new(
            &y_top,
            Some(&y_bottom),
            Rgb420ChromaRows::new(&chroma, &chroma, &chroma, &chroma, &chroma, &chroma),
            &mut dst_top,
            Some(&mut dst_bottom),
        ))
        .expect("non-empty normalized row pair");

        assert_eq!(normalized.y_top.len(), 6);
        assert_eq!(normalized.y_bottom.expect("bottom luma").len(), 6);
        assert_eq!(normalized.chroma.min_width(), 3);
        assert_eq!(normalized.dst_top.len(), 18);
        assert_eq!(normalized.dst_bottom.expect("bottom output").len(), 18);
    }

    #[test]
    fn row_pair_rejects_zero_complete_pixels() {
        let chroma = [0_u8; 1];
        let mut dst = [];
        assert!(normalize_simd_row_pair(Rgb420RowPair::new(
            &[1],
            None,
            Rgb420ChromaRows::new(&chroma, &chroma, &chroma, &chroma, &chroma, &chroma,),
            &mut dst,
            None,
        ))
        .is_none());
    }
}
