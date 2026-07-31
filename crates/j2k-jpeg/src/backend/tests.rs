// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::color::upsample::upsample_h2v2_fancy_rows;
use crate::color::ycbcr::ycbcr_to_rgb;
use alloc::vec;
use alloc::vec::Vec;

use super::scalar;

macro_rules! rgb420_pair {
    (
        $y_top:expr,
        $y_bottom:expr,
        $prev_cb:expr,
        $curr_cb:expr,
        $next_cb:expr,
        $prev_cr:expr,
        $curr_cr:expr,
        $next_cr:expr,
        $dst_top:expr,
        $dst_bottom:expr $(,)?
    ) => {
        super::Rgb420RowPair::new(
            $y_top,
            $y_bottom,
            super::Rgb420ChromaRows::new(
                $prev_cb, $curr_cb, $next_cb, $prev_cr, $curr_cr, $next_cr,
            ),
            $dst_top,
            $dst_bottom,
        )
    };
}

macro_rules! rgb420_cropped_pair {
    (
        $y_top:expr,
        $y_bottom:expr,
        $prev_cb:expr,
        $curr_cb:expr,
        $next_cb:expr,
        $prev_cr:expr,
        $curr_cr:expr,
        $next_cr:expr,
        $crop_start:expr,
        $crop_width:expr,
        $dst_top:expr,
        $dst_bottom:expr $(,)?
    ) => {
        super::Rgb420CroppedRowPair::new(
            rgb420_pair!(
                $y_top,
                $y_bottom,
                $prev_cb,
                $curr_cb,
                $next_cb,
                $prev_cr,
                $curr_cr,
                $next_cr,
                $dst_top,
                $dst_bottom,
            ),
            super::Rgb420Crop::new($crop_start, $crop_width),
        )
    };
}

#[derive(Clone, Copy)]
struct Rgb420Fixture<'a> {
    y_top: &'a [u8],
    y_bottom: Option<&'a [u8]>,
    chroma: super::Rgb420ChromaRows<'a>,
}

impl<'a> Rgb420Fixture<'a> {
    fn new(
        y_top: &'a [u8],
        y_bottom: Option<&'a [u8]>,
        cb: [&'a [u8]; 3],
        cr: [&'a [u8]; 3],
    ) -> Self {
        let [prev_cb, curr_cb, next_cb] = cb;
        let [prev_cr, curr_cr, next_cr] = cr;
        Self {
            y_top,
            y_bottom,
            chroma: super::Rgb420ChromaRows::new(
                prev_cb, curr_cb, next_cb, prev_cr, curr_cr, next_cr,
            ),
        }
    }

    fn request<'b>(
        self,
        dst_top: &'b mut [u8],
        dst_bottom: Option<&'b mut [u8]>,
    ) -> super::Rgb420RowPair<'b>
    where
        'a: 'b,
    {
        super::Rgb420RowPair::new(self.y_top, self.y_bottom, self.chroma, dst_top, dst_bottom)
    }

    fn cropped_request<'b>(
        self,
        crop_start: usize,
        crop_width: usize,
        dst_top: &'b mut [u8],
        dst_bottom: Option<&'b mut [u8]>,
    ) -> super::Rgb420CroppedRowPair<'b>
    where
        'a: 'b,
    {
        super::Rgb420CroppedRowPair::new(
            self.request(dst_top, dst_bottom),
            super::Rgb420Crop::new(crop_start, crop_width),
        )
    }
}

fn assert_ycbcr_rows_match_scalar(
    label: &str,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    fill: impl FnOnce(&[u8], &[u8], &[u8], &mut [u8]),
) {
    let mut expected = vec![0xAA; y.len() * 3];
    let mut actual = expected.clone();
    scalar::fill_rgb_row_from_ycbcr(y, cb, cr, &mut expected);
    fill(y, cb, cr, &mut actual);
    assert_eq!(actual, expected, "{label}");
}

#[cfg(target_arch = "x86_64")]
fn assert_gray_rows_match_scalar(label: &str, gray: &[u8], fill: impl FnOnce(&[u8], &mut [u8])) {
    let mut expected = vec![0xAA; gray.len() * 3];
    let mut actual = expected.clone();
    scalar::fill_rgb_row_from_gray(gray, &mut expected);
    fill(gray, &mut actual);
    assert_eq!(actual, expected, "{label}");
}

#[cfg(target_arch = "x86_64")]
fn assert_rgb_rows_match_scalar(
    label: &str,
    r: &[u8],
    g: &[u8],
    b: &[u8],
    fill: impl FnOnce(&[u8], &[u8], &[u8], &mut [u8]),
) {
    let mut expected = vec![0xAA; r.len() * 3];
    let mut actual = expected.clone();
    scalar::fill_rgb_row_from_rgb(r, g, b, &mut expected);
    fill(r, g, b, &mut actual);
    assert_eq!(actual, expected, "{label}");
}

fn assert_rgb420_pair_matches_scalar(
    label: &str,
    kind: super::BackendKind,
    fixture: Rgb420Fixture<'_>,
) {
    let mut expected_top = vec![0xAA; fixture.y_top.len() * 3];
    let mut expected_bottom = fixture.y_bottom.map(|row| vec![0xAA; row.len() * 3]);
    scalar::fill_rgb_row_pair_from_420(
        fixture.request(&mut expected_top, expected_bottom.as_deref_mut()),
    );

    let mut actual_top = vec![0xAA; fixture.y_top.len() * 3];
    let mut actual_bottom = fixture.y_bottom.map(|row| vec![0xAA; row.len() * 3]);
    super::Backend { kind }
        .fill_rgb_row_pair_from_420(fixture.request(&mut actual_top, actual_bottom.as_deref_mut()));

    assert_eq!(actual_top, expected_top, "{label}: top row");
    assert_eq!(actual_bottom, expected_bottom, "{label}: bottom row");
}

fn assert_rgb420_cropped_pair_matches_scalar(
    label: &str,
    kind: super::BackendKind,
    fixture: Rgb420Fixture<'_>,
    crop_start: usize,
    crop_width: usize,
) {
    let mut expected_top = vec![0xAA; crop_width * 3];
    let mut expected_bottom = fixture.y_bottom.map(|_| vec![0xAA; crop_width * 3]);
    scalar::fill_rgb_row_pair_from_420_cropped(fixture.cropped_request(
        crop_start,
        crop_width,
        &mut expected_top,
        expected_bottom.as_deref_mut(),
    ));

    let mut actual_top = vec![0xAA; crop_width * 3];
    let mut actual_bottom = fixture.y_bottom.map(|_| vec![0xAA; crop_width * 3]);
    super::Backend { kind }.fill_rgb_row_pair_from_420_cropped(fixture.cropped_request(
        crop_start,
        crop_width,
        &mut actual_top,
        actual_bottom.as_deref_mut(),
    ));

    assert_eq!(actual_top, expected_top, "{label}: top row");
    assert_eq!(actual_bottom, expected_bottom, "{label}: bottom row");
}

fn assert_rgb420_pair_uses_safe_prefix(
    label: &str,
    kind: super::BackendKind,
    fixture: Rgb420Fixture<'_>,
    safe_width: usize,
) {
    let mut expected_top = vec![0xAA; fixture.y_top.len() * 3];
    let mut expected_bottom = fixture.y_bottom.map(|row| vec![0xAA; row.len() * 3]);
    let expected_y_bottom = fixture.y_bottom.map(|row| &row[..safe_width]);
    let expected_dst_bottom = expected_bottom
        .as_deref_mut()
        .map(|row| &mut row[..safe_width * 3]);
    scalar::fill_rgb_row_pair_from_420(super::Rgb420RowPair::new(
        &fixture.y_top[..safe_width],
        expected_y_bottom,
        fixture.chroma,
        &mut expected_top[..safe_width * 3],
        expected_dst_bottom,
    ));

    let mut actual_top = vec![0xAA; fixture.y_top.len() * 3];
    let mut actual_bottom = fixture.y_bottom.map(|row| vec![0xAA; row.len() * 3]);
    super::Backend { kind }
        .fill_rgb_row_pair_from_420(fixture.request(&mut actual_top, actual_bottom.as_deref_mut()));

    assert_eq!(actual_top, expected_top, "{label}: top row");
    assert_eq!(actual_bottom, expected_bottom, "{label}: bottom row");
}

#[cfg(target_arch = "x86_64")]
fn x86_fixture_byte(index: usize) -> u8 {
    u8::try_from(index).expect("x86 fixture index must fit in u8")
}

#[test]
fn gray_rows_expand_to_equal_rgb_channels() {
    let gray = [10u8, 40, 90, 200];
    let mut dst = vec![0u8; gray.len() * 3];
    scalar::fill_rgb_row_from_gray(&gray, &mut dst);
    assert_eq!(dst, vec![10, 10, 10, 40, 40, 40, 90, 90, 90, 200, 200, 200]);
}

#[test]
fn ycbcr_rows_match_per_pixel_reference() {
    let y = [16u8, 40, 90, 200];
    let cb = [128u8, 100, 200, 180];
    let cr = [128u8, 220, 10, 90];
    let mut dst = vec![0u8; y.len() * 3];
    scalar::fill_rgb_row_from_ycbcr(&y, &cb, &cr, &mut dst);

    let expected: Vec<u8> = y
        .iter()
        .zip(cb.iter())
        .zip(cr.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();

    assert_eq!(dst, expected);
}

#[test]
fn ycbcr_420_row_pair_matches_reference() {
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128, 136];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];
    let mut dst_top = vec![0u8; y_top.len() * 3];
    let mut dst_bot = vec![0u8; y_bot.len() * 3];

    scalar::fill_rgb_row_pair_from_420(rgb420_pair!(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut dst_top,
        Some(&mut dst_bot),
    ));

    let mut cb_top = vec![0u8; y_top.len()];
    let mut cb_bot = vec![0u8; y_top.len()];
    let mut cr_top = vec![0u8; y_top.len()];
    let mut cr_bot = vec![0u8; y_top.len()];
    upsample_h2v2_fancy_rows(
        &prev_cb,
        &curr_cb,
        &next_cb,
        y_top.len(),
        &mut cb_top,
        &mut cb_bot,
    );
    upsample_h2v2_fancy_rows(
        &prev_cr,
        &curr_cr,
        &next_cr,
        y_top.len(),
        &mut cr_top,
        &mut cr_bot,
    );

    let expected_top: Vec<u8> = y_top
        .iter()
        .zip(cb_top.iter())
        .zip(cr_top.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();
    let expected_bot: Vec<u8> = y_bot
        .iter()
        .zip(cb_bot.iter())
        .zip(cr_bot.iter())
        .flat_map(|((&y, &cb), &cr)| {
            let (r, g, b) = ycbcr_to_rgb(y, cb, cr);
            [r, g, b]
        })
        .collect();

    assert_eq!(dst_top, expected_top);
    assert_eq!(dst_bot, expected_bot);
}

#[test]
fn backend_scalar_420_row_pair_matches_reference_for_odd_widths() {
    let y_top = [16u8, 24, 32, 40, 48, 56, 64];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    assert_rgb420_pair_matches_scalar(
        "scalar dispatch odd-width 4:2:0 pair",
        super::BackendKind::Scalar,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[test]
fn backend_scalar_420_row_pair_handles_missing_bottom_row() {
    let y_top = [16u8, 24, 32, 40, 48];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    assert_rgb420_pair_matches_scalar(
        "scalar dispatch top-only 4:2:0 pair",
        super::BackendKind::Scalar,
        Rgb420Fixture::new(
            &y_top,
            None,
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the bounded row index"
)]
fn backend_scalar_420_cropped_row_pair_matches_full_width_crop() {
    let backend = super::Backend {
        kind: super::BackendKind::Scalar,
    };
    let width = 73usize;
    let crop_start = 3usize;
    let crop_width = 53usize;
    let chroma_width = width.div_ceil(2);
    let y_top: Vec<u8> = (0..width)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let y_bot: Vec<u8> = (0..width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(19)))
        .collect();
    let prev_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(17)).wrapping_add(41))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(29)).wrapping_add(13))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(43)).wrapping_add(7))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(11)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(23)))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(31)))
        .collect();

    let mut expected_top = vec![0u8; width * 3];
    let mut expected_bot = vec![0u8; width * 3];
    scalar::fill_rgb_row_pair_from_420(rgb420_pair!(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        Some(&mut expected_bot),
    ));

    let mut actual_top = vec![0u8; crop_width * 3];
    let mut actual_bot = vec![0u8; crop_width * 3];
    backend.fill_rgb_row_pair_from_420_cropped(rgb420_cropped_pair!(
        &y_top,
        Some(&y_bot),
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        crop_start,
        crop_width,
        &mut actual_top,
        Some(&mut actual_bot),
    ));

    let crop_bytes = crop_width * 3;
    let crop_byte_start = crop_start * 3;
    assert_eq!(
        actual_top,
        expected_top[crop_byte_start..crop_byte_start + crop_bytes]
    );
    assert_eq!(
        actual_bot,
        expected_bot[crop_byte_start..crop_byte_start + crop_bytes]
    );
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the bounded row index"
)]
fn backend_scalar_420_cropped_top_only_matches_full_width_crop() {
    let backend = super::Backend {
        kind: super::BackendKind::Scalar,
    };
    let width = 31usize;
    let crop_start = 1usize;
    let crop_width = 17usize;
    let chroma_width = width.div_ceil(2);
    let y_top: Vec<u8> = (0..width)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    let prev_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(23)).wrapping_add(19))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(31)))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(7)).wrapping_add(113))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(13)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(29)).wrapping_add(3))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(17)))
        .collect();

    let mut expected_top = vec![0u8; width * 3];
    scalar::fill_rgb_row_pair_from_420(rgb420_pair!(
        &y_top,
        None,
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        &mut expected_top,
        None,
    ));

    let mut actual_top = vec![0u8; crop_width * 3];
    backend.fill_rgb_row_pair_from_420_cropped(rgb420_cropped_pair!(
        &y_top,
        None,
        &prev_cb,
        &curr_cb,
        &next_cb,
        &prev_cr,
        &curr_cr,
        &next_cr,
        crop_start,
        crop_width,
        &mut actual_top,
        None,
    ));

    let crop_bytes = crop_width * 3;
    let crop_byte_start = crop_start * 3;
    assert_eq!(
        actual_top,
        expected_top[crop_byte_start..crop_byte_start + crop_bytes]
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_ycbcr_rows_match_scalar_reference_for_tail_widths() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let cb = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let cr = [0u8, 15, 32, 64, 96, 120, 128, 136, 160, 192, 224, 240, 255];
    assert_ycbcr_rows_match_scalar(
        "AVX2 tail width",
        &y,
        &cb,
        &cr,
        super::x86::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_ycbcr_rows_use_shortest_safe_prefix() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y = [16u8, 40, 90, 200, 12, 24, 48, 96];
    let cb = [128u8, 100, 200, 180, 90];
    let cr = [128u8, 220, 10, 90, 70, 60, 50];
    assert_ycbcr_rows_match_scalar(
        "AVX2 shortest safe prefix",
        &y,
        &cb,
        &cr,
        super::x86::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_gray_rows_match_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let gray = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    assert_gray_rows_match_scalar(
        "AVX2 gray row",
        &gray,
        super::x86::fill_rgb_row_from_gray_for_test,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_rgb_rows_match_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let r = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let g = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let b = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    assert_rgb_rows_match_scalar(
        "AVX2 RGB row",
        &r,
        &g,
        &b,
        super::x86::fill_rgb_row_from_rgb_for_test,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_ycbcr_rows_match_scalar_reference_across_multiple_chunks() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let len = 31usize;
    let y: Vec<u8> = (0..len)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub(x86_fixture_byte(i).wrapping_mul(29)))
        .collect();
    let cr: Vec<u8> = (0..len)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(53)).wrapping_add(97))
        .collect();
    assert_ycbcr_rows_match_scalar(
        "AVX2 multiple chunks",
        &y,
        &cb,
        &cr,
        super::x86::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_matches_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128, 136];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    assert_rgb420_pair_matches_scalar(
        "AVX2 even-width 4:2:0 pair",
        super::BackendKind::Avx2,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_matches_scalar_reference_for_odd_widths() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y_top = [16u8, 24, 32, 40, 48, 56, 64];
    let y_bot = [80u8, 88, 96, 104, 112, 120, 128];
    let prev_cb = [120u8, 100, 140, 160];
    let curr_cb = [110u8, 90, 130, 170];
    let next_cb = [100u8, 80, 120, 180];
    let prev_cr = [130u8, 150, 170, 190];
    let curr_cr = [140u8, 160, 180, 200];
    let next_cr = [150u8, 170, 190, 210];

    assert_rgb420_pair_matches_scalar(
        "AVX2 odd-width 4:2:0 pair",
        super::BackendKind::Avx2,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_handles_missing_bottom_row() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y_top = [16u8, 24, 32, 40, 48];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    assert_rgb420_pair_matches_scalar(
        "AVX2 top-only 4:2:0 pair",
        super::BackendKind::Avx2,
        Rgb420Fixture::new(
            &y_top,
            None,
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_cropped_row_pair_matches_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let width = 73usize;
    let crop_start = 3usize;
    let crop_width = 53usize;
    let chroma_width = width.div_ceil(2);
    let y_top: Vec<u8> = (0..width)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let y_bot: Vec<u8> = (0..width)
        .map(|i| 255u8.wrapping_sub(x86_fixture_byte(i).wrapping_mul(19)))
        .collect();
    let prev_cb: Vec<u8> = (0..chroma_width)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(17)).wrapping_add(41))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_width)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(29)).wrapping_add(13))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_width)
        .map(|i| (x86_fixture_byte(i).wrapping_mul(43)).wrapping_add(7))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub(x86_fixture_byte(i).wrapping_mul(11)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub(x86_fixture_byte(i).wrapping_mul(23)))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub(x86_fixture_byte(i).wrapping_mul(31)))
        .collect();

    assert_rgb420_cropped_pair_matches_scalar(
        "AVX2 cropped 4:2:0 pair",
        super::BackendKind::Avx2,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
        crop_start,
        crop_width,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_420_row_pair_uses_shortest_safe_prefix() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72, 80];
    let y_bot = [80u8, 88, 96, 104];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    let safe_width = y_bot.len();
    assert_rgb420_pair_uses_safe_prefix(
        "AVX2 shortest safe 4:2:0 prefix",
        super::BackendKind::Avx2,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
        safe_width,
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_backend_prefers_cropped_420_region_when_available() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }

    let backend = super::Backend {
        kind: super::BackendKind::Avx2,
    };

    assert!(backend.prefers_cropped_420_region(4096, 257));
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_ycbcr_rows_match_scalar_reference_for_tail_widths() {
    let y = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let cb = [255u8, 240, 200, 180, 160, 140, 128, 120, 96, 64, 32, 16, 0];
    let cr = [0u8, 15, 32, 64, 96, 120, 128, 136, 160, 192, 224, 240, 255];
    assert_ycbcr_rows_match_scalar(
        "NEON tail width",
        &y,
        &cb,
        &cr,
        super::neon::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_ycbcr_rows_use_shortest_safe_prefix() {
    let y = [16u8, 40, 90, 200, 12, 24, 48, 96];
    let cb = [128u8, 100, 200, 180, 90];
    let cr = [128u8, 220, 10, 90, 70, 60, 50];
    assert_ycbcr_rows_match_scalar(
        "NEON shortest safe prefix",
        &y,
        &cb,
        &cr,
        super::neon::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the bounded row index"
)]
fn neon_ycbcr_rows_match_scalar_reference_across_multiple_chunks() {
    let len = 31usize;
    let y: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(29)))
        .collect();
    let cr: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    assert_ycbcr_rows_match_scalar(
        "NEON multiple chunks",
        &y,
        &cb,
        &cr,
        super::neon::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the row index to exercise tails"
)]
fn neon_ycbcr_rows_match_scalar_reference_for_offset_subslice_and_odd_tail_width() {
    let len = 255usize;
    let y_buf: Vec<u8> = (0..len + 3)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let cb_buf: Vec<u8> = (0..len + 3)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(29)))
        .collect();
    let cr_buf: Vec<u8> = (0..len + 3)
        .map(|i| ((i as u8).wrapping_mul(53)).wrapping_add(97))
        .collect();
    let y = &y_buf[1..=len];
    let cb = &cb_buf[1..=len];
    let cr = &cr_buf[1..=len];
    assert_ycbcr_rows_match_scalar(
        "NEON offset subslice and odd tail",
        y,
        cb,
        cr,
        super::neon::fill_rgb_row_from_ycbcr_for_test,
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_matches_scalar_reference_for_tail_widths() {
    let y_top = [0u8, 16, 33, 64, 96, 127, 128, 129, 160, 192, 224, 255, 12];
    let y_bot = [12u8, 255, 224, 192, 160, 129, 128, 127, 96, 64, 33, 16, 0];
    let prev_cb = [255u8, 240, 200, 180, 160, 140, 128];
    let curr_cb = [240u8, 220, 180, 160, 140, 120, 96];
    let next_cb = [220u8, 200, 160, 140, 120, 96, 64];
    let prev_cr = [0u8, 15, 32, 64, 96, 120, 128];
    let curr_cr = [16u8, 32, 64, 96, 120, 136, 160];
    let next_cr = [32u8, 64, 96, 120, 136, 160, 192];

    assert_rgb420_pair_matches_scalar(
        "NEON tail-width 4:2:0 pair",
        super::BackendKind::Neon,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the bounded row index"
)]
fn neon_420_row_pair_matches_scalar_reference_across_multiple_chunks() {
    let len = 31usize;
    let y_top: Vec<u8> = (0..len)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let y_bot: Vec<u8> = (0..len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(19)))
        .collect();
    let chroma_len = len.div_ceil(2);
    let prev_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(17)).wrapping_add(41))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(29)).wrapping_add(13))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_len)
        .map(|i| ((i as u8).wrapping_mul(43)).wrapping_add(7))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(11)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(23)))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_len)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(31)))
        .collect();

    assert_rgb420_pair_matches_scalar(
        "NEON multi-chunk 4:2:0 pair",
        super::BackendKind::Neon,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_uses_shortest_safe_prefix() {
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72, 80];
    let y_bot = [80u8, 88, 96, 104];
    let prev_cb = [120u8, 100, 140];
    let curr_cb = [110u8, 90, 130];
    let next_cb = [100u8, 80, 120];
    let prev_cr = [130u8, 150, 170];
    let curr_cr = [140u8, 160, 180];
    let next_cr = [150u8, 170, 190];

    let safe_width = y_bot.len();
    assert_rgb420_pair_uses_safe_prefix(
        "NEON shortest safe 4:2:0 prefix",
        super::BackendKind::Neon,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
        safe_width,
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_420_row_pair_handles_missing_bottom_row() {
    let y_top = [16u8, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96];
    let prev_cb = [120u8, 100, 140, 160, 180, 200];
    let curr_cb = [110u8, 90, 130, 170, 190, 210];
    let next_cb = [100u8, 80, 120, 180, 200, 220];
    let prev_cr = [130u8, 150, 170, 190, 210, 230];
    let curr_cr = [140u8, 160, 180, 200, 220, 240];
    let next_cr = [150u8, 170, 190, 210, 230, 250];

    assert_rgb420_pair_matches_scalar(
        "NEON top-only 4:2:0 pair",
        super::BackendKind::Neon,
        Rgb420Fixture::new(
            &y_top,
            None,
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
    );
}

#[cfg(target_arch = "aarch64")]
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the synthetic byte pattern intentionally wraps the bounded row index"
)]
fn neon_420_cropped_row_pair_matches_scalar_reference_across_chunks() {
    let width = 73usize;
    let crop_start = 3usize;
    let crop_width = 53usize;
    let chroma_width = width.div_ceil(2);
    let y_top: Vec<u8> = (0..width)
        .map(|i| ((i as u8).wrapping_mul(37)).wrapping_add(11))
        .collect();
    let y_bot: Vec<u8> = (0..width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(19)))
        .collect();
    let prev_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(17)).wrapping_add(41))
        .collect();
    let curr_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(29)).wrapping_add(13))
        .collect();
    let next_cb: Vec<u8> = (0..chroma_width)
        .map(|i| ((i as u8).wrapping_mul(43)).wrapping_add(7))
        .collect();
    let prev_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(11)))
        .collect();
    let curr_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(23)))
        .collect();
    let next_cr: Vec<u8> = (0..chroma_width)
        .map(|i| 255u8.wrapping_sub((i as u8).wrapping_mul(31)))
        .collect();

    assert_rgb420_cropped_pair_matches_scalar(
        "NEON cropped 4:2:0 pair",
        super::BackendKind::Neon,
        Rgb420Fixture::new(
            &y_top,
            Some(&y_bot),
            [&prev_cb, &curr_cb, &next_cb],
            [&prev_cr, &curr_cr, &next_cr],
        ),
        crop_start,
        crop_width,
    );
}
