// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::{fill_rgb_row_pair_from_420, fill_rgb_row_pair_from_420_cropped};
use crate::backend::{scalar, Rgb420ChromaRows, Rgb420Crop, Rgb420CroppedRowPair, Rgb420RowPair};

struct TopOnlyFixture {
    y: Vec<u8>,
    prev_cb: Vec<u8>,
    curr_cb: Vec<u8>,
    next_cb: Vec<u8>,
    prev_cr: Vec<u8>,
    curr_cr: Vec<u8>,
    next_cr: Vec<u8>,
}

impl TopOnlyFixture {
    fn new(width: usize) -> Self {
        let chroma_width = width.div_ceil(2);
        let fixture_byte = |index| u8::try_from(index).expect("fixture index must fit in u8");
        Self {
            y: (0..width)
                .map(|index| fixture_byte(index).wrapping_mul(37).wrapping_add(11))
                .collect(),
            prev_cb: (0..chroma_width)
                .map(|index| fixture_byte(index).wrapping_mul(17).wrapping_add(41))
                .collect(),
            curr_cb: (0..chroma_width)
                .map(|index| fixture_byte(index).wrapping_mul(29).wrapping_add(13))
                .collect(),
            next_cb: (0..chroma_width)
                .map(|index| fixture_byte(index).wrapping_mul(43).wrapping_add(7))
                .collect(),
            prev_cr: (0..chroma_width)
                .map(|index| 255u8.wrapping_sub(fixture_byte(index).wrapping_mul(11)))
                .collect(),
            curr_cr: (0..chroma_width)
                .map(|index| 255u8.wrapping_sub(fixture_byte(index).wrapping_mul(23)))
                .collect(),
            next_cr: (0..chroma_width)
                .map(|index| 255u8.wrapping_sub(fixture_byte(index).wrapping_mul(31)))
                .collect(),
        }
    }

    fn chroma(&self) -> Rgb420ChromaRows<'_> {
        Rgb420ChromaRows::new(
            &self.prev_cb,
            &self.curr_cb,
            &self.next_cb,
            &self.prev_cr,
            &self.curr_cr,
            &self.next_cr,
        )
    }
}

fn top_only_rows<'a>(fixture: &'a TopOnlyFixture, dst: &'a mut [u8]) -> Rgb420RowPair<'a> {
    Rgb420RowPair::new(&fixture.y, None, fixture.chroma(), dst, None)
}

#[test]
fn cropped_top_only_neon_matches_scalar_across_alignment_and_partial_chunks() {
    let fixture = TopOnlyFixture::new(73);

    for (crop_start, crop_width) in [(3, 53), (0, 31), (2, 14)] {
        let crop = Rgb420Crop::new(crop_start, crop_width);
        let mut expected = vec![0u8; crop_width * 3];
        scalar::fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            top_only_rows(&fixture, &mut expected),
            crop,
        ));

        let mut actual = vec![0u8; crop_width * 3];
        fill_rgb_row_pair_from_420_cropped(Rgb420CroppedRowPair::new(
            top_only_rows(&fixture, &mut actual),
            crop,
        ));

        assert_eq!(
            actual, expected,
            "top-only crop differs at start {crop_start}, width {crop_width}"
        );
    }
}

#[test]
fn top_only_neon_tail_matches_scalar_for_partial_and_full_chunks() {
    for width in [31, 32] {
        let fixture = TopOnlyFixture::new(width);
        let mut expected = vec![0u8; width * 3];
        scalar::fill_rgb_row_pair_from_420(top_only_rows(&fixture, &mut expected));

        let mut actual = vec![0u8; width * 3];
        fill_rgb_row_pair_from_420(top_only_rows(&fixture, &mut actual));

        assert_eq!(actual, expected, "top-only tail differs at width {width}");
    }
}
