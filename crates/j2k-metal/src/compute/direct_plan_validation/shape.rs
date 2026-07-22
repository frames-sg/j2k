// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::J2kRect;

use super::super::{
    J2kDirectStoreStep, PreparedClassicSubBand, PreparedClassicSubBandGroup, PreparedDirectIdwt,
    PreparedHtSubBand, PreparedHtSubBandGroup,
};

pub(in crate::compute) fn classic_group_shapes_match(
    first: &PreparedClassicSubBandGroup,
    other: &PreparedClassicSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

pub(in crate::compute) fn ht_group_shapes_match(
    first: &PreparedHtSubBandGroup,
    other: &PreparedHtSubBandGroup,
) -> bool {
    first.end_step == other.end_step
        && first.total_coefficients == other.total_coefficients
        && first.members.len() == other.members.len()
        && first
            .members
            .iter()
            .zip(&other.members)
            .all(|(left, right)| left.offset_elements == right.offset_elements)
}

pub(in crate::compute) fn classic_sub_band_shapes_match(
    first: &PreparedClassicSubBand,
    other: &PreparedClassicSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

pub(in crate::compute) fn ht_sub_band_shapes_match(
    first: &PreparedHtSubBand,
    other: &PreparedHtSubBand,
) -> bool {
    first.width == other.width && first.height == other.height
}

fn rect_shapes_match(first: J2kRect, other: J2kRect) -> bool {
    first.x0 == other.x0 && first.y0 == other.y0 && first.x1 == other.x1 && first.y1 == other.y1
}

pub(in crate::compute) fn idwt_shapes_match(
    first: &PreparedDirectIdwt,
    other: &PreparedDirectIdwt,
) -> bool {
    first.step.transform == other.step.transform
        && rect_shapes_match(first.step.rect, other.step.rect)
        && first.output_window.x0 == other.output_window.x0
        && first.output_window.y0 == other.output_window.y0
        && first.output_window.x1 == other.output_window.x1
        && first.output_window.y1 == other.output_window.y1
        && rect_shapes_match(first.step.ll, other.step.ll)
        && rect_shapes_match(first.step.hl, other.step.hl)
        && rect_shapes_match(first.step.lh, other.step.lh)
        && rect_shapes_match(first.step.hh, other.step.hh)
}

pub(in crate::compute) fn store_shapes_match(
    first: &J2kDirectStoreStep,
    other: &J2kDirectStoreStep,
) -> bool {
    rect_shapes_match(first.input_rect, other.input_rect)
        && first.source_x == other.source_x
        && first.source_y == other.source_y
        && first.copy_width == other.copy_width
        && first.copy_height == other.copy_height
        && first.output_width == other.output_width
        && first.output_height == other.output_height
        && first.output_x == other.output_x
        && first.output_y == other.output_y
        && first.addend.to_bits() == other.addend.to_bits()
}
