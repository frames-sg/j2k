// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{J2kRect, J2kRequiredBandRegion};

fn region(x0: u32, y0: u32, x1: u32, y1: u32) -> J2kRequiredBandRegion {
    J2kRequiredBandRegion { x0, y0, x1, y1 }
}

fn rect(x0: u32, y0: u32, x1: u32, y1: u32) -> J2kRect {
    J2kRect { x0, y0, x1, y1 }
}

#[test]
fn expanded_within_rect_clamps_each_edge_in_absolute_coordinates() {
    let bounds = rect(100, 200, 150, 260);

    assert_eq!(
        region(102, 205, 145, 258).expanded_within_rect(10, bounds),
        region(100, 200, 150, 260)
    );
    assert_eq!(
        region(110, 215, 140, 250).expanded_within_rect(0, bounds),
        region(110, 215, 140, 250)
    );
}

#[test]
fn expanded_within_rect_saturates_overflow_before_clamping() {
    let maximum = u32::MAX;
    let bounds = rect(maximum - 20, maximum - 30, maximum, maximum);

    assert_eq!(
        region(maximum - 2, maximum - 3, maximum, maximum).expanded_within_rect(maximum, bounds),
        region(maximum - 20, maximum - 30, maximum, maximum)
    );
}

#[test]
fn expanded_within_rect_keeps_a_degenerate_bounding_rect_empty() {
    let empty = region(9, 11, 9, 11);
    let empty_bounds = rect(9, 11, 9, 11);

    assert_eq!(empty.expanded_within_rect(u32::MAX, empty_bounds), empty);
    assert_eq!(empty.width(), 0);
    assert_eq!(empty.height(), 0);
}

#[test]
fn union_returns_the_commutative_bounding_envelope() {
    let left = region(4, 20, 12, 40);
    let right = region(10, 5, 30, 25);
    let expected = region(4, 5, 30, 40);

    assert_eq!(left.union(right), expected);
    assert_eq!(right.union(left), expected);
    assert_eq!(left.union(left), left);
}

#[test]
fn union_handles_empty_and_maximum_boundary_regions_without_arithmetic() {
    let non_empty = region(u32::MAX - 10, u32::MAX - 8, u32::MAX, u32::MAX);
    let empty_on_boundary = region(u32::MAX, u32::MAX, u32::MAX, u32::MAX);

    assert_eq!(non_empty.union(empty_on_boundary), non_empty);
    assert_eq!(empty_on_boundary.union(non_empty), non_empty);
}

#[test]
fn to_rect_preserves_non_empty_empty_and_maximum_coordinates() {
    for required in [
        region(3, 5, 11, 17),
        region(7, 9, 7, 9),
        region(u32::MAX - 1, u32::MAX - 2, u32::MAX, u32::MAX),
    ] {
        assert_eq!(
            required.to_rect(),
            rect(required.x0, required.y0, required.x1, required.y1)
        );
    }
}

#[test]
fn from_required_region_matches_the_explicit_rect_conversion() {
    for required in [
        region(0, 0, 1, 1),
        region(42, 24, 42, 24),
        region(u32::MAX - 4, u32::MAX - 3, u32::MAX, u32::MAX),
    ] {
        assert_eq!(J2kRect::from(required), required.to_rect());
    }
}
