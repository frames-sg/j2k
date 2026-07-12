// SPDX-License-Identifier: MIT OR Apache-2.0

use super::plan::{tag_tree_node_count, IdwtWorkspaceTracker};
use super::reuse::{discard_stale_capacity, ReallocationBudget};
use super::{
    release_unused_roi_workspace, roi_workspace_bytes, BuildWorkspace, DecompositionAllocationPlan,
    Segment, DEFAULT_MAX_DECODE_BYTES,
};
use crate::error::{DecodeError, ValidationError};
use crate::j2c::build::{Decomposition, SubBand, SubBandType};
use crate::j2c::decode::{DecodeAllocationBudget, DecompositionStorage};
use crate::j2c::rect::IntRect;
use crate::j2c::tag_tree::TagTree;
use alloc::vec::Vec;
use core::mem::size_of;

fn coefficient_plan(baseline: usize, coefficients: usize) -> DecompositionAllocationPlan {
    let mut plan =
        DecompositionAllocationPlan::new(baseline, false, false, BuildWorkspace::CoefficientsOnly)
            .expect("baseline");
    plan.add_coefficients(coefficients).expect("target count");
    plan
}

#[test]
fn allocation_plan_tag_tree_count_matches_builder_for_edge_shapes() {
    for (width, height) in [
        (0, 0),
        (0, 4),
        (4, 0),
        (1, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (3, 5),
        (8, 8),
    ] {
        let mut nodes = Vec::new();
        let _tree = TagTree::new(width, height, &mut nodes);
        assert_eq!(
            nodes.len(),
            tag_tree_node_count(width, height).unwrap(),
            "{width}x{height} tag tree"
        );
    }
}

#[test]
fn same_size_second_tile_reuses_capacity_without_charging_a_second_owner() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(2)
        .expect("small test capacity");
    let target = storage.coefficients.capacity();
    let coefficient_bytes = target * size_of::<f32>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - coefficient_bytes;
    let mut plan = coefficient_plan(baseline, target);

    discard_stale_capacity(&mut storage, &plan);
    assert_eq!(storage.coefficients.capacity(), target);
    plan.account_live_workspace(&storage, 0)
        .expect("same-size reuse fits exactly");
    assert_eq!(plan.total_bytes, DEFAULT_MAX_DECODE_BYTES);

    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    reallocation
        .reserve(&mut storage.coefficients, target)
        .expect("same target is allocation-free");
    assert_eq!(storage.coefficients.capacity(), target);
}

#[test]
fn stale_capacity_is_released_even_when_only_a_later_phase_would_need_the_space() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(4)
        .expect("small test capacity");
    storage
        .segments
        .try_reserve_exact(2)
        .expect("small segment capacity");
    let baseline = DEFAULT_MAX_DECODE_BYTES - 2 * size_of::<f32>();
    let mut plan = coefficient_plan(baseline, 1);

    // The logical build has room without consulting a failure fallback, but
    // keeping either stale owner would steal capacity from packet/decode work.
    discard_stale_capacity(&mut storage, &plan);
    assert_eq!(storage.coefficients.capacity(), 0);
    assert_eq!(storage.segments.capacity(), 0);
    plan.account_live_workspace(&storage, 0)
        .expect("logical request remains admissible");
}

#[test]
fn retained_segment_capacity_is_added_once_outside_the_structural_total() {
    let mut storage = DecompositionStorage::default();
    storage
        .segments
        .try_reserve_exact(2)
        .expect("small test capacity");
    let segment_bytes = storage.segments.capacity() * size_of::<Segment<'_>>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - segment_bytes;
    let mut plan = coefficient_plan(baseline, 0);

    plan.account_live_workspace(&storage, 0)
        .expect("segment capacity fits once");
    assert_eq!(plan.total_bytes, baseline);
    storage.structural_workspace_bytes = plan.total_bytes;
    DecodeAllocationBudget::for_storage(&storage)
        .expect("downstream adds the separate segment owner once");
}

#[test]
fn roi_release_restores_the_exact_reserved_boundary() {
    let sub_bands = 7;
    let decompositions = 3;
    let components = 2;
    let bytes = roi_workspace_bytes(sub_bands, decompositions, components).expect("ROI bytes");
    let mut storage = DecompositionStorage::default();
    let empty = IntRect::from_xywh(0, 0, 0, 0);
    for _ in 0..sub_bands {
        storage.sub_bands.push(SubBand {
            sub_band_type: SubBandType::LowLow,
            rect: empty,
            precincts: 0..0,
            coefficients: 0..0,
        });
    }
    for _ in 0..decompositions {
        storage.decompositions.push(Decomposition {
            sub_bands: [0; 3],
            rect: empty,
        });
    }
    storage.structural_workspace_bytes = DEFAULT_MAX_DECODE_BYTES;
    release_unused_roi_workspace(&mut storage, components).expect("planned ROI bytes were present");
    assert_eq!(
        storage.structural_workspace_bytes,
        DEFAULT_MAX_DECODE_BYTES - bytes
    );
    let mut budget = DecodeAllocationBudget::from_live_bytes(storage.structural_workspace_bytes)
        .expect("released baseline");
    budget
        .include_elements::<u8>(bytes)
        .expect("exact released boundary is reusable");
    assert!(budget.include_elements::<u8>(1).is_err());
}

#[test]
fn skipped_resolution_idwt_workspace_uses_padded_active_high_water() {
    let ll = IntRect::from_xywh(0, 0, 2, 2);
    let rects = [
        IntRect::from_xywh(0, 0, 4, 3),
        IntRect::from_xywh(0, 0, 8, 5),
        IntRect::from_xywh(0, 0, 16, 9),
    ];

    let mut skipped_one = IdwtWorkspaceTracker::new(Some(2));
    for rect in rects {
        skipped_one.observe(rect);
    }
    assert_eq!(skipped_one.finish(ll).unwrap(), Some(54 + 20));

    let mut full = IdwtWorkspaceTracker::new(Some(3));
    for rect in rects {
        full.observe(rect);
    }
    assert_eq!(full.finish(ll).unwrap(), Some(170 + 54));

    let mut ll_only = IdwtWorkspaceTracker::new(Some(0));
    for rect in rects {
        ll_only.observe(rect);
    }
    assert_eq!(ll_only.finish(ll).unwrap(), Some(4));
}

#[test]
fn impossible_metadata_count_rejects_at_the_logical_preflight() {
    let mut plan = coefficient_plan(0, 0);
    plan.layers = DEFAULT_MAX_DECODE_BYTES / size_of::<super::Layer>() + 1;
    assert_eq!(
        plan.validate_minimum_live_workspace(0),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn reusable_growth_drops_the_empty_old_buffer_at_the_transient_boundary() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(1)
        .expect("small test capacity");
    let target = storage.coefficients.capacity() + 1;
    let baseline = DEFAULT_MAX_DECODE_BYTES - target * size_of::<f32>();
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");

    reallocation
        .reserve(&mut storage.coefficients, target)
        .expect("dropping the empty old buffer makes the exact final target fit");
    assert!(storage.coefficients.capacity() >= target);
}

#[test]
fn reusable_capacity_growth_rejects_a_logical_target_over_the_aggregate_cap() {
    let mut storage = DecompositionStorage::default();
    storage
        .coefficients
        .try_reserve_exact(1)
        .expect("small test capacity");
    let coefficient_bytes = storage.coefficients.capacity() * size_of::<f32>();
    let baseline = DEFAULT_MAX_DECODE_BYTES - coefficient_bytes;
    let mut reallocation =
        ReallocationBudget::for_storage(&storage, baseline).expect("live baseline");
    let target = storage.coefficients.capacity() + 1;

    assert_eq!(
        reallocation.reserve(&mut storage.coefficients, target),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}
