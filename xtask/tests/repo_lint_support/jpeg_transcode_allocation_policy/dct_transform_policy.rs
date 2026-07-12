// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::rust_function_policy::FunctionCalls;
use super::super::{assert_pattern_checks, repo_root, PatternCheck};

const INFALLIBLE_GEOMETRY_GROWTH: &[&str] = &[
    "Vec::with_capacity",
    "vec",
    "collect",
    "to_vec",
    "reserve",
    "reserve_exact",
    "resize",
];

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn calls(label: &str, source: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse(label, source, function)
}

#[test]
fn dct53_geometry_allocations_follow_aggregate_preflight() {
    let source = read("crates/j2k-transcode/src/dct53_2d.rs");
    let direct = calls(
        "DCT 5/3 direct projection",
        &source,
        "dct8x8_blocks_to_dwt53_float_linear_with_scratch",
    );
    direct.assert_ordered(
        "DCT 5/3 direct aggregate preflight",
        &[
            "validate_grid",
            "validate_direct_workspace",
            "ensure_sample_len",
            "try_vec_with_capacity",
        ],
    );
    direct.assert_propagated(
        "DCT 5/3 direct allocation propagation",
        &[
            "validate_direct_workspace",
            "ensure_sample_len",
            "checked_allocation_len",
            "try_vec_with_capacity",
        ],
    );

    let reference = calls(
        "DCT 5/3 reference projection",
        &source,
        "dct8x8_blocks_then_dwt53_float",
    );
    reference.assert_ordered(
        "DCT 5/3 reference aggregate preflight",
        &[
            "validate_grid",
            "checked_allocation_len",
            "validate_reference_workspace",
            "try_vec_with_capacity",
            "linearized_53_2d_from_plane",
        ],
    );
    reference.assert_propagated(
        "DCT 5/3 reference allocation propagation",
        &[
            "checked_allocation_len",
            "validate_reference_workspace",
            "try_vec_with_capacity",
        ],
    );

    let plane = calls(
        "DCT 5/3 sample-plane projection",
        &source,
        "linearized_53_2d_from_plane",
    );
    plane.assert_ordered(
        "DCT 5/3 plane aggregate preflight",
        &[
            "checked_allocation_len",
            "validate_plane_workspace",
            "try_vec_with_capacity",
            "try_vec_filled",
        ],
    );
    for (label, function) in [
        ("DCT 5/3 direct projection", direct),
        ("DCT 5/3 reference projection", reference),
        ("DCT 5/3 plane projection", plane),
    ] {
        function.assert_absent(label, INFALLIBLE_GEOMETRY_GROWTH);
    }
}

#[test]
fn dct53_symbolic_rows_use_shared_codec_math_without_the_basis_loop() {
    let source = read("crates/j2k-transcode/src/dct53_2d.rs");
    calls(
        "DCT 5/3 symbolic weight rows",
        &source,
        "write_symbolic_weight_rows",
    )
    .assert_ordered(
        "DCT 5/3 shared symbolic row construction",
        &["linearized_dwt53_row", "taps", "push_weight_tap"],
    );
    calls("DCT 5/3 weight row growth", &source, "resize_weight_rows").assert_ordered(
        "DCT 5/3 fallible row and tap reservation",
        &["try_vec_resize_with", "clear", "try_vec_reserve_len"],
    );
    assert_pattern_checks(&[PatternCheck::new("DCT 5/3 symbolic source", &source)
        .required(&[
            "use j2k_codec_math::dwt::{",
            "linearized_dwt53_row",
            "DWT53_MAX_HIGH_LINEAR_TAPS",
            "DWT53_MAX_LINEAR_TAPS",
        ])
        .forbidden(&[
            "let mut basis = vec![0.0; sample_len]",
            "for sample_idx in 0..sample_len",
            "fn linearized_53_from_sample_slice(",
            "fn transpose_band(",
            "fn column_from_rows(",
        ])]);
}

#[test]
fn dct97_geometry_allocations_follow_aggregate_preflight() {
    let source = read("crates/j2k-transcode/src/dct97_2d.rs");
    let grid = calls(
        "DCT 9/7 grid projection",
        &source,
        "dct8x8_blocks_then_dwt97_float_with_scratch",
    );
    grid.assert_ordered(
        "DCT 9/7 grid aggregate preflight",
        &[
            "validate_grid",
            "checked_allocation_len",
            "validate_grid_workspace",
            "try_vec_resize_with",
        ],
    );
    let plane = calls(
        "DCT 9/7 plane projection",
        &source,
        "linearized_97_2d_from_plane_with_scratch",
    );
    plane.assert_ordered(
        "DCT 9/7 plane aggregate preflight",
        &[
            "validate_sample_plane",
            "validate_plane_workspace",
            "linearized_97_2d_from_plane_with_plane_scratch",
        ],
    );
    let storage = calls(
        "DCT 9/7 plane storage",
        &source,
        "linearized_97_2d_from_plane_with_plane_scratch",
    );
    storage.assert_contains(
        "DCT 9/7 fallible geometry storage",
        &[
            "try_vec_resize_with",
            "try_vec_reserve_len",
            "try_vec_filled",
        ],
    );
    for (label, function) in [
        ("DCT 9/7 grid projection", grid),
        ("DCT 9/7 plane projection", plane),
        ("DCT 9/7 plane storage", storage),
    ] {
        function.assert_absent(label, INFALLIBLE_GEOMETRY_GROWTH);
    }

    for function in [
        "linearized_97_split_contiguous_into",
        "linearized_97_split_strided_into",
    ] {
        let split = calls("DCT 9/7 split workspace", &source, function);
        split.assert_ordered(
            "DCT 9/7 split reserve-before-growth",
            &["clear", "try_vec_reserve_len"],
        );
        split.assert_absent("DCT 9/7 split workspace", INFALLIBLE_GEOMETRY_GROWTH);
    }
}

#[test]
fn dct_allocation_failures_keep_typed_transform_and_transcode_categories() {
    let allocation = read("crates/j2k-transcode/src/allocation.rs");
    let transform_error = read("crates/j2k-transcode/src/dct_grid.rs");
    let transcode_error = read("crates/j2k-transcode/src/jpeg_to_htj2k/error.rs");
    let dct53 = read("crates/j2k-transcode/src/dct53_2d.rs");
    let dct97 = read("crates/j2k-transcode/src/dct97_2d.rs");
    assert_pattern_checks(&[
        PatternCheck::new("DCT allocation error conversion", &allocation).required(&[
            "impl From<TranscodeAllocationError> for DctTransformError",
            "Self::MemoryCapExceeded { requested, cap }",
            "Self::HostAllocationFailed { bytes }",
        ]),
        PatternCheck::new("typed DCT transform error", &transform_error).required(&[
            "pub enum DctTransformError",
            "MemoryCapExceeded",
            "HostAllocationFailed",
            "impl From<DctGridError> for DctTransformError",
        ]),
        PatternCheck::new("typed JPEG transcode DCT mapping", &transcode_error).required(&[
            "fn map_transform_error(",
            "DctTransformError::MemoryCapExceeded { requested, cap }",
            "JpegToHtj2kError::MemoryCapExceeded { requested, cap }",
            "DctTransformError::HostAllocationFailed { bytes }",
            "JpegToHtj2kError::HostAllocationFailed { bytes }",
        ]),
        PatternCheck::new("DCT 5/3 aggregate regression", &dct53)
            .required(&["reference_workspace_rejects_aggregate_before_any_single_vector_hits_cap"]),
        PatternCheck::new("DCT 9/7 aggregate regression", &dct97)
            .required(&["grid_workspace_rejects_aggregate_before_any_single_vector_hits_cap"]),
    ]);
}

#[test]
fn dct_transform_policy_stays_focused() {
    assert!(
        include_str!("dct_transform_policy.rs").lines().count() < 275,
        "DCT transform allocation policy must stay focused"
    );
}
