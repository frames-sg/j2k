// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::rust_function_policy::FunctionCalls;
use super::super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

const INFALLIBLE_GEOMETRY_GROWTH: &[&str] = &[
    "Vec::with_capacity",
    "vec",
    "collect",
    "to_vec",
    "reserve",
    "reserve_exact",
];

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn calls(label: &str, source: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse(label, source, function)
}

fn progressive_sources() -> String {
    read_source_files(
        repo_root(),
        &[
            "crates/j2k-jpeg/src/entropy/progressive.rs",
            "crates/j2k-jpeg/src/entropy/progressive/model.rs",
            "crates/j2k-jpeg/src/entropy/progressive/allocation.rs",
            "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
            "crates/j2k-jpeg/src/entropy/progressive/render.rs",
            "crates/j2k-jpeg/src/entropy/progressive/tests.rs",
        ],
    )
}

#[test]
fn progressive_coefficient_accumulator_preflights_aggregate_live_storage() {
    let allocation = read("crates/j2k-jpeg/src/entropy/progressive/allocation.rs");
    let scan = read("crates/j2k-jpeg/src/entropy/progressive/scan.rs");
    let allocate = calls(
        "progressive coefficient accumulator",
        &allocation,
        "allocate_coefficients",
    );
    allocate.assert_ordered(
        "progressive aggregate preflight before coefficient reserves",
        &[
            "validate_coefficient_workspace",
            "checked_phase_capacity",
            "try_reserve_for_len_with_live_budget",
            "checked_allocation_len",
            "try_reserve_for_len_with_live_budget",
        ],
    );
    allocate.assert_propagated(
        "progressive fallible coefficient allocation",
        &[
            "validate_coefficient_workspace",
            "checked_phase_capacity",
            "try_reserve_for_len_with_live_budget",
            "checked_allocation_len",
        ],
    );
    allocate.assert_absent(
        "progressive coefficient accumulator",
        INFALLIBLE_GEOMETRY_GROWTH,
    );

    let budget = calls(
        "progressive coefficient workspace",
        &allocation,
        "validate_coefficient_workspace",
    );
    budget.assert_contains(
        "progressive checked aggregate workspace",
        &[
            "checked_allocation_bytes",
            "checked_allocation_len",
            "checked_add_allocation_bytes",
        ],
    );
    budget.assert_propagated(
        "progressive workspace overflow propagation",
        &[
            "checked_allocation_bytes",
            "checked_allocation_len",
            "checked_add_allocation_bytes",
        ],
    );
    calls(
        "progressive scan accumulator",
        &scan,
        "decode_progressive_dct_blocks",
    )
    .assert_ordered(
        "progressive coefficient allocation before scan accumulation",
        &["allocate_coefficients", "decode_progressive_scan"],
    );
}

#[test]
fn progressive_extraction_preflights_all_live_planes_before_decode() {
    let transcode = read("crates/j2k-jpeg/src/transcode.rs");
    calls(
        "progressive DCT extraction",
        &transcode,
        "extract_dct_blocks",
    )
    .assert_ordered(
        "progressive extraction aggregate preflight",
        &[
            "validate_progressive_extraction_workspace",
            "decode_progressive_dct_blocks",
            "build_progressive_components",
        ],
    );
    let budget = calls(
        "progressive extraction workspace",
        &transcode,
        "validate_progressive_extraction_workspace",
    );
    budget.assert_contains(
        "progressive decoded and output plane accounting",
        &[
            "checked_allocation_bytes",
            "checked_add_allocation_bytes",
            "checked_allocation_len",
            "usize::from",
        ],
    );
    budget.assert_propagated(
        "progressive extraction workspace propagation",
        &[
            "checked_allocation_bytes",
            "checked_add_allocation_bytes",
            "checked_allocation_len",
        ],
    );

    let build = calls(
        "progressive extracted component construction",
        &transcode,
        "build_progressive_components",
    );
    build.assert_contains(
        "progressive fallible output construction",
        &["try_reserve_for_len_with_live_budget"],
    );
    build.assert_propagated(
        "progressive output allocation propagation",
        &["try_reserve_for_len_with_live_budget"],
    );
    build.assert_absent(
        "progressive extracted component construction",
        INFALLIBLE_GEOMETRY_GROWTH,
    );
}

#[test]
fn jpeg_allocation_failures_keep_typed_categories() {
    let allocation = read("crates/j2k-jpeg/src/allocation.rs");
    let error = read("crates/j2k-jpeg/src/error.rs");
    let progressive = progressive_sources();
    let transcode = read("crates/j2k-jpeg/src/transcode.rs");
    assert_pattern_checks(&[
        PatternCheck::new("JPEG checked allocation helpers", &allocation).required(&[
            "checked_mul(size_of::<T>())",
            "checked_add(additional)",
            "try_host_vec_with_capacity(capacity).map_err(host_allocation_error)",
            "try_host_vec_filled(len, value).map_err(host_allocation_error)",
            "JpegError::HostAllocationFailed",
            "JpegError::MemoryCapExceeded",
        ]),
        PatternCheck::new("typed JPEG allocation errors", &error)
            .required(&["MemoryCapExceeded", "HostAllocationFailed"]),
        PatternCheck::new("progressive aggregate regression", &progressive)
            .required(&["coefficient_workspace_rejects_aggregate_component_planes"]),
        PatternCheck::new("progressive extraction aggregate regression", &transcode)
            .required(&["progressive_extraction_rejects_aggregate_live_planes"]),
    ]);
}

#[test]
fn progressive_allocation_policy_stays_focused() {
    assert!(
        include_str!("progressive_policy.rs").lines().count() < 225,
        "progressive transcode allocation policy must stay focused"
    );
}
