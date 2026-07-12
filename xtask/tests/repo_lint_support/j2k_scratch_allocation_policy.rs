// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible aggregate ownership for reusable J2K row-decode scratch.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn j2k_row_scratch_preflights_the_aggregate_and_reserves_fallibly() {
    let scratch = read("crates/j2k/src/scratch.rs");
    assert_pattern_checks(&[
        PatternCheck::new("J2K row scratch aggregate plan", &scratch).required(&[
            "fn checked_scratch_bytes(",
            "let requested = checked_scratch_bytes(packed_len, row_len)?;",
            "if requested > cap",
            "BufferError::AllocationTooLarge",
            "fn ensure_scratch_capacity_within_cap(",
            "let planned_row_capacity = self.row_u16.capacity().max(row_len);",
            "self.packed_bytes.capacity()",
            "self.row_u16.capacity()",
        ]),
        PatternCheck::new("J2K row scratch fallible growth", &scratch)
            .required(&[
                "fn reserve_replacing_before_growth<T>(",
                "*values = Vec::new();",
                ".try_reserve_exact(target_len)",
                "BufferError::HostAllocationFailed",
            ])
            .forbidden(&["Vec::with_capacity", "vec!["]),
    ]);
    let packed_growth = scratch
        .find("&mut self.packed_bytes,")
        .expect("packed scratch growth");
    let capacity_reconciliation = scratch
        .find("let planned_row_capacity =")
        .expect("packed allocator-capacity reconciliation");
    let row_growth = scratch
        .find("&mut self.row_u16,")
        .expect("u16 row scratch growth");
    assert!(
        packed_growth < capacity_reconciliation && capacity_reconciliation < row_growth,
        "allocator-reported packed capacity must be reconciled before any u16-row growth"
    );
}

#[test]
fn j2k_row_decode_propagates_typed_scratch_failures_before_sink_callbacks() {
    let rows = read("crates/j2k/src/view/rows.rs");
    let scratch = read("crates/j2k/src/scratch.rs");
    assert_pattern_checks(&[
        PatternCheck::new("J2K row scratch propagation", &rows).required(&[
            ".packed_bytes(max_stripe_len)",
            ".packed_bytes_and_row_u16(max_stripe_len, samples_per_row)",
            ".checked_sub(row_scratch_bytes)",
            "DecodeRowsError::Decode(J2kError::Buffer(error))",
        ]),
        PatternCheck::new("J2K row scratch boundary regressions", &scratch).required(&[
            "aggregate_row_scratch_has_an_exact_byte_boundary",
            "stale_capacity_is_released_before_a_mixed_scratch_request",
            "allocator_capacity_overage_is_reconciled_before_row_growth",
            "row_scratch_overflow_is_typed_before_mutation",
        ]),
    ]);
}

#[test]
fn j2k_view_owners_stay_split_by_decode_responsibility() {
    let root = repo_root();
    let facade = read("crates/j2k/src/view.rs");
    assert_pattern_checks(
        &[PatternCheck::new("J2K view responsibility split", &facade)
            .required(&["mod rows;", "mod traits;"])
            .forbidden(&["impl TileBatchDecode", "fn bounded_row_stripe_layout("])],
    );

    for (relative, max_lines) in [
        ("crates/j2k/src/view.rs", 550),
        ("crates/j2k/src/view/rows.rs", 330),
        ("crates/j2k/src/view/traits.rs", 210),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below {max_lines} lines"
        );
    }
}
