// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::JpegAllocationSources;
use super::{assert_ordered, calls};

pub(super) fn assert_policy(sources: &JpegAllocationSources) {
    calls(
        "JPEG checkpoint builder",
        &sources.checkpoint_build,
        "build_checkpoint_plan_mapped_from_validated_with_live_budget",
    )
    .assert_ordered(
        "JPEG checkpoint allocation and traversal",
        &[
            "plan_checkpoint_build_from_validated",
            "try_checkpoint_vec_with_live_budget",
            "terminated_with_live_budget",
            "BitReader::new",
            "push_planned_checkpoint",
        ],
    );
    assert!(
        sources
            .checkpoint_planning
            .contains("checked_checkpoint_phase_bytes::<T>(")
            && sources
                .checkpoint_allocation
                .contains("checked_actual_checkpoint_live_bytes::<T>(")
            && sources
                .checkpoint_cache_allocation
                .contains("retained_baseline_bytes")
            && sources
                .checkpoint_cache_allocation
                .contains("*checkpoints = Vec::new();")
            && sources.checkpoint_allocation.contains("values.capacity()")
            && sources
                .checkpoint_eoi
                .contains("struct ValidatedScanBytes<'a>")
            && sources.checkpoint_eoi.contains("reader_bytes.capacity()")
            && sources
                .checkpoint_build
                .contains("plan.dc_table(component)?")
            && sources
                .checkpoint_build
                .contains("plan.ac_table(component)?")
            && !sources.checkpoint_eoi.contains("fn entropy_eoi_end("),
        "checkpoint planning must count aggregate live bytes and allocator-returned capacities"
    );
    assert_ordered(
        "JPEG lazy checkpoint cache",
        &sources.checkpoint_cache,
        &[
            "fn checkpoint_before_mcu(",
            "let required_capacity =",
            "reserve_checkpoint_capacity(",
            "if cache.checkpoints.is_empty()",
            "extend_non_restart_checkpoints(",
        ],
    );

    let tests = format!(
        "{}{}{}",
        sources.checkpoint_allocation_tests,
        sources.checkpoint_build_tests,
        sources.checkpoint_eoi_tests
    );
    for regression in [
        "checkpoint_and_terminated_reader_live_byte_boundary_is_exact",
        "allocator_returned_checkpoint_capacity_is_postchecked",
        "checkpoint_cache_growth_counts_decoder_baseline_old_and_replacement_exactly",
        "failed_actual_cache_postcheck_releases_the_retained_allocation",
        "checkpoint_plan_applies_the_combined_cap_to_a_missing_eoi_copy",
        "terminated_scan_appends_only_the_missing_eoi_bytes",
    ] {
        assert!(
            tests.contains(regression),
            "missing checkpoint regression {regression}"
        );
    }
}
