// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::JpegAllocationSources;
use super::calls;

pub(super) fn assert_policy(sources: &JpegAllocationSources) {
    assert_packet_owners_remain_move_only(sources);
    assert_packet_build_contracts(sources);
    assert_packet_regressions(sources);
}

fn assert_packet_owners_remain_move_only(sources: &JpegAllocationSources) {
    for packet in [
        "JpegFast420PacketV1",
        "JpegFast422PacketV1",
        "JpegFast444PacketV1",
        "JpegGrayPacketV1",
    ] {
        let declaration = format!("pub struct {packet}");
        let (prefix, _) = sources
            .fast_packet_types
            .split_once(&declaration)
            .unwrap_or_else(|| panic!("missing fast-packet owner {packet}"));
        let derive = prefix
            .rsplit("#[derive(")
            .next()
            .unwrap_or_else(|| panic!("missing derive for {packet}"));
        assert!(
            derive.starts_with("Debug, PartialEq, Eq)]"),
            "{packet} must remain move-only because its entropy and metadata vectors can approach the shared host cap"
        );
        assert!(
            !derive.starts_with("Debug, Clone") && !derive.starts_with("Clone"),
            "{packet} must not regain infallible Clone"
        );
    }
}

fn assert_packet_build_contracts(sources: &JpegAllocationSources) {
    let inspect = calls(
        "JPEG color fast-packet builder",
        &sources.fast_packet_build,
        "build_color_fast_packet",
    );
    inspect.assert_ordered(
        "JPEG color packet single-parse handoff",
        &[
            "JpegView::parse",
            "ColorFastHeader::inspect",
            "Decoder::from_view",
            "build_color_fast_packet_from_decoder",
        ],
    );
    let build = calls(
        "JPEG color fast-packet builder",
        &sources.fast_packet_build,
        "build_color_fast_packet_from_decoder",
    );
    build.assert_ordered(
        "JPEG color packet lifecycle",
        &[
            "validate_scan_bytes",
            "inspect_entropy_segments_allow_missing_eoi",
            "retained_decoder_allocation_bytes",
            "terminated_copy_len",
            "checked_color_packet_live_bytes",
            "build_fast_entropy_checkpoints",
            "terminated_with_live_budget",
            "scan_live_bytes",
            "extract_entropy_segments_from_layout",
        ],
    );
    inspect.assert_absent(
        "JPEG color packet duplicate parsing",
        &["Decoder::new", "parse_header"],
    );
    assert!(
        sources
            .fast_packet_checkpoints
            .contains("build_checkpoint_plan_mapped_from_validated_with_live_budget(")
            && sources
                .fast_packet_checkpoints
                .contains("#[cfg(test)]\npub(super) fn packet_checkpoints_from_device(")
            && sources
                .fast_packet_allocation
                .contains("checked_actual_vec_live_bytes::<T>(")
            && sources.fast_packet_allocation.contains("values.capacity()")
            && !sources.fast_packet_build.contains("parse_header(")
            && !sources.fast_packet_build.contains("Decoder::new("),
        "fast packets must construct the final checkpoint form directly under actual-capacity checks"
    );
    assert_eq!(
        sources
            .fast_packet_entropy
            .matches("fn inspect_entropy_segments_with_missing_eoi(")
            .count(),
        1,
        "entropy inspection must share one implementation"
    );
}

fn assert_packet_regressions(sources: &JpegAllocationSources) {
    let tests = format!(
        "{}{}{}{}",
        sources.fast_packet_allocation_tests,
        sources.fast_packet_behavior_tests,
        sources.fast_packet_checkpoint_tests,
        sources.fast_packet_source_tests
    );
    for regression in [
        "entropy_and_packet_live_byte_boundaries_are_exact",
        "allocator_returned_packet_capacity_is_postchecked",
        "packets_accept_missing_eoi_without_changing_materialized_entropy",
        "malformed_packet_input_returns_an_error",
        "direct_packet_checkpoints_match_device_checkpoint_conversion",
        "packet_builders_reuse_their_single_parsed_view",
    ] {
        assert!(
            tests.contains(regression),
            "missing fast-packet regression {regression}"
        );
    }
}
