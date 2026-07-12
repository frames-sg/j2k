// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, calls, JpegTranscodeSources, PatternCheck};

#[test]
fn jpeg_transcode_workspace_preflight_precedes_decode_owned_allocations() {
    let sources = JpegTranscodeSources::read();
    calls(
        &sources,
        "single JPEG transcode preparation",
        "prepare_single_transcode",
    )
    .assert_ordered(
        "single JPEG header workspace preflight",
        &[
            "validate_transcode_options",
            "validate_jpeg_transcode_workspace",
            "extract_dct_blocks",
        ],
    );
    for function in ["prepare_integer_batch_tile", "prepare_float97_batch_tile"] {
        calls(&sources, "JPEG batch tile preparation", function).assert_ordered(
            "JPEG batch tile workspace preflight",
            &["validate_jpeg_transcode_workspace", "extract_dct_blocks"],
        );
    }
    calls(
        &sources,
        "JPEG batch transcode",
        "jpeg_tile_batch_to_htj2k_with_scratch",
    )
    .assert_ordered(
        "JPEG batch aggregate workspace preflight",
        &[
            "validate_batch_route",
            "try_vec_with_capacity",
            "prepare_integer_batch_tile",
        ],
    );
    calls(
        &sources,
        "JPEG batch route validation",
        "validate_batch_route",
    )
    .assert_ordered(
        "JPEG batch route validates options and aggregate workspace",
        &["validate_transcode_options", "validate_batch_workspace"],
    );
    calls(&sources, "JPEG batch workspace", "validate_batch_workspace").assert_ordered(
        "JPEG batch workspace aggregation",
        &["validate_jpeg_transcode_workspace", "batch_workspace_bytes"],
    );
    calls(
        &sources,
        "JPEG batch workspace bytes",
        "batch_workspace_bytes",
    )
    .assert_ordered(
        "JPEG batch fixed metadata before valid tile peaks",
        &["fixed_metadata_peak", "checked_add_allocation_bytes"],
    );

    let workspace = calls(&sources, "JPEG workspace model", "workspace_from_info");
    workspace.assert_ordered(
        "JPEG workspace checked allocation model",
        &[
            "workspace_geometry",
            "checked_allocation_bytes",
            "checked_add_allocation_bytes",
            "ensure_allocation_bytes",
        ],
    );
    workspace.assert_propagated(
        "JPEG workspace checked allocation model",
        &[
            "workspace_geometry",
            "checked_allocation_bytes",
            "checked_add_allocation_bytes",
            "ensure_allocation_bytes",
        ],
    );
    calls(&sources, "JPEG workspace geometry", "workspace_geometry").assert_contains(
        "JPEG workspace overflow checks",
        &["checked_mul", "checked_add"],
    );

    assert_pattern_checks(&[PatternCheck::new(
        "JPEG workspace preflight regressions",
        &sources.full_combined(),
    )
    .required(&[
        "public_transcode_rejects_huge_header_geometry_before_entropy_allocation",
        "batch_preserves_an_individually_oversized_header_as_a_tile_error",
        "batch_rejects_valid_header_plans_that_exceed_the_aggregate_cap",
        "huge_sof_geometry_is_rejected_before_dct_extraction",
    ])]);
}
