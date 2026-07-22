// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible live-allocation contract for JPEG output and decode scratch.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_owned_output_and_public_reusable_buffer_remain_fallible() {
    let root = repo_root();
    let output_format =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/output_format.rs"))
            .expect("read JPEG output-format source");
    let decoder = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder.rs"))
        .expect("read JPEG decoder module");
    let decode_allocation =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/allocation.rs"))
            .expect("read JPEG decode-allocation source");
    let output_buffer = fs::read_to_string(root.join("crates/j2k-jpeg/src/output_buffer.rs"))
        .expect("read JPEG output-buffer source");
    let warning_ownership =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/warning_ownership.rs"))
            .expect("read JPEG warning-ownership source");
    let tile = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/tile.rs"))
        .expect("read JPEG tile planning source");
    let core_error = fs::read_to_string(root.join("crates/j2k-core/src/error.rs"))
        .expect("read shared buffer errors");

    assert_pattern_checks(&[
        PatternCheck::new("owned JPEG output allocation", &output_format)
            .required(&[
                "try_vec_filled(len, 0)",
                "fn allocate_output_buffer_with_live_budget(",
                "checked_live_phase_bytes(*live_bytes, len, cap)?;",
                "checked_live_phase_bytes(*live_bytes, output.capacity(), cap)?;",
            ])
            .forbidden(&["alloc::vec![0; len]", "set_len(len)"]),
        PatternCheck::new("reusable JPEG output allocation", &output_buffer)
            .required(&[
                "try_host_vec_filled(len, 0)",
                "self.bytes.capacity() > max_bytes",
                "self.clear_storage();",
                "ensure_output_capacity(bytes.capacity(), max_bytes)?;",
                "BufferError::HostAllocationFailed",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone)]",
                "bytes: alloc::vec![0; len]",
                "try_host_vec_resize(&mut self.bytes, len, 0)",
            ]),
        PatternCheck::new("shared typed buffer allocation failure", &core_error).required(&[
            "#[non_exhaustive]",
            "HostAllocationFailed {",
            "bytes: usize",
            "what: &'static str",
        ]),
        PatternCheck::new("focused JPEG live-allocation owner", &decoder)
            .required(&["mod allocation;"]),
        PatternCheck::new(
            "JPEG retained metadata and phase budget",
            &decode_allocation,
        )
        .required(&[
            "fn decode_workspace_cap(",
            "fn decode_phase_live_bytes(",
            "external_live_bytes",
            "self.decode_scratch_bytes(workspace_cap)?",
            "fn compute_progressive_scratch_bytes(",
            "fn checked_workspace_add(",
            "warning_merge_peak_bytes(warning_capacity)?",
        ]),
        PatternCheck::new("JPEG warning-owner capacity sequence", &warning_ownership)
            .required(&[
                "MAX_DECODE_SCAN_WARNINGS",
                "fn merged_warnings(",
                "fn try_clone_warnings(",
                "try_reserve_for_len_with_live_budget(",
                "fn merged_warning_capacity_bytes(",
                "fn warning_merge_peak_bytes(",
                "fn ensure_warning_capacity_peak(",
                "warning_capacity_peak_accepts_exact_and_rejects_one_over",
                "warning_merge_preserves_values_under_the_shared_capacity",
            ])
            .forbidden(&[
                "try_vec_with_capacity(warning_count)",
                "try_vec_with_capacity(warnings.len())",
            ]),
        PatternCheck::new("JPEG batch warning-result claim", &tile)
            .required(&["merged_warning_capacity_bytes(decoder.warnings.capacity())?"]),
    ]);

    assert!(
        warning_ownership.lines().count() < 165,
        "JPEG warning ownership must stay below its focused-module ratchet"
    );
}

#[test]
fn jpeg_pool_and_render_storage_keep_one_fallible_live_budget() {
    let root = repo_root();
    let pool = fs::read_to_string(root.join("crates/j2k-jpeg/src/internal/scratch.rs"))
        .expect("read JPEG scratch pool");
    let routing = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing.rs"))
        .expect("read JPEG decode routing");
    let owned_output =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/routing/owned_output.rs"))
            .expect("read JPEG owned-output routing");
    let routing = format!("{routing}\n{owned_output}");
    let sequential = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/sequential.rs"))
        .expect("read JPEG sequential routing");
    let writer_scratch =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/core_traits.rs"))
            .expect("read JPEG writer scratch source");
    let render_sources = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/lossless_render.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes/allocation.rs",
        ],
    );
    let external_temp_sources = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/lossless_render.rs",
            "crates/j2k-jpeg/src/decoder/lossless_region.rs",
            "crates/j2k-jpeg/src/decoder/extended12/rgba.rs",
        ],
    );
    let progressive = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/progressive.rs",
            "crates/j2k-jpeg/src/entropy/progressive/model.rs",
            "crates/j2k-jpeg/src/entropy/progressive/allocation.rs",
            "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
            "crates/j2k-jpeg/src/entropy/progressive/render.rs",
        ],
    );
    let sequential_dct = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/dct.rs",
            "crates/j2k-jpeg/src/entropy/sequential/dct/allocation.rs",
        ],
    );

    assert_pool_routing_and_reconciliation(&pool, &routing, &sequential, &writer_scratch);
    assert_render_plane_budget(&render_sources);
    assert_pattern_checks(&[
        PatternCheck::new(
            "nested lossless and 12-bit external output budget",
            &external_temp_sources,
        )
        .required(&[
            "external_live_bytes: usize",
            "decode_phase_live_bytes(external_live_bytes)?",
            "allocate_output_buffer_with_live_budget(",
            "let nested_external_live_bytes =",
            "checked_live_phase_bytes(external_live_bytes, rgb.capacity(), workspace_cap)?;",
        ])
        .forbidden(&[
            "allocate_output_buffer(full_len)?",
            "allocate_output_buffer(rgb_len)?",
        ]),
        PatternCheck::new("progressive decode phase allocation", &progressive)
            .required(&[
                "external_live_bytes: usize",
                "decode_progressive_dct_blocks(plan, bytes, external_live_bytes)?",
                "checked_phase_capacity(",
                "drop(coeffs);",
                "fn validate_component_image_workspace(",
                "fn component_image_capacity_bytes(",
                "try_reserve_for_len_with_live_budget(",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone)]\npub(crate) struct ProgressiveDctBlocks",
                "let mut dc_predictors = try_vec_filled",
                "let mut images = try_vec_with_capacity",
                "let mut a = try_vec_filled",
            ]),
        PatternCheck::new("sequential DCT decode phase allocation", &sequential_dct)
            .required(&[
                "try_reserve_for_len_with_live_budget(",
                "fn validate_actual_dct_lifecycle(",
                "pub(crate) fn capacity_bytes(&self)",
                "pub(crate) fn plane_capacity_bytes(&self)",
                "lifecycle.workspace_cap",
                "drop(prev_dc);",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub(crate) struct DecodedDctBlocks",
                "try_vec_filled(component_count",
                "try_vec_with_capacity(component_count)",
            ]),
    ]);
}

fn assert_render_plane_budget(render_sources: &str) {
    assert_pattern_checks(&[PatternCheck::new(
        "lossless and 12-bit plane allocation",
        render_sources,
    )
    .required(&[
        "if layout.total_bytes > self.plan.scratch_bytes",
        "try_reserve_for_len_with_live_budget(",
        "fn preflight_plane_specs",
        "fn allocate_plane(",
        "dct_blocks.capacity_bytes()",
        "ensure_progressive12_coefficient_capacities",
    ])
    .forbidden(&[
        "let mut c0 = vec![",
        "pixels: vec![0u16;",
        "try_vec_filled(layout.luma_len, P::default())?",
        "ensure_sampled_plane_capacities",
    ])]);
}

fn assert_pool_routing_and_reconciliation(
    pool: &str,
    routing: &str,
    sequential: &str,
    writer_scratch: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("reusable JPEG scratch pool", pool)
            .required(&[
                "fn prepare_for(",
                ") -> Result<(), JpegError>",
                "try_reserve_for_len",
                "fn release_retained_allocations",
                "fn reconcile_external_workspace",
                "fn release_for_external_workspace",
                "fn ensure_retained_capacity",
                "try_reserve_for_len_with_live_budget",
                "sequential_requires_growth",
                "lossless_requires_growth",
                "detached_sink_bytes",
            ])
            .forbidden(&[
                "self.prev_dc.resize(n, 0)",
                "rows.top_row.resize(width.saturating_mul(6), 0)",
            ]),
        PatternCheck::new("JPEG request live-allocation preflight", routing).required(&[
            "additional_decode_scratch_bytes(",
            "prepare_decode_workspace_with_additional(",
            "allocate_output_buffer_with_live_budget(",
            "checked_add_allocation_bytes(external_live_bytes, out.capacity())?",
            "decode_into_output_format_with_scratch_and_external(",
            "decode_region_into_output_format_with_scratch_and_external(",
            "external_live_bytes",
        ]),
        PatternCheck::new("JPEG non-pool workspace reconciliation", sequential).required(&[
            "let pool_backed = self.progressive_plan.is_none()",
            "super::SofKind::Lossless | super::SofKind::Extended12",
            "if pool_backed",
            "pool.reconcile_external_workspace(owned_output_bytes, workspace_cap)?;",
            "pool.release_for_external_workspace(requested, workspace_cap)?;",
        ]),
        PatternCheck::new("JPEG crop and progressive row scratch", writer_scratch)
            .required(&[
                "let row_bytes = checked_allocation_len::<u8>(scaled_width, 3)?;",
                "try_reserve_for_len_with_live_budget(",
                "fn prepare_rgb_rows(&mut self) -> Result<(), JpegError>",
                "let rgb_rows_bytes = checked_allocation_len::<u8>(rgb_row_len, 2)?;",
                "self.top_row = Vec::new();",
                "self.bottom_row = Vec::new();",
            ])
            .forbidden(&[
                "r: Vec::new()",
                "dst.resize(width, 0)",
                "top_row: vec![0; row_len]",
                "let r = try_vec_filled(scaled_width, 0)?;",
                "ensure_row_capacities(&[&r, &g, &b], row_bytes)?;",
                "try_resize_filled(&mut self.top_row, self.rgb_row_len, 0)?;",
            ]),
    ]);
}

#[test]
fn jpeg_decode_allocation_boundaries_and_reuse_remain_covered() {
    let sources = read_source_files(
        repo_root(),
        &[
            "crates/j2k-jpeg/src/allocation.rs",
            "crates/j2k-jpeg/src/decoder/allocation.rs",
            "crates/j2k-jpeg/src/decoder/core_traits.rs",
            "crates/j2k-jpeg/src/decoder/sequential.rs",
            "crates/j2k-jpeg/src/decoder/scratch.rs",
            "crates/j2k-jpeg/src/entropy/progressive.rs",
            "crates/j2k-jpeg/src/entropy/progressive/model.rs",
            "crates/j2k-jpeg/src/entropy/progressive/allocation.rs",
            "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
            "crates/j2k-jpeg/src/entropy/progressive/render.rs",
            "crates/j2k-jpeg/src/entropy/progressive/tests.rs",
            "crates/j2k-jpeg/src/internal/scratch.rs",
            "crates/j2k-jpeg/src/decoder/output_format.rs",
            "crates/j2k-jpeg/src/decoder/routing/owned_output.rs",
            "crates/j2k-jpeg/src/decoder/warning_ownership.rs",
            "crates/j2k-jpeg/src/decoder/tests.rs",
            "crates/j2k-jpeg/src/output_buffer.rs",
            "crates/j2k-jpeg/src/transcode.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("JPEG decode allocation regressions", &sources).required(&[
            "owned_output_and_scratch_share_an_exact_cap_boundary",
            "sequential_pool_formula_counts_outer_metadata_rows_and_payloads",
            "nested_lossless_rgba_intermediates_are_aggregated",
            "progressive12_scratch_counts_live_u16_render_planes",
            "sink_rows_have_an_exact_aggregate_cap_boundary",
            "same_or_smaller_sink_rows_reuse_capacity_within_cap",
            "stale_capacity_is_released_instead_of_rejecting_a_valid_request",
            "allocator_failure_keeps_its_public_typed_category",
            "resize_drops_stale_capacity_before_using_a_smaller_cap",
            "actual_allocator_capacity_is_checked_against_the_cap",
            "actual_retained_capacity_is_checked_and_released",
            "final_fit_stale_growth_still_counts_old_and_new_storage",
            "prior_actual_overcapacity_affects_the_next_transient_peak",
            "actual_vector_capacity_is_checked_against_the_selected_cap",
            "external_rows_reduce_the_remaining_progressive_phase_capacity",
            "retained_decoder_metadata_reduces_the_extraction_workspace",
            "owned_output_and_nested_lossless_rgba_temps_share_one_boundary",
            "non_pool_decode_releases_stale_capacity_even_when_it_would_fit",
            "owned_decode_external_live_boundary_counts_output_and_scratch_exactly",
        ]),
    ]);
}

#[test]
fn jpeg_shared_allocation_helpers_postcheck_actual_capacity() {
    let allocation = fs::read_to_string(repo_root().join("crates/j2k-jpeg/src/allocation.rs"))
        .expect("read JPEG allocation helpers");

    assert_pattern_checks(&[
        PatternCheck::new("JPEG shared Vec capacity postcheck", &allocation)
            .required(&[
                "pub(crate) fn try_vec_with_capacity<T>",
                "pub(crate) fn try_vec_filled<T: Clone>",
                "pub(crate) fn try_reserve_for_len<T>",
                "pub(crate) fn try_reserve_for_len_with_live_budget<T>",
                "fn try_reserve_for_len_with_budget<T>",
                "fn ensure_vec_capacity_bytes<T>",
                "ensure_vec_capacity_bytes(&values, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;",
            ])
            .normalized_required(&[
                "let actual_bytes = values.capacity().checked_mul(size_of::<T>()).ok_or(",
                "ensure_budget_bytes(actual_bytes, cap)",
            ]),
    ]);
}

#[test]
fn jpeg_sequential_warning_output_uses_one_fallible_owner() {
    let root = repo_root();
    let restart =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/restart.rs"))
            .expect("read sequential restart source");
    let fast_paths = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/rgb444.rs",
            "crates/j2k-jpeg/src/entropy/sequential/fast420/mod.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("fallible sequential scan warnings", &restart).required(&[
            "let mut warnings = try_vec_with_capacity(1)?;",
            "warnings.push(Warning::MissingEoi);",
        ]),
        PatternCheck::new("shared sequential scan finalization", &fast_paths)
            .required(&["finish_scan(&mut br, true)"])
            .forbidden(&["fn finish_fast_tile_scan", "let mut warnings = Vec::new();"]),
    ]);
}
