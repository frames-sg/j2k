use std::fs;

const SUPPORT_DOC: &str = "docs/public-support.md";
const CONFORMANCE_MANIFEST: &str = "corpus/j2k-conformance/manifest.tsv";

const REQUIRED_COLUMNS: &[&str] = &[
    "ID",
    "Status",
    "Existing subsystem to extend",
    "Reused helpers/APIs",
    "Self-check test",
    "Comparator/publication gate",
    "Remaining limitation",
];

const REQUIRED_SUPPORT_IDS: &[&str] = &[
    "part1-signed-samples",
    "part1-high-bit-depths",
    "part1-component-counts",
    "part1-progression-tileparts",
    "part1-packet-markers",
    "part1-roi-maxshift",
    "jp2-color-component-metadata",
    "part15-ht-refinement-decode",
    "part15-ht-refinement-encode",
    "jph-wrapper",
    "j2k-to-htj2k-recode",
    "benchmark-publication-gates",
    "external-speed-comparisons",
];

const REQUIRED_TEST_REFERENCES: &[&str] = &[
    "inspect_spec_valid_component_count_above_u8_is_reported_exactly",
    "native_parse_accepts_spec_component_count_above_u8",
    "poc_marker_preserves_wide_component_bounds",
    "poc_marker_accepts_wide_all_components_sentinel",
    "classic_encode_roundtrips_more_than_four_components",
    "htj2k_encode_roundtrips_more_than_four_components",
    "precomputed_53_encode_roundtrips_more_than_four_components",
    "precomputed_97_encode_roundtrips_more_than_four_components",
    "precomputed_encode_writes_component_sampling_in_siz",
    "precomputed_classic_53_encode_preserves_component_sampling_in_siz",
    "raw_pixel_encode_rejects_component_sampling_without_component_sized_dwt",
    "lossless_component_plane_encode_preserves_sampling_for_classic_and_htj2k",
    "lossless_component_plane_encode_round_trips_full_resolution_gray29_planes",
    "lossless_component_plane_encode_round_trips_full_resolution_gray35_planes",
    "lossless_component_plane_encode_round_trips_sampled_high_bit_planes",
    "lossless_component_plane_encode_round_trips_sampled_high_bit_multi_tile_planes",
    "lossless_component_plane_encode_round_trips_unaligned_sampled_high_bit_multi_tile_planes",
    "typed_component_plane_encode_preserves_mixed_metadata_for_classic_and_htj2k",
    "lossless_typed_component_plane_encode_preserves_mixed_metadata",
    "lossless_typed_component_plane_encode_round_trips_mixed_high_bit_metadata",
    "lossless_typed_component_plane_encode_round_trips_mixed_35_bit_metadata",
    "lossless_typed_component_plane_encode_round_trips_mixed_high_bit_multi_tile_metadata",
    "decode_components_exposes_public_sampling_metadata",
    "lossless_encode_facade_roundtrips_more_than_four_components",
    "cpu_lossless_round_trips_component_count_above_u8",
    "cpu_classic_lossy_roundtrips_more_than_four_components",
    "cpu_lossless_multi_tile_codestream_decodes",
    "cpu_classic_lossy_multi_tile_codestream_decodes",
    "cpu_lossless_emits_multiple_tile_parts_that_strict_decode_uses",
    "cpu_classic_lossy_emits_multiple_tile_parts_that_strict_decode_uses",
    "cpu_lossless_emits_tlm_for_multiple_tile_parts",
    "cpu_classic_lossy_emits_tlm_for_multiple_tile_parts",
    "tile_header_poc_changes_packet_iteration_order",
    "strict_decode_accepts_matching_plt_packet_length",
    "strict_decode_accepts_matching_plm_packet_length",
    "strict_decode_accepts_ppt_separated_packet_header",
    "strict_decode_accepts_ppm_separated_packet_header",
    "cpu_lossless_emits_packet_markers_that_strict_decode_uses",
    "cpu_lossless_emits_ppm_and_ppt_that_strict_decode_uses",
    "cpu_lossless_multi_tile_emits_ppm_and_ppt_that_strict_decode_uses",
    "cpu_lossless_emits_ppm_and_ppt_across_multiple_tile_parts_that_strict_decode_uses",
    "ppm_marker_writer_splits_at_packet_header_boundaries",
    "ppt_marker_writer_splits_large_payloads",
    "packet_header_validation_allows_chunked_ppm_and_ppt_payloads",
    "cpu_classic_lossy_emits_plt_and_plm_that_strict_decode_uses",
    "cpu_classic_lossy_emits_sop_and_eph_that_strict_decode_uses",
    "cpu_classic_lossy_emits_ppm_and_ppt_that_strict_decode_uses",
    "cpu_classic_lossy_multi_tile_emits_ppm_and_ppt_that_strict_decode_uses",
    "cpu_classic_lossy_emits_ppm_and_ppt_across_multiple_tile_parts_that_strict_decode_uses",
    "roi_maxshift_inverse_preserves_background_and_unshifts_roi_coefficients",
    "classic_scalar_decode_applies_nonzero_roi_maxshift",
    "tile_header_rgn_marker_with_zero_shift_is_a_noop",
    "encode_whole_component_roi_maxshift_roundtrips_and_writes_rgn",
    "encode_rectangular_roi_maxshift_roundtrips_and_writes_rgn",
    "encode_rejects_ambiguous_whole_component_and_rectangular_roi",
    "cpu_lossless_rectangular_roi_roundtrips_and_writes_rgn",
    "cpu_lossless_multi_tile_rectangular_roi_roundtrips_and_writes_rgn",
    "cpu_lossless_htj2k_rectangular_roi_roundtrips_at_31_coded_bitplanes",
    "cpu_lossless_classic_high_bit_rectangular_roi_roundtrips_at_50_coded_bitplanes",
    "cpu_lossless_classic_high_bit_rectangular_roi_rejects_56_coded_bitplanes_explicitly",
    "cpu_lossy_rectangular_roi_writes_rgn_and_decodes",
    "tile_header_rgn_with_explicit_style_is_rejected",
    "main_header_rgn_with_explicit_style_is_rejected",
    "iso_p0_03_tile_header_roi_maxshift_matches_reference_when_available",
    "inspect_ht_jph_reports_file_wrapper",
    "inspect_ht_jp2_rejects_file_type_mismatch",
    "inspect_accepts_legal_38_bit_component_metadata",
    "inspect_reports_legal_38_bit_component_metadata",
    "classic_coefficient_state_preserves_38_bit_magnitude",
    "classic_tier1_round_trips_38_bit_coefficients",
    "direct_ht_block_roundtrip_31_bit_cleanup_path",
    "classic_decode_adapter_accepts_legal_38_bit_roi_bitplane_count",
    "forward_53_i64_round_trips_38_bit_values",
    "forward_dwt_i64_matches_f32_path_for_exact_range",
    "classic_reversible_i64_encode_writes_29_bit_codestream_metadata",
    "classic_reversible_i64_decode_round_trips_29_bit_native_bytes",
    "classic_reversible_i64_decode_native_components_preserves_29_bit_plane",
    "classic_reversible_i64_decode_round_trips_signed_29_bit_native_bytes",
    "classic_reversible_i64_decode_round_trips_rgb29_rct_native_bytes",
    "classic_reversible_i64_decode_native_region_crops_29_bit_bytes",
    "classic_reversible_i64_decode_round_trips_31_bit_native_bytes_without_dwt",
    "htj2k_reversible_i64_decode_round_trips_29_bit_native_bytes_without_dwt",
    "htj2k_reversible_i64_decode_round_trips_31_bit_native_bytes_without_dwt",
    "classic_reversible_i64_encode_rejects_38_bit_beyond_no_quant_bitplane_limit",
    "cpu_lossless_classic_gray29_round_trips_native_bytes",
    "cpu_lossless_classic_gray32_dwt_round_trips_native_bytes",
    "cpu_lossless_classic_gray35_dwt_round_trips_native_bytes",
    "cpu_lossless_classic_gray37_without_dwt_round_trips_native_bytes",
    "cpu_lossless_classic_gray29_multi_tile_round_trips_native_bytes",
    "cpu_lossless_classic_signed_gray29_round_trips_native_bytes",
    "cpu_lossless_classic_rgb29_rct_round_trips_native_bytes",
    "cpu_lossless_classic_rejects_gray38_explicitly",
    "cpu_lossless_htj2k_gray31_without_dwt_round_trips_native_bytes",
    "cpu_lossless_htj2k_high_bit_dwt_rejects_explicitly",
    "cpu_lossless_htj2k_gray29_without_dwt_round_trips_native_bytes",
    "lossy_sample_descriptor_accepts_part1_high_bit_depths",
    "cpu_classic_lossy_gray29_decodes_native_bytes",
    "cpu_classic_lossy_gray38_decodes_native_bytes",
    "cpu_lossy_htj2k_high_bit_rejects_explicitly",
    "write_siz_marker_uses_per_component_sample_info",
    "write_qcc_marker_for_component_quantization_override",
    "written_qcc_marker_overrides_component_quantization_on_parse",
    "ht_capability_word_uses_max_component_precision",
    "decoder_exposes_component_planes",
    "decode_region_components_exposes_plane_dimensions",
    "native_bytes_per_sample_tracks_high_bit_depths",
    "native_sample_packing_writes_high_bit_unsigned_little_endian_bytes",
    "native_sample_packing_writes_high_bit_signed_little_endian_bytes",
    "unsigned_gray24_roundtrips_through_native_bytes",
    "signed_gray24_roundtrips_through_native_bytes",
    "cpu_lossless_round_trips_gray24",
    "cpu_lossless_round_trips_signed_gray24",
    "decode_native_rejects_mixed_component_bit_depths",
    "decode_native_components_handles_mixed_component_metadata",
    "decode_native_region_components_preserves_plane_metadata",
    "decode_native_components_exposes_mixed_public_planes",
    "decode_native_components_exposes_high_bit_public_plane",
    "decode_native_region_components_exposes_high_bit_public_plane",
    "decode_native_region_components_covers_sampled_high_bit_public_planes",
    "decode_native_accepts_gt24_bit_integer_packed_output",
    "decode_components_rejects_gt24_bit_float_planes",
    "signed_gray8_roundtrips_through_component_planes_and_native_bytes",
    "signed_gray16_roundtrips_through_native_bytes",
    "decode_components_exposes_signed_gray8_public_samples",
    "decode_components_exposes_signed_gray16_public_samples",
    "cpu_classic_lossy_preserves_signed_component_metadata",
    "wrap_ht_codestream_as_jph_inspects_and_decodes",
    "wrap_jph_rejects_classic_codestream",
    "inspect_rejects_jp2_file_type_with_ht_codestream",
    "jph_file_type_rejects_classic_codestream",
    "jp2_file_type_rejects_htj2k_codestream",
    "jph_file_type_accepts_htj2k_codestream",
    "wrap_writes_bpcc_for_mixed_precision_and_signedness",
    "wrap_allows_icc_color_spec_for_non_enumerated_component_counts",
    "wrap_preserves_inspected_icc_metadata_when_rewrapping",
    "wrap_preserves_inspected_icc_metadata_when_rewrapping_jph",
    "wrap_preserves_multiple_colr_boxes_when_rewrapping",
    "wrap_writes_cdef_for_explicit_srgb_alpha",
    "wrap_preserves_premultiplied_alpha_cdef_and_decodes_rgba",
    "inspect_and_decode_jp2_palette_component_mapping_metadata",
    "inspect_and_decode_jp2_signed_palette_metadata",
    "wrap_writes_palette_component_mapping_and_channel_definitions",
    "missing_ihdr_returns_invalid_box",
    "missing_colr_returns_invalid_box",
    "invalid_ihdr_compression_type_returns_invalid_box",
    "ihdr_dimension_mismatch_returns_invalid_box",
    "ihdr_bpc_mismatch_returns_invalid_box",
    "bpcc_precision_mismatch_returns_invalid_box",
    "premultiplied_opacity_cdef_sets_alpha",
    "unspecified_cdef_association_decodes",
    "decode_rgba16_roi_scaled_and_region_scaled_preserve_alpha",
    "inspect_rejects_missing_colr",
    "inspect_rejects_invalid_ihdr_compression_type",
    "inspect_rejects_ihdr_dimension_mismatch",
    "inspect_rejects_ihdr_bpc_mismatch",
    "inspect_rejects_bpcc_precision_mismatch",
    "lossy_97_source_uses_pixel_preserving_recode",
    "signed_source_uses_pixel_preserving_recode",
    "four_component_source_uses_pixel_preserving_recode",
    "mixed_typed_source_uses_pixel_preserving_recode",
    "high_bit_source_uses_pixel_preserving_recode",
    "lossy_sampled_source_uses_pixel_fallback_and_preserves_sampling",
    "raw_htj2k_lossless_can_be_wrapped_as_jph_without_reencode",
    "recode_can_emit_jph_file_wrapper",
    "recode_jph_preserves_input_icc_color_spec",
    "recode_jph_preserves_multiple_colr_boxes_for_coefficient_path",
    "recode_jph_preserves_channel_definition_metadata_for_coefficient_path",
    "recode_jph_drops_palette_metadata_on_pixel_fallback",
    "recode_jph_drops_component_mapping_metadata_on_sampled_pixel_fallback",
    "recode_subsampled_classic_53_uses_coefficient_path_and_preserves_sampling",
    "cpu_htj2k_lossy_reports_rate_granularity",
    "cpu_htj2k_lossy_three_quality_layers_use_three_pass_segment_granularity",
    "ht_target_coding_passes_tracks_ht_quality_layers",
    "preencoded_htj2k97_preserves_refinement_segments_in_packet_body",
    "prequantized_htj2k97_accepts_empty_high_subbands",
    "ht_cpu_fallback_encodes_two_pass_sigprop_refinement",
    "ht_cpu_fallback_sigprop_refinement_encodes_new_significance_bits",
    "ht_cpu_fallback_encodes_three_pass_magref_refinement",
    "ht_cpu_fallback_rejects_unsupported_refinement_pass_count",
    "accelerator_facade_ht_lossless_quality_layers_request_refinement_passes",
    "ht_layer_contributions_split_cleanup_and_refinement_across_layers",
    "htj2k_lossy_quality_layers_decode_split_refinement_layer",
    "public_decode_matches_openhtj2k_refinement_fixtures",
    "cargo xtask adoption-report",
];

const REQUIRED_COMPARATOR_REFERENCES: &[&str] = &[
    "OpenJPEG",
    "Grok",
    "OpenJPH",
    "Kakadu",
    "publication_eligible",
];

const REQUIRED_MANIFEST_IDS: &[&str] = &[
    "part1_core_lossless_53",
    "part1_core_lossy_97_layers_precincts",
    "part1_poc_tlm_sop",
    "part1_plt_sop_eph",
    "openhtj2k_ds0_ht_12_b11",
    "openhtj2k_ds0_ht_09_b11",
    "plm_iso_vector_absent",
    "jpx_part2_deferred",
    "icc_roundtrip_deferred",
    "encode_gt16_deferred",
];

/// Verify the public J2K/HTJ2K support matrix stays synchronized with tests,
/// conformance rows, and publication gates.
pub(crate) fn public_support(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let mut require_final_statuses = false;
    for arg in args {
        match arg.as_str() {
            "--final" => require_final_statuses = true,
            "-h" | "--help" => return Err(public_support_usage()),
            other => {
                return Err(format!(
                    "unknown public-support argument `{other}`\n{}",
                    public_support_usage()
                ));
            }
        }
    }

    let doc = read(SUPPORT_DOC)?;
    let manifest = read(CONFORMANCE_MANIFEST)?;
    let mut failures = Vec::new();

    require_contains_all(&doc, REQUIRED_COLUMNS, SUPPORT_DOC, &mut failures);
    require_contains_all(&doc, REQUIRED_SUPPORT_IDS, SUPPORT_DOC, &mut failures);
    require_contains_all(&doc, REQUIRED_TEST_REFERENCES, SUPPORT_DOC, &mut failures);
    require_contains_all(
        &doc,
        REQUIRED_COMPARATOR_REFERENCES,
        SUPPORT_DOC,
        &mut failures,
    );
    require_contains_all(
        &manifest,
        REQUIRED_MANIFEST_IDS,
        CONFORMANCE_MANIFEST,
        &mut failures,
    );

    for id in manifest_ids(&manifest) {
        if !doc.contains(id) && !id.starts_with('#') {
            failures.push(format!(
                "{SUPPORT_DOC} does not mention conformance manifest row `{id}`"
            ));
        }
    }

    if !doc.contains("jpx_part2_deferred")
        || !doc.contains("Out of scope")
        || !doc.contains("Part 2")
    {
        failures.push(
            "JPX/Part 2 deferral must be documented as explicit out-of-scope policy".to_string(),
        );
    }

    if require_final_statuses {
        require_final_support_statuses(&doc, &mut failures);
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

fn public_support_usage() -> String {
    "usage: cargo xtask public-support [--final]".to_string()
}

fn read(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|err| format!("read {path}: {err}"))
}

fn require_contains_all(haystack: &str, needles: &[&str], path: &str, failures: &mut Vec<String>) {
    for needle in needles {
        if !haystack.contains(needle) {
            failures.push(format!("{path} is missing `{needle}`"));
        }
    }
}

fn manifest_ids(manifest: &str) -> impl Iterator<Item = &str> {
    manifest.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        line.split('\t').next()
    })
}

fn require_final_support_statuses(doc: &str, failures: &mut Vec<String>) {
    for required_id in REQUIRED_SUPPORT_IDS {
        match support_row_status(doc, required_id) {
            Some("Done") => {}
            Some(status) => failures.push(format!(
                "{SUPPORT_DOC} row `{required_id}` must be `Done` for final support, found `{status}`"
            )),
            None => failures.push(format!(
                "{SUPPORT_DOC} is missing support matrix row `{required_id}`"
            )),
        }
    }
}

fn support_row_status<'a>(doc: &'a str, id: &str) -> Option<&'a str> {
    doc.lines().find_map(|line| {
        let line = line.trim();
        if !line.starts_with('|') {
            return None;
        }
        let mut cells = line.split('|').map(str::trim);
        cells.next()?;
        let row_id = cells.next()?;
        let status = cells.next()?;
        (row_id == id).then_some(status)
    })
}

#[cfg(test)]
mod tests {
    use super::{manifest_ids, support_row_status};

    #[test]
    fn manifest_ids_skips_header_comments() {
        let ids = manifest_ids("# id\tpath\nrow_a\ta\n\nrow_b\tb").collect::<Vec<_>>();

        assert_eq!(ids, ["row_a", "row_b"]);
    }

    #[test]
    fn support_row_status_reads_markdown_table_rows() {
        let doc = "\
| ID | Status | Existing subsystem to extend |
| --- | --- | --- |
| part1-signed-samples | Done | x |
";

        assert_eq!(
            support_row_status(doc, "part1-signed-samples"),
            Some("Done")
        );
        assert_eq!(support_row_status(doc, "missing"), None);
    }
}
