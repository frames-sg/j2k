// SPDX-License-Identifier: MIT OR Apache-2.0

const DCT97_BENCH: &str = include_str!("../benches/dct97.rs");

#[test]
fn dct97_benchmark_groups_are_stable() {
    for expected in [
        "dct53_metal_projection",
        "jpeg_to_htj2k_wsi_53",
        "reversible_dct53_metal_projection",
        "reversible_dct53_batch_metal_projection",
        "jpeg_to_htj2k_wsi_integer_53",
        "jpeg_to_htj2k_wsi_integer_53_tile_batch",
        "metal_auto",
        "rayon_batch",
        "metal_auto_batch",
        "metal_explicit_batch",
        "rayon_224x224",
        "rayon_512x512",
        "rayon_1024x1024",
        "rayon_2048x2048",
        "batch_1",
        "batch_8",
        "batch_32",
        "batch_128",
        "batch_256",
        "batch_512",
        "rayon_224x224_batch_1",
        "metal_explicit_224x224_batch_1",
        "rayon_512x512_batch_512",
        "rayon_1024x1024_batch_128",
        "rayon_2048x2048_batch_32",
        "dct97_metal_idct_dwt",
        "cpu_idct_dwt_224x224",
        "metal_explicit_224x224",
        "cpu_idct_dwt_512x512",
        "metal_explicit_512x512",
        "cpu_idct_dwt_1024x1024",
        "metal_explicit_1024x1024",
        "cpu_idct_dwt_2048x2048",
        "metal_explicit_2048x2048",
        "jpeg_to_htj2k_wsi_97",
        "srgb_ybr420_224",
        "srgb_ybr420_512",
        "srgb_ybr420_1024",
        "srgb_ybr420_2048",
        "srgb_ybr420_224_batch_128",
        "srgb_ybr420_512_batch_128",
        "srgb_ybr420_1024_batch_128",
        "srgb_ybr420_2048_batch_128",
        "srgb_ybr420_2048_batch_256",
        "srgb_ybr420_2048_batch_512",
        "p3_like_ybr444_224",
        "p3_like_ybr444_512",
        "p3_like_ybr444_1024",
        "p3_like_ybr444_2048",
        "p3_like_ybr444_224_batch_128",
        "p3_like_ybr444_512_batch_256",
        "p3_like_ybr444_1024_batch_512",
        "p3_like_ybr444_2048_batch_512",
        "ycbcr_like_ybr420_224",
        "ycbcr_like_ybr420_512",
        "ycbcr_like_ybr420_1024",
        "ycbcr_like_ybr420_2048",
        "ycbcr_like_ybr420_224_batch_512",
        "ycbcr_like_ybr420_512_batch_512",
        "ycbcr_like_ybr420_1024_batch_512",
        "ycbcr_like_ybr420_2048_batch_512",
    ] {
        assert!(
            DCT97_BENCH.contains(expected),
            "missing benchmark marker {expected}"
        );
    }
}

#[test]
fn dct97_benchmark_emits_transcode_batch_profile_rows() {
    for expected in [
        "J2K_TRANSCODE_METAL_PROFILE_STAGES",
        "emit_transcode_batch_profile",
        "profile_stage_mode_from_env",
        "ProfileStageMode::Disabled",
        "ProfileStageMode::Rows",
        "ProfileStageMode::Summary",
        "TRANSCODE_BATCH_PROFILE_SUMMARY",
        "Option<j2k_profile::ProfileSummary>",
        "ProfileSummary::new",
        "format_transcode_profile_fields",
        "format_profile_key_value_fields_with_limits",
        "with_max_output_bytes",
        "checked_sub(PROFILE_PREFIX.len())",
        "eprintln!(\"j2k_profile{fields}\")",
        "use j2k_profile::emit_profile_error",
        "transcode_batch_profile_row",
        "transcode_batch_profile_format",
        "transcode_batch_summary_record",
        "record_str",
        "row.codec()",
        "row.op()",
        "row.path()",
        "report.profile_row(context, request)",
        "format_transcode_profile_fields(row.fields())",
        "TranscodeBatchProfileRequest::Cpu",
        "TranscodeBatchProfileRequest::MetalAuto",
        "TranscodeBatchProfileRequest::MetalExplicit",
        "benchmark_name.as_str()",
        "pipeline_map",
        "emit_pipeline_map",
        "stage.resident_handoff_count",
        "recommend_next_stage",
        "map.recommendation.evidence_dispatches",
    ] {
        assert!(
            DCT97_BENCH.contains(expected),
            "missing transcode batch profile marker {expected}"
        );
    }
    assert!(
        !DCT97_BENCH.contains("format!(\"j2k_profile"),
        "profile emission must not allocate a second unbounded row string"
    );
}
