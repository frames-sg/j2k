// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ensure settled experiments remove temporary controls and retain evidence surfaces.

use super::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn rejected_cuda_dwt97_column_quantize_has_no_candidate_or_active_switch() {
    const RETIRED_SWITCH: &str = "J2K_CUDA_DISABLE_DWT97_FUSED_COLUMN_QUANTIZE";
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-transcode-cuda/Cargo.toml").required(&[
                "j2k-cuda-j2k-engine/cuda-oxide-j2k-encode",
                "j2k-cuda-j2k-engine/cuda-oxide-htj2k-encode",
            ]),
            FilePatternCheck::new("crates/j2k-cuda-transcode-engine/src/build_flags.rs")
                .forbidden(&[RETIRED_SWITCH, "dwt97_fused_column_quantize_disabled"]),
            FilePatternCheck::new("crates/j2k-cuda-transcode-engine/src/transcode/htj2k97.rs")
                .required(&[
                    "input: Dwt97BatchInput::I16(blocks)",
                    "launch_transcode_dwt97_quantize_codeblock_bands",
                ]),
            FilePatternCheck::new("crates/j2k-transcode-cuda/benches/dwt97.rs")
                .required(&[
                    "staged_column_lift_quantize",
                    "DctGridI16ToHtj2k97CodeBlockJob",
                    "dct_grid_i16_to_htj2k97_preencoded_batch",
                    "new_explicit_resident_ht_encode",
                    "assert_staged_route",
                    "transcode_batch_with_accelerator",
                    "temporary_float_band_bytes",
                    "product_temporary_float_band_bytes",
                    "product_temporary_float_band_traffic_bytes",
                    "resident_stage_input_sha256",
                    "output_sha256",
                    "for tile in &probe.tiles",
                    "P13 product tile codestream decodes",
                    "const BATCH_SIZE: usize = 16",
                    "const DIMENSION: usize = 512",
                    "resident_preencode_512x512_batch_16",
                    "srgb_ybr420_512_batch_16",
                ])
                .forbidden(&[
                    RETIRED_SWITCH,
                    "unfused_column_lift_quantize",
                    "fused_column_lift_quantize",
                    "DctGridToHtj2k97CodeBlockJob",
                    "dct_grid_to_htj2k97_codeblock_batch",
                ]),
            FilePatternCheck::new("docs/env-vars.md").required(&[RETIRED_SWITCH, "Historical P13"]),
        ],
    );
}

#[test]
fn promoted_jpeg_staged_encode_has_no_benchmark_switch_or_legacy_route() {
    const RETIRED_SWITCH: &str = "J2K_JPEG_METAL_DISABLE_STAGED_BASELINE_ENCODE";

    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/compute/encode.rs")
                .required(&[
                    "jpeg_baseline_encode_precompute_batch",
                    "jpeg_baseline_encode_entropy_from_coeffs_batch",
                ])
                .forbidden(&[RETIRED_SWITCH, "jpeg_baseline_encode_batch"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/benches/encode_baseline.rs")
                .required(&["path=staged", "coefficient_scratch_bytes", "output_sha256"])
                .forbidden(&[RETIRED_SWITCH]),
            FilePatternCheck::new("docs/env-vars.md").required(&[RETIRED_SWITCH, "Historical P18"]),
        ],
    );
}
