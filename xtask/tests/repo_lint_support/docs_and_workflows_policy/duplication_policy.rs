// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{
    assert_contains_all_normalized, assert_file_pattern_checks, assert_pattern_checks,
    read_source_files, repo_root, FilePatternCheck, PatternCheck,
};

mod cache_identity;
mod classic_mq;

#[test]
fn component_plane_metadata_accessors_are_shared() {
    let root = repo_root();
    let native_color = fs::read_to_string(root.join("crates/j2k-native/src/color.rs"))
        .expect("read native color module");
    let facade_decode = fs::read_to_string(root.join("crates/j2k/src/decode/component_handoff.rs"))
        .expect("read j2k component handoff");

    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-native/src/lib.rs")
                .named("native component-plane accessor macro")
                .required(&[
                    "#[doc(hidden)]",
                    "#[macro_export]",
                    "macro_rules! __j2k_component_plane_metadata_accessors",
                ]),
            FilePatternCheck::new("crates/j2k/src/decode/component_handoff.rs")
                .named("j2k component handoff")
                .forbidden(&["macro_rules! impl_component_plane_metadata_accessors"]),
        ],
    );
    for (name, source, expected_macro, expected_calls) in [
        (
            "native color",
            native_color.as_str(),
            "crate::__j2k_component_plane_metadata_accessors!();",
            2,
        ),
        (
            "j2k decode facade",
            facade_decode.as_str(),
            "j2k_native::__j2k_component_plane_metadata_accessors!();",
            2,
        ),
    ] {
        assert_eq!(
            source.matches(expected_macro).count(),
            expected_calls,
            "{name} must use the shared component-plane accessor macro"
        );
    }
}

#[test]
fn ht_code_block_scalar_fallback_lives_in_trait_default() {
    let root = repo_root();
    let backend = fs::read_to_string(root.join("crates/j2k-native/src/backend.rs"))
        .expect("read native backend trait");
    let trait_source = backend
        .split_once("pub trait HtCodeBlockDecoder")
        .expect("native backend trait must define HtCodeBlockDecoder")
        .1;
    assert_contains_all_normalized(
        "HT code-block scalar fallback",
        trait_source,
        &[
            "fn decode_code_block(\n        &mut self,",
            "decode_ht_code_block_scalar(job, output)",
        ],
    );

    for relative in [
        "crates/j2k-metal/src/classic.rs",
        "crates/j2k-metal/src/idwt.rs",
        "crates/j2k-metal/src/mct.rs",
        "crates/j2k-metal/src/store.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let production = source
            .split_once("#[cfg(test)]")
            .map_or(source.as_str(), |(prod, _)| prod);
        assert!(
            !production.contains("fn decode_code_block("),
            "{relative} must inherit the shared scalar HT fallback instead of restating it"
        );
    }

    let composite =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/code_block_decoder.rs"))
            .expect("read Metal composite code-block decoder");
    assert!(
        composite.contains("self.ht.decode_code_block(job, output)")
            && !composite.contains("decode_ht_code_block_scalar("),
        "Metal composite decoder must delegate HT blocks instead of copying the scalar fallback"
    );
}

#[test]
fn packet_progression_ordering_uses_shared_packetization_contract() {
    let root = repo_root();
    let packet_contract =
        fs::read_to_string(root.join("crates/j2k-types/src/lib.rs")).expect("read j2k-types");
    let native_encode =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/packet_plan.rs"))
            .expect("read native encode packet plan");
    let native_encode_options =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/options.rs"))
            .expect("read native encode options");
    let native_codestream =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/codestream_write.rs"))
            .expect("read native codestream writer");
    let metal_packet_plan =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode/packet_plan.rs"))
            .expect("read Metal packet plan");
    let metal_capacity =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/encode_capacity.rs"))
            .expect("read Metal encode capacity");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-types packet progression contract", &packet_contract).required(&[
            "pub fn sort_packet_descriptors_for_progression",
            "pub const fn codestream_order_code",
        ]),
        PatternCheck::new("native encode packetization option", &native_encode_options)
            .required(&["packetization_order(self)"]),
        PatternCheck::new("native encode packet descriptor ordering", &native_encode)
            .required(&["crate::sort_packet_descriptors_for_progression("])
            .forbidden(&["fn sort_packet_descriptors_for_progression("]),
        PatternCheck::new(
            "native codestream progression byte mapping",
            &native_codestream,
        )
        .required(&[".codestream_order_code()"]),
        PatternCheck::new("Metal packet plan progression ordering", &metal_packet_plan)
            .required(&["sort_packet_descriptors_for_progression("])
            .forbidden(&["fn sort_lossless_device_packet_descriptors("]),
        PatternCheck::new("Metal capacity progression byte mapping", &metal_capacity)
            .required(&[".codestream_order_code()"]),
    ]);
}

#[test]
fn idwt_required_region_propagation_uses_shared_native_helper() {
    let root = repo_root();
    let direct_roi = fs::read_to_string(root.join("crates/j2k-native/src/direct_roi.rs"))
        .expect("read native direct ROI helper");
    let direct_roi_tests =
        fs::read_to_string(root.join("crates/j2k-native/src/direct_roi/region_tests.rs"))
            .expect("read native direct ROI helper tests");
    let native_roi = fs::read_to_string(root.join("crates/j2k-native/src/j2c/roi.rs"))
        .expect("read native ROI planner");
    let cuda_direct =
        fs::read_to_string(root.join("crates/j2k-cuda/src/direct_plan/required_regions.rs"))
            .expect("read CUDA direct-plan required regions");
    let metal_direct_roi =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
            .expect("read Metal direct ROI");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native direct ROI IDWT helper", &direct_roi).required(&[
            "pub fn idwt_required_input_windows",
            "pub fn idwt_required_input_window_for_rects",
            "pub const fn idwt_required_output_margin",
            "pub struct J2kRequiredBandRegion",
            "mod region_tests;",
        ]),
        PatternCheck::new("j2k-native direct ROI region contracts", &direct_roi_tests).required(&[
            "fn expanded_within_rect_clamps_each_edge_in_absolute_coordinates()",
            "fn expanded_within_rect_saturates_overflow_before_clamping()",
            "fn union_returns_the_commutative_bounding_envelope()",
            "fn to_rect_preserves_non_empty_empty_and_maximum_coordinates()",
            "fn from_required_region_matches_the_explicit_rect_conversion()",
        ]),
    ]);
    assert!(
        direct_roi_tests.lines().count() < 100,
        "native direct ROI region contracts must stay below their focused 100-line ratchet"
    );

    for (relative, source) in [
        (
            "crates/j2k-cuda/src/direct_plan/required_regions.rs",
            &cuda_direct,
        ),
        (
            "crates/j2k-metal/src/compute/direct_roi.rs",
            &metal_direct_roi,
        ),
    ] {
        assert_pattern_checks(&[PatternCheck::new(relative, source)
            .required(&["idwt_required_input_windows(", "expanded_within_band("])
            .forbidden(&[
                "fn idwt_input_required_region(",
                "fn idwt_required_output_margin(",
                "struct RequiredBandRegion {",
                "struct BandRequiredRegion {",
                "j2k_native::idwt_band_index",
            ])]);
    }

    assert_pattern_checks(&[
        PatternCheck::new("native ROI IDWT window arithmetic", &native_roi)
            .required(&[
                "idwt_required_input_window_for_rects(",
                "crate::idwt_required_output_margin(",
            ])
            .forbidden(&["fn idwt_input_required_region(", "fn idwt_band_index("]),
    ]);
}

#[test]
fn metal_direct_required_region_retain_uses_shared_job_helper() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute ROI module", &compute).required(&["mod direct_roi;"]),
        PatternCheck::new("Metal direct required-region retain helper", &direct_roi).required(&[
            "trait RequiredRegionJob",
            "impl RequiredRegionJob for J2kClassicCleanupBatchJob",
            "impl RequiredRegionJob for J2kHtCleanupBatchJob",
            "fn retain_jobs_for_required_region<J: RequiredRegionJob>",
            "retain_jobs_for_required_region(jobs, required);",
        ]),
    ]);
    assert_eq!(
        direct_roi.matches("jobs.retain(|job|").count(),
        1,
        "Metal direct classic/HT required-region retain must have one shared retain body"
    );
}

#[test]
fn metal_direct_sub_band_group_scan_uses_shared_helper() {
    let root = repo_root();
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_prepare =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare.rs"))
            .expect("read Metal direct prepare");

    assert_pattern_checks(&[
        PatternCheck::new("Metal compute prepare module", &compute)
            .required(&["mod direct_prepare;"]),
        PatternCheck::new("Metal direct sub-band grouping helper", &direct_prepare).required(&[
            "fn prepare_sub_band_groups<'a, SubBand: 'a, Group>",
            "prepare_sub_band_groups(",
            "PreparedDirectGrayscaleStep::ClassicSubBand(sub_band)",
            "PreparedDirectGrayscaleStep::HtSubBand(sub_band)",
            "prepare_classic_sub_band_group,",
            "prepare_ht_sub_band_group,",
        ]),
    ]);
    assert_eq!(
        direct_prepare
            .matches("while step_idx < steps.len()")
            .count(),
        1,
        "Metal direct classic/HT sub-band grouping must have one shared scan loop"
    );
}

#[test]
fn metal_hybrid_region_scaled_cache_uses_shared_scope() {
    let root = repo_root();
    let hybrid =
        fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs")).expect("read hybrid");

    assert_pattern_checks(&[
        PatternCheck::new("Metal hybrid region-scaled cache scope", &hybrid)
            .required(&[
                "enum RegionScaledColorPlanCache",
                "Uncached",
                "Global",
                "Session(&'a crate::MetalBackendSession)",
                "fn build_region_scaled_direct_plan_with_cache(",
                "fn build_region_scaled_direct_color_plan_cached_with_cache(",
                "RegionScaledColorPlanCache::Uncached",
                "RegionScaledColorPlanCache::Session(session)",
            ])
            .forbidden(&["fn build_region_scaled_direct_color_plan_cached_with_session("]),
    ]);
    assert_eq!(
        hybrid.matches("match fmt {").count(),
        1,
        "Metal hybrid direct region-scaled format dispatch must stay single-sourced"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "wavelet constant provenance is checked across every backend in one policy matrix"
)]
fn wavelet_and_idct_constants_use_codec_math_sources() {
    let root = repo_root();
    let codec_math = fs::read_to_string(root.join("crates/j2k-codec-math/src/lib.rs"))
        .expect("read j2k-codec-math lib");
    let metal_shader_source =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/shader_source.rs"))
            .expect("read j2k-metal shader source");
    let metal_forward_transform =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/forward_transform.rs"))
            .expect("read j2k-metal forward transform");
    let forward_dwt_shader = fs::read_to_string(root.join("crates/j2k-metal/src/fdwt.metal"))
        .expect("read j2k-metal fdwt shader");
    let inverse_dwt_shader = fs::read_to_string(root.join("crates/j2k-metal/src/idwt.metal"))
        .expect("read j2k-metal idwt shader");
    let transcode_metal =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/metal/runtime.rs"))
            .expect("read transcode Metal runtime");
    let metal_transcode_dct97 =
        fs::read_to_string(root.join("crates/j2k-transcode-metal/src/dct97.metal"))
            .expect("read transcode Metal dct97 shader");
    let cpu_transcode_dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode CPU dct97 module");
    let cuda_transcode = read_source_files(
        root,
        &[
            "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/main.rs",
            "crates/j2k-cuda-runtime/src/cuda_oxide_transcode/simt/src/constants.rs",
        ],
    );

    assert_file_pattern_checks(
        root,
        &[FilePatternCheck::new("crates/j2k-metal/Cargo.toml")
            .named("j2k-metal codec math dependency")
            .required(&["j2k-codec-math"])],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "j2k-codec-math generated Metal DWT97 constants",
        &codec_math,
    )
    .required(&[
        "pub const DWT97_CONSTANTS_METAL",
        "include_str!(\"../generated/dwt97_constants.metal\")",
    ])]);
    let generated_idx = metal_shader_source
        .find("j2k_codec_math::generated::DWT97_CONSTANTS_METAL")
        .expect("j2k-metal shader source must splice generated DWT97 constants");
    let idwt_idx = metal_shader_source
        .find("../idwt.metal")
        .expect("j2k-metal shader source must include IDWT shader");
    let fdwt_idx = metal_shader_source
        .find("../fdwt.metal")
        .expect("j2k-metal shader source must include FDWT shader");
    assert!(
        generated_idx < idwt_idx && generated_idx < fdwt_idx,
        "j2k-metal shader source must splice generated DWT97 constants before IDWT/FDWT shaders"
    );
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2k-transcode-metal generated DWT97 constants",
            &transcode_metal,
        )
        .required(&["j2k_codec_math::generated::DWT97_CONSTANTS_METAL"]),
        PatternCheck::new("j2k-transcode CPU DCT97 constants", &cpu_transcode_dct97)
            .required(&[
                "j2k_codec_math::dwt::DWT97_ALPHA_F64",
                "j2k_codec_math::dwt::DWT97_BETA_F64",
                "j2k_codec_math::dwt::DWT97_GAMMA_F64",
                "j2k_codec_math::dwt::DWT97_DELTA_F64",
                "j2k_codec_math::dwt::DWT97_KAPPA_F64",
                "j2k_codec_math::dwt::DWT97_INV_KAPPA_F64",
            ])
            .forbidden(&[
                "-1.586_134_342",
                "-0.052_980_118",
                "0.882_911_075",
                "0.443_506_852",
                "1.230_174_104",
            ]),
        PatternCheck::new("j2k-metal host DWT97 constants", &metal_forward_transform).required(&[
            "j2k_codec_math::dwt::DWT97_ALPHA_F32",
            "j2k_codec_math::dwt::DWT97_BETA_F32",
            "j2k_codec_math::dwt::DWT97_GAMMA_F32",
            "j2k_codec_math::dwt::DWT97_DELTA_F32",
        ]),
    ]);

    for (relative, source) in [
        ("crates/j2k-metal/src/fdwt.metal", &forward_dwt_shader),
        ("crates/j2k-metal/src/idwt.metal", &inverse_dwt_shader),
        (
            "crates/j2k-transcode-metal/src/dct97.metal",
            &metal_transcode_dct97,
        ),
    ] {
        assert!(
            source.contains("CODEC_MATH_DWT97") || source.contains("CODEC_MATH_IDWT97"),
            "{relative} must use generated codec-math DWT constants"
        );
        assert_pattern_checks(&[PatternCheck::new(relative, source).forbidden(&[
            "1.586134",
            "0.052980",
            "0.882911",
            "0.443506",
            "1.230174",
            "J2K_FDWT97_",
            "DCT97_ALPHA",
            "DCT97_BETA",
            "DCT97_GAMMA",
            "DCT97_DELTA",
            "DCT97_KAPPA",
        ])]);
    }

    assert_pattern_checks(&[PatternCheck::new(
        "CUDA Oxide transcode SIMT IDCT constants",
        &cuda_transcode,
    )
    .required(&["use j2k_codec_math::jpeg::idct;", "idct::FIX_0_298631336"])
    .forbidden(&[
        "const CONST_BITS: i32 = 13",
        "const FIX_0_298631336: i32 = 2446",
    ])]);
}

#[test]
fn jp2_box_parsing_is_native_owned_with_facade_adapter_only() {
    let root = repo_root();
    let native_jp2 = read_source_files(
        root,
        &[
            "crates/j2k-native/src/jp2/mod.rs",
            "crates/j2k-native/src/jp2/container.rs",
        ],
    );
    let native_lib =
        fs::read_to_string(root.join("crates/j2k-native/src/lib.rs")).expect("read native lib");
    let facade_boxes = fs::read_to_string(root.join("crates/j2k/src/parse/boxes.rs"))
        .expect("read facade JP2 adapter");

    assert!(
        native_jp2.contains("pub fn inspect_jp2_container")
            && native_jp2.contains("fn parse_jp2_container_with_strict")
            && native_jp2.contains("parse_jp2_container_with_strict_and_retained_baseline("),
        "j2k-native must own the JP2/JPH container box walk used by native decode"
    );
    assert!(
        native_lib.contains("inspect_jp2_container"),
        "j2k-native must re-export the JP2 container inspection bridge for facade adapters"
    );
    assert!(
        facade_boxes.contains("inspect_jp2_container(input)")
            && !facade_boxes.contains("fn read_box_header")
            && !facade_boxes.contains("fn parse_jp2h")
            && !facade_boxes.contains("fn parse_pclr")
            && !facade_boxes.contains("fn parse_cmap")
            && !facade_boxes.contains("fn parse_cdef")
            && !facade_boxes.contains("fn walk_top_level_boxes"),
        "j2k facade JP2 parsing must be an adapter over j2k-native, not a second box parser"
    );
}

#[test]
fn native_classic_and_ht_parallel_copyback_share_one_helper() {
    let root = repo_root();
    let decode_shell = fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode.rs"))
        .expect("read native J2K decode module");
    let decode_subband =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode/subband.rs"))
            .expect("read native J2K subband decode module");
    let decode_parallel =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/decode/subband/parallel.rs"))
            .expect("read native J2K parallel subband decode module");
    let decode = format!("{decode_shell}\n{decode_subband}\n{decode_parallel}");

    assert_pattern_checks(&[PatternCheck::new(
        "native classic/HT decoded-block copyback",
        decode.as_str(),
    )
    .required(&[
        "trait DecodedSubBandBlock",
        "impl DecodedSubBandBlock for DecodedClassicBlock",
        "impl DecodedSubBandBlock for DecodedHtBlock",
        "fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>",
        "copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)",
        "decoded_classic_block_copyback_covers_full_block",
        "decoded_ht_block_copyback_covers_partial_edge_block",
        "decoded_block_copyback_rejects_out_of_bounds_blocks",
    ])]);
    let helper_start = decode
        .find("fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>")
        .expect("shared decoded-block copyback helper");
    let helper_rest = &decode[helper_start..];
    let helper_end = helper_rest
        .find("#[cfg(test)]")
        .expect("end of shared decoded-block copyback helper");
    let helper = &helper_rest[..helper_end];
    assert_eq!(
        helper.matches("let dst_start =").count(),
        1,
        "native decoded-block copyback destination bounds/indexing must have one implementation"
    );
    assert_eq!(
        decode
            .matches(".copy_from_slice(&block.coefficients()")
            .count(),
        1,
        "native decoded-block coefficient row copy must have one implementation"
    );
}

#[test]
fn copied_test_fixture_helpers_live_in_shared_support() {
    let root = repo_root();
    let test_support = fs::read_to_string(root.join("crates/j2k-test-support/src/lib.rs"))
        .expect("read j2k-test-support");
    let compare = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare/images.rs"))
        .expect("read compare encode image module");
    let dct97 = fs::read_to_string(root.join("crates/j2k-transcode/src/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT module");
    let dct97_test = fs::read_to_string(root.join("crates/j2k-transcode/tests/dct97_2d.rs"))
        .expect("read transcode 9/7 DCT test");
    let dwt_diff =
        fs::read_to_string(root.join("crates/j2k-transcode-test-support/src/dwt_diff.rs"))
            .expect("read shared transcode DWT diff test support");
    let jpeg_batch = fs::read_to_string(root.join("crates/j2k-jpeg/tests/batch.rs"))
        .expect("read JPEG batch tests");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-test-support shared PNM helper", &test_support).required(&[
            "pub fn write_pnm(",
            "pub fn read_pnm_image(",
            "pub struct PnmImage",
            "fn read_pnm_token",
        ]),
        PatternCheck::new("compare encode shared PNM helper use", &compare)
            .required(&[
                "j2k_test_support::write_pnm",
                "j2k_test_support::read_pnm_image",
            ])
            .forbidden(&[
                "struct PnmParser",
                "fn parse_pnm_u32(",
                "fs::File::create(path)",
                "write!(file,",
            ]),
    ]);

    assert_pattern_checks(&[
        PatternCheck::new("9/7 transcode internal diff helper", &dct97)
            .required(&[
                "#[cfg(test)]\nimpl Dwt97TwoDimensional<f64>",
                "pub(crate) fn max_abs_diff(&self, other: &Self) -> f64",
            ])
            .forbidden(&["pub fn max_abs_diff(&self, other: &Self) -> f64"]),
        PatternCheck::new("shared transcode DWT diff helper", &dwt_diff).required(&[
            "pub fn max_abs_diff_53(",
            "pub fn max_abs_diff_97(",
            "fn max_abs_diff_bands(",
        ]),
        PatternCheck::new(
            "9/7 transcode integration test shared diff helper",
            &dct97_test,
        )
        .required(&[
            "use j2k_transcode_test_support::max_abs_diff_97;",
            "max_abs_diff_97(&",
        ])
        .forbidden(&["mod dwt_diff;", "fn max_abs_diff("]),
    ]);

    let ycbcr12_start = jpeg_batch
        .find("fn session_batch_decode_extended12_ycbcr444_matches_single_tile_decode")
        .expect("12-bit YCbCr session batch test section");
    let ycbcr12_end = jpeg_batch
        .find("fn session_batch_decode_12bit_rgba16_matches_single_tile_decode")
        .expect("end of 12-bit YCbCr session batch test section");
    let ycbcr12_section = &jpeg_batch[ycbcr12_start..ycbcr12_end];
    assert_eq!(
        ycbcr12_section
            .matches("assert_session_batch_decode(")
            .count(),
        8,
        "12-bit YCbCr session batch cases must share the batch assertion helper"
    );
    assert!(
        !ycbcr12_section.contains("let mut outputs = vec![vec![0u8; expected.len()]"),
        "12-bit YCbCr session batch cases must not reintroduce duplicated output/job setup"
    );
}
