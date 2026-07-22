// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

const REMOVED_SYMBOLS: &[(&str, &[&str])] = &[
    (
        "crates/j2k-core/src/context.rs",
        &["pub struct DecoderContext"] as &[&str],
    ),
    (
        "crates/j2k-core/src/batch/collection.rs",
        &["pub fn try_collect_ordered_batch_results<T, E>("],
    ),
    (
        "crates/j2k-native/src/image.rs",
        &[
            "fn prepare_decoded_image<'ctx>(",
            "fn prepare_decoded_image_with_retained_baseline<'ctx>(",
            "fn prepare_decoded_image_with_ht_decoder<'ctx>(",
            "fn prepare_decoded_image_with_region<'ctx>(",
            "fn prepare_decoded_image_with_region_and_ht_decoder<'ctx>(",
            "fn decode_with_output_region(",
            "fn decode_with_output_region_and_ht_decoder(",
        ],
    ),
    (
        "crates/j2k-native/src/j2c/encode/precomputed/api53.rs",
        &[
            "fn encode_precomputed_53_with_mct_and_accelerator(",
            "fn encode_precomputed_53_with_component_sample_info_and_accelerator(",
        ],
    ),
    (
        "crates/j2k-native/src/image/native/allocation.rs",
        &["fn for_component_pack("],
    ),
    (
        "crates/j2k-native/src/j2c/encode/packet_plan.rs",
        &["fn public_packetization_progression_order("],
    ),
    (
        "crates/j2k-cuda/src/decoder/api.rs",
        &[
            "self.decode_to_cuda_resident_surface_impl(",
            "self.decode_region_to_cuda_resident_surface_impl(",
            "self.decode_scaled_to_cuda_resident_surface_impl(",
            "self.decode_region_scaled_to_cuda_resident_surface_impl(",
        ],
    ),
    (
        "crates/j2k-metal/src/decoder/direct_paths.rs",
        &[
            "fn decode_region_scaled_direct_to_surface(",
            "fn decode_region_scaled_direct_to_surface_with_session(",
        ],
    ),
    (
        "crates/j2k-transcode-metal/src/metal/reversible.rs",
        &["fn dispatch_reversible_dwt53_batch_with_runtime("],
    ),
    (
        "crates/j2k-metal/src/hybrid.rs",
        &["fn region_scaled_color_plan_cache_key("],
    ),
    (
        "crates/j2k-test-support/src/fixtures.rs",
        &[
            "pub fn jpeg_baseline_420_16x16(",
            "pub fn jpeg_grayscale_8x8(",
            "pub fn jpeg_baseline_444_8x8(",
            "pub fn jpeg_baseline_444_8x8_rgb(",
            "pub fn jpeg_baseline_422_16x8(",
            "pub fn jpeg_baseline_422_16x8_rgb(",
            "pub fn jpeg_baseline_420_restart_32x16(",
            "pub fn jpeg_baseline_420_restart_32x16_rgb(",
            "pub fn minimal_gray8_jpeg(",
        ],
    ),
    (
        "crates/j2k-test-support/src/pixels.rs",
        &["pub fn crop_interleaved_u8("],
    ),
    ("crates/j2k/src/recode/pixel.rs", &["fn recode_components("]),
    ("crates/j2k/src/adapter/mod.rs", &["mod encode_stage;"]),
    (
        "crates/j2k/src/lib.rs",
        &["pub use adapter::encode_stage::{"],
    ),
    (
        "crates/j2k-jpeg/src/decoder/output_format.rs",
        &["fn allocate_output_buffer("],
    ),
    (
        "xtask/src/coverage/parsing.rs",
        &["fn normalize_lcov_path("],
    ),
    (
        "crates/j2k/src/encode.rs",
        &["fn j2k_lossy_position_progression_decomposition_levels("],
    ),
    (
        "crates/j2k-types/src/resident.rs",
        &[
            "self.input.width()",
            "self.input.height()",
            "self.input.num_components()",
            "self.input.bit_depth()",
            "self.input.signed()",
        ],
    ),
];

#[test]
fn removed_pass_through_layers_stay_absent() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/traits.rs")
                .named("direct codec context ownership")
                .required(&["ctx: &mut Self::Context"]),
            FilePatternCheck::new("crates/j2k-core/src/batch/collection.rs")
                .named("bounded ordered collection ownership")
                .required(&["pub fn try_collect_ordered_batch_results_with_limits"]),
            FilePatternCheck::new("crates/j2k-native/src/image.rs")
                .named("native full decode pipeline")
                .required(&["fn decode_image<'ctx>(", "set_output_region(output_region)"]),
            FilePatternCheck::new("crates/j2k-cuda/src/encode/htj2k/resident.rs")
                .named("resident encode input ownership")
                .required(&["job.input.width()", "job.input.num_components()"]),
            FilePatternCheck::new("crates/j2k/src/lib.rs")
                .named("direct encode-stage contract ownership")
                .required(&["pub use j2k_types::{"]),
            FilePatternCheck::new("xtask/src/coverage/parsing.rs")
                .named("direct coverage path normalization")
                .required(&["normalize_coverage_path(path, root)?"]),
        ],
    );
    for &(relative, forbidden) in REMOVED_SYMBOLS {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert_pattern_checks(&[
            PatternCheck::new("removed pass-through layer", &source).forbidden(forbidden)
        ]);
    }
}
