// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod native_contracts;

fn assert_focused_j2k_encode_module(path: &str, source: &str) {
    assert!(
        source.lines().count() < 800,
        "crates/j2k/src/{path} must stay below the focused-module line-count ratchet"
    );
    assert!(
        !source.contains("use super::*") && !source.contains("include!("),
        "crates/j2k/src/{path} must keep explicit Rust module boundaries"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "native encode ownership checks are intentionally reviewed as one fail-closed matrix"
)]
fn native_encode_options_and_tile_parts_live_in_focused_modules() {
    let root = repo_root();
    let encode = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode.rs"))
        .expect("read native encode module");
    let options = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/options.rs"))
        .expect("read native encode options module");
    let tile_parts =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/tile_parts.rs"))
            .expect("read native encode tile-part module");
    let tile_parts_consume =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/tile_parts/consume.rs"))
            .expect("read native consuming tile-part module");
    let tile_parts_finalize =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/tile_parts/finalize.rs"))
            .expect("read native borrowed tile-part finalization module");
    let tile_parts_implementation = [
        tile_parts.as_str(),
        tile_parts_consume.as_str(),
        tile_parts_finalize.as_str(),
    ]
    .join("\n");
    let precomputed_shell =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed.rs"))
            .expect("read native encode precomputed module");
    let precomputed_accelerator = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/precomputed/accelerator.rs"),
    )
    .expect("read native precomputed accelerator module");
    let precomputed_api53 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/api53.rs"))
            .expect("read native precomputed 5-3 API module");
    let precomputed_api97 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/api97.rs"))
            .expect("read native precomputed 9-7 API module");
    let precomputed_batch97 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/batch97.rs"))
            .expect("read native precomputed 9-7 batch module");
    let precomputed_batch97_prepare = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/precomputed/batch97/prepare.rs"),
    )
    .expect("read native precomputed 9-7 batch preparation module");
    let precomputed_batch97_finalize = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/precomputed/batch97/finalize.rs"),
    )
    .expect("read native precomputed 9-7 batch finalization module");
    let precomputed_compact97 =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/compact97.rs"))
            .expect("read native compact preencoded 9-7 module");
    let precomputed_compact97_construction = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/precomputed/compact97/construction.rs"),
    )
    .expect("read native compact preencoded 9-7 construction module");
    let precomputed_packets =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/packets.rs"))
            .expect("read native precomputed packet module");
    let precomputed_validation =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/precomputed/validation.rs"))
            .expect("read native precomputed validation module");
    let precomputed = [
        precomputed_shell.as_str(),
        precomputed_accelerator.as_str(),
        precomputed_api53.as_str(),
        precomputed_api97.as_str(),
        precomputed_batch97.as_str(),
        precomputed_batch97_prepare.as_str(),
        precomputed_batch97_finalize.as_str(),
        precomputed_compact97.as_str(),
        precomputed_compact97_construction.as_str(),
        precomputed_packets.as_str(),
        precomputed_validation.as_str(),
    ]
    .join("\n");
    let packet_plan =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/packet_plan.rs"))
            .expect("read native encode packet-plan module");
    let rate_control_implementation = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/encode/rate_control.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment/legacy.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted.rs",
            "crates/j2k-native/src/j2c/encode/rate_control/assignment/accounted/classic.rs",
        ],
    );
    let roi_plan = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/roi_plan.rs"))
        .expect("read native encode ROI planning module");
    let roi_construction = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/single_tile/plan/construction/roi.rs"),
    )
    .expect("read native encode ROI construction module");
    let samples = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/samples.rs"))
        .expect("read native encode sample helper module");
    let i64_packetize =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/i64_packetize.rs"))
            .expect("read native encode i64 packetization module");
    let single_tile =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/single_tile.rs"))
            .expect("read native encode single-tile module");
    let api_helpers =
        fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/api_helpers.rs"))
            .expect("read native encode public API helper module");
    let codec_math_dwt = fs::read_to_string(root.join("crates/j2k-codec-math/src/dwt.rs"))
        .expect("read shared codec-math DWT policy");
    let subband = fs::read_to_string(root.join("crates/j2k-native/src/j2c/encode/subband.rs"))
        .expect("read native encode subband preparation module");
    let typed_i64_subband = fs::read_to_string(
        root.join("crates/j2k-native/src/j2c/encode/typed_i64/prepare/subband.rs"),
    )
    .expect("read native typed-i64 subband preparation module");
    let subband_implementation = [subband.as_str(), typed_i64_subband.as_str()].join("\n");

    assert!(
        encode.lines().count() < 800,
        "j2c/encode.rs must stay below the post-split line-count ratchet"
    );
    assert!(
        precomputed_api97.lines().count() < 800,
        "j2c/encode/precomputed/api97.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_batch97.lines().count() < 150,
        "j2c/encode/precomputed/batch97.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_batch97_prepare.lines().count() < 300,
        "j2c/encode/precomputed/batch97/prepare.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_batch97_finalize.lines().count() < 250,
        "j2c/encode/precomputed/batch97/finalize.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_compact97.lines().count() < 500,
        "j2c/encode/precomputed/compact97.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        precomputed_compact97_construction.lines().count() < 420,
        "j2c/encode/precomputed/compact97/construction.rs must stay below the focused-module line-count ratchet"
    );
    assert!(
        roi_plan.lines().count() < 300,
        "j2c/encode/roi_plan.rs must stay below the ROI planning line-count ratchet"
    );
    assert!(
        subband.lines().count() < 650,
        "j2c/encode/subband.rs must stay below the subband preparation line-count ratchet"
    );

    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs option module shell", &encode).required(&[
            "mod options;",
            "pub use self::options",
            "EncodeOptions",
        ]),
    ]);
    for option_type in [
        "pub struct EncodeOptions",
        "pub struct EncodeComponentPlane",
        "pub struct EncodeTypedComponentPlane",
        "pub struct EncodeRoiRegion",
        "pub enum EncodeProgressionOrder",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs option type exclusion", &encode)
                .forbidden(&[option_type]),
            PatternCheck::new("j2c/encode/options.rs option type ownership", &options)
                .required(&[option_type]),
        ]);
    }
    for helper in [
        "fn validate_irreversible_quantization_scale",
        "pub(super) fn validate_irreversible_quantization_profile",
        "pub(super) fn validate_precinct_exponents_for_options",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs option validation helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new(
                "j2c/encode/options.rs option validation helper ownership",
                &options,
            )
            .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs tile-part module shell", &encode).required(&[
            "mod tile_parts;",
            "write_single_tile_packetized_codestream",
            "validate_packet_header_marker_payloads",
        ]),
    ]);
    for helper in [
        "struct EncodedTilePart",
        "fn consume_packetized_tile_into_tile_parts",
        "fn write_single_tile_packetized_codestream_for_session",
        "fn validate_packet_header_marker_payload",
        "fn validate_packet_header_marker_payloads",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs tile-part helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new(
                "j2c/encode/tile_parts module-family helper ownership",
                &tile_parts_implementation,
            )
            .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[PatternCheck::new(
        "j2c/encode/tile-parts focused module graph",
        &tile_parts,
    )
    .required(&["mod consume;", "mod finalize;"])]);
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs focused module wiring", &encode).required(&[
            "mod precomputed;",
            "pub use self::precomputed::{",
            "mod packet_plan;",
            "mod rate_control;",
            "mod roi_plan;",
            "mod samples;",
            "mod i64_packetize;",
            "mod single_tile;",
            "mod subband;",
        ]),
        PatternCheck::new("precomputed.rs 9-7 batch wiring", &precomputed_shell).required(&[
            "mod batch97;",
            "encode_precomputed_htj2k_97_batch_owned_with_accelerator",
            "encode_precomputed_htj2k_97_batch_with_accelerator",
        ]),
        PatternCheck::new("precomputed.rs compact 9-7 wiring", &precomputed_shell)
            .required(&["mod compact97;"]),
        PatternCheck::new(
            "precomputed/compact97.rs retained ownership",
            &precomputed_compact97,
        )
        .required(&[
            "mod construction;",
            "NativeEncodeRetainedInput::from_owner_bytes",
            "try_compact_packetization_accelerator(",
            "reconcile_compact_final_codestream(",
        ]),
        PatternCheck::new("precomputed/api97.rs batch exclusion", &precomputed_api97)
            .forbidden(&["pub fn encode_precomputed_htj2k_97_batch_with_accelerator("]),
        PatternCheck::new(
            "precomputed/batch97.rs batch ownership",
            &precomputed_batch97,
        )
        .required(&[
            "mod finalize;",
            "mod prepare;",
            "pub fn encode_precomputed_htj2k_97_batch_with_accelerator(",
            "pub fn encode_precomputed_htj2k_97_batch_owned_with_accelerator(",
            "drop(images);",
        ]),
        PatternCheck::new(
            "precomputed/batch97 preparation ownership",
            &precomputed_batch97_prepare,
        )
        .required(&[
            "prepare_batch_plans(",
            "try_reserve_exact(images.len())",
            "encode_prepared_resolution_packets_for_session(",
            "split_encoded_packets(",
        ]),
        PatternCheck::new(
            "precomputed/batch97 final ownership",
            &precomputed_batch97_finalize,
        )
        .required(&[
            "packetize_and_finalize_batch(",
            "batch_iteration_live_bytes(",
            "write_single_tile_packetized_codestream_for_session(",
            "codestream.capacity()",
        ]),
    ]);
    for helper in [
        "struct I64PacketizeRequest",
        "pub(super) fn packetize_i64_component_resolution_packets",
    ] {
        assert_pattern_checks(&[
            PatternCheck::new("j2c/encode.rs i64 packetization helper exclusion", &encode)
                .forbidden(&[helper]),
            PatternCheck::new(
                "j2c/encode/i64_packetize.rs helper ownership",
                &i64_packetize,
            )
            .required(&[helper]),
        ]);
    }
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode.rs API helper module shell", &encode).required(&[
            "mod api_helpers;",
            "public_sub_band_type",
            "internal_sub_band_type",
            "deinterleave_to_f32",
            "use j2k_codec_math::dwt::max_decomposition_levels;",
        ]),
    ]);
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode.rs single-tile implementation exclusion",
            &encode,
        )
        .forbidden(&["fn encode_impl("]),
        PatternCheck::new(
            "j2c/encode/single_tile.rs single-tile implementation",
            &single_tile,
        )
        .required(&["pub(super) fn encode_impl("]),
    ]);
    let precomputed_helpers = [
        "pub fn encode_precomputed_j2k_53",
        "pub fn encode_precomputed_htj2k_97",
        "pub fn encode_preencoded_htj2k_97",
        "pub(in crate::j2c::encode) fn validate_precomputed_dwt97_geometry",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/precomputed.rs precomputed helpers",
            precomputed.as_str(),
        )
        .required(&precomputed_helpers),
        PatternCheck::new("j2c/encode.rs precomputed helper exclusion", &encode)
            .forbidden(&precomputed_helpers),
        PatternCheck::new(
            "precomputed DWT adapter forwarding macro",
            precomputed.as_str(),
        )
        .required(&["macro_rules! forward_precomputed_encode_stage_hooks"]),
    ]);
    assert_eq!(
        precomputed
            .matches("forward_precomputed_encode_stage_hooks!();")
            .count(),
        1,
        "the shared direct-coefficient adapter must own one forwarding implementation"
    );
    for forwarded in [
        "fn dispatch_report(",
        "fn encode_quantize_subband(",
        "fn encode_tier1_code_block(",
        "fn encode_tier1_code_blocks(",
        "fn encode_ht_code_block(",
        "fn encode_ht_code_blocks(",
        "fn prefer_parallel_cpu_code_block_fallback(",
        "fn prefer_parallel_cpu_tile_encode(",
        "fn encode_packetization(",
    ] {
        assert_eq!(
            precomputed.matches(forwarded).count(),
            1,
            "precomputed DWT forwarding hook `{forwarded}` must live once in the shared macro"
        );
    }
    for defaulted in [
        "fn encode_deinterleave(",
        "fn encode_forward_rct(",
        "fn encode_forward_ict(",
        "fn encode_ht_subband(",
        "fn encode_htj2k_tile(",
    ] {
        assert_pattern_checks(&[PatternCheck::new(
            "precomputed DWT defaulted hook exclusions",
            precomputed.as_str(),
        )
        .forbidden(&[defaulted])]);
    }
    let packet_plan_helpers = [
        "pub(super) fn packet_descriptors_for_order",
        "pub(super) fn packetize_resolution_packets_with_options",
        "pub(super) fn ordered_prepared_resolution_packets",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/packet_plan.rs packet-plan helpers",
            &packet_plan,
        )
        .required(&packet_plan_helpers),
        PatternCheck::new("j2c/encode.rs packet-plan helper exclusion", &encode)
            .forbidden(&packet_plan_helpers),
    ]);
    let rate_control_helpers = [
        "fn classic_multilayer_code_block_style",
        "struct ClassicLayerBudgetAllocator",
        "fn assign_classic_segment_layers_by_slope",
        "fn assign_ht_segment_layers_by_budget",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode/rate-control module-family helpers",
            &rate_control_implementation,
        )
        .required(&rate_control_helpers),
        PatternCheck::new("j2c/encode.rs rate-control helper exclusion", &encode)
            .forbidden(&rate_control_helpers),
    ]);
    let roi_plan_helpers = [
        "pub(super) struct ComponentRoiEncodePlan",
        "pub(super) struct ComponentRoiEncodeRegion",
        "fn validate_roi_shift_for_max(",
        "pub(super) fn roi_subband_scale",
        "pub(super) fn max_total_bitplanes_for_components",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/roi_plan.rs ROI planning helpers", &roi_plan)
            .required(&roi_plan_helpers),
        PatternCheck::new("j2c/encode.rs ROI planning helper exclusion", &encode)
            .forbidden(&roi_plan_helpers),
        PatternCheck::new("single-tile ROI construction ownership", &roi_construction).required(&[
            "fn try_roi_plans(",
            "planned_region_count",
            "fn append_roi_region(",
        ]),
    ]);
    let sample_helpers = [
        "pub(super) fn raw_pixel_bytes_per_sample",
        "pub(super) fn read_le_sample_value",
        "pub(super) fn sign_extend_sample",
        "pub(super) fn native_samples_equal",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/samples.rs sample helpers", &samples)
            .required(&sample_helpers),
        PatternCheck::new("j2c/encode.rs sample helper exclusion", &encode)
            .forbidden(&sample_helpers),
    ]);
    let subband_helpers = [
        "fn apply_roi_maxshift_encode_for_session",
        "fn shift_roi_coefficient_i64",
        "fn roi_region_subband_window",
        "pub(super) fn prepare_subband(",
        "fn prepare_subband_for_session(",
        "pub(super) struct I64SubbandEncodeSettings",
        "pub(super) fn prepare_packed_subband_i64",
        "fn code_block_shapes_for_session",
        "fn subband_range_bits",
    ];
    assert_pattern_checks(&[
        PatternCheck::new(
            "j2c/encode subband module-family helpers",
            &subband_implementation,
        )
        .required(&subband_helpers),
        PatternCheck::new("j2c/encode.rs subband helper exclusion", &encode)
            .forbidden(&subband_helpers),
    ]);
    let api_helpers_patterns = [
        "pub(super) fn public_sub_band_type",
        "pub(super) fn internal_sub_band_type",
        "pub(super) fn default_public_code_block_style",
        "pub(crate) fn deinterleave_to_f32",
        "pub(crate) fn deinterleave_rgb8_unsigned_to_f32",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("j2c/encode/api_helpers.rs API helpers", &api_helpers)
            .required(&api_helpers_patterns),
        PatternCheck::new("j2c/encode.rs API helper exclusion", &encode)
            .forbidden(&api_helpers_patterns),
        PatternCheck::new("shared codec-math DWT geometry policy", &codec_math_dwt).required(&[
            "pub const fn max_decomposition_levels(width: u32, height: u32) -> u8",
            "while minimum_dimension > 1",
            "maximum_decomposition_levels_are_const_and_use_the_shorter_axis",
            "maximum_decomposition_levels_match_power_of_two_boundaries",
        ]),
        PatternCheck::new("native encoder shared DWT geometry consumer", &encode)
            .required(&["use j2k_codec_math::dwt::max_decomposition_levels;"])
            .forbidden(&["fn max_decomposition_levels("]),
        PatternCheck::new("native API helper DWT clone exclusion", &api_helpers)
            .forbidden(&["fn max_decomposition_levels("]),
    ]);
}

#[test]
fn j2k_encode_facade_lives_in_focused_modules() {
    let root = repo_root();
    let facade =
        fs::read_to_string(root.join("crates/j2k/src/encode.rs")).expect("read J2K encode facade");
    let contracts = fs::read_to_string(root.join("crates/j2k/src/encode/contracts.rs"))
        .expect("read J2K encode contracts");
    let samples = fs::read_to_string(root.join("crates/j2k/src/encode/samples.rs"))
        .expect("read J2K encode samples");
    let native = fs::read_to_string(root.join("crates/j2k/src/encode/native.rs"))
        .expect("read J2K native encode bridge");
    let routing = fs::read_to_string(root.join("crates/j2k/src/encode/routing.rs"))
        .expect("read J2K encode routing");
    let lossy = fs::read_to_string(root.join("crates/j2k/src/encode/lossy.rs"))
        .expect("read J2K lossy encode targeting");
    let validation = read_source_files(
        root,
        &[
            "crates/j2k/src/encode/validation.rs",
            "crates/j2k/src/encode/validation/component.rs",
            "crates/j2k/src/encode/validation/decode.rs",
        ],
    );

    for (path, source) in [
        ("encode.rs", facade.as_str()),
        ("encode/contracts.rs", contracts.as_str()),
        ("encode/samples.rs", samples.as_str()),
        ("encode/native.rs", native.as_str()),
        ("encode/routing.rs", routing.as_str()),
        ("encode/lossy.rs", lossy.as_str()),
        ("encode/validation.rs", validation.as_str()),
    ] {
        assert_focused_j2k_encode_module(path, source);
    }

    assert_pattern_checks(&[
        PatternCheck::new("J2K encode facade wiring", &facade).required(&[
            "mod contracts;",
            "pub use self::contracts::{",
            "mod samples;",
            "pub use self::samples::{",
            "mod native;",
            "mod routing;",
            "mod lossy;",
            "mod validation;",
            "pub fn encode_j2k_lossless(",
            "pub fn encode_j2k_lossy(",
            "pub fn j2k_lossless_decomposition_levels(",
        ]),
        PatternCheck::new("J2K encode facade ownership exclusions", &facade).forbidden(&[
            "pub enum EncodeBackendPreference",
            "pub struct J2kLosslessSamples",
            "struct RequiredEncodeStages",
            "struct LossyAttempt",
            "fn validate_lossless_roundtrip(",
            "use self::contracts::*",
            "use self::samples::*",
        ]),
        PatternCheck::new("J2K encode contract ownership", &contracts).required(&[
            "pub enum EncodeBackendPreference",
            "pub struct J2kLosslessEncodeOptions",
            "pub enum J2kRateTarget",
            "pub struct J2kLossyEncodeOptions",
            "pub struct EncodedJ2k",
            "pub struct EncodedLossyJ2k",
        ]),
        PatternCheck::new("J2K encode sample ownership", &samples).required(&[
            "pub struct J2kLosslessSamples",
            "pub struct J2kLosslessComponentSamples",
            "pub struct J2kLosslessTypedComponentSamples",
            "pub struct J2kLossySamples",
            "pub(super) fn raw_pixel_bytes_per_sample",
        ]),
        PatternCheck::new("J2K native encode bridge ownership", &native).required(&[
            "pub(super) fn encode_cpu(",
            "pub(super) fn native_roi_regions_for_samples(",
            "pub(super) fn native_lossless_options(",
            "pub(super) fn native_lossy_options(",
        ]),
        PatternCheck::new("J2K encode routing ownership", &routing).required(&[
            "pub(super) fn resolve_accelerated_encode_backend(",
            "pub(super) struct RequiredEncodeStages",
            "pub(super) fn required_encode_stages(",
            "pub(super) fn required_lossy_encode_stages(",
        ]),
        PatternCheck::new("J2K lossy target ownership", &lossy).required(&[
            "pub(super) struct LossyAttempt",
            "pub(super) fn encode_lossy_targeted(",
            "pub(super) fn encode_lossy_to_byte_target(",
            "pub(super) fn encode_lossy_to_psnr_target(",
            "pub(super) fn target_bytes_for_bpp(",
        ]),
        PatternCheck::new("J2K encode validation ownership", &validation).required(&[
            "pub(super) fn validate_lossy_roundtrip(",
            "pub(super) fn validate_lossless_roundtrip(",
            "pub(in crate::encode) fn validate_lossless_component_roundtrip(",
            "pub(in crate::encode) fn validate_lossless_typed_component_roundtrip(",
        ]),
    ]);
}

#[test]
fn jpeg_to_htj2k_options_live_in_focused_module() {
    let root = repo_root();
    let transcode = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k.rs"))
        .expect("read JPEG-to-HTJ2K transcode module");
    let options =
        fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/options.rs"))
            .expect("read JPEG-to-HTJ2K options module");
    let report = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/report.rs"))
        .expect("read JPEG-to-HTJ2K report module");
    let error = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/error.rs"))
        .expect("read JPEG-to-HTJ2K error module");
    let batch = fs::read_to_string(root.join("crates/j2k-transcode/src/jpeg_to_htj2k/batch.rs"))
        .expect("read JPEG-to-HTJ2K batch module");

    assert!(
        transcode.lines().count() < 1_770,
        "jpeg_to_htj2k.rs must stay below the post-split line-count ratchet"
    );

    let option_items = [
        "pub const JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE",
        "pub struct JpegToHtj2kEncodeOptions",
        "pub struct JpegToHtj2kOptions",
        "pub enum JpegToHtj2kCoefficientPath",
        "fn native_progression_order",
    ];
    let report_items = [
        "pub struct BatchTranscodeReport",
        "pub enum TranscodeBatchProfileRequest",
        "pub struct TranscodeTimingReport",
        "pub struct TranscodeReport",
    ];
    let error_items = [
        "pub enum JpegToHtj2kError",
        "pub(super) fn dct53_transform_error",
    ];
    let batch_facade_items = [
        "pub fn jpeg_to_htj2k_batch",
        "pub(super) fn jpeg_tile_batch_to_htj2k_with_scratch",
    ];
    let batch_items = [
        batch_facade_items[0],
        batch_facade_items[1],
        "pub(super) struct IntegerBatchTile",
        "pub(super) fn encode_float97_batch_tile",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("jpeg_to_htj2k options module shell", &transcode)
            .required(&[
                "mod options;",
                "pub use self::options",
                "JpegToHtj2kOptions",
            ])
            .forbidden(&option_items),
        PatternCheck::new("jpeg_to_htj2k option item ownership", &options).required(&option_items),
        PatternCheck::new("jpeg_to_htj2k support module wiring", &transcode).required(&[
            "mod report;",
            "mod error;",
            "mod batch;",
        ]),
        PatternCheck::new("jpeg_to_htj2k report item exclusion", &transcode)
            .forbidden(&report_items),
        PatternCheck::new("jpeg_to_htj2k error item exclusion", &transcode).forbidden(&error_items),
        PatternCheck::new("jpeg_to_htj2k batch item exclusion", &transcode).forbidden(&batch_items),
        PatternCheck::new("jpeg_to_htj2k report item ownership", &report).required(&report_items),
        PatternCheck::new("jpeg_to_htj2k error item ownership", &error).required(&error_items),
        PatternCheck::new("jpeg_to_htj2k batch facade ownership", &batch)
            .required(&batch_facade_items),
    ]);
}
