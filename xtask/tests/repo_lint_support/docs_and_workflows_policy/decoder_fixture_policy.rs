// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod shared_codec;

#[test]
fn jpeg_decoder_upsample_sample_width_twins_use_generic_helpers() {
    let root = repo_root();
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
            "crates/j2k-jpeg/src/decoder/extended12/writers.rs",
            "crates/j2k-jpeg/src/decoder/lossless_helpers.rs",
        ],
    );

    assert_pattern_checks(&[PatternCheck::new(
        "JPEG decoder sample-width upsample helpers",
        &decoder,
    )
    .required(&[
        "trait UpsampleSample",
        "impl UpsampleSample for u8",
        "impl UpsampleSample for u16",
        "fn upsample_h2v1_sample_at<S: UpsampleSample>",
        "fn upsample_h2v2_rows_at<S: UpsampleSample>",
        "upsample_h2v1_sample_at(",
        "upsample_h2v2_rows_at(current, near, output_width, output_x)",
        "upsample_h2v2_u16_rows_at(",
    ])
    .forbidden(&[
        "fn upsample_h2v1_12bit_at",
        "fn upsample_h2v2_12bit_at",
        "3 * u32::from(row[sample])",
        "let colsum = |index: usize| 3 * u32::from(current[index])",
    ])]);
}

#[test]
fn mirrored_twin_unification_record_is_current() {
    let root = repo_root();
    let record = fs::read_to_string(root.join("engineering/mirrored-twin-unification.md"))
        .expect("read mirrored-twin unification record");
    let compute = fs::read_to_string(root.join("crates/j2k-metal/src/compute.rs"))
        .expect("read Metal compute root");
    let direct_prepare_classic =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_prepare/classic.rs"))
            .expect("read Metal classic direct preparation");
    let direct_roi = fs::read_to_string(root.join("crates/j2k-metal/src/compute/direct_roi.rs"))
        .expect("read Metal direct ROI");
    let hybrid =
        fs::read_to_string(root.join("crates/j2k-metal/src/hybrid.rs")).expect("read hybrid");
    let decoder = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/extended12.rs",
            "crates/j2k-jpeg/src/decoder/extended12/planes.rs",
            "crates/j2k-jpeg/src/decoder/extended12/upsample.rs",
            "crates/j2k-jpeg/src/decoder/extended12/writers.rs",
        ],
    );
    let neon = fs::read_to_string(root.join("crates/j2k-jpeg/src/backend/neon.rs"))
        .expect("read JPEG NEON backend");
    let native_idwt = read_source_files(
        root,
        &[
            "crates/j2k-native/src/j2c/idwt.rs",
            "crates/j2k-native/src/j2c/idwt/orchestrate.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("mirrored-twin unification record", &record).required(&[
            "## Unified Families",
            "## Documented Waivers",
            "## Golden Checks",
            "Metal direct required-region retain",
            "Metal direct sub-band group scanning",
            "Metal hybrid region-scaled planning",
            "JPEG sample-width upsample helpers",
            "Extended12 versus Progressive12 JPEG decode is not a safe merge target",
            "NEON `dual` versus `top_only` row-pair kernels are intentionally separate",
            "Native IDWT f32 versus i64 remains separate",
            "cargo test -p j2k-jpeg --test decode_into progressive12_ycbcr420",
            "cargo test -p j2k-jpeg --test neon_hot_paths",
        ]),
        PatternCheck::new("Metal direct twin-unification module shell", &compute)
            .required(&["mod direct_prepare;", "mod direct_roi;"]),
        PatternCheck::new("Metal direct retain unification", &direct_roi)
            .required(&["fn retain_jobs_for_required_region<J: RequiredRegionJob>"]),
        PatternCheck::new(
            "Metal direct sub-band grouping unification",
            &direct_prepare_classic,
        )
        .required(&["fn prepare_sub_band_groups<'a, SubBand: 'a, Group>"]),
        PatternCheck::new("Metal hybrid region-scaled planning unification", &hybrid)
            .required(&["enum RegionScaledColorPlanCache"]),
        PatternCheck::new("JPEG sample-width and extended12 waiver evidence", &decoder).required(
            &[
                "trait UpsampleSample",
                "struct Extended12WriteRegion",
                "fn render_progressive12_color_planes(",
                "fn decode_extended12_color_planes(",
            ],
        ),
        PatternCheck::new("JPEG NEON row-pair waiver evidence", &neon).required(&[
            "unsafe fn fill_rgb_row_pair_from_420_neon(",
            "unsafe fn fill_rgb_row_pair_from_420_neon_top_only(",
        ]),
        PatternCheck::new("native IDWT f32/i64 waiver evidence", &native_idwt)
            .required(&["pub(crate) fn apply(", "fn apply_i64("]),
    ]);
}

#[test]
fn jpeg_fixture_builders_tables_and_reference_decode_are_split() {
    let root = repo_root();
    let module = fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures.rs"))
        .expect("read JPEG fixture module");
    let builders =
        fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures/builders.rs"))
            .expect("read JPEG fixture builders");
    let reference = fs::read_to_string(
        root.join("crates/j2k-test-support/src/jpeg_fixtures/reference_decode.rs"),
    )
    .expect("read JPEG fixture reference decode helpers");
    let tables =
        fs::read_to_string(root.join("crates/j2k-test-support/src/jpeg_fixtures/tables.rs"))
            .expect("read JPEG fixture tables");

    assert_pattern_checks(&[
        PatternCheck::new("jpeg_fixtures.rs small re-export shell", &module)
            .required(&[
                "mod builders;",
                "mod reference_decode;",
                "mod tables;",
                "pub use builders::*;",
                "pub use tables::*;",
            ])
            .forbidden(&["pub fn "]),
        PatternCheck::new("jpeg fixture builders.rs fixture builders", &builders)
            .required(&[
                "pub fn minimal_baseline_420_jpeg",
                "pub fn extended_12bit_rgb_8x8_jpeg",
                "pub fn lossless_predictor_rgb_3x3_jpeg",
                "pub fn progressive_12bit_cmyk_8x8_jpeg",
            ])
            .forbidden(&[
                "pub const LOSSLESS_GRAYSCALE_3X3_PIXELS",
                "fn ycbcr8_pixels_to_rgb8",
                "enum ColorSpaceFixture",
                "fn upsample_h2v2_12bit_for_fixture",
            ]),
        PatternCheck::new("jpeg fixture tables.rs table ownership", &tables).required(&[
            "pub const LOSSLESS_GRAYSCALE_3X3_PIXELS",
            "pub(super) const LOSSLESS_RGB_8BIT_422_4X2_C0",
            "pub(super) struct Lossless422Planes",
            "pub(super) c0:",
        ]),
        PatternCheck::new(
            "jpeg fixture reference_decode.rs helper ownership",
            &reference,
        )
        .required(&[
            "pub(super) fn ycbcr8_pixels_to_rgb8",
            "pub(super) fn ycbcr16_pixels_to_rgb16",
            "pub(super) enum ColorSpaceFixture",
            "pub(super) fn upsample_h2v2_12bit_for_fixture",
        ])
        .forbidden(&["_jpeg() -> Vec<u8>"]),
    ]);
    assert!(
        module.lines().count() < 50,
        "jpeg_fixtures.rs must stay below the re-export shell line-count ratchet"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "compare-binary ownership and duplication guards form one cohesive policy"
)]
fn compare_bins_use_library_common_helpers() {
    let root = repo_root();
    let common = fs::read_to_string(root.join("crates/j2k-compare/src/common.rs"))
        .expect("read j2k-compare common library module");
    let lib = fs::read_to_string(root.join("crates/j2k-compare/src/lib.rs"))
        .expect("read j2k-compare lib");
    let fixture = fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare.rs"))
        .expect("read fixture compare module");
    let fixture_cli =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/cli.rs"))
            .expect("read fixture compare CLI module");
    let fixture_manifest =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/manifest.rs"))
            .expect("read fixture compare manifest module");
    let fixture_rows =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/rows.rs"))
            .expect("read fixture compare rows module");
    let fixture_comparators =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/comparators.rs"))
            .expect("read fixture compare comparators module");
    let fixture_gates =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/gates.rs"))
            .expect("read fixture compare publication gates module");
    let fixture_types =
        fs::read_to_string(root.join("crates/j2k-compare/src/fixture_compare/types.rs"))
            .expect("read fixture compare types module");
    let encode_cli = fs::read_to_string(root.join("crates/j2k-compare/src/encode_compare/cli.rs"))
        .expect("read encode compare CLI module");
    let fixture_bin =
        fs::read_to_string(root.join("crates/j2k-compare/src/bin/jp2k_fixture_compare.rs"))
            .expect("read fixture compare bin");
    let encode_bin =
        fs::read_to_string(root.join("crates/j2k-compare/src/bin/jp2k_encode_compare.rs"))
            .expect("read encode compare bin");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-compare library modules", &lib).required(&[
            "pub mod common;",
            "pub mod fixture_compare;",
            "pub mod encode_compare;",
        ]),
        PatternCheck::new("fixture compare bin launcher", &fixture_bin)
            .required(&["j2k_compare::fixture_compare::main();"]),
        PatternCheck::new("encode compare bin launcher", &encode_bin)
            .required(&["j2k_compare::encode_compare::main();"]),
    ]);
    assert!(
        fixture.lines().count() < 600,
        "fixture_compare.rs must stay below the focused-coordinator line-count ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare manifest module shell", &fixture)
            .required(&["mod manifest;"])
            .forbidden(&[
                "fn fixture_manifest_from_env(",
                "fn external_fixture_metadata(",
            ]),
        PatternCheck::new("fixture_compare/manifest.rs ownership", &fixture_manifest).required(&[
            "pub(super) fn fixture_manifest_from_env",
            "pub(super) fn external_fixture_metadata",
        ]),
        PatternCheck::new("fixture_compare rows module shell", &fixture)
            .required(&["mod rows;"])
            .forbidden(&[
                "fn measurement_row(",
                "fn mixed_measurement_row(",
                "fn skip_row(",
                "fn mixed_skip_row(",
            ]),
        PatternCheck::new("fixture_compare/rows.rs ownership", &fixture_rows).required(&[
            "pub(super) fn measurement_row",
            "pub(super) fn mixed_measurement_row",
            "pub(super) fn skip_row",
            "pub(super) fn mixed_skip_row",
        ]),
        PatternCheck::new("fixture_compare comparators module shell", &fixture)
            .required(&["mod comparators;"])
            .forbidden(&[
                "fn decode_openjph_once(",
                "fn decode_kakadu_once(",
                "fn read_cli_pnm_output(",
                "OPENJPH_TEMP_COUNTER",
                "KAKADU_TEMP_COUNTER",
            ]),
        PatternCheck::new(
            "fixture_compare/comparators.rs ownership",
            &fixture_comparators,
        )
        .required(&[
            "pub(super) fn decode_openjph_once",
            "pub(super) fn decode_kakadu_once",
            "pub(super) fn openjph_is_available",
            "pub(super) fn kakadu_is_available",
            "fn read_cli_pnm_output",
        ]),
        PatternCheck::new("fixture_compare gates module shell", &fixture)
            .required(&["mod gates;"])
            .forbidden(&[
                "fn publication_blockers(",
                "fn publication_gate_skipped_comparators_label(",
                "fn require_mixed_fixture_group(",
                "fn external_unique_input_count_for_format_operation(",
            ]),
        PatternCheck::new("fixture_compare/gates.rs ownership", &fixture_gates).required(&[
            "pub(super) fn publication_blockers",
            "pub(super) fn publication_gate_skipped_comparators_label",
            "fn require_mixed_fixture_group",
            "fn external_unique_input_count_for_format_operation",
        ]),
        PatternCheck::new("fixture_compare types module shell", &fixture)
            .required(&["mod types;"])
            .forbidden(&[
                "enum BenchmarkMode",
                "enum Codec",
                "enum Container",
                "enum Operation",
                "enum OperationClass",
                "enum DecoderKind",
            ]),
        PatternCheck::new("fixture_compare/types.rs ownership", &fixture_types).required(&[
            "pub(super) enum BenchmarkMode",
            "pub(super) enum Codec",
            "pub(super) enum Container",
            "pub(super) enum Operation",
            "pub(super) enum OperationClass",
            "pub(super) enum DecoderKind",
        ]),
    ]);
    assert!(
        !root
            .join("crates/j2k-compare/src/bin/common/mod.rs")
            .exists(),
        "compare bins must not reintroduce a bin-local common module"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-compare common helper ownership", &common).required(&[
            "pub struct BatchSizeConfig",
            "pub struct BatchSizeEnv",
            "pub fn batch_size_config_from_env",
            "pub fn batch_size_config_from_values",
            "pub fn legacy_batch_sizes_from_env",
        ]),
        PatternCheck::new("fixture_compare shared batch-size helper use", &fixture_cli)
            .required(&[
                "use super::{",
                "common::batch_size_config_from_env(",
                "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
            ])
            .forbidden(&[
                "mod common;",
                "struct BatchSizeConfig",
                "fn batch_size_config_from_values",
                "fn legacy_batch_sizes_from_env",
            ]),
        PatternCheck::new("encode_compare shared batch-size helper use", &encode_cli)
            .required(&[
                "use super::{",
                "common::batch_size_config_from_env(",
                "J2K_ENCODE_COMPARE_CASE_BATCH_SIZES",
            ])
            .forbidden(&[
                "mod common;",
                "struct BatchSizeConfig",
                "fn batch_size_config_from_values",
                "fn legacy_batch_sizes_from_env",
            ]),
    ]);
}

#[test]
fn deinterleave_reference_has_only_checked_public_entrypoint() {
    let root = repo_root();
    let native = fs::read_to_string(root.join("crates/j2k-native/src/scalar/encode.rs"))
        .expect("read native scalar encode module");
    let scalar = fs::read_to_string(root.join("crates/j2k-native/src/scalar.rs"))
        .expect("read native scalar module");
    let native_root = fs::read_to_string(root.join("crates/j2k-native/src/lib.rs"))
        .expect("read native crate root");
    let cuda_parity = fs::read_to_string(root.join("crates/j2k-cuda/tests/htj2k_encode_parity.rs"))
        .expect("read CUDA parity tests");
    let metal_parity = fs::read_to_string(root.join("crates/j2k-metal/src/encode/tests.rs"))
        .expect("read Metal encode tests");
    let metal_bench =
        fs::read_to_string(root.join("crates/j2k-metal/tests/encode_auto_routing_benchmark.rs"))
            .expect("read Metal encode benchmark tests");

    assert_pattern_checks(&[
        PatternCheck::new("j2k-native checked deinterleave reference", &native)
            .required(&[
                "pub fn try_deinterleave_reference",
                "checked_deinterleave_reference_bytes_per_sample",
                "checked_decode_byte_len3",
                "ValidationError::InvalidComponentMetadata",
            ])
            .forbidden(&["pub fn deinterleave_reference", ".expect("]),
        PatternCheck::new("j2k-native scalar checked re-export", &scalar)
            .required(&["try_deinterleave_reference"])
            .forbidden(&[" deinterleave_reference,"]),
        PatternCheck::new("j2k-native root checked re-export", &native_root)
            .required(&["try_deinterleave_reference"])
            .forbidden(&[" deinterleave_reference,"]),
        PatternCheck::new(
            "CUDA parity tests checked deinterleave reference",
            &cuda_parity,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
        PatternCheck::new(
            "Metal encode tests checked deinterleave reference",
            &metal_parity,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
        PatternCheck::new(
            "Metal encode routing tests checked deinterleave reference",
            &metal_bench,
        )
        .required(&["try_deinterleave_reference"])
        .forbidden(&[" deinterleave_reference,", "= deinterleave_reference("]),
    ]);
}

#[test]
fn decode_strictness_policy_is_explicit_and_warns_on_lenient_default() {
    let root = repo_root();
    let native = fs::read_to_string(root.join("crates/j2k-native/src/image.rs"))
        .expect("read native image module");
    let facade_decode =
        fs::read_to_string(root.join("crates/j2k/src/decode.rs")).expect("read facade decode");
    let facade_view = read_source_files(
        root,
        &["crates/j2k/src/view.rs", "crates/j2k/src/view/traits.rs"],
    );
    let facade_batch = read_source_files(
        root,
        &["crates/j2k/src/batch.rs", "crates/j2k/src/batch/direct.rs"],
    );
    let crate_readme =
        fs::read_to_string(root.join("crates/j2k/README.md")).expect("read j2k README");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("read architecture docs");

    assert_pattern_checks(&[
        PatternCheck::new("native DecodeSettings strictness constructors", &native).required(&[
            "pub const fn lenient() -> Self",
            "pub const fn strict() -> Self",
            "pub const fn lenient_tolerance_enabled",
            "Self::lenient()",
        ]),
        PatternCheck::new("j2k facade decode warnings", &facade_decode).required(&[
            "pub enum J2kDecodeWarning",
            "LenientDecodeMode",
            "decode_warnings_for_settings",
            "DecodeOutcome<J2kDecodeWarning>",
        ]),
        PatternCheck::new("j2k facade view warning propagation", &facade_view).required(&[
            "type Warning = J2kDecodeWarning",
            "decode_warnings_for_settings(DecodeSettings::default())",
        ]),
        PatternCheck::new("j2k facade batch warning propagation", &facade_batch).required(&[
            "DecodeOutcome<J2kDecodeWarning>",
            "decode_warnings_for_settings(DecodeSettings::default())",
        ]),
        PatternCheck::new("j2k README decode strictness policy", &crate_readme).required(&[
            "DecodeSettings::strict()",
            "J2kDecodeWarning::LenientDecodeMode",
        ]),
        PatternCheck::new("architecture decode strictness policy", &architecture).required(&[
            "DecodeSettings::strict()",
            "J2kDecodeWarning::LenientDecodeMode",
        ]),
    ]);
}
