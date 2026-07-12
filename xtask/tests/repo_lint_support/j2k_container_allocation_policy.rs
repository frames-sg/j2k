// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only JP2 metadata handoffs and single-allocation wrapper/recode ratchets.

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn public_container_and_recode_owners_do_not_expose_infallible_clone() {
    let facade_metadata = read_source_files(repo_root(), &["crates/j2k/src/metadata.rs"]);
    let facade_recode = read_source_files(repo_root(), &["crates/j2k/src/recode.rs"]);
    let native_metadata = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/jp2/container.rs",
            "crates/j2k-native/src/jp2/metadata.rs",
        ],
    );
    let native_recode = read_source_files(repo_root(), &["crates/j2k-native/src/j2c/recode.rs"]);

    assert_pattern_checks(&[
        PatternCheck::new("facade public JP2 metadata owners", &facade_metadata).forbidden(&[
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct J2kSupportInfo",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct J2kFileMetadata",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct J2kPaletteMetadata",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum J2kColorSpec",
        ]),
        PatternCheck::new("native public JP2 metadata owners", &native_metadata).forbidden(&[
            "#[derive(Debug, Clone)]\npub struct Jp2Container",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct Jp2FileMetadata",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum Jp2ColorSpec",
            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct Jp2PaletteMetadata",
        ]),
        PatternCheck::new("facade recode output owner", &facade_recode)
            .forbidden(&["#[derive(Debug, Clone, PartialEq, Eq)]\npub struct ReencodedHtj2k"]),
        PatternCheck::new("native coefficient recode owner", &native_recode)
            .forbidden(&["#[derive(Debug, Clone)]\npub struct Reversible53CoefficientImage"]),
    ]);
}

#[test]
fn lightweight_codestream_inspection_reserves_component_metadata_fallibly() {
    let inspect = read_source_files(repo_root(), &["crates/j2k-native/src/inspect.rs"]);
    let regressions = read_source_files(repo_root(), &["crates/j2k-native/src/tests.rs"]);

    assert_pattern_checks(&[
        PatternCheck::new("native lightweight codestream inspection", &inspect)
            .required(&[
                "J2kCodestreamHeaderError::HostAllocationFailed",
                "try_reserve_exact(component_len)",
                "size_of::<J2kCodestreamComponentHeader>()",
            ])
            .forbidden(&["Vec::with_capacity(usize::from(component_count))"]),
        PatternCheck::new("maximum component inspection regression", &regressions).required(&[
            "inspect_fallibly_materializes_the_maximum_component_metadata",
            "metadata.component_info.len()",
        ]),
    ]);
}

#[test]
fn native_jp2_private_graph_stays_move_only_and_fallible() {
    let native = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/jp2/mod.rs",
            "crates/j2k-native/src/jp2/container.rs",
            "crates/j2k-native/src/jp2/metadata.rs",
            "crates/j2k-native/src/jp2/colr.rs",
            "crates/j2k-native/src/jp2/pclr.rs",
            "crates/j2k-native/src/jp2/cmap.rs",
            "crates/j2k-native/src/jp2/cdef.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new("native JP2 ownership graph", &native)
        .required(&[
            "Jp2AllocationBudget",
            "fn allocated_bytes(&self) -> Result<usize>",
            "public_metadata_from_boxes(boxes: ImageBoxes)",
            "NativeColorSpace::Icc(profile) => Jp2ColorSpec::IccProfile",
            "entries,",
            "release_capacity",
        ])
        .forbidden(&[
            ".to_vec()",
            ".clone()",
            ".collect",
            "Vec::with_capacity",
            "#[derive(Debug, Clone, Default)]\npub(crate) struct ImageBoxes",
            "#[derive(Debug, Clone)]\npub(crate) struct ColorSpecificationBox",
            "#[derive(Debug, Clone)]\npub(crate) struct PaletteBox",
            "#[derive(Debug, Clone)]\npub(crate) struct ComponentMappingBox",
            "#[derive(Debug, Clone)]\npub(crate) struct ChannelDefinitionBox",
        ])]);
}

#[test]
fn facade_container_conversion_stays_consuming_and_clone_free() {
    let facade = read_source_files(
        repo_root(),
        &[
            "crates/j2k/src/parse/boxes.rs",
            "crates/j2k/src/parse/codestream.rs",
            "crates/j2k/src/parse/mod.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("facade JP2 consuming conversion", &facade)
            .required(&[
                "file_metadata_from_native(container.metadata)?",
                "ParseAllocationBudget::from_live_bytes",
                "NativeColorSpec::IccProfile { profile }",
                "J2kColorSpec::IccProfile { profile }",
                "parsed.into_parts",
                "release_capacity",
            ])
            .forbidden(&[
                ".to_vec()",
                ".clone()",
                ".collect",
                "Vec::with_capacity",
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub(crate) struct ParsedImageInfo",
            ]),
    ]);
}

#[test]
fn parsed_image_metadata_stays_inside_native_decode_and_output_budgets() {
    let ownership = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/image/allocation.rs",
            "crates/j2k-native/src/image.rs",
            "crates/j2k-native/src/j2c/codestream/allocation/header.rs",
            "crates/j2k-native/src/j2c/codestream/header.rs",
            "crates/j2k-native/src/j2c/codestream/header/allocation.rs",
            "crates/j2k-native/src/j2c/codestream/size.rs",
            "crates/j2k-native/src/j2c/mod.rs",
            "crates/j2k-native/src/jp2/mod.rs",
            "crates/j2k-native/src/jp2/container.rs",
            "crates/j2k-native/src/jp2/metadata.rs",
        ],
    );
    let decode = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/j2c/decode.rs",
            "crates/j2k-native/src/j2c/decode/direct_plan.rs",
            "crates/j2k-native/src/j2c/recode.rs",
            "crates/j2k-native/src/j2c/tile.rs",
            "crates/j2k-native/src/j2c/tile/metadata.rs",
        ],
    );
    let output = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/color/postprocess.rs",
            "crates/j2k-native/src/image/allocation.rs",
            "crates/j2k-native/src/image/native.rs",
            "crates/j2k-native/src/image/native/allocation.rs",
            "crates/j2k-native/src/image/output_api.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("native parsed-Image retained owner", &ownership).required(&[
            "retained_container_metadata_bytes",
            "retained_header_bytes(header)?",
            "boxes.allocated_bytes()?",
            "profile.capacity()",
            "Image::from_parsed_parts",
            "self.retained_metadata_bytes()?",
            "component_sizes.capacity()",
            "parse_raw_with_retained_baseline",
            "HeaderMarkerBudget::with_retained_baseline",
            "HEADER_ALLOCATION_WHAT",
            "DecodeError::AllocationTooLarge",
            "size_marker(reader, marker_budget.remaining_bytes())?",
            "try_with_synthetic_color_specification",
        ]),
        PatternCheck::new("native parsed-Image decode baseline", &decode).required(&[
            "retained_image_bytes: usize",
            "tile::parse(&mut reader, header, retained_image_bytes)?",
            "TileMetadataBudget::for_image",
            "minimum_inherited_tile_bytes(main_header)?",
            "metadata_budget.validate_owner_graph(&tiles)?",
        ]),
        PatternCheck::new("native parsed-Image output baseline", &output).required(&[
            "retained_image_bytes: usize",
            "Self::for_decoded_channels(",
            "include_capacity_overage",
            "include_color_space_clone_overage",
            "DecodeOwnerBudget::for_components",
            "include_components",
            "allocation: DecodeOwnerBudget",
            "MAX_EXACT_F32_INTEGER_BITS",
            "integer_container: exact_values",
        ]),
        PatternCheck::new("native component sampling rule", &output)
            .required(&[
                "pub(super) fn component_plane_sampling_at",
                "self.component_plane_sampling_at(component_idx)",
            ])
            .forbidden(&["fn native_component_sampling"]),
    ]);
    for relative in [
        "crates/j2k-native/src/color/postprocess.rs",
        "crates/j2k-native/src/image/allocation.rs",
        "crates/j2k-native/src/image/native.rs",
        "crates/j2k-native/src/image/native/allocation.rs",
        "crates/j2k-native/src/image/output_api.rs",
    ] {
        let source = read_source_files(repo_root(), &[relative]);
        let production = source.split("#[cfg(test)]").next().unwrap_or(&source);
        assert!(
            !production.contains("Vec::with_capacity") && !production.contains(".collect::<Vec"),
            "{relative} production allocation must stay fallible and non-collecting"
        );
    }
}

#[test]
fn native_component_output_regressions_cover_shared_rules() {
    let regressions = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/image/allocation.rs",
            "crates/j2k-native/src/color/postprocess.rs",
            "crates/j2k-native/tests/encode_coefficients.rs",
            "crates/j2k-native/tests/empty_cmap.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "native component-output regressions",
        &regressions,
    )
    .required(&[
        "component_owner_budget_accepts_exact_cap_and_rejects_one_over",
        "shared_decode_budget_uses_simd_and_integer_capacities",
        "assert_eq!(owned_sampling, borrowed_sampling);",
        "resolved palette columns must use the display grid",
        "components.has_alpha()",
        "high_precision_sycc_palette_does_not_reuse_pretransform_integer_shadow",
    ])]);
}

#[test]
fn wrapper_and_recode_keep_one_output_owner_and_no_nested_payload_vectors() {
    let wrapper = read_source_files(
        repo_root(),
        &[
            "crates/j2k/src/wrap.rs",
            "crates/j2k/src/wrap/color.rs",
            "crates/j2k/src/wrap/metadata.rs",
            "crates/j2k/src/wrap/plan.rs",
            "crates/j2k/src/wrap/writer.rs",
            "crates/j2k/src/wrap/allocation.rs",
        ],
    );
    let recode = read_source_files(
        repo_root(),
        &[
            "crates/j2k/src/recode.rs",
            "crates/j2k/src/recode/allocation.rs",
            "crates/j2k/src/recode/coefficient.rs",
            "crates/j2k/src/recode/component_grid.rs",
            "crates/j2k/src/recode/component_grid/resolved.rs",
            "crates/j2k/src/recode/output.rs",
            "crates/j2k/src/recode/pixel.rs",
            "crates/j2k/src/recode/validation.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("exact JP2/JPH writer", &wrapper)
            .required(&[
                "WrapPlan::build",
                "allocate_output(plan.total_len, retained_bytes)",
                "CheckedWriter",
                "writer exceeded its exact allocation plan",
                "wrap_recode_jph_codestream",
            ])
            .forbidden(&[".to_vec()", ".collect", "Vec::with_capacity", "push_box("]),
        PatternCheck::new("ownership-aware HTJ2K recode", &recode)
            .required(&[
                "allocation::copy_bytes",
                "output::wrap_borrowed_jph",
                "output::finalize_owned",
                "codestream.capacity()",
                "component_grid::plane_data",
                "component_grid::resolved_plane_data",
                "RecodeAllocationBudget",
            ])
            .forbidden(&[".to_vec()", ".collect", "Vec::with_capacity"]),
    ]);
}

#[test]
fn recode_roundtrip_uses_one_paired_native_decode_budget() {
    let native = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/image/allocation.rs",
            "crates/j2k-native/src/image/compare.rs",
            "crates/j2k-native/src/image.rs",
            "crates/j2k-native/src/image/output_api.rs",
        ],
    );
    let validation = read_source_files(repo_root(), &["crates/j2k/src/recode/validation.rs"]);
    let orchestration = read_source_files(repo_root(), &["crates/j2k/src/recode.rs"]);

    assert_pattern_checks(&[
        PatternCheck::new("paired native comparison budget", &native).required(&[
            "decoded_samples_equal",
            "decoded_samples_equal_with_retained_bytes",
            "retained_encoded_bytes.capacity()",
            "paired_metadata",
            "let source_bytes = source",
            ".allocated_bytes()",
            "encoded_baseline",
            "decode_native_with_context_and_retained_baseline",
            "decode_native_components_with_context_and_retained_baseline",
        ]),
        PatternCheck::new("recode paired validation", &validation)
            .required(&[".decoded_samples_equal_with_retained_bytes(&encoded_image, encoded)"])
            .forbidden(&[
                "source_image.decode_native()",
                "encoded_image.decode_native()",
                "source_image.decode_native_components()",
                "encoded_image.decode_native_components()",
            ]),
    ]);
    let drop_metadata = orchestration
        .find("drop(parsed);")
        .expect("recode must release parsed facade metadata before validation");
    let validate = orchestration
        .find("validation::roundtrip(")
        .expect("recode must retain CPU round-trip validation");
    assert!(
        drop_metadata < validate,
        "facade parsed metadata must be dropped before paired native decode"
    );
    let comparison = read_source_files(repo_root(), &["crates/j2k-native/src/image/compare.rs"]);
    assert!(
        comparison.lines().count() < 190,
        "native paired comparison must stay below its 190-line responsibility ratchet"
    );
}

#[test]
fn recode_roundtrip_parse_accounts_every_live_validation_owner() {
    let native = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/image.rs",
            "crates/j2k-native/src/j2c/mod.rs",
            "crates/j2k-native/src/jp2/mod.rs",
            "crates/j2k-native/src/jp2/container.rs",
            "crates/j2k-native/src/jp2/tests.rs",
            "crates/j2k-native/src/color.rs",
        ],
    );
    let validation = read_source_files(repo_root(), &["crates/j2k/src/recode/validation.rs"]);
    let boundary_tests = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/image/compare/tests.rs",
            "crates/j2k-native/tests/empty_cmap.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("retained native validation parse", &native).required(&[
            "pub fn new_with_retained_baseline",
            "j2c::parse_with_retained_baseline",
            "jp2::parse_with_retained_baseline",
            "parse_raw_with_retained_baseline",
            "retained_baseline_bytes",
            "from_parsed_parts_with_retained_baseline",
            "retained_jp2_parse_baseline_covers_nested_icc_and_palette_owners",
            "retained_color_profile_peak_accepts_exact_cap_and_rejects_one_over",
            "implicit_mapping_allocation_counts_external_baseline_at_exact_boundary",
            "sycc_conversion_discards_pretransform_integer_shadows",
        ]),
        PatternCheck::new("paired validation parse call path", &validation).required(&[
            "let source_image = Image::new_with_retained_baseline",
            "source_image.retained_allocation_bytes()",
            "encoded.capacity().saturating_add(source_metadata_bytes)",
            "Image::new_with_retained_baseline(encoded, &settings, encoded_parse_baseline)",
        ]),
        PatternCheck::new("retained parse boundary", &boundary_tests).required(&[
            "retained_parse_baseline_is_enforced_before_parser_growth",
            "DEFAULT_MAX_DECODE_BYTES - 1",
            "paired_comparison_preserves_mixed_palette_precision_and_signedness",
            "16-bit signed palette differences must not be truncated",
            "paired_comparison_preserves_palette_integers_above_f32_precision",
            "adjacent 25-bit palette values above 2^24 must remain distinguishable",
            "retained_full_jp2_parse_baseline_covers_implicit_palette_mapping",
            "retained_full_jp2_parse_baseline_covers_icc_profile_clone",
            "assert_full_parse_respects_retained_baseline",
        ]),
    ]);
}

#[test]
fn wrapper_and_recode_responsibility_modules_stay_focused() {
    for (relative, max_lines) in [
        ("crates/j2k/src/wrap.rs", 325),
        ("crates/j2k/src/wrap/plan.rs", 175),
        ("crates/j2k/src/wrap/color.rs", 175),
        ("crates/j2k/src/wrap/metadata.rs", 310),
        ("crates/j2k/src/wrap/writer.rs", 335),
        ("crates/j2k/src/wrap/allocation.rs", 100),
        ("crates/j2k/src/recode.rs", 275),
        ("crates/j2k/src/recode/allocation.rs", 200),
        ("crates/j2k/src/recode/coefficient.rs", 135),
        ("crates/j2k/src/recode/component_grid.rs", 150),
        ("crates/j2k/src/recode/component_grid/resolved.rs", 130),
        ("crates/j2k/src/recode/output.rs", 75),
        ("crates/j2k/src/recode/pixel.rs", 225),
        ("crates/j2k/src/recode/validation.rs", 85),
    ] {
        let source = read_source_files(repo_root(), &[relative]);
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
    }
}

#[test]
fn native_image_decode_responsibilities_stay_focused() {
    for (relative, max_lines) in [
        ("crates/j2k-native/src/image.rs", 800),
        ("crates/j2k-native/src/image/direct_api.rs", 130),
        ("crates/j2k-native/src/image/output_api.rs", 400),
        ("crates/j2k-native/src/image/native.rs", 500),
        ("crates/j2k-native/src/image/native/allocation.rs", 425),
        (
            "crates/j2k-native/src/image/native/allocation/tests.rs",
            200,
        ),
    ] {
        let source = read_source_files(repo_root(), &[relative]);
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
    }
}

#[test]
fn native_jp2_responsibilities_stay_split_and_focused() {
    let root = repo_root();
    let facade = read_source_files(root, &["crates/j2k-native/src/jp2/mod.rs"]);
    let container = read_source_files(root, &["crates/j2k-native/src/jp2/container.rs"]);
    let metadata = read_source_files(root, &["crates/j2k-native/src/jp2/metadata.rs"]);
    let image_header = read_source_files(root, &["crates/j2k-native/src/jp2/image_header.rs"]);
    let validation = read_source_files(root, &["crates/j2k-native/src/jp2/validation.rs"]);

    assert_pattern_checks(&[
        PatternCheck::new("native JP2 facade wiring", &facade)
            .required(&[
                "mod container;",
                "mod image_header;",
                "mod metadata;",
                "mod validation;",
                "pub use self::container::{",
                "pub use self::metadata::{",
            ])
            .forbidden(&[
                "pub struct Jp2FileMetadata",
                "fn parse_jp2_container_with_strict(",
                "fn parse_image_header(",
                "fn validate_component_precision_metadata(",
            ]),
        PatternCheck::new("native JP2 container ownership", &container)
            .required(&[
                "pub fn inspect_jp2_container(",
                "pub fn extract_jp2_codestream_payload(",
                "pub(crate) fn parse_with_retained_baseline(",
                "fn parse_jp2_container_with_strict(",
                "pub(super) fn parse_jp2_header_box(",
                "fn count_color_specification_boxes(",
            ])
            .forbidden(&[
                "pub struct Jp2FileMetadata",
                "fn parse_component_descriptor(",
                "fn validate_component_precision_metadata(",
            ]),
        PatternCheck::new("native JP2 metadata ownership", &metadata)
            .required(&[
                "pub(crate) struct ImageBoxes",
                "pub struct Jp2FileMetadata",
                "pub(super) fn public_metadata_from_boxes(",
                "fn public_color_spec(",
                "fn allocated_bytes(&self) -> Result<usize>",
            ])
            .forbidden(&[
                "fn parse_jp2_header_box(",
                "fn validate_codestream_file_kind(",
            ]),
        PatternCheck::new("native JP2 image-header ownership", &image_header).required(&[
            "pub(super) fn parse_image_header(",
            "pub(super) fn parse_bits_per_component(",
            "fn parse_component_descriptor(",
        ]),
        PatternCheck::new("native JP2 consistency ownership", &validation).required(&[
            "pub(super) fn validate_codestream_file_kind(",
            "pub(super) fn validate_image_header_matches_codestream(",
            "pub(super) fn validate_component_precision_metadata(",
            "fn resolved_image_component_descriptor(",
        ]),
    ]);

    for (relative, max_lines) in [
        ("crates/j2k-native/src/jp2/mod.rs", 60),
        ("crates/j2k-native/src/jp2/allocation.rs", 200),
        ("crates/j2k-native/src/jp2/box.rs", 165),
        ("crates/j2k-native/src/jp2/cdef.rs", 110),
        ("crates/j2k-native/src/jp2/cmap.rs", 90),
        ("crates/j2k-native/src/jp2/colr.rs", 175),
        ("crates/j2k-native/src/jp2/container.rs", 425),
        ("crates/j2k-native/src/jp2/icc.rs", 100),
        ("crates/j2k-native/src/jp2/image_header.rs", 80),
        ("crates/j2k-native/src/jp2/metadata.rs", 500),
        ("crates/j2k-native/src/jp2/pclr.rs", 110),
        ("crates/j2k-native/src/jp2/tests.rs", 225),
        ("crates/j2k-native/src/jp2/validation.rs", 175),
    ] {
        let source = read_source_files(root, &[relative]);
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its {max_lines}-line responsibility ratchet"
        );
        assert!(
            !source.contains("use super::*") && !source.contains("include!("),
            "{relative} must keep explicit Rust module boundaries"
        );
    }
}
