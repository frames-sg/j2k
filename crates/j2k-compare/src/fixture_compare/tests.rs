// SPDX-License-Identifier: MIT OR Apache-2.0

use super::fixtures::{collect_j2k_paths, container_from_path_and_bytes};
use super::metadata::{
    external_manifest_covered_case_count, external_manifest_missing_case_count,
    mixed_external_group_distinct_inputs_label, skipped_comparators_label,
};
use super::{
    canonicalize_manifest_row_path, publication_blockers, unique_input_count, BenchmarkMode, Codec,
    Container, DecoderKind, FixtureCase, MixedFixtureBatch, Operation, OperationClass,
    DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES,
};
use crate::common;
use j2k_core::{Downscale, PixelFormat, Rect};
use std::path::Path;

fn test_batch_size_config_from_values(
    case_batch_sizes: Option<&str>,
    mixed_batch_sizes: Option<&str>,
    legacy: Option<Vec<usize>>,
) -> Result<common::BatchSizeConfig, String> {
    common::batch_size_config_from_values(
        case_batch_sizes,
        mixed_batch_sizes,
        legacy,
        "J2K_FIXTURE_COMPARE_CASE_BATCH_SIZES",
        "J2K_FIXTURE_COMPARE_MIXED_BATCH_SIZES",
        DEFAULT_CASE_BATCH_SIZES,
        DEFAULT_MIXED_BATCH_SIZES,
    )
}

#[test]
fn decode_batch_config_defaults_keep_large_batches_mixed_only() {
    let config =
        test_batch_size_config_from_values(None, None, None).expect("default batch config parses");

    assert_eq!(config.case_batch_sizes, DEFAULT_CASE_BATCH_SIZES);
    assert_eq!(config.mixed_batch_sizes, DEFAULT_MIXED_BATCH_SIZES);
}

#[test]
fn decode_batch_config_split_env_overrides_legacy_independently() {
    let config = test_batch_size_config_from_values(Some("3"), None, Some(vec![2, 4]))
        .expect("case override with legacy config parses");

    assert_eq!(config.case_batch_sizes, vec![3]);
    assert_eq!(config.mixed_batch_sizes, vec![2, 4]);

    let config = test_batch_size_config_from_values(None, Some("8,16"), Some(vec![2, 4]))
        .expect("mixed override with legacy config parses");

    assert_eq!(config.case_batch_sizes, vec![2, 4]);
    assert_eq!(config.mixed_batch_sizes, vec![8, 16]);
}

#[test]
fn decode_manifest_path_remaps_to_supplied_fixture_root_by_suffix() {
    let root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-manifest-remap-test")
        .join(std::process::id().to_string());
    let fixture_root = root.join("decode-fixtures");
    let fixture = fixture_root.join("classic").join("sample.jp2");
    std::fs::create_dir_all(fixture.parent().expect("fixture parent")).expect("create dirs");
    std::fs::write(&fixture, b"jp2").expect("fixture");

    let resolved = canonicalize_manifest_row_path(
        "/old/worktree/target/j2k-public-corpora/materialized-kodak/decode-fixtures/classic/sample.jp2",
        Path::new("/unused"),
        &[fixture_root],
        "fixture manifest",
        Path::new("fixtures.tsv"),
        2,
    )
    .expect("remap stale absolute path");

    assert_eq!(resolved, fixture.canonicalize().expect("canonical fixture"));
}

#[test]
fn recursive_external_discovery_accepts_jhc_and_empty_dirs() {
    let root = std::env::current_dir()
        .expect("current dir")
        .join("target")
        .join("j2k-fixture-recursive-discovery-test")
        .join(std::process::id().to_string());
    let empty = root.join("empty");
    let nested = root.join("nested").join("deeper");
    std::fs::create_dir_all(&empty).expect("create empty dir");
    std::fs::create_dir_all(&nested).expect("create nested dir");
    let fixture = nested.join("tile.jhc");
    std::fs::write(&fixture, b"not-a-real-codestream").expect("write fixture");
    std::fs::write(nested.join("ignore.txt"), b"ignore").expect("write ignored file");

    let mut empty_paths = Vec::new();
    collect_j2k_paths(&empty, &mut empty_paths).expect("collect empty dir");
    assert!(empty_paths.is_empty());

    let mut paths = Vec::new();
    collect_j2k_paths(&root, &mut paths).expect("collect recursive paths");
    assert_eq!(paths, vec![fixture]);
    assert_eq!(
        container_from_path_and_bytes(Path::new("tile.jhc"), b"not-a-real-codestream"),
        Container::Jhc
    );
}

#[test]
fn manifest_status_and_source_digest_own_external_publication_counts() {
    let covered = fixture_case(
        "covered",
        "external:fixture",
        Some("source-a"),
        Container::Jp2,
    );
    let mut missing = fixture_case(
        "missing",
        "external:fixture",
        Some("source-b"),
        Container::RawCodestream,
    );
    missing.manifest_status = "missing".to_string();

    assert_eq!(
        external_manifest_covered_case_count(std::slice::from_ref(&covered)),
        1
    );
    assert_eq!(
        external_manifest_missing_case_count(std::slice::from_ref(&missing)),
        1
    );

    let mut jp2_variant = covered.clone();
    jp2_variant.bytes = b"different container bytes".to_vec();
    jp2_variant.container = Container::Jph;
    assert_eq!(unique_input_count(&[covered, jp2_variant]), 1);
}

#[test]
fn mixed_labels_skip_labels_blockers_and_rows_have_direct_owners() {
    let mut gray = fixture_case("gray", "external:gray", Some("gray-a"), Container::Jph);
    gray.format = PixelFormat::Gray8;
    gray.codec = Codec::Htj2k;
    gray.operation = Operation::RegionScaled {
        roi: Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        },
        scale: Downscale::Quarter,
    };
    let mut rgb = fixture_case("rgb", "external:rgb", Some("rgb-a"), Container::Jp2);
    rgb.format = PixelFormat::Rgb8;

    let mixed = MixedFixtureBatch {
        name: "external_mixed_gray8_full".to_string(),
        cases: vec![gray.clone(), rgb.clone()],
        format: PixelFormat::Gray8,
        operation_class: OperationClass::Full,
    };
    assert_eq!(
        mixed_external_group_distinct_inputs_label(std::slice::from_ref(&mixed)),
        "external_mixed_gray8_full:2"
    );

    let mixed_row = super::rows::mixed_skip_row(
        BenchmarkMode::Capability,
        DecoderKind::OpenJpeg,
        &mixed,
        2,
        4,
        "synthetic-mixed-skip",
    );
    let mixed_columns = mixed_row.split('\t').collect::<Vec<_>>();
    assert_eq!(mixed_columns.len(), 29);
    assert_eq!(
        &mixed_columns[..10],
        [
            "openjpeg",
            "external_mixed_gray8_full",
            "capability",
            "skipped",
            "external:mixed",
            "natural-image",
            "unit-corpus",
            "cc0",
            "unit-fixture",
            "covered",
        ]
    );
    assert_eq!(mixed_columns.last(), Some(&"synthetic-mixed-skip"));

    let skipped = skipped_comparators_label(BenchmarkMode::Capability, &[gray.clone()]);
    assert!(skipped.contains("openjpeg:openjpeg-htj2k-roi-scaled-noncomparable"));
    assert!(skipped.contains("openjpeg:openjpeg-external-gray-roi-scaled-noncomparable"));

    let blockers = publication_blockers(
        BenchmarkMode::PortableNative,
        1,
        &[1],
        &[1],
        false,
        &[gray.clone()],
        &[],
    );
    assert!(blockers.contains(&"mixed-external-batches-missing".to_string()));
    assert!(blockers.contains(&"external-case-count-below-24".to_string()));

    let row = super::rows::skip_row(
        BenchmarkMode::Capability,
        super::DecoderKind::OpenJpeg,
        &gray,
        1,
        1,
        "openjpeg-htj2k-roi-scaled-noncomparable",
    );
    assert_eq!(row.split('\t').count(), 29);
}

#[test]
fn decode_helpers_cover_crop_labels_flattening_and_mixed_rotation() {
    let gray = (0_u8..16).collect::<Vec<_>>();
    let cropped = super::decode::crop_interleaved(
        &gray,
        (4, 4),
        Rect {
            x: 1,
            y: 1,
            w: 2,
            h: 2,
        },
        PixelFormat::Gray8,
    )
    .expect("gray crop");
    assert_eq!(cropped, [5, 6, 9, 10]);

    let rgb = (0_u8..24).collect::<Vec<_>>();
    let cropped = super::decode::crop_interleaved(
        &rgb,
        (4, 2),
        Rect {
            x: 2,
            y: 0,
            w: 2,
            h: 2,
        },
        PixelFormat::Rgb8,
    )
    .expect("RGB crop");
    assert_eq!(cropped, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]);

    assert!(super::decode::crop_interleaved(
        &gray,
        (4, 4),
        Rect {
            x: 3,
            y: 3,
            w: 2,
            h: 2,
        },
        PixelFormat::Gray8,
    )
    .unwrap_err()
    .contains("exceeds"));
    assert!(super::decode::crop_interleaved(
        &gray[..15],
        (4, 4),
        Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        },
        PixelFormat::Gray8,
    )
    .unwrap_err()
    .contains("scaled source length"));

    assert_eq!(
        super::decode::flatten_outputs(vec![vec![1, 2], vec![], vec![3]]),
        [1, 2, 3]
    );
    assert_eq!(
        super::decode::pixel_format_label(PixelFormat::Gray8),
        "gray8"
    );
    assert_eq!(super::decode::pixel_format_label(PixelFormat::Rgb8), "rgb8");
    assert_eq!(
        super::decode::pixel_format_label(PixelFormat::Rgba8),
        "unsupported"
    );

    let first = fixture_case("first", "test", None, Container::RawCodestream);
    let second = fixture_case("second", "test", None, Container::RawCodestream);
    let mixed = MixedFixtureBatch {
        name: "rotation".to_string(),
        cases: vec![first, second],
        format: PixelFormat::Rgb8,
        operation_class: OperationClass::Full,
    };
    assert_eq!(super::decode::mixed_case_at(&mixed, 0).name, "first");
    assert_eq!(super::decode::mixed_case_at(&mixed, 3).name, "second");
}

#[test]
fn decode_routing_helpers_fail_before_external_processes_for_unsupported_shapes() {
    let mut case = fixture_case("unsupported", "test", None, Container::RawCodestream);
    case.format = PixelFormat::Rgba8;
    let error = super::decode::decode_external_once(
        BenchmarkMode::Capability,
        &case,
        DecoderKind::OpenJpeg,
        &case.bytes,
    )
    .expect_err("unsupported output format");
    assert!(error.contains("does not support Rgba8"));

    assert_eq!(
        super::decode::decode_method_label(BenchmarkMode::Capability, DecoderKind::OpenJph, &case,),
        "openjph-cli-process-output-pnm"
    );
    assert_eq!(
        super::decode::decode_method_label(BenchmarkMode::Capability, DecoderKind::Kakadu, &case,),
        "kakadu-cli-process-output-pnm"
    );
    assert_eq!(
        super::decode::decode_method_label(BenchmarkMode::Capability, DecoderKind::OpenJpeg, &case,),
        "native"
    );
    assert!(super::decode::decode_external_region_scaled_emulated_once(
        &case,
        DecoderKind::OpenJpeg,
        &case.bytes,
    )
    .unwrap_err()
    .contains("non-ROI+scaled"));
}

fn fixture_case(
    name: &str,
    input_source: &str,
    source_fnv1a64: Option<&str>,
    container: Container,
) -> FixtureCase {
    FixtureCase {
        name: name.to_string(),
        input_source: input_source.to_string(),
        corpus_category: "natural-image".to_string(),
        corpus_name: "unit-corpus".to_string(),
        license_status: "cc0".to_string(),
        encode_command: "unit-fixture".to_string(),
        manifest_status: "covered".to_string(),
        source_fnv1a64: source_fnv1a64.map(str::to_string),
        codec: Codec::Classic,
        container,
        bytes: name.as_bytes().to_vec(),
        dimensions: (128, 128),
        format: PixelFormat::Rgb8,
        operation: Operation::Full,
    }
}
