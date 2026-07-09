// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    canonicalize_manifest_row_path, collect_j2k_paths, container_from_path_and_bytes,
    external_manifest_covered_case_count, external_manifest_missing_case_count,
    mixed_external_group_distinct_inputs_label, publication_blockers, skipped_comparators_label,
    unique_input_count, BenchmarkMode, Codec, Container, FixtureCase, MixedFixtureBatch, Operation,
    OperationClass, DEFAULT_CASE_BATCH_SIZES, DEFAULT_MIXED_BATCH_SIZES,
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

    assert_eq!(external_manifest_covered_case_count(&[covered.clone()]), 1);
    assert_eq!(external_manifest_missing_case_count(&[missing.clone()]), 1);

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
        mixed_external_group_distinct_inputs_label(&[mixed]),
        "external_mixed_gray8_full:2"
    );

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
