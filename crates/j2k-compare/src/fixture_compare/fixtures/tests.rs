// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::PathBuf};

use j2k_core::{Downscale, PixelFormat, Rect};

use super::{
    case_from_bytes, codec_from_bytes, collect_j2k_paths, container_from_bytes,
    container_from_path_and_bytes, encode_gray, encode_lossless, external_scaled_roi,
    generated_metadata, is_j2k_path, load_external_fixture_cases, mixed_external_batches,
    pixel_format, should_emit_external_region_scaled, Codec, Container, FixtureMetadata, Operation,
};

#[test]
fn fixture_classification_covers_supported_extensions_containers_and_codecs() {
    for path in [
        "image.j2k",
        "image.J2C",
        "image.jp2",
        "image.JPH",
        "image.jhc",
    ] {
        assert!(is_j2k_path(std::path::Path::new(path)), "{path}");
    }
    assert!(!is_j2k_path(std::path::Path::new("image.png")));
    assert!(!is_j2k_path(std::path::Path::new("image")));

    let jp2_signature = [0, 0, 0, 12, b'j', b'P', b' ', b' ', 0];
    assert_eq!(container_from_bytes(&jp2_signature), Container::Jp2);
    assert_eq!(
        container_from_bytes(b"raw codestream"),
        Container::RawCodestream
    );
    assert_eq!(
        container_from_path_and_bytes(std::path::Path::new("image.JPH"), b"raw"),
        Container::Jph
    );
    assert_eq!(
        container_from_path_and_bytes(std::path::Path::new("image.jhc"), &jp2_signature),
        Container::Jhc
    );
    assert_eq!(
        container_from_path_and_bytes(std::path::Path::new("image.bin"), &jp2_signature),
        Container::Jp2
    );

    assert_eq!(pixel_format(1, 8), Some(PixelFormat::Gray8));
    assert_eq!(pixel_format(3, 8), Some(PixelFormat::Rgb8));
    assert_eq!(pixel_format(2, 8), None);
    assert_eq!(pixel_format(1, 12), None);

    let classic = encode_gray(8, 8, Codec::Classic).expect("encode classic fixture");
    let htj2k = encode_gray(8, 8, Codec::Htj2k).expect("encode HTJ2K fixture");
    assert_eq!(codec_from_bytes(&classic), Codec::Classic);
    assert_eq!(codec_from_bytes(&htj2k), Codec::Htj2k);
    assert_eq!(codec_from_bytes(b"not a codestream"), Codec::Unknown);
}

#[test]
fn case_materialization_validates_codestream_shape_and_roi() {
    let metadata = generated_metadata("unit-generated");
    let error = result_error(
        case_from_bytes(
            "invalid",
            metadata.clone(),
            Codec::Classic,
            Container::RawCodestream,
            b"not a codestream".to_vec(),
            Operation::Full,
        ),
        "invalid codestream",
    );
    assert!(error.contains("invalid: inspect:"));

    let two_component = encode_lossless(&[0, 1, 2, 3, 4, 5, 6, 7], 2, 2, 2, Codec::Classic)
        .expect("encode two-component fixture");
    let error = result_error(
        case_from_bytes(
            "two-component",
            metadata.clone(),
            Codec::Classic,
            Container::RawCodestream,
            two_component,
            Operation::Full,
        ),
        "unsupported output shape",
    );
    assert!(error.contains("unsupported output shape for benchmark"));

    let bytes = encode_gray(64, 64, Codec::Classic).expect("encode gray fixture");
    let error = result_error(
        case_from_bytes(
            "outside-roi",
            metadata,
            Codec::Classic,
            Container::RawCodestream,
            bytes,
            Operation::Region(Rect {
                x: 32,
                y: 32,
                w: 64,
                h: 64,
            }),
        ),
        "out-of-bounds ROI",
    );
    assert!(error.contains("outside-roi: ROI"));
    assert!(error.contains("exceeds"));
}

#[test]
fn external_loader_reports_directory_and_fixture_failures() {
    let root = test_root("loader-errors");
    let missing = root.join("missing");
    let error = result_error(
        load_external_fixture_cases(&missing, None),
        "missing directory",
    );
    assert!(error.contains("is not a directory"));

    let empty = root.join("empty");
    fs::create_dir_all(&empty).expect("create empty fixture directory");
    let error = result_error(
        load_external_fixture_cases(&empty, None),
        "empty fixture directory",
    );
    assert!(error.contains("contains no .j2k/.j2c/.jp2/.jph/.jhc fixtures"));

    fs::write(empty.join("invalid.j2k"), b"not a codestream").expect("write invalid fixture");
    let error = result_error(load_external_fixture_cases(&empty, None), "invalid fixture");
    assert!(error.contains("inspect external fixture"));

    let unsupported = root.join("unsupported");
    fs::create_dir_all(&unsupported).expect("create unsupported fixture directory");
    let two_component = encode_lossless(&[0, 1, 2, 3, 4, 5, 6, 7], 2, 2, 2, Codec::Classic)
        .expect("encode two-component fixture");
    fs::write(unsupported.join("two-component.j2k"), two_component)
        .expect("write unsupported fixture");
    let error = result_error(
        load_external_fixture_cases(&unsupported, None),
        "unsupported fixture",
    );
    assert!(error.contains("unsupported benchmark shape: components=2 bit_depth=8"));

    #[cfg(unix)]
    {
        let unreadable = root.join("unreadable");
        fs::create_dir_all(&unreadable).expect("create unreadable fixture directory");
        std::os::unix::fs::symlink(
            unreadable.join("missing-target"),
            unreadable.join("broken.j2k"),
        )
        .expect("create broken fixture symlink");
        let error = result_error(
            load_external_fixture_cases(&unreadable, None),
            "unreadable fixture",
        );
        assert!(error.contains("read "));
        assert!(error.contains("broken.j2k"));
    }

    let error = collect_j2k_paths(&missing, &mut Vec::new()).expect_err("missing scan root fails");
    assert!(error.contains("read external input dir"));
}

#[test]
fn external_loader_sorts_cases_and_applies_region_scaled_corpus_policy() {
    let root = test_root("loader-success").join("kodak");
    fs::create_dir_all(&root).expect("create fixture directory");
    fs::write(
        root.join("b-small.j2k"),
        encode_gray(64, 64, Codec::Classic).expect("encode small fixture"),
    )
    .expect("write small fixture");
    fs::write(
        root.join("a-large.j2k"),
        encode_gray(128, 128, Codec::Classic).expect("encode large fixture"),
    )
    .expect("write large fixture");

    let cases = load_external_fixture_cases(&root, None).expect("load external fixtures");
    assert_eq!(
        cases
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>(),
        [
            "external_a-large_full",
            "external_a-large_roi64_q4",
            "external_b-small_full",
        ]
    );
    assert_eq!(cases[0].corpus_category, "natural-image");
    assert_eq!(cases[0].format, PixelFormat::Gray8);
    assert_eq!(cases[0].codec, Codec::Classic);
    assert_eq!(cases[0].container, Container::RawCodestream);
    assert!(matches!(
        cases[1].operation,
        Operation::RegionScaled {
            scale: Downscale::Quarter,
            ..
        }
    ));
    assert_eq!(cases[2].dimensions, (64, 64));

    assert_eq!(
        external_scaled_roi((200, 128)),
        Rect {
            x: 68,
            y: 32,
            w: 64,
            h: 64,
        }
    );
    assert!(should_emit_external_region_scaled(&metadata(
        "natural-image"
    )));
    assert!(should_emit_external_region_scaled(&metadata(
        "medical-domain"
    )));
    assert!(should_emit_external_region_scaled(&metadata(
        "remote-sensing"
    )));
    assert!(!should_emit_external_region_scaled(&metadata("interop")));
}

#[test]
fn mixed_external_batches_group_compatible_distinct_inputs_only() {
    let bytes = encode_gray(64, 64, Codec::Classic).expect("encode mixed fixture");
    let base = case_from_bytes(
        "first",
        metadata("natural-image"),
        Codec::Classic,
        Container::RawCodestream,
        bytes.clone(),
        Operation::Full,
    )
    .expect("materialize first case");

    let mut duplicate = base.clone();
    duplicate.name = "duplicate".to_string();
    let mut distinct = base.clone();
    distinct.name = "distinct".to_string();
    distinct.bytes = encode_gray(68, 64, Codec::Classic).expect("encode distinct fixture");
    distinct.dimensions = (68, 64);
    let mut region = distinct.clone();
    region.name = "region".to_string();
    region.operation = Operation::Region(Rect {
        x: 0,
        y: 0,
        w: 32,
        h: 32,
    });
    let mut generated = distinct.clone();
    generated.name = "generated".to_string();
    generated.input_source = "j2k-generated".to_string();

    let groups = mixed_external_batches(&[base, duplicate, distinct, region, generated]);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].name, "external_mixed_gray8_full");
    assert_eq!(groups[0].format, PixelFormat::Gray8);
    assert_eq!(groups[0].cases.len(), 3);
    assert!(groups[0]
        .cases
        .iter()
        .all(|case| case.input_source.starts_with("external:")));
}

#[test]
fn generated_encoder_rejects_invalid_samples_and_unknown_codec() {
    let error =
        encode_lossless(&[], 1, 1, 1, Codec::Classic).expect_err("invalid sample length must fail");
    assert!(!error.is_empty());

    let error = encode_lossless(&[0], 1, 1, 1, Codec::Unknown)
        .expect_err("unknown generated codec must fail");
    assert_eq!(error, "cannot encode generated fixture for unknown codec");
}

fn test_root(label: &str) -> PathBuf {
    let root = std::env::current_dir()
        .expect("current directory")
        .join("target")
        .join("j2k-fixture-loader-tests")
        .join(format!("{label}-{}", std::process::id()));
    fs::create_dir_all(&root).expect("create fixture test root");
    root
}

fn metadata(corpus_category: &str) -> FixtureMetadata {
    FixtureMetadata {
        input_source: "external:unit".to_string(),
        corpus_category: corpus_category.to_string(),
        corpus_name: "unit-corpus".to_string(),
        license_status: "generated-test".to_string(),
        encode_command: "unit-encoder".to_string(),
        manifest_status: "covered".to_string(),
        source_fnv1a64: None,
    }
}

fn result_error<T>(result: Result<T, String>, context: &str) -> String {
    match result {
        Ok(_) => panic!("{context} unexpectedly succeeded"),
        Err(error) => error,
    }
}
