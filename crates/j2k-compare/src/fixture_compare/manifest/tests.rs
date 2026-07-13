// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::HashMap, fs, path::PathBuf};

use super::{
    external_fixture_metadata, fixture_manifest_from_path, parse_manifest_codec,
    parse_manifest_container, Codec, Container, FixtureManifest, ManifestEntry,
};
use j2k_test_support::fnv1a64_hex;

#[test]
fn manifest_parser_skips_comments_and_applies_optional_defaults() {
    let root = test_root("defaults");
    let fixture = root.join("sample.j2k");
    fs::write(&fixture, b"fixture").expect("write fixture");
    let manifest_path = root.join("fixtures.tsv");
    write_manifest(
        &manifest_path,
        "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n\
         # generated test row follows\n\
         sample.j2k\t natural-image \t\t\t\t\t\t\t\n",
    );

    let manifest = fixture_manifest_from_path(&manifest_path, &[]).expect("parse fixture manifest");
    let canonical_fixture = fixture.canonicalize().expect("canonical fixture");
    let entry = manifest
        .entries
        .get(&canonical_fixture)
        .expect("manifest entry");

    assert_eq!(entry.corpus_category, "natural-image");
    assert_eq!(entry.corpus_name, "not-recorded");
    assert_eq!(entry.license_status, "not-recorded");
    assert_eq!(entry.encode_command, "not-recorded");
    assert_eq!(entry.input_fnv1a64, None);
    assert_eq!(entry.source_fnv1a64, None);
    assert_eq!(entry.codec, None);
    assert_eq!(entry.container, None);
}

#[test]
fn manifest_parser_reports_read_header_and_row_structure_errors() {
    let root = test_root("structure-errors");
    let fixture = root.join("sample.j2k");
    fs::write(&fixture, b"fixture").expect("write fixture");
    let manifest_path = root.join("fixtures.tsv");

    let missing_path = root.join("missing.tsv");
    let error = result_error(
        fixture_manifest_from_path(&missing_path, &[]),
        "missing manifest",
    );
    assert!(error.contains("read J2K_FIXTURE_COMPARE_MANIFEST"));

    for (text, expected) in [
        (" \n\t\n", "is empty"),
        (
            "corpus_category\n",
            "fixture manifest is missing required \"path\" column",
        ),
        (
            "path\n",
            "fixture manifest is missing required \"corpus_category\" column",
        ),
        (
            "path\tcorpus_category\nsample.j2k\n",
            "fixture manifest row 2 is missing \"corpus_category\" field",
        ),
        (
            "path\tcorpus_category\nsample.j2k\t  \n",
            "fixture manifest row 2 has empty required \"corpus_category\" field",
        ),
        (
            "path\tcorpus_category\tcorpus_name\nsample.j2k\tinterop\n",
            "fixture manifest row 2 is missing \"corpus_name\" field",
        ),
        (
            "path\tcorpus_category\nmissing.j2k\tinterop\n",
            "fixture manifest",
        ),
    ] {
        write_manifest(&manifest_path, text);
        let error = result_error(
            fixture_manifest_from_path(&manifest_path, &[]),
            "invalid manifest",
        );
        assert!(
            error.contains(expected),
            "expected {expected:?} in error: {error}"
        );
    }
}

#[test]
fn manifest_parser_rejects_duplicate_paths_and_invalid_type_pins() {
    let root = test_root("invalid-pins");
    let fixture = root.join("sample.j2k");
    fs::write(&fixture, b"fixture").expect("write fixture");
    let manifest_path = root.join("fixtures.tsv");

    for (text, expected) in [
        (
            "path\tcorpus_category\nsample.j2k\tinterop\nsample.j2k\tinterop\n",
            "row 3 duplicates path sample.j2k",
        ),
        (
            "path\tcorpus_category\tcodec\nsample.j2k\tinterop\tjpeg\n",
            "row 2 has invalid codec \"jpeg\"",
        ),
        (
            "path\tcorpus_category\tcontainer\nsample.j2k\tinterop\ttiff\n",
            "row 2 has invalid container \"tiff\"",
        ),
    ] {
        write_manifest(&manifest_path, text);
        let error = result_error(
            fixture_manifest_from_path(&manifest_path, &[]),
            "invalid manifest",
        );
        assert!(
            error.contains(expected),
            "expected {expected:?} in error: {error}"
        );
    }
}

#[test]
fn manifest_type_parsers_accept_all_documented_spellings() {
    for (value, expected) in [
        (None, None),
        (Some("j2k"), Some(Codec::Classic)),
        (Some("classic"), Some(Codec::Classic)),
        (Some("htj2k"), Some(Codec::Htj2k)),
        (Some("unknown"), Some(Codec::Unknown)),
    ] {
        assert_eq!(parse_manifest_codec(value, 7), Ok(expected));
    }

    for (value, expected) in [
        (None, None),
        (Some("raw-codestream"), Some(Container::RawCodestream)),
        (Some("j2k"), Some(Container::RawCodestream)),
        (Some("j2c"), Some(Container::RawCodestream)),
        (Some("jp2"), Some(Container::Jp2)),
        (Some("jph"), Some(Container::Jph)),
        (Some("jhc"), Some(Container::Jhc)),
    ] {
        assert_eq!(parse_manifest_container(value, 8), Ok(expected));
    }
}

#[test]
fn external_metadata_distinguishes_unlisted_unpinned_and_pinned_fixtures() {
    let root = test_root("metadata-status");
    let fixture = root.join("kodak").join("sample.j2k");
    fs::create_dir_all(fixture.parent().expect("fixture parent")).expect("create fixture parent");
    let bytes = b"fixture bytes";
    fs::write(&fixture, bytes).expect("write fixture");
    let canonical_fixture = fixture.canonicalize().expect("canonical fixture");

    let unlisted = external_fixture_metadata(
        &fixture,
        bytes,
        Codec::Classic,
        Container::RawCodestream,
        None,
    )
    .expect("infer unlisted metadata");
    assert_eq!(unlisted.corpus_category, "natural-image");
    assert_eq!(unlisted.corpus_name, "path-inferred");
    assert_eq!(unlisted.license_status, "not-recorded");
    assert_eq!(unlisted.encode_command, "not-recorded");
    assert_eq!(unlisted.manifest_status, "not-covered");
    assert_eq!(unlisted.source_fnv1a64, None);

    let empty_manifest = FixtureManifest {
        entries: HashMap::new(),
    };
    let absent = external_fixture_metadata(
        &fixture,
        bytes,
        Codec::Classic,
        Container::RawCodestream,
        Some(&empty_manifest),
    )
    .expect("infer absent-entry metadata");
    assert_eq!(absent.manifest_status, "not-covered");

    let unpinned_manifest = manifest_with(
        canonical_fixture.clone(),
        manifest_entry(None, None, None, Some("source-digest")),
    );
    let unpinned = external_fixture_metadata(
        &fixture,
        bytes,
        Codec::Classic,
        Container::RawCodestream,
        Some(&unpinned_manifest),
    )
    .expect("read unpinned metadata");
    assert_eq!(unpinned.corpus_category, "interop");
    assert_eq!(unpinned.corpus_name, "unit-corpus");
    assert_eq!(unpinned.license_status, "generated-test");
    assert_eq!(unpinned.encode_command, "unit-encoder");
    assert_eq!(unpinned.manifest_status, "covered-unpinned");
    assert_eq!(unpinned.source_fnv1a64.as_deref(), Some("source-digest"));

    let pinned_manifest = manifest_with(
        canonical_fixture,
        manifest_entry(
            Some(fnv1a64_hex(bytes)),
            Some(Codec::Classic),
            Some(Container::RawCodestream),
            Some("source-digest"),
        ),
    );
    let pinned = external_fixture_metadata(
        &fixture,
        bytes,
        Codec::Classic,
        Container::RawCodestream,
        Some(&pinned_manifest),
    )
    .expect("read pinned metadata");
    assert_eq!(pinned.manifest_status, "covered");
}

#[test]
fn external_metadata_rejects_hash_codec_and_container_mismatches() {
    let root = test_root("metadata-mismatches");
    let fixture = root.join("sample.j2k");
    let bytes = b"fixture bytes";
    fs::write(&fixture, bytes).expect("write fixture");
    let canonical_fixture = fixture.canonicalize().expect("canonical fixture");

    let hash_manifest = manifest_with(
        canonical_fixture.clone(),
        manifest_entry(
            Some("incorrect".to_string()),
            Some(Codec::Classic),
            Some(Container::RawCodestream),
            None,
        ),
    );
    let error = result_error(
        external_fixture_metadata(
            &fixture,
            bytes,
            Codec::Classic,
            Container::RawCodestream,
            Some(&hash_manifest),
        ),
        "hash mismatch",
    );
    assert!(error.contains("hash mismatch: manifest incorrect != actual"));

    let codec_manifest = manifest_with(
        canonical_fixture.clone(),
        manifest_entry(
            None,
            Some(Codec::Htj2k),
            Some(Container::RawCodestream),
            None,
        ),
    );
    let error = result_error(
        external_fixture_metadata(
            &fixture,
            bytes,
            Codec::Classic,
            Container::RawCodestream,
            Some(&codec_manifest),
        ),
        "codec mismatch",
    );
    assert!(error.contains("codec mismatch: manifest htj2k != detected j2k"));

    let container_manifest = manifest_with(
        canonical_fixture,
        manifest_entry(None, Some(Codec::Classic), Some(Container::Jp2), None),
    );
    let error = result_error(
        external_fixture_metadata(
            &fixture,
            bytes,
            Codec::Classic,
            Container::RawCodestream,
            Some(&container_manifest),
        ),
        "container mismatch",
    );
    assert!(error.contains("container mismatch: manifest jp2 != detected raw-codestream"));
}

#[test]
fn external_metadata_rejects_unrepresentable_and_missing_paths() {
    let error = result_error(
        external_fixture_metadata(
            std::path::Path::new("invalid\nfixture.j2k"),
            b"",
            Codec::Classic,
            Container::RawCodestream,
            None,
        ),
        "control character",
    );
    assert!(error.contains("path contains a control character"));

    let missing = test_root("metadata-missing").join("missing.j2k");
    let manifest = FixtureManifest {
        entries: HashMap::new(),
    };
    let error = result_error(
        external_fixture_metadata(
            &missing,
            b"",
            Codec::Classic,
            Container::RawCodestream,
            Some(&manifest),
        ),
        "missing path",
    );
    assert!(error.contains("canonicalize external fixture"));
}

fn test_root(label: &str) -> PathBuf {
    let root = std::env::current_dir()
        .expect("current directory")
        .join("target")
        .join("j2k-fixture-manifest-tests")
        .join(format!("{label}-{}", std::process::id()));
    fs::create_dir_all(&root).expect("create manifest test root");
    root
}

fn write_manifest(path: &std::path::Path, text: &str) {
    fs::write(path, text).expect("write fixture manifest");
}

fn manifest_with(path: PathBuf, entry: ManifestEntry) -> FixtureManifest {
    FixtureManifest {
        entries: HashMap::from([(path, entry)]),
    }
}

fn manifest_entry(
    input_fnv1a64: Option<String>,
    codec: Option<Codec>,
    container: Option<Container>,
    source_fnv1a64: Option<&str>,
) -> ManifestEntry {
    ManifestEntry {
        corpus_category: "interop".to_string(),
        corpus_name: "unit-corpus".to_string(),
        license_status: "generated-test".to_string(),
        encode_command: "unit-encoder".to_string(),
        input_fnv1a64,
        source_fnv1a64: source_fnv1a64.map(str::to_string),
        codec,
        container,
    }
}

fn result_error<T>(result: Result<T, String>, context: &str) -> String {
    match result {
        Ok(_) => panic!("{context} unexpectedly succeeded"),
        Err(error) => error,
    }
}
