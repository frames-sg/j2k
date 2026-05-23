// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::corpus_validation::{
    load_external_wsi_fixtures, validate_transcode_corpus, CorpusFixture, CorpusValidationError,
    CorpusValidationOptions,
};
use signinum_transcode::TranscodeValidationClassification;
use std::{fs, path::PathBuf};

#[test]
fn committed_conformance_fixtures_produce_error_report() {
    let fixtures = conformance_fixtures();

    let report = validate_transcode_corpus(&fixtures, &CorpusValidationOptions::default())
        .expect("validate committed transcode corpus");

    assert_eq!(report.fixture_count, fixtures.len());
    assert!(report.sample_count >= 64);
    assert_eq!(report.exact_match_count, report.sample_count);
    assert_eq!(report.max_abs_error, 0);
    assert_eq!(
        report.classification,
        TranscodeValidationClassification::Exact
    );
    assert_eq!(
        report.histogram_buckets.get(&0).copied(),
        Some(report.sample_count)
    );
    assert!(report
        .fixtures
        .iter()
        .all(|fixture| fixture.classification == TranscodeValidationClassification::Exact));
}

#[test]
fn external_wsi_loader_discovers_limited_jpeg_inputs() {
    let root = unique_temp_dir("signinum-transcode-corpus");
    let nested = root.join("nested");
    fs::create_dir_all(&nested).expect("create temp corpus");
    fs::write(root.join("ignore.txt"), b"not a jpeg").expect("write ignored file");
    fs::write(root.join("a.jpg"), conformance_fixtures()[0].bytes).expect("write first jpeg");
    fs::write(nested.join("b.jpeg"), conformance_fixtures()[1].bytes).expect("write nested jpeg");

    let options = CorpusValidationOptions {
        external_wsi_roots: vec![root.clone()],
        external_tile_limit: 1,
        ..CorpusValidationOptions::default()
    };

    let fixtures = load_external_wsi_fixtures(&options).expect("load external WSI fixtures");

    assert_eq!(fixtures.len(), 1);
    assert!(fixtures[0].name.ends_with("a.jpg"));
    let report = validate_transcode_corpus(&[fixtures[0].as_fixture()], &options)
        .expect("validate loaded external fixture");
    assert_eq!(report.fixture_count, 1);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn external_wsi_loader_can_require_configured_roots() {
    let options = CorpusValidationOptions {
        require_external_wsi: true,
        ..CorpusValidationOptions::default()
    };

    let err = load_external_wsi_fixtures(&options).expect_err("missing roots should fail");

    assert!(matches!(
        err,
        CorpusValidationError::MissingRequiredExternalCorpus(_)
    ));
}

fn conformance_fixtures() -> Vec<CorpusFixture<'static>> {
    vec![
        CorpusFixture {
            name: "grayscale_8x8",
            bytes: include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg"),
        },
        CorpusFixture {
            name: "baseline_444_8x8",
            bytes: include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_444_8x8.jpg"),
        },
        CorpusFixture {
            name: "baseline_422_16x8",
            bytes: include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_422_16x8.jpg"),
        },
        CorpusFixture {
            name: "baseline_420_16x16",
            bytes: include_bytes!(
                "../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg"
            ),
        },
    ]
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{unique}", std::process::id()))
}
