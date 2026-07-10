// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, fs, path::Path, process::Command};

use super::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn conformance_manifest_hashes_and_generator_cover_committed_fixtures() {
    let root = repo_root();
    let conformance = root.join("corpus/conformance");
    let support_conformance = root.join("crates/j2k-test-support/fixtures/conformance");
    let manifest_text =
        fs::read_to_string(conformance.join("manifest.json")).expect("read conformance manifest");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_text).expect("parse conformance manifest");
    let generator =
        fs::read_to_string(conformance.join("generate.sh")).expect("read conformance generator");
    let fixtures = manifest["fixtures"]
        .as_array()
        .expect("conformance manifest fixtures array");
    let mut listed_files = BTreeSet::new();

    assert!(
        manifest["libjpeg_turbo_version"].as_str().is_some(),
        "conformance manifest must record the libjpeg-turbo generator version"
    );

    for fixture in fixtures {
        for (path_key, hash_key) in [("input", "input_sha256"), ("reference", "reference_sha256")] {
            let filename = fixture[path_key]
                .as_str()
                .unwrap_or_else(|| panic!("fixture missing `{path_key}`"));
            let expected_hash = fixture[hash_key]
                .as_str()
                .unwrap_or_else(|| panic!("{filename} missing `{hash_key}`"));
            assert_eq!(
                expected_hash.len(),
                64,
                "{filename} must record a SHA-256 hash"
            );

            let corpus_path = conformance.join(filename);
            assert!(
                corpus_path.exists(),
                "conformance manifest lists missing file {filename}"
            );
            assert_eq!(
                sha256_hex(&corpus_path),
                expected_hash,
                "conformance manifest hash is stale for {filename}"
            );

            let support_path = support_conformance.join(filename);
            assert!(
                support_path.exists(),
                "j2k-test-support conformance copy is missing {filename}"
            );
            assert_eq!(
                sha256_hex(&support_path),
                expected_hash,
                "j2k-test-support conformance copy drifted from corpus/{filename}"
            );

            assert!(
                generator.contains(filename),
                "conformance generator does not regenerate manifest file {filename}"
            );
            listed_files.insert(filename.to_string());
        }
    }

    for dir in [&conformance, &support_conformance] {
        assert_committed_conformance_files_are_listed(dir, &listed_files);
    }
}

#[test]
fn openhtj2k_fixture_dirs_have_license_notice_coverage() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-native/fixtures/htj2k/README.md").required(&[
                "OpenHTJ2K",
                "ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f",
                "LICENSE.OpenHTJ2K",
            ]),
            FilePatternCheck::new("crates/j2k-native/fixtures/htj2k/LICENSE.OpenHTJ2K")
                .required(&["BSD 3-Clause License", "Osamu Watanabe"]),
            FilePatternCheck::new("crates/j2k-test-support/fixtures/htj2k/README.md").required(&[
                "OpenHTJ2K",
                "ffe5acf9f1eedb87c36c3fd2134fdc1ddea5e75f",
                "LICENSE.OpenHTJ2K",
            ]),
            FilePatternCheck::new("crates/j2k-test-support/fixtures/htj2k/LICENSE.OpenHTJ2K")
                .required(&["BSD 3-Clause License", "Osamu Watanabe"]),
            FilePatternCheck::new("NOTICES.md").required(&[
                "crates/j2k-native/fixtures/htj2k",
                "crates/j2k-native/fixtures/htj2k/LICENSE.OpenHTJ2K",
                "crates/j2k-test-support/fixtures/htj2k",
                "crates/j2k-test-support/fixtures/htj2k/LICENSE.OpenHTJ2K",
            ]),
        ],
    );
}

fn assert_committed_conformance_files_are_listed(dir: &Path, listed_files: &BTreeSet<String>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read {}: {err}", dir.display())) {
        let entry = entry.expect("read conformance entry");
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !matches!(ext, "jpg" | "rgb" | "gray") {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("utf-8 fixture filename");
        assert!(
            listed_files.contains(filename),
            "conformance fixture {filename} is missing from manifest.json"
        );
    }
}

fn sha256_hex(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .or_else(|_| {
            Command::new("shasum")
                .args(["-a", "256"])
                .arg(path)
                .output()
        })
        .unwrap_or_else(|err| panic!("hash {}: {err}", path.display()));
    assert!(
        output.status.success(),
        "hash command failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("missing hash output for {}", path.display()))
        .to_string()
}

#[test]
fn corpus_readme_does_not_claim_committed_fixtures_are_absent() {
    assert_file_pattern_checks(
        repo_root(),
        &[FilePatternCheck::new("corpus/README.md").forbidden(&["intentionally empty"])],
    );
}
