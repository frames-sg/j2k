// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    env, fs,
    path::{Component, Path, PathBuf},
};

use j2k_native::{DecodeSettings, Image};

const CONFORMANCE_ENV: &str = "J2K_ISO_CONFORMANCE_DIR";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    Blocking,
    KnownLimitation,
    Investigate,
    OutOfScope,
}

#[derive(Debug)]
struct Vector {
    id: String,
    path: PathBuf,
    classification: Classification,
    features: String,
    reason: String,
}

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

fn manifest_path() -> PathBuf {
    repo_root().join("corpus/j2k-conformance/manifest.tsv")
}

fn load_manifest() -> Vec<Vector> {
    let text = fs::read_to_string(manifest_path()).expect("read J2K conformance manifest");
    text.lines()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            Some(parse_manifest_line(line_idx + 1, line))
        })
        .collect()
}

fn parse_manifest_line(line_number: usize, line: &str) -> Vector {
    let fields: Vec<_> = line.split('\t').collect();
    assert_eq!(
        fields.len(),
        5,
        "manifest line {line_number} must contain id, path, classification, features, reason"
    );
    let path = PathBuf::from(fields[1]);
    assert!(
        !path.is_absolute()
            && !path
                .components()
                .any(|component| matches!(component, Component::ParentDir)),
        "manifest line {line_number} path must stay relative to the ISO vector root"
    );
    Vector {
        id: fields[0].to_string(),
        path,
        classification: parse_classification(line_number, fields[2]),
        features: fields[3].to_string(),
        reason: fields[4].to_string(),
    }
}

fn parse_classification(line_number: usize, value: &str) -> Classification {
    match value {
        "blocking" => Classification::Blocking,
        "known-limitation" => Classification::KnownLimitation,
        "investigate" => Classification::Investigate,
        "out-of-scope" => Classification::OutOfScope,
        _ => panic!("manifest line {line_number} has invalid classification {value:?}"),
    }
}

#[test]
fn iso_conformance_manifest_is_release_classified() {
    let vectors = load_manifest();
    assert!(
        vectors
            .iter()
            .any(|vector| vector.classification == Classification::Blocking),
        "manifest must contain at least one blocking vector"
    );
    for vector in vectors {
        assert!(!vector.id.is_empty(), "vector id must not be empty");
        assert!(
            !vector.features.is_empty(),
            "{} must list exercised features",
            vector.id
        );
        if matches!(
            vector.classification,
            Classification::KnownLimitation | Classification::OutOfScope
        ) {
            assert!(
                !vector.reason.is_empty(),
                "{} non-blocking row must document the deferred feature",
                vector.id
            );
        }
        assert_ne!(
            vector.classification,
            Classification::Investigate,
            "{} must be classified before release signoff",
            vector.id
        );
    }
}

#[test]
fn iso_conformance_manifest_blocks_release_shipped_features() {
    let vectors = load_manifest();
    for required_feature in [
        "part1-core",
        "part15-core",
        "poc",
        "precincts",
        "progression-orders",
        "tlm",
        "plt",
        "sop",
        "eph",
    ] {
        assert!(
            vectors.iter().any(|vector| {
                vector.classification == Classification::Blocking
                    && vector
                        .features
                        .split(';')
                        .any(|feature| feature == required_feature)
            }),
            "manifest must include a blocking vector for shipped feature {required_feature}"
        );
    }
    assert!(
        vectors.iter().any(|vector| {
            vector.features.split(';').any(|feature| feature == "plm")
                && (vector.classification == Classification::Blocking
                    || (vector.classification == Classification::KnownLimitation
                        && vector
                            .features
                            .split(';')
                            .any(|feature| feature == "conformance-coverage-gap")))
        }),
        "manifest must include a blocking PLM vector or document the ISO coverage gap"
    );
}

#[test]
fn iso_conformance_flags_missing_blocking_vectors_as_signoff_failures() {
    let vector_root = env::temp_dir().join(format!("j2k-missing-blocking-{}", std::process::id()));
    fs::create_dir_all(&vector_root).expect("create temporary vector root");
    let vectors = vec![Vector {
        id: "missing_blocking".to_string(),
        path: PathBuf::from("part1/missing.j2k"),
        classification: Classification::Blocking,
        features: "part1-core".to_string(),
        reason: "blocking vector required".to_string(),
    }];

    let missing = missing_blocking_vectors(&vectors, &vector_root);

    fs::remove_dir_all(&vector_root).expect("remove temporary vector root");
    assert_eq!(missing, vec!["missing_blocking".to_string()]);
}

fn missing_blocking_vectors(vectors: &[Vector], vector_root: &Path) -> Vec<String> {
    vectors
        .iter()
        .filter(|vector| vector.classification == Classification::Blocking)
        .filter(|vector| !vector_root.join(&vector.path).exists())
        .map(|vector| vector.id.clone())
        .collect()
}

#[test]
fn env_gated_iso_conformance_blocks_only_shipped_features() {
    let Some(vector_root) = env::var_os(CONFORMANCE_ENV).map(PathBuf::from) else {
        return;
    };
    let vectors = load_manifest();
    let missing_blocking = missing_blocking_vectors(&vectors, &vector_root);
    assert!(
        missing_blocking.is_empty(),
        "blocking ISO conformance vectors are missing from {}: {}",
        vector_root.display(),
        missing_blocking.join(", ")
    );

    for vector in vectors {
        match vector.classification {
            Classification::Investigate => {
                panic!("{} is still investigate at release signoff", vector.id);
            }
            Classification::KnownLimitation => {
                eprintln!(
                    "known limitation {}: {} ({})",
                    vector.id, vector.reason, vector.features
                );
            }
            Classification::OutOfScope => {
                eprintln!(
                    "out of scope {}: {} ({})",
                    vector.id, vector.reason, vector.features
                );
            }
            Classification::Blocking => {
                let path = vector_root.join(&vector.path);
                let bytes = fs::read(&path)
                    .unwrap_or_else(|err| panic!("read blocking vector {}: {err}", vector.id));
                let image = Image::new(&bytes, &DecodeSettings::default())
                    .unwrap_or_else(|err| panic!("parse blocking vector {}: {err}", vector.id));
                image
                    .decode_native()
                    .unwrap_or_else(|err| panic!("decode blocking vector {}: {err}", vector.id));
            }
        }
    }
}
