// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use j2k_t803::{EncoderIcs, EncoderIut, EncoderMarker, EncoderMatrix};

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

fn read(path: &str) -> String {
    fs::read_to_string(repo_root().join(path)).expect("read committed encoder evidence source")
}

#[test]
fn committed_matrix_is_pinned_and_each_ics_matches_its_inventory() {
    let matrix = EncoderMatrix::parse(&read("corpus/j2k-conformance/encoder-matrix-v1.toml"))
        .expect("valid committed encoder matrix");

    for (iut, path) in [
        (
            EncoderIut::Cpu,
            "corpus/j2k-conformance/encoder-ics-cpu.toml",
        ),
        (
            EncoderIut::Cuda,
            "corpus/j2k-conformance/encoder-ics-cuda.toml",
        ),
        (
            EncoderIut::Metal,
            "corpus/j2k-conformance/encoder-ics-metal.toml",
        ),
    ] {
        let ics = EncoderIcs::parse(&read(path)).expect("valid committed Annex F ICS");
        ics.validate_against(&matrix)
            .expect("ICS must pin its exact matrix inventory");
        assert_eq!(ics.iut, iut);
    }
}

#[test]
fn matrix_covers_every_exposed_part1_marker_and_declared_pair() {
    let matrix = EncoderMatrix::parse(&read("corpus/j2k-conformance/encoder-matrix-v1.toml"))
        .expect("valid committed encoder matrix");

    for marker in [
        EncoderMarker::Rgn,
        EncoderMarker::Tlm,
        EncoderMarker::Plm,
        EncoderMarker::Plt,
        EncoderMarker::Ppm,
        EncoderMarker::Ppt,
        EncoderMarker::Sop,
        EncoderMarker::Eph,
    ] {
        assert!(
            matrix
                .cases
                .iter()
                .any(|case| case.markers.contains(&marker)),
            "matrix does not exercise {marker:?}"
        );
    }
}

#[test]
fn matrix_rejects_case_tampering_before_execution() {
    let text = read("corpus/j2k-conformance/encoder-matrix-v1.toml");
    let tampered = text.replacen(
        "pattern = \"checkerboard\"",
        "pattern = \"deterministic-noise\"",
        1,
    );

    let error = EncoderMatrix::parse(&tampered).expect_err("case hash mismatch must fail");

    assert!(error.to_string().contains("SHA-256"));
}
