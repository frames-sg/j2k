// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path};

use j2k_t803::{
    EncoderBlockCoding, EncoderIcs, EncoderIut, EncoderMarker, EncoderMatrix, EncoderOperation,
    EncoderPayload,
};

const MATRIX_PATH: &str = "corpus/j2k-conformance/encoder-matrix-v2.toml";

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

#[test]
fn matrix_covers_cpu_coefficient_recode_to_raw_htj2k_and_jph() {
    let matrix = EncoderMatrix::parse(&read(MATRIX_PATH)).expect("valid committed encoder matrix");
    let recode_cases = matrix
        .cases
        .iter()
        .filter(|case| case.operation == EncoderOperation::Recode)
        .collect::<Vec<_>>();

    assert_eq!(recode_cases.len(), 2);
    assert!(recode_cases
        .iter()
        .all(|case| case.iuts == [EncoderIut::Cpu]));
    assert!(recode_cases
        .iter()
        .any(|case| case.payload == EncoderPayload::Codestream));
    assert!(recode_cases
        .iter()
        .any(|case| case.payload == EncoderPayload::Jph));
    assert!(recode_cases
        .iter()
        .any(|case| case.source_payload == Some(EncoderPayload::Codestream)));
    assert!(recode_cases
        .iter()
        .any(|case| case.source_payload == Some(EncoderPayload::Jp2)));
}

fn read(path: &str) -> String {
    fs::read_to_string(repo_root().join(path)).expect("read committed encoder evidence source")
}

#[test]
fn committed_matrix_is_pinned_and_each_ics_matches_its_inventory() {
    let matrix = EncoderMatrix::parse(&read(MATRIX_PATH)).expect("valid committed encoder matrix");

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
    let matrix = EncoderMatrix::parse(&read(MATRIX_PATH)).expect("valid committed encoder matrix");

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
fn matrix_covers_classic_ht_codestream_and_jph_outputs() {
    let matrix = EncoderMatrix::parse(&read(MATRIX_PATH)).expect("valid committed encoder matrix");

    assert_eq!(matrix.schema_version, 2);
    assert!(matrix.cases.iter().any(|case| {
        case.block_coding == EncoderBlockCoding::Classic
            && case.payload == EncoderPayload::Codestream
    }));
    assert!(matrix.cases.iter().any(|case| {
        case.block_coding == EncoderBlockCoding::HighThroughput
            && case.payload == EncoderPayload::Codestream
    }));
    assert!(matrix.cases.iter().any(|case| {
        case.block_coding == EncoderBlockCoding::HighThroughput
            && case.payload == EncoderPayload::Jph
    }));
    assert!(matrix.cases.iter().all(|case| {
        case.payload != EncoderPayload::Jph
            || (case.block_coding == EncoderBlockCoding::HighThroughput
                && case.markers.contains(&EncoderMarker::Cap))
    }));
}

#[test]
fn matrix_covers_ht_boundaries_beyond_the_pairwise_rows() {
    let matrix = EncoderMatrix::parse(&read(MATRIX_PATH)).expect("valid committed encoder matrix");
    let ht_encode = |case: &&j2k_t803::EncoderCase| {
        case.operation == EncoderOperation::Encode
            && case.block_coding == EncoderBlockCoding::HighThroughput
    };

    for bit_depth in [1, 16, 30] {
        assert!(
            matrix
                .cases
                .iter()
                .filter(ht_encode)
                .any(|case| case.bit_depth == bit_depth),
            "HT matrix does not cover {bit_depth}-bit input"
        );
    }
    assert!(matrix.cases.iter().any(|case| {
        case.operation == EncoderOperation::Encode
            && case.block_coding == EncoderBlockCoding::Classic
            && case.bit_depth == 31
    }));
    for components in [2, 4, 5] {
        assert!(
            matrix
                .cases
                .iter()
                .filter(ht_encode)
                .any(|case| case.components == components),
            "HT matrix does not cover {components} components"
        );
    }
    for levels in [0, 5] {
        assert!(
            matrix
                .cases
                .iter()
                .filter(ht_encode)
                .any(|case| case.decomposition_levels == levels),
            "HT matrix does not cover decomposition boundary {levels}"
        );
    }
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
                .filter(ht_encode)
                .any(|case| case.markers.contains(&marker)),
            "HT matrix does not exercise {marker:?}"
        );
    }
    for id in [
        "part15-layers-lossless-3",
        "part15-layers-lossy-3",
        "part15-planar-sampled",
        "part15-planar-typed",
        "part15-precinct-lossy",
        "part15-roi-lossless",
        "part15-tile-multitile",
    ] {
        let case = matrix
            .cases
            .iter()
            .find(|case| case.id == id)
            .unwrap_or_else(|| panic!("HT matrix case {id} is missing"));
        assert!(ht_encode(&case));
        assert!(case.iuts.contains(&EncoderIut::Cpu));
    }

    let roi = matrix
        .cases
        .iter()
        .find(|case| case.id == "part15-roi-lossless")
        .expect("HT ROI matrix case");
    assert_eq!(
        roi.reference_decoder,
        j2k_t803::EncoderReferenceDecoder::OpenHtj2k
    );
    assert!(matrix
        .cases
        .iter()
        .filter(|case| case.id != roi.id)
        .all(|case| case.reference_decoder == j2k_t803::EncoderReferenceDecoder::OpenJpeg));
}

#[test]
fn matrix_rejects_case_tampering_before_execution() {
    let text = read(MATRIX_PATH);
    let tampered = text.replacen(
        "pattern = \"checkerboard\"",
        "pattern = \"deterministic-noise\"",
        1,
    );

    let error = EncoderMatrix::parse(&tampered).expect_err("case hash mismatch must fail");

    assert!(error.to_string().contains("SHA-256"));
}

#[test]
fn matrix_uses_one_fixed_pairwise_contract() {
    let text = read(MATRIX_PATH);
    assert!(!text.contains("[[pairwise_scopes]]"));

    let without_part1_membership = text.replace("pairwise_scope = \"part1\"", "");
    let error = EncoderMatrix::parse(&without_part1_membership)
        .expect_err("the fixed Part 1 pairwise contract must remain mandatory");

    assert!(error.to_string().contains("Part1 pairwise rows"));
}
