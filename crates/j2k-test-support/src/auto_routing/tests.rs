// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use j2k_core::{CompressedPayloadKind, CompressedTransferSyntax};
use serde_json::json;
use sha2::{Digest, Sha256};

use super::{
    auto_routing_route_cell, load_auto_routing_manifest, load_auto_routing_pnm,
    validate_auto_routing_decode_identity, write_auto_routing_evidence, AutoRoutingBackend,
    AutoRoutingCodec, AutoRoutingContainer, AutoRoutingEvidence, AutoRoutingExecution,
    AutoRoutingOperation, AutoRoutingPlatform,
};

#[test]
fn manifest_loader_hashes_and_classifies_bounded_external_inputs() {
    let root = temp_dir("manifest");
    let decode = root.join("decode/sample.j2k");
    let encode = root.join("encode/sample.ppm");
    fs::create_dir_all(decode.parent().unwrap()).unwrap();
    fs::create_dir_all(encode.parent().unwrap()).unwrap();
    fs::write(&decode, b"decode bytes").unwrap();
    fs::write(&encode, b"P6\n1 1\n255\n\x01\x02\x03").unwrap();
    let manifest = root.join("manifest.json");
    let manifest_value = json!({
        "schema_version": 1,
        "corpus": "routing-fixture",
        "source_url": "https://example.invalid/routing-fixture",
        "cases": [
            {
                "id": "decode-case",
                "path": "decode/sample.j2k",
                "kind": "decode",
                "pixel_format": "rgb8",
                "sha256": sha256(b"decode bytes")
            },
            {
                "id": "encode-case",
                "path": "encode/sample.ppm",
                "kind": "encode",
                "pixel_format": "rgb8",
                "sha256": sha256(b"P6\n1 1\n255\n\x01\x02\x03")
            }
        ]
    });
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&manifest_value).unwrap(),
    )
    .unwrap();

    let loaded = load_auto_routing_manifest(&manifest, &root).unwrap();

    assert_eq!(loaded.manifest.corpus, "routing-fixture");
    assert_eq!(loaded.workloads.len(), 2);
    assert_eq!(loaded.workloads[0].bytes, b"decode bytes");
    assert_eq!(loaded.workloads[1].bytes, b"P6\n1 1\n255\n\x01\x02\x03");
    assert_eq!(
        loaded.manifest_sha256,
        sha256(&fs::read(&manifest).unwrap())
    );
    let pnm = load_auto_routing_pnm(&loaded.workloads[1]).unwrap();
    assert_eq!((pnm.width, pnm.height, pnm.components), (1, 1, 3));
    assert_eq!(pnm.pixels, [1, 2, 3]);
}

#[test]
fn manifest_loader_rejects_escape_hash_mismatch_and_duplicate_ids() {
    let root = temp_dir("invalid");
    fs::write(root.join("sample.j2k"), b"sample").unwrap();
    let manifest = root.join("manifest.json");
    let base = json!({
        "schema_version": 1,
        "corpus": "routing-fixture",
        "source_url": "https://example.invalid/routing-fixture",
        "cases": [{
            "id": "case",
            "path": "../outside.j2k",
            "kind": "decode",
            "pixel_format": "gray8",
            "sha256": sha256(b"sample")
        }]
    });
    fs::write(&manifest, serde_json::to_vec(&base).unwrap()).unwrap();
    assert!(load_auto_routing_manifest(&manifest, &root)
        .unwrap_err()
        .contains("safe relative path"));

    let mut mismatch = base.clone();
    mismatch["cases"][0]["path"] = json!("sample.j2k");
    mismatch["cases"][0]["sha256"] = json!("0".repeat(64));
    fs::write(&manifest, serde_json::to_vec(&mismatch).unwrap()).unwrap();
    assert!(load_auto_routing_manifest(&manifest, &root)
        .unwrap_err()
        .contains("SHA-256 mismatch"));

    let mut duplicate = mismatch;
    duplicate["cases"][0]["sha256"] = json!(sha256(b"sample"));
    let repeated = duplicate["cases"][0].clone();
    duplicate["cases"].as_array_mut().unwrap().push(repeated);
    fs::write(&manifest, serde_json::to_vec(&duplicate).unwrap()).unwrap();
    assert!(load_auto_routing_manifest(&manifest, &root)
        .unwrap_err()
        .contains("unique ids"));
}

#[test]
fn manifest_v2_classifies_part15_codestream_jph_and_encoder_workloads() {
    let root = temp_dir("part15-manifest");
    for (path, bytes) in [
        ("decode/sample.j2k", b"ht codestream".as_slice()),
        ("decode/sample.jph", b"jph file".as_slice()),
        (
            "encode/sample.ppm",
            b"P6\n1 1\n255\n\x01\x02\x03".as_slice(),
        ),
    ] {
        let path = root.join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }
    let manifest = root.join("manifest.json");
    let manifest_value = json!({
        "schema_version": 2,
        "corpus": "part15-routing-fixture",
        "source_url": "https://example.invalid/part15-routing-fixture",
        "cases": [
            {
                "id": "ht-codestream",
                "path": "decode/sample.j2k",
                "kind": "decode",
                "codec": "htj2k-part-15",
                "container": "codestream",
                "pixel_format": "rgb8",
                "sha256": sha256(b"ht codestream")
            },
            {
                "id": "jph-file",
                "path": "decode/sample.jph",
                "kind": "decode",
                "codec": "htj2k-part-15",
                "container": "jph",
                "pixel_format": "rgb8",
                "sha256": sha256(b"jph file")
            },
            {
                "id": "ht-encode",
                "path": "encode/sample.ppm",
                "kind": "encode",
                "codec": "htj2k-part-15",
                "container": "codestream",
                "pixel_format": "rgb8",
                "sha256": sha256(b"P6\n1 1\n255\n\x01\x02\x03")
            }
        ]
    });
    fs::write(&manifest, serde_json::to_vec(&manifest_value).unwrap()).unwrap();

    let loaded = load_auto_routing_manifest(&manifest, &root).unwrap();

    assert_eq!(loaded.manifest.schema_version, 2);
    assert_eq!(loaded.workloads.len(), 3);
    assert!(loaded
        .workloads
        .iter()
        .all(|workload| workload.codec == AutoRoutingCodec::Htj2kPart15));
    assert_eq!(
        loaded.workloads[0].container,
        AutoRoutingContainer::Codestream
    );
    assert_eq!(loaded.workloads[1].container, AutoRoutingContainer::Jph);
    validate_auto_routing_decode_identity(
        &loaded.workloads[0],
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    )
    .unwrap();
    validate_auto_routing_decode_identity(
        &loaded.workloads[1],
        CompressedTransferSyntax::HtJpeg2000Lossless,
        CompressedPayloadKind::JphFile,
    )
    .unwrap();
    let error = validate_auto_routing_decode_identity(
        &loaded.workloads[1],
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jp2File,
    )
    .expect_err("declared JPH identity must be checked against production inspection");
    assert!(error.contains("does not match inspected"), "{error}");
    let pnm = load_auto_routing_pnm(&loaded.workloads[2]).unwrap();
    assert_eq!(pnm.codec, AutoRoutingCodec::Htj2kPart15);
    let mut wrapped_encode = loaded.workloads[2].clone();
    wrapped_encode.container = AutoRoutingContainer::Jph;
    let error = load_auto_routing_pnm(&wrapped_encode)
        .expect_err("routing encode currently measures codestream output only");
    assert!(error.contains("codestream output"), "{error}");
}

#[test]
fn manifest_v2_requires_explicit_legal_codec_container_pairs() {
    let root = temp_dir("part15-manifest-invalid");
    fs::write(root.join("sample.j2k"), b"sample").unwrap();
    let manifest = root.join("manifest.json");
    let mut value = json!({
        "schema_version": 2,
        "corpus": "part15-routing-fixture",
        "source_url": "https://example.invalid/part15-routing-fixture",
        "cases": [{
            "id": "case",
            "path": "sample.j2k",
            "kind": "decode",
            "pixel_format": "gray8",
            "sha256": sha256(b"sample")
        }]
    });
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let error = load_auto_routing_manifest(&manifest, &root)
        .expect_err("schema v2 must identify codec and container");
    assert!(error.contains("codec and container"), "{error}");

    value["cases"][0]["codec"] = json!("jpeg-2000-part-1");
    value["cases"][0]["container"] = json!("jph");
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let error =
        load_auto_routing_manifest(&manifest, &root).expect_err("Part 1 cannot be declared as JPH");
    assert!(error.contains("codec/container pair"), "{error}");
}

#[test]
fn evidence_writer_is_deterministic_and_refuses_invalid_route_labels() {
    let root = temp_dir("evidence");
    let output = root.join("nested/evidence.json");
    let mut evidence = AutoRoutingEvidence {
        schema_version: 1,
        candidate_sha: "1".repeat(40),
        backend: AutoRoutingBackend::Metal,
        platform: AutoRoutingPlatform {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            hardware: "Apple M fixture".to_string(),
            driver: "fixture driver".to_string(),
        },
        external_manifest_sha256: "2".repeat(64),
        external_case_count: 2,
        cells: vec![auto_routing_route_cell(
            "decode-case",
            AutoRoutingOperation::FullDecode,
            "auto-routing_full-decode_decode-case",
            "3".repeat(64),
        )],
    };

    write_auto_routing_evidence(&output, &evidence).unwrap();
    let first = fs::read(&output).unwrap();
    write_auto_routing_evidence(&output, &evidence).unwrap();
    assert_eq!(fs::read(&output).unwrap(), first);
    assert!(first.ends_with(b"\n"));

    evidence.cells[0].hybrid.execution = AutoRoutingExecution::Cpu;
    assert!(write_auto_routing_evidence(&output, &evidence)
        .unwrap_err()
        .contains("hybrid route"));
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn temp_dir(label: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-test-support-auto-routing-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}
