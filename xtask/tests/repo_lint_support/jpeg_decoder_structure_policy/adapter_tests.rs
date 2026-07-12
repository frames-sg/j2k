// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership ratchets for JPEG adapter production and integration tests.

use std::{fs, path::Path};

use super::super::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

const JPEG_DEVICE_PLAN_TEST_LEAVES: &[(&str, usize)] = &[
    ("crates/j2k-jpeg/tests/device_plan/support.rs", 175),
    (
        "crates/j2k-jpeg/tests/device_plan/capability_routing.rs",
        200,
    ),
    ("crates/j2k-jpeg/tests/device_plan/color_components.rs", 225),
    (
        "crates/j2k-jpeg/tests/device_plan/device_plan_basics.rs",
        185,
    ),
    ("crates/j2k-jpeg/tests/device_plan/malformed_inputs.rs", 240),
    ("crates/j2k-jpeg/tests/device_plan/progressive.rs", 345),
    (
        "crates/j2k-jpeg/tests/device_plan/extended/color_and_sampling.rs",
        565,
    ),
    (
        "crates/j2k-jpeg/tests/device_plan/extended/four_component.rs",
        350,
    ),
    ("crates/j2k-jpeg/tests/device_plan/lossless/app14.rs", 190),
    (
        "crates/j2k-jpeg/tests/device_plan/lossless/grayscale.rs",
        140,
    ),
    (
        "crates/j2k-jpeg/tests/device_plan/lossless/sampling.rs",
        375,
    ),
    ("crates/j2k-jpeg/tests/device_plan/lossless/ycbcr.rs", 190),
];

fn read_repo_file(root: &Path, relative_path: &str) -> String {
    fs::read_to_string(root.join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

#[test]
fn jpeg_device_plan_integration_tests_use_focused_real_modules() {
    let root = repo_root();
    let target = read_repo_file(root, "crates/j2k-jpeg/tests/device_plan.rs");
    let suite = read_repo_file(root, "crates/j2k-jpeg/tests/device_plan/mod.rs");
    let extended = read_repo_file(root, "crates/j2k-jpeg/tests/device_plan/extended.rs");
    let lossless = read_repo_file(root, "crates/j2k-jpeg/tests/device_plan/lossless.rs");

    assert!(
        target.lines().count() <= 8
            && target.contains("#[path = \"device_plan/mod.rs\"]")
            && target.contains("mod suite;")
            && target.matches("mod ").count() == 1,
        "device_plan.rs must remain a minimal integration-target shell"
    );
    for declaration in [
        "mod capability_routing;",
        "mod color_components;",
        "mod device_plan_basics;",
        "mod extended;",
        "mod lossless;",
        "mod malformed_inputs;",
        "mod progressive;",
        "mod support;",
    ] {
        assert!(
            suite.contains(declaration),
            "device_plan suite lost required boundary {declaration}"
        );
    }
    assert!(suite.lines().count() <= 15 && suite.matches("mod ").count() == 8);
    for declaration in ["mod color_and_sampling;", "mod four_component;"] {
        assert!(extended.contains(declaration));
    }
    assert!(extended.lines().count() <= 8 && extended.matches("mod ").count() == 2);
    for declaration in [
        "mod app14;",
        "mod grayscale;",
        "mod sampling;",
        "mod ycbcr;",
    ] {
        assert!(lossless.contains(declaration));
    }
    assert!(lossless.lines().count() <= 10 && lossless.matches("mod ").count() == 4);

    let mut test_count = 0usize;
    for (relative_path, max_lines) in JPEG_DEVICE_PLAN_TEST_LEAVES {
        let source = read_repo_file(root, relative_path);
        let line_count = source.lines().count();
        assert!(
            line_count <= *max_lines,
            "{relative_path} grew to {line_count} lines; split it before exceeding {max_lines}"
        );
        assert!(
            !source.contains("include!(") && !source.contains("#[path") && !source.contains("::*"),
            "{relative_path} must remain a real module with explicit imports"
        );
        test_count += source.matches("#[test]").count();
    }
    assert_eq!(
        test_count, 70,
        "the device-plan split must preserve its behavior-test inventory"
    );
}

#[test]
fn jpeg_fast_packet_owner_uses_focused_real_modules() {
    let root = repo_root();
    let owner = fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet.rs"))
        .expect("read JPEG fast-packet owner");
    let module_limits = [
        ("allocation.rs", 130usize),
        ("build.rs", 250),
        ("build/gray.rs", 100),
        ("build/materialization.rs", 40),
        ("cache.rs", 90),
        ("checkpoints.rs", 175),
        ("entropy.rs", 250),
        ("error.rs", 125),
        ("family.rs", 75),
        ("header.rs", 230),
        ("types.rs", 330),
        ("tests.rs", 30),
    ];

    assert!(
        owner.lines().count() <= 30 && owner.matches("mod ").count() == 10,
        "JPEG fast-packet root must remain a minimal module owner"
    );
    for declaration in [
        "mod allocation;",
        "mod build;",
        "mod cache;",
        "mod checkpoints;",
        "mod entropy;",
        "mod error;",
        "mod family;",
        "mod header;",
        "mod types;",
        "mod tests;",
    ] {
        assert!(
            owner.contains(declaration),
            "JPEG fast-packet owner lost required boundary {declaration}"
        );
    }
    assert_pattern_checks(&[PatternCheck::new("JPEG fast-packet owner", &owner)
        .required(&[
            "pub use build::{",
            "build_fast420_packet",
            "build_fast422_packet",
            "build_fast444_packet",
            "build_gray_packet",
            "pub use cache::{",
            "pub use error::{FastPacketError, TableKind};",
            "pub use family::{classify_color_fast_packet_family, JpegFastPacketFamily};",
            "pub use types::{",
            "JpegCanonicalHuffmanTable",
            "JpegEntropyCheckpointV1",
            "JpegFast420PacketV1",
            "JpegFast422PacketV1",
            "JpegFast444PacketV1",
            "JpegGrayPacketV1",
            "JpegHuffmanTable",
        ])
        .forbidden(&[
            "pub struct JpegFast420PacketV1",
            "fn build_color_fast_packet",
            "fn packet_checkpoints_from_device",
            "fn extract_entropy_segments",
        ])]);

    for (filename, max_lines) in module_limits {
        let relative = format!("crates/j2k-jpeg/src/adapter/fast_packet/{filename}");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let line_count = source.lines().count();
        assert!(
            line_count <= max_lines,
            "{relative} grew to {line_count} lines; split it before exceeding {max_lines}"
        );
        assert!(
            !source.contains("include!(") && !source.contains("#[path") && !source.contains("::*"),
            "{relative} must remain a real module with explicit imports"
        );
    }

    let build = fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/build.rs"))
        .expect("read JPEG fast-packet builder");
    let header = fs::read_to_string(root.join("crates/j2k-jpeg/src/adapter/fast_packet/header.rs"))
        .expect("read JPEG fast-packet header adapter");
    assert_eq!(
        header.matches(".get(usize::from(slot))").count(),
        2,
        "quant and defensive Huffman selectors must remain bounds checked"
    );
    assert!(!build.contains("tables[slot as usize]") && !header.contains("tables[slot as usize]"));
}

#[test]
fn jpeg_fast_packet_accessors_stay_out_of_public_api() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-jpeg/src/adapter/fast_packet/types.rs")
                .named("shared JPEG fast-packet ABI types")
                .required(&[
                    "pub struct JpegFast420PacketV1",
                    "pub struct JpegFast422PacketV1",
                    "pub struct JpegFast444PacketV1",
                ])
                .forbidden(&[
                    "pub trait JpegColorFastPacket",
                    "impl_color_fast_packet_access",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg/src/adapter/mod.rs")
                .named("j2k-jpeg adapter facade")
                .forbidden(&["JpegColorFastPacket"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/owned_decode/plan.rs")
                .named("CUDA owned-decode plan")
                .required(&[
                    "macro_rules! fast_rgb8_packet_parts",
                    "fn build_cuda_rgb8_plan_data",
                ])
                .forbidden(&[
                    "JpegColorFastPacket",
                    "trait JpegColorFastPacket",
                    "macro_rules! impl_color_fast_packet_access",
                    "macro_rules! cuda_decode_plan",
                ]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/compute/fast_packets/descriptors.rs")
                .named("Metal fast packets")
                .required(&[
                    "trait FastSubsampledPacket",
                    "macro_rules! impl_fast_subsampled_packet_accessors",
                ])
                .forbidden(&[
                    "JpegColorFastPacket",
                    "trait JpegColorFastPacket",
                    "macro_rules! impl_color_fast_packet_access",
                ]),
            FilePatternCheck::stable_api_snapshots()
                .named("stable API snapshot union")
                .forbidden(&[
                    "pub trait j2k_jpeg::adapter::JpegColorFastPacket",
                    "j2k_jpeg::adapter::JpegColorFastPacket::",
                ]),
        ],
    );
}
