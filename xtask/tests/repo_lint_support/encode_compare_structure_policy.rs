// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the encode-comparison harness.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

fn assert_line_budget(relative_path: &str, source: &str, max_lines: usize) {
    let line_count = source.lines().count();
    assert!(
        line_count < max_lines,
        "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
    );
}

#[test]
fn encode_compare_stays_split_by_responsibility() {
    let shell = read("crates/j2k-compare/src/encode_compare.rs");
    let types = read("crates/j2k-compare/src/encode_compare/types.rs");
    let cli = read("crates/j2k-compare/src/encode_compare/cli.rs");
    let images = read("crates/j2k-compare/src/encode_compare/images.rs");
    let tools = read("crates/j2k-compare/src/encode_compare/tools.rs");
    let validation = read("crates/j2k-compare/src/encode_compare/validation.rs");
    let measurement = read("crates/j2k-compare/src/encode_compare/measurement.rs");
    let render = read("crates/j2k-compare/src/encode_compare/render.rs");

    for (path, source, max_lines) in [
        ("encode_compare.rs", shell.as_str(), 350),
        ("encode_compare/types.rs", types.as_str(), 200),
        ("encode_compare/cli.rs", cli.as_str(), 200),
        ("encode_compare/images.rs", images.as_str(), 525),
        ("encode_compare/tools.rs", tools.as_str(), 400),
        ("encode_compare/validation.rs", validation.as_str(), 325),
        ("encode_compare/measurement.rs", measurement.as_str(), 325),
        ("encode_compare/render.rs", render.as_str(), 725),
    ] {
        assert_line_budget(path, source, max_lines);
        assert!(
            !source.contains("use super::*"),
            "crates/j2k-compare/src/{path} must keep explicit module imports"
        );
        assert!(
            !source.contains("include!("),
            "crates/j2k-compare/src/{path} must remain a real Rust module"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("encode_compare coordinator", &shell)
            .required(&[
                "mod types;",
                "mod cli;",
                "mod images;",
                "mod tools;",
                "mod validation;",
                "mod measurement;",
                "mod render;",
                "fn run() -> Result<(), String>",
            ])
            .forbidden(&[
                "struct ImageCase",
                "fn batch_size_config_from_env(",
                "fn load_external_image_cases(",
                "fn all_encoder_tools(",
                "fn validate_encoded_profile(",
                "fn measure_case_rows(",
                "fn emit_metadata(",
            ]),
        PatternCheck::new("encode_compare type ownership", &types).required(&[
            "pub(super) struct ImageCase",
            "pub(super) enum EncoderKind",
            "pub(super) struct Measurement",
            "pub(super) struct MetadataInput",
        ]),
        PatternCheck::new("encode_compare CLI ownership", &cli).required(&[
            "pub(super) fn encode_one",
            "pub(super) fn validate_tool_gates",
            "pub(super) fn batch_size_config_from_env",
            "pub(super) fn encode_work_dir",
        ]),
        PatternCheck::new("encode_compare image ownership", &images).required(&[
            "pub(super) fn load_external_image_cases",
            "pub(super) fn encode_manifest_from_env",
            "pub(super) fn collect_source_image_paths",
            "pub(super) fn read_raster_image",
            "pub(super) fn read_pnm",
        ]),
        PatternCheck::new("encode_compare tool ownership", &tools).required(&[
            "pub(super) fn all_encoder_tools",
            "pub(super) fn run_encoder_once",
            "pub(super) fn command_version_label",
            "pub(super) fn selected_encoders_label",
        ]),
        PatternCheck::new("encode_compare validation ownership", &validation).required(&[
            "pub(super) fn validate_encoded_profile",
            "pub(super) struct CodProfile",
            "pub(super) fn cod_profile",
            "pub(super) fn decode_encoded_output",
        ]),
        PatternCheck::new("encode_compare measurement ownership", &measurement).required(&[
            "pub(super) fn measure_case_rows",
            "pub(super) fn measure_mixed_rows",
            "pub(super) fn measure_case_encoder_once",
            "pub(super) fn measurement",
        ]),
        PatternCheck::new("encode_compare render ownership", &render).required(&[
            "pub(super) fn emit_metadata",
            "pub(super) fn publication_blockers",
            "pub(super) fn measurement_row",
            "pub(super) fn mixed_measurement_row",
            "pub(super) fn unique_image_count",
        ]),
    ]);
}
