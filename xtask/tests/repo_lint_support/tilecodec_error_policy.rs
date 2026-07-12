// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed-source ownership policy for tile-codec I/O failures.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn tilecodec_io_failures_preserve_typed_sources() {
    let root = repo_root();
    let error = fs::read_to_string(root.join("crates/j2k-tilecodec/src/error.rs"))
        .expect("read tilecodec error module");
    let production = error
        .split_once("#[cfg(test)]")
        .map_or(error.as_str(), |(production, _)| production);
    let consumers = read_source_files(
        root,
        &[
            "crates/j2k-tilecodec/src/bounded.rs",
            "crates/j2k-tilecodec/src/deflate.rs",
            "crates/j2k-tilecodec/src/zstd_codec.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("tilecodec typed I/O error model", production)
            .required(&[
                "#[non_exhaustive]",
                "pub enum TileCodecError",
                "Malformed {",
                "Io {",
                "source: std::io::Error,",
                "ErrorKind::UnexpectedEof => truncated_input(context)",
                "ErrorKind::InvalidData | ErrorKind::InvalidInput => TileCodecError::Malformed",
            ])
            .normalized_required(&[
                "pub(crate) fn malformed_io_error(error: std::io::Error, context: &'static str) -> TileCodecError",
                "pub(crate) fn input_or_backend_io_error( error: std::io::Error, context: &'static str, ) -> TileCodecError",
            ])
            .forbidden(&[
                "Backend(String)",
                "TileCodecError::Backend",
                "error.to_string()",
                "format!(\"{context}: {error}\")",
                "error: &std::io::Error",
            ]),
        PatternCheck::new("tilecodec owned I/O handoff", &consumers)
            .required(&[
                "input_or_backend_io_error(error, \"bounded decode\")",
                "input_or_backend_io_error(raw_error, \"deflate decode failed\")",
                "malformed_io_error(error, \"zstd decoder init\")",
                "malformed_io_error(error, \"zstd decode failed\")",
            ])
            .forbidden(&[
                "input_or_backend_io_error(&error",
                "input_or_backend_io_error(&raw_error",
                "malformed_io_error(&error",
            ]),
        PatternCheck::new("tilecodec I/O source behavior coverage", &error).required(&[
            "operational_io_error_preserves_source_kind_context_and_display",
            "malformed_io_errors_preserve_original_kinds_and_sources",
            "malformed_only_helper_preserves_non_eof_io_source",
            "unexpected_eof_remains_a_truncated_input_without_io_source",
            "decoder_status_messages_use_an_invalid_data_source",
        ]),
    ]);

    assert_eq!(
        production.matches("source: std::io::Error,").count(),
        2,
        "malformed and operational I/O variants must own their typed source"
    );
}
