// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership ratchets for JPEG entropy restart-marker consumption.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

mod counts;

#[test]
fn jpeg_restart_marker_consumption_has_one_bit_reader_owner() {
    let root = repo_root();
    let bit_reader_path = "crates/j2k-jpeg/src/internal/bit_reader.rs";
    let bit_reader = fs::read_to_string(root.join(bit_reader_path))
        .unwrap_or_else(|error| panic!("read {bit_reader_path}: {error}"));
    let bit_reader_production = bit_reader
        .split_once("#[cfg(test)]")
        .map_or(bit_reader.as_str(), |(production, _)| production);

    let lossless = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/lossless_helpers.rs"))
        .expect("read JPEG lossless helpers");
    let progressive =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/progressive/scan.rs"))
            .expect("read JPEG progressive scan");
    let sequential_restart =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/restart.rs"))
            .expect("read JPEG sequential restart helpers");
    let sequential_dct =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/dct.rs"))
            .expect("read JPEG sequential DCT traversal");
    let checkpoint =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/internal/checkpoint/build.rs"))
            .expect("read JPEG checkpoint builder");
    let consumers = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/decoder/lossless_helpers.rs",
            "crates/j2k-jpeg/src/entropy/progressive/scan.rs",
            "crates/j2k-jpeg/src/entropy/sequential/restart.rs",
            "crates/j2k-jpeg/src/entropy/sequential/dct.rs",
            "crates/j2k-jpeg/src/internal/checkpoint/build.rs",
        ],
    );

    assert_restart_patterns(
        &bit_reader,
        bit_reader_production,
        &lossless,
        &progressive,
        &sequential_restart,
        &sequential_dct,
        &checkpoint,
    );
    counts::assert_restart_counts(
        &consumers,
        &lossless,
        &progressive,
        &sequential_restart,
        &sequential_dct,
        &checkpoint,
    );
}

fn assert_restart_patterns(
    bit_reader: &str,
    bit_reader_production: &str,
    lossless: &str,
    progressive: &str,
    sequential_restart: &str,
    sequential_dct: &str,
    checkpoint: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("BitReader restart-marker transaction", bit_reader_production)
            .required(&["pub(crate) fn consume_restart_marker("])
            .normalized_required(&[
                "if self.bits == 0 { self.refill_one_byte(); }",
                "let marker = self .take_marker() .ok_or(JpegError::UnexpectedEoi { mcu_at, mcu_total })?;",
                "let expected = 0xd0 | expected_rst;",
                "offset: self.position(),",
                "self.reset_at_restart();",
                "Ok((expected_rst + 1) & 0x07)",
            ]),
        PatternCheck::new("restart-marker behavior coverage", bit_reader).required(&[
            "restart_markers_consume_fill_and_advance_the_expected_sequence",
            "wrong_restart_marker_preserves_offset_and_buffered_padding",
            "missing_or_truncated_restart_marker_keeps_mcu_coordinates",
            "stuffed_ff_is_entropy_data_not_a_restart_marker",
            "validated_restart_discards_only_then_buffered_padding",
        ]),
        PatternCheck::new("lossless restart reset ordering", lossless).normalized_required(&[
            "br.reset_at_restart(); *expected_rst = br.consume_restart_marker(",
        ]),
        PatternCheck::new("progressive restart state ordering", progressive)
            .normalized_required(&[
                "*expected_rst = br.consume_restart_marker(",
                "dc_predictors.fill(0); *eob_run = 0;",
            ]),
        PatternCheck::new("sequential restart delegation", sequential_restart)
            .required(&["*expected_rst = br.consume_restart_marker("]),
        PatternCheck::new("sequential DCT restart state ordering", sequential_dct)
            .normalized_required(&[
                "if consume_restart_marker_if_due(",
                ")? { storage.prev_dc.fill(0); mcus_since_restart = 0; }",
            ]),
        PatternCheck::new("checkpoint restart state ordering", checkpoint)
            .normalized_required(&[
                ".consume_restart_marker(expected_rst, mcu_index, build.total_mcus) .map_err(E::from)?;",
                "prev_dc.fill(0); mcus_since_restart = 0; push_planned_checkpoint(",
            ]),
    ]);
}
