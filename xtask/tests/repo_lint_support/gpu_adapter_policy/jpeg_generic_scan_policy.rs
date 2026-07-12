// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic JPEG sequential scan-driver ownership policy.

use super::*;

#[test]
fn jpeg_generic_output_modes_share_one_typed_scan_driver() {
    let root = repo_root();
    let entry = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/generic.rs"))
        .expect("read generic sequential entry points");
    let driver =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/generic/driver.rs"))
            .expect("read generic sequential scan driver");
    let row =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/sequential/generic/row.rs"))
            .expect("read generic sequential MCU row kernel");
    let regressions = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/tests.rs"))
        .expect("read JPEG decoder regressions");

    assert_pattern_checks(&[
        PatternCheck::new("generic JPEG scan entry points", &entry)
            .required(&[
                "mod driver;",
                "mod row;",
                "struct ComponentStripeEmitter",
                "struct RgbStripeEmitter",
                "ScanOutputMode::ComponentRows",
                "ScanOutputMode::InterleavedRgb",
                "decode_scan_rows(",
            ])
            .forbidden(&[
                "restart_seek_for_mcu",
                "skip_to_mcu",
                "decode_mcu_row(",
                "clippy::too_many_lines",
                "macro_rules!",
            ]),
        PatternCheck::new("typed generic JPEG scan driver", &driver)
            .required(&[
                "struct ScanSetup",
                "struct ScanBuffers",
                "trait StripeEmitter",
                "fn decode_scan_rows",
                "restart_seek_for_mcu(",
                "skip_to_mcu(",
                "for mcu_row in setup.first_decode_mcu_row + 1..setup.decode_mcu_row_end",
                "emitter.emit(StripeEmit {",
                "finish_scan(&mut br, setup.decode_mcu_row_end == setup.mcu_rows)",
            ])
            .forbidden(&["emit_stripe_rgb(", "emit_stripe(", "macro_rules!"]),
        PatternCheck::new("focused generic JPEG MCU row kernel", &row).required(&[
            "fn decode_mcu_row(",
            "consume_restart_marker_if_due(",
            "decode_block_with_activity(",
            "skip_block(",
        ]),
        PatternCheck::new("generic scan output-mode parity regression", &regressions).required(&[
            "shared_generic_scan_driver_preserves_restart_region_output_mode_parity",
            "decode_scan_baseline_rgb(",
            "decode_scan_baseline(",
        ]),
    ]);

    assert_eq!(
        entry.matches("decode_scan_rows(").count(),
        2,
        "both output entry points must delegate to the one typed driver"
    );
    assert!(entry.lines().count() < 180);
    assert!(driver.lines().count() < 260);
    assert!(row.lines().count() < 230);
}
