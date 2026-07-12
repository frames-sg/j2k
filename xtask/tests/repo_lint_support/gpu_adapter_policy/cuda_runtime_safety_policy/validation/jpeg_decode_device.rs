// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_runtime, PatternCheck};

#[test]
fn jpeg_decode_device_checks_huffman_indexes_and_reports_invalid_plans() {
    let device = read_runtime("cuda_oxide_jpeg_decode/simt/src/main.rs");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA JPEG checked device Huffman indexing", &device).required(&[
            "const JPEG_HUFFMAN_VALUE_CAPACITY: u32 = 256;",
            "fn checked_huffman_value_index(",
            "code.checked_add(val_offset)",
            "index >= values_len || index >= JPEG_HUFFMAN_VALUE_CAPACITY",
            "fn decode_symbol(",
            "fn decode_symbol_real(",
            "set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);",
        ]),
        PatternCheck::new("CUDA JPEG explicit invalid-plan status", &device).required(&[
            "const JPEG_STATUS_INVALID: u32 = 3;",
            "fn checkpoint_bit_position(",
            "fn validate_decode_thread_range(",
            "params.mcus_per_row != expected_mcus_per_row",
            "params.mcus_per_row > u32::MAX / params.mcu_rows",
            "first_checkpoint.mcu_index != 0",
            "previous.mcu_index >= start_mcu || previous_bit_position >= current_bit_position",
            "next.mcu_index <= start_mcu",
            "let end_mcu = if gid + 1 < params.checkpoint_count",
            "set_error(status, JPEG_STATUS_INVALID",
            "store_decode_status(status, gid, thread_status);",
        ]),
    ]);
    assert_eq!(
        device.matches("checked_huffman_value_index(").count(),
        3,
        "both JPEG symbol decoders must share the fixed-capacity index guard"
    );
    assert_eq!(
        device.matches("validate_decode_thread_range(").count(),
        4,
        "all three RGB8 sampling kernels must share explicit plan validation"
    );
    assert!(
        device.matches("JPEG_STATUS_INVALID").count() >= 10,
        "malformed JPEG decode plans must report explicit invalid statuses"
    );
    assert_eq!(
        device.matches("while mcu < end_mcu").count(),
        3,
        "every RGB8 sampling kernel must decode its complete validated half-open MCU range"
    );
    assert_eq!(
        device.matches("mcu += 1;").count(),
        3,
        "every RGB8 sampling kernel must advance exactly once per stored MCU"
    );
}
