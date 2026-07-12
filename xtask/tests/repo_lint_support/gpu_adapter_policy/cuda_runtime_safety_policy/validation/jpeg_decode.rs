// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_runtime, PatternCheck};

mod preflight;

#[test]
fn cuda_jpeg_decode_validation_leaves_stay_focused() {
    for (relative, max_lines) in [
        ("allocation.rs", 100usize),
        ("allocation/phase.rs", 100),
        ("allocation/tests.rs", 100),
        ("jpeg/decode.rs", 300),
        ("jpeg/diagnostics.rs", 250),
        ("jpeg/validation.rs", 125),
        ("jpeg/validation/decode_plan.rs", 250),
        ("jpeg/validation/decode_plan/checkpoints.rs", 150),
        ("jpeg/validation/huffman.rs", 150),
        ("jpeg/validation/tests.rs", 200),
        ("jpeg/validation/tests/decode_security.rs", 350),
    ] {
        let source = read_runtime(relative);
        assert!(
            source.lines().count() < max_lines,
            "CUDA {relative} must stay below its {max_lines}-line focus ratchet"
        );
    }
}

#[test]
fn jpeg_decode_plan_requires_exact_grid_partition_and_addressing() {
    let decode_plan = read_runtime("jpeg/validation/decode_plan.rs");
    let checkpoints = read_runtime("jpeg/validation/decode_plan/checkpoints.rs");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA JPEG exact sampling MCU grid", &decode_plan).required(&[
            "CudaJpegRgb8Sampling::Fast420 => (16, 16)",
            "CudaJpegRgb8Sampling::Fast422 => (16, 8)",
            "CudaJpegRgb8Sampling::Fast444 => (8, 8)",
            "let expected_mcus_per_row = width.div_ceil(mcu_width);",
            "let expected_mcu_rows = height.div_ceil(mcu_height);",
            "plan.mcus_per_row != expected_mcus_per_row",
            ".checked_mul(expected_mcu_rows)",
        ]),
        PatternCheck::new("CUDA JPEG exact checkpoint partition", &checkpoints).required(&[
            "fn validate_entropy_checkpoints(",
            "if first.mcu_index != 0",
            "fn validate_complete_mcu_partition(",
            "for (index, pair) in checkpoints.windows(2).enumerate()",
            "let start_mcu = pair[0].mcu_index;",
            "let end_mcu = pair[1].mcu_index;",
            "if start_mcu >= end_mcu",
            "if end_mcu >= total_mcus",
            "range ends at total_mcus",
            "complete, non-overlapping coverage",
            "checkpoint.entropy_pos > entropy_len",
            "checkpoint.bit_count > 63",
            "checkpoint.bit_acc & unused_mask != 0",
            "consumed_bits <= previous",
        ]),
        PatternCheck::new("CUDA JPEG u32 output addressing", &decode_plan).required(&[
            "const U32_ADDRESSABLE_BYTES: u64 = u32::MAX as u64 + 1;",
            "u64::from(out_stride)",
            ".checked_mul(u64::from(height - 1))",
            ".and_then(|prefix| prefix.checked_add(u64::from(row_bytes)))",
            "if output_len_u64 > U32_ADDRESSABLE_BYTES",
            "exceeds the kernel's u32 byte addressing",
        ]),
    ]);
}

#[test]
fn jpeg_decode_requires_canonical_role_checked_huffman_tables() {
    let validation = read_runtime("jpeg/validation.rs");
    let huffman = read_runtime("jpeg/validation/huffman.rs");
    assert_pattern_checks(&[
        PatternCheck::new("CUDA JPEG Huffman validation integration", &validation).required(&[
            "huffman::validate_entropy_huffman_tables(plan)?;",
            "mod huffman;",
        ]),
        PatternCheck::new("CUDA JPEG canonical Huffman validation", &huffman).required(&[
            "const JPEG_HUFFMAN_CAPACITY: u32 = 256;",
            "table.values_len == 0 || table.values_len > JPEG_HUFFMAN_CAPACITY",
            "table.max_code[0] != -1 || table.val_offset[0] != 0",
            "let mut next_code = 0i64;",
            "let mut value_cursor = 0i64;",
            "max_code == code_limit - 1",
            "val_offset != value_cursor - next_code",
            "value_cursor != i64::from(table.values_len)",
        ]),
        PatternCheck::new("CUDA JPEG role-specific Huffman symbols", &huffman).required(&[
            "HuffmanRole::Dc => symbol <= 11",
            "HuffmanRole::Ac =>",
            "size <= 10 && (size != 0 || matches!(symbol, 0x00 | 0xf0))",
            "validate_rgb8_huffman_tables(",
            "validate_entropy_huffman_tables(",
        ]),
    ]);
}
