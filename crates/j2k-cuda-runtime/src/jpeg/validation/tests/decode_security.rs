// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::super::{
    validate_jpeg_entropy_chunk_plan, validate_jpeg_rgb8_plan, validate_jpeg_rgb8_plan_with_pitch,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::assert_jpeg_invalid_argument;
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::{CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan};
use crate::{
    CudaJpegEntropyCheckpoint, CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};

pub(super) fn decode_plan(checkpoints: &[CudaJpegEntropyCheckpoint]) -> CudaJpegRgb8DecodePlan<'_> {
    let huffman = valid_huffman_table();
    CudaJpegRgb8DecodePlan {
        sampling: CudaJpegRgb8Sampling::Fast444,
        dimensions: (1, 1),
        mcus_per_row: 1,
        mcu_rows: 1,
        entropy_bytes: &[0],
        entropy_checkpoints: checkpoints,
        y_quant: [1; 64],
        cb_quant: [1; 64],
        cr_quant: [1; 64],
        y_dc_table: huffman,
        y_ac_table: huffman,
        cb_dc_table: huffman,
        cb_ac_table: huffman,
        cr_dc_table: huffman,
        cr_ac_table: huffman,
    }
}

fn valid_huffman_table() -> CudaJpegHuffmanTable {
    let mut values = [0; 256];
    values[0] = 0;
    CudaJpegHuffmanTable::from_jpeg_bits_values(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1,
        values,
    )
    .expect("one-code JPEG Huffman table")
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
fn decode_plan_for(
    sampling: CudaJpegRgb8Sampling,
    dimensions: (u32, u32),
    checkpoints: &[CudaJpegEntropyCheckpoint],
) -> CudaJpegRgb8DecodePlan<'_> {
    let mut plan = decode_plan(checkpoints);
    plan.sampling = sampling;
    plan.dimensions = dimensions;
    let (mcu_width, mcu_height) = match sampling {
        CudaJpegRgb8Sampling::Fast420 => (16, 16),
        CudaJpegRgb8Sampling::Fast422 => (16, 8),
        CudaJpegRgb8Sampling::Fast444 => (8, 8),
    };
    plan.mcus_per_row = dimensions.0.div_ceil(mcu_width);
    plan.mcu_rows = dimensions.1.div_ceil(mcu_height);
    plan
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_accepts_exact_odd_edge_grids_for_every_sampling() {
    let checkpoints = [CudaJpegEntropyCheckpoint::default()];
    for (sampling, dimensions, expected_grid) in [
        (CudaJpegRgb8Sampling::Fast420, (17, 17), (2, 2)),
        (CudaJpegRgb8Sampling::Fast422, (17, 9), (2, 2)),
        (CudaJpegRgb8Sampling::Fast444, (9, 9), (2, 2)),
    ] {
        let plan = decode_plan_for(sampling, dimensions, &checkpoints);
        let validated = validate_jpeg_rgb8_plan(&plan).expect("exact MCU grid must validate");
        assert_eq!(validated.params.mcus_per_row, expected_grid.0);
        assert_eq!(validated.params.mcu_rows, expected_grid.1);
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_accepts_complete_nonuniform_checkpoint_partition() {
    let checkpoints = [
        CudaJpegEntropyCheckpoint::default(),
        CudaJpegEntropyCheckpoint {
            mcu_index: 2,
            entropy_pos: 1,
            ..CudaJpegEntropyCheckpoint::default()
        },
        CudaJpegEntropyCheckpoint {
            mcu_index: 5,
            entropy_pos: 2,
            ..CudaJpegEntropyCheckpoint::default()
        },
    ];
    let mut plan = decode_plan_for(CudaJpegRgb8Sampling::Fast444, (32, 16), &checkpoints);
    plan.entropy_bytes = &[0, 0];

    let validated = validate_jpeg_rgb8_plan(&plan).expect("complete MCU partition");
    assert_eq!(validated.params.mcus_per_row, 4);
    assert_eq!(validated.params.mcu_rows, 2);
    assert_eq!(validated.params.checkpoint_count, 3);
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_rejects_grid_and_checkpoint_coverage_gaps() {
    let checkpoints = [CudaJpegEntropyCheckpoint::default()];
    let mut wrong_grid = decode_plan_for(CudaJpegRgb8Sampling::Fast444, (9, 9), &checkpoints);
    wrong_grid.mcus_per_row = 1;
    assert_jpeg_invalid_argument(validate_jpeg_rgb8_plan(&wrong_grid), "MCU grid");

    let starts_after_zero = [CudaJpegEntropyCheckpoint {
        mcu_index: 1,
        ..CudaJpegEntropyCheckpoint::default()
    }];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan_for(
            CudaJpegRgb8Sampling::Fast444,
            (16, 8),
            &starts_after_zero,
        )),
        "must start at MCU zero",
    );

    let duplicate = [
        CudaJpegEntropyCheckpoint::default(),
        CudaJpegEntropyCheckpoint {
            mcu_index: 0,
            entropy_pos: 1,
            ..CudaJpegEntropyCheckpoint::default()
        },
    ];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan_for(
            CudaJpegRgb8Sampling::Fast444,
            (16, 8),
            &duplicate,
        )),
        "not strictly MCU-ordered",
    );

    let starts_at_end = [
        CudaJpegEntropyCheckpoint::default(),
        CudaJpegEntropyCheckpoint {
            mcu_index: 1,
            entropy_pos: 1,
            ..CudaJpegEntropyCheckpoint::default()
        },
    ];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan(&starts_at_end)),
        "beyond the MCU range",
    );
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_rejects_malformed_checkpoint_bit_state() {
    let malformed_initial = [CudaJpegEntropyCheckpoint {
        entropy_pos: 1,
        ..CudaJpegEntropyCheckpoint::default()
    }];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan(&malformed_initial)),
        "initial decoder state",
    );

    for (second, expected) in [
        (
            CudaJpegEntropyCheckpoint {
                mcu_index: 1,
                entropy_pos: 2,
                ..CudaJpegEntropyCheckpoint::default()
            },
            "beyond the entropy payload",
        ),
        (
            CudaJpegEntropyCheckpoint {
                mcu_index: 1,
                entropy_pos: 1,
                bit_count: 64,
                ..CudaJpegEntropyCheckpoint::default()
            },
            "more than 63 buffered bits",
        ),
        (
            CudaJpegEntropyCheckpoint {
                mcu_index: 1,
                entropy_pos: 1,
                bit_count: 1,
                bit_acc: 1,
                ..CudaJpegEntropyCheckpoint::default()
            },
            "unused accumulator bits",
        ),
        (
            CudaJpegEntropyCheckpoint {
                mcu_index: 1,
                entropy_pos: 1,
                reserved: 1,
                ..CudaJpegEntropyCheckpoint::default()
            },
            "nonzero reserved state",
        ),
    ] {
        let checkpoints = [CudaJpegEntropyCheckpoint::default(), second];
        assert_jpeg_invalid_argument(
            validate_jpeg_rgb8_plan(&decode_plan_for(
                CudaJpegRgb8Sampling::Fast444,
                (16, 8),
                &checkpoints,
            )),
            expected,
        );
    }

    let checkpoints = [
        CudaJpegEntropyCheckpoint::default(),
        CudaJpegEntropyCheckpoint {
            mcu_index: 1,
            entropy_pos: 1,
            bit_count: 1,
            bit_acc: 1,
            ..CudaJpegEntropyCheckpoint::default()
        },
    ];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan_for(
            CudaJpegRgb8Sampling::Fast444,
            (16, 8),
            &checkpoints,
        )),
        "unused accumulator bits",
    );

    let nonadvancing_bits = [
        CudaJpegEntropyCheckpoint::default(),
        CudaJpegEntropyCheckpoint {
            mcu_index: 1,
            entropy_pos: 1,
            bit_count: 8,
            ..CudaJpegEntropyCheckpoint::default()
        },
    ];
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&decode_plan_for(
            CudaJpegRgb8Sampling::Fast444,
            (16, 8),
            &nonadvancing_bits,
        )),
        "does not advance through the entropy payload",
    );
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_rejects_u32_unaddressable_output_before_allocation() {
    let checkpoints = [CudaJpegEntropyCheckpoint::default()];
    let plan = decode_plan_for(
        CudaJpegRgb8Sampling::Fast444,
        (65_500, 65_500),
        &checkpoints,
    );
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&plan),
        "exceeds the kernel's u32 byte addressing",
    );

    let narrow = decode_plan_for(CudaJpegRgb8Sampling::Fast444, (1, 2), &checkpoints);
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan_with_pitch(&narrow, u32::MAX as usize),
        "exceeds the kernel's u32 byte addressing",
    );
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_decode_validation_rejects_noncanonical_and_role_invalid_huffman_tables() {
    let checkpoints = [CudaJpegEntropyCheckpoint::default()];
    let mut noncanonical = decode_plan(&checkpoints);
    noncanonical.y_dc_table.val_offset[1] = 1;
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&noncanonical),
        "non-canonical value offset",
    );

    let mut too_many = decode_plan(&checkpoints);
    too_many.y_dc_table.values_len = 257;
    assert_jpeg_invalid_argument(validate_jpeg_rgb8_plan(&too_many), "value count 257");

    let mut oversized_dc_symbol = decode_plan(&checkpoints);
    oversized_dc_symbol.y_dc_table.values[0] = 12;
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&oversized_dc_symbol),
        "invalid baseline value",
    );

    assert_jpeg_invalid_argument(
        CudaJpegHuffmanTable::from_jpeg_bits_values(
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            2,
            [0; 256],
        ),
        "forbidden all-ones code",
    );
    let mut all_ones = valid_huffman_table();
    all_ones.max_code[1] = 1;
    all_ones.values_len = 2;
    let mut invalid_all_ones = decode_plan(&checkpoints);
    invalid_all_ones.y_dc_table = all_ones;
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&invalid_all_ones),
        "forbidden all-ones code",
    );

    let mut malformed_ac_symbol = decode_plan(&checkpoints);
    malformed_ac_symbol.y_ac_table.values[0] = 0x10;
    assert_jpeg_invalid_argument(
        validate_jpeg_rgb8_plan(&malformed_ac_symbol),
        "invalid baseline value",
    );
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
#[test]
fn jpeg_entropy_diagnostic_validates_huffman_tables_before_launch() {
    let table = valid_huffman_table();
    let mut plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig::default(),
        entropy_bytes: &[0],
        y_dc_table: table,
        y_ac_table: table,
        cb_dc_table: table,
        cb_ac_table: table,
        cr_dc_table: table,
        cr_ac_table: table,
    };
    plan.cb_ac_table.max_code[1] = 2;
    assert_jpeg_invalid_argument(
        validate_jpeg_entropy_chunk_plan(&plan, 1),
        "non-canonical code bounds",
    );
}
