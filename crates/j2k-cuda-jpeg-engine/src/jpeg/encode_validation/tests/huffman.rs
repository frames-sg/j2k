// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    invalid_message, valid_gray, valid_rgb444, validate, TestTables, ENTROPY_LEN, INPUT_LEN,
};

#[test]
fn accepts_empty_missing_and_canonical_prefix_free_huffman_entries() {
    let params = valid_rgb444();
    let empty = TestTables::default();
    assert!(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &empty).is_ok());

    let mut canonical = TestTables::default();
    canonical.huff_dc_luma.lens[0] = 1;
    canonical.huff_dc_luma.codes[0] = 0b0;
    canonical.huff_dc_luma.lens[1] = 2;
    canonical.huff_dc_luma.codes[1] = 0b10;
    canonical.huff_dc_luma.lens[2] = 3;
    canonical.huff_dc_luma.codes[2] = 0b110;
    assert!(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &canonical).is_ok());
}

#[test]
fn rejects_quantization_and_huffman_inputs_that_break_kernel_indexes() {
    let params = valid_rgb444();

    let mut zero_quant = TestTables::default();
    zero_quant.q_luma[63] = 0;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &zero_quant))
            .contains("entry 63 must be nonzero")
    );

    let mut long_code = TestTables::default();
    long_code.huff_ac_luma.lens[255] = 17;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &long_code))
            .contains("above 16")
    );

    let mut code_does_not_fit = TestTables::default();
    code_does_not_fit.huff_dc_luma.lens[1] = 1;
    code_does_not_fit.huff_dc_luma.codes[1] = 2;
    assert!(invalid_message(validate(
        &[params],
        INPUT_LEN,
        0,
        ENTROPY_LEN,
        &code_does_not_fit,
    ))
    .contains("does not fit"));

    let mut unused_chroma = TestTables::default();
    unused_chroma.q_chroma[0] = 0;
    assert!(validate(&[valid_gray()], INPUT_LEN, 0, ENTROPY_LEN, &unused_chroma).is_ok());
}

#[test]
fn rejects_duplicate_prefix_conflicting_noncanonical_and_all_ones_codes() {
    let params = valid_rgb444();

    let mut duplicate = TestTables::default();
    duplicate.huff_dc_luma.lens[0] = 1;
    duplicate.huff_dc_luma.lens[1] = 1;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &duplicate))
            .contains("canonical prefix-free")
    );

    let mut prefix_conflict = TestTables::default();
    prefix_conflict.huff_dc_luma.lens[0] = 1;
    prefix_conflict.huff_dc_luma.lens[1] = 2;
    prefix_conflict.huff_dc_luma.codes[1] = 0b00;
    assert!(invalid_message(validate(
        &[params],
        INPUT_LEN,
        0,
        ENTROPY_LEN,
        &prefix_conflict,
    ))
    .contains("canonical prefix-free"));

    let mut noncanonical_gap = TestTables::default();
    noncanonical_gap.huff_dc_luma.lens[0] = 2;
    noncanonical_gap.huff_dc_luma.codes[0] = 0b01;
    assert!(invalid_message(validate(
        &[params],
        INPUT_LEN,
        0,
        ENTROPY_LEN,
        &noncanonical_gap,
    ))
    .contains("canonical prefix-free"));

    let mut all_ones = TestTables::default();
    all_ones.huff_dc_luma.lens[0] = 1;
    all_ones.huff_dc_luma.codes[0] = 0b1;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &all_ones))
            .contains("all-ones")
    );
}
