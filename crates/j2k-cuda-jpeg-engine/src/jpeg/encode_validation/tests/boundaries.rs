// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{invalid_message, valid_rgb444, validate, TestTables, ENTROPY_LEN, INPUT_LEN};

const U32_ADDRESSABLE_BYTES: u64 = 1u64 << 32;

#[test]
fn accepts_exact_input_and_entropy_ends_then_rejects_one_byte_short() {
    let tables = TestTables::default();
    let params = valid_rgb444();
    let exact_input_len = 16 + 8 * 64 + 51;
    let exact_entropy_len = 32 + 256;
    assert!(validate(&[params], exact_input_len, 0, exact_entropy_len, &tables,).is_ok());
    assert!(invalid_message(validate(
        &[params],
        exact_input_len - 1,
        0,
        exact_entropy_len,
        &tables,
    ))
    .contains("beyond allocation"));
    assert!(invalid_message(validate(
        &[params],
        exact_input_len,
        0,
        exact_entropy_len - 1,
        &tables,
    ))
    .contains("beyond allocation"));
}

#[test]
fn accepts_exact_u32_index_end_and_rejects_the_next_byte() {
    let tables = TestTables::default();
    let maximum_addressable = usize::try_from(U32_ADDRESSABLE_BYTES)
        .expect("test host represents the CUDA u32 byte-index space");
    let mut params = valid_rgb444();
    params.entropy_offset_bytes = 1;
    params.entropy_capacity = u32::MAX;
    assert!(validate(&[params], INPUT_LEN, 0, maximum_addressable, &tables,).is_ok());

    params.entropy_offset_bytes = 2;
    assert!(invalid_message(validate(
        &[params],
        INPUT_LEN,
        0,
        maximum_addressable,
        &tables,
    ))
    .contains("u32 byte indexes"));
}

#[test]
fn accepts_last_u32_input_index_and_rejects_row_wrap() {
    let tables = TestTables::default();
    let maximum_addressable = usize::try_from(U32_ADDRESSABLE_BYTES)
        .expect("test host represents the CUDA u32 byte-index space");
    let mut params = valid_rgb444();
    params.input_height = 2;
    params.output_height = 2;
    params.mcu_rows = 1;
    params.pitch_bytes = u32::MAX - 50;
    let exact_input_len = maximum_addressable + 16;
    assert!(validate(&[params], exact_input_len, 0, ENTROPY_LEN, &tables,).is_ok());

    params.pitch_bytes += 1;
    assert!(invalid_message(validate(
        &[params],
        exact_input_len + 1,
        0,
        ENTROPY_LEN,
        &tables,
    ))
    .contains("row footprint"));
}

#[test]
fn rejects_zero_out_of_allocation_and_oversized_entropy_ranges() {
    let tables = TestTables::default();
    let mut params = valid_rgb444();
    params.entropy_capacity = 0;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &tables))
            .contains("must be nonzero")
    );

    params = valid_rgb444();
    params.entropy_offset_bytes = 900;
    params.entropy_capacity = 200;
    assert!(
        invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &tables))
            .contains("beyond allocation")
    );

    let too_large_allocation = usize::try_from(U32_ADDRESSABLE_BYTES + 1)
        .expect("test host represents a value above the CUDA u32 index space");
    assert!(invalid_message(validate(
        &[valid_rgb444()],
        INPUT_LEN,
        0,
        too_large_allocation,
        &tables,
    ))
    .contains("allocation exceeds u32"));
}
