// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    validate_jpeg_baseline_encode_request, CudaJpegBaselineEncodeTableRefs,
    CudaJpegBaselineEncodeValidation,
};
use crate::jpeg::{
    CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
};
use crate::CudaError;

mod boundaries;
mod huffman;
mod launch_geometry;

const INPUT_LEN: usize = 1_024;
const ENTROPY_LEN: usize = 1_024;
const LARGE_BATCH: usize = 20_000;

type ParamsMutation = (&'static str, fn(&mut CudaJpegBaselineEncodeParams));

struct TestTables {
    q_luma: [u8; 64],
    q_chroma: [u8; 64],
    huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable,
    huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable,
    huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable,
    huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable,
}

impl Default for TestTables {
    fn default() -> Self {
        Self {
            q_luma: [1; 64],
            q_chroma: [1; 64],
            huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
            huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
            huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
            huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
        }
    }
}

impl TestTables {
    fn refs(&self) -> CudaJpegBaselineEncodeTableRefs<'_> {
        CudaJpegBaselineEncodeTableRefs {
            q_luma: &self.q_luma,
            q_chroma: &self.q_chroma,
            huff_dc_luma: &self.huff_dc_luma,
            huff_ac_luma: &self.huff_ac_luma,
            huff_dc_chroma: &self.huff_dc_chroma,
            huff_ac_chroma: &self.huff_ac_chroma,
        }
    }
}

fn valid_rgb444() -> CudaJpegBaselineEncodeParams {
    CudaJpegBaselineEncodeParams {
        input_offset_bytes: 16,
        input_width: 17,
        input_height: 9,
        output_width: 17,
        output_height: 9,
        pitch_bytes: 64,
        mcus_per_row: 3,
        mcu_rows: 2,
        restart_interval_mcus: 0,
        format: CudaJpegBaselineEncodeFormat::Rgb8.abi(),
        components: 3,
        max_h: 1,
        max_v: 1,
        h0: 1,
        v0: 1,
        h1: 1,
        v1: 1,
        h2: 1,
        v2: 1,
        entropy_offset_bytes: 32,
        entropy_capacity: 256,
    }
}

fn valid_gray() -> CudaJpegBaselineEncodeParams {
    CudaJpegBaselineEncodeParams {
        format: CudaJpegBaselineEncodeFormat::Gray8.abi(),
        components: 1,
        pitch_bytes: 17,
        max_h: 1,
        max_v: 1,
        h0: 1,
        v0: 1,
        h1: 0,
        v1: 0,
        h2: 0,
        v2: 0,
        ..valid_rgb444()
    }
}

fn validate(
    params: &[CudaJpegBaselineEncodeParams],
    input_len: usize,
    bound_input_offset: usize,
    entropy_len: usize,
    tables: &TestTables,
) -> Result<CudaJpegBaselineEncodeValidation, CudaError> {
    validate_jpeg_baseline_encode_request(
        0,
        input_len,
        bound_input_offset,
        params,
        entropy_len,
        tables.refs(),
        0,
    )
}

fn invalid_message(result: Result<CudaJpegBaselineEncodeValidation, CudaError>) -> String {
    match result {
        Err(CudaError::InvalidArgument { message }) => message,
        Err(error) => panic!("expected invalid argument, got {error}"),
        Ok(validation) => panic!("expected invalid argument, got {validation:?}"),
    }
}

#[test]
fn accepts_supported_sampling_shapes_and_retains_single_offsets() {
    let tables = TestTables::default();
    let rgb444 = valid_rgb444();
    let validation =
        validate(&[rgb444], INPUT_LEN, 8, ENTROPY_LEN, &tables).expect("valid RGB 4:4:4 request");
    assert_eq!(validation.tile_count, 1);
    assert_eq!(validation.first_tile.input_ptr, 24);
    assert_eq!(validation.first_tile.entropy_offset, 32);
    assert_eq!(validation.first_tile.entropy_capacity, 256);

    let gray = valid_gray();
    assert!(validate(&[gray], INPUT_LEN, 0, ENTROPY_LEN, &tables).is_ok());

    let mut rgb422 = valid_rgb444();
    (rgb422.max_h, rgb422.h0, rgb422.mcus_per_row) = (2, 2, 2);
    assert!(validate(&[rgb422], INPUT_LEN, 0, ENTROPY_LEN, &tables).is_ok());

    let mut rgb420 = rgb422;
    (rgb420.max_v, rgb420.v0, rgb420.mcu_rows) = (2, 2, 1);
    assert!(validate(&[rgb420], INPUT_LEN, 0, ENTROPY_LEN, &tables).is_ok());
}

#[test]
fn rejects_empty_requests_and_each_dimension_or_geometry_field() {
    let tables = TestTables::default();
    assert!(
        invalid_message(validate(&[], INPUT_LEN, 0, ENTROPY_LEN, &tables)).contains("at least one")
    );

    let mutations: &[ParamsMutation] = &[
        ("input_width", |params| params.input_width = 0),
        ("input_height", |params| params.input_height = 0),
        ("output_width", |params| params.output_width = 0),
        ("output_height", |params| params.output_height = 0),
        ("input_exceeds_width", |params| params.input_width = 18),
        ("input_exceeds_height", |params| params.input_height = 10),
        ("marker_width", |params| params.output_width = 65_536),
        ("marker_height", |params| params.output_height = 65_536),
        ("restart_interval", |params| {
            params.restart_interval_mcus = 65_536;
        }),
        ("mcus_per_row", |params| params.mcus_per_row = 2),
        ("mcu_rows", |params| params.mcu_rows = 1),
    ];
    for (field, mutate) in mutations {
        let mut params = valid_rgb444();
        mutate(&mut params);
        assert!(
            validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &tables).is_err(),
            "invalid {field} must fail validation"
        );
    }
}

#[test]
fn rejects_each_format_component_and_sampling_field() {
    let tables = TestTables::default();
    let mutations: &[ParamsMutation] = &[
        ("format", |params| params.format = 7),
        ("components", |params| params.components = 2),
        ("max_h", |params| params.max_h = 2),
        ("max_v", |params| params.max_v = 2),
        ("h0", |params| params.h0 = 2),
        ("v0", |params| params.v0 = 2),
        ("h1", |params| params.h1 = 2),
        ("v1", |params| params.v1 = 2),
        ("h2", |params| params.h2 = 2),
        ("v2", |params| params.v2 = 2),
    ];
    for (field, mutate) in mutations {
        let mut params = valid_rgb444();
        mutate(&mut params);
        let message = invalid_message(validate(&[params], INPUT_LEN, 0, ENTROPY_LEN, &tables));
        assert!(message.contains("sampling"), "invalid {field}: {message}");
    }
}

#[test]
fn rejects_pitch_and_every_input_range_failure_before_launch() {
    let tables = TestTables::default();
    let mut short_pitch = valid_rgb444();
    short_pitch.pitch_bytes = 50;
    assert!(
        invalid_message(validate(&[short_pitch], INPUT_LEN, 0, ENTROPY_LEN, &tables))
            .contains("smaller than row byte count")
    );

    let mut row_product_overflow = valid_rgb444();
    row_product_overflow.input_height = 3;
    row_product_overflow.output_height = 3;
    row_product_overflow.pitch_bytes = u32::MAX;
    row_product_overflow.mcu_rows = 1;
    assert!(invalid_message(validate(
        &[row_product_overflow],
        usize::MAX,
        0,
        ENTROPY_LEN,
        &tables,
    ))
    .contains("last input-row"));

    let mut linear_index_overflow = valid_rgb444();
    linear_index_overflow.input_height = 2;
    linear_index_overflow.output_height = 2;
    linear_index_overflow.pitch_bytes = u32::MAX;
    linear_index_overflow.mcu_rows = 1;
    assert!(invalid_message(validate(
        &[linear_index_overflow],
        usize::MAX,
        0,
        ENTROPY_LEN,
        &tables,
    ))
    .contains("row footprint"));

    let params = valid_rgb444();
    assert!(
        invalid_message(validate(&[params], 578, 0, ENTROPY_LEN, &tables))
            .contains("beyond allocation")
    );
    assert!(invalid_message(validate(
        &[params],
        usize::MAX,
        usize::MAX,
        ENTROPY_LEN,
        &tables,
    ))
    .contains("input offset overflows"));

    let pointer_overflow = validate_jpeg_baseline_encode_request(
        u64::MAX - 8,
        INPUT_LEN,
        0,
        &[params],
        ENTROPY_LEN,
        tables.refs(),
        0,
    );
    assert!(invalid_message(pointer_overflow).contains("device input pointer"));
}

#[test]
fn accepts_adjacent_entropy_ranges_and_rejects_aliases_by_original_index() {
    let tables = TestTables::default();
    let mut first = valid_rgb444();
    first.entropy_offset_bytes = 500;
    first.entropy_capacity = 100;
    let mut second = valid_rgb444();
    second.entropy_offset_bytes = 400;
    second.entropy_capacity = 100;
    assert!(validate(&[first, second], INPUT_LEN, 0, ENTROPY_LEN, &tables).is_ok());

    second.entropy_capacity = 101;
    let message = invalid_message(validate(
        &[first, second],
        INPUT_LEN,
        0,
        ENTROPY_LEN,
        &tables,
    ));
    assert!(message.contains("tiles 1 and 0 overlap"), "{message}");
}

#[test]
fn large_entropy_range_sweep_finds_overlap_without_quadratic_pair_scanning() {
    let tables = TestTables::default();
    let mut params = Vec::new();
    params
        .try_reserve_exact(LARGE_BATCH + 1)
        .expect("reserve adversarial batch");
    for index in 0..LARGE_BATCH {
        let mut tile = valid_rgb444();
        tile.entropy_offset_bytes = u32::try_from(index * 2).expect("test offset fits u32");
        tile.entropy_capacity = 1;
        params.push(tile);
    }
    assert!(validate(&params, INPUT_LEN, 0, LARGE_BATCH * 2, &tables,).is_ok());

    let mut hidden_overlap = valid_rgb444();
    hidden_overlap.entropy_offset_bytes =
        u32::try_from((LARGE_BATCH - 1) * 2).expect("test offset fits u32");
    hidden_overlap.entropy_capacity = 1;
    params.push(hidden_overlap);
    let message = invalid_message(validate(&params, INPUT_LEN, 0, LARGE_BATCH * 2, &tables));
    assert!(message.contains("overlap"), "{message}");
}

#[test]
fn later_invalid_batch_tile_fails_the_whole_preflight() {
    let tables = TestTables::default();
    let mut first = valid_rgb444();
    first.entropy_offset_bytes = 0;
    let mut later = valid_rgb444();
    later.input_offset_bytes = 900;
    later.entropy_offset_bytes = 256;
    let message = invalid_message(validate(
        &[first, later],
        INPUT_LEN,
        0,
        ENTROPY_LEN,
        &tables,
    ));
    assert!(message.contains("tile 1"), "{message}");
    assert!(message.contains("beyond allocation"), "{message}");
}
