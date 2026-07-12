// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{validate_jpeg_buffer_context, validate_jpeg_context_matches, JPEG_CONTEXT_MISMATCH};
use crate::{
    CudaContext, CudaDeviceBuffer, CudaError, CudaJpegBaselineEncodeFormat,
    CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob,
    CudaJpegEntropyCheckpoint,
};

mod decode_security;

use decode_security::decode_plan;

fn assert_jpeg_context_mismatch<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, JPEG_CONTEXT_MISMATCH);
        }
        Err(error) => panic!("expected a JPEG CUDA context mismatch, got {error}"),
        Ok(_) => panic!("expected a JPEG CUDA context mismatch"),
    }
}

fn encode_job(input: &CudaDeviceBuffer) -> CudaJpegBaselineEntropyEncodeJob<'_> {
    CudaJpegBaselineEntropyEncodeJob {
        input,
        input_offset: 0,
        params: CudaJpegBaselineEncodeParams::default(),
        q_luma: [0; 64],
        q_chroma: [0; 64],
        huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
        entropy_capacity: 0,
    }
}

fn encode_batch_job(
    input: &CudaDeviceBuffer,
    params: Vec<CudaJpegBaselineEncodeParams>,
) -> CudaJpegBaselineEntropyEncodeBatchJob<'_> {
    CudaJpegBaselineEntropyEncodeBatchJob {
        input,
        params,
        q_luma: [0; 64],
        q_chroma: [0; 64],
        huff_dc_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_ac_luma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_dc_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
        huff_ac_chroma: CudaJpegBaselineEncodeHuffmanTable::default(),
        entropy_capacity: 0,
    }
}

fn valid_encode_params() -> CudaJpegBaselineEncodeParams {
    CudaJpegBaselineEncodeParams {
        input_offset_bytes: 0,
        input_width: 8,
        input_height: 8,
        output_width: 8,
        output_height: 8,
        pitch_bytes: 24,
        mcus_per_row: 1,
        mcu_rows: 1,
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
        entropy_offset_bytes: 0,
        entropy_capacity: 256,
    }
}

fn assert_jpeg_invalid_argument<T>(result: Result<T, CudaError>, expected: &str) {
    match result {
        Err(CudaError::InvalidArgument { message }) => assert!(
            message.contains(expected),
            "expected {expected:?} in CUDA JPEG validation error, got {message:?}"
        ),
        Err(error) => panic!("expected a CUDA JPEG invalid argument, got {error}"),
        Ok(_) => panic!("expected a CUDA JPEG invalid argument"),
    }
}

#[test]
fn jpeg_context_validation_accepts_empty_and_matching_inputs() {
    assert!(validate_jpeg_context_matches([]).is_ok());
    assert!(validate_jpeg_context_matches([true, true, true]).is_ok());
}

#[test]
fn jpeg_context_validation_rejects_each_mismatched_input_position() {
    for matches in [
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ] {
        assert_jpeg_context_mismatch(validate_jpeg_context_matches(matches));
    }
}

#[test]
fn safe_jpeg_apis_reject_foreign_buffers_and_keep_empty_batch_noop() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
    let local_buffer = context.allocate(0).expect("local empty buffer");
    let foreign_buffer = foreign_context.allocate(0).expect("foreign empty buffer");
    assert!(validate_jpeg_buffer_context(&context, [&local_buffer]).is_ok());

    let checkpoints = [CudaJpegEntropyCheckpoint::default()];
    assert_jpeg_context_mismatch(context.decode_jpeg_rgb8_owned_into(
        &decode_plan(&checkpoints),
        &foreign_buffer,
        3,
    ));
    assert_jpeg_context_mismatch(
        context.encode_jpeg_baseline_entropy(&encode_job(&foreign_buffer)),
    );
    assert_jpeg_context_mismatch(
        context.encode_jpeg_baseline_entropy_batch(&encode_batch_job(
            &foreign_buffer,
            vec![CudaJpegBaselineEncodeParams::default()],
        )),
    );

    let empty = context
        .encode_jpeg_baseline_entropy_batch(&encode_batch_job(&foreign_buffer, Vec::new()))
        .expect("an empty JPEG encode batch remains a no-op");
    assert!(empty.is_empty());
}

#[test]
fn safe_jpeg_encode_apis_reject_invalid_ranges_before_kernel_launch() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let context = CudaContext::system_default().expect("launch CUDA context");
    let input = context.allocate(1_024).expect("allocate JPEG input");

    let mut single = encode_job(&input);
    single.params = valid_encode_params();
    single.input_offset = 900;
    single.q_luma = [1; 64];
    single.q_chroma = [1; 64];
    single.entropy_capacity = 256;
    assert_jpeg_invalid_argument(
        context.encode_jpeg_baseline_entropy(&single),
        "beyond allocation",
    );

    let mut first = valid_encode_params();
    first.entropy_offset_bytes = 0;
    let mut second = valid_encode_params();
    second.entropy_offset_bytes = 128;
    let mut batch = encode_batch_job(&input, vec![first, second]);
    batch.q_luma = [1; 64];
    batch.q_chroma = [1; 64];
    batch.entropy_capacity = 512;
    assert_jpeg_invalid_argument(
        context.encode_jpeg_baseline_entropy_batch(&batch),
        "entropy ranges for tiles 0 and 1 overlap",
    );
}
