use super::{
    checked_f32_words_byte_len, f32_slice_as_bytes_mut, format_idwt_batch_trace_row,
    idwt_batch_kernel_mode, idwt_batch_trace_row, idwt_batch_uses_cooperative_53,
    jpeg_entropy_overflow_count, pool_fit_buffer_index_by_len, validate_dct_block_grid,
    CudaContext, CudaDwt97BatchGeometry, CudaError, CudaExecutionStats,
    CudaExternalDeviceBufferViewMut, CudaHtj2k97CodeblockBatchWithPoolRequest,
    CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob,
    CudaHtj2kDecodeTables, CudaHtj2kDequantizeTarget, CudaHtj2kEncodeCodeBlockJob,
    CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResidentTarget, CudaHtj2kEncodeTables,
    CudaJ2kIdwtBatchKernelMode, CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget,
    CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob, CudaJ2kRect, CudaJpegChunkedEntropyConfig,
    CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable, CudaKernelName, CudaQueuedHtj2kCleanup,
};
#[cfg(feature = "cuda-oxide-j2k-ml")]
use super::{CudaJ2kMlKernelConfig, CudaJ2kMlLayout, CudaJ2kMlNormalization, CudaJ2kMlSample};

fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
}

#[test]
fn cuda_context_identity_distinguishes_clones_from_independent_contexts_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let cloned = context.clone();
    let independent = CudaContext::system_default().expect("independent CUDA context");

    assert!(context.is_same_context(&cloned));
    assert!(!context.is_same_context(&independent));
}

#[test]
fn retained_primary_context_identity_and_release_are_balanced_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let first = CudaContext::retain_primary(0).expect("retain primary context");
    let second = CudaContext::retain_primary(0).expect("retain primary context again");
    let owned = CudaContext::system_default().expect("independent owned context");

    assert!(first.is_same_context(&second));
    assert!(!first.is_same_context(&owned));
    assert_eq!(first.device_ordinal(), 0);

    drop(first);
    drop(second);
    let retained_again = CudaContext::retain_primary(0).expect("retain primary after release");
    assert_eq!(retained_again.device_ordinal(), 0);
}

#[test]
fn external_cuda_view_rejects_foreign_context_and_never_owns_memory_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let foreign = CudaContext::system_default().expect("foreign CUDA context");
    let mut allocation = context.allocate(16).expect("device allocation");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();

    // SAFETY: `allocation` owns the live range and is exclusively borrowed by
    // the view for the duration of this scope.
    let view = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 4, &mut allocation)
    }
    .expect("external view");
    assert_eq!(view.device_ptr(), ptr);
    assert_eq!(view.byte_len(), 16);
    drop(view);

    // The external view has no ownership: the original allocation remains
    // live and is still responsible for freeing its memory.
    assert_eq!(allocation.device_ptr(), ptr);
    assert_eq!(allocation.byte_len(), 16);

    // SAFETY: the allocation remains live and exclusively borrowed, but the
    // deliberately foreign context must reject its pointer identity.
    let error = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&foreign, ptr, len, 4, &mut allocation)
    }
    .expect_err("foreign context must fail");
    assert!(matches!(error, CudaError::InvalidArgument { .. }));
}

#[cfg(feature = "cuda-oxide-j2k-ml")]
#[test]
fn j2k_ml_external_destination_checks_batch_offsets_before_launch_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let source = context.upload(&[1, 2, 3, 4]).expect("source upload");
    let mut allocation = context.allocate(4).expect("destination allocation");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: `allocation` owns this live four-byte range and the view holds
    // its exclusive borrow until validation returns.
    let mut destination = unsafe {
        CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 1, &mut allocation)
    }
    .expect("external view");

    let error = context
        .j2k_ml_convert_into_external(
            source.device_ptr(),
            source.byte_len(),
            &mut destination,
            CudaJ2kMlKernelConfig {
                width: 2,
                height: 2,
                channels: 1,
                sample: CudaJ2kMlSample::U8,
                layout: CudaJ2kMlLayout::ChannelsFirst,
                destination_offset_elements: 1,
                normalization: CudaJ2kMlNormalization::Integer,
            },
        )
        .expect_err("offset must exceed destination bounds");
    assert!(matches!(error, CudaError::OutputTooSmall { .. }));
    drop(destination);
    assert_eq!(allocation.device_ptr(), ptr);
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn cuda_transcode_kernel_gate() -> bool {
    if super::transcode_kernels_built() {
        return true;
    }
    assert!(
        !j2k_test_support::cuda_strict_oxide_required(),
        "J2K_REQUIRE_CUDA_OXIDE_BUILD is set but transcode kernels were not built"
    );
    eprintln!(
        "{} gate=J2K_REQUIRE_CUDA_OXIDE_BUILD context={} reason=transcode-kernels-not-built",
        j2k_test_support::GPU_TEST_SKIP_MARKER,
        module_path!()
    );
    false
}

mod pipeline;

#[test]
fn jpeg_chunked_entropy_config_counts_bit_subsequences() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 4,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };

    assert_eq!(config.subsequence_bits(), 128);
    assert_eq!(config.subsequence_count_for_entropy_bytes(0).unwrap(), 0);
    assert_eq!(config.subsequence_count_for_entropy_bytes(1).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(16).unwrap(), 1);
    assert_eq!(config.subsequence_count_for_entropy_bytes(17).unwrap(), 2);
}

#[test]
fn checked_f32_words_byte_len_rejects_multiplication_overflow() {
    assert_eq!(checked_f32_words_byte_len(2).expect("byte len"), 8);
    assert!(matches!(
        checked_f32_words_byte_len(usize::MAX),
        Err(CudaError::LengthTooLarge { len }) if len == usize::MAX
    ));
}

#[test]
fn validate_dct_block_grid_checks_shape_and_coefficient_count() {
    let grid = validate_dct_block_grid(2, 1, 15, 8, 3, 384, "invalid").expect("valid grid");

    assert_eq!(grid.block_count, 2);
    assert_eq!(grid.expected_coeffs, 384);
    assert_eq!((grid.low_width, grid.high_width), (8, 7));
    assert_eq!((grid.low_height, grid.high_height), (4, 4));
    assert!(matches!(
        validate_dct_block_grid(2, 1, 15, 8, 3, 383, "invalid"),
        Err(CudaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        validate_dct_block_grid(2, 1, 15, 8, 0, 0, "invalid"),
        Err(CudaError::InvalidArgument { .. })
    ));
    assert!(matches!(
        validate_dct_block_grid(usize::MAX, 2, 1, 1, 1, 64, "invalid"),
        Err(CudaError::LengthTooLarge { .. })
    ));
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[test]
fn cuda_oxide_reversible53_transcode_matches_scalar_fixture_when_required() {
    if !cuda_runtime_gate() || !cuda_transcode_kernel_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut blocks = [0i16; 64];
    for (index, value) in [
        (0, 80),
        (1, -24),
        (2, 13),
        (3, 5),
        (5, -3),
        (8, 31),
        (9, -11),
        (10, 7),
        (16, -9),
        (17, 4),
        (18, 3),
        (27, -5),
        (36, 6),
        (45, -4),
        (54, 2),
        (63, -1),
    ] {
        blocks[index] = value;
    }

    let bands = context
        .j2k_transcode_reversible_dwt53(&blocks, 1, 1, 8, 8)
        .expect("cuda-oxide reversible 5/3 transcode");

    assert_eq!((bands.low_width, bands.low_height), (4, 4));
    assert_eq!((bands.high_width, bands.high_height), (4, 4));
    assert_eq!(
        bands.ll.as_slice(),
        &[14, 8, 12, 22, 13, 7, 14, 22, 8, 7, 12, 15, 6, 3, 5, 7]
    );
    assert_eq!(
        bands.hl.as_slice(),
        &[2, -1, -1, 5, 1, -4, 2, 0, -1, 1, 0, 3, 3, -3, 2, 0]
    );
    assert_eq!(
        bands.lh.as_slice(),
        &[2, 1, -1, 2, 2, -1, 3, 0, -1, 3, -1, 1, 1, -4, -1, -2]
    );
    assert_eq!(
        bands.hh.as_slice(),
        &[1, 2, -1, -4, 1, -1, 0, 1, -1, -1, 1, -2, -5, 2, -1, -1]
    );
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[test]
fn cuda_oxide_dwt97_transcode_matches_scalar_fixture_when_required() {
    if !cuda_runtime_gate() || !cuda_transcode_kernel_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut blocks = [0.0f32; 64];
    for (index, value) in [
        (0, 80.0),
        (1, -24.0),
        (2, 13.0),
        (3, 5.0),
        (5, -3.0),
        (8, 31.0),
        (9, -11.0),
        (10, 7.0),
        (16, -9.0),
        (17, 4.0),
        (18, 3.0),
        (27, -5.0),
        (36, 6.0),
        (45, -4.0),
        (54, 2.0),
        (63, -1.0),
    ] {
        blocks[index] = value;
    }

    let bands = context
        .j2k_transcode_dwt97(&blocks, 1, 1, 8, 8)
        .expect("cuda-oxide 9/7 transcode");

    assert_eq!((bands.low_width, bands.low_height), (4, 4));
    assert_eq!((bands.high_width, bands.high_height), (4, 4));
    assert_f32_slice_close(
        &bands.ll,
        &[
            12.144_072, 8.567_899, 11.216_426, 20.388_594, 11.476_019, 7.618_125, 12.952_319,
            19.958_328, 7.468_019, 6.779_34, 10.701_953, 14.315_73, 4.983_001, 3.069_523,
            4.546_064, 6.695_241,
        ],
        0.02,
    );
    assert_f32_slice_close(
        &bands.hl,
        &[
            0.579_117, -0.765_21, -1.113_766, 3.008_691, 1.415_966, -2.878_618, 2.173_036,
            -0.629_188, -0.239_748, 0.239_237, -0.885_278, 2.500_556, 1.929_175, -2.255_519,
            1.123_41, 0.191_912,
        ],
        0.02,
    );
    assert_f32_slice_close(
        &bands.lh,
        &[
            -0.314_113, 0.534_82, -1.107_942, 1.062_559, 0.976_02, -1.180_377, 1.861_77,
            -0.696_248, -1.241_956, 2.006_542, -1.112_403, 0.853_18, 0.104_077, -3.326_791,
            0.079_872, -2.094_714,
        ],
        0.02,
    );
    assert_f32_slice_close(
        &bands.hh,
        &[
            -0.434_17, 1.497_277, -0.967_611, -6.657_543, 1.496_545, -1.963_292, -2.252_154,
            3.941_389, -0.968_106, -2.252_748, 1.867_451, -1.252_69, -6.656_182, 3.949_171,
            -1.248_663, 0.544_539,
        ],
        0.02,
    );
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "driver symbol inventory is one fail-closed runtime contract"
)]
fn cuda_oxide_dwt97_batch_and_quantize_paths_match_reference_when_required() {
    const WIDE_PATTERN: [f32; 17] = [
        -2.0, -1.75, -1.25, -0.5, 0.0, 0.25, 0.75, 1.0, 1.5, 2.0, -2.5, 2.5, -3.0, 3.0, -0.25, 0.5,
        1.25,
    ];

    if !cuda_runtime_gate() || !cuda_transcode_kernel_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let first = dwt97_fixture_blocks(1.0);
    let second = dwt97_fixture_blocks(-1.0);
    let mut blocks = Vec::with_capacity(128);
    blocks.extend_from_slice(&first);
    blocks.extend_from_slice(&second);

    let expected_first = context
        .j2k_transcode_dwt97(&first, 1, 1, 8, 8)
        .expect("single first DWT97");
    let expected_second = context
        .j2k_transcode_dwt97(&second, 1, 1, 8, 8)
        .expect("single second DWT97");
    let (batch, _) = context
        .j2k_transcode_dwt97_batch_with_pool(super::CudaDwt97BatchWithPoolRequest {
            blocks: &blocks,
            geometry: CudaDwt97BatchGeometry {
                item_count: 2,
                block_cols: 1,
                block_rows: 1,
                width: 8,
                height: 8,
            },
            pool: &pool,
        })
        .expect("cuda-oxide DWT97 batch");
    assert_eq!(batch.len(), 2);
    assert_dwt97_bands_close(&batch[0], &expected_first, 0.02);
    assert_dwt97_bands_close(&batch[1], &expected_second, 0.02);

    let wide_block_cols = 129;
    let wide_width = 1032;
    let wide_height = 8;
    let mut wide_blocks = vec![0.0f32; wide_block_cols * 64];
    for (index, value) in wide_blocks.iter_mut().enumerate() {
        *value = WIDE_PATTERN[index % WIDE_PATTERN.len()];
    }
    let wide_expected = context
        .j2k_transcode_dwt97(&wide_blocks, wide_block_cols, 1, wide_width, wide_height)
        .expect("single wide DWT97");
    let (wide_batch, _) = context
        .j2k_transcode_dwt97_batch_with_pool(super::CudaDwt97BatchWithPoolRequest {
            blocks: &wide_blocks,
            geometry: CudaDwt97BatchGeometry {
                item_count: 1,
                block_cols: wide_block_cols,
                block_rows: 1,
                width: wide_width,
                height: wide_height,
            },
            pool: &pool,
        })
        .expect("wide cuda-oxide DWT97 batch");
    assert_eq!(wide_batch.len(), 1);
    assert_dwt97_bands_close(&wide_batch[0], &wide_expected, 0.02);

    let params = super::CudaHtj2k97QuantizeParams {
        inv_delta_ll: 1.0,
        inv_delta_hl: 1.25,
        inv_delta_lh: 0.75,
        inv_delta_hh: 2.0,
        cb_width: 64,
        cb_height: 64,
    };
    let expected_codeblocks = expected_dwt97_codeblocks(&batch, params);
    let (quantized, _) = context
        .j2k_transcode_htj2k97_codeblock_batch_with_pool(CudaHtj2k97CodeblockBatchWithPoolRequest {
            blocks: &blocks,
            geometry: CudaDwt97BatchGeometry {
                item_count: 2,
                block_cols: 1,
                block_rows: 1,
                width: 8,
                height: 8,
            },
            params,
            pool: &pool,
        })
        .expect("cuda-oxide staged DWT97 quantize batch");
    assert_eq!(quantized, expected_codeblocks);

    let first_i16 = dwt97_fixture_i16_blocks(1);
    let second_i16 = dwt97_fixture_i16_blocks(-1);
    let mut i16_blocks = Vec::with_capacity(128);
    i16_blocks.extend_from_slice(&first_i16);
    i16_blocks.extend_from_slice(&second_i16);
    let (fused, _) = context
        .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
            super::CudaHtj2k97I16CodeblockBatchWithPoolRequest {
                blocks: &i16_blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: 2,
                    block_cols: 1,
                    block_rows: 1,
                    width: 8,
                    height: 8,
                },
                params,
                pool: &pool,
            },
        )
        .expect("cuda-oxide fused i16 DWT97 quantize batch");
    assert_eq!(download_device_codeblock_bands(&fused), expected_codeblocks);
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn assert_f32_slice_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn assert_dwt97_bands_close(
    actual: &super::CudaTranscodeDwt97Bands,
    expected: &super::CudaTranscodeDwt97Bands,
    tolerance: f32,
) {
    assert_eq!(
        (
            actual.low_width,
            actual.low_height,
            actual.high_width,
            actual.high_height,
        ),
        (
            expected.low_width,
            expected.low_height,
            expected.high_width,
            expected.high_height,
        )
    );
    assert_f32_slice_close(&actual.ll, &expected.ll, tolerance);
    assert_f32_slice_close(&actual.hl, &expected.hl, tolerance);
    assert_f32_slice_close(&actual.lh, &expected.lh, tolerance);
    assert_f32_slice_close(&actual.hh, &expected.hh, tolerance);
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn dwt97_fixture_blocks(scale: f32) -> [f32; 64] {
    let mut blocks = [0.0f32; 64];
    for (index, value) in DWT97_FIXTURE_VALUES {
        blocks[index] = f32::from(value) * scale;
    }
    blocks
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn dwt97_fixture_i16_blocks(scale: i16) -> [i16; 64] {
    let mut blocks = [0i16; 64];
    for (index, value) in DWT97_FIXTURE_VALUES {
        blocks[index] = value * scale;
    }
    blocks
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
const DWT97_FIXTURE_VALUES: [(usize, i16); 16] = [
    (0, 80),
    (1, -24),
    (2, 13),
    (3, 5),
    (5, -3),
    (8, 31),
    (9, -11),
    (10, 7),
    (16, -9),
    (17, 4),
    (18, 3),
    (27, -5),
    (36, 6),
    (45, -4),
    (54, 2),
    (63, -1),
];

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn expected_dwt97_codeblocks(
    batch: &[super::CudaTranscodeDwt97Bands],
    params: super::CudaHtj2k97QuantizeParams,
) -> super::CudaHtj2k97CodeblockBands {
    let first = batch.first().expect("non-empty DWT97 batch");
    let mut ll = Vec::new();
    let mut hl = Vec::new();
    let mut lh = Vec::new();
    let mut hh = Vec::new();
    for bands in batch {
        ll.extend(
            bands
                .ll
                .iter()
                .map(|&value| quantize_dwt97_deadzone(value, params.inv_delta_ll)),
        );
        hl.extend(
            bands
                .hl
                .iter()
                .map(|&value| quantize_dwt97_deadzone(value, params.inv_delta_hl)),
        );
        lh.extend(
            bands
                .lh
                .iter()
                .map(|&value| quantize_dwt97_deadzone(value, params.inv_delta_lh)),
        );
        hh.extend(
            bands
                .hh
                .iter()
                .map(|&value| quantize_dwt97_deadzone(value, params.inv_delta_hh)),
        );
    }
    super::CudaHtj2k97CodeblockBands {
        ll,
        hl,
        lh,
        hh,
        item_count: batch.len(),
        low_width: first.low_width,
        low_height: first.low_height,
        high_width: first.high_width,
        high_height: first.high_height,
    }
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
#[expect(
    clippy::cast_possible_truncation,
    reason = "test mirrors CUDA deadzone quantization for bounded fixture coefficients"
)]
fn quantize_dwt97_deadzone(value: f32, inv_delta: f32) -> i32 {
    let sign = if value < 0.0 { -1 } else { 1 };
    sign * (value.abs() * inv_delta).floor() as i32
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn download_device_codeblock_bands(
    bands: &super::CudaHtj2k97DeviceCodeblockBands,
) -> super::CudaHtj2k97CodeblockBands {
    let low_low_len = bands.item_count * bands.low_width * bands.low_height;
    let high_low_len = bands.item_count * bands.high_width * bands.low_height;
    let low_high_len = bands.item_count * bands.low_width * bands.high_height;
    let high_high_len = bands.item_count * bands.high_width * bands.high_height;
    super::CudaHtj2k97CodeblockBands {
        ll: download_pooled_i32(&bands.ll, low_low_len),
        hl: download_pooled_i32(&bands.hl, high_low_len),
        lh: download_pooled_i32(&bands.lh, low_high_len),
        hh: download_pooled_i32(&bands.hh, high_high_len),
        item_count: bands.item_count,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    }
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn download_pooled_i32(buffer: &super::CudaPooledDeviceBuffer, len: usize) -> Vec<i32> {
    let mut output = vec![0i32; len];
    buffer
        .copy_to_host(super::i32_slice_as_bytes_mut(&mut output))
        .expect("download pooled i32 buffer");
    output
}

#[test]
fn jpeg_chunked_entropy_report_has_one_less_overflow_than_subsequence_count() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: 1,
        sequence_len: 8,
        max_overflow_subsequences: 2,
    };
    let subsequences = config.subsequence_count_for_entropy_bytes(16).unwrap();

    assert_eq!(subsequences, 4);
    assert_eq!(jpeg_entropy_overflow_count(subsequences), 3);
    assert_eq!(jpeg_entropy_overflow_count(0), 0);
}

#[test]
fn jpeg_chunked_entropy_config_rejects_zero_subsequence_or_sequence() {
    let zero_words = CudaJpegChunkedEntropyConfig {
        subsequence_words: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };
    let zero_sequence = CudaJpegChunkedEntropyConfig {
        sequence_len: 0,
        ..CudaJpegChunkedEntropyConfig::default()
    };

    assert!(zero_words.validate().is_err());
    assert!(zero_sequence.validate().is_err());
}

#[test]
fn jpeg_chunked_entropy_config_rejects_subsequence_bit_overflow() {
    let config = CudaJpegChunkedEntropyConfig {
        subsequence_words: (u32::MAX / 32) + 1,
        ..CudaJpegChunkedEntropyConfig::default()
    };

    assert!(config.validate().is_err());
    assert!(config.subsequence_count_for_entropy_bytes(1).is_err());
}

#[test]
fn jpeg_chunked_entropy_report_summarizes_sync_quality() {
    let report = CudaJpegChunkedEntropyReport {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 4,
            sequence_len: 8,
            max_overflow_subsequences: 2,
        },
        entropy_bytes: 4096,
        states: vec![
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 0,
                end_bit: 128,
                bit_pos: 128,
                symbol_count: 10,
                block_phase: 0,
                zigzag_index: 0,
                reserved: 0,
            },
            CudaJpegEntropySyncState {
                code: 0,
                start_bit: 128,
                end_bit: 256,
                bit_pos: 256,
                symbol_count: 9,
                block_phase: 3,
                zigzag_index: 12,
                reserved: 0,
            },
        ],
        overflows: vec![CudaJpegEntropyOverflowState {
            code: 0,
            from_subsequence: 0,
            to_subsequence: 1,
            overflow_bits: 96,
            synchronized: 1,
            reserved: [0; 3],
        }],
        execution: CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 0,
            hardware_decode: false,
        },
    };

    assert_eq!(report.subsequence_count(), 2);
    assert_eq!(report.synchronized_overflow_count(), 1);
    assert_eq!(report.max_overflow_bits(), Some(96));
    assert_eq!(report.failed_state_count(), 0);
}

#[test]
fn jpeg_entropy_self_sync_returns_empty_report_for_empty_entropy_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("cuda context");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig::default(),
        entropy_bytes: &[],
        y_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        y_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cb_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_dc_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
        cr_ac_table: CudaJpegHuffmanTable::from_jpeg_bits_values([0; 16], 0, [0; 256])
            .expect("empty huffman table"),
    };

    let report = context
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("empty diagnostic report");
    assert_eq!(report.subsequence_count(), 0);
    assert_eq!(report.overflows.len(), 0);
}

#[cfg(all(
    feature = "cuda-oxide-jpeg-decode",
    not(j2k_cuda_oxide_jpeg_decode_built)
))]
#[test]
fn cuda_oxide_jpeg_decode_missing_build_error_mentions_strict_gate() {
    let error = super::build_flags::ensure_cuda_oxide_jpeg_decode_ptx_built()
        .expect_err("missing JPEG Oxide PTX should be reported");
    let message = error.to_string();
    assert!(message.contains("cuda-oxide JPEG decode PTX was not built"));
    assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
}

#[cfg(all(
    feature = "cuda-oxide-htj2k-decode",
    not(j2k_cuda_oxide_htj2k_decode_built)
))]
#[test]
fn cuda_oxide_htj2k_decode_missing_build_error_mentions_strict_gate() {
    let error = super::build_flags::ensure_cuda_oxide_htj2k_decode_ptx_built()
        .expect_err("missing HTJ2K Oxide PTX should be reported");
    let message = error.to_string();
    assert!(message.contains("cuda-oxide HTJ2K decode PTX was not built"));
    assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
}

#[cfg(all(
    feature = "cuda-oxide-htj2k-encode",
    not(j2k_cuda_oxide_htj2k_encode_built)
))]
#[test]
fn cuda_oxide_htj2k_encode_missing_build_error_mentions_strict_gate() {
    let error = super::build_flags::ensure_cuda_oxide_htj2k_encode_ptx_built()
        .expect_err("missing HTJ2K encode Oxide PTX should be reported");
    let message = error.to_string();
    assert!(message.contains("cuda-oxide HTJ2K encode PTX was not built"));
    assert!(message.contains("J2K_REQUIRE_CUDA_OXIDE_BUILD"));
}

#[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
#[test]
fn cuda_oxide_jpeg_entropy_self_sync_decodes_zero_stream_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let mut bits = [0u8; 16];
    bits[0] = 1;
    let table = CudaJpegHuffmanTable::from_jpeg_bits_values(bits, 1, [0; 256]).expect("zero table");
    let entropy = [0u8; 2];
    let context = CudaContext::system_default().expect("cuda context");
    let plan = CudaJpegChunkedEntropyPlan {
        config: CudaJpegChunkedEntropyConfig {
            subsequence_words: 1,
            sequence_len: 8,
            max_overflow_subsequences: 1,
        },
        entropy_bytes: &entropy,
        y_dc_table: table,
        y_ac_table: table,
        cb_dc_table: table,
        cb_ac_table: table,
        cr_dc_table: table,
        cr_ac_table: table,
    };

    let report = context
        .diagnose_jpeg_420_entropy_self_sync(&plan)
        .expect("cuda-oxide JPEG entropy self-sync");
    assert_eq!(report.subsequence_count(), 1);
    assert_eq!(report.overflows.len(), 0);
    assert_eq!(report.execution.kernel_dispatches(), 1);
    assert_eq!(report.states[0].code, 0);
    assert_eq!(report.states[0].start_bit, 0);
    assert_eq!(report.states[0].end_bit, 16);
    assert_eq!(report.states[0].bit_pos, 16);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "kernel metadata inventory is one exact host/device parity contract"
)]
fn runtime_raii_primitives_smoke_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut pinned = context.pinned_host_buffer(16).expect("pinned host buffer");
    pinned.as_mut_slice().copy_from_slice(&[7u8; 16]);
    assert_eq!(pinned.as_slice(), &[7u8; 16]);
    let pinned_upload = context
        .upload_pinned(&[1u8, 2, 3, 4])
        .expect("pinned upload");
    let mut uploaded = [0u8; 4];
    pinned_upload
        .copy_to_host(&mut uploaded)
        .expect("download pinned upload");
    assert_eq!(uploaded, [1, 2, 3, 4]);
    let pinned_float_upload = context
        .upload_f32_pinned(&[1.25, -2.5])
        .expect("pinned f32 upload");
    let mut downloaded_float_values = [0.0f32; 2];
    pinned_float_upload
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut downloaded_float_values))
        .expect("download pinned f32 upload");
    assert!((downloaded_float_values[0] - 1.25).abs() < f32::EPSILON);
    assert!((downloaded_float_values[1] + 2.5).abs() < f32::EPSILON);
    let pinned_integer_upload = context
        .upload_i32_pinned(&[7, -11])
        .expect("pinned i32 upload");
    let mut downloaded_integer_values = [0i32; 2];
    pinned_integer_upload
        .copy_to_host(super::i32_slice_as_bytes_mut(
            &mut downloaded_integer_values,
        ))
        .expect("download pinned i32 upload");
    assert_eq!(downloaded_integer_values, [7, -11]);
    let ranged_upload = context
        .upload(&[9u8, 8, 7, 6, 5, 4])
        .expect("range-copy upload");
    let mut range = [0u8; 3];
    ranged_upload
        .copy_range_to_host(2, &mut range)
        .expect("copy device range");
    assert_eq!(range, [7, 6, 5]);
    let mut uninit_range = Vec::with_capacity(3);
    ranged_upload
        .copy_range_to_host_uninit(1, uninit_range.spare_capacity_mut())
        .expect("copy device range into spare capacity");
    // SAFETY: copy_range_to_host_uninit returned success after writing
    // exactly three bytes into the Vec spare capacity.
    unsafe {
        uninit_range.set_len(3);
    }
    assert_eq!(uninit_range, [8, 7, 6]);
    let pool = context.buffer_pool();
    let pooled_upload = pool.upload(&[3u8, 1, 4, 1]).expect("pooled upload");
    let pooled_output = super::copy_pooled_bytes_to_vec_uninit(&pooled_upload, 4)
        .expect("copy pooled bytes into spare capacity");
    assert_eq!(pooled_output, [3, 1, 4, 1]);

    let module = context
        .preload_kernel_module(CudaKernelName::CopyU8)
        .expect("preload copy kernel");
    assert_eq!(module.entrypoint(), "j2k_copy_u8");

    let stream = context.create_stream().expect("CUDA stream");
    let start = context.create_event().expect("start event");
    let end = context.create_event().expect("end event");
    start.record(&stream).expect("record start");
    end.record(&stream).expect("record end");
    end.synchronize().expect("synchronize event");
    let elapsed = super::CudaEvent::elapsed_time_us(&start, &end).expect("elapsed time");
    assert!(elapsed >= 0.0);

    let pool = context.buffer_pool();
    {
        let buffer = pool.take(32).expect("pooled buffer");
        assert!(buffer.device_ptr() != 0);
        assert_eq!(buffer.byte_len(), 32);
        assert!(buffer.allocation_byte_len() >= 32);
    }
    let cached_count = pool.cached_count().expect("cached count");
    assert_eq!(cached_count, 1);
    {
        let buffer = pool.take(16).expect("reused pooled buffer");
        assert_eq!(buffer.byte_len(), 16);
        assert!(buffer.allocation_byte_len() >= 32);
    }

    let samples = [1.25f32, -2.5, 3.75, 4.5];
    {
        let buffer = pool.upload_f32(&samples).expect("pooled f32 upload");
        assert_eq!(
            buffer.byte_len(),
            samples.len() * std::mem::size_of::<f32>()
        );
        let mut downloaded = vec![0.0f32; samples.len()];
        buffer
            .copy_to_host(f32_slice_as_bytes_mut(&mut downloaded))
            .expect("download pooled f32 upload");
        assert_eq!(downloaded, samples);
    }
    let i16_samples = [-12i16, 7, 19, -4];
    {
        let buffer = pool
            .upload_i16_pinned(&i16_samples)
            .expect("pooled pinned i16 upload");
        assert_eq!(
            buffer.byte_len(),
            i16_samples.len() * std::mem::size_of::<i16>()
        );
        let mut downloaded_bytes = vec![0u8; std::mem::size_of_val(&i16_samples)];
        buffer
            .copy_to_host(&mut downloaded_bytes)
            .expect("download pooled pinned i16 upload");
        let downloaded = downloaded_bytes
            .chunks_exact(std::mem::size_of::<i16>())
            .map(|chunk| i16::from_ne_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        assert_eq!(downloaded, i16_samples);
    }
    let cached_after_upload = pool.cached_count().expect("cached after upload");
    assert!(cached_after_upload >= cached_count);
}

#[test]
fn pooled_i16_pinned_upload_is_size_gated() {
    assert!(super::should_use_pinned_pooled_i16_upload(4 * 1024 * 1024));
    assert!(!super::should_use_pinned_pooled_i16_upload(
        4 * 1024 * 1024 + 1
    ));
}

#[test]
fn pooled_buffer_selection_uses_smallest_sufficient_fit() {
    let buffers = [(1usize, 32usize), (0, 64)];

    assert_eq!(
        pool_fit_buffer_index_by_len(buffers.iter().copied(), 16),
        Some(1)
    );
    let mut large_pool = (0..1024).map(|index| (index, 8usize)).collect::<Vec<_>>();
    large_pool[1022] = (1022, 32);
    large_pool[1023] = (1023, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(large_pool.iter().copied(), 16),
        Some(1022)
    );
    let mut recent_fit_pool = (0..4096).map(|index| (index, 8usize)).collect::<Vec<_>>();
    recent_fit_pool[4094] = (4094, 32);
    recent_fit_pool[4095] = (4095, 64);

    assert_eq!(
        pool_fit_buffer_index_by_len(recent_fit_pool.iter().copied(), 16),
        Some(4094)
    );
    let fallback_pool = (0..4096)
        .map(|index| match index.cmp(&3000) {
            std::cmp::Ordering::Less => (index, 8usize),
            std::cmp::Ordering::Equal => (index, 32),
            std::cmp::Ordering::Greater => (index, 64),
        })
        .collect::<Vec<_>>();

    assert_eq!(
        pool_fit_buffer_index_by_len(fallback_pool.iter().copied(), 16),
        Some(3000)
    );
}

#[test]
fn pooled_take_with_trace_reports_allocation_and_reuse_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let (fresh, fresh_trace) = pool.take_with_trace(32).expect("fresh traced take");

    assert_eq!(fresh.byte_len(), 32);
    assert_eq!(fresh_trace.requested_len, 32);
    assert_eq!(fresh_trace.free_count_before, 0);
    assert_eq!(fresh_trace.scanned_count, 0);
    assert!(!fresh_trace.reused);
    assert!(fresh_trace.allocation_byte_len >= 32);
    drop(fresh);

    let (reused, reuse_trace) = pool.take_with_trace(16).expect("reused traced take");

    assert_eq!(reused.byte_len(), 16);
    assert_eq!(reuse_trace.requested_len, 16);
    assert_eq!(reuse_trace.free_count_before, 1);
    assert_eq!(reuse_trace.scanned_count, 1);
    assert!(reuse_trace.reused);
    assert!(reuse_trace.allocation_byte_len >= 32);
}

#[test]
fn pooled_buffer_can_detach_and_recycle_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let raw = pool
        .take(32)
        .expect("pooled buffer")
        .into_device_buffer()
        .expect("detach pooled buffer");
    assert_eq!(pool.cached_count().expect("cached after detach"), 0);

    pool.recycle(raw).expect("explicit recycle");
    assert_eq!(pool.cached_count().expect("cached after recycle"), 1);

    let (_reused, trace) = pool.take_with_trace(16).expect("reused traced take");
    assert!(trace.reused);
    assert!(trace.allocation_byte_len >= 32);
}

#[test]
fn htj2k_encoded_codeblock_reports_segment_lengths_from_status() {
    let encoded = super::CudaHtj2kEncodedCodeBlock {
        data: vec![0u8; 10],
        status: super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            detail: 0,
            data_len: 10,
            number_of_coding_passes: 3,
            missing_bit_planes: 4,
            reserved0: 7,
            reserved1: 3,
            reserved2: 0,
        },
        execution: super::CudaExecutionStats::default(),
        stage_timings: super::CudaHtj2kEncodeStageTimings::default(),
    };

    assert_eq!(encoded.cleanup_length(), 7);
    assert_eq!(encoded.refinement_length(), 3);
}

fn htj2k_multi_input_compact_job(
    job: super::CudaHtj2kEncodeKernelJob,
) -> super::CudaHtj2kEncodeMultiInputKernelJob {
    super::CudaHtj2kEncodeMultiInputKernelJob {
        coefficient_ptr: 0x1000,
        coefficient_offset: job.coefficient_offset,
        coefficient_stride: job.coefficient_stride,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        output_offset: job.output_offset,
        output_capacity: job.output_capacity,
        target_coding_passes: job.target_coding_passes,
    }
}

fn assert_compact_jobs_match_for_single_and_multi_input(
    statuses: &[super::CudaHtj2kEncodeStatus],
    kernel_jobs: &[super::CudaHtj2kEncodeKernelJob],
) -> Result<(Vec<super::CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    let multi_input_jobs = kernel_jobs
        .iter()
        .copied()
        .map(htj2k_multi_input_compact_job)
        .collect::<Vec<_>>();
    let mut single_budget = super::allocation::HostPhaseBudget::new("test compact jobs");
    let mut multi_budget = super::allocation::HostPhaseBudget::new("test compact jobs");
    let single = super::htj2k_encode_compact_jobs(statuses, kernel_jobs, &mut single_budget);
    let multi = super::htj2k_encode_compact_jobs_multi_input(
        statuses,
        &multi_input_jobs,
        &mut multi_budget,
    );
    match (single, multi) {
        (Ok(single), Ok(multi)) => {
            assert_eq!(single, multi);
            Ok(single)
        }
        (Err(single), Err(multi)) => {
            assert_eq!(format!("{single:?}"), format!("{multi:?}"));
            Err(single)
        }
        (single, multi) => panic!(
            "single and multi-input compact planners diverged: single={single:?} multi={multi:?}"
        ),
    }
}

#[test]
fn htj2k_encode_compact_jobs_accept_empty_batches() {
    let (compact_jobs, compact_len) =
        assert_compact_jobs_match_for_single_and_multi_input(&[], &[]).expect("empty compact plan");

    assert!(compact_jobs.is_empty());
    assert_eq!(compact_len, 0);
}

#[test]
fn htj2k_encode_compact_jobs_pack_actual_payloads() {
    let capacity = u32::try_from(super::HTJ2K_ENCODE_OUTPUT_CAPACITY)
        .expect("HTJ2K encode output capacity fits u32");
    let double_capacity = capacity
        .checked_mul(2)
        .expect("test output capacity fits u32");
    let kernel_jobs = [
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 0,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: 0,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 4096,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: capacity,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
        super::CudaHtj2kEncodeKernelJob {
            coefficient_offset: 8192,
            coefficient_stride: 64,
            width: 64,
            height: 64,
            total_bitplanes: 8,
            output_offset: double_capacity,
            output_capacity: capacity,
            target_coding_passes: 1,
        },
    ];
    let statuses = [
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 12,
            reserved2: 0x8001_8002,
            ..super::CudaHtj2kEncodeStatus::default()
        },
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 0,
            ..super::CudaHtj2kEncodeStatus::default()
        },
        super::CudaHtj2kEncodeStatus {
            code: super::HTJ2K_STATUS_OK,
            data_len: 7,
            ..super::CudaHtj2kEncodeStatus::default()
        },
    ];

    let (compact_jobs, compact_len) =
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs)
            .expect("valid compact jobs");

    assert_eq!(compact_len, 19);
    assert_eq!(
        compact_jobs,
        vec![
            super::CudaHtj2kEncodeCompactJob {
                source_offset: 0,
                compact_offset: 0,
                data_len: 12,
                reserved: 0x8001_8002,
            },
            super::CudaHtj2kEncodeCompactJob {
                source_offset: capacity,
                compact_offset: 12,
                data_len: 0,
                reserved: 0,
            },
            super::CudaHtj2kEncodeCompactJob {
                source_offset: double_capacity,
                compact_offset: 12,
                data_len: 7,
                reserved: 0,
            },
        ]
    );
}

#[test]
fn htj2k_encode_compact_jobs_accept_exact_capacity_payloads() {
    let kernel_jobs = [super::CudaHtj2kEncodeKernelJob {
        coefficient_offset: 0,
        coefficient_stride: 64,
        width: 64,
        height: 64,
        total_bitplanes: 8,
        output_offset: 11,
        output_capacity: 5,
        target_coding_passes: 1,
    }];
    let statuses = [super::CudaHtj2kEncodeStatus {
        code: super::HTJ2K_STATUS_OK,
        data_len: 5,
        reserved2: 9,
        ..super::CudaHtj2kEncodeStatus::default()
    }];

    let (compact_jobs, compact_len) =
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs)
            .expect("exact-capacity compact job");

    assert_eq!(compact_len, 5);
    assert_eq!(
        compact_jobs,
        vec![super::CudaHtj2kEncodeCompactJob {
            source_offset: 11,
            compact_offset: 0,
            data_len: 5,
            reserved: 9,
        }]
    );
}

#[test]
fn htj2k_encode_compact_jobs_reject_payloads_larger_than_capacity() {
    let kernel_jobs = [super::CudaHtj2kEncodeKernelJob {
        coefficient_offset: 0,
        coefficient_stride: 64,
        width: 64,
        height: 64,
        total_bitplanes: 8,
        output_offset: 0,
        output_capacity: 5,
        target_coding_passes: 1,
    }];
    let statuses = [super::CudaHtj2kEncodeStatus {
        code: super::HTJ2K_STATUS_OK,
        data_len: 6,
        ..super::CudaHtj2kEncodeStatus::default()
    }];

    assert!(matches!(
        assert_compact_jobs_match_for_single_and_multi_input(&statuses, &kernel_jobs),
        Err(CudaError::LengthTooLarge { len }) if len == 6
    ));
}

#[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
#[test]
fn cuda_oxide_htj2k_compact_codeblocks_assembles_payload_when_required() {
    const J2K_HT_MEL_SIZE: usize = 192;
    const J2K_HT_VLC_SIZE: usize = 3072 - J2K_HT_MEL_SIZE;
    const J2K_HT_MS_SIZE: usize = (16384usize * 16).div_ceil(15);
    const J2K_HT_MEL_OFFSET: usize = J2K_HT_MS_SIZE;
    const J2K_HT_VLC_OFFSET: usize = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;
    const J2K_HT_COMPACT_ASSEMBLE_FLAG: u32 = 0x8000_0000;

    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let source_offset = 3usize;
    let plain_source_offset = source_offset + J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE + 8;
    let mut scratch = vec![0u8; plain_source_offset + 4];
    scratch[source_offset..source_offset + 3].copy_from_slice(&[10, 11, 12]);
    scratch[source_offset + J2K_HT_MEL_OFFSET..source_offset + J2K_HT_MEL_OFFSET + 2]
        .copy_from_slice(&[20, 21]);
    let vlc_start = source_offset + J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE - 3;
    scratch[vlc_start..vlc_start + 3].copy_from_slice(&[30, 31, 32]);
    scratch[plain_source_offset..plain_source_offset + 4].copy_from_slice(&[40, 41, 42, 43]);
    let jobs = [
        super::CudaHtj2kEncodeCompactJob {
            source_offset: u32::try_from(source_offset).expect("source offset fits"),
            compact_offset: 0,
            data_len: 8,
            reserved: J2K_HT_COMPACT_ASSEMBLE_FLAG | 2 | (3 << 15),
        },
        super::CudaHtj2kEncodeCompactJob {
            source_offset: u32::try_from(plain_source_offset).expect("plain offset fits"),
            compact_offset: 8,
            data_len: 4,
            reserved: 0,
        },
    ];
    let expected = [10, 11, 12, 20, 21, 30, 0x15, 0, 40, 41, 42, 43];

    let scratch_buffer = context.upload(&scratch).expect("scratch upload");
    let compact_buffer = context.allocate(expected.len()).expect("compact output");
    let jobs_buffer = context
        .upload(super::bytes::htj2k_encode_compact_jobs_as_bytes(&jobs))
        .expect("compact job upload");

    context
        .launch_htj2k_compact_codeblocks(&scratch_buffer, &compact_buffer, &jobs_buffer, jobs.len())
        .expect("cuda-oxide compact codeblocks");
    let mut actual = vec![0u8; expected.len()];
    compact_buffer
        .copy_to_host(&mut actual)
        .expect("download compact output");

    assert_eq!(actual, expected);
}

#[test]
fn htj2k_encode_tables_feed_resident_region_encode_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
        coefficient_offset: 0,
        coefficient_stride: 2,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblock_regions_resident(
            &coefficients,
            4,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &vlc_table0,
                vlc_table1: &vlc_table1,
                uvlc_table: &uvlc_table,
            },
        )
        .expect("resource-backed resident HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
}

#[test]
fn htj2k_encode_resident_region_reuses_pool_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockRegionJob {
        coefficient_offset: 0,
        coefficient_stride: 2,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            &coefficients,
            4,
            &jobs,
            &resources,
            &pool,
        )
        .expect("pooled resource-backed resident HTJ2K encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
    assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
}

#[test]
fn htj2k_encode_codeblocks_resident_reuses_pool_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let coefficients = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("resident coefficients");
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let encoded = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &coefficients,
            4,
            &jobs,
            &resources,
            &pool,
        )
        .expect("pooled resource-backed resident HTJ2K codeblock encode");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert_eq!(encoded.code_blocks().len(), 1);
    assert!(pool.cached_count().expect("cached pooled encode buffers") >= 3);
}

#[test]
fn htj2k_encode_multi_resident_inputs_match_separate_batches_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");
    let first = context
        .upload_i32_pinned(&[0, 0, 0, 0])
        .expect("first resident coefficients");
    let second = context
        .upload_i32_pinned(&[0, 0])
        .expect("second resident coefficients");
    let first_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];
    let second_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 1,
        total_bitplanes: 1,
        target_coding_passes: 1,
    }];

    let first_separate = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &first,
            4,
            &first_jobs,
            &resources,
            &pool,
        )
        .expect("first separate resident encode");
    let second_separate = context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &second,
            2,
            &second_jobs,
            &resources,
            &pool,
        )
        .expect("second separate resident encode");

    let combined = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
            &[
                CudaHtj2kEncodeResidentTarget {
                    coefficients: &first,
                    coefficient_count: 4,
                    jobs: &first_jobs,
                },
                CudaHtj2kEncodeResidentTarget {
                    coefficients: &second,
                    coefficient_count: 2,
                    jobs: &second_jobs,
                },
            ],
            &resources,
            &pool,
        )
        .expect("combined resident encode");

    assert_eq!(combined.execution().kernel_dispatches(), 1);
    assert_eq!(combined.code_blocks().len(), 2);
    assert_eq!(
        combined.code_blocks()[0].data(),
        first_separate.code_blocks()[0].data()
    );
    assert_eq!(
        combined.code_blocks()[1].data(),
        second_separate.code_blocks()[0].data()
    );
    let timings = combined.stage_timings();
    assert_eq!(
        timings.ht_encode_us,
        timings
            .ht_kernel_us
            .saturating_add(timings.ht_status_readback_us)
            .saturating_add(timings.ht_compact_us)
            .saturating_add(timings.ht_output_readback_us)
    );
    assert!(timings.ht_kernel_us > 0);
    assert!(timings.ht_status_readback_us > 0);
}

#[test]
fn htj2k97_resident_batch_returns_pooled_quantized_bands_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let blocks = vec![0.0f32; 64];
    let params = super::CudaHtj2k97QuantizeParams {
        inv_delta_ll: 1.0,
        inv_delta_hl: 1.0,
        inv_delta_lh: 1.0,
        inv_delta_hh: 1.0,
        cb_width: 64,
        cb_height: 64,
    };

    let (bands, _) = context
        .j2k_transcode_htj2k97_codeblock_batch_resident_with_pool(
            CudaHtj2k97CodeblockBatchWithPoolRequest {
                blocks: &blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: 1,
                    block_cols: 1,
                    block_rows: 1,
                    width: 8,
                    height: 8,
                },
                params,
                pool: &pool,
            },
        )
        .expect("resident HTJ2K 9/7 codeblock batch");

    assert!(bands.ll.as_device_buffer().is_some());
    assert!(bands.hl.as_device_buffer().is_some());
    assert!(bands.lh.as_device_buffer().is_some());
    assert!(bands.hh.as_device_buffer().is_some());
    let cached_while_bands_live = pool.cached_count().expect("cached buffers while live");

    drop(bands);

    assert!(pool.cached_count().expect("cached buffers after drop") >= cached_while_bands_live + 4);
}

#[test]
fn htj2k_encode_rejects_unsupported_refinement_pass_count_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [0, 2, -3, 1];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 3,
        target_coding_passes: 4,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("unsupported HTJ2K encode pass count is explicit");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 5);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_rejects_lossy_zero_sigprop_request_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [0, 2, -3, 4];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 3,
        target_coding_passes: 2,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("target-2 zero SigProp cannot silently drop low coefficient bits");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 6);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_rejects_unreachable_target_three_sigprop_coefficients_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let coefficients = [3, 0, 0, 0];
    let jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: 2,
        height: 2,
        total_bitplanes: 4,
        target_coding_passes: 3,
    }];

    let error = context
        .encode_htj2k_codeblocks(
            &coefficients,
            &jobs,
            CudaHtj2kEncodeTables {
                vlc_table0: &[0u16; 2048],
                vlc_table1: &[0u16; 2048],
                uvlc_table: &[0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES],
            },
        )
        .expect_err("isolated target-3 SigProp coefficient is explicitly unsupported");

    match error {
        CudaError::KernelStatus {
            kernel,
            code,
            detail,
        } => {
            assert_eq!(kernel, "j2k_htj2k_encode_codeblocks");
            assert_eq!(code, super::HTJ2K_STATUS_UNSUPPORTED);
            assert_eq!(detail, 6);
        }
        other => panic!("unexpected CUDA encode error: {other:?}"),
    }
}

#[test]
fn htj2k_encode_resources_feed_one_job_batch_encode_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let vlc_table0 = [0u16; 2048];
    let vlc_table1 = [0u16; 2048];
    let uvlc_table = vec![0u8; super::HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let resources = context
        .upload_htj2k_encode_resources(CudaHtj2kEncodeTables {
            vlc_table0: &vlc_table0,
            vlc_table1: &vlc_table1,
            uvlc_table: &uvlc_table,
        })
        .expect("encode resources");

    let encoded = context
        .encode_htj2k_codeblocks_with_resources(
            &[0, 0, 0, 0],
            &[CudaHtj2kEncodeCodeBlockJob {
                coefficient_offset: 0,
                width: 2,
                height: 2,
                total_bitplanes: 1,
                target_coding_passes: 1,
            }],
            &resources,
        )
        .expect("resource-backed one-job HTJ2K encode");
    let block = encoded
        .code_blocks()
        .first()
        .expect("one encoded code block");

    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    // An all-zero codeblock has no significant bitplanes, so the encoder emits zero
    // coding passes (matching native ht_block_encode::encode_code_block).
    assert_eq!(block.num_coding_passes(), 0);
    assert_eq!(block.cleanup_length(), 0);
    assert_eq!(block.data().len(), 0);
    assert_eq!(block.refinement_length(), 0);
}

#[test]
fn default_stream_timer_reports_elapsed_time_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![17u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_us(|| context.copy_with_kernel(&input))
        .expect("timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
#[test]
fn cuda_oxide_copy_u8_matches_builtin_copy_and_cpu_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = (0..4099)
        .map(|index| u8::try_from((index * 31 + 17) % 251).expect("modulo 251 fits u8"))
        .collect::<Vec<_>>();

    let builtin = context
        .copy_with_kernel(&input)
        .expect("builtin CUDA copy kernel");
    let cuda_oxide = context
        .copy_with_cuda_oxide_kernel(&input)
        .expect("cuda-oxide CUDA copy kernel");

    let mut builtin_bytes = vec![0u8; input.len()];
    builtin
        .buffer()
        .copy_to_host(&mut builtin_bytes)
        .expect("download builtin CUDA copy");
    let mut cuda_oxide_bytes = vec![0u8; input.len()];
    cuda_oxide
        .buffer()
        .copy_to_host(&mut cuda_oxide_bytes)
        .expect("download cuda-oxide CUDA copy");

    assert_eq!(builtin.execution().kernel_dispatches(), 1);
    assert_eq!(cuda_oxide.execution().kernel_dispatches(), 1);
    assert_eq!(builtin_bytes, input);
    assert_eq!(cuda_oxide_bytes, input);
    assert_eq!(cuda_oxide_bytes, builtin_bytes);
}

#[test]
fn named_default_stream_timer_is_available_for_profiling_ranges_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let input = vec![23u8; 4096];
    let (output, elapsed_us) = context
        .time_default_stream_named_us("j2k.test.copy", || context.copy_with_kernel(&input))
        .expect("named timed CUDA copy kernel");

    assert_eq!(output.execution().kernel_dispatches(), 1);
    assert!(elapsed_us > 0);
}

#[test]
fn typed_device_view_reports_element_count_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let mut aligned = context.allocate(16).expect("aligned buffer");
    let view = aligned.typed_view::<u32>().expect("typed immutable view");
    assert_eq!(view.len(), 4);
    let mut_view = aligned.typed_view_mut::<u64>().expect("typed mutable view");
    assert_eq!(mut_view.len(), 2);

    let unaligned = context.allocate(3).expect("unaligned buffer");
    let error = unaligned
        .typed_view::<u16>()
        .expect_err("unaligned typed view");
    assert!(matches!(
        error,
        CudaError::LengthNotElementAligned {
            bytes: 3,
            element_size: 2
        }
    ));
}

#[test]
fn kernel_module_names_cover_htj2k_decode_and_encode_stages() {
    let cases = [
        (
            CudaKernelName::Htj2kDecodeCodeblocks,
            "j2k_htj2k_decode_codeblocks",
        ),
        (
            CudaKernelName::Htj2kDecodeCodeblocksMultiCleanupDequantize,
            "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCodeblocks,
            "j2k_dequantize_htj2k_codeblocks",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCodeblocksMulti,
            "j2k_dequantize_htj2k_codeblocks_multi",
        ),
        (
            CudaKernelName::J2kDequantizeHtj2kCleanupJobsMulti,
            "j2k_dequantize_htj2k_cleanup_jobs_multi",
        ),
        (CudaKernelName::J2kIdwtInterleave, "j2k_idwt_interleave"),
        (
            CudaKernelName::J2kIdwtInterleaveHorizontal53Multi,
            "j2k_idwt_interleave_horizontal_53_multi",
        ),
        (
            CudaKernelName::J2kIdwtInterleaveHorizontal97Multi,
            "j2k_idwt_interleave_horizontal_97_multi",
        ),
        (
            CudaKernelName::J2kIdwtHorizontal53,
            "j2k_idwt_horizontal_53",
        ),
        (
            CudaKernelName::J2kIdwtHorizontal97,
            "j2k_idwt_horizontal_97",
        ),
        (
            CudaKernelName::J2kIdwtVertical53Multi,
            "j2k_idwt_vertical_53_multi",
        ),
        (
            CudaKernelName::J2kIdwtVertical97Multi,
            "j2k_idwt_vertical_97_multi",
        ),
        (
            CudaKernelName::J2kIdwtVertical97MultiCols4,
            "j2k_idwt_vertical_97_multi_cols4",
        ),
        (CudaKernelName::J2kIdwtVertical53, "j2k_idwt_vertical_53"),
        (CudaKernelName::J2kIdwtVertical97, "j2k_idwt_vertical_97"),
        (CudaKernelName::J2kInverseMct, "j2k_inverse_mct"),
        (CudaKernelName::J2kStoreGray8, "j2k_store_gray8"),
        (CudaKernelName::J2kStoreGray16, "j2k_store_gray16"),
        (CudaKernelName::J2kStoreRgb8, "j2k_store_rgb8"),
        (
            CudaKernelName::J2kStoreRgb8MctBatch,
            "j2k_store_rgb8_mct_batch",
        ),
        (CudaKernelName::J2kStoreRgb16, "j2k_store_rgb16"),
        (CudaKernelName::J2kStoreRgb16Mct, "j2k_store_rgb16_mct"),
        (
            CudaKernelName::Htj2kEncodeCodeblocks,
            "j2k_htj2k_encode_codeblocks",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInput,
            "j2k_htj2k_encode_codeblocks_multi_input",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup,
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup",
        ),
        (
            CudaKernelName::Htj2kEncodeCodeblocksMultiInputCleanup64,
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64",
        ),
        (
            CudaKernelName::Htj2kCompactCodeblocks,
            "j2k_htj2k_compact_codeblocks",
        ),
        (
            CudaKernelName::Htj2kPacketizeCleanup,
            "j2k_htj2k_packetize_cleanup",
        ),
    ];

    for (kernel, entrypoint) in cases {
        assert_eq!(kernel.entrypoint(), entrypoint);
        let raw_entrypoint = kernel.kernel().entrypoint();
        assert_eq!(
            &raw_entrypoint[..raw_entrypoint.len() - 1],
            entrypoint.as_bytes()
        );
        assert_eq!(raw_entrypoint.last(), Some(&0));
    }
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "paired forward/inverse transform buffers intentionally share stage terminology"
)]
fn htj2k_empty_codeblock_decode_zero_fills_coefficients_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let output = context
        .decode_htj2k_codeblocks(
            &[],
            &[],
            CudaHtj2kDecodeTables {
                vlc_table0: &first_vlc,
                vlc_table1: &later_vlc,
                uvlc_table0: &first_uvlc,
                uvlc_table1: &later_uvlc,
            },
            8,
        )
        .expect("empty HTJ2K decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");

    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "paired forward/inverse transform buffers intentionally share stage terminology"
)]
fn htj2k_empty_codeblock_decode_with_resources_zero_fills_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode tables");
    let resources = context
        .upload_htj2k_decode_resources_with_tables(&[], &tables)
        .expect("decode resources");

    let output = context
        .decode_htj2k_codeblocks_with_resources(&resources, &[], 8)
        .expect("resource-backed empty HTJ2K decode");
    let mut actual = vec![f32::NAN; 8];
    output
        .coefficients()
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download coefficients");

    assert_eq!(actual, vec![0.0; 8]);
    assert_eq!(output.execution().kernel_dispatches(), 0);
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "paired forward/inverse transform buffers intentionally share stage terminology"
)]
fn htj2k_decode_table_resources_feed_multiple_payload_uploads_when_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let first_vlc = [0u16; 1024];
    let later_vlc = [0u16; 1024];
    let first_uvlc = [0u16; 320];
    let later_uvlc = [0u16; 256];
    let tables = context
        .upload_htj2k_decode_table_resources(CudaHtj2kDecodeTables {
            vlc_table0: &first_vlc,
            vlc_table1: &later_vlc,
            uvlc_table0: &first_uvlc,
            uvlc_table1: &later_uvlc,
        })
        .expect("decode table resources");

    let first_resources = context
        .upload_htj2k_decode_resources_with_tables(&[0xAA, 0x55], &tables)
        .expect("first payload resources");
    let second_resources = context
        .upload_htj2k_decode_resources_with_tables(&[0x11, 0x22, 0x33], &tables)
        .expect("second payload resources");

    assert!(std::sync::Arc::ptr_eq(
        &first_resources.tables.as_ref().expect("first tables").inner,
        &second_resources
            .tables
            .as_ref()
            .expect("second tables")
            .inner
    ));
    assert_eq!(first_resources.payload_len, 2);
    assert_eq!(second_resources.payload_len, 3);
}

#[test]
fn j2k_inverse_dwt_single_dispatches_parallel_stages_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");

    let output = context
        .j2k_inverse_dwt_single_device(
            &ll,
            &hl,
            &lh,
            &hh,
            CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 2,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                irreversible97: 0,
            },
        )
        .expect("CUDA inverse DWT");

    assert_eq!(output.execution().kernel_dispatches(), 3);
    let mut actual = vec![0.0f32; 4];
    output
        .buffer()
        .copy_to_host(super::f32_slice_as_bytes_mut(&mut actual))
        .expect("download inverse DWT");
    assert_eq!(actual, vec![7.0, 9.0, 10.0, 13.0]);
}

#[test]
fn j2k_inverse_dwt_single_reuses_pool_when_runtime_required() {
    if !cuda_runtime_gate() {
        return;
    }

    let context = CudaContext::system_default().expect("CUDA context");
    let pool = context.buffer_pool();
    let ll = context
        .upload(super::f32_slice_as_bytes(&[10.0]))
        .expect("upload LL");
    let hl = context
        .upload(super::f32_slice_as_bytes(&[2.0]))
        .expect("upload HL");
    let lh = context
        .upload(super::f32_slice_as_bytes(&[4.0]))
        .expect("upload LH");
    let hh = context
        .upload(super::f32_slice_as_bytes(&[1.0]))
        .expect("upload HH");

    let output = context
        .j2k_inverse_dwt_single_device_with_pool(
            &ll,
            &hl,
            &lh,
            &hh,
            CudaJ2kIdwtJob {
                rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 2,
                    y1: 2,
                },
                ll_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hl_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                lh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                hh_rect: CudaJ2kRect {
                    x0: 0,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                },
                irreversible97: 0,
            },
            &pool,
        )
        .expect("pooled CUDA inverse DWT");

    assert_eq!(output.execution().kernel_dispatches(), 3);
    let cached_while_live = pool.cached_count().expect("cached while live");

    drop(output);

    assert!(pool.cached_count().expect("cached after drop") > cached_while_live);
}
