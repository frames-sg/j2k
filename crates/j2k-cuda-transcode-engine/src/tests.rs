use super::{validate_dct_block_grid, CudaError};
#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
use super::{CudaContext, CudaTranscodeEngine};
#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
use super::{CudaDwt97BatchGeometry, CudaHtj2k97CodeblockBatchWithPoolRequest};
#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn cuda_runtime_gate() -> bool {
    j2k_test_support::cuda_runtime_gate(module_path!())
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

#[test]
fn pooled_i16_pinned_upload_is_size_gated() {
    assert!(super::transcode::should_use_pinned_pooled_i16_upload(
        4 * 1024 * 1024
    ));
    assert!(!super::transcode::should_use_pinned_pooled_i16_upload(
        4 * 1024 * 1024 + 1
    ));
}

fn codeblock_major_from_row_major<T: Copy>(
    band: &[T],
    width: usize,
    height: usize,
    cb_width: usize,
    cb_height: usize,
) -> Vec<T> {
    assert_eq!(band.len(), width * height);
    assert!(cb_width > 0 && cb_height > 0);
    let mut output = Vec::new();
    output
        .try_reserve_exact(band.len())
        .expect("test code-block oracle allocation");
    for cby in 0..height.div_ceil(cb_height) {
        for cbx in 0..width.div_ceil(cb_width) {
            let block_width = (width - cbx * cb_width).min(cb_width);
            let block_height = (height - cby * cb_height).min(cb_height);
            for local_y in 0..block_height {
                let row_start = (cby * cb_height + local_y) * width + cbx * cb_width;
                output.extend_from_slice(&band[row_start..row_start + block_width]);
            }
        }
    }
    output
}

#[test]
fn codeblock_major_oracle_reorders_multiple_horizontal_blocks() {
    let row_major = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    assert_eq!(
        codeblock_major_from_row_major(&row_major, 5, 2, 4, 2),
        [0, 1, 2, 3, 5, 6, 7, 8, 4, 9]
    );
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

    let bands = CudaTranscodeEngine::new(&context)
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

    let bands = CudaTranscodeEngine::new(&context)
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
    const WIDE_I16_PATTERN: [i16; 17] =
        [-8, -7, -5, -2, 0, 1, 3, 4, 6, 8, -10, 10, -12, 12, -1, 2, 5];

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

    let expected_first = CudaTranscodeEngine::new(&context)
        .j2k_transcode_dwt97(&first, 1, 1, 8, 8)
        .expect("single first DWT97");
    let expected_second = CudaTranscodeEngine::new(&context)
        .j2k_transcode_dwt97(&second, 1, 1, 8, 8)
        .expect("single second DWT97");
    let (batch, _) = CudaTranscodeEngine::new(&context)
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
    let wide_expected = CudaTranscodeEngine::new(&context)
        .j2k_transcode_dwt97(&wide_blocks, wide_block_cols, 1, wide_width, wide_height)
        .expect("single wide DWT97");
    let (wide_batch, _) = CudaTranscodeEngine::new(&context)
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
    let (quantized, _) = CudaTranscodeEngine::new(&context)
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

    let wide_i16_blocks = (0..wide_block_cols * 64)
        .map(|index| WIDE_I16_PATTERN[index % WIDE_I16_PATTERN.len()])
        .collect::<Vec<_>>();
    let wide_i16_as_f32 = wide_i16_blocks
        .iter()
        .copied()
        .map(f32::from)
        .collect::<Vec<_>>();
    let (wide_i16_reference_bands, _) = CudaTranscodeEngine::new(&context)
        .j2k_transcode_dwt97_batch_with_pool(super::CudaDwt97BatchWithPoolRequest {
            blocks: &wide_i16_as_f32,
            geometry: CudaDwt97BatchGeometry {
                item_count: 1,
                block_cols: wide_block_cols,
                block_rows: 1,
                width: wide_width,
                height: wide_height,
            },
            pool: &pool,
        })
        .expect("wide f32 CUDA DWT97 reference batch");
    let wide_i16_expected_codeblocks = expected_dwt97_codeblocks(&wide_i16_reference_bands, params);
    let (wide_i16_quantized, wide_i16_timings) = CudaTranscodeEngine::new(&context)
        .j2k_transcode_htj2k97_codeblock_i16_batch_resident_with_pool(
            super::CudaHtj2k97I16CodeblockBatchWithPoolRequest {
                blocks: &wide_i16_blocks,
                geometry: CudaDwt97BatchGeometry {
                    item_count: 1,
                    block_cols: wide_block_cols,
                    block_rows: 1,
                    width: wide_width,
                    height: wide_height,
                },
                params,
                pool: &pool,
            },
        )
        .expect("wide staged i16 CUDA DWT97 quantize batch");
    assert_eq!(
        download_device_codeblock_bands(&wide_i16_quantized),
        wide_i16_expected_codeblocks
    );
    assert!(
        wide_i16_timings.column_lift_us > 0,
        "wide i16 path must retain the staged column lift after P13 rejection"
    );
    assert!(wide_i16_timings.quantize_codeblock_us > 0);

    let first_i16 = dwt97_fixture_i16_blocks(1);
    let second_i16 = dwt97_fixture_i16_blocks(-1);
    let mut i16_blocks = Vec::with_capacity(128);
    i16_blocks.extend_from_slice(&first_i16);
    i16_blocks.extend_from_slice(&second_i16);
    let (staged_i16, _) = CudaTranscodeEngine::new(&context)
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
        .expect("cuda-oxide staged i16 DWT97 quantize batch");
    assert_eq!(
        download_device_codeblock_bands(&staged_i16),
        expected_codeblocks
    );
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
        let quantize_band = |band: &[f32], inv_delta: f32, width: usize, height: usize| {
            let quantized = band
                .iter()
                .map(|&value| quantize_dwt97_deadzone(value, inv_delta))
                .collect::<Vec<_>>();
            codeblock_major_from_row_major(
                &quantized,
                width,
                height,
                params.cb_width,
                params.cb_height,
            )
        };
        ll.extend(quantize_band(
            &bands.ll,
            params.inv_delta_ll,
            bands.low_width,
            bands.low_height,
        ));
        hl.extend(quantize_band(
            &bands.hl,
            params.inv_delta_hl,
            bands.high_width,
            bands.low_height,
        ));
        lh.extend(quantize_band(
            &bands.lh,
            params.inv_delta_lh,
            bands.low_width,
            bands.high_height,
        ));
        hh.extend(quantize_band(
            &bands.hh,
            params.inv_delta_hh,
            bands.high_width,
            bands.high_height,
        ));
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
