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
    assert_dwt97_bands_bits_eq(&bands, &canonical_reference_bands(&blocks, 1, 1, 8, 8));

    // Odd sizes, one-sample axes, and an axis above the cooperative
    // shared-memory row limit.
    for (block_cols, block_rows, width, height) in [
        (2, 2, 13, 11),
        (1, 2, 1, 9),
        (2, 1, 9, 1),
        (130, 1, 1033, 7),
    ] {
        let grid = (0..block_cols * block_rows * 64)
            .map(|index| blocks[index % 64] - f32::from(u8::try_from(index % 13).expect("small")))
            .collect::<Vec<_>>();
        let (batch, _) = CudaTranscodeEngine::new(&context)
            .j2k_transcode_dwt97_batch_with_pool(super::CudaDwt97BatchWithPoolRequest {
                blocks: &grid,
                geometry: CudaDwt97BatchGeometry {
                    item_count: 1,
                    block_cols,
                    block_rows,
                    width,
                    height,
                },
                pool: &context.buffer_pool(),
            })
            .expect("cuda-oxide 9/7 batch transcode");
        assert_dwt97_bands_bits_eq(
            &batch[0],
            &canonical_reference_bands(&grid, block_cols, block_rows, width, height),
        );
    }
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
    assert_dwt97_bands_bits_eq(&batch[0], &expected_first);
    assert_dwt97_bands_bits_eq(&batch[1], &expected_second);
    assert_dwt97_bands_bits_eq(&batch[0], &canonical_reference_bands(&first, 1, 1, 8, 8));

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
    assert_dwt97_bands_bits_eq(&wide_batch[0], &wide_expected);
    assert_dwt97_bands_bits_eq(
        &wide_batch[0],
        &canonical_reference_bands(&wide_blocks, wide_block_cols, 1, wide_width, wide_height),
    );

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
        "wide i16 path must retain the staged column lift"
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
fn assert_f32_slice_bits_eq(actual: &[f32], expected: &[f32], band: &str) {
    assert_eq!(actual.len(), expected.len(), "{band} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{band}[{index}]: actual={actual}, expected={expected}"
        );
    }
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn assert_dwt97_bands_bits_eq(
    actual: &super::CudaTranscodeDwt97Bands,
    expected: &super::CudaTranscodeDwt97Bands,
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
    assert_f32_slice_bits_eq(&actual.ll, &expected.ll, "ll");
    assert_f32_slice_bits_eq(&actual.hl, &expected.hl, "hl");
    assert_f32_slice_bits_eq(&actual.lh, &expected.lh, "lh");
    assert_f32_slice_bits_eq(&actual.hh, &expected.hh, "hh");
}

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
struct HostArithmetic;

#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
impl j2k_codec_math::transcode97::Transcode97Arithmetic for HostArithmetic {
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// Canonical binary32 bands from `j2k_codec_math::transcode97`: per-block
/// IDCT, then interleaved lifting of every row and every column.
#[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
fn canonical_reference_bands(
    blocks: &[f32],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> super::CudaTranscodeDwt97Bands {
    use j2k_codec_math::transcode97::{forward_lift97, idct8x8};

    assert_eq!(blocks.len(), block_cols * block_rows * 64);
    let mut plane = vec![0.0f32; width * height];
    for (block_index, block) in blocks.chunks_exact(64).enumerate() {
        let coefficients: [f32; 64] = block.try_into().expect("64 coefficients");
        let samples = idct8x8::<HostArithmetic>(&coefficients);
        let (block_x, block_y) = (block_index % block_cols, block_index / block_cols);
        for local_y in 0..8 {
            for local_x in 0..8 {
                let (x, y) = (block_x * 8 + local_x, block_y * 8 + local_y);
                if x < width && y < height {
                    plane[y * width + x] = samples[local_y * 8 + local_x];
                }
            }
        }
    }
    for row in plane.chunks_mut(width) {
        forward_lift97::<HostArithmetic>(row);
    }
    for x in 0..width {
        let mut column = (0..height)
            .map(|y| plane[y * width + x])
            .collect::<Vec<_>>();
        forward_lift97::<HostArithmetic>(&mut column);
        for (y, value) in column.into_iter().enumerate() {
            plane[y * width + x] = value;
        }
    }
    let (low_width, high_width) = (width.div_ceil(2), width / 2);
    let (low_height, high_height) = (height.div_ceil(2), height / 2);
    let band = |x_parity: usize, y_parity: usize, band_width: usize, band_height: usize| {
        (0..band_height)
            .flat_map(|y| (0..band_width).map(move |x| (x, y)))
            .map(|(x, y)| plane[(2 * y + y_parity) * width + 2 * x + x_parity])
            .collect::<Vec<_>>()
    };
    super::CudaTranscodeDwt97Bands {
        ll: band(0, 0, low_width, low_height),
        hl: band(1, 0, high_width, low_height),
        lh: band(0, 1, low_width, high_height),
        hh: band(1, 1, high_width, high_height),
        low_width,
        low_height,
        high_width,
        high_height,
    }
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
fn quantize_dwt97_deadzone(value: f32, inv_delta: f32) -> i32 {
    j2k_codec_math::transcode97::quantize97::<HostArithmetic>(value, inv_delta)
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
