// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
#[cfg(target_os = "macos")]
use signinum_transcode::accelerator::{
    DctGridToHtj2k97CodeBlockJob, Htj2k97CodeBlockOptions, J2kSubBandType,
    PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband,
};
#[cfg(target_os = "macos")]
use signinum_transcode::dct97_2d::{
    dct8x8_blocks_then_dwt97_float_with_scratch, Dct97GridScratch, Dwt97TwoDimensional,
};
use signinum_transcode_metal::weights::{Dwt97WeightRows, SparseDwt97WeightRows};
use signinum_transcode_metal::MetalDctToWaveletStageAccelerator;
#[cfg(not(target_os = "macos"))]
use signinum_transcode_metal::MetalTranscodeError;
#[cfg(target_os = "macos")]
use signinum_transcode_metal::METAL_UNAVAILABLE;

#[test]
fn explicit_metal_reports_unavailable_on_non_macos() {
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let blocks = vec![[[0.0; 8]; 8]];
    let result = accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 1,
        block_rows: 1,
        width: 8,
        height: 8,
    });

    #[cfg(not(target_os = "macos"))]
    assert_eq!(
        result.expect_err("explicit Metal is unavailable off macOS"),
        MetalTranscodeError::MetalUnavailable.as_static_str()
    );

    #[cfg(target_os = "macos")]
    let _ = result;
}

#[test]
fn auto_metal_falls_back_for_tiny_jobs() {
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();
    let blocks = vec![[[0.0; 8]; 8]];
    let output = accelerator
        .dct_grid_to_dwt97(DctGridToDwt97Job {
            blocks: &blocks,
            block_cols: 1,
            block_rows: 1,
            width: 8,
            height: 8,
        })
        .expect("auto accelerator can decline tiny job");

    assert!(output.is_none());
    assert_eq!(accelerator.dwt97_attempts(), 1);
    assert_eq!(accelerator.dwt97_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_metal_uses_cpu_for_97_jobs_by_default() {
    let blocks = structured_blocks(64, 64);
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();

    match accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 64,
        block_rows: 64,
        width: 512,
        height: 512,
    }) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("auto Metal should leave 9/7 jobs on the optimized CPU path"),
        Err(message) if message == METAL_UNAVAILABLE => {}
        Err(message) => panic!("auto Metal 9/7 accelerator failed: {message}"),
    }

    assert_eq!(accelerator.dwt97_attempts(), 1);
    assert_eq!(accelerator.dwt97_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_matches_scalar_for_structured_cases() {
    let blocks = structured_blocks(2, 2);
    let mut scalar_scratch = Dct97GridScratch::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    for (width, height) in [(8, 8), (13, 11), (16, 16)] {
        let actual = match accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
            blocks: &blocks,
            block_cols: 2,
            block_rows: 2,
            width,
            height,
        }) {
            Ok(Some(output)) => output,
            Ok(None) => panic!("explicit Metal accelerator must not silently fall back"),
            Err(message) if message == METAL_UNAVAILABLE => {
                eprintln!("skipping Metal coefficient test because no Metal device is available");
                return;
            }
            Err(message) => panic!("explicit Metal accelerator failed: {message}"),
        };
        let expected = dct8x8_blocks_then_dwt97_float_with_scratch(
            &blocks,
            2,
            2,
            width,
            height,
            &mut scalar_scratch,
        )
        .expect("scalar 9/7 IDCT path accepts covered grid");

        let max_diff = max_abs_diff(&actual, &expected);
        assert!(
            max_diff <= 2.0e-2,
            "Metal 9/7 DCT transform diverged for {width}x{height}: {max_diff}"
        );
    }

    assert_eq!(accelerator.dwt97_dispatches(), 3);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_batch_matches_scalar_for_structured_cases() {
    let first = structured_blocks(2, 2);
    let second = structured_blocks_with_offset(2, 2, 97.0);
    let jobs = [
        DctGridToDwt97Job {
            blocks: &first,
            block_cols: 2,
            block_rows: 2,
            width: 13,
            height: 11,
        },
        DctGridToDwt97Job {
            blocks: &second,
            block_cols: 2,
            block_rows: 2,
            width: 13,
            height: 11,
        },
    ];
    let mut scalar_scratch = Dct97GridScratch::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let actual = match accelerator.dct_grid_to_dwt97_batch(&jobs) {
        Ok(Some(output)) => output,
        Ok(None) => panic!("explicit Metal batch accelerator must not silently fall back"),
        Err(message) if message == METAL_UNAVAILABLE => {
            eprintln!("skipping Metal batch coefficient test because no Metal device is available");
            return;
        }
        Err(message) => panic!("explicit Metal batch accelerator failed: {message}"),
    };

    assert_eq!(actual.len(), jobs.len());
    for (actual, job) in actual.iter().zip(jobs.iter()) {
        let expected = dct8x8_blocks_then_dwt97_float_with_scratch(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            &mut scalar_scratch,
        )
        .expect("scalar 9/7 IDCT path accepts covered grid");

        let max_diff = max_abs_diff(actual, &expected);
        assert!(
            max_diff <= 2.0e-2,
            "Metal 9/7 batch transform diverged: {max_diff}"
        );
    }

    assert_eq!(accelerator.dwt97_batch_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_batch_reports_idct_row_and_column_stage_timings() {
    let batch_blocks = [
        structured_blocks_with_offset(4, 4, 0.0),
        structured_blocks_with_offset(4, 4, 3.0),
    ];
    let jobs: Vec<_> = batch_blocks
        .iter()
        .map(|blocks| DctGridToDwt97Job {
            blocks,
            block_cols: 4,
            block_rows: 4,
            width: 29,
            height: 31,
        })
        .collect();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    match accelerator.dct_grid_to_dwt97_batch(&jobs) {
        Ok(Some(_)) => {}
        Ok(None) => panic!("explicit Metal batch accelerator must not silently fall back"),
        Err(message) if message == METAL_UNAVAILABLE => {
            eprintln!("skipping Metal batch timing test because no Metal device is available");
            return;
        }
        Err(message) => panic!("explicit Metal batch accelerator failed: {message}"),
    }

    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("Metal 9/7 batch records backend stage timings");
    assert!(timings.pack_upload_us > 0);
    assert!(timings.idct_row_lift_us > 0);
    assert!(timings.column_lift_us > 0);
    assert_eq!(timings.quantize_codeblock_us, 0);
    assert!(timings.readback_us > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_dct97_codeblock_batch_matches_scalar_quantized_layout() {
    let batch_blocks = [
        structured_blocks_with_offset(4, 4, 0.0),
        structured_blocks_with_offset(4, 4, 37.0),
    ];
    let jobs: Vec<_> = batch_blocks
        .iter()
        .map(|blocks| DctGridToHtj2k97CodeBlockJob {
            blocks,
            block_cols: 4,
            block_rows: 4,
            width: 29,
            height: 31,
            x_rsiz: 1,
            y_rsiz: 1,
        })
        .collect();
    let options = Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 2,
        code_block_height_exp: 2,
        irreversible_quantization_scale: 2.5,
    };
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let actual = match accelerator.dct_grid_to_htj2k97_codeblock_batch(&jobs, options) {
        Ok(Some(output)) => output,
        Ok(None) => {
            panic!("explicit Metal code-block batch accelerator must not silently fall back")
        }
        Err(message) if message == METAL_UNAVAILABLE => {
            eprintln!("skipping Metal code-block batch test because no Metal device is available");
            return;
        }
        Err(message) => panic!("explicit Metal code-block batch accelerator failed: {message}"),
    };

    assert_eq!(actual.len(), jobs.len());
    let mut scalar_scratch = Dct97GridScratch::default();
    for (actual, job) in actual.iter().zip(jobs.iter()) {
        let dwt = dct8x8_blocks_then_dwt97_float_with_scratch(
            job.blocks,
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            &mut scalar_scratch,
        )
        .expect("scalar 9/7 IDCT path accepts covered grid");
        let expected =
            prequantized_component_from_dwt_for_test(&dwt, options, job.x_rsiz, job.y_rsiz);

        assert_prequantized_component_layout_eq(actual, &expected);
        assert_prequantized_component_coefficients_close(actual, &expected, 1);
    }

    assert_eq!(accelerator.dwt97_batch_dispatches(), 1);
    assert_eq!(accelerator.htj2k97_codeblock_batch_attempts(), 1);
    assert_eq!(accelerator.htj2k97_codeblock_batch_dispatches(), 1);
    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("Metal code-block batch records backend stage timings");
    assert!(timings.pack_upload_us > 0);
    assert!(timings.idct_row_lift_us > 0);
    assert!(timings.column_lift_us > 0);
    assert!(timings.quantize_codeblock_us > 0);
    assert!(timings.readback_us > 0);
}

#[test]
fn weight_rows_match_expected_geometry_for_supported_lengths() {
    for sample_len in [8_usize, 13, 16] {
        let rows = Dwt97WeightRows::for_len(sample_len);

        assert_eq!(rows.low.len(), sample_len.div_ceil(2));
        assert_eq!(rows.high.len(), sample_len / 2);
        assert!(rows.low.iter().all(|row| row.len() == sample_len));
        assert!(rows.high.iter().all(|row| row.len() == sample_len));
        assert!(rows
            .low
            .iter()
            .all(|row| row.iter().any(|&value| value.to_bits() != 0)));
        assert!(rows
            .high
            .iter()
            .all(|row| row.iter().any(|&value| value.to_bits() != 0)));
    }
}

#[test]
fn weight_rows_are_deterministic() {
    let first = Dwt97WeightRows::for_len(13);
    let second = Dwt97WeightRows::for_len(13);

    assert_eq!(f32_rows_to_bits(&first.low), f32_rows_to_bits(&second.low));
    assert_eq!(
        f32_rows_to_bits(&first.high),
        f32_rows_to_bits(&second.high)
    );
}

#[test]
fn sparse_weight_rows_reconstruct_dense_rows_for_wsi_lengths() {
    for sample_len in [8_usize, 13, 16, 512, 1024, 2048] {
        let dense = Dwt97WeightRows::for_len(sample_len);
        let sparse = SparseDwt97WeightRows::for_len(sample_len);

        assert!(sparse.max_taps_per_row() <= 16);
        assert_eq!(sparse.low.len(), dense.low.len());
        assert_eq!(sparse.high.len(), dense.high.len());
        assert_eq!(reconstruct_sparse_rows(&sparse.low, sample_len), dense.low);
        assert_eq!(
            reconstruct_sparse_rows(&sparse.high, sample_len),
            dense.high
        );
    }
}

fn f32_rows_to_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn reconstruct_sparse_rows(
    rows: &[signinum_transcode_metal::weights::SparseWeightRow],
    sample_len: usize,
) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|row| {
            let mut dense = vec![0.0; sample_len];
            for tap in &row.taps {
                dense[tap.sample_idx] = tap.weight;
            }
            dense
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn max_abs_diff(actual: &Dwt97TwoDimensional<f64>, expected: &Dwt97TwoDimensional<f64>) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    actual
        .ll
        .iter()
        .zip(expected.ll.iter())
        .chain(actual.hl.iter().zip(expected.hl.iter()))
        .chain(actual.lh.iter().zip(expected.lh.iter()))
        .chain(actual.hh.iter().zip(expected.hh.iter()))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

#[cfg(target_os = "macos")]
fn prequantized_component_from_dwt_for_test(
    dwt: &Dwt97TwoDimensional<f64>,
    options: Htj2k97CodeBlockOptions,
    x_rsiz: u8,
    y_rsiz: u8,
) -> PrequantizedHtj2k97Component {
    PrequantizedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        resolutions: vec![
            PrequantizedHtj2k97Resolution {
                subbands: vec![prequantized_subband_from_coefficients_for_test(
                    &dwt.ll,
                    dwt.low_width,
                    dwt.low_height,
                    J2kSubBandType::LowLow,
                    quantized_97_total_bitplanes_for_test(options, J2kSubBandType::LowLow),
                    options,
                )],
            },
            PrequantizedHtj2k97Resolution {
                subbands: vec![
                    prequantized_subband_from_coefficients_for_test(
                        &dwt.hl,
                        dwt.high_width,
                        dwt.low_height,
                        J2kSubBandType::HighLow,
                        quantized_97_total_bitplanes_for_test(options, J2kSubBandType::HighLow),
                        options,
                    ),
                    prequantized_subband_from_coefficients_for_test(
                        &dwt.lh,
                        dwt.low_width,
                        dwt.high_height,
                        J2kSubBandType::LowHigh,
                        quantized_97_total_bitplanes_for_test(options, J2kSubBandType::LowHigh),
                        options,
                    ),
                    prequantized_subband_from_coefficients_for_test(
                        &dwt.hh,
                        dwt.high_width,
                        dwt.high_height,
                        J2kSubBandType::HighHigh,
                        quantized_97_total_bitplanes_for_test(options, J2kSubBandType::HighHigh),
                        options,
                    ),
                ],
            },
        ],
    }
}

#[cfg(target_os = "macos")]
fn prequantized_subband_from_coefficients_for_test(
    coefficients: &[f64],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Subband {
    let quantized = quantize_97_subband_for_test(coefficients, sub_band_type, options);
    let cb_width = 1usize << (options.code_block_width_exp + 2);
    let cb_height = 1usize << (options.code_block_height_exp + 2);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            let mut block_coefficients = Vec::with_capacity(block_width * block_height);
            for y in 0..block_height {
                let row_start = (y0 + y) * width + x0;
                block_coefficients
                    .extend_from_slice(&quantized[row_start..row_start + block_width]);
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: block_coefficients,
                width: block_width as u32,
                height: block_height as u32,
            });
        }
    }

    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes,
        code_blocks,
    }
}

#[cfg(target_os = "macos")]
fn quantize_97_subband_for_test(
    coefficients: &[f64],
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Vec<i32> {
    let delta = quantized_97_delta_for_test(options, sub_band_type);
    let inv_delta = 1.0 / delta;

    coefficients
        .iter()
        .map(|&coefficient| {
            let sign = if coefficient < 0.0 { -1 } else { 1 };
            sign * (coefficient.abs() * inv_delta).floor() as i32
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn quantized_97_total_bitplanes_for_test(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> u8 {
    let (exponent, _) = quantized_97_step_for_test(options, sub_band_type);
    options
        .guard_bits
        .saturating_add(exponent)
        .saturating_sub(1)
}

#[cfg(target_os = "macos")]
fn quantized_97_delta_for_test(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> f64 {
    let log_gain = match sub_band_type {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow | J2kSubBandType::LowHigh => 1,
        J2kSubBandType::HighHigh => 2,
    };
    let range_bits = i32::from(options.bit_depth) + log_gain;
    let (exponent, mantissa) = quantized_97_step_for_test(options, sub_band_type);
    pow2i_f64_for_test(range_bits - i32::from(exponent)) * (1.0 + f64::from(mantissa) / 2048.0)
}

#[cfg(target_os = "macos")]
fn quantized_97_step_for_test(
    options: Htj2k97CodeBlockOptions,
    _sub_band_type: J2kSubBandType,
) -> (u8, u16) {
    let base_delta = pow2i_f64_for_test(-i32::from(options.guard_bits))
        * f64::from(options.irreversible_quantization_scale);
    let floor_log2 = base_delta.log2().floor() as i32;
    let mut exponent = i32::from(options.bit_depth) - floor_log2;
    let normalized = base_delta / pow2i_f64_for_test(floor_log2);
    let mut mantissa = ((normalized - 1.0) * 2048.0).round() as i32;

    if mantissa >= 2048 {
        exponent -= 1;
        mantissa = 0;
    }

    (
        u8::try_from(exponent.clamp(0, 31)).expect("clamped exponent fits u8"),
        u16::try_from(mantissa.clamp(0, 2047)).expect("clamped mantissa fits u16"),
    )
}

#[cfg(target_os = "macos")]
fn pow2i_f64_for_test(exp: i32) -> f64 {
    if exp >= 0 {
        f64::from(1u32 << exp.cast_unsigned())
    } else {
        1.0 / f64::from(1u32 << (-exp).cast_unsigned())
    }
}

#[cfg(target_os = "macos")]
fn assert_prequantized_component_layout_eq(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
) {
    assert_eq!(actual.x_rsiz, expected.x_rsiz);
    assert_eq!(actual.y_rsiz, expected.y_rsiz);
    assert_eq!(actual.resolutions.len(), expected.resolutions.len());
    for (actual_resolution, expected_resolution) in
        actual.resolutions.iter().zip(expected.resolutions.iter())
    {
        assert_eq!(
            actual_resolution.subbands.len(),
            expected_resolution.subbands.len()
        );
        for (actual_subband, expected_subband) in actual_resolution
            .subbands
            .iter()
            .zip(expected_resolution.subbands.iter())
        {
            assert_eq!(actual_subband.sub_band_type, expected_subband.sub_band_type);
            assert_eq!(actual_subband.num_cbs_x, expected_subband.num_cbs_x);
            assert_eq!(actual_subband.num_cbs_y, expected_subband.num_cbs_y);
            assert_eq!(
                actual_subband.total_bitplanes,
                expected_subband.total_bitplanes
            );
            assert_eq!(
                actual_subband.code_blocks.len(),
                expected_subband.code_blocks.len()
            );
            for (actual_block, expected_block) in actual_subband
                .code_blocks
                .iter()
                .zip(expected_subband.code_blocks.iter())
            {
                assert_eq!(actual_block.width, expected_block.width);
                assert_eq!(actual_block.height, expected_block.height);
                assert_eq!(
                    actual_block.coefficients.len(),
                    expected_block.coefficients.len()
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn assert_prequantized_component_coefficients_close(
    actual: &PrequantizedHtj2k97Component,
    expected: &PrequantizedHtj2k97Component,
    max_abs_error: i32,
) {
    for (actual_resolution, expected_resolution) in
        actual.resolutions.iter().zip(expected.resolutions.iter())
    {
        for (actual_subband, expected_subband) in actual_resolution
            .subbands
            .iter()
            .zip(expected_resolution.subbands.iter())
        {
            for (actual_block, expected_block) in actual_subband
                .code_blocks
                .iter()
                .zip(expected_subband.code_blocks.iter())
            {
                for (&actual_coefficient, &expected_coefficient) in actual_block
                    .coefficients
                    .iter()
                    .zip(expected_block.coefficients.iter())
                {
                    assert!(
                        (actual_coefficient - expected_coefficient).abs() <= max_abs_error,
                        "quantized coefficient diverged: actual {actual_coefficient}, expected {expected_coefficient}"
                    );
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn structured_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = Vec::with_capacity(block_cols * block_rows);
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let mut block = [[0.0; 8]; 8];
            block[0][0] = 384.0 + (block_x * 19 + block_y * 23) as f64;
            block[0][1] = -17.0 + block_x as f64;
            block[1][0] = 11.0 - block_y as f64;
            block[2][3] = 7.0;
            block[4][4] = -3.0;
            block[7][7] = 2.0;
            blocks.push(block);
        }
    }
    blocks
}

#[cfg(target_os = "macos")]
fn structured_blocks_with_offset(
    block_cols: usize,
    block_rows: usize,
    offset: f64,
) -> Vec<[[f64; 8]; 8]> {
    let mut blocks = structured_blocks(block_cols, block_rows);
    for block in &mut blocks {
        block[0][0] += offset;
        block[3][2] -= offset / 7.0;
    }
    blocks
}
