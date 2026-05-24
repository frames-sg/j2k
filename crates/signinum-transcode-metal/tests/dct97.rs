// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
#[cfg(target_os = "macos")]
use signinum_transcode::dct97_2d::{
    dct8x8_blocks_to_dwt97_float_linear_with_scratch, Dct97GridScratch, Dwt97TwoDimensional,
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
fn auto_metal_dispatches_97_for_224_square_jobs() {
    let blocks = structured_blocks(28, 28);
    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto();

    match accelerator.dct_grid_to_dwt97(DctGridToDwt97Job {
        blocks: &blocks,
        block_cols: 28,
        block_rows: 28,
        width: 224,
        height: 224,
    }) {
        Ok(Some(_)) => {}
        Ok(None) => panic!("auto Metal should dispatch measured 224x224 9/7 jobs"),
        Err(message) if message == METAL_UNAVAILABLE => {
            eprintln!("skipping Metal auto threshold test because no Metal device is available");
            return;
        }
        Err(message) => panic!("auto Metal 9/7 accelerator failed: {message}"),
    }

    assert_eq!(accelerator.dwt97_attempts(), 1);
    assert_eq!(accelerator.dwt97_dispatches(), 1);
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
        let expected = dct8x8_blocks_to_dwt97_float_linear_with_scratch(
            &blocks,
            2,
            2,
            width,
            height,
            &mut scalar_scratch,
        )
        .expect("scalar 9/7 projection accepts covered grid");

        let max_diff = max_abs_diff(&actual, &expected);
        assert!(
            max_diff <= 2.0e-2,
            "Metal 9/7 DCT projection diverged for {width}x{height}: {max_diff}"
        );
    }

    assert_eq!(accelerator.dwt97_dispatches(), 3);
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
