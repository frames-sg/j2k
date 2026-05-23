// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::accelerator::{DctGridToDwt97Job, DctToWaveletStageAccelerator};
use signinum_transcode_metal::weights::Dwt97WeightRows;
use signinum_transcode_metal::MetalDctToWaveletStageAccelerator;
#[cfg(not(target_os = "macos"))]
use signinum_transcode_metal::MetalTranscodeError;

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

fn f32_rows_to_bits(rows: &[Vec<f32>]) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}
