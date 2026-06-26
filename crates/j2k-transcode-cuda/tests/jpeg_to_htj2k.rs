// SPDX-License-Identifier: MIT OR Apache-2.0
//
// End-to-end pipeline parity: drive the full JPEG -> HTJ2K transcode through the
// CUDA accelerator with the fused 9/7 code-block path active (supports_*
// = true), and assert the produced codestream decodes correctly. This exercises
// the CUDA-specific glue the isolated hook parity tests do not: DCT repack,
// multi-component (YCbCr 4:2:0) job construction, sampling factors, and the
// prequantized encode integration. Mirrors j2k-transcode-metal's
// ycbcr_420_batch_transcodes_with_explicit_metal_97_codeblock_path.
//
// Compiled only with `cuda-runtime`; asserts only on the CUDA runner.
#![cfg(feature = "cuda-runtime")]

use j2k_native::{DecodeSettings, Image};
use j2k_test_support::{cuda_runtime_required, jpeg_baseline_420_16x16};
use j2k_transcode::{
    JpegTileBatchInput, JpegToHtj2kCoefficientPath, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
};
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;

#[test]
fn ycbcr_420_batch_transcodes_to_htj2k_with_explicit_cuda_97_codeblock_path() {
    if !cuda_runtime_required() {
        return;
    }

    let jpeg = jpeg_baseline_420_16x16();
    let inputs = vec![
        JpegTileBatchInput {
            bytes: jpeg.as_slice(),
        };
        4
    ];
    let options = JpegToHtj2kOptions::lossy_97();
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit();

    let batch = transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut accelerator)
        .expect("explicit CUDA 9/7 code-block batch transcode should succeed on the runner");

    // Pipeline-level accounting: 4 tiles x 3 components = 12 jobs, grouped into
    // luma plus the compatible chroma pair, no scalar fallback.
    assert_eq!(batch.report.tile_count, inputs.len());
    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(batch.report.failed_tiles, 0);
    assert_eq!(batch.report.timings.batch_jobs, 12);
    assert_eq!(batch.report.timings.accelerator_dispatches, 2);
    assert_eq!(batch.report.timings.accelerator_dispatched_jobs, 12);
    assert_eq!(batch.report.timings.cpu_fallback_jobs, 0);

    // Real staged timings flowed through from the GPU code-block path.
    assert!(batch.report.timings.dwt97_batch_pack_upload_us > 0);
    assert!(batch.report.timings.dwt97_batch_idct_row_lift_us > 0);
    assert!(batch.report.timings.dwt97_batch_column_lift_us > 0);
    assert!(batch.report.timings.dwt97_batch_quantize_codeblock_us > 0);
    assert!(batch.report.timings.dwt97_batch_readback_us > 0);

    // The code-block path is a staged 9/7 batch plus quantize, so it registers as
    // both (single-job 9/7 never used).
    assert_eq!(accelerator.dwt97_attempts(), 0);
    assert_eq!(accelerator.dwt97_batch_attempts(), 2);
    assert_eq!(accelerator.dwt97_batch_dispatches(), 2);
    assert_eq!(accelerator.htj2k97_codeblock_batch_attempts(), 2);
    assert_eq!(accelerator.htj2k97_codeblock_batch_dispatches(), 2);

    for tile in batch.tiles {
        let tile = tile.expect("valid 9/7 code-block tile transcodes");
        let decoded = Image::new(&tile.codestream, &DecodeSettings::default())
            .expect("native parser accepts generated CUDA code-block 9/7 HTJ2K")
            .decode_native()
            .expect("native decoder accepts generated CUDA code-block 9/7 HTJ2K");
        assert_eq!((decoded.width, decoded.height), (16, 16));
        assert_eq!(decoded.num_components, 3);
        assert_eq!(
            tile.report.coefficient_path,
            JpegToHtj2kCoefficientPath::FloatDirectLinear97
        );
        assert_eq!(
            tile.report.path,
            "native_component_sampling_float_direct_97"
        );
        assert!(tile.report.float_reference_metrics.is_none());
        assert_component_sampling(&tile.codestream, &[(1, 1), (2, 2), (2, 2)]);
    }
}

/// Assert the SIZ marker's per-component sub-sampling factors (`XRsiz`, `YRsiz`).
fn assert_component_sampling(codestream: &[u8], expected: &[(u8, u8)]) {
    let siz = find_marker(codestream, 0x51).expect("SIZ marker");
    let component_info = siz + 40;
    for (component_index, &(x_rsiz, y_rsiz)) in expected.iter().enumerate() {
        let offset = component_info + component_index * 3;
        assert_eq!(codestream[offset + 1], x_rsiz);
        assert_eq!(codestream[offset + 2], y_rsiz);
    }
}

fn find_marker(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xff, marker])
}
