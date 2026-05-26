// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use signinum_j2k_native::{DecodeSettings, Image};
#[cfg(target_os = "macos")]
use signinum_transcode::{
    JpegTileBatchInput, JpegToHtj2kCoefficientPath, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
};
#[cfg(target_os = "macos")]
use signinum_transcode_metal::{MetalDctToWaveletStageAccelerator, METAL_UNAVAILABLE};

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_97_and_native_sampling() {
    let jpeg = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::lossy_97()
    };
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let encoded = match transcoder.transcode_with_accelerator(jpeg, &options, &mut accelerator) {
        Ok(encoded) => encoded,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!(
                "skipping Metal transcode integration test because no Metal device is available"
            );
            return;
        }
        Err(error) => panic!("explicit Metal 9/7 transcode failed: {error}"),
    };
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated Metal 9/7 HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated Metal 9/7 HTJ2K");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(
        encoded.report.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    );
    assert_eq!(
        encoded.report.path,
        "native_component_sampling_float_direct_97"
    );
    assert_eq!(metrics.total, 384);
    assert_eq!(metrics.max_abs_error, 0);
    assert_eq!(accelerator.dwt97_attempts(), 3);
    assert_eq!(accelerator.dwt97_dispatches(), 3);
    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);
    assert_report_sampling(
        &encoded.report.components,
        &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)],
    );
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 2), (2, 2)]);
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_53_and_native_sampling() {
    let jpeg = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
    let options = JpegToHtj2kOptions {
        coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear53,
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::lossless_53()
    };
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let encoded = match transcoder.transcode_with_accelerator(jpeg, &options, &mut accelerator) {
        Ok(encoded) => encoded,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!(
                "skipping Metal transcode integration test because no Metal device is available"
            );
            return;
        }
        Err(error) => panic!("explicit Metal 5/3 transcode failed: {error}"),
    };
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated Metal 5/3 HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated Metal 5/3 HTJ2K");
    let metrics = encoded
        .report
        .float_reference_metrics
        .as_ref()
        .expect("float reference metrics are reported");

    assert_eq!(
        encoded.report.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear53
    );
    assert_eq!(
        encoded.report.path,
        "native_component_sampling_float_direct_53"
    );
    assert_eq!(metrics.total, 384);
    assert_eq!(metrics.max_abs_error, 0);
    assert_eq!(accelerator.dwt53_attempts(), 3);
    assert_eq!(accelerator.dwt53_dispatches(), 3);
    assert_eq!((decoded.width, decoded.height), (16, 16));
    assert_eq!(decoded.num_components, 3);
    assert_report_sampling(
        &encoded.report.components,
        &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)],
    );
    assert_component_sampling(&encoded.codestream, &[(1, 1), (2, 2), (2, 2)]);
}

#[cfg(target_os = "macos")]
#[test]
fn grayscale_jpeg_transcodes_to_htj2k_with_explicit_metal_reversible_53_batch() {
    assert_explicit_metal_integer53_matches_scalar(
        include_bytes!("../fixtures/conformance/grayscale_8x8.jpg"),
        "full_resolution_components_integer_direct_53",
        &[(8, 8, 1, 1)],
        &[(1, 1)],
        1,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_444_jpeg_transcodes_to_htj2k_with_explicit_metal_reversible_53_batch() {
    assert_explicit_metal_integer53_matches_scalar(
        include_bytes!("../fixtures/conformance/baseline_444_8x8.jpg"),
        "full_resolution_components_integer_direct_53",
        &[(8, 8, 1, 1), (8, 8, 1, 1), (8, 8, 1, 1)],
        &[(1, 1), (1, 1), (1, 1)],
        1,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_422_jpeg_transcodes_to_htj2k_with_explicit_metal_reversible_53_batch() {
    assert_explicit_metal_integer53_matches_scalar(
        include_bytes!("../fixtures/conformance/baseline_422_16x8.jpg"),
        "native_component_sampling_integer_direct_53",
        &[(16, 8, 1, 1), (8, 8, 2, 1), (8, 8, 2, 1)],
        &[(1, 1), (2, 1), (2, 1)],
        2,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_reversible_53_batch() {
    assert_explicit_metal_integer53_matches_scalar(
        include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg"),
        "native_component_sampling_integer_direct_53",
        &[(16, 16, 1, 1), (8, 8, 2, 2), (8, 8, 2, 2)],
        &[(1, 1), (2, 2), (2, 2)],
        2,
    );
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_batch_transcodes_with_explicit_metal_reversible_53_across_tiles() {
    let jpeg = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let options = JpegToHtj2kOptions {
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::lossless_53()
    };
    let mut scalar_transcoder = JpegToHtj2kTranscoder::default();
    let scalar = scalar_transcoder
        .transcode(jpeg, &options)
        .expect("scalar IntegerDirect53 transcode succeeds");
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let batch = match transcoder.transcode_batch_with_accelerator(
        &inputs,
        &options,
        &mut accelerator,
    ) {
        Ok(batch) => batch,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!(
                    "skipping Metal reversible batch transcode integration test because no Metal device is available"
                );
            return;
        }
        Err(error) => panic!("explicit Metal reversible 5/3 batch transcode failed: {error}"),
    };

    assert_eq!(batch.report.tile_count, inputs.len());
    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(batch.report.failed_tiles, 0);
    assert_eq!(batch.report.reversible_dwt53_batches, 3);
    assert_eq!(batch.report.reversible_dwt53_batch_jobs, 12);
    assert_eq!(accelerator.reversible_dwt53_attempts(), 0);
    assert_eq!(accelerator.reversible_dwt53_batch_attempts(), 3);
    assert_eq!(accelerator.reversible_dwt53_batch_dispatches(), 3);
    for tile in batch.tiles {
        let tile = tile.expect("valid tile transcodes");
        assert_eq!(tile.codestream, scalar.codestream);
        assert_component_sampling(&tile.codestream, &[(1, 1), (2, 2), (2, 2)]);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_batch_transcodes_with_explicit_metal_97_across_tiles() {
    let jpeg = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let options = JpegToHtj2kOptions {
        validate_against_float_reference: true,
        ..JpegToHtj2kOptions::lossy_97()
    };
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let batch = match transcoder.transcode_batch_with_accelerator(
        &inputs,
        &options,
        &mut accelerator,
    ) {
        Ok(batch) => batch,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!("skipping Metal 9/7 batch transcode integration test because no Metal device is available");
            return;
        }
        Err(error) => panic!("explicit Metal 9/7 batch transcode failed: {error}"),
    };

    assert_eq!(batch.report.tile_count, inputs.len());
    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(batch.report.failed_tiles, 0);
    assert_eq!(batch.report.timings.batch_jobs, 12);
    assert_eq!(batch.report.timings.accelerator_dispatches, 3);
    assert_eq!(batch.report.timings.accelerator_dispatched_jobs, 12);
    assert_eq!(batch.report.timings.cpu_fallback_jobs, 0);
    assert!(batch.report.timings.dwt97_batch_pack_upload_us > 0);
    assert!(batch.report.timings.dwt97_batch_idct_row_lift_us > 0);
    assert!(batch.report.timings.dwt97_batch_column_lift_us > 0);
    assert_eq!(batch.report.timings.dwt97_batch_quantize_codeblock_us, 0);
    assert!(batch.report.timings.dwt97_batch_readback_us > 0);
    assert_eq!(accelerator.dwt97_attempts(), 0);
    assert_eq!(accelerator.dwt97_batch_attempts(), 3);
    assert_eq!(accelerator.dwt97_batch_dispatches(), 3);
    for tile in batch.tiles {
        let tile = tile.expect("valid 9/7 tile transcodes");
        assert_eq!(
            tile.report.coefficient_path,
            JpegToHtj2kCoefficientPath::FloatDirectLinear97
        );
        assert_eq!(
            tile.report.path,
            "native_component_sampling_float_direct_97"
        );
        assert_eq!(
            tile.report
                .float_reference_metrics
                .as_ref()
                .expect("float reference metrics are reported")
                .max_abs_error,
            0
        );
        assert_component_sampling(&tile.codestream, &[(1, 1), (2, 2), (2, 2)]);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_batch_transcodes_with_explicit_metal_97_codeblock_path() {
    let jpeg = include_bytes!("../fixtures/conformance/baseline_420_16x16.jpg");
    let inputs = vec![JpegTileBatchInput { bytes: jpeg }; 4];
    let options = JpegToHtj2kOptions::lossy_97();
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let batch = match transcoder.transcode_batch_with_accelerator(
        &inputs,
        &options,
        &mut accelerator,
    ) {
        Ok(batch) => batch,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!("skipping Metal 9/7 code-block batch transcode integration test because no Metal device is available");
            return;
        }
        Err(error) => panic!("explicit Metal 9/7 code-block batch transcode failed: {error}"),
    };

    assert_eq!(batch.report.tile_count, inputs.len());
    assert_eq!(batch.report.successful_tiles, inputs.len());
    assert_eq!(batch.report.failed_tiles, 0);
    assert_eq!(batch.report.timings.batch_jobs, 12);
    assert_eq!(batch.report.timings.accelerator_dispatches, 3);
    assert_eq!(batch.report.timings.accelerator_dispatched_jobs, 12);
    assert_eq!(batch.report.timings.cpu_fallback_jobs, 0);
    assert!(batch.report.timings.dwt97_batch_pack_upload_us > 0);
    assert!(batch.report.timings.dwt97_batch_idct_row_lift_us > 0);
    assert!(batch.report.timings.dwt97_batch_column_lift_us > 0);
    assert!(batch.report.timings.dwt97_batch_quantize_codeblock_us > 0);
    assert!(batch.report.timings.dwt97_batch_readback_us > 0);
    assert_eq!(accelerator.dwt97_attempts(), 0);
    assert_eq!(accelerator.dwt97_batch_attempts(), 3);
    assert_eq!(accelerator.dwt97_batch_dispatches(), 3);
    assert_eq!(accelerator.htj2k97_codeblock_batch_attempts(), 3);
    assert_eq!(accelerator.htj2k97_codeblock_batch_dispatches(), 3);
    for tile in batch.tiles {
        let tile = tile.expect("valid 9/7 code-block tile transcodes");
        let decoded = Image::new(&tile.codestream, &DecodeSettings::default())
            .expect("native parser accepts generated Metal code-block 9/7 HTJ2K")
            .decode_native()
            .expect("native decoder accepts generated Metal code-block 9/7 HTJ2K");
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

#[cfg(target_os = "macos")]
fn assert_explicit_metal_integer53_matches_scalar(
    jpeg: &[u8],
    expected_path: &str,
    expected_report_sampling: &[(u32, u32, u8, u8)],
    expected_codestream_sampling: &[(u8, u8)],
    expected_batch_dispatches: usize,
) {
    let options = JpegToHtj2kOptions {
        validate_against_integer_reference: true,
        ..JpegToHtj2kOptions::lossless_53()
    };
    let mut scalar_transcoder = JpegToHtj2kTranscoder::default();
    let scalar = scalar_transcoder
        .transcode(jpeg, &options)
        .expect("scalar IntegerDirect53 transcode succeeds");
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();

    let encoded = match transcoder.transcode_with_accelerator(jpeg, &options, &mut accelerator) {
        Ok(encoded) => encoded,
        Err(error) if error.to_string().contains(METAL_UNAVAILABLE) => {
            eprintln!(
                "skipping Metal reversible transcode integration test because no Metal device is available"
            );
            return;
        }
        Err(error) => panic!("explicit Metal reversible 5/3 transcode failed: {error}"),
    };
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated Metal reversible 5/3 HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated Metal reversible 5/3 HTJ2K");
    let metrics = encoded
        .report
        .integer_reference_metrics
        .as_ref()
        .expect("integer reference metrics are reported");

    assert_eq!(encoded.codestream, scalar.codestream);
    assert_eq!(
        encoded.report.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    );
    assert_eq!(encoded.report.path, expected_path);
    assert_eq!(metrics.max_abs_error, 0);
    assert_eq!(metrics.exact_matches, metrics.total);
    assert_eq!(accelerator.reversible_dwt53_attempts(), 0);
    assert_eq!(accelerator.reversible_dwt53_dispatches(), 0);
    assert_eq!(
        accelerator.reversible_dwt53_batch_attempts(),
        expected_batch_dispatches
    );
    assert_eq!(
        accelerator.reversible_dwt53_batch_dispatches(),
        expected_batch_dispatches
    );
    assert_eq!(
        (decoded.width, decoded.height),
        (encoded.report.width, encoded.report.height)
    );
    assert_eq!(decoded.num_components, expected_report_sampling.len() as u8);
    assert_report_sampling(&encoded.report.components, expected_report_sampling);
    assert_component_sampling(&encoded.codestream, expected_codestream_sampling);
}

#[cfg(target_os = "macos")]
fn assert_report_sampling(
    components: &[signinum_transcode::TranscodeComponentReport],
    expected: &[(u32, u32, u8, u8)],
) {
    assert_eq!(components.len(), expected.len());
    for (component, &(width, height, x_rsiz, y_rsiz)) in components.iter().zip(expected.iter()) {
        assert_eq!((component.width, component.height), (width, height));
        assert_eq!((component.x_rsiz, component.y_rsiz), (x_rsiz, y_rsiz));
    }
}

#[cfg(target_os = "macos")]
fn assert_component_sampling(codestream: &[u8], expected: &[(u8, u8)]) {
    let siz = find_marker(codestream, 0x51).expect("SIZ marker");
    let component_info = siz + 40;
    for (component_index, &(x_rsiz, y_rsiz)) in expected.iter().enumerate() {
        let offset = component_info + component_index * 3;
        assert_eq!(codestream[offset + 1], x_rsiz);
        assert_eq!(codestream[offset + 2], y_rsiz);
    }
}

#[cfg(target_os = "macos")]
fn find_marker(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xff, marker])
}
