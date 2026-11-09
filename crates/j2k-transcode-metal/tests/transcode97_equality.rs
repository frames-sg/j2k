// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend byte equality for the irreversible 9/7 JPEG-to-HTJ2K
//! transcode: explicit Metal must emit exactly the CPU codestream for every
//! matrix input and option variant, single and batched. Binary64 precision
//! is CPU-only: explicit Metal rejects it and Auto leaves it on the CPU.
//!
//! Set `J2K_REQUIRE_METAL_RUNTIME=1` to fail instead of skipping when no
//! Metal device is available.

use j2k_transcode::{
    JpegTileBatchInput, JpegToHtj2kError, JpegToHtj2kTranscoder, TranscodePrecision,
    TranscodeStageError,
};
use j2k_transcode_metal::MetalDctToWaveletStageAccelerator;
#[cfg(target_os = "macos")]
use j2k_transcode_test_support::transcode97_matrix::{matrix_jpegs, MatrixVariant};
use j2k_transcode_test_support::transcode97_matrix::{matrix_options, MATRIX_VARIANTS};

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_metal_matches_cpu_bytes_for_the_matrix() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut transcoder = JpegToHtj2kTranscoder::default();
    for jpeg in matrix_jpegs() {
        for variant in MATRIX_VARIANTS {
            let options = matrix_options(variant);
            let cpu = transcoder
                .transcode(&jpeg.bytes, &options)
                .unwrap_or_else(|error| panic!("CPU {} / {}: {error}", jpeg.name, variant.name))
                .codestream;

            let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
            let single = match transcoder.transcode_with_accelerator(
                &jpeg.bytes,
                &options,
                &mut accelerator,
            ) {
                Ok(encoded) => encoded,
                Err(JpegToHtj2kError::Accelerator(TranscodeStageError::DeviceUnavailable)) => {
                    j2k_test_support::metal_device_unavailable_is_skip(module_path!());
                    return;
                }
                Err(error) => panic!("Metal {} / {}: {error}", jpeg.name, variant.name),
            };
            assert!(
                single.report.timings.accelerator_work_observed(),
                "Metal {} / {} did not dispatch",
                jpeg.name,
                variant.name
            );
            assert_eq!(
                single.codestream, cpu,
                "Metal single-tile bytes differ from CPU for {} / {}",
                jpeg.name, variant.name
            );

            assert_metal_batch_matches(&mut transcoder, &jpeg.bytes, variant, &cpu, jpeg.name);
        }
    }
}

#[cfg(target_os = "macos")]
fn assert_metal_batch_matches(
    transcoder: &mut JpegToHtj2kTranscoder,
    bytes: &[u8],
    variant: MatrixVariant,
    cpu: &[u8],
    name: &str,
) {
    let options = matrix_options(variant);
    let tiles = [JpegTileBatchInput { bytes }; 3];
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let batch = transcoder
        .transcode_batch_with_accelerator(&tiles, &options, &mut accelerator)
        .unwrap_or_else(|error| panic!("Metal batch {name} / {}: {error}", variant.name));
    assert!(
        batch.report.timings.accelerator_work_observed(),
        "Metal batch {name} / {} did not dispatch",
        variant.name
    );
    for tile in batch.tiles {
        let tile = tile
            .unwrap_or_else(|error| panic!("Metal batch tile {name} / {}: {error}", variant.name));
        assert_eq!(
            tile.codestream, cpu,
            "Metal batch bytes differ from CPU for {name} / {}",
            variant.name
        );
    }
}

#[test]
fn explicit_metal_rejects_f64_precision() {
    let jpeg = j2k_test_support::JPEG_BASELINE_420_16X16;
    let mut options = matrix_options(MATRIX_VARIANTS[0]);
    options.precision = TranscodePrecision::F64;
    let mut transcoder = JpegToHtj2kTranscoder::default();

    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let single = transcoder.transcode_with_accelerator(jpeg, &options, &mut accelerator);
    assert!(
        matches!(
            single,
            Err(JpegToHtj2kError::Accelerator(
                TranscodeStageError::Unsupported(_)
            ))
        ),
        "explicit Metal must reject F64, got {:?}",
        single.map(|encoded| encoded.codestream.len())
    );

    let tiles = [JpegTileBatchInput { bytes: jpeg }; 2];
    let mut accelerator = MetalDctToWaveletStageAccelerator::new_explicit();
    let batch = transcoder.transcode_batch_with_accelerator(&tiles, &options, &mut accelerator);
    assert!(
        matches!(
            batch,
            Err(JpegToHtj2kError::Accelerator(
                TranscodeStageError::Unsupported(_)
            ))
        ),
        "explicit Metal batch must reject F64"
    );
}

#[test]
fn auto_metal_runs_f64_precision_on_the_cpu() {
    let jpeg = j2k_test_support::JPEG_BASELINE_444_RESTART_67X45;
    let mut options = matrix_options(MATRIX_VARIANTS[1]);
    options.precision = TranscodePrecision::F64;
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let cpu = transcoder
        .transcode(jpeg, &options)
        .expect("CPU F64 transcode")
        .codestream;

    let mut accelerator = MetalDctToWaveletStageAccelerator::for_auto()
        .with_auto_dwt97_min_samples(0)
        .with_auto_dwt97_batch_thresholds(0, 0);
    let auto = transcoder
        .transcode_with_accelerator(jpeg, &options, &mut accelerator)
        .expect("Auto F64 transcode runs on the CPU");
    assert_eq!(auto.codestream, cpu);
    assert_eq!(accelerator.dwt97_dispatches(), 0);
    assert_eq!(accelerator.dwt97_batch_dispatches(), 0);
}
