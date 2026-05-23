// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use signinum_j2k_native::{DecodeSettings, Image};
#[cfg(target_os = "macos")]
use signinum_transcode::{JpegToHtj2kCoefficientPath, JpegToHtj2kOptions, JpegToHtj2kTranscoder};
#[cfg(target_os = "macos")]
use signinum_transcode_metal::{MetalDctToWaveletStageAccelerator, METAL_UNAVAILABLE};

#[cfg(target_os = "macos")]
#[test]
fn ycbcr_420_jpeg_transcodes_to_htj2k_with_explicit_metal_97_and_native_sampling() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");
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
