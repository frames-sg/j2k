// SPDX-License-Identifier: Apache-2.0

use signinum_j2k_native::{DecodeSettings, Image};
use signinum_transcode::{jpeg_to_htj2k, JpegToHtj2kError, JpegToHtj2kOptions};

#[test]
fn grayscale_8x8_jpeg_transcodes_to_decodable_htj2k() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/grayscale_8x8.jpg");

    let encoded = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect("transcode grayscale JPEG to HTJ2K");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("native parser accepts generated HTJ2K")
        .decode_native()
        .expect("native decoder accepts generated HTJ2K");

    assert_eq!((encoded.report.width, encoded.report.height), (8, 8));
    assert_eq!(encoded.report.component_count, 1);
    assert_eq!((decoded.width, decoded.height), (8, 8));
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn ycbcr_jpeg_is_explicitly_out_of_initial_e2e_scope() {
    let jpeg = include_bytes!("../../signinum-jpeg/fixtures/conformance/baseline_420_16x16.jpg");

    let err = jpeg_to_htj2k(jpeg, &JpegToHtj2kOptions::default())
        .expect_err("YCbCr expansion is not implemented yet");

    assert!(matches!(err, JpegToHtj2kError::Unsupported(_)));
}
