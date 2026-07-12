// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{Decoder, PixelFormat, Warning};
use j2k_test_support::progressive_8x8_jpeg;

fn decode_rgb(bytes: &[u8]) -> (Vec<u8>, Vec<Warning>) {
    let decoder = Decoder::new(bytes).expect("progressive decoder");
    let (width, height) = decoder.info().dimensions;
    let stride = usize::try_from(width * 3).expect("fixture stride");
    let mut pixels = vec![0; stride * usize::try_from(height).expect("fixture height")];
    let outcome = decoder
        .decode_into(&mut pixels, stride, PixelFormat::Rgb8)
        .expect("progressive decode");
    (pixels, outcome.warnings)
}

#[test]
fn valid_multiscan_and_missing_eoi_decode_identically_with_typed_warning() {
    let complete = progressive_8x8_jpeg();
    let mut missing_eoi = complete.clone();
    missing_eoi.truncate(missing_eoi.len() - 2);

    let (complete_pixels, complete_warnings) = decode_rgb(&complete);
    let (missing_pixels, missing_warnings) = decode_rgb(&missing_eoi);

    assert_eq!(missing_pixels, complete_pixels);
    assert!(!complete_warnings.contains(&Warning::MissingEoi));
    assert!(missing_warnings.contains(&Warning::MissingEoi));
}
