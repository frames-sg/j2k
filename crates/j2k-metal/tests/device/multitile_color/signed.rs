// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn prepared_metal_color_preserves_signed_rgb_in_resident_output() {
    if !should_run_metal_runtime() {
        return;
    }
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let fixture = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-s12-53-single-raw")
        .expect("checked-in single-tile OpenJPH signed RGB12 fixture");
    let inputs = vec![EncodedImage::full(Arc::from(fixture.encoded))];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU signed RGB12 oracle");
    let CpuBatchSamples::I16(expected) = expected.groups()[0].samples() else {
        panic!("signed RGB12 must use I16 batch storage")
    };
    let expected_bytes = expected
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(inputs).expect("prepare signed RGB group");
    assert!(prepared.errors().is_empty());
    assert_eq!(
        prepared.groups()[0].info().sample_type,
        NativeSampleType::I16
    );
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::Htj2kOffsetPlan
    );
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode signed RGB resident group");
    assert!(result.group_errors().is_empty());
    let surface = &result.groups()[0].surfaces()[0];
    assert_eq!(surface.pixel_format(), PixelFormat::RgbI16);
    assert_eq!(
        surface
            .as_bytes()
            .expect("resident signed RGB bytes")
            .as_ref(),
        expected_bytes
    );
}
