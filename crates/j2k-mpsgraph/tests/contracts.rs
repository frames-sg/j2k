// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchAlpha, BatchCodecRoute, BatchGroupInfo, BatchLayout, BatchWaveletTransform};
use j2k_core::{
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, PixelLayout, SampleType,
};
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
use j2k_mpsgraph::Error;
use j2k_mpsgraph::{MpsGraphElementType, MpsGraphTensorSpec};

fn group_info(color: PixelLayout, sample_type: SampleType, layout: BatchLayout) -> BatchGroupInfo {
    let (precision, signed) = match sample_type {
        SampleType::U8 => (8, false),
        SampleType::U16 => (16, false),
        SampleType::I16 => (16, true),
        _ => unreachable!("test covers the fast native batch contract"),
    };
    BatchGroupInfo {
        dimensions: (17, 11),
        color,
        alpha: BatchAlpha::None,
        precision,
        signed,
        sample_type,
        layout,
        colorspace: Colorspace::SRgb,
        route: BatchCodecRoute::Htj2k,
        transform: BatchWaveletTransform::Reversible53,
        transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
        payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
    }
}

#[test]
fn every_native_group_maps_to_rank_four_mpsgraph_shape_and_dtype() {
    for (color, channels) in [
        (PixelLayout::Gray, 1),
        (PixelLayout::Rgb, 3),
        (PixelLayout::Rgba, 4),
    ] {
        for (sample_type, expected_type) in [
            (SampleType::U8, MpsGraphElementType::U8),
            (SampleType::U16, MpsGraphElementType::U16),
            (SampleType::I16, MpsGraphElementType::I16),
        ] {
            let nchw = MpsGraphTensorSpec::from_group_info(
                &group_info(color, sample_type, BatchLayout::Nchw),
                5,
            )
            .expect("native NCHW group");
            assert_eq!(nchw.shape(), [5, channels, 11, 17]);
            assert_eq!(nchw.element_type(), expected_type);

            let nhwc = MpsGraphTensorSpec::from_group_info(
                &group_info(color, sample_type, BatchLayout::Nhwc),
                5,
            )
            .expect("native NHWC group");
            assert_eq!(nhwc.shape(), [5, 11, 17, channels]);
            assert_eq!(nhwc.element_type(), expected_type);
        }
    }
}

#[test]
fn tensor_spec_rejects_empty_groups() {
    let error = MpsGraphTensorSpec::from_group_info(
        &group_info(PixelLayout::Rgb, SampleType::U8, BatchLayout::Nhwc),
        0,
    )
    .expect_err("empty MPSGraph batches are invalid");
    assert!(error.to_string().contains("nonzero"));
}

#[test]
fn explicit_tensor_spec_rejects_overflow_and_zero_dimensions() {
    assert!(MpsGraphTensorSpec::new([usize::MAX, 2, 1, 1], MpsGraphElementType::U16,).is_err());
    assert!(MpsGraphTensorSpec::new([1, 0, 1, 3], MpsGraphElementType::U8).is_err());
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
#[test]
fn non_apple_api_consistently_reports_unsupported_platform() {
    use std::sync::Arc;

    use j2k::{prepare_batch, BatchDecodeOptions, EncodedImage};
    use j2k_mpsgraph::{MpsGraphBatchDecoder, MpsGraphProgram, SubmittedMpsGraphRun};
    use j2k_test_support::htj2k_rgb8_fixture;

    let options = BatchDecodeOptions::default();
    assert!(matches!(
        MpsGraphBatchDecoder::system_default(options),
        Err(Error::UnsupportedPlatform)
    ));

    let mut decoder = MpsGraphBatchDecoder;
    let encoded = Arc::<[u8]>::from(htj2k_rgb8_fixture(8, 8));
    let prepared = prepare_batch(vec![EncodedImage::full(encoded)], options)
        .expect("valid fallback batch preparation");
    assert_eq!(prepared.errors(), []);
    let group = prepared
        .groups()
        .first()
        .expect("one homogeneous fallback group");
    assert!(matches!(
        decoder.prepare(Vec::new()),
        Err(Error::UnsupportedPlatform)
    ));
    assert!(matches!(
        decoder.prepare_prepared_images(Vec::new()),
        Err(Error::UnsupportedPlatform)
    ));
    assert!(matches!(
        decoder.decode(Vec::new()),
        Err(Error::UnsupportedPlatform)
    ));
    assert!(matches!(
        decoder.decode_prepared(&prepared),
        Err(Error::UnsupportedPlatform)
    ));
    assert!(matches!(
        decoder.decode_prepared_images(Vec::new()),
        Err(Error::UnsupportedPlatform)
    ));

    let program = MpsGraphProgram;
    assert!(matches!(
        decoder.submit_prepared_group(&program, group),
        Err(Error::UnsupportedPlatform)
    ));
    assert!(matches!(
        decoder.run_prepared_group(&program, group),
        Err(Error::UnsupportedPlatform)
    ));

    let submitted = SubmittedMpsGraphRun;
    assert!(!submitted.is_complete());
    assert!(matches!(submitted.wait(), Err(Error::UnsupportedPlatform)));
}
