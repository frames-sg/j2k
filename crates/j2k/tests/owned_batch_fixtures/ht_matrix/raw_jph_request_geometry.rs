// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    wrap_j2k_codestream, BatchCodecRoute, BatchDecodeOptions, BatchLayout, BatchWaveletTransform,
    CompressedPayloadKind, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, Downscale,
    EncodedImage, J2kDecoder, J2kFileWrapOptions, J2kScratchPool, PixelFormat, Rect,
};
use j2k_test_support::{
    openhtj2k_refinement_fixture, openhtj2k_refinement_odd_fixture,
    openhtj2k_refinement_odd_pixels, openhtj2k_refinement_pixels,
};

#[test]
fn independent_openhtj2k_raw_and_derived_jph_outputs_are_exact_and_indexed() {
    let raw = openhtj2k_refinement_fixture();
    let expected = openhtj2k_refinement_pixels();
    // The codestream and oracle are independent OpenHT/OpenJPH artifacts; only
    // the JPH container around those bytes is constructed by this repository.
    let jph = wrap_j2k_codestream(raw, J2kFileWrapOptions::jph()).expect("wrap OpenHT fixture");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![
            EncodedImage::full(Arc::from(raw)),
            EncodedImage::full(Arc::from([0_u8, 1, 2, 3])),
            EncodedImage::full(Arc::from(raw)),
            EncodedImage::full(Arc::from(jph)),
        ])
        .expect("decode raw Part 15 and JPH batch");

    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 1);
    assert_eq!(result.groups().len(), 2);
    let raw_group = result
        .groups()
        .iter()
        .find(|group| group.info().payload_kind == CompressedPayloadKind::Jpeg2000Codestream)
        .expect("raw Part 15 group");
    assert_eq!(raw_group.source_indices(), [0, 2]);
    assert_eq!(raw_group.info().route, BatchCodecRoute::Htj2k);
    assert_eq!(
        raw_group.info().transform,
        BatchWaveletTransform::Reversible53
    );
    assert_eq!(raw_group.info().precision, 8);
    assert!(!raw_group.info().signed);
    let CpuBatchSamples::U8(raw_samples) = raw_group.samples() else {
        panic!("OpenHT raw group must retain u8 samples")
    };
    assert_eq!(raw_samples, &[expected, expected].concat());

    let jph_group = result
        .groups()
        .iter()
        .find(|group| group.info().payload_kind == CompressedPayloadKind::JphFile)
        .expect("JPH group");
    assert_eq!(jph_group.source_indices(), [3]);
    let CpuBatchSamples::U8(jph_samples) = jph_group.samples() else {
        panic!("OpenHT JPH group must retain u8 samples")
    };
    assert_eq!(jph_samples, expected);
}

#[test]
fn independent_odd_openhtj2k_fixture_supports_roi_and_reduction() {
    const WIDTH: u32 = 17;
    const HEIGHT: u32 = 37;
    let encoded = Arc::<[u8]>::from(openhtj2k_refinement_odd_fixture());
    let expected = openhtj2k_refinement_odd_pixels();
    let roi = Rect {
        x: 3,
        y: 5,
        w: 9,
        h: 21,
    };
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![
            EncodedImage::full(Arc::clone(&encoded)),
            EncodedImage::new(Arc::clone(&encoded), DecodeRequest::Region { roi }),
            EncodedImage::new(
                Arc::clone(&encoded),
                DecodeRequest::RegionReduced {
                    roi,
                    scale: Downscale::Half,
                },
            ),
        ])
        .expect("decode independent odd OpenHT request matrix");

    assert!(
        result.errors().is_empty(),
        "request errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), 3);
    assert!(result
        .groups()
        .iter()
        .all(|group| group.info().route == BatchCodecRoute::Htj2k));

    let full = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [0])
        .expect("full independent group");
    assert_eq!(
        full.decoded_rects(),
        [Rect {
            x: 0,
            y: 0,
            w: WIDTH,
            h: HEIGHT,
        }]
    );
    let CpuBatchSamples::U8(full_samples) = full.samples() else {
        panic!("independent odd full group must retain u8 samples")
    };
    assert_eq!(full_samples, expected);

    let region = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [1])
        .expect("independent ROI group");
    assert_eq!(region.decoded_rects(), [roi]);
    let CpuBatchSamples::U8(region_samples) = region.samples() else {
        panic!("independent odd ROI group must retain u8 samples")
    };
    let expected_region = (roi.y..roi.y + roi.h)
        .flat_map(|y| {
            let start = (y * WIDTH + roi.x) as usize;
            expected[start..start + roi.w as usize].iter().copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(region_samples, &expected_region);

    let scaled_roi = roi.scaled_covering(Downscale::Half);
    let mut scalar = J2kDecoder::new(&encoded).expect("scalar odd OpenHT decoder");
    let mut oracle = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize];
    let outcome = scalar
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut oracle,
            scaled_roi.w as usize,
            PixelFormat::Gray8,
            roi,
            Downscale::Half,
        )
        .expect("scalar odd OpenHT region-reduced oracle");
    assert_eq!(outcome.decoded, scaled_roi);
    let region_reduced = result
        .groups()
        .iter()
        .find(|group| group.source_indices() == [2])
        .expect("independent region-reduced group");
    assert_eq!(region_reduced.decoded_rects(), [scaled_roi]);
    let CpuBatchSamples::U8(region_reduced_samples) = region_reduced.samples() else {
        panic!("independent odd region-reduced group must retain u8 samples")
    };
    assert_eq!(region_reduced_samples, &oracle);
}
