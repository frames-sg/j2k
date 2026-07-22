// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn classic_raw_and_jp2_regression_outputs_are_bit_exact() {
    const WIDTH: u32 = 11;
    const HEIGHT: u32 = 7;
    let pixels = (0..WIDTH * HEIGHT)
        .map(|index| ((index * 29 + 7) & 0xff) as u8)
        .collect::<Vec<_>>();
    let raw = encode(
        &pixels,
        WIDTH,
        HEIGHT,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode classic raw fixture");
    let jp2 = wrap_j2k_codestream(&raw, J2kFileWrapOptions::jp2()).expect("wrap classic JP2");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(options);
    let result = decoder
        .decode(vec![
            EncodedImage::full(Arc::from(raw)),
            EncodedImage::full(Arc::from(jp2)),
        ])
        .expect("decode classic raw/JP2 batch");

    assert!(result.errors().is_empty());
    assert_eq!(result.groups().len(), 2);
    for (index, payload_kind) in [
        (0, CompressedPayloadKind::Jpeg2000Codestream),
        (1, CompressedPayloadKind::Jp2File),
    ] {
        let group = result
            .groups()
            .iter()
            .find(|group| group.source_indices() == [index])
            .expect("classic wrapper group");
        assert_eq!(group.info().payload_kind, payload_kind);
        assert_eq!(group.info().route, BatchCodecRoute::Classic);
        assert_eq!(group.info().transform, BatchWaveletTransform::Reversible53);
        let CpuBatchSamples::U8(samples) = group.samples() else {
            panic!("classic fixture must retain u8 samples")
        };
        assert_eq!(samples, &pixels);
    }
}

fn u16_from_le_bytes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

fn assert_full_and_roi_u16(
    groups: &[CpuBatchGroup],
    samples: &[u16],
    dimensions: (u32, u32),
    roi: Rect,
) {
    let (width, height) = dimensions;
    let full = groups
        .iter()
        .find(|group| group.source_indices() == [0])
        .expect("full group");
    assert_eq!(
        full.decoded_rects(),
        [Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }]
    );
    let CpuBatchSamples::U16(full_samples) = full.samples() else {
        panic!("full Gray12 group must retain u16 samples")
    };
    assert_eq!(full_samples, samples);

    let region = groups
        .iter()
        .find(|group| group.source_indices() == [1])
        .expect("ROI group");
    assert_eq!(region.decoded_rects(), [roi]);
    let CpuBatchSamples::U16(region_samples) = region.samples() else {
        panic!("ROI Gray12 group must retain u16 samples")
    };
    let expected_region = (roi.y..roi.y + roi.h)
        .flat_map(|y| {
            let start = (y * width + roi.x) as usize;
            samples[start..start + roi.w as usize].iter().copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(region_samples, &expected_region);
}

fn assert_reduced_u16(groups: &[CpuBatchGroup], encoded: &[u8], dimensions: (u32, u32), roi: Rect) {
    let reduced_dimensions = (dimensions.0.div_ceil(2), dimensions.1.div_ceil(2));
    let mut reduced_bytes =
        vec![0_u8; reduced_dimensions.0 as usize * reduced_dimensions.1 as usize * 2];
    J2kDecoder::new(encoded)
        .expect("scalar reduced decoder")
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut reduced_bytes,
            reduced_dimensions.0 as usize * 2,
            PixelFormat::Gray16,
            Downscale::Half,
        )
        .expect("scalar reduced oracle");
    let reduced = groups
        .iter()
        .find(|group| group.source_indices() == [2])
        .expect("reduced group");
    let CpuBatchSamples::U16(reduced_samples) = reduced.samples() else {
        panic!("reduced Gray12 group must retain u16 samples")
    };
    assert_eq!(reduced_samples, &u16_from_le_bytes(&reduced_bytes));

    let scaled_roi = roi.scaled_covering(Downscale::Half);
    let mut region_reduced_bytes = vec![0_u8; scaled_roi.w as usize * scaled_roi.h as usize * 2];
    let outcome = J2kDecoder::new(encoded)
        .expect("scalar region-reduced decoder")
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut region_reduced_bytes,
            scaled_roi.w as usize * 2,
            PixelFormat::Gray16,
            roi,
            Downscale::Half,
        )
        .expect("scalar region-reduced oracle");
    assert_eq!(outcome.decoded, scaled_roi);
    let region_reduced = groups
        .iter()
        .find(|group| group.source_indices() == [3])
        .expect("region-reduced group");
    assert_eq!(region_reduced.decoded_rects(), [scaled_roi]);
    let CpuBatchSamples::U16(region_reduced_samples) = region_reduced.samples() else {
        panic!("region-reduced Gray12 group must retain u16 samples")
    };
    assert_eq!(
        region_reduced_samples,
        &u16_from_le_bytes(&region_reduced_bytes)
    );
}

#[test]
fn multitile_multilevel_classic_batch_supports_full_roi_and_reduction() {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 80;
    let samples = (0..WIDTH * HEIGHT)
        .map(|index| ((index * 613 + index / 11) & 0x0fff) as u16)
        .collect::<Vec<_>>();
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let source = J2kLosslessSamples::new(&bytes, WIDTH, HEIGHT, 1, 12, false)
        .expect("multi-tile Gray12 source");
    let encoded = Arc::<[u8]>::from(
        encode_j2k_lossless(
            source,
            &J2kLosslessEncodeOptions::default()
                .with_cpu_only_backend()
                .with_max_decomposition_levels(Some(3))
                .with_tile_size(Some((48, 40))),
        )
        .expect("encode odd multi-tile classic fixture")
        .codestream,
    );
    let roi = Rect {
        x: 5,
        y: 7,
        w: 41,
        h: 29,
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
                DecodeRequest::Reduced {
                    scale: Downscale::Half,
                },
            ),
            EncodedImage::new(
                Arc::clone(&encoded),
                DecodeRequest::RegionReduced {
                    roi,
                    scale: Downscale::Half,
                },
            ),
        ])
        .expect("decode odd classic request matrix");

    assert!(
        result.errors().is_empty(),
        "request errors: {:?}",
        result.errors()
    );
    assert_eq!(result.groups().len(), 4);
    assert!(result
        .groups()
        .iter()
        .all(|group| group.info().route == BatchCodecRoute::Classic));
    assert_full_and_roi_u16(result.groups(), &samples, (WIDTH, HEIGHT), roi);
    assert_reduced_u16(result.groups(), &encoded, (WIDTH, HEIGHT), roi);
}
