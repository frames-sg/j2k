// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchColor, BatchDecodeOptions, BatchItemError, BatchLayout, CpuBatchDecoder, DecodeRequest,
    Downscale, EncodedImage, NativeSampleType, NonRepresentableReason, PreparationDepth, Rect,
};
use j2k_native::{encode_component_planes_53, EncodeComponentPlane, EncodeOptions};

use super::oracles::{decoded_samples_for_source, native_request_oracle};

/// RGB codestream without MCT whose second and third components use `sampling`.
fn sampled_rgb_fixture(
    width: u32,
    height: u32,
    sampling: (u8, u8),
    precision: u8,
    ht: bool,
) -> Vec<u8> {
    sampled_fixture(width, height, &[(1, 1), sampling, sampling], precision, ht)
}

/// Codestream without MCT with one component per entry of `factors`.
fn sampled_fixture(
    width: u32,
    height: u32,
    factors: &[(u8, u8)],
    precision: u8,
    ht: bool,
) -> Vec<u8> {
    let max = (1_u32 << precision) - 1;
    let data: Vec<Vec<u8>> = factors
        .iter()
        .enumerate()
        .map(|(component, &(sx, sy))| {
            let samples = width.div_ceil(u32::from(sx)) * height.div_ceil(u32::from(sy));
            (0..samples)
                .flat_map(|index| {
                    let component = u32::try_from(component).expect("component index");
                    let sample = (index * 37 + component * 101) % (max + 1);
                    if precision <= 8 {
                        vec![u8::try_from(sample).expect("8-bit sample")]
                    } else {
                        u16::try_from(sample)
                            .expect("16-bit sample")
                            .to_le_bytes()
                            .to_vec()
                    }
                })
                .collect()
        })
        .collect();
    let planes: Vec<_> = data
        .iter()
        .zip(factors)
        .map(|(data, &(x_rsiz, y_rsiz))| EncodeComponentPlane {
            data,
            x_rsiz,
            y_rsiz,
        })
        .collect();
    encode_component_planes_53(
        &planes,
        width,
        height,
        precision,
        false,
        &EncodeOptions {
            use_ht_block_coding: ht,
            use_mct: false,
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode sampled fixture")
}

#[test]
fn cpu_batch_upsamples_subsampled_components_like_single_image_decode() {
    const WIDTH: u32 = 17;
    const HEIGHT: u32 = 13;
    let roi = Rect {
        x: 3,
        y: 1,
        w: 9,
        h: 7,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];

    for ht in [false, true] {
        for factors in [
            [(1, 1), (2, 1), (2, 1)],
            [(1, 1), (2, 2), (2, 2)],
            [(1, 1), (1, 2), (1, 2)],
            [(1, 1), (3, 2), (3, 2)],
            // No unit-sampled component: decode still targets the image grid
            // because the factors differ.
            [(1, 2), (2, 1), (2, 1)],
        ] {
            for (precision, sample_type) in [(8, NativeSampleType::U8), (12, NativeSampleType::U16)]
            {
                let encoded =
                    Arc::<[u8]>::from(sampled_fixture(WIDTH, HEIGHT, &factors, precision, ht));
                for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
                    let mut options = BatchDecodeOptions::default();
                    options.workers = NonZeroUsize::new(1);
                    options.layout = layout;
                    let mut decoder = CpuBatchDecoder::new(options);
                    let prepared = decoder
                        .prepare(
                            requests
                                .into_iter()
                                .map(|request| EncodedImage::new(Arc::clone(&encoded), request))
                                .collect(),
                        )
                        .expect("prepare sampled request matrix");
                    let case =
                        format!("ht={ht} factors={factors:?} precision={precision} {layout:?}");
                    assert!(
                        prepared.errors().is_empty(),
                        "{case}: {:?}",
                        prepared.errors()
                    );
                    assert!(
                        prepared.groups().iter().all(|group| {
                            group.info().color == BatchColor::Rgb
                                && group.info().sample_type == sample_type
                                && group.images().iter().all(|image| {
                                    image.preparation_depth() == PreparationDepth::MetadataOnly
                                })
                        }),
                        "{case}"
                    );

                    let result = decoder
                        .decode_prepared(&prepared)
                        .expect("decode sampled request matrix");
                    assert!(result.errors().is_empty(), "{case}: {:?}", result.errors());
                    for source_index in 0..requests.len() {
                        let image = prepared
                            .groups()
                            .iter()
                            .flat_map(j2k::PreparedBatchGroup::images)
                            .find(|image| image.source_index() == source_index)
                            .expect("prepared sampled source");
                        assert_eq!(
                            decoded_samples_for_source(&result, source_index),
                            native_request_oracle(image, layout),
                            "{case} source={source_index}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn subsampled_and_unit_sampled_rgb_tiles_decode_in_separate_groups() {
    let unit = Arc::<[u8]>::from(sampled_rgb_fixture(16, 16, (1, 1), 8, false));
    let sampled = Arc::<[u8]>::from(sampled_rgb_fixture(16, 16, (2, 1), 8, false));
    let mut options = BatchDecodeOptions::default();
    options.workers = NonZeroUsize::new(1);
    options.layout = BatchLayout::Nhwc;
    let mut decoder = CpuBatchDecoder::new(options);
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::clone(&unit)),
            EncodedImage::full(Arc::clone(&sampled)),
            EncodedImage::full(Arc::clone(&unit)),
            EncodedImage::full(sampled),
        ])
        .expect("prepare mixed sampling");
    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());

    // The unit-sampled tiles keep their offset plan; subsampled tiles retain
    // metadata only, so the two never share a backend execution group.
    let mut groups = prepared
        .groups()
        .iter()
        .map(|group| {
            (
                group.source_indices().to_vec(),
                group.images()[0].preparation_depth(),
            )
        })
        .collect::<Vec<_>>();
    groups.sort_unstable_by_key(|(indices, _)| indices[0]);
    assert_eq!(
        groups,
        [
            (vec![0, 2], PreparationDepth::ClassicOffsetPlan),
            (vec![1, 3], PreparationDepth::MetadataOnly),
        ]
    );

    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode mixed sampling");
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    for source_index in 0..4 {
        let image = prepared
            .groups()
            .iter()
            .flat_map(j2k::PreparedBatchGroup::images)
            .find(|image| image.source_index() == source_index)
            .expect("prepared source");
        assert_eq!(
            decoded_samples_for_source(&result, source_index),
            native_request_oracle(image, BatchLayout::Nhwc),
            "source={source_index}"
        );
    }
}

#[test]
fn uniformly_subsampled_images_remain_nonrepresentable() {
    // Native decode shrinks the image to the shared component grid, which the
    // batch's reference-grid geometry does not describe.
    let inputs = [
        sampled_fixture(17, 13, &[(2, 2)], 8, false),
        sampled_fixture(17, 13, &[(2, 1); 3], 8, false),
        sampled_fixture(17, 13, &[(2, 1); 3], 8, true),
    ];
    let mut options = BatchDecodeOptions::default();
    options.workers = NonZeroUsize::new(1);
    let prepared = CpuBatchDecoder::new(options)
        .prepare(
            inputs
                .into_iter()
                .map(|bytes| EncodedImage::full(Arc::<[u8]>::from(bytes)))
                .collect(),
        )
        .expect("prepare uniformly subsampled inputs");
    assert!(prepared.groups().is_empty());
    let reasons = prepared
        .errors()
        .iter()
        .map(|error| match error.source {
            BatchItemError::NonRepresentableBatchOutput { reason } => (error.index, reason),
            _ => panic!("expected representability error: {error:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reasons,
        [
            (0, NonRepresentableReason::ComponentSubsampling),
            (1, NonRepresentableReason::ComponentSubsampling),
            (2, NonRepresentableReason::ComponentSubsampling),
        ]
    );
}
