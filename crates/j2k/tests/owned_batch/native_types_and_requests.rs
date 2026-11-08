// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, DecodeRequest, Downscale, EncodedImage,
    NativeSampleType, PreparationDepth, Rect,
};

use super::fixtures::htj2k_native_fixture;
use super::oracles::{decoded_samples_for_source, native_request_oracle};

#[test]
fn exact_operations_preserve_native_samples_and_independent_sessions() {
    use super::fixtures::htj2k_native_fixture_with_levels;
    use super::oracles::native_exact_request_oracle;
    use j2k::{prepare_batch, J2kDecodeControl, PreparedImageOperation};
    let regions = [
        Rect::full((65, 49)),
        Rect {
            x: 9,
            y: 11,
            w: 33,
            h: 25,
        },
    ];
    for (precision, signed) in [(7, false), (12, false), (16, true)] {
        let encoded: Arc<[u8]> =
            htj2k_native_fixture_with_levels(3, precision, signed, 65, 49, 4).into();
        let prepared = prepare_batch(
            vec![EncodedImage::full(encoded)],
            BatchDecodeOptions::default(),
        )
        .unwrap();
        assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
        let image = &prepared.groups()[0].images()[0];
        std::thread::scope(|scope| {
            for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
                scope.spawn(move || {
                    let mut options = BatchDecodeOptions::default();
                    options.layout = layout;
                    options.workers = NonZeroUsize::new(1);
                    let mut session = CpuBatchDecoder::new(options);
                    let mut control = J2kDecodeControl::default();
                    control.max_working_memory_bytes = NonZeroUsize::new(8 << 20);
                    for levels in [4, 2] {
                        let operations = regions.map(|roi| {
                            let operation = PreparedImageOperation::new(image, roi, levels);
                            if layout == BatchLayout::Nhwc {
                                operation.with_control(&control)
                            } else {
                                operation
                            }
                        });
                        let result = session.decode_operations(&operations).unwrap();
                        assert!(result.errors().is_empty(), "{:?}", result.errors());
                        for (index, roi) in regions.into_iter().enumerate() {
                            let factor = 1_u32 << levels;
                            let output = Rect {
                                x: roi.x / factor,
                                y: roi.y / factor,
                                w: (roi.x + roi.w).div_ceil(factor) - roi.x / factor,
                                h: (roi.y + roi.h).div_ceil(factor) - roi.y / factor,
                            };
                            assert_eq!(
                                decoded_samples_for_source(&result, index),
                                native_exact_request_oracle(image, output, levels, layout),
                                "precision={precision} signed={signed} layout={layout:?}"
                            );
                        }
                    }
                });
            }
        });
    }
}

#[test]
fn prepared_htj2k_gray_and_rgb_support_native_types_and_requests_exactly() {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 6;
    let roi = Rect {
        x: 1,
        y: 1,
        w: 5,
        h: 3,
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

    for components in [1_u16, 3] {
        for (precision, signed, sample_type) in [
            (8, false, NativeSampleType::U8),
            (12, false, NativeSampleType::U16),
            (16, true, NativeSampleType::I16),
        ] {
            let encoded = Arc::<[u8]>::from(htj2k_native_fixture(
                components, precision, signed, WIDTH, HEIGHT,
            ));
            for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
                let options = {
                    let mut options = BatchDecodeOptions::default();
                    options.workers = NonZeroUsize::new(1);
                    options.layout = layout;
                    options
                };
                let mut decoder = CpuBatchDecoder::new(options);
                let prepared = decoder
                    .prepare(
                        requests
                            .into_iter()
                            .map(|request| EncodedImage::new(Arc::clone(&encoded), request))
                            .collect(),
                    )
                    .expect("prepare Gray/RGB request matrix");
                assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
                assert!(prepared.groups().iter().all(|group| {
                    group.info().sample_type == sample_type
                        && group.info().color.channels() == components as usize
                        && group.images().iter().all(|image| {
                            image.preparation_depth() == PreparationDepth::Htj2kOffsetPlan
                        })
                }));

                let result = decoder
                    .decode_prepared(&prepared)
                    .expect("decode Gray/RGB request matrix");
                assert!(result.errors().is_empty(), "{:?}", result.errors());
                for source_index in 0..requests.len() {
                    let prepared_image = prepared
                        .groups()
                        .iter()
                        .flat_map(j2k::PreparedBatchGroup::images)
                        .find(|image| image.source_index() == source_index)
                        .expect("prepared Gray/RGB source");
                    assert_eq!(
                        decoded_samples_for_source(&result, source_index),
                        native_request_oracle(prepared_image, layout),
                        "components={components} precision={precision} signed={signed} {layout:?} source={source_index}"
                    );
                }
                let stats = decoder.workspace_stats();
                assert_eq!(stats.preparation_calls(), requests.len() as u64);
                assert_eq!(stats.prepared_plan_decode_calls(), requests.len() as u64);
                // Each request is a group of one image, which decodes its
                // plan with inner parallelism and flattens no payloads.
                assert_eq!(stats.flattened_group_plans(), 0);
                assert_eq!(stats.output_group_allocations(), requests.len() as u64);
                assert_eq!(stats.output_compaction_copied_samples(), 0);
            }
        }
    }
}
