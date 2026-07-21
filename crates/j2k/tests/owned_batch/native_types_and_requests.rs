// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, DecodeRequest, Downscale, EncodedImage,
    NativeSampleType, PreparationDepth, Rect,
};

use super::fixtures::htj2k_native_fixture;
use super::oracles::{decoded_samples_for_source, native_request_oracle};

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
                let options = BatchDecodeOptions {
                    workers: NonZeroUsize::new(1),
                    layout,
                    ..BatchDecodeOptions::default()
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
                assert!(prepared.errors().is_empty());
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
                assert!(result.errors().is_empty());
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
                assert_eq!(stats.flattened_group_plans(), requests.len() as u64);
                assert_eq!(stats.output_group_allocations(), requests.len() as u64);
                assert_eq!(stats.output_compaction_copied_samples(), 0);
            }
        }
    }
}
