// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reduced regions that end where a tile begins once tile bounds are rounded
//! onto the reduced grid.
//!
//! The `OpenJPH` fixtures are 19 x 13 images in 11 x 7 tiles. At half scale the
//! second tile column starts at column 6, but tile selection rounds bounds
//! outward and keeps that tile for regions ending at column 6; the tile then
//! contributes no samples.

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, DecodeRequest, Downscale, EncodedImage, Rect,
};
use j2k_test_support::openjph_batch_fixtures;

use super::oracles::{decoded_samples_for_source, native_request_oracle};

fn assert_batch_matches_single_image(encoded: &Arc<[u8]>, requests: &[DecodeRequest], case: &str) {
    let mut options = BatchDecodeOptions::default();
    options.workers = NonZeroUsize::new(4);
    options.layout = BatchLayout::Nhwc;
    let mut decoder = CpuBatchDecoder::new(options);
    let prepared = decoder
        .prepare(
            requests
                .iter()
                .map(|&request| EncodedImage::new(Arc::clone(encoded), request))
                .collect(),
        )
        .unwrap_or_else(|error| panic!("{case}: prepare: {error}"));
    let failures = prepared
        .errors()
        .iter()
        .take(3)
        .map(|error| format!("{:?}: {error}", requests[error.index]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{case}: {} of {} requests failed to prepare, first {failures:?}",
        prepared.errors().len(),
        requests.len()
    );

    let result = decoder
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("{case}: decode: {error}"));
    assert!(result.errors().is_empty(), "{case}: {:?}", result.errors());
    for image in prepared
        .groups()
        .iter()
        .flat_map(j2k::PreparedBatchGroup::images)
    {
        let source_index = image.source_index();
        assert_eq!(
            decoded_samples_for_source(&result, source_index),
            native_request_oracle(image, BatchLayout::Nhwc),
            "{case} request={:?}",
            requests[source_index]
        );
    }
}

#[test]
fn every_half_scale_region_of_a_tiled_image_prepares_and_decodes() {
    // The reproduction in #298: about a quarter of these failed to prepare.
    let fixture = openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-u8-53-raw")
        .expect("tiled RGB fixture");
    let (width, height) = (fixture.width, fixture.height);
    let requests = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| {
            (1..=height - y).flat_map(move |h| {
                (1..=width - x).map(move |w| DecodeRequest::RegionReduced {
                    roi: Rect { x, y, w, h },
                    scale: Downscale::Half,
                })
            })
        })
        .collect::<Vec<_>>();
    let encoded = Arc::<[u8]>::from(fixture.encoded);
    assert_batch_matches_single_image(&encoded, &requests, fixture.name);
}

#[test]
fn reduced_regions_ending_at_rounded_tile_edges_prepare_for_every_fixture() {
    // Regions that end at, start at, or straddle the tile edges at x = 11 and
    // y = 7, at both reductions the two-level fixtures allow.
    let regions = [
        (0, 0, 11, 1),
        (0, 0, 11, 7),
        (0, 0, 19, 7),
        (0, 0, 11, 13),
        (10, 6, 2, 2),
        (11, 7, 8, 6),
        (5, 3, 7, 5),
        (0, 0, 19, 13),
    ];
    for fixture in openjph_batch_fixtures() {
        let requests = [Downscale::Half, Downscale::Quarter]
            .into_iter()
            .flat_map(|scale| {
                regions
                    .iter()
                    .map(move |&(x, y, w, h)| DecodeRequest::RegionReduced {
                        roi: Rect { x, y, w, h },
                        scale,
                    })
            })
            .collect::<Vec<_>>();
        let encoded = Arc::<[u8]>::from(fixture.encoded);
        assert_batch_matches_single_image(&encoded, &requests, fixture.name);
    }
}
