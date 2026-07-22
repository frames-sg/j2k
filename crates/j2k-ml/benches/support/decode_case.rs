// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
use j2k::{DecodeRequest, Downscale, PreparedBatch, Rect};
use j2k_ml::BurnBatchDecode;

pub(crate) fn requests(
    dimensions: (u32, u32),
    include_region_reduced: bool,
) -> Vec<(&'static str, DecodeRequest, u32)> {
    let (width, height) = dimensions;
    let roi = centered_roi(dimensions);
    let mut requests = vec![
        ("full", DecodeRequest::Full, width.saturating_mul(height)),
        (
            "roi",
            DecodeRequest::Region { roi },
            roi.w.saturating_mul(roi.h),
        ),
        (
            "reduced",
            DecodeRequest::Reduced {
                scale: Downscale::Half,
            },
            width.div_ceil(2).saturating_mul(height.div_ceil(2)),
        ),
    ];
    if include_region_reduced {
        requests.push((
            "roi_reduced",
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
            roi.w.div_ceil(2).saturating_mul(roi.h.div_ceil(2)),
        ));
    }
    requests
}

pub(crate) fn decoded_pixels_per_batch(output_pixels: u32, batch_size: usize) -> u64 {
    u64::from(output_pixels)
        .saturating_mul(u64::try_from(batch_size).expect("benchmark batch size fits u64"))
}

pub(crate) fn require_prepared_success(prepared: &PreparedBatch) {
    assert!(
        prepared.errors().is_empty(),
        "benchmark preparation returned indexed errors: {:?}",
        prepared.errors()
    );
    assert_eq!(
        prepared.groups().len(),
        1,
        "benchmark workload must prepare exactly one homogeneous group"
    );
}

pub(crate) fn require_burn_success<B: Backend>(decoded: BurnBatchDecode<B>) -> BurnBatchDecode<B> {
    assert!(
        decoded.errors.is_empty(),
        "benchmark adapter returned indexed errors: {:?}",
        decoded.errors
    );
    assert!(
        decoded.group_errors.is_empty(),
        "benchmark adapter returned group errors: {:?}",
        decoded.group_errors
    );
    assert_eq!(
        decoded.groups.len(),
        1,
        "benchmark workload must materialize exactly one Burn tensor group"
    );
    decoded
}

fn centered_roi((width, height): (u32, u32)) -> Rect {
    Rect {
        x: width / 4,
        y: height / 4,
        w: (width / 2).max(1),
        h: (height / 2).max(1),
    }
}
