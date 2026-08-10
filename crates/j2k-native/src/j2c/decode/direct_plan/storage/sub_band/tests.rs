// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{strip_classic_payload_owners, strip_grayscale_payload_owners};
use crate::{
    HtOwnedSubBandPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kOwnedSubBandPlan,
    J2kRect,
};

#[test]
fn referenced_ownership_strips_each_coder_without_rejecting_the_other() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let mut plan = J2kDirectGrayscalePlan {
        dimensions: (2, 2),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::ClassicSubBand(J2kOwnedSubBandPlan {
                band_id: 1,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 2,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
        ],
    };

    assert_eq!(
        strip_grayscale_payload_owners(&mut plan, &[]).expect("strip HT owners"),
        0
    );
    assert_eq!(
        strip_classic_payload_owners(&mut plan).expect("strip classic owners"),
        0
    );
}
