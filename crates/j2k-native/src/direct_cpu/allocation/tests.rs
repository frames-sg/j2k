// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{component_band_count, DirectAllocationBudget, DirectWorkspaceBudget};
use crate::error::{DecodeError, ValidationError};
use crate::{
    HtOwnedSubBandPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kOwnedSubBandPlan,
    J2kRect, DEFAULT_MAX_DECODE_BYTES,
};

#[test]
fn aggregate_budget_has_an_exact_shared_cap_boundary() {
    let mut budget = DirectAllocationBudget {
        bytes: DEFAULT_MAX_DECODE_BYTES - 1,
    };
    budget.include_bytes(1).expect("exact boundary fits");
    assert_eq!(
        budget.include_bytes(1),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn actual_scalar_workspace_uses_the_remaining_direct_budget() {
    let budget = DirectWorkspaceBudget {
        base_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
        peak_bytes: DEFAULT_MAX_DECODE_BYTES - 1,
    };
    budget.validate_workspace(1).expect("exact boundary fits");
    assert_eq!(
        budget.validate_workspace(2),
        Err(DecodeError::Validation(ValidationError::ImageTooLarge))
    );
}

#[test]
fn mixed_entropy_steps_reserve_one_shared_band_owner() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let plan = J2kDirectGrayscalePlan {
        dimensions: (2, 2),
        bit_depth: 8,
        steps: vec![
            J2kDirectGrayscaleStep::ClassicSubBand(J2kOwnedSubBandPlan {
                band_id: 7,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
            J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan {
                band_id: 7,
                rect,
                width: 2,
                height: 2,
                irreversible_midpoint: false,
                jobs: Vec::new(),
            }),
        ],
    };

    assert_eq!(component_band_count(&plan).expect("band count"), 1);
}
