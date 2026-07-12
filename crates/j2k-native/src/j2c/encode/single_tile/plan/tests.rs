// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{build_single_tile_plan, validate_encode_request, ValidatedEncodeRoute};
use crate::j2c::encode::{
    BlockCodingMode, EncodeOptions, EncodeRoiRegion, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeRetainedInput, NativeEncodeSession,
};

fn build_roi_plan(
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<super::SingleTilePlan> {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        use_ht_block_coding: true,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    };
    let validated = validate_encode_request(
        64,
        8,
        8,
        1,
        8,
        &options,
        BlockCodingMode::HighThroughput,
        &[],
        session,
    )?;
    let ValidatedEncodeRoute::SingleTile(validated) = validated else {
        return Err(NativeEncodePipelineError::internal_invariant(
            "plan test unexpectedly routed to multi-tile encode",
        ));
    };
    build_single_tile_plan(
        validated,
        8,
        8,
        1,
        8,
        false,
        &options,
        BlockCodingMode::HighThroughput,
        &[EncodeRoiRegion {
            component: 0,
            x: 1,
            y: 1,
            width: 4,
            height: 4,
            shift: 12,
        }],
        &[],
        session,
    )
}

#[test]
fn single_tile_plan_construction_accepts_its_exact_measured_cap() {
    let discovery = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("plan discovery session");
    let discovered = build_roi_plan(&discovery).expect("discover retained plan size");
    let plan_bytes = super::super::ownership::single_tile_plan_retained_bytes(&discovered)
        .expect("measure retained plan");

    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), plan_bytes)
        .expect("exact plan session");
    let plan = build_roi_plan(&exact).expect("exact-cap plan construction");
    assert_eq!(
        super::super::ownership::single_tile_plan_retained_bytes(&plan)
            .expect("measure exact-cap plan"),
        plan_bytes
    );
}

#[test]
fn single_tile_plan_construction_rejects_one_byte_below_measured_cap() {
    let discovery = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("plan discovery session");
    let discovered = build_roi_plan(&discovery).expect("discover retained plan size");
    let plan_bytes = super::super::ownership::single_tile_plan_retained_bytes(&discovered)
        .expect("measure retained plan");
    let cap = plan_bytes.checked_sub(1).expect("plan retains bytes");
    let constrained = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("constrained plan session");

    let Err(error) = build_roi_plan(&constrained) else {
        panic!("cap-minus-one plan construction unexpectedly succeeded");
    };
    assert!(matches!(
        error.into_encode_error(),
        crate::EncodeError::AllocationTooLarge {
            requested,
            cap: actual_cap,
            ..
        } if requested > actual_cap && actual_cap == cap
    ));
}
