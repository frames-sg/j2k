// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodeRetainedInput, NativeEncodeSession,
};
use crate::EncodeError;

fn build_plan_with_cap(cap: usize) -> NativeEncodePipelineResult<TypedI64HighBitPlan> {
    let data = [0_u8; 32];
    let planes = [EncodeTypedComponentPlane {
        data: &data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 25,
        signed: false,
    }];
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("valid typed i64 plan cap");
    TypedI64HighBitPlan::try_new(&planes, &EncodeOptions::default(), 1, 0, &session)
}

#[test]
fn plan_construction_accepts_exact_live_peak_and_rejects_one_byte_less() {
    let plan = build_plan_with_cap(crate::DEFAULT_MAX_CODEC_BYTES)
        .expect("discover typed i64 plan capacity");
    let mut transient_steps = Vec::<QuantStepSize>::new();
    transient_steps
        .try_reserve_exact(4)
        .expect("small typed i64 step fixture");
    let exact_cap = plan.retained_bytes().expect("retained typed i64 plan")
        + transient_steps.capacity() * core::mem::size_of::<QuantStepSize>();
    build_plan_with_cap(exact_cap).expect("exact typed i64 construction peak");
    let error = build_plan_with_cap(exact_cap - 1)
        .err()
        .expect("one byte below typed i64 construction peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}
