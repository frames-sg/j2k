// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::*;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodeRetainedInput, NativeEncodeSession,
};
use crate::EncodeError;

fn pixels() -> Vec<u8> {
    let values = [0_u32, 1 << 24, (1 << 25) - 1, 123];
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn deinterleave_with_cap(cap: usize) -> NativeEncodePipelineResult<Vec<Vec<i64>>> {
    let pixels = pixels();
    let session = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("valid test cap");
    try_deinterleave_to_i64(&I64DeinterleaveRequest {
        pixels: &pixels,
        num_pixels: 2,
        num_components: 2,
        bit_depth: 25,
        signed: false,
        retained_base_bytes: 7,
        session: &session,
    })
}

#[test]
fn deinterleave_is_value_exact_and_enforces_actual_capacity_peak() {
    let discovered = deinterleave_with_cap(crate::DEFAULT_MAX_CODEC_BYTES)
        .expect("discover exact i64 component capacity");
    assert_eq!(
        discovered,
        vec![
            vec![-(1_i64 << 24), (1_i64 << 24) - 1],
            vec![0, 123 - (1_i64 << 24)],
        ]
    );
    let exact_cap = 7 + component_planes_retained_bytes(&discovered).expect("component bytes");
    deinterleave_with_cap(exact_cap).expect("exact deinterleave capacity");
    let error = deinterleave_with_cap(exact_cap - 1)
        .expect_err("one byte below the deinterleave peak must fail");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}
