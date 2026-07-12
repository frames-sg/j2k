// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::encode::NativeEncodePipelineError;
use crate::{EncodeError, NativeEncodeRetainedInput};
use alloc::vec;

fn session_with_cap(cap: usize) -> NativeEncodeSession<'static> {
    NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), cap)
        .expect("test session baseline")
}

fn loop_request<'a, 'input>(
    options: &'a EncodeOptions,
    component_sample_info: &'a [EncodeComponentSampleInfo],
    session: &'a NativeEncodeSession<'input>,
) -> LoopPlanRequest<'a, 'input> {
    LoopPlanRequest {
        width: 8,
        height: 8,
        tile_width: 4,
        tile_height: 4,
        num_components: 1,
        bit_depth: 8,
        options,
        roi_regions: &[],
        component_sample_info,
        block_coding_mode: BlockCodingMode::Classic,
        session,
    }
}

#[test]
fn loop_plan_accepts_exact_observed_peak_and_rejects_cap_minus_one() {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        quality_layer_byte_targets: {
            let mut values = Vec::new();
            values.try_reserve_exact(1).expect("small option owner");
            values.push(32);
            values
        },
        ..EncodeOptions::default()
    };
    let component_info = [EncodeComponentSampleInfo {
        bit_depth: 8,
        signed: false,
    }];
    let discovery = session_with_cap(crate::DEFAULT_MAX_CODEC_BYTES);
    let discovered = build_loop_plan(&loop_request(&options, &component_info, &discovery))
        .expect("discover loop plan");
    let (steps, component_steps) =
        build_step_graph(8, 1, 1, &options, &component_info).expect("discover step graph");
    let step_peak = step_graph_retained_bytes(&steps, &component_steps).expect("step bytes");
    let exact_cap = step_peak.max(discovered.retained_bytes());

    let exact = session_with_cap(exact_cap);
    build_loop_plan(&loop_request(&options, &component_info, &exact))
        .expect("loop plan at exact cap");

    let over = session_with_cap(exact_cap - 1);
    let error = build_loop_plan(&loop_request(&options, &component_info, &over))
        .err()
        .expect("loop plan is one byte over cap");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge { .. })
    ));
}

#[test]
fn final_plan_accepts_exact_observed_peak_and_rejects_cap_minus_one() {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let tile_bodies = Vec::new();
    let discovery = session_with_cap(crate::DEFAULT_MAX_CODEC_BYTES);
    let loop_plan =
        build_loop_plan(&loop_request(&options, &[], &discovery)).expect("discover loop plan");
    let discovered = loop_plan
        .into_final_plan(&FinalPlanRequest {
            width: 8,
            height: 8,
            tile_width: 4,
            tile_height: 4,
            num_components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            roi_regions: &[],
            component_sample_info: &[],
            block_coding_mode: BlockCodingMode::Classic,
            tile_bodies: &tile_bodies,
            session: &discovery,
        })
        .expect("discover final plan");
    let (steps, component_steps) =
        build_step_graph(8, 1, 1, &options, &[]).expect("discover step graph");
    let exact_cap = step_graph_retained_bytes(&steps, &component_steps).expect("step bytes")
        + encode_params_retained_bytes(&discovered.params).expect("parameter bytes")
        + discovered.quant_params.capacity() * core::mem::size_of::<(u16, u16)>();

    let exact = session_with_cap(exact_cap);
    let loop_plan = build_loop_plan(&loop_request(&options, &[], &exact)).expect("exact loop plan");
    loop_plan
        .into_final_plan(&FinalPlanRequest {
            width: 8,
            height: 8,
            tile_width: 4,
            tile_height: 4,
            num_components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            roi_regions: &[],
            component_sample_info: &[],
            block_coding_mode: BlockCodingMode::Classic,
            tile_bodies: &tile_bodies,
            session: &exact,
        })
        .expect("final plan at exact cap");

    let over = session_with_cap(exact_cap - 1);
    let loop_plan =
        build_loop_plan(&loop_request(&options, &[], &over)).expect("loop plan remains below cap");
    let error = loop_plan
        .into_final_plan(&FinalPlanRequest {
            width: 8,
            height: 8,
            tile_width: 4,
            tile_height: 4,
            num_components: 1,
            bit_depth: 8,
            signed: false,
            options: &options,
            roi_regions: &[],
            component_sample_info: &[],
            block_coding_mode: BlockCodingMode::Classic,
            tile_bodies: &tile_bodies,
            session: &over,
        })
        .err()
        .expect("final plan is one byte over cap");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge {
            what: "multi-tile final plan construction",
            ..
        })
    ));
}

#[test]
fn precinct_validation_happens_before_loop_allocation() {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        precinct_exponents: vec![(6, 6)],
        ..EncodeOptions::default()
    };
    let session = session_with_cap(0);
    let error = build_loop_plan(&loop_request(&options, &[], &session))
        .err()
        .expect("invalid precinct plan");
    assert!(matches!(
        error,
        NativeEncodePipelineError::InvalidInput(
            "precinct exponent count must match resolution level count"
        )
    ));
}
