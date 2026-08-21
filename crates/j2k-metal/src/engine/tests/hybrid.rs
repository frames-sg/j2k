// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::direct_grayscale_execute::execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test;
use super::super::{
    execute_hybrid_cpu_tier1_direct_color_plan_batch, flattened_hybrid_cpu_decode_batches_for_test,
    hybrid_cpu_decode_inputs_for_test, hybrid_cpu_decode_worker_count,
    hybrid_cpu_decode_worker_inits_for_test, hybrid_stacked_component_batches_for_test,
    new_command_buffer, prepare_direct_color_plan,
    reset_flattened_hybrid_cpu_decode_batches_for_test, reset_hybrid_cpu_decode_inputs_for_test,
    reset_hybrid_cpu_decode_worker_inits_for_test, reset_hybrid_stacked_component_batches_for_test,
    reset_stacked_component_batches_for_test, reset_thread_hybrid_cpu_decode_inputs_for_test,
    should_flatten_hybrid_cpu_tier1_color_batch, stacked_component_batches_for_test,
    thread_hybrid_cpu_decode_inputs_for_test, try_encode_stacked_mct_rgb8_direct_color_batch,
    with_runtime, DirectColorBatchCommandBuffers, DirectHybridStageTimings, DirectTier1Mode,
    PreparedDirectColorPlan, StackedDirectColorBatchRequest,
};
use super::{
    hybrid_support::{prepared_direct_color_tier1_input_count, HYBRID_COUNTER_TEST_LOCK},
    runtime::should_run_metal_runtime,
};
use j2k_core::PixelFormat;
use j2k_native::{
    encode, DecodeSettings, DecoderContext, EncodeOptions, Image, J2kWaveletTransform,
};
use std::sync::Arc;

#[test]
fn hybrid_rgb8_batch_uses_stacked_component_graph() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_stacked_component_batches_for_test();
    reset_hybrid_cpu_decode_worker_inits_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared],
        PixelFormat::Rgb8,
    )
    .expect("hybrid RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert!(
        hybrid_stacked_component_batches_for_test() >= 3,
        "hybrid RGB batch should stack each component plane instead of encoding each tile/component serially"
    );
    assert!(
        hybrid_cpu_decode_worker_inits_for_test() > 0,
        "hybrid RGB batch should use worker-local CPU decode scratch instead of per-input decode/flatten"
    );
}

#[test]
fn hybrid_rgb8_repeated_batch_decodes_shared_tier1_inputs_once() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let prepared = Arc::new(prepare_direct_color_plan(&plan).expect("prepared color plan"));
    let unique_tier1_inputs = prepared_direct_color_tier1_input_count(&prepared);
    assert!(
        unique_tier1_inputs > 0,
        "fixture should have Tier-1 inputs to decode"
    );
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();
    reset_thread_hybrid_cpu_decode_inputs_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared.clone(), prepared],
        PixelFormat::Rgb8,
    )
    .expect("hybrid repeated RGB8 batch");

    assert_eq!(surfaces.len(), 3);
    assert!(
        hybrid_cpu_decode_inputs_for_test() >= unique_tier1_inputs,
        "repeated RGB hybrid batches should decode the shared coefficient inputs"
    );
}

#[test]
fn hybrid_rgb8_distinct_batch_keeps_tier1_inputs_separate() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes_a = encode(
        &j2k_test_support::gradient_variant_u8(32, 32, 3, 0),
        32,
        32,
        3,
        8,
        false,
        &options,
    )
    .expect("encode first rgb8");
    let bytes_b = encode(
        &j2k_test_support::gradient_variant_u8(32, 32, 3, 7),
        32,
        32,
        3,
        8,
        false,
        &options,
    )
    .expect("encode second rgb8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_color_plan_with_context(&mut context_a)
        .expect("first direct color plan");
    let plan_b = image_b
        .build_direct_color_plan_with_context(&mut context_b)
        .expect("second direct color plan");
    let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
    let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
    let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
        + prepared_direct_color_tier1_input_count(&prepared_b);
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared_a, prepared_b],
        PixelFormat::Rgb8,
    )
    .expect("hybrid distinct RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert_ne!(
        surfaces[0].as_bytes().expect("surface byte access"),
        surfaces[1].as_bytes().expect("surface byte access"),
        "distinct RGB inputs must not reuse the first tile's decoded coefficients"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        expected_inputs,
        "distinct RGB hybrid batches should decode each tile's own Tier-1 inputs"
    );
}

#[test]
fn incompatible_later_color_component_does_not_encode_partial_stacked_work() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 3);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes = encode(&pixels, 32, 32, 3, 8, false, &options).expect("encode rgb8");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    let plan = image
        .build_direct_color_plan_with_context(&mut context)
        .expect("direct color plan");
    let first = Arc::new(prepare_direct_color_plan(&plan).expect("first prepared color plan"));
    let mut incompatible =
        prepare_direct_color_plan(&plan).expect("incompatible prepared color plan");
    incompatible.component_plans[1].dimensions = (31, 32);
    let plans = [first, Arc::new(incompatible)];
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_stacked_component_batches_for_test();

    with_runtime(|runtime| {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut stage_timings = DirectHybridStageTimings::default();
        let mut retained_buffers = Vec::with_capacity(128);
        let mut status_checks = Vec::with_capacity(32);
        let mut scratch_buffers = Vec::with_capacity(64);
        let result =
            try_encode_stacked_mct_rgb8_direct_color_batch(StackedDirectColorBatchRequest {
                runtime,
                command_buffers: DirectColorBatchCommandBuffers::single(&command_buffer),
                plans: &plans,
                tier1_mode: DirectTier1Mode::Metal,
                force_flattened_cpu_tier1: false,
                stage_timings: &mut stage_timings,
                retained_buffers: &mut retained_buffers,
                status_checks: &mut status_checks,
                scratch_buffers: &mut scratch_buffers,
            })?;
        assert!(result.is_none(), "incompatible component must use fallback");
        assert!(
            retained_buffers.is_empty() && status_checks.is_empty() && scratch_buffers.is_empty(),
            "preflight rejection must not retain partial stacked execution resources"
        );
        Ok(())
    })
    .expect("probe stacked color preflight");

    assert_eq!(
        stacked_component_batches_for_test(),
        0,
        "fallback must be selected before any stacked component commands are encoded"
    );
}

#[test]
fn hybrid_rgb8_flattened_cpu_tier1_batch_uses_one_decode_queue() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels_a = j2k_test_support::gradient_variant_u8(32, 32, 3, 0);
    let pixels_b = j2k_test_support::gradient_variant_u8(32, 32, 3, 11);
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let bytes_a = encode(&pixels_a, 32, 32, 3, 8, false, &options).expect("encode first rgb8");
    let bytes_b = encode(&pixels_b, 32, 32, 3, 8, false, &options).expect("encode second rgb8");
    let image_a = Image::new(&bytes_a, &DecodeSettings::default()).expect("first image");
    let image_b = Image::new(&bytes_b, &DecodeSettings::default()).expect("second image");
    let mut context_a = DecoderContext::default();
    let mut context_b = DecoderContext::default();
    let plan_a = image_a
        .build_direct_color_plan_with_context(&mut context_a)
        .expect("first direct color plan");
    let plan_b = image_b
        .build_direct_color_plan_with_context(&mut context_b)
        .expect("second direct color plan");
    let prepared_a = Arc::new(prepare_direct_color_plan(&plan_a).expect("first prepared"));
    let prepared_b = Arc::new(prepare_direct_color_plan(&plan_b).expect("second prepared"));
    let expected_inputs = prepared_direct_color_tier1_input_count(&prepared_a)
        + prepared_direct_color_tier1_input_count(&prepared_b);
    let _guard = HYBRID_COUNTER_TEST_LOCK
        .lock()
        .expect("hybrid counter lock");
    reset_hybrid_cpu_decode_inputs_for_test();
    reset_flattened_hybrid_cpu_decode_batches_for_test();

    let surfaces = execute_flattened_hybrid_cpu_tier1_direct_color_plan_batch_for_test(
        &[prepared_a, prepared_b],
        PixelFormat::Rgb8,
    )
    .expect("flattened hybrid distinct RGB8 batch");

    assert_eq!(surfaces.len(), 2);
    assert_ne!(
        surfaces[0].as_bytes().expect("surface byte access"),
        surfaces[1].as_bytes().expect("surface byte access"),
        "flattened distinct RGB hybrid batches must keep each tile's coefficients separate"
    );
    assert!(
        hybrid_cpu_decode_inputs_for_test() >= expected_inputs,
        "flattened RGB hybrid batches should still decode every distinct Tier-1 input"
    );
    assert!(
        flattened_hybrid_cpu_decode_batches_for_test() >= 1,
        "flattened RGB hybrid should collect Tier-1 work through the flattened CPU decode queue"
    );
}

#[test]
fn flattened_cpu_tier1_default_gate_targets_large_distinct_batches_only() {
    fn color_plan(width: u32, height: u32) -> Arc<PreparedDirectColorPlan> {
        Arc::new(PreparedDirectColorPlan {
            dimensions: (width, height),
            bit_depths: [8, 8, 8],
            alpha_bit_depth: None,
            signed: false,
            mct: true,
            transform: J2kWaveletTransform::Reversible53,
            component_plans: Vec::new(),
        })
    }

    let repeated = vec![color_plan(1024, 1024); 16];
    assert!(
        !should_flatten_hybrid_cpu_tier1_color_batch(&repeated),
        "repeated RGB batches already win through shared Tier-1 decode and should not use the flattened distinct scheduler"
    );

    let small_distinct = (0..16).map(|_| color_plan(256, 256)).collect::<Vec<_>>();
    assert!(
        !should_flatten_hybrid_cpu_tier1_color_batch(&small_distinct),
        "small RGB batches measured slower with flattened Tier-1 and should stay on the grouped path"
    );

    let large_distinct = (0..16).map(|_| color_plan(1024, 1024)).collect::<Vec<_>>();
    assert!(
        should_flatten_hybrid_cpu_tier1_color_batch(&large_distinct),
        "large distinct RGB explicit hybrid batches measured faster with flattened Tier-1"
    );
}

#[test]
fn hybrid_cpu_decode_worker_count_allows_two_way_small_batch_parallelism() {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    if available < 2 {
        return;
    }

    assert_eq!(
        hybrid_cpu_decode_worker_count(2),
        2,
        "two independent hybrid CPU Tier-1 inputs should be able to use two workers"
    );
}
