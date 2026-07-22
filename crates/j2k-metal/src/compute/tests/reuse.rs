// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    execute_hybrid_cpu_tier1_direct_color_plan_batch, hybrid_repeated_output_blits_for_test,
    prepare_direct_color_plan, reset_hybrid_cpu_decode_inputs_for_test,
    reset_hybrid_repeated_output_blits_for_test, reset_thread_hybrid_cpu_decode_inputs_for_test,
    thread_hybrid_cpu_decode_inputs_for_test,
};
use super::{
    hybrid_support::{
        cached_direct_color_tier1_input_count, prepared_direct_color_tier1_input_count,
        HYBRID_COUNTER_TEST_LOCK,
    },
    runtime::should_run_metal_runtime,
};
use j2k_core::PixelFormat;
use j2k_native::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};
use std::sync::Arc;

#[test]
fn hybrid_rgb8_reused_plan_caches_cpu_tier1_inputs_across_calls() {
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
        &[prepared.clone(), prepared.clone()],
        PixelFormat::Rgb8,
    )
    .expect("first hybrid repeated RGB8 batch");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(
        cached_direct_color_tier1_input_count(&prepared),
        unique_tier1_inputs,
        "first RGB hybrid call should cache every decoded CPU Tier-1 input"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        unique_tier1_inputs,
        "first RGB hybrid call should decode each shared Tier-1 input once"
    );

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[prepared.clone(), prepared.clone()],
        PixelFormat::Rgb8,
    )
    .expect("second hybrid repeated RGB8 batch");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(
        cached_direct_color_tier1_input_count(&prepared),
        unique_tier1_inputs,
        "second RGB hybrid call should keep every decoded CPU Tier-1 input cached"
    );
    assert_eq!(
        thread_hybrid_cpu_decode_inputs_for_test(),
        unique_tier1_inputs,
        "second RGB hybrid call must reuse cached CPU Tier-1 coefficients without re-decoding"
    );
}

#[test]
fn hybrid_rgb8_repeated_batch_decodes_once_and_blits_distinct_outputs() {
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
    reset_hybrid_repeated_output_blits_for_test();

    let surfaces = execute_hybrid_cpu_tier1_direct_color_plan_batch(
        &[
            prepared.clone(),
            prepared.clone(),
            prepared.clone(),
            prepared,
        ],
        PixelFormat::Rgb8,
    )
    .expect("hybrid repeated RGB8 batch");

    assert_eq!(surfaces.len(), 4);
    let surface_bytes = surfaces[0].as_bytes().expect("surface byte access").len();
    let offsets = surfaces
        .iter()
        .map(|surface| {
            surface
                .metal_buffer_trusted()
                .expect("resident Metal surface")
                .1
        })
        .collect::<Vec<_>>();
    assert_eq!(
        offsets,
        (0..surfaces.len())
            .map(|index| index * surface_bytes)
            .collect::<Vec<_>>(),
        "repeated outputs must retain distinct Metal buffer offsets"
    );
    for surface in &surfaces[1..] {
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            surfaces[0].as_bytes().expect("surface byte access"),
            "repeated outputs should remain byte-identical"
        );
    }
    assert_eq!(
        hybrid_repeated_output_blits_for_test(),
        2,
        "repeated RGB hybrid batches should duplicate packed output surfaces with logarithmic Metal blit ranges"
    );
}
