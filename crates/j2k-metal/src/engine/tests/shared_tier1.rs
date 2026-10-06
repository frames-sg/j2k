// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    direct_tier1_input_buffer_prepares_for_test, execute_prepared_direct_color_plan_batch,
    prepare_direct_color_plan, prepare_referenced_classic_rgba_plan,
    reset_direct_tier1_input_buffer_prepares_for_test, reset_shared_classic_tier1_passes_for_test,
    reset_stacked_component_batches_for_test, shared_classic_tier1_passes_for_test,
    stacked_component_batches_for_test, submit_prepared_direct_color_plan_batches_into_groups,
    ColorGroupSubmission, DirectDestinationConsumerOrdering, MetalRuntime, PreparedDirectColorPlan,
};
use super::classic::{built_grayscale_plan_tier1_buffers, cpu_decode, encoded_classic};
use super::runtime::should_run_metal_runtime;
use crate::engine::abi::J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_native::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};
use std::sync::Arc;

fn encoded_rgb(seed: u8, (width, height): (u32, u32)) -> Vec<u8> {
    let pixels = (0..width * height * 3)
        .map(|index| {
            u8::try_from((index.wrapping_mul(u32::from(seed) + 7) ^ (index / width)) & 255)
                .expect("masked fixture sample fits u8")
        })
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, 3, 8, false, &options).expect("encode classic RGB")
}

fn prepared_plan(bytes: &[u8]) -> PreparedDirectColorPlan {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("image");
    let plan = image
        .build_direct_color_plan_with_context(&mut DecoderContext::default())
        .expect("direct color plan");
    prepare_direct_color_plan(&plan).expect("prepared color plan")
}

fn cpu_rgb8(bytes: &[u8], (width, height): (u32, u32)) -> Vec<u8> {
    let mut decoder = j2k::J2kDecoder::new(bytes).expect("CPU decoder");
    let row_bytes = width as usize * 3;
    let mut output = vec![0; row_bytes * height as usize];
    decoder
        .decode_into(&mut output, row_bytes, PixelFormat::Rgb8)
        .expect("CPU decode");
    output
}

fn destination(
    runtime: &MetalRuntime,
    (width, height): (u32, u32),
    format: PixelFormat,
    count: usize,
) -> (crate::metal_types::Buffer, MetalImageDestination) {
    let row_bytes = width as usize * format.channels();
    let image_bytes = row_bytes * height as usize;
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        &runtime.device,
        image_bytes * count,
    )
    .expect("destination allocation");
    let layout =
        MetalImageLayout::new_batch(0, (width, height), row_bytes, format, count, image_bytes)
            .expect("destination layout");
    // SAFETY: the fresh allocation has one writer and is read only after the
    // submission completes.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout).expect("destination")
    };
    (buffer, destination)
}

#[test]
fn shared_classic_tier1_reports_failure_against_its_own_group_and_source() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = Arc::new(MetalRuntime::new().expect("Metal runtime"));
    let landscape = (64, 48);
    let portrait = (48, 64);
    let first_bytes = [encoded_rgb(1, landscape), encoded_rgb(2, landscape)];
    let second_bytes = [encoded_rgb(3, portrait), encoded_rgb(4, portrait)];
    let first = first_bytes
        .iter()
        .map(|bytes| Arc::new(prepared_plan(bytes)))
        .collect::<Vec<_>>();
    let mut broken = prepared_plan(&second_bytes[1]);
    // Claim segmentation symbols the coded data does not contain: the block
    // passes host validation and fails strict decoding on the GPU.
    let job = broken.component_plans[0]
        .classic_groups
        .first_mut()
        .and_then(|group| {
            group
                .jobs
                .iter_mut()
                .find(|job| job.coded_len != 0 && job.number_of_coding_passes > 3)
        })
        .expect("fixture has a grouped classic job with several passes");
    job.style_flags |= J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
    job.strict = 1;
    let second = vec![Arc::new(prepared_plan(&second_bytes[0])), Arc::new(broken)];

    let (first_output, first_destination) = destination(&runtime, landscape, PixelFormat::Rgb8, 2);
    let (_, second_destination) = destination(&runtime, portrait, PixelFormat::Rgb8, 2);
    reset_shared_classic_tier1_passes_for_test();
    let mut results = submit_prepared_direct_color_plan_batches_into_groups(
        &runtime,
        vec![
            ColorGroupSubmission {
                plans: &first,
                fmt: PixelFormat::Rgb8,
                layout: BatchLayout::Nhwc,
                destination: &first_destination,
                source_indices: Some(&[0, 2]),
                consumer_ordering: DirectDestinationConsumerOrdering::HostCompletionOnly,
            },
            ColorGroupSubmission {
                plans: &second,
                fmt: PixelFormat::Rgb8,
                layout: BatchLayout::Nhwc,
                destination: &second_destination,
                source_indices: Some(&[1, 3]),
                consumer_ordering: DirectDestinationConsumerOrdering::HostCompletionOnly,
            },
        ],
    )
    .expect("no session-fatal failure");
    assert_eq!(shared_classic_tier1_passes_for_test(), 1);
    let second_result = results.pop().expect("second group result");
    let first_result = results.pop().expect("first group result");

    let error = second_result
        .expect("second group submits")
        .wait()
        .expect_err("second group holds the bad job");
    let message = error.to_string();
    assert!(message.contains("for source 3"), "{message}");
    first_result
        .expect("first group submits")
        .wait()
        .expect("first group is unaffected");
    drop(first_destination);
    // SAFETY: the first group completed and released its destination.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&first_output, 0, 64 * 48 * 3 * 2)
            .expect("first group bytes")
    };
    let expected = first_bytes
        .iter()
        .flat_map(|bytes| cpu_rgb8(bytes, landscape))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

/// One group's plans and their shared output dimensions.
type PlanGroup<'a> = (&'a [Arc<PreparedDirectColorPlan>], (u32, u32));

/// Submits each group into its own destination through the shared dispatch,
/// waits, and returns the decoded bytes of every group.
fn decode_groups(
    runtime: &Arc<MetalRuntime>,
    groups: &[PlanGroup<'_>],
    format: PixelFormat,
) -> Vec<Vec<u8>> {
    let destinations = groups
        .iter()
        .map(|(plans, dimensions)| destination(runtime, *dimensions, format, plans.len()))
        .collect::<Vec<_>>();
    let submissions = groups
        .iter()
        .zip(&destinations)
        .map(|((plans, _), (_, destination))| ColorGroupSubmission {
            plans,
            fmt: format,
            layout: BatchLayout::Nhwc,
            destination,
            source_indices: None,
            consumer_ordering: DirectDestinationConsumerOrdering::HostCompletionOnly,
        })
        .collect();
    for result in submit_prepared_direct_color_plan_batches_into_groups(runtime, submissions)
        .expect("no session-fatal failure")
    {
        result
            .expect("group submits")
            .wait()
            .expect("group decodes");
    }
    groups
        .iter()
        .zip(destinations)
        .map(|((plans, (width, height)), (output, destination))| {
            drop(destination);
            let len = *width as usize * *height as usize * format.channels() * plans.len();
            // SAFETY: the group completed and released its destination.
            unsafe {
                j2k_metal_support::checked_buffer_read_vec::<u8>(&output, 0, len)
                    .expect("group bytes")
            }
        })
        .collect()
}

fn built_tier1_buffers(plans: &[Arc<PreparedDirectColorPlan>]) -> usize {
    plans
        .iter()
        .flat_map(|plan| &plan.component_plans)
        .map(built_grayscale_plan_tier1_buffers)
        .sum()
}

#[test]
fn batch_rgb_routes_leave_per_plan_tier1_buffers_unbuilt() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = Arc::new(MetalRuntime::new().expect("Metal runtime"));
    let landscape = (64, 48);
    let portrait = (48, 64);
    let landscape_bytes = [encoded_rgb(5, landscape), encoded_rgb(6, landscape)];
    let portrait_bytes = [encoded_rgb(7, portrait), encoded_rgb(8, portrait)];
    let landscape_plans = landscape_bytes
        .iter()
        .map(|bytes| Arc::new(prepared_plan(bytes)))
        .collect::<Vec<_>>();
    let portrait_plans = portrait_bytes
        .iter()
        .map(|bytes| Arc::new(prepared_plan(bytes)))
        .collect::<Vec<_>>();
    let expected_landscape = landscape_bytes
        .iter()
        .flat_map(|bytes| cpu_rgb8(bytes, landscape))
        .collect::<Vec<_>>();
    let expected_portrait = portrait_bytes
        .iter()
        .flat_map(|bytes| cpu_rgb8(bytes, portrait))
        .collect::<Vec<_>>();

    reset_direct_tier1_input_buffer_prepares_for_test();
    reset_shared_classic_tier1_passes_for_test();
    let decoded = decode_groups(
        &runtime,
        &[(&landscape_plans, landscape), (&portrait_plans, portrait)],
        PixelFormat::Rgb8,
    );
    assert_eq!(shared_classic_tier1_passes_for_test(), 1);
    assert_eq!(decoded, [expected_landscape.clone(), expected_portrait]);

    reset_stacked_component_batches_for_test();
    let surfaces = execute_prepared_direct_color_plan_batch(&landscape_plans, PixelFormat::Rgb8)
        .expect("stacked batch decode");
    assert!(stacked_component_batches_for_test() > 0);
    let stacked = surfaces
        .iter()
        .flat_map(|surface| surface.as_bytes().expect("surface bytes").into_owned())
        .collect::<Vec<_>>();
    assert_eq!(stacked, expected_landscape);

    assert_eq!(built_tier1_buffers(&landscape_plans), 0);
    assert_eq!(built_tier1_buffers(&portrait_plans), 0);
    assert_eq!(
        direct_tier1_input_buffer_prepares_for_test(),
        0,
        "batch RGB routes upload their own Tier-1 copies"
    );
}

#[test]
fn component_route_builds_tier1_buffers_once_per_plan() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = Arc::new(MetalRuntime::new().expect("Metal runtime"));
    // RGBA cannot stack as RGB, so each component decodes on its own.
    let bytes = j2k_test_support::wrap_jp2_rgba_codestream(&encoded_classic(32, 32, 4), 32, 32, 8);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("RGBA image");
    let referenced = image
        .build_referenced_classic_plan_region_with_context(
            &mut DecoderContext::default(),
            (0, 0, 32, 32),
        )
        .expect("referenced RGBA plan");
    let plans = [Arc::new(
        prepare_referenced_classic_rgba_plan(&referenced, &bytes, false)
            .expect("prepared RGBA plan"),
    )];
    let expected = cpu_decode(&bytes, 32, PixelFormat::Rgba8);

    reset_direct_tier1_input_buffer_prepares_for_test();
    reset_stacked_component_batches_for_test();
    for pass in ["first", "second"] {
        let decoded = decode_groups(&runtime, &[(&plans, (32, 32))], PixelFormat::Rgba8);
        assert_eq!(decoded, std::slice::from_ref(&expected), "{pass} pass");
        let built = built_tier1_buffers(&plans);
        assert!(
            built > 0,
            "the per-component route reads the plan's buffers"
        );
        assert_eq!(
            direct_tier1_input_buffer_prepares_for_test(),
            3 * built,
            "{pass} pass"
        );
    }
    assert_eq!(stacked_component_batches_for_test(), 0);
}
