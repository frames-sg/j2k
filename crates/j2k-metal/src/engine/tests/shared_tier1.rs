// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    prepare_direct_color_plan, reset_shared_classic_tier1_passes_for_test,
    shared_classic_tier1_passes_for_test, submit_prepared_direct_color_plan_batches_into_groups,
    ColorGroupSubmission, DirectDestinationConsumerOrdering, MetalRuntime, PreparedDirectColorPlan,
};
use super::runtime::should_run_metal_runtime;
use crate::engine::abi::J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS;
use j2k::BatchLayout;
use j2k_core::PixelFormat;
use j2k_metal_support::{MetalImageDestination, MetalImageLayout};
use j2k_native::{encode, DecodeSettings, DecoderContext, EncodeOptions, Image};
use std::sync::Arc;

fn encoded_rgb(seed: u8, (width, height): (u32, u32)) -> Vec<u8> {
    let pixels = (0..width * height * 3)
        .map(|index| (index.wrapping_mul(u32::from(seed) + 7) ^ (index / width)) as u8)
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
    count: usize,
) -> (crate::metal_types::Buffer, MetalImageDestination) {
    let row_bytes = width as usize * 3;
    let image_bytes = row_bytes * height as usize;
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        &runtime.device,
        image_bytes * count,
    )
    .expect("destination allocation");
    let layout = MetalImageLayout::new_batch(
        0,
        (width, height),
        row_bytes,
        PixelFormat::Rgb8,
        count,
        image_bytes,
    )
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

    let (first_output, first_destination) = destination(&runtime, landscape, 2);
    let (_, second_destination) = destination(&runtime, portrait, 2);
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
