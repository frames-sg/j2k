// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::{
    dispatch_3d_pipeline, BatchLayout, Buffer, Error, MetalImageDestination, MetalRuntime,
    PixelFormat, PreparedDirectColorPlan,
};
use crate::engine::abi::J2kNativeColorBatchStoreParams;

mod plan;

use self::plan::{plan_exact_native_color_store, NativeColorStorePlan};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct NativeColorStoreConfig {
    pub(super) format: PixelFormat,
    pub(super) layout: BatchLayout,
    pub(super) image_count: usize,
    pub(super) broadcast_planes: bool,
    pub(super) destination_image_index: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_exact_native_color_batch_store_in_encoder(
    runtime: &MetalRuntime,
    encoder: &crate::metal_types::ComputeCommandEncoderRef,
    planes: &[Buffer],
    plan: &PreparedDirectColorPlan,
    config: NativeColorStoreConfig,
    destination: &MetalImageDestination,
) -> Result<(), Error> {
    let NativeColorStorePlan {
        channels,
        destination_offset,
        params,
        pipeline,
    } = plan_exact_native_color_store(runtime, planes, plan, config, destination)?;
    match planes {
        [r, g, b] => encoder.memory_barrier_with_resources(&[r, g, b]),
        [r, g, b, a] => encoder.memory_barrier_with_resources(&[r, g, b, a]),
        _ => unreachable!("plane count was validated against the native color format"),
    }
    encoder.setComputePipelineState(pipeline);
    for (index, plane) in planes.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(plane), 0);
    }
    // SAFETY: the checked destination owns this exact dense group range until
    // the submitted command buffer has completed.
    encoder.set_buffer(
        channels as u64,
        Some(unsafe { destination.raw_buffer() }),
        u64::try_from(destination_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal stacked exact color destination offset exceeds u64".to_string(),
        })?,
    );
    encoder.set_bytes::<J2kNativeColorBatchStoreParams>(channels as u64 + 1, &params);
    dispatch_3d_pipeline(
        encoder,
        pipeline,
        (params.width, params.height, params.batch_count),
    );
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::engine::{
        checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, new_command_buffer,
        new_compute_command_encoder, new_shared_buffer, with_runtime,
    };

    #[test]
    fn irreversible_native_color_store_rounds_centered_half_ties_to_even_before_shift() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        with_runtime(|runtime| {
            let plane0 = copied_slice_buffer(&runtime.device, &[0.5_f32, -1.5])?;
            let plane1 = copied_slice_buffer(&runtime.device, &[0.0_f32, 0.0])?;
            let plane2 = copied_slice_buffer(&runtime.device, &[0.0_f32, 0.0])?;
            let rgb_planes = [&plane0, &plane1, &plane2];

            let rgb8 = new_shared_buffer(&runtime.device, 6)?;
            dispatch_test_store(
                runtime,
                &runtime.store_native_rgb_batch_u8,
                &rgb_planes,
                &rgb8,
                J2kNativeColorBatchStoreParams {
                    width: 2,
                    height: 1,
                    plane_stride: 2,
                    output_row_stride: 6,
                    output_item_stride: 6,
                    batch_count: 1,
                    layout: 1,
                    mct: 1,
                    transform: 1,
                    signed: 0,
                    bit_depths: [8, 8, 8, 0],
                },
            )?;
            assert_eq!(
                checked_buffer_slice::<u8>(&rgb8, 6, "irreversible RGB8 tie output")?,
                [128, 128, 128, 126, 126, 126]
            );

            let alpha = copied_slice_buffer(&runtime.device, &[7.0_f32, 9.0])?;
            let irreversible_rgba_planes = [&plane0, &plane1, &plane2, &alpha];
            let rgba_sample_count = 2_usize * 4;
            let rgba_output_bytes = rgba_sample_count
                .checked_mul(size_of::<u16>())
                .expect("RGBA16 output fixture byte length");
            let rgba16 = new_shared_buffer(&runtime.device, rgba_output_bytes)?;
            dispatch_test_store(
                runtime,
                &runtime.store_native_rgba_batch_u16,
                &irreversible_rgba_planes,
                &rgba16,
                J2kNativeColorBatchStoreParams {
                    width: 2,
                    height: 1,
                    plane_stride: 2,
                    output_row_stride: 8,
                    output_item_stride: 8,
                    batch_count: 1,
                    layout: 1,
                    mct: 1,
                    transform: 1,
                    signed: 0,
                    bit_depths: [16, 16, 16, 16],
                },
            )?;
            assert_eq!(
                checked_buffer_slice::<u16>(&rgba16, 8, "irreversible RGBA16 tie output")?,
                [32768, 32768, 32768, 7, 32766, 32766, 32766, 9]
            );
            Ok(())
        })
        .expect("irreversible native color tie stores");
    }

    fn dispatch_test_store(
        runtime: &MetalRuntime,
        pipeline: &crate::metal_types::ComputePipelineState,
        planes: &[&Buffer],
        output: &Buffer,
        params: J2kNativeColorBatchStoreParams,
    ) -> Result<(), Error> {
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.setComputePipelineState(pipeline);
        for (index, plane) in planes.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(plane), 0);
        }
        encoder.set_buffer(planes.len() as u64, Some(output), 0);
        encoder.set_bytes::<J2kNativeColorBatchStoreParams>(planes.len() as u64 + 1, &params);
        dispatch_3d_pipeline(
            &encoder,
            pipeline,
            (params.width, params.height, params.batch_count),
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)
    }
}
