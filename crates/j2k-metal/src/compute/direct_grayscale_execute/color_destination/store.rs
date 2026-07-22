// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_3d_pipeline, size_of, BatchLayout, Buffer, Error, MetalImageDestination, MetalRuntime,
    PixelFormat, PreparedDirectColorPlan,
};
use crate::compute::abi::J2kNativeColorBatchStoreParams;

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
    encoder: &metal::ComputeCommandEncoderRef,
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
    encoder.set_compute_pipeline_state(pipeline);
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
    encoder.set_bytes(
        channels as u64 + 1,
        size_of::<J2kNativeColorBatchStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        pipeline,
        (params.width, params.height, params.batch_count),
    );
    Ok(())
}
