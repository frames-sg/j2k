// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::destination_index_validation::validate_stacked_color_destination_indices;
use super::{
    dispatch_3d_pipeline, size_of, BatchLayout, Buffer, Error, J2kNativeColorBatchStoreParams,
    J2kWaveletTransform, MetalImageDestination, MetalRuntime, PixelFormat, PreparedDirectColorPlan,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_arguments,
    reason = "the stacked exact store keeps destination layout and native codec metadata explicit"
)]
#[expect(
    clippy::too_many_lines,
    reason = "validation, ABI construction, resource binding, and dispatch form one ordered Metal encoder transaction"
)]
pub(super) fn encode_exact_native_color_batch_store_in_encoder(
    runtime: &MetalRuntime,
    encoder: &metal::ComputeCommandEncoderRef,
    planes: &[Buffer],
    plan: &PreparedDirectColorPlan,
    fmt: PixelFormat,
    layout: BatchLayout,
    count: usize,
    broadcast_planes: bool,
    destination_image_index: usize,
    destination: &MetalImageDestination,
) -> Result<(), Error> {
    let channels = fmt.channels();
    if planes.len() != channels {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal stacked exact color store",
            reason: "stacked plane count does not match output channels",
        });
    }
    let destination_layout = destination.layout();
    let bytes_per_sample = fmt.bytes_per_sample();
    let plane_stride = usize::try_from(plan.dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(plan.dimensions.1 as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal stacked exact color plane size overflow".to_string(),
        })?;
    validate_stacked_color_destination_indices(
        plan.dimensions,
        channels,
        layout,
        count,
        broadcast_planes,
    )?;
    let destination_end =
        destination_image_index
            .checked_add(count)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal exact color destination image range overflow".to_string(),
            })?;
    if destination_end > destination_layout.image_count() {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal exact color store",
            reason: "destination image range exceeds the validated output group",
        });
    }
    let destination_offset = destination_layout
        .image_offset_bytes(destination_image_index)
        .and_then(|offset| destination_layout.byte_offset().checked_add(offset))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal exact color destination offset overflow".to_string(),
        })?;
    let params = J2kNativeColorBatchStoreParams {
        width: plan.dimensions.0,
        height: plan.dimensions.1,
        plane_stride: if broadcast_planes {
            0
        } else {
            u32::try_from(plane_stride).map_err(|_| Error::MetalKernel {
                message: "J2K Metal stacked exact color plane stride exceeds u32".to_string(),
            })?
        },
        output_row_stride: u32::try_from(destination_layout.pitch_bytes() / bytes_per_sample)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal stacked exact color row stride exceeds u32".to_string(),
            })?,
        output_item_stride: u32::try_from(
            destination_layout.image_stride_bytes() / bytes_per_sample,
        )
        .map_err(|_| Error::MetalKernel {
            message: "J2K Metal stacked exact color image stride exceeds u32".to_string(),
        })?,
        batch_count: u32::try_from(count).map_err(|_| Error::MetalKernel {
            message: "J2K Metal stacked exact color batch count exceeds u32".to_string(),
        })?,
        layout: match layout {
            BatchLayout::Nchw => 0,
            BatchLayout::Nhwc => 1,
            _ => {
                return Err(Error::UnsupportedMetalRequest {
                    reason: "J2K Metal exact color destination received an unknown batch layout",
                })
            }
        },
        mct: u32::from(plan.mct),
        transform: match plan.transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        },
        signed: u32::from(plan.signed),
        bit_depths: [
            u32::from(plan.bit_depths[0]),
            u32::from(plan.bit_depths[1]),
            u32::from(plan.bit_depths[2]),
            plan.alpha_bit_depth.map_or(0, u32::from),
        ],
    };
    let pipeline = match fmt {
        PixelFormat::Rgb8 => &runtime.store_native_rgb_batch_u8,
        PixelFormat::Rgb16 => &runtime.store_native_rgb_batch_u16,
        PixelFormat::RgbI16 => &runtime.store_native_rgb_batch_i16,
        PixelFormat::Rgba8 => &runtime.store_native_rgba_batch_u8,
        PixelFormat::Rgba16 => &runtime.store_native_rgba_batch_u16,
        PixelFormat::RgbaI16 => &runtime.store_native_rgba_batch_i16,
        _ => return Err(Error::UnsupportedMetalRequest {
            reason:
                "J2K Metal stacked exact color destination supports native RGB/RGBA integers only",
        }),
    };
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
