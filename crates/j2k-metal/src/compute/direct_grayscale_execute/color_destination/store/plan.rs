// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::destination_index_validation::validate_stacked_color_destination_indices;
use super::{
    BatchLayout, Buffer, Error, MetalImageDestination, MetalRuntime, NativeColorStoreConfig,
    PixelFormat, PreparedDirectColorPlan,
};
use crate::compute::abi::J2kNativeColorBatchStoreParams;
use j2k_native::J2kWaveletTransform;

pub(super) struct NativeColorStorePlan<'a> {
    pub(super) channels: usize,
    pub(super) destination_offset: usize,
    pub(super) params: J2kNativeColorBatchStoreParams,
    pub(super) pipeline: &'a metal::ComputePipelineState,
}

pub(super) fn plan_exact_native_color_store<'a>(
    runtime: &'a MetalRuntime,
    planes: &[Buffer],
    plan: &PreparedDirectColorPlan,
    config: NativeColorStoreConfig,
    destination: &MetalImageDestination,
) -> Result<NativeColorStorePlan<'a>, Error> {
    let NativeColorStoreConfig {
        format,
        layout,
        image_count,
        broadcast_planes,
        destination_image_index,
    } = config;
    let fmt = format;
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
        image_count,
        broadcast_planes,
    )?;
    let destination_end = destination_image_index
        .checked_add(image_count)
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
        batch_count: u32::try_from(image_count).map_err(|_| Error::MetalKernel {
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
    let pipeline = native_color_store_pipeline(runtime, fmt)?;
    Ok(NativeColorStorePlan {
        channels,
        destination_offset,
        params,
        pipeline,
    })
}

fn native_color_store_pipeline(
    runtime: &MetalRuntime,
    format: PixelFormat,
) -> Result<&metal::ComputePipelineState, Error> {
    match format {
        PixelFormat::Rgb8 => Ok(&runtime.store_native_rgb_batch_u8),
        PixelFormat::Rgb16 => Ok(&runtime.store_native_rgb_batch_u16),
        PixelFormat::RgbI16 => Ok(&runtime.store_native_rgb_batch_i16),
        PixelFormat::Rgba8 => Ok(&runtime.store_native_rgba_batch_u8),
        PixelFormat::Rgba16 => Ok(&runtime.store_native_rgba_batch_u16),
        PixelFormat::RgbaI16 => Ok(&runtime.store_native_rgba_batch_i16),
        _ => Err(Error::UnsupportedMetalRequest {
            reason:
                "J2K Metal stacked exact color destination supports native RGB/RGBA integers only",
        }),
    }
}
