// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    prepared_direct_color_plan_supports_runtime, Arc, BatchLayout, Error, MetalImageDestination,
    MetalRuntime, PixelFormat, PreparedDirectColorPlan,
};

pub(super) fn validate_color_group(
    runtime: &MetalRuntime,
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    layout: BatchLayout,
    destination: &MetalImageDestination,
) -> Result<(), Error> {
    let first = &plans[0];
    if !matches!(
        fmt,
        PixelFormat::Rgb8
            | PixelFormat::Rgb16
            | PixelFormat::RgbI16
            | PixelFormat::Rgba8
            | PixelFormat::Rgba16
            | PixelFormat::RgbaI16
    ) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact color destination supports native RGB/RGBA integers only",
        });
    }
    if !matches!(layout, BatchLayout::Nchw | BatchLayout::Nhwc) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact RGB destination received an unknown batch layout",
        });
    }
    let bit_depths = first
        .bit_depths
        .iter()
        .copied()
        .chain(first.alpha_bit_depth);
    let bit_depths_supported = match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
            !first.signed && bit_depths.clone().all(|depth| (1..=8).contains(&depth))
        }
        PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            !first.signed && bit_depths.clone().all(|depth| (9..=16).contains(&depth))
        }
        PixelFormat::RgbI16 | PixelFormat::RgbaI16 => {
            first.signed && bit_depths.clone().all(|depth| (1..=16).contains(&depth))
        }
        _ => false,
    };
    if !bit_depths_supported {
        return Err(Error::UnsupportedMetalRequest {
            reason:
                "J2K Metal exact RGB destination sample width does not match component precision",
        });
    }
    let expected_components = fmt.channels();
    let expects_alpha = expected_components == 4;
    if plans.iter().any(|plan| {
        plan.dimensions != first.dimensions
            || plan.bit_depths != first.bit_depths
            || plan.alpha_bit_depth != first.alpha_bit_depth
            || plan.signed != first.signed
            || plan.component_plans.len() != expected_components
            || plan.alpha_bit_depth.is_some() != expects_alpha
            || plan
                .component_plans
                .iter()
                .any(|component| component.dimensions != first.dimensions)
    }) {
        return Err(Error::UnsupportedMetalRequest {
            reason:
                "J2K Metal exact color destination requires homogeneous matching component plans",
        });
    }
    if plans
        .iter()
        .any(|plan| !prepared_direct_color_plan_supports_runtime(plan, fmt))
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact color destination contains a code-block plan unsupported by the Metal runtime",
        });
    }
    destination
        .validate_device(&runtime.device)
        .and_then(|()| destination.validate_batch(first.dimensions, fmt, plans.len()))
        .map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K Metal exact RGB group destination validation failed",
                source,
            )
        })?;

    let destination_layout = destination.layout();
    let tight_row_bytes = usize::try_from(first.dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(fmt.bytes_per_pixel()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal exact RGB row byte count overflow".to_string(),
        })?;
    let tight_image_bytes = tight_row_bytes
        .checked_mul(first.dimensions.1 as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal exact RGB image byte count overflow".to_string(),
        })?;
    if destination_layout.pitch_bytes() != tight_row_bytes
        || destination_layout.image_stride_bytes() != tight_image_bytes
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact RGB destination must be one dense contiguous group",
        });
    }
    Ok(())
}
