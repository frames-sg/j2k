// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-image execution for prepared region-scaled direct plans.

use j2k_core::{Downscale, PixelFormat, Rect};
use metal::Device;

use super::planning::{
    build_region_scaled_direct_plan, build_region_scaled_direct_plan_with_session,
    is_direct_region_scaled_runtime_fallback_error, PreparedRegionScaledDirectPlan,
    RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
};
use crate::{Error, Surface};

pub(crate) fn decode_region_scaled_direct_to_surface(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Option<Surface>, Error> {
    let Some(prepared) = build_region_scaled_direct_plan(input, fmt, roi, scale)? else {
        return Ok(None);
    };
    execute_region_scaled_direct_plan(prepared, fmt)
}
pub(crate) fn decode_region_scaled_direct_to_surface_with_session(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    session: &crate::MetalBackendSession,
) -> Result<Option<Surface>, Error> {
    crate::compute::with_runtime_for_session(session, |_| {
        let Some(prepared) =
            build_region_scaled_direct_plan_with_session(input, fmt, roi, scale, session)?
        else {
            return Ok(None);
        };
        execute_region_scaled_direct_plan_with_device(prepared, fmt, session.device_handle())
    })
}

fn execute_region_scaled_direct_plan(
    plan: PreparedRegionScaledDirectPlan,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    match plan {
        PreparedRegionScaledDirectPlan::Gray(plan) => {
            match crate::compute::execute_prepared_direct_grayscale_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => {
            match crate::compute::execute_hybrid_cpu_tier1_direct_color_plan(plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => {
                    Err(Error::UnsupportedMetalRequest {
                        reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
                    })
                }
                Err(error) => Err(error),
            }
        }
    }
}

fn execute_region_scaled_direct_plan_with_device(
    plan: PreparedRegionScaledDirectPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Option<Surface>, Error> {
    match plan {
        PreparedRegionScaledDirectPlan::Gray(plan) => {
            match crate::compute::execute_prepared_direct_grayscale_plan_with_device(
                &plan, fmt, device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => {
            match crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_with_device(
                plan, fmt, device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => {
                    Err(Error::UnsupportedMetalRequest {
                        reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
                    })
                }
                Err(error) => Err(error),
            }
        }
    }
}
