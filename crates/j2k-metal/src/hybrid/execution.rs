// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-image execution for prepared region-scaled direct plans.

use crate::metal_types::Device;
use j2k_core::{Downscale, PixelFormat, Rect};

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
    crate::engine::with_runtime_for_session(session, |_| {
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
            match crate::engine::execute_prepared_direct_grayscale_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => execute_color_plan(
            || crate::engine::execute_prepared_direct_color_plan(plan.clone(), fmt),
            || crate::engine::execute_hybrid_cpu_tier1_direct_color_plan(plan.clone(), fmt),
        ),
    }
}

fn execute_region_scaled_direct_plan_with_device(
    plan: PreparedRegionScaledDirectPlan,
    fmt: PixelFormat,
    device: &Device,
) -> Result<Option<Surface>, Error> {
    match plan {
        PreparedRegionScaledDirectPlan::Gray(plan) => {
            match crate::engine::execute_prepared_direct_grayscale_plan_with_device(
                &plan, fmt, device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PreparedRegionScaledDirectPlan::Color(plan) => execute_color_plan(
            || {
                crate::engine::execute_prepared_direct_color_plan_with_device(
                    plan.clone(),
                    fmt,
                    device,
                )
            },
            || {
                crate::engine::execute_hybrid_cpu_tier1_direct_color_plan_with_device(
                    plan.clone(),
                    fmt,
                    device,
                )
            },
        ),
    }
}

/// Runs a colour plan on the Metal route, as a full-image decode does, and on
/// the CPU Tier-1 route when that one rejects the plan's code blocks, which
/// the CPU decodes whatever their size.
fn execute_color_plan(
    metal: impl FnOnce() -> Result<Surface, Error>,
    cpu_tier1: impl FnOnce() -> Result<Surface, Error>,
) -> Result<Option<Surface>, Error> {
    let surface = match metal() {
        Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => cpu_tier1(),
        result => result,
    };
    match surface {
        Ok(surface) => Ok(Some(surface)),
        Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Err(
            Error::capability_rejected(j2k_core::CapabilityRejection::unsupported_operation(
                RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
            )),
        ),
        Err(error) => Err(error),
    }
}
