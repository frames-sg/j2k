// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch routing for region-scaled Metal direct decode.

use std::sync::Arc;

use j2k_core::{Downscale, PixelFormat, Rect};

use super::cache::RegionScaledColorPlanCache;
use super::plan_resolution::try_resolve_plans_in_order;
use super::planning::{
    build_region_scaled_direct_color_plan_cached_with_cache, build_region_scaled_direct_gray_plan,
};
use crate::{Error, Surface};

pub(crate) fn decode_region_scaled_grayscale_batch_direct_to_device_routed(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
    session: Option<&crate::MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K MetalDirect region-scaled grayscale batch does not support {fmt:?}"
            ),
        });
    }

    let decode = || {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal region-scaled grayscale batch plan",
        );
        let mut plans =
            budget.try_vec(requests.len(), "J2K Metal region-scaled grayscale plans")?;
        for (input, roi, scale) in requests {
            let plan = build_region_scaled_direct_gray_plan(input.as_ref(), *roi, *scale)?;
            plans.push(Arc::new(plan));
        }
        crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
    };
    match session {
        Some(session) => crate::compute::with_runtime_for_session(session, |_| decode()),
        None => decode(),
    }
}

#[cfg(test)]
pub(crate) fn decode_region_scaled_color_batch_direct_to_device(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    decode_region_scaled_color_batch_direct_to_device_routed(requests, fmt, None)
}

pub(crate) fn decode_region_scaled_color_batch_direct_to_device_routed(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
    session: Option<&crate::MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled color batch does not support {fmt:?}"),
        });
    }

    let decode = || {
        let cache = match session {
            Some(session) => RegionScaledColorPlanCache::Session(session),
            None => RegionScaledColorPlanCache::Global,
        };
        let plans = if let Some((input, roi, scale)) = repeated_region_scaled_request(requests) {
            let plan = build_region_scaled_direct_color_plan_cached_with_cache(
                input.as_ref(),
                fmt,
                roi,
                scale,
                cache,
            )?;
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal repeated region-scaled color batch plan",
            );
            budget.try_filled(
                requests.len(),
                plan,
                "J2K Metal repeated region-scaled color plans",
            )?
        } else {
            try_resolve_plans_in_order(requests, |(input, roi, scale)| {
                build_region_scaled_direct_color_plan_cached_with_cache(
                    input.as_ref(),
                    fmt,
                    *roi,
                    *scale,
                    cache,
                )
            })?
        };
        crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)
    };
    match session {
        Some(session) => crate::compute::with_runtime_for_session(session, |_| decode()),
        None => decode(),
    }
}

#[cfg(test)]
pub(crate) fn decode_repeated_region_scaled_color_batch_direct_to_device(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    decode_repeated_region_scaled_color_batch_direct_to_device_routed(
        input, roi, scale, fmt, count, None,
    )
}

pub(crate) fn decode_repeated_region_scaled_color_batch_direct_to_device_routed(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
    fmt: PixelFormat,
    count: usize,
    session: Option<&crate::MetalBackendSession>,
) -> Result<Vec<Surface>, Error> {
    if count == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect repeated region-scaled color batch requires count > 0"
                .to_string(),
        });
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect region-scaled color batch does not support {fmt:?}"),
        });
    }

    let decode = || {
        let cache = match session {
            Some(session) => RegionScaledColorPlanCache::Session(session),
            None => RegionScaledColorPlanCache::Global,
        };
        let plan =
            build_region_scaled_direct_color_plan_cached_with_cache(input, fmt, roi, scale, cache)?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal repeated region-scaled color batch plan",
        );
        let plans =
            budget.try_filled(count, plan, "J2K Metal repeated region-scaled color plans")?;
        crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)
    };
    match session {
        Some(session) => crate::compute::with_runtime_for_session(session, |_| decode()),
        None => decode(),
    }
}

fn repeated_region_scaled_request(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
) -> Option<(&Arc<[u8]>, Rect, Downscale)> {
    let (first_input, first_roi, first_scale) = requests.first()?;
    requests
        .iter()
        .all(|(input, roi, scale)| {
            *roi == *first_roi
                && *scale == *first_scale
                && (Arc::ptr_eq(input, first_input) || input.as_ref() == first_input.as_ref())
        })
        .then_some((first_input, *first_roi, *first_scale))
}
