// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native direct-plan construction for region-scaled Metal decode.

use std::{sync::Arc, time::Instant};

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};

use super::cache::RegionScaledColorPlanCache;
use super::profile::emit_region_scaled_color_plan_build_timings;
use crate::error::native_decode_error;
use crate::profile_env::elapsed_since_us;
use crate::session::PreparedPlanCacheKey;
use crate::{direct, Error, J2kDecoder};

pub(super) const RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED: &str =
    "J2K Metal ROI+scaled hybrid decode currently supports single-tile RGB direct plans for Rgb8/Rgba8/Rgb16";

pub(super) enum PreparedRegionScaledDirectPlan {
    Gray(crate::compute::PreparedDirectGrayscalePlan),
    Color(Arc<crate::compute::PreparedDirectColorPlan>),
}

pub(super) fn build_region_scaled_direct_plan(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Option<PreparedRegionScaledDirectPlan>, Error> {
    build_region_scaled_direct_plan_with_cache(
        input,
        fmt,
        roi,
        scale,
        RegionScaledColorPlanCache::Uncached,
    )
}

pub(super) fn build_region_scaled_direct_plan_with_session(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    session: &crate::MetalBackendSession,
) -> Result<Option<PreparedRegionScaledDirectPlan>, Error> {
    build_region_scaled_direct_plan_with_cache(
        input,
        fmt,
        roi,
        scale,
        RegionScaledColorPlanCache::Session(session),
    )
}

fn build_region_scaled_direct_plan_with_cache(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    cache: RegionScaledColorPlanCache<'_>,
) -> Result<Option<PreparedRegionScaledDirectPlan>, Error> {
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            match build_region_scaled_direct_gray_plan(input, roi, scale) {
                Ok(plan) => Ok(Some(PreparedRegionScaledDirectPlan::Gray(plan))),
                Err(error) if is_direct_region_scaled_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            }
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => {
            Ok(Some(PreparedRegionScaledDirectPlan::Color(
                build_region_scaled_direct_color_plan_cached_with_cache(
                    input, fmt, roi, scale, cache,
                )?,
            )))
        }
        _ => Ok(None),
    }
}

#[doc(hidden)]
pub(crate) fn benchmark_region_scaled_direct_plan_prepare(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<(), Error> {
    if build_region_scaled_direct_plan(input, fmt, roi, scale)?.is_some() {
        Ok(())
    } else {
        Err(Error::UnsupportedMetalRequest {
            reason: "J2K MetalDirect ROI+scaled plan preparation is unsupported for this benchmark input",
        })
    }
}

pub(super) fn build_region_scaled_direct_gray_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<crate::compute::PreparedDirectGrayscalePlan, Error> {
    let image = build_region_scaled_native_image(input, scale)?;
    let mut context = NativeDecoderContext::default();
    let output_region = roi.scaled_covering(scale);
    let plan = match image.build_direct_grayscale_plan_region_with_context(
        &mut context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) => plan,
        Err(error) if direct::is_unsupported_direct_plan_error(&error) => {
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect region-scaled batch currently supports grayscale direct plans only: {error}"
                ),
                reason: crate::MetalDirectFallbackReason::UnsupportedPlan,
            });
        }
        Err(error) => return Err(native_decode_error(error)),
    };
    let mut prepared = crate::compute::prepare_direct_grayscale_plan(&plan)?;
    crate::compute::crop_prepared_direct_grayscale_plan_to_output_region(
        &mut prepared,
        output_region,
    )?;
    Ok(prepared)
}

fn build_region_scaled_direct_color_plan(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
) -> Result<crate::compute::PreparedDirectColorPlan, Error> {
    #[cfg(test)]
    super::cache::record_region_scaled_color_plan_build_for_test();

    let profile_stages = crate::compute::metal_profile_stages_enabled();
    let total_started = profile_stages.then(Instant::now);
    let native_image_started = profile_stages.then(Instant::now);
    let image = build_region_scaled_native_image(input, scale)?;
    let native_image_us = native_image_started
        .map(elapsed_since_us)
        .unwrap_or_default();
    let direct_plan_started = profile_stages.then(Instant::now);
    let mut context = NativeDecoderContext::default();
    let output_region = roi.scaled_covering(scale);
    let plan = match image.build_direct_color_plan_region_with_context(
        &mut context,
        (
            output_region.x,
            output_region.y,
            output_region.w,
            output_region.h,
        ),
    ) {
        Ok(plan) => plan,
        Err(error) if direct::is_unsupported_direct_plan_error(&error) => {
            return Err(Error::UnsupportedMetalRequest {
                reason: RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED,
            });
        }
        Err(error) => return Err(native_decode_error(error)),
    };
    let direct_plan_us = direct_plan_started
        .map(elapsed_since_us)
        .unwrap_or_default();
    let prepare_started = profile_stages.then(Instant::now);
    let mut prepared = crate::compute::prepare_direct_color_plan_for_cpu_upload(&plan)?;
    let prepare_us = prepare_started.map(elapsed_since_us).unwrap_or_default();
    let crop_started = profile_stages.then(Instant::now);
    crate::compute::crop_prepared_direct_color_plan_to_output_region(&mut prepared, output_region)?;
    let crop_us = crop_started.map(elapsed_since_us).unwrap_or_default();
    if let Some(started) = total_started {
        emit_region_scaled_color_plan_build_timings(
            native_image_us,
            direct_plan_us,
            prepare_us,
            crop_us,
            elapsed_since_us(started),
        );
    }
    Ok(prepared)
}

pub(super) fn build_region_scaled_direct_color_plan_cached_with_cache(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    cache: RegionScaledColorPlanCache<'_>,
) -> Result<Arc<crate::compute::PreparedDirectColorPlan>, Error> {
    let cache_key = PreparedPlanCacheKey::region_scaled_color(input, fmt, roi, scale);
    if let Some(plan) = cache.get(cache_key)? {
        return Ok(plan);
    }

    let plan = Arc::new(build_region_scaled_direct_color_plan(input, roi, scale)?);
    cache.store(cache_key, plan.clone())?;
    Ok(plan)
}

fn build_region_scaled_native_image(
    input: &[u8],
    scale: Downscale,
) -> Result<NativeImage<'_>, Error> {
    let decoder = J2kDecoder::new(input)?;
    let dims = decoder.inner.info().dimensions;
    let target_dims = (
        dims.0.div_ceil(scale.denominator()),
        dims.1.div_ceil(scale.denominator()),
    );
    let settings = NativeDecodeSettings {
        target_resolution: Some(target_dims),
        ..NativeDecodeSettings::default()
    };
    let image = NativeImage::new(input, &settings).map_err(native_decode_error)?;
    Ok(image)
}

pub(super) fn is_direct_region_scaled_runtime_fallback_error(error: &Error) -> bool {
    crate::decoder::is_direct_runtime_fallback_error(error)
}
