// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage,
};
use metal::Device;

use crate::error::native_decode_error;
use crate::profile_env::{
    decode_profile_label, elapsed_since_us, MetalDirectProfileRow, MetalProfileFormat,
};
use crate::session::{prepared_plan_cache_error, PreparedPlanCache, PreparedPlanCacheKey};
use crate::{direct, Error, J2kDecoder, Surface};

pub(crate) const RGB_REGION_SCALED_METAL_DIRECT_UNSUPPORTED: &str =
    "J2K Metal ROI+scaled hybrid decode currently supports single-tile RGB direct plans for Rgb8/Rgba8/Rgb16";
pub(crate) const REGION_SCALED_COLOR_PLAN_CACHE_CAP: usize = 128;

mod plan_resolution;

use self::plan_resolution::try_resolve_plans_in_order;

static REGION_SCALED_COLOR_PLAN_CACHE: OnceLock<
    Mutex<PreparedPlanCache<Arc<crate::compute::PreparedDirectColorPlan>>>,
> = OnceLock::new();

#[cfg(test)]
macro_rules! test_atomic_counter {
    ($counter:ident, $reset:ident, $load:ident) => {
        static $counter: AtomicUsize = AtomicUsize::new(0);

        pub(crate) fn $reset() {
            $counter.store(0, Ordering::Relaxed);
        }

        pub(crate) fn $load() -> usize {
            $counter.load(Ordering::Relaxed)
        }
    };
}

#[cfg(test)]
test_atomic_counter!(
    REGION_SCALED_COLOR_PLAN_BUILDS,
    reset_region_scaled_color_plan_builds_for_test,
    region_scaled_color_plan_builds_for_test
);
#[cfg(test)]
static REGION_SCALED_COLOR_PLAN_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn region_scaled_color_plan_test_lock_for_test() -> std::sync::MutexGuard<'static, ()> {
    REGION_SCALED_COLOR_PLAN_TEST_LOCK
        .lock()
        .expect("region scaled color plan test lock")
}

#[cfg(test)]
pub(crate) fn reset_region_scaled_color_plan_cache_for_test() {
    if let Some(cache) = REGION_SCALED_COLOR_PLAN_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.clear();
        }
    }
}

enum PreparedRegionScaledDirectPlan {
    Gray(crate::compute::PreparedDirectGrayscalePlan),
    Color(Arc<crate::compute::PreparedDirectColorPlan>),
}

#[derive(Clone, Copy)]
enum RegionScaledColorPlanCache<'a> {
    Uncached,
    Global,
    Session(&'a crate::MetalBackendSession),
}

impl RegionScaledColorPlanCache<'_> {
    fn get(
        self,
        key: PreparedPlanCacheKey<'_>,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
        match self {
            Self::Uncached => Ok(None),
            Self::Global => cached_region_scaled_direct_color_plan(key),
            Self::Session(session) => {
                cached_region_scaled_direct_color_plan_with_session(session, key)
            }
        }
    }

    fn store(
        self,
        key: PreparedPlanCacheKey<'_>,
        plan: Arc<crate::compute::PreparedDirectColorPlan>,
    ) -> Result<(), Error> {
        match self {
            Self::Uncached => Ok(()),
            Self::Global => {
                plan.disable_dynamic_cpu_tier1_retention()?;
                store_region_scaled_direct_color_plan(key, plan)
            }
            Self::Session(session) => {
                plan.disable_dynamic_cpu_tier1_retention()?;
                store_region_scaled_direct_color_plan_with_session(session, key, plan)
            }
        }
    }
}

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
    let Some(prepared) =
        build_region_scaled_direct_plan_with_session(input, fmt, roi, scale, session)?
    else {
        return Ok(None);
    };
    execute_region_scaled_direct_plan_with_device(prepared, fmt, session.device_handle())
}

fn build_region_scaled_direct_plan(
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

fn build_region_scaled_direct_plan_with_session(
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

pub(crate) fn decode_region_scaled_grayscale_batch_direct_to_device(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
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

    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal region-scaled grayscale batch plan",
    );
    let mut plans = budget.try_vec(requests.len(), "J2K Metal region-scaled grayscale plans")?;
    for (input, roi, scale) in requests {
        let plan = build_region_scaled_direct_gray_plan(input.as_ref(), *roi, *scale)?;
        plans.push(Arc::new(plan));
    }
    crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
}

pub(crate) fn decode_region_scaled_color_batch_direct_to_device(
    requests: &[(Arc<[u8]>, Rect, Downscale)],
    fmt: PixelFormat,
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

    if let Some((input, roi, scale)) = repeated_region_scaled_request(requests) {
        let plan = build_region_scaled_direct_color_plan_cached(input.as_ref(), fmt, roi, scale)?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal repeated region-scaled color batch plan",
        );
        let plans = budget.try_filled(
            requests.len(),
            plan,
            "J2K Metal repeated region-scaled color plans",
        )?;
        return crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt);
    }

    let plans = try_resolve_plans_in_order(requests, |(input, roi, scale)| {
        build_region_scaled_direct_color_plan_cached(input.as_ref(), fmt, *roi, *scale)
    })?;
    crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)
}

pub(crate) fn decode_repeated_region_scaled_color_batch_direct_to_device(
    input: &[u8],
    roi: Rect,
    scale: Downscale,
    fmt: PixelFormat,
    count: usize,
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

    let plan = build_region_scaled_direct_color_plan_cached(input, fmt, roi, scale)?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal repeated region-scaled color batch plan",
    );
    let plans = budget.try_filled(count, plan, "J2K Metal repeated region-scaled color plans")?;
    crate::compute::execute_hybrid_cpu_tier1_direct_color_plan_batch(&plans, fmt)
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

fn build_region_scaled_direct_gray_plan(
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
    REGION_SCALED_COLOR_PLAN_BUILDS.fetch_add(1, Ordering::Relaxed);

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

fn build_region_scaled_direct_color_plan_cached(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Arc<crate::compute::PreparedDirectColorPlan>, Error> {
    build_region_scaled_direct_color_plan_cached_with_cache(
        input,
        fmt,
        roi,
        scale,
        RegionScaledColorPlanCache::Global,
    )
}

fn build_region_scaled_direct_color_plan_cached_with_cache(
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

fn cached_region_scaled_direct_color_plan_with_session(
    session: &crate::MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
    let mut guard =
        session
            .region_scaled_color_plan_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "session region-scaled color prepared-plan cache",
            })?;
    Ok(guard.get(key).cloned())
}

fn store_region_scaled_direct_color_plan_with_session(
    session: &crate::MetalBackendSession,
    key: PreparedPlanCacheKey<'_>,
    plan: Arc<crate::compute::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    let mut guard =
        session
            .region_scaled_color_plan_cache
            .lock()
            .map_err(|_| Error::MetalStatePoisoned {
                state: "session region-scaled color prepared-plan cache",
            })?;
    evict_one_region_scaled_color_plan_if_needed(&mut guard, key, plan)
}

fn cached_region_scaled_direct_color_plan(
    key: PreparedPlanCacheKey<'_>,
) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
    let cache = REGION_SCALED_COLOR_PLAN_CACHE
        .get_or_init(|| Mutex::new(PreparedPlanCache::new(REGION_SCALED_COLOR_PLAN_CACHE_CAP)));
    let mut guard = cache.lock().map_err(|_| Error::MetalStatePoisoned {
        state: "global region-scaled color prepared-plan cache",
    })?;
    Ok(guard.get(key).cloned())
}

fn store_region_scaled_direct_color_plan(
    key: PreparedPlanCacheKey<'_>,
    plan: Arc<crate::compute::PreparedDirectColorPlan>,
) -> Result<(), Error> {
    let cache = REGION_SCALED_COLOR_PLAN_CACHE
        .get_or_init(|| Mutex::new(PreparedPlanCache::new(REGION_SCALED_COLOR_PLAN_CACHE_CAP)));
    let mut guard = cache.lock().map_err(|_| Error::MetalStatePoisoned {
        state: "global region-scaled color prepared-plan cache",
    })?;
    evict_one_region_scaled_color_plan_if_needed(&mut guard, key, plan)
}

fn evict_one_region_scaled_color_plan_if_needed<T: crate::session::PreparedPlanCacheValue>(
    cache: &mut PreparedPlanCache<T>,
    key: PreparedPlanCacheKey<'_>,
    value: T,
) -> Result<(), Error> {
    cache.insert(key, value).map(|_| ()).map_err(|error| {
        prepared_plan_cache_error(
            "Metal region-scaled prepared-plan cache update failed",
            error,
        )
    })
}

fn emit_region_scaled_color_plan_build_timings(
    native_image_us: u128,
    direct_plan_us: u128,
    prepare_us: u128,
    crop_us: u128,
    total_us: u128,
) {
    if !crate::compute::metal_profile_stages_enabled() {
        return;
    }

    let label = match decode_profile_label() {
        Ok(label) => label,
        Err(error) => {
            j2k_profile::emit_profile_error("metal_hybrid_plan_label", &error);
            return;
        }
    };
    for (stage, elapsed_us) in [
        ("native_image", native_image_us),
        ("direct_color_plan", direct_plan_us),
        ("prepare_cpu_upload", prepare_us),
        ("crop_prepared_plan", crop_us),
        ("plan_total", total_us),
    ] {
        let processor = plan_stage_processor(stage);
        let metric = plan_stage_metric(stage);
        let metric_kind = plan_stage_metric_kind(stage);
        let aggregation = plan_stage_aggregation(stage);
        crate::profile_env::emit_metal_profile_row(
            "j2k",
            "decode",
            "metal_cpu_hybrid_plan",
            &MetalDirectProfileRow {
                pipeline: "decode_hybrid",
                label: &label,
                stage,
                processor,
                metric,
                metric_kind,
                aggregation,
                fmt: MetalProfileFormat::Family("Rgb"),
                batch_count: 1,
                elapsed_us,
            },
        );
    }
}

fn plan_stage_processor(stage: &str) -> &'static str {
    match stage {
        "native_image" | "direct_color_plan" | "prepare_cpu_upload" | "crop_prepared_plan" => "cpu",
        _ => "hybrid",
    }
}

fn plan_stage_metric(stage: &str) -> &'static str {
    match stage {
        "native_image" => "native_image_us",
        "direct_color_plan" => "direct_color_plan_us",
        "prepare_cpu_upload" => "prepare_cpu_upload_us",
        "crop_prepared_plan" => "crop_prepared_plan_us",
        "plan_total" => "plan_total_us",
        _ => "wall_us",
    }
}

fn plan_stage_metric_kind(_stage: &str) -> &'static str {
    "wall_elapsed"
}

fn plan_stage_aggregation(stage: &str) -> &'static str {
    match stage {
        "plan_total" => "inclusive",
        _ => "exclusive",
    }
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

fn is_direct_region_scaled_runtime_fallback_error(error: &Error) -> bool {
    crate::decoder::is_direct_runtime_fallback_error(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    fn should_run_metal_runtime() -> bool {
        j2k_test_support::metal_runtime_gate(module_path!())
    }

    fn encoded_rgb8_tile_for_region_scaled_plan_cache(seed: u8) -> Arc<[u8]> {
        let mut pixels = j2k_test_support::gradient_u8(64, 64, 3);
        for pixel in pixels.chunks_exact_mut(3) {
            pixel[0] = pixel[0].wrapping_add(seed);
            pixel[1] = pixel[1].wrapping_add(seed.wrapping_mul(3));
            pixel[2] = pixel[2].wrapping_add(seed.wrapping_mul(5));
        }
        let options = j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        };
        Arc::<[u8]>::from(
            j2k_native::encode(&pixels, 64, 64, 3, 8, false, &options).expect("encode rgb8"),
        )
    }

    fn region_scaled_plan_cache_roi() -> Rect {
        Rect {
            x: 8,
            y: 8,
            w: 32,
            h: 32,
        }
    }

    #[test]
    fn known_repeated_region_scaled_color_batch_builds_one_plan() {
        let _guard = region_scaled_color_plan_test_lock_for_test();
        reset_region_scaled_color_plan_builds_for_test();
        let input = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
        let roi = Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        };

        let result = decode_repeated_region_scaled_color_batch_direct_to_device(
            input.as_ref(),
            roi,
            Downscale::Half,
            PixelFormat::Rgb8,
            4,
        );

        assert!(result.is_err());
        assert_eq!(region_scaled_color_plan_builds_for_test(), 1);
    }

    #[test]
    fn known_repeated_region_scaled_color_batch_rejects_zero_count() {
        let _guard = region_scaled_color_plan_test_lock_for_test();
        reset_region_scaled_color_plan_builds_for_test();
        let result = decode_repeated_region_scaled_color_batch_direct_to_device(
            &[1_u8, 2, 3, 4],
            Rect {
                x: 0,
                y: 0,
                w: 64,
                h: 64,
            },
            Downscale::Half,
            PixelFormat::Rgb8,
            0,
        );

        assert!(matches!(result, Err(Error::MetalKernel { .. })));
        assert_eq!(region_scaled_color_plan_builds_for_test(), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn known_repeated_region_scaled_color_batch_reuses_cached_plan_across_calls() {
        if !should_run_metal_runtime() {
            return;
        }

        if Device::system_default().is_none() {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }

        let _guard = region_scaled_color_plan_test_lock_for_test();
        let input = encoded_rgb8_tile_for_region_scaled_plan_cache(17);
        let roi = region_scaled_plan_cache_roi();
        reset_region_scaled_color_plan_cache_for_test();
        reset_region_scaled_color_plan_builds_for_test();

        for _ in 0..2 {
            let surfaces = decode_repeated_region_scaled_color_batch_direct_to_device(
                input.as_ref(),
                roi,
                Downscale::Quarter,
                PixelFormat::Rgb8,
                4,
            )
            .expect("repeated RGB region-scaled batch");
            assert_eq!(surfaces.len(), 4);
        }

        assert_eq!(
            region_scaled_color_plan_builds_for_test(),
            1,
            "same RGB ROI+scaled batch should reuse the prepared direct color plan across calls"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn known_distinct_region_scaled_color_batch_reuses_cached_plans_across_calls() {
        if !should_run_metal_runtime() {
            return;
        }

        if Device::system_default().is_none() {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
            return;
        }

        let _guard = region_scaled_color_plan_test_lock_for_test();
        let first = encoded_rgb8_tile_for_region_scaled_plan_cache(29);
        let second = encoded_rgb8_tile_for_region_scaled_plan_cache(43);
        let roi = region_scaled_plan_cache_roi();
        let requests = vec![
            (first, roi, Downscale::Quarter),
            (second, roi, Downscale::Quarter),
        ];
        reset_region_scaled_color_plan_cache_for_test();
        reset_region_scaled_color_plan_builds_for_test();

        for _ in 0..2 {
            let surfaces =
                decode_region_scaled_color_batch_direct_to_device(&requests, PixelFormat::Rgb8)
                    .expect("distinct RGB region-scaled batch");
            assert_eq!(surfaces.len(), requests.len());
        }

        assert_eq!(
            region_scaled_color_plan_builds_for_test(),
            2,
            "same distinct RGB ROI+scaled inputs should reuse prepared direct color plans across calls"
        );
    }
}
