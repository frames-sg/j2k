// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use j2k_native::{
    DecodeSettings as NativeDecodeSettings, Image as NativeImage, J2kDirectGrayscalePlan,
};
#[cfg(target_os = "macos")]
use metal::Device;

#[cfg(target_os = "macos")]
use super::surface::CPU_STAGED_METAL_REQUIRES_EXPLICIT_API;
use super::J2kDecoder;
#[cfg(target_os = "macos")]
use crate::direct;
#[cfg(target_os = "macos")]
use crate::error::{adapter_backend_error, native_decode_j2k_error};
#[cfg(target_os = "macos")]
use crate::session::{
    cached_session_direct_color_plan, cached_session_direct_gray_plan, direct_gray_plan_cache_key,
    direct_plan_cache_key, store_session_direct_color_plan, store_session_direct_gray_plan,
};
#[cfg(target_os = "macos")]
use crate::{Error, MetalBackendSession, MetalDirectFallbackReason, Surface};

macro_rules! define_ensure_prepared_direct_plan {
    (
        with_session: $with_session:ident,
        plain: $plain:ident,
        prepare_fresh: $prepare_fresh:ident,
        plan_field: $plan_field:ident,
        prepared_field: $prepared_field:ident,
        prepared_ty: $prepared_ty:path,
        cache_key: $cache_key:ident,
        cached: $cached:ident,
        store: $store:ident,
        build: $build:ident,
        prepare: $prepare:path,
        label: $label:literal
    ) => {
        #[cfg(target_os = "macos")]
        fn $with_session(
            &mut self,
            session: &MetalBackendSession,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            let cache_key = $cache_key(self.bytes);
            if self.$prepared_field.is_none() {
                if let Some((plan, prepared)) = $cached(session, cache_key) {
                    self.$plan_field = Some(plan);
                    self.$prepared_field = Some(prepared);
                }
            }
            self.$prepare_fresh(Some((session, cache_key)))
        }

        #[cfg(target_os = "macos")]
        fn $plain(&mut self) -> Result<Option<Arc<$prepared_ty>>, Error> {
            self.$prepare_fresh(None)
        }

        #[cfg(target_os = "macos")]
        fn $prepare_fresh(
            &mut self,
            session_cache: Option<(&MetalBackendSession, u64)>,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            if self.$prepared_field.is_none() {
                self.ensure_native_image()?;
                let (Some(image), native_context) =
                    (self.native_image.as_ref(), &mut self.native_context)
                else {
                    return Err(Error::Decode(adapter_backend_error(
                        "native image cache missing".to_string(),
                    )));
                };
                let plan = match image.$build(native_context) {
                    Ok(plan) => plan,
                    Err(error) if direct::is_unsupported_direct_plan_error(&error) => {
                        return Ok(None);
                    }
                    Err(error) => {
                        return Err(Error::Decode(adapter_backend_error(format!(
                            "failed to build J2K MetalDirect {} plan: {error}",
                            $label
                        ))));
                    }
                };
                let prepared = Arc::new($prepare(&plan)?);
                if let Some((session, cache_key)) = session_cache {
                    $store(session, cache_key, &plan, prepared.clone());
                }
                self.$plan_field = Some(plan);
                self.$prepared_field = Some(prepared);
            }
            Ok(self.$prepared_field.clone())
        }
    };
}

#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_DIM: u32 = 512;
#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_COUNT: usize = 16;

impl J2kDecoder<'_> {
    #[cfg(target_os = "macos")]
    pub(super) fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.bytes, &NativeDecodeSettings::default())
                    .map_err(native_decode_j2k_error)?,
            );
        }
        Ok(())
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_gray_plan_with_session,
        plain: ensure_prepared_direct_gray_plan,
        prepare_fresh: prepare_fresh_direct_gray_plan,
        plan_field: native_direct_gray_plan,
        prepared_field: native_prepared_direct_gray_plan,
        prepared_ty: crate::compute::PreparedDirectGrayscalePlan,
        cache_key: direct_gray_plan_cache_key,
        cached: cached_session_direct_gray_plan,
        store: store_session_direct_gray_plan,
        build: build_direct_grayscale_plan_with_context,
        prepare: crate::compute::prepare_direct_grayscale_plan,
        label: "grayscale"
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_color_plan_with_session,
        plain: ensure_prepared_direct_color_plan,
        prepare_fresh: prepare_fresh_direct_color_plan,
        plan_field: native_direct_color_plan,
        prepared_field: native_prepared_direct_color_plan,
        prepared_ty: crate::compute::PreparedDirectColorPlan,
        cache_key: direct_plan_cache_key,
        cached: cached_session_direct_color_plan,
        store: store_session_direct_color_plan,
        build: build_direct_color_plan_with_context,
        prepare: crate::compute::prepare_direct_color_plan,
        label: "color"
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_direct_to_surface(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan()? else {
                return Ok(None);
            };
            return Ok(Some(
                crate::compute::execute_prepared_direct_grayscale_plan(&plan, fmt)?,
            ));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan()? else {
                return Ok(None);
            };
            return match crate::compute::execute_prepared_direct_color_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_color_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_direct_to_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan_with_session(session)? else {
                return Ok(None);
            };
            return Ok(Some(
                crate::compute::execute_prepared_direct_grayscale_plan_with_device(
                    &plan,
                    fmt,
                    session.device_handle(),
                )?,
            ));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan_with_session(session)? else {
                return Ok(None);
            };
            return match crate::compute::execute_prepared_direct_color_plan_with_device(
                &plan,
                fmt,
                session.device_handle(),
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_color_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_region_scaled_direct_to_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<Option<Surface>, Error> {
        crate::hybrid::decode_region_scaled_direct_to_surface(self.bytes, fmt, roi, scale)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_region_scaled_direct_to_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Option<Surface>, Error> {
        crate::hybrid::decode_region_scaled_direct_to_surface_with_session(
            self.bytes, fmt, roi, scale, session,
        )
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_full_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(adapter_backend_error(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_to_surface(image, native_context, fmt)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_full_to_metal_surface_with_device(
        &mut self,
        fmt: PixelFormat,
        device: &Device,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(adapter_backend_error(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_to_surface_with_device(image, native_context, fmt, device)
    }

    #[cfg(target_os = "macos")]
    fn decode_repeated_grayscale_cpu_to_surfaces(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(count);
        for _ in 0..count {
            surfaces.push(self.decode_to_cpu_surface(fmt)?);
        }
        Ok(surfaces)
    }

    #[cfg(target_os = "macos")]
    fn should_auto_use_direct_for_repeated(
        plan: &J2kDirectGrayscalePlan,
        fmt: PixelFormat,
        count: usize,
    ) -> bool {
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) || count == 0 {
            return false;
        }

        let max_dim = plan.dimensions.0.max(plan.dimensions.1);
        max_dim >= AUTO_REPEATED_GRAYSCALE_MIN_DIM && count >= AUTO_REPEATED_GRAYSCALE_MIN_COUNT
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(adapter_backend_error(
                    "native image cache missing".to_string(),
                )));
            };
            let plan = image
                .build_direct_grayscale_plan_with_context(native_context)
                .map_err(native_decode_j2k_error)?;
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_prepared_direct_gray_plan.as_ref() else {
            return Ok(Vec::new());
        };
        crate::compute::execute_repeated_prepared_direct_grayscale_plan(plan, fmt, count)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_color_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let surface = self.decode_to_surface_impl(fmt, BackendRequest::Metal)?;
        Ok(vec![surface; count])
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_auto_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        let dims = self.inner.info().dimensions;
        if dims.0.max(dims.1) < AUTO_REPEATED_GRAYSCALE_MIN_DIM
            || count < AUTO_REPEATED_GRAYSCALE_MIN_COUNT
        {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(adapter_backend_error(
                    "native image cache missing".to_string(),
                )));
            };
            let Ok(plan) = image.build_direct_grayscale_plan_with_context(native_context) else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_direct_gray_plan.as_ref() else {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        };
        if Self::should_auto_use_direct_for_repeated(plan, fmt, count) {
            let Some(prepared) = self.native_prepared_direct_gray_plan.as_ref() else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            crate::compute::execute_repeated_prepared_direct_grayscale_plan(prepared, fmt, count)
        } else {
            self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count)
        }
    }
}

#[cfg(target_os = "macos")]
fn is_direct_color_runtime_fallback_error(error: &Error) -> bool {
    is_direct_runtime_fallback_error(error)
}

#[cfg(target_os = "macos")]
pub(crate) fn is_direct_runtime_fallback_error(error: &Error) -> bool {
    error.is_direct_fallback()
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_grayscale_batch_direct_to_device(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full grayscale batch does not support {fmt:?}"),
        });
    }

    let mut plans = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let Some(plan) = decoder.ensure_prepared_direct_gray_plan()? else {
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full grayscale Gray8/Gray16 only; fmt={fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
            });
        };
        plans.push(plan);
    }
    crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_color_batch_direct_to_device(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full color batch does not support {fmt:?}"),
        });
    }

    let mut plans = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let Some(plan) = decoder.ensure_prepared_direct_color_plan()? else {
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full RGB color only; fmt={fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
            });
        };
        plans.push(plan);
    }
    match crate::compute::execute_prepared_direct_color_plan_batch(&plans, fmt) {
        Ok(surfaces) => Ok(surfaces),
        Err(error) if is_direct_color_runtime_fallback_error(&error) => {
            Err(Error::UnsupportedMetalRequest {
                reason: CPU_STAGED_METAL_REQUIRES_EXPLICIT_API,
            })
        }
        Err(error) => Err(error),
    }
}
