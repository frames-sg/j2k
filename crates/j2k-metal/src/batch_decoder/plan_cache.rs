// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent prepared-plan caches for exact Metal batch output.

use super::{
    Arc, DecodeRequest, Error, MetalBackendSession, MetalBatchDecoder, PixelFormat,
    PreparedBatchGroup, PreparedImage,
};

fn prepare_referenced_gray_plan(
    session: &MetalBackendSession,
    image: &PreparedImage,
) -> Result<Option<crate::compute::PreparedDirectGrayscalePlan>, Error> {
    if let Some(prepared) = image.htj2k_plan() {
        if !prepared.is_grayscale() {
            return Ok(None);
        }
        let referenced = prepared
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "J2K Metal does not recognize the prepared HTJ2K grayscale plan adapter",
            })?;
        return crate::compute::with_runtime_for_session(session, |_| {
            crate::compute::prepare_referenced_htj2k_grayscale_plan(referenced, image.bytes())
                .map(Some)
        });
    }
    let Some(prepared) = image.classic_plan() else {
        return Ok(None);
    };
    if !prepared.is_grayscale() {
        return Ok(None);
    }
    let referenced = prepared
        .adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedClassicPlan>()
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal does not recognize the prepared classic grayscale plan adapter",
        })?;
    crate::compute::with_runtime_for_session(session, |_| {
        crate::compute::prepare_referenced_classic_grayscale_plan(referenced, image.bytes())
            .map(Some)
    })
}

#[cfg(target_os = "macos")]
fn prepare_referenced_color_plan(
    session: &MetalBackendSession,
    image: &PreparedImage,
    fmt: PixelFormat,
) -> Result<Option<crate::compute::PreparedDirectColorPlan>, Error> {
    let rgba = matches!(
        fmt,
        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaI16
    );
    let signed = matches!(fmt, PixelFormat::RgbI16 | PixelFormat::RgbaI16);
    let plan = if let Some(prepared) = image.htj2k_plan() {
        if (rgba && !prepared.is_rgba()) || (!rgba && !prepared.is_color()) {
            return Ok(None);
        }
        let referenced = prepared
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "J2K Metal does not recognize the prepared HTJ2K color plan adapter",
            })?;
        crate::compute::with_runtime_for_session(session, |_| {
            if rgba {
                crate::compute::prepare_referenced_htj2k_rgba_plan(
                    referenced,
                    image.bytes(),
                    signed,
                )
            } else {
                crate::compute::prepare_referenced_htj2k_color_plan(
                    referenced,
                    image.bytes(),
                    signed,
                )
            }
        })?
    } else if let Some(prepared) = image.classic_plan() {
        if (rgba && !prepared.is_rgba()) || (!rgba && !prepared.is_color()) {
            return Ok(None);
        }
        let referenced = prepared
            .adapter_view()
            .downcast_ref::<j2k_native::J2kReferencedClassicPlan>()
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "J2K Metal does not recognize the prepared classic color plan adapter",
            })?;
        crate::compute::with_runtime_for_session(session, |_| {
            if rgba {
                crate::compute::prepare_referenced_classic_rgba_plan(
                    referenced,
                    image.bytes(),
                    signed,
                )
            } else {
                crate::compute::prepare_referenced_classic_color_plan(
                    referenced,
                    image.bytes(),
                    signed,
                )
            }
        })?
    } else {
        return Ok(None);
    };
    Ok(Some(plan))
}

#[cfg(target_os = "macos")]
pub(super) type PreparedGrayPlanCache =
    crate::session::PreparedPlanCache<Arc<crate::compute::PreparedDirectGrayscalePlan>>;

#[cfg(target_os = "macos")]
pub(super) type PreparedColorPlanCache =
    crate::session::PreparedPlanCache<Arc<crate::compute::PreparedDirectColorPlan>>;

#[cfg(target_os = "macos")]
pub(super) const PREPARED_BATCH_PLAN_CACHE_CAP: usize = 128;
impl MetalBatchDecoder {
    #[cfg(target_os = "macos")]
    pub(super) fn prepared_gray_group_plans(
        &mut self,
        group: &PreparedBatchGroup,
        fmt: PixelFormat,
        asynchronous: bool,
    ) -> Result<Vec<Arc<crate::compute::PreparedDirectGrayscalePlan>>, Error> {
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(group.images().len())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared grayscale group destination",
                source,
            })?;
        for image in group.images() {
            if let Some(plan) = self.prepared_gray_plan_for_image(image, fmt)? {
                plans.push(plan);
                continue;
            }
            if image.request() != DecodeRequest::Full {
                return Err(Error::UnsupportedMetalRequest {
                    reason: if asynchronous {
                        "J2K Metal asynchronous external ROI/reduction requires a prepared HTJ2K offset plan"
                    } else {
                        "J2K Metal external ROI/reduction requires a prepared HTJ2K offset plan"
                    },
                });
            }
            plans.push(
                crate::decoder::prepare_full_grayscale_direct_plan_with_session(
                    image.bytes(),
                    fmt,
                    self.backend_session(),
                )?,
            );
        }
        Ok(plans)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepared_color_group_plans(
        &mut self,
        group: &PreparedBatchGroup,
        fmt: PixelFormat,
    ) -> Result<Vec<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
        let mut plans = Vec::new();
        plans
            .try_reserve_exact(group.images().len())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K Metal prepared exact color group destination",
                source,
            })?;
        for image in group.images() {
            let Some(plan) = self.prepared_color_plan_for_image(image, fmt)? else {
                return Err(Error::UnsupportedMetalRequest {
                    reason: if matches!(
                        fmt,
                        PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaI16
                    ) {
                        "J2K Metal exact RGBA final-store requires a prepared classic or HTJ2K four-component offset plan"
                    } else {
                        "J2K Metal exact RGB final-store requires a prepared classic or HTJ2K three-component offset plan"
                    },
                });
            };
            plans.push(plan);
        }
        Ok(plans)
    }

    #[cfg(target_os = "macos")]
    fn prepared_gray_plan_for_image(
        &mut self,
        image: &PreparedImage,
        fmt: PixelFormat,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>, Error> {
        let key = crate::session::PreparedPlanCacheKey::prepared_gray(
            image.bytes(),
            image.request(),
            fmt,
        );
        if let Some(plan) = self.prepared_gray_plans.get(key) {
            return Ok(Some(plan.clone()));
        }
        let Some(plan) = prepare_referenced_gray_plan(self.backend_session(), image)? else {
            return Ok(None);
        };
        let plan = Arc::new(plan);
        self.prepared_gray_plans
            .insert(key, plan.clone())
            .map_err(|error| {
                crate::session::direct_plan_cache::prepared_plan_cache_error(
                    "J2K Metal prepared codec grayscale plan cache",
                    error,
                )
            })?;
        Ok(Some(plan))
    }

    #[cfg(target_os = "macos")]
    fn prepared_color_plan_for_image(
        &mut self,
        image: &PreparedImage,
        fmt: PixelFormat,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
        let key = crate::session::PreparedPlanCacheKey::prepared_color(
            image.bytes(),
            image.request(),
            fmt,
        );
        if let Some(plan) = self.prepared_color_plans.get(key) {
            return Ok(Some(plan.clone()));
        }
        let Some(plan) = prepare_referenced_color_plan(self.backend_session(), image, fmt)? else {
            return Ok(None);
        };
        let plan = Arc::new(plan);
        self.prepared_color_plans
            .insert(key, plan.clone())
            .map_err(|error| {
                crate::session::direct_plan_cache::prepared_plan_cache_error(
                    "J2K Metal prepared codec color plan cache",
                    error,
                )
            })?;
        Ok(Some(plan))
    }
}
