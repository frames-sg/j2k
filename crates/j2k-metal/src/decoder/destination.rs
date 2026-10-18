// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-image decode into caller-owned Metal storage.

use j2k::BatchLayout;
use j2k_core::PixelFormat;

use super::J2kDecoder;
use crate::engine::DirectDestinationConsumerOrdering;
use crate::{Error, MetalBackendSession, MetalDirectFallbackReason, MetalImageDestination};

impl J2kDecoder<'_> {
    /// Decode the full image directly into caller-owned Metal storage.
    ///
    /// The destination's layout must describe one image with the image's
    /// dimensions; rows may be padded and the image may start at a nonzero
    /// offset. Its pixel format selects the output, with the same samples
    /// [`Self::decode_request_to_device`] produces for that format:
    ///
    /// - `Gray8` or `Gray16` for single-component images, at any precision.
    /// - `Rgb8` for 8-bit three-component images, and `Rgb16` for
    ///   three-component images of 9 to 16 bits.
    ///
    /// The final store writes into that range instead of a new
    /// [`Surface`](crate::Surface), so a stream of same-sized frames can
    /// reuse one allocation; bytes outside the range are not written. Like
    /// [`Self::decode_request_to_device`], the decoder keeps its prepared plan
    /// for later calls.
    ///
    /// This returns once the GPU work has completed, after which the caller
    /// may read the destination.
    ///
    /// # Errors
    ///
    /// Returns a Metal support error when the destination's device, size, or
    /// pixel format does not fit the image; a capability rejection for other
    /// formats and precisions; [`Error::MetalDirectFallback`] with
    /// [`MetalDirectFallbackReason::UnsupportedPlan`] when the codestream
    /// cannot take the direct Metal route, in which case
    /// [`Self::decode_request_to_device`] still decodes it; and a decode error
    /// for invalid codestreams or GPU failures.
    pub fn decode_into_metal(&mut self, destination: &MetalImageDestination) -> Result<(), Error> {
        self.decode_into_metal_on(None, destination)
    }

    /// [`Self::decode_into_metal`] on the device and queue of `session`.
    ///
    /// The destination must belong to the session's device. Prepared plans
    /// are shared through the session's plan cache, as for
    /// [`Self::decode_request_to_device_with_session`].
    ///
    /// # Errors
    ///
    /// As for [`Self::decode_into_metal`].
    pub fn decode_into_metal_with_session(
        &mut self,
        destination: &MetalImageDestination,
        session: &MetalBackendSession,
    ) -> Result<(), Error> {
        self.decode_into_metal_on(Some(session), destination)
    }

    fn decode_into_metal_on(
        &mut self,
        session: Option<&MetalBackendSession>,
        destination: &MetalImageDestination,
    ) -> Result<(), Error> {
        let fmt = destination.layout().pixel_format();
        let info = self.inner.info();
        let (dimensions, components, bit_depth) =
            (info.dimensions, info.components, info.bit_depth);
        // The store below also checks the destination's device.
        destination
            .validate_image(dimensions, fmt)
            .map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K Metal decode destination validation failed",
                    source,
                )
            })?;
        let supported = match (fmt, components) {
            (PixelFormat::Gray8 | PixelFormat::Gray16, 1) => true,
            // The color destination store writes samples at their native
            // precision, which equals the surface conversion only when the
            // output sample holds the source precision unscaled.
            (PixelFormat::Rgb8, 3) => bit_depth == 8,
            (PixelFormat::Rgb16, 3) => (9..=16).contains(&bit_depth),
            _ => false,
        };
        if !supported {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    "J2K Metal decode into a destination supports Gray8 and Gray16 for one component, Rgb8 for 8-bit RGB, and Rgb16 for 9- to 16-bit RGB",
                ),
            ));
        }
        match session {
            // Run on the session's runtime, as surface decodes do, so its
            // buffer pools and caches carry over between frames.
            Some(session) => crate::engine::with_runtime_for_session(session, |_| {
                self.decode_supported_into_metal(Some(session), fmt, destination)
            }),
            None => self.decode_supported_into_metal(None, fmt, destination),
        }
    }

    fn decode_supported_into_metal(
        &mut self,
        session: Option<&MetalBackendSession>,
        fmt: PixelFormat,
        destination: &MetalImageDestination,
    ) -> Result<(), Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let plan = match session {
                Some(session) => {
                    self.ensure_prepared_direct_gray_plan_with_session(fmt, session)?
                }
                None => self.ensure_prepared_direct_gray_plan()?,
            }
            .ok_or_else(|| unsupported_plan(fmt))?;
            return crate::engine::execute_prepared_direct_grayscale_plan_into(
                &plan,
                fmt,
                destination,
            );
        }
        let plan = match session {
            Some(session) => self.ensure_prepared_direct_color_plan_with_session(fmt, session)?,
            None => self.ensure_prepared_direct_color_plan()?,
        }
        .ok_or_else(|| unsupported_plan(fmt))?;
        crate::engine::submit_prepared_direct_color_plan_batch_into_group(
            crate::engine::current_runtime()?,
            &[plan],
            fmt,
            BatchLayout::Nhwc,
            destination,
            None,
            DirectDestinationConsumerOrdering::HostCompletionOnly,
        )?
        .wait()?;
        Ok(())
    }
}

fn unsupported_plan(fmt: PixelFormat) -> Error {
    Error::MetalDirectFallback {
        message: format!(
            "J2K Metal decode into a destination could not build a direct plan; fmt={fmt:?}"
        ),
        reason: MetalDirectFallbackReason::UnsupportedPlan,
    }
}
