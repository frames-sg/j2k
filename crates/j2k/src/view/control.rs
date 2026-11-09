// SPDX-License-Identifier: MIT OR Apache-2.0

//! Controls carried by one decode operation, separate from reusable source facts.

use alloc::sync::Arc;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::J2kError;

/// Cooperative cancellation shared with a running decode operation.
///
/// Clones address the same flag. Create a new token for an independent operation;
/// cancellation is permanent for a token and does not modify encoded input.
#[derive(Debug, Clone, Default)]
pub struct J2kCancellationToken {
    flag: Arc<AtomicBool>,
}

impl J2kCancellationToken {
    /// Create an uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask associated operations to stop at their next cooperative checkpoint.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }

    pub(crate) fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag)
    }

    /// Identity of the shared flag: equal exactly for clones of one token.
    pub(crate) fn flag_ptr(&self) -> *const AtomicBool {
        Arc::as_ptr(&self.flag)
    }
}

/// Resource limits and cancellation for one decode operation.
///
/// These controls are independent of reusable preparation and of the existing
/// batch scheduling estimate in [`crate::BatchDecodeOptions`]. Native working
/// memory includes parsed native metadata and execution scratch; caller-owned
/// compressed input, caller-owned immutable source preparation, and caller-supplied
/// output buffers are separate. Newly derived operation plans and codec-owned
/// conversion intermediates and outputs count while they are assembled. This is not a
/// process RSS limit and cannot raise the native engine's built-in ceiling.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct J2kDecodeControl {
    /// Optional upper bound for native decode working allocations.
    pub max_working_memory_bytes: Option<NonZeroUsize>,
    /// Optional upper bound for the final packed output span, including row gaps.
    pub max_output_bytes: Option<usize>,
    /// Optional cooperative cancellation token.
    pub cancellation: Option<J2kCancellationToken>,
}

impl J2kDecodeControl {
    pub(crate) fn with_native_context<'a, T>(
        &self,
        context: &mut j2k_native::DecoderContext<'a>,
        operation: impl FnOnce(&mut j2k_native::DecoderContext<'a>) -> Result<T, J2kError>,
    ) -> Result<T, J2kError> {
        self.check_cancelled()?;
        context
            .with_operation_control(self.to_native(), operation)
            .map_err(|error| self.normalize_error(error))
    }

    pub(crate) fn native_error(&self, error: j2k_native::DecodeError) -> J2kError {
        self.normalize_error(J2kError::from_native_decode_error(error))
    }

    pub(crate) fn normalize_error(&self, mut error: J2kError) -> J2kError {
        if self.working_baseline() != 0 {
            if let J2kError::ResourceLimit { requested, cap, .. } = &mut error {
                // Some checks precede the virtual working reservation. Keep
                // their original typed source, and never guess which request
                // byte count includes that reservation.
                if *cap == Some(j2k_native::DEFAULT_MAX_DECODE_BYTES) {
                    *requested = None;
                    *cap = Some(j2k_native::DEFAULT_MAX_DECODE_BYTES - self.working_baseline());
                }
            }
        }
        error
    }

    pub(crate) fn to_native(&self) -> j2k_native::DecodeOperationControl {
        j2k_native::DecodeOperationControl::new(
            self.max_working_memory_bytes.map(NonZeroUsize::get),
            self.cancellation.as_ref().map(J2kCancellationToken::flag),
        )
    }
    pub(crate) fn check_cancelled(&self) -> Result<(), J2kError> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(J2kCancellationToken::is_cancelled)
        {
            return Err(J2kError::Cancelled);
        }
        Ok(())
    }

    pub(crate) fn check_output(&self, bytes: usize) -> Result<(), J2kError> {
        self.check_cancelled()?;
        if let Some(cap) = self.max_output_bytes {
            j2k_core::ensure_allocation_within_cap(bytes, cap, "decode operation output")?;
        }
        Ok(())
    }

    pub(crate) fn working_baseline(&self) -> usize {
        j2k_native::DEFAULT_MAX_DECODE_BYTES.saturating_sub(
            self.max_working_memory_bytes
                .map_or(j2k_native::DEFAULT_MAX_DECODE_BYTES, NonZeroUsize::get),
        )
    }
}

impl super::J2kDecoder<'_> {
    /// [`Self::max_reduction_levels`] under operation-specific limits and
    /// cancellation, including the tile-header scan it may need.
    ///
    /// # Errors
    /// Returns metadata, tile-header, resource-limit or cancellation errors.
    pub fn max_reduction_levels_with_control(
        &mut self,
        control: &J2kDecodeControl,
    ) -> Result<u8, J2kError> {
        let result = self.source_max_reduction_levels_with_control(control);
        self.trim_preparation_cache();
        result
    }

    /// Plan explicit sRGB region output under operation-specific limits and
    /// cancellation, as [`Self::srgb8_region_decode_plan`] plans it.
    ///
    /// The output limit is checked against tight packing.
    /// # Errors
    /// Returns the errors of [`Self::srgb8_region_decode_plan`], or a
    /// resource-limit or cancellation error.
    pub fn srgb8_region_decode_plan_with_control(
        &mut self,
        roi: crate::Rect,
        levels: u8,
        control: &J2kDecodeControl,
    ) -> Result<super::J2kRegionDecodePlan, J2kError> {
        let result = self
            .region_decode_plan_with_interpretation_and_control(None, roi, levels, control)
            .and_then(|plan| {
                control.check_output(plan.buffer_bytes)?;
                Ok(plan)
            });
        self.trim_preparation_cache();
        result
    }

    /// Plan native-sample region output under operation-specific limits and
    /// cancellation, as [`Self::native_region_decode_plan`] plans it.
    ///
    /// The output limit is checked against tight packing.
    /// # Errors
    /// Returns the errors of [`Self::native_region_decode_plan`], or a
    /// resource-limit or cancellation error.
    pub fn native_region_decode_plan_with_control(
        &mut self,
        roi: crate::Rect,
        levels: u8,
        control: &J2kDecodeControl,
    ) -> Result<super::J2kNativeRegionDecodePlan, J2kError> {
        let result = self
            .native_region_decode_plan_with_interpretation(roi, levels, control)
            .and_then(|plan| {
                control.check_output(plan.region.buffer_bytes)?;
                Ok(plan)
            });
        self.trim_preparation_cache();
        result
    }

    /// Plan a packed region under operation-specific limits and cancellation.
    ///
    /// The output limit is checked against tight packing here; decoding also
    /// checks the caller's actual stride. Planning does not allocate output.
    /// # Errors
    /// Returns geometry, capability, resource-limit or cancellation errors.
    pub fn region_decode_plan_with_control(
        &mut self,
        fmt: crate::PixelFormat,
        roi: crate::Rect,
        levels: u8,
        control: &J2kDecodeControl,
    ) -> Result<super::J2kRegionDecodePlan, J2kError> {
        let result = self
            .region_decode_plan_with_interpretation_and_control(Some(fmt), roi, levels, control)
            .and_then(|plan| {
                control.check_output(plan.buffer_bytes)?;
                Ok(plan)
            });
        self.trim_preparation_cache();
        result
    }

    /// Decode a packed source region at an exact reduction under operation controls.
    ///
    /// Caller-supplied output is excluded from working memory. Its required span,
    /// including row gaps but excluding final padding, obeys the output limit.
    /// Cancellation is cooperative; an error can leave a prefix written. A token
    /// selects bounded native serial blocks and row bands, avoiding bulk work
    /// that cannot be interrupted. No request silently changes its reduction.
    ///
    /// # Errors
    /// Returns the geometry and format errors of [`Self::region_decode_plan`],
    /// resource-limit or cancellation errors, or a decode error.
    pub fn decode_region_scaled_pow2_into_with_control(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: crate::PixelFormat,
        roi: crate::Rect,
        levels: u8,
        control: &J2kDecodeControl,
    ) -> Result<crate::decode::J2kDecodeOutcome, J2kError> {
        let result = (|| {
            let plan = self.region_decode_plan_with_interpretation_and_control(
                Some(fmt),
                roi,
                levels,
                control,
            )?;
            let span = plan.buffer_len(stride)?;
            control.check_output(span)?;
            super::validate_buffer(
                (plan.output_region.w, plan.output_region.h),
                out.len(),
                stride,
                fmt,
            )?;
            let image = self
                .image
                .as_ref()
                .ok_or_else(|| J2kError::internal_backend("internal image cache missing"))?;
            let warnings = crate::decode::decode_warnings_for_image(image);
            control.with_native_context(&mut self.native_context, |context| {
                crate::decode::decode_image_region_into_with_native_context(
                    image,
                    context,
                    out,
                    stride,
                    fmt,
                    plan.output_region,
                )
            })?;
            Ok(j2k_core::DecodeOutcome::new(plan.output_region, warnings))
        })();
        self.trim_preparation_cache();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::error::Error;

    #[test]
    fn early_native_budget_failure_preserves_source_without_guessing_request_bytes() {
        let bytes = include_bytes!(
            "../../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
        );
        let image = j2k_native::Image::new(bytes, &j2k_native::DecodeSettings::default()).unwrap();
        let mut context = j2k_native::DecoderContext::default();
        let control = J2kDecodeControl {
            max_working_memory_bytes: NonZeroUsize::new(16 * 1024 * 1024),
            ..J2kDecodeControl::default()
        };
        let error = control
            .with_native_context(&mut context, |context| {
                image
                    .decode_region_with_context_and_retained_capacity(
                        (0, 0, 1, 1),
                        context,
                        j2k_native::DEFAULT_MAX_DECODE_BYTES,
                    )
                    .map(|_| ())
                    .map_err(J2kError::from_native_decode_error)
            })
            .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("working memory limit 16777216 bytes"),
            "{message}"
        );
        assert!(
            !message.contains(&j2k_native::DEFAULT_MAX_DECODE_BYTES.to_string()),
            "{message}"
        );
        let J2kError::ResourceLimit {
            requested,
            cap,
            source,
        } = error
        else {
            panic!("expected resource limit");
        };
        assert_eq!(requested, None);
        assert_eq!(cap, Some(16 * 1024 * 1024));
        let native = source
            .source()
            .unwrap()
            .downcast_ref::<j2k_native::DecodeError>()
            .unwrap();
        assert!(
            matches!(native, j2k_native::DecodeError::AllocationTooLarge { requested, cap, .. }
            if *requested > j2k_native::DEFAULT_MAX_DECODE_BYTES && *cap == j2k_native::DEFAULT_MAX_DECODE_BYTES)
        );
    }
}
