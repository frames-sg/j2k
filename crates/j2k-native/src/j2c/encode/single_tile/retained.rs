// SPDX-License-Identifier: MIT OR Apache-2.0

//! Repeated irreversible encodes of one image at different quantization scales.
//!
//! Deinterleave, the forward ICT and the 9/7 DWT never read the scale, so they
//! run once; validation, planning, quantization, Tier-1, packets and the
//! codestream rerun for every candidate. Later phases run with the retained
//! coefficients as their session baseline, where an ordinary encode passes them
//! to Tier-1 as `source_retained_bytes`, so they can only refuse more: any
//! error drops the coefficients and returns the ordinary encode's result.

use super::super::multitile::encode_options_retained_bytes;
use super::super::{
    block_coding_mode, CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions, EncodeRoiRegion,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeRetainedInput,
    NativeEncodeSession, Vec,
};
use super::ownership::single_tile_plan_retained_bytes;
use super::{
    build_single_tile_plan, encode_tile_packets, finalize_staged_codestream,
    prepare_accelerated_components, prepared_transforms_retained_bytes, profile,
    validate_encode_request, AcceleratedComponentRequest, OwnedDwtComponent, SingleTilePlan,
    TransformStageTimings, ValidatedEncodeRoute,
};
use crate::{EncodeResult, DEFAULT_MAX_CODEC_BYTES};

/// CPU encoder for one image at a sequence of irreversible quantization scales.
///
/// Each [`encode_at_scale`](Self::encode_at_scale) call returns exactly what
/// [`encode_with_roi_regions`](crate::encode_with_roi_regions) returns with
/// [`EncodeOptions::irreversible_quantization_scale`] replaced.
/// [`retained_bytes`](Self::retained_bytes) reports what stays live between
/// calls. Implementation-facing: rate search in the `j2k` facade drives it.
#[doc(hidden)]
pub struct IrreversibleScaleEncoder<'a> {
    pixels: &'a [u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: EncodeOptions,
    roi_regions: &'a [EncodeRoiRegion],
    /// Cap of this encoder's own sessions; tests lower it.
    cap: usize,
    transform: Option<RetainedTransform>,
    /// Cleared once retention is ineligible or was refused.
    reuse: bool,
}

struct RetainedTransform {
    decompositions: Vec<OwnedDwtComponent>,
    bytes: usize,
    /// Plan owner bytes the transform stage was budgeted with.
    plan_bytes: usize,
    /// Reported in the first candidate's profile row only.
    timings: TransformStageTimings,
}

impl<'a> IrreversibleScaleEncoder<'a> {
    /// Describe the image and options; nothing runs until the first call.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "this codec boundary mirrors encode_with_roi_regions' explicit geometry and sample representation"
    )]
    pub fn new(
        pixels: &'a [u8],
        width: u32,
        height: u32,
        num_components: u16,
        bit_depth: u8,
        signed: bool,
        options: EncodeOptions,
        roi_regions: &'a [EncodeRoiRegion],
    ) -> Self {
        let reuse = !options.reversible && !options.validate_high_throughput_codestream;
        Self {
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            options,
            roi_regions,
            cap: DEFAULT_MAX_CODEC_BYTES,
            transform: None,
            reuse,
        }
    }

    /// Encode at `quantization_scale`.
    ///
    /// # Errors
    ///
    /// Returns the error [`encode_with_roi_regions`](crate::encode_with_roi_regions)
    /// returns for the same request at this scale.
    pub fn encode_at_scale(&mut self, quantization_scale: f32) -> EncodeResult<Vec<u8>> {
        self.options.irreversible_quantization_scale = quantization_scale;
        if self.reuse && self.transform.is_none() {
            // A failure here is the ordinary encode's own, returned below.
            if let Ok(transform) = self.transform_tile() {
                self.reuse = transform.is_some();
                self.transform = transform;
            }
        }
        if let Some(mut transform) = self.transform.take() {
            if let Ok(codestream) = self.encode_transformed(&mut transform) {
                self.transform = Some(transform);
                return Ok(codestream);
            }
            // Not a silent failure: the ordinary encode below is authoritative.
            self.reuse = false;
        }
        crate::encode_with_roi_regions(
            self.pixels,
            self.width,
            self.height,
            self.num_components,
            self.bit_depth,
            self.signed,
            &self.options,
            self.roi_regions,
        )
    }

    /// Heap bytes kept live between calls: retained coefficients and options.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        let transform = self
            .transform
            .as_ref()
            .map_or(0, |transform| transform.bytes);
        encode_options_retained_bytes(&self.options)
            .map_or(usize::MAX, |options| options.saturating_add(transform))
    }

    /// Validate and plan the whole-image tile; `None` for tiled requests.
    fn plan(
        &self,
        session: &NativeEncodeSession<'_>,
    ) -> NativeEncodePipelineResult<Option<SingleTilePlan>> {
        let mode = block_coding_mode(&self.options);
        let ValidatedEncodeRoute::SingleTile(validated) = validate_encode_request(
            self.pixels.len(),
            self.width,
            self.height,
            self.num_components,
            self.bit_depth,
            &self.options,
            mode,
            &[],
            session,
        )?
        else {
            return Ok(None);
        };
        build_single_tile_plan(
            validated,
            self.width,
            self.height,
            self.num_components,
            self.bit_depth,
            self.signed,
            &self.options,
            mode,
            self.roi_regions,
            &[],
            session,
        )
        .map(Some)
    }

    /// The ordinary encode's transform stage. The CPU-only accelerator declines
    /// the whole-tile HT hook, so this staged route is the one it takes.
    fn transform_tile(&self) -> NativeEncodePipelineResult<Option<RetainedTransform>> {
        let session =
            NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), self.cap)?;
        let Some(plan) = self.plan(&session)? else {
            return Ok(None);
        };
        let prepared = prepare_accelerated_components(
            &AcceleratedComponentRequest {
                pixels: self.pixels,
                width: self.width,
                height: self.height,
                num_components: self.num_components,
                bit_depth: self.bit_depth,
                signed: self.signed,
                options: &self.options,
                plan: &plan,
                profile_enabled: profile::profile_stages_enabled(),
                session: &session,
            },
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        )?;
        Ok(Some(RetainedTransform {
            bytes: prepared_transforms_retained_bytes(&prepared)?,
            plan_bytes: single_tile_plan_retained_bytes(&plan)?,
            timings: TransformStageTimings {
                deinterleave: prepared.deinterleave_us,
                mct: prepared.mct_us,
                dwt: prepared.dwt_us,
            },
            decompositions: prepared.decompositions,
        }))
    }

    /// Quantize, code, packetize and write one candidate from `transform`.
    fn encode_transformed(
        &self,
        transform: &mut RetainedTransform,
    ) -> NativeEncodePipelineResult<Vec<u8>> {
        let timings = core::mem::take(&mut transform.timings);
        let session = NativeEncodeSession::try_with_cap(
            NativeEncodeRetainedInput::from_owner_bytes(&transform.decompositions, transform.bytes),
            self.cap,
        )?;
        let profile_enabled = profile::profile_stages_enabled();
        let total_start = profile::profile_now(profile_enabled);
        let invariant = || NativeEncodePipelineError::internal_invariant("retained plan changed");
        let mut plan = self.plan(&session)?.ok_or_else(invariant)?;
        // The transform stage was budgeted with the first plan's owners.
        if single_tile_plan_retained_bytes(&plan)? > transform.plan_bytes {
            return Err(invariant());
        }
        let encoded = encode_tile_packets(
            self.width,
            self.height,
            self.num_components,
            self.bit_depth,
            &self.options,
            &[],
            &plan,
            &transform.decompositions,
            0,
            profile_enabled,
            &session,
            &mut CpuOnlyJ2kEncodeStageAccelerator,
        )?;
        plan.params.required_ht_magnitude_bound =
            encoded.packetized_tile.required_ht_magnitude_bound;
        finalize_staged_codestream(
            &self.options,
            &plan.into_codestream_final_plan(),
            timings,
            &encoded,
            profile_enabled,
            total_start,
            &session,
        )
    }
}

#[cfg(test)]
mod tests;
