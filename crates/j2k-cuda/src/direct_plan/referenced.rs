// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
};

use super::{
    classic::referenced::{append_referenced_classic_subband, referenced_classic_payload_bytes},
    ht::{
        append_referenced_ht_subband_with_mode, referenced_payload_bytes, ReferencedHtPayloadMode,
    },
    shared::{self, CudaPlanOwners},
    CudaHtj2kDecodePlan, Error, PLAN_PAYLOAD_TOO_LARGE, REFERENCED_PLAN_PAYLOAD_MISMATCH,
};
use crate::allocation::HostPhaseBudget;

impl CudaHtj2kDecodePlan {
    pub(crate) fn from_referenced_tile_grayscale_plan_direct_source(
        plan: &J2kDirectGrayscalePlan,
        ht_payloads: &[HtCodeBlockPayloadRanges],
        encoded: &[u8],
        output_format: PixelFormat,
        output_origin: (u32, u32),
        output_dimensions: (u32, u32),
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Self, Error> {
        Self::from_referenced_tile_grayscale_plan_with_mode(
            plan,
            ht_payloads,
            &[],
            &[],
            encoded,
            output_format,
            output_origin,
            output_dimensions,
            ReferencedHtPayloadMode::DirectSource,
            host_budget,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the tile adapter explicitly carries both referenced entropy sources, output geometry, shared arena, and allocation budget"
    )]
    pub(crate) fn from_referenced_tile_grayscale_plan_into_shared(
        plan: &J2kDirectGrayscalePlan,
        ht_payloads: &[HtCodeBlockPayloadRanges],
        classic_payloads: &[J2kClassicCodeBlockPayload],
        classic_ranges: &[J2kCodestreamRange],
        encoded: &[u8],
        output_format: PixelFormat,
        output_origin: (u32, u32),
        output_dimensions: (u32, u32),
        shared_payload: &mut Vec<u8>,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Self, Error> {
        let payload_bytes = referenced_payload_bytes(encoded, ht_payloads)?
            .checked_add(referenced_classic_payload_bytes(
                encoded,
                classic_payloads,
                classic_ranges,
            )?)
            .ok_or(Error::capability_rejected(
                j2k_core::CapabilityRejection::resource_limit(PLAN_PAYLOAD_TOO_LARGE),
            ))?;
        if payload_bytes != 0 {
            host_budget.try_vec_reserve(shared_payload, payload_bytes)?;
        }
        Self::from_referenced_tile_grayscale_plan_with_mode(
            plan,
            ht_payloads,
            classic_payloads,
            classic_ranges,
            encoded,
            output_format,
            output_origin,
            output_dimensions,
            ReferencedHtPayloadMode::Compact(shared_payload),
            host_budget,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the common referenced-plan conversion retains explicit entropy sources, output geometry, payload mode, and allocation budget"
    )]
    fn from_referenced_tile_grayscale_plan_with_mode(
        plan: &J2kDirectGrayscalePlan,
        ht_payloads: &[HtCodeBlockPayloadRanges],
        classic_payloads: &[J2kClassicCodeBlockPayload],
        classic_ranges: &[J2kCodestreamRange],
        encoded: &[u8],
        output_format: PixelFormat,
        output_origin: (u32, u32),
        output_dimensions: (u32, u32),
        mut payload_mode: ReferencedHtPayloadMode<'_>,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Self, Error> {
        let mut owners = CudaPlanOwners::from_referenced_plan_with_budget(plan, host_budget)?;
        let mut ht_payloads = ht_payloads.iter();
        let mut classic_payloads = classic_payloads.iter();
        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::HtSubBand(subband) => {
                    append_referenced_ht_subband_with_mode(
                        &mut owners,
                        subband,
                        None,
                        &mut ht_payloads,
                        encoded,
                        &mut payload_mode,
                    )?;
                }
                J2kDirectGrayscaleStep::ClassicSubBand(subband) => {
                    let ReferencedHtPayloadMode::Compact(shared_payload) = &mut payload_mode else {
                        return Err(Error::capability_rejected(
                            j2k_core::CapabilityRejection::unsupported_operation(
                                "direct-source CUDA HTJ2K payloads do not support classic code blocks",
                            ),
                        ));
                    };
                    append_referenced_classic_subband(
                        &mut owners,
                        subband,
                        None,
                        &mut classic_payloads,
                        classic_ranges,
                        encoded,
                        shared_payload,
                    )?;
                }
                J2kDirectGrayscaleStep::Idwt(step) => owners.append_idwt(*step)?,
                J2kDirectGrayscaleStep::Store(step) => {
                    owners
                        .store_steps
                        .push(shared::convert_referenced_tile_store_step(
                            *step,
                            output_dimensions,
                        )?);
                }
            }
        }
        if ht_payloads.next().is_some() || classic_payloads.next().is_some() {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::geometry_mismatch(REFERENCED_PLAN_PAYLOAD_MISMATCH),
            ));
        }
        owners.finish(plan, output_format, output_origin, output_dimensions)
    }
}
