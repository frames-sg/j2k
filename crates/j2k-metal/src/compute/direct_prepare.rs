// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment, J2kHtCleanupBatchJob};
use super::direct_roi::BandRequiredRegion;
use super::{
    classic_style_flags, prepare_direct_tier1_input_buffer, with_runtime, CpuTier1CoefficientCache,
    DirectTier1Mode, Error, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedClassicSubBandGroupMember, PreparedDirectColorPlan,
    PreparedDirectGrayscalePlan, PreparedDirectGrayscaleStep, PreparedDirectIdwt,
    PreparedHtPayloadSource, PreparedHtSubBand, PreparedHtSubBandGroup,
    PreparedHtSubBandGroupMember,
};
#[cfg(target_os = "macos")]
use j2k_native::{
    HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange,
    J2kDirectGrayscalePlan as NativeGrayscalePlan, J2kReferencedClassicPlan,
    J2kReferencedHtj2kPlan,
};

mod classic;
mod color;
mod grayscale;
mod ht;
mod referenced;

pub(in crate::compute) use self::classic::{
    prepare_classic_sub_band, prepare_classic_sub_band_groups, prepare_sub_band_groups,
};
use self::classic::{prepare_referenced_classic_sub_band, ReferencedClassicPayloadCursor};
pub(crate) use self::color::{
    prepare_referenced_classic_color_plan, prepare_referenced_classic_rgba_plan,
    prepare_referenced_htj2k_color_plan, prepare_referenced_htj2k_rgba_plan,
};
pub(in crate::compute) use self::grayscale::prepare_direct_grayscale_plan_for_cpu_upload;
pub(crate) use self::grayscale::{
    prepare_direct_grayscale_plan, prepare_referenced_classic_grayscale_plan,
    prepare_referenced_htj2k_grayscale_plan,
};
use self::ht::prepare_referenced_ht_sub_band;
pub(in crate::compute) use self::ht::{prepare_ht_sub_band, prepare_ht_sub_band_groups};
use self::referenced::{
    append_referenced_classic_component_steps, append_referenced_htj2k_component_steps,
    finish_referenced_component_plan, validate_payload_record_span,
    validate_referenced_component_metadata,
};
