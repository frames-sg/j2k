// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated packed-subband geometry and prepared-output metadata.

use alloc::vec::Vec;

use super::PackedSubbandRequest;
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodePipelineResult, PreparedEncodeCodeBlock,
    PreparedEncodeSubband,
};

#[derive(Clone, Copy)]
pub(super) struct PackedSubbandPlan {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) total_bitplanes: u8,
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
    pub(super) block_count: usize,
}

impl PackedSubbandPlan {
    pub(super) fn try_new(
        request: &PackedSubbandRequest<'_, '_>,
    ) -> NativeEncodePipelineResult<Self> {
        let exponent = u8::try_from(request.step_size.exponent).map_err(|_| {
            NativeEncodePipelineError::internal_invariant(
                "quantization exponent exceeds supported range",
            )
        })?;
        let total_bitplanes = request
            .settings
            .guard_bits
            .saturating_add(exponent)
            .saturating_sub(1)
            .checked_add(request.settings.roi_shift)
            .ok_or_else(|| {
                NativeEncodePipelineError::unsupported(
                    "ROI maxshift exceeds supported coded bitplane count",
                )
            })?;
        let width = request.view.width();
        let height = request.view.height();
        let num_cbs_x = width.div_ceil(request.settings.cb_width);
        let num_cbs_y = height.div_ceil(request.settings.cb_height);
        let block_count = usize::try_from(num_cbs_x)
            .map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("code-block count exceeds usize")
            })?
            .checked_mul(usize::try_from(num_cbs_y).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("code-block count exceeds usize")
            })?)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("code-block count"))?;
        Ok(Self {
            width,
            height,
            total_bitplanes,
            num_cbs_x,
            num_cbs_y,
            block_count,
        })
    }
}

pub(super) fn empty_prepared_subband(
    request: &PackedSubbandRequest<'_, '_>,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: Vec::new(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: 0,
        num_cbs_y: 0,
        code_block_width: request.settings.cb_width,
        code_block_height: request.settings.cb_height,
        width: request.view.width(),
        height: request.view.height(),
        sub_band_type: request.sub_band_type,
        total_bitplanes: 0,
        block_coding_mode: request.settings.block_coding_mode,
        ht_target_coding_passes: request.settings.ht_target_coding_passes,
    }
}

pub(super) fn prepared_subband(
    request: &PackedSubbandRequest<'_, '_>,
    plan: PackedSubbandPlan,
    code_blocks: Vec<PreparedEncodeCodeBlock>,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x: plan.num_cbs_x,
        num_cbs_y: plan.num_cbs_y,
        code_block_width: request.settings.cb_width,
        code_block_height: request.settings.cb_height,
        width: plan.width,
        height: plan.height,
        sub_band_type: request.sub_band_type,
        total_bitplanes: plan.total_bitplanes,
        block_coding_mode: request.settings.block_coding_mode,
        ht_target_coding_passes: request.settings.ht_target_coding_passes,
    }
}
