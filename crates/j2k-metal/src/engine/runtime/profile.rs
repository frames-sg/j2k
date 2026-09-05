// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{ComputePipelineState, Device};
use j2k_metal_support::{MetalPipelineLoader, MetalSupportError};

pub(in crate::engine) struct ClassicTier1ProfileKernels {
    pub(in crate::engine) density: ComputePipelineState,
    pub(in crate::engine) raw_pack: ComputePipelineState,
    pub(in crate::engine) arithmetic_pack: ComputePipelineState,
    pub(in crate::engine) symbol_plan: ComputePipelineState,
    pub(in crate::engine) pass_plan: ComputePipelineState,
    pub(in crate::engine) token_emit: ComputePipelineState,
    pub(in crate::engine) split_token_emit: ComputePipelineState,
    pub(in crate::engine) split_mq_byte_token_emit: ComputePipelineState,
    pub(in crate::engine) token_pack: ComputePipelineState,
    pub(in crate::engine) split_token_pack: ComputePipelineState,
}

impl ClassicTier1ProfileKernels {
    pub(super) fn new(device: &Device) -> Result<Self, MetalSupportError> {
        let source = super::super::shader_source::profile_shader_source();
        let loader = MetalPipelineLoader::new(device, &source)?;
        Ok(Self {
            density: loader.pipeline("j2k_profile_classic_tier1_density_bypass_u16_32")?,
            raw_pack: loader.pipeline("j2k_profile_classic_tier1_raw_pack_bypass_u16_32")?,
            arithmetic_pack: loader
                .pipeline("j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32")?,
            symbol_plan: loader.pipeline("j2k_plan_classic_tier1_symbols_bypass_u16_32")?,
            pass_plan: loader.pipeline("j2k_plan_classic_tier1_passes_bypass_u16_32")?,
            token_emit: loader.pipeline("j2k_emit_classic_tier1_tokens_bypass_u16_32")?,
            split_token_emit: loader
                .pipeline("j2k_emit_classic_tier1_split_tokens_bypass_u16_32")?,
            split_mq_byte_token_emit: loader
                .pipeline("j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32")?,
            token_pack: loader.pipeline("j2k_pack_classic_tier1_tokens_bypass_u16_32")?,
            split_token_pack: loader
                .pipeline("j2k_pack_classic_tier1_split_tokens_bypass_u16_32")?,
        })
    }
}
