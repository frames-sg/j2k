// SPDX-License-Identifier: Apache-2.0

use metal::ComputePipelineState;

use crate::profile_env::classic_selective_bypass_disabled;

use super::{
    J2kClassicEncodeBatchJob, MetalRuntime, J2K_CLASSIC_ENCODE_32_MAX_HEIGHT,
    J2K_CLASSIC_ENCODE_32_MAX_WIDTH, J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES,
    J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS, J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS,
    J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS, J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT,
};

pub(super) fn classic_resident_style_flags_from_env() -> u32 {
    if classic_selective_bypass_disabled() {
        0
    } else {
        J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS
    }
}

pub(super) fn classic_cod_block_style_from_flags(flags: u32) -> u32 {
    let mut style = 0u32;
    if (flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) != 0 {
        style |= 0x01;
    }
    if (flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0 {
        style |= 0x02;
    }
    if (flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0 {
        style |= 0x04;
    }
    if (flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0 {
        style |= 0x08;
    }
    if (flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0 {
        style |= 0x20;
    }
    style
}

pub(super) fn classic_tier1_gpu_token_pack_supported(jobs: &[J2kClassicEncodeBatchJob]) -> bool {
    !jobs.is_empty()
        && classic_encode_code_blocks_pipeline_kind(jobs)
            == J2kClassicEncodePipelineKind::BypassU16_32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum J2kClassicEncodePipelineKind {
    Generic,
    Generic32,
    Bypass32,
    BypassU16_32,
    Style0,
    Style0_32,
}

pub(super) fn classic_encode_code_blocks_pipeline_kind(
    jobs: &[J2kClassicEncodeBatchJob],
) -> J2kClassicEncodePipelineKind {
    let all_32 = jobs.iter().all(|job| {
        job.width <= J2K_CLASSIC_ENCODE_32_MAX_WIDTH
            && job.height <= J2K_CLASSIC_ENCODE_32_MAX_HEIGHT
    });
    let all_style0 = jobs.iter().all(|job| job.style_flags == 0);
    let all_bypass = jobs
        .iter()
        .all(|job| job.style_flags == J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS);
    let all_u16_bitplanes = jobs.iter().all(|job| job.total_bitplanes <= 16);
    match (all_style0, all_bypass, all_32, all_u16_bitplanes) {
        (true, _, true, _) => J2kClassicEncodePipelineKind::Style0_32,
        (true, _, false, _) => J2kClassicEncodePipelineKind::Style0,
        (false, true, true, true) => J2kClassicEncodePipelineKind::BypassU16_32,
        (false, true, true, false) => J2kClassicEncodePipelineKind::Bypass32,
        (false, _, true, _) => J2kClassicEncodePipelineKind::Generic32,
        (false, _, false, _) => J2kClassicEncodePipelineKind::Generic,
    }
}

pub(super) fn classic_encode_code_blocks_pipeline<'a>(
    runtime: &'a MetalRuntime,
    jobs: &[J2kClassicEncodeBatchJob],
) -> &'a ComputePipelineState {
    match classic_encode_code_blocks_pipeline_kind(jobs) {
        J2kClassicEncodePipelineKind::Generic => &runtime.classic_encode_code_blocks,
        J2kClassicEncodePipelineKind::Generic32 => &runtime.classic_encode_code_blocks_32,
        J2kClassicEncodePipelineKind::Bypass32 => &runtime.classic_encode_code_blocks_bypass_32,
        J2kClassicEncodePipelineKind::BypassU16_32 => {
            &runtime.classic_encode_code_blocks_bypass_u16_32
        }
        J2kClassicEncodePipelineKind::Style0 => &runtime.classic_encode_code_blocks_style0,
        J2kClassicEncodePipelineKind::Style0_32 => &runtime.classic_encode_code_blocks_style0_32,
    }
}
