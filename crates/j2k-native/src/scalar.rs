// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::error::bail;
use crate::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, apply_roi_maxshift_inverse_i64,
    checked_decode_byte_len3, j2c, profile, DecodingError, EncodedHtJ2kCodeBlock,
    EncodedJ2kCodeBlock, HtCleanupEncodeDistribution, HtCodeBlockDecodeJob,
    HtCodeBlockDecodePhaseLimit, J2kCodeBlockDecodeJob, J2kCodeBlockSegment, J2kCodeBlockStyle,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
    J2kPacketizationEncodeJob, J2kSubBandDecodeJob, J2kSubBandType, J2kTier1TokenSegment, Result,
    ValidationError, MAX_CLASSIC_DECODE_BITPLANES, MAX_DEINTERLEAVE_REFERENCE_BIT_DEPTH,
};

mod classic_decode;
use self::classic_decode::{checked_code_block_output_layout, CodeBlockOutputLayout};
pub use self::classic_decode::{
    decode_j2k_code_block_scalar, decode_j2k_code_block_scalar_profiled,
    decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_profiled, decode_j2k_sub_band_scalar,
    J2kCodeBlockDecodeProfile, J2kCodeBlockDecodeWorkspace,
};
mod encode;
pub use self::encode::{
    collect_ht_cleanup_encode_distribution, encode_ht_code_block_scalar,
    encode_ht_code_block_scalar_with_passes, encode_j2k_code_block_scalar_with_style,
    encode_j2k_packetization_scalar, forward_dwt53_reference, forward_dwt97_reference,
    forward_ict_reference, forward_rct_reference, pack_j2k_code_block_scalar_from_tier1_tokens,
    quantize_reversible_reference, quantize_subband_reference, try_deinterleave_reference,
};
mod ht_decode;
pub use self::ht_decode::{
    decode_ht_code_block_scalar, decode_ht_code_block_scalar_until_phase,
    decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_profiled, HtCodeBlockDecodeProfile,
    HtCodeBlockDecodeWorkspace,
};

fn internal_j2k_sub_band_type(sub_band_type: J2kSubBandType) -> j2c::build::SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => j2c::build::SubBandType::LowLow,
        J2kSubBandType::HighLow => j2c::build::SubBandType::HighLow,
        J2kSubBandType::LowHigh => j2c::build::SubBandType::LowHigh,
        J2kSubBandType::HighHigh => j2c::build::SubBandType::HighHigh,
    }
}

fn internal_j2k_code_block_style(style: J2kCodeBlockStyle) -> j2c::codestream::CodeBlockStyle {
    j2c::codestream::CodeBlockStyle {
        selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
        reset_context_probabilities: style.reset_context_probabilities,
        termination_on_each_pass: style.termination_on_each_pass,
        vertically_causal_context: style.vertically_causal_context,
        segmentation_symbols: style.segmentation_symbols,
        high_throughput_block_coding: false,
    }
}
