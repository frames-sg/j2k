// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::engine::abi::J2kHtUvlcEncodeTableEntry;
use crate::metal_types::{Buffer, ComputePipelineState, Device};
use j2k_metal_support::{checked_shared_buffer_with_slice, MetalPipelineLoader, MetalSupportError};
use j2k_native::{ht_uvlc_encode_table, ht_vlc_encode_table0, ht_vlc_encode_table1};

pub(crate) struct EncodeKernels {
    pub(in crate::engine) lossless_deinterleave_to_planes: ComputePipelineState,
    pub(in crate::engine) lossless_deinterleave_rct_rgb8_to_planes: ComputePipelineState,
    pub(in crate::engine) lossless_extract_coefficients: ComputePipelineState,
    pub(in crate::engine) fdwt53_horizontal: ComputePipelineState,
    pub(in crate::engine) fdwt53_vertical: ComputePipelineState,
    pub(in crate::engine) fdwt53_horizontal_batched: ComputePipelineState,
    pub(in crate::engine) fdwt53_vertical_batched: ComputePipelineState,
    pub(in crate::engine) fdwt97_lift_horizontal: ComputePipelineState,
    pub(in crate::engine) fdwt97_lift_vertical: ComputePipelineState,
    pub(in crate::engine) fdwt97_deinterleave_horizontal: ComputePipelineState,
    pub(in crate::engine) fdwt97_deinterleave_vertical: ComputePipelineState,
    pub(in crate::engine) forward_rct: ComputePipelineState,
    pub(in crate::engine) forward_ict: ComputePipelineState,
    pub(in crate::engine) encode_deinterleave_mct: ComputePipelineState,
    pub(in crate::engine) quantize_subband: ComputePipelineState,
    pub(in crate::engine) lossy_extract_quantized_coefficients: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_block: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks_32: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks_bypass_32: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks_bypass_u16_32: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks_style0: ComputePipelineState,
    pub(in crate::engine) classic_encode_code_blocks_style0_32: ComputePipelineState,
    pub(in crate::engine) ht_encode_code_block: ComputePipelineState,
    pub(in crate::engine) ht_encode_code_blocks: ComputePipelineState,
    pub(in crate::engine) packet_block_prepare_resident_classic: ComputePipelineState,
    pub(in crate::engine) packet_block_prepare_resident_ht: ComputePipelineState,
    pub(in crate::engine) packet_encode: ComputePipelineState,
    pub(in crate::engine) packet_encode_batched: ComputePipelineState,
    pub(in crate::engine) packet_encode_resident_classic_batched: ComputePipelineState,
    pub(in crate::engine) packet_payload_copy_batched: ComputePipelineState,
    pub(in crate::engine) lossless_codestream_assemble: ComputePipelineState,
    pub(in crate::engine) lossless_codestream_assemble_batched: ComputePipelineState,
    pub(in crate::engine) ht_vlc_encode_table0: Buffer,
    pub(in crate::engine) ht_vlc_encode_table1: Buffer,
    pub(in crate::engine) ht_uvlc_encode_table: Buffer,
}

impl EncodeKernels {
    pub(super) fn new(device: &Device) -> Result<Self, MetalSupportError> {
        let source = super::super::shader_source::encode_shader_source();
        let loader = MetalPipelineLoader::new(device, &source)?;
        let ht_uvlc_encode_rows = (*ht_uvlc_encode_table()).map(J2kHtUvlcEncodeTableEntry::from);
        Ok(Self {
            lossless_deinterleave_to_planes: loader
                .pipeline("j2k_lossless_deinterleave_to_planes")?,
            lossless_deinterleave_rct_rgb8_to_planes: loader
                .pipeline("j2k_lossless_deinterleave_rct_rgb8_to_planes")?,
            lossless_extract_coefficients: loader.pipeline("j2k_lossless_extract_coefficients")?,
            fdwt53_horizontal: loader.pipeline("j2k_forward_dwt53_horizontal")?,
            fdwt53_vertical: loader.pipeline("j2k_forward_dwt53_vertical")?,
            fdwt53_horizontal_batched: loader.pipeline("j2k_forward_dwt53_horizontal_batched")?,
            fdwt53_vertical_batched: loader.pipeline("j2k_forward_dwt53_vertical_batched")?,
            fdwt97_lift_horizontal: loader.pipeline("j2k_forward_dwt97_lift_horizontal")?,
            fdwt97_lift_vertical: loader.pipeline("j2k_forward_dwt97_lift_vertical")?,
            fdwt97_deinterleave_horizontal: loader
                .pipeline("j2k_forward_dwt97_deinterleave_horizontal")?,
            fdwt97_deinterleave_vertical: loader
                .pipeline("j2k_forward_dwt97_deinterleave_vertical")?,
            forward_rct: loader.pipeline("j2k_forward_rct")?,
            forward_ict: loader.pipeline("j2k_forward_ict")?,
            encode_deinterleave_mct: loader.pipeline("j2k_encode_deinterleave_mct")?,
            quantize_subband: loader.pipeline("j2k_quantize_subband")?,
            lossy_extract_quantized_coefficients: loader
                .pipeline("j2k_lossy_extract_quantized_coefficients")?,
            classic_encode_code_block: loader.pipeline("j2k_encode_classic_code_block")?,
            classic_encode_code_blocks: loader.pipeline("j2k_encode_classic_code_blocks")?,
            classic_encode_code_blocks_32: loader.pipeline("j2k_encode_classic_code_blocks_32")?,
            classic_encode_code_blocks_bypass_32: loader
                .pipeline("j2k_encode_classic_code_blocks_bypass_32")?,
            classic_encode_code_blocks_bypass_u16_32: loader
                .pipeline("j2k_encode_classic_code_blocks_bypass_u16_32")?,
            classic_encode_code_blocks_style0: loader
                .pipeline("j2k_encode_classic_code_blocks_style0")?,
            classic_encode_code_blocks_style0_32: loader
                .pipeline("j2k_encode_classic_code_blocks_style0_32")?,
            ht_encode_code_block: loader.pipeline("j2k_encode_ht_code_block")?,
            ht_encode_code_blocks: loader.pipeline("j2k_encode_ht_code_blocks")?,
            packet_block_prepare_resident_classic: loader
                .pipeline("j2k_prepare_packet_blocks_from_classic_status")?,
            packet_block_prepare_resident_ht: loader
                .pipeline("j2k_prepare_packet_blocks_from_ht_status")?,
            packet_encode: loader.pipeline("j2k_encode_packetization")?,
            packet_encode_batched: loader.pipeline("j2k_encode_packetization_batched")?,
            packet_encode_resident_classic_batched: loader
                .pipeline("j2k_encode_packetization_resident_classic_batched")?,
            packet_payload_copy_batched: loader.pipeline("j2k_copy_packet_payload_batched")?,
            lossless_codestream_assemble: loader
                .pipeline("j2k_assemble_lossless_classic_codestream")?,
            lossless_codestream_assemble_batched: loader
                .pipeline("j2k_assemble_lossless_codestream_batched")?,
            ht_vlc_encode_table0: checked_shared_buffer_with_slice(device, ht_vlc_encode_table0())?,
            ht_vlc_encode_table1: checked_shared_buffer_with_slice(device, ht_vlc_encode_table1())?,
            ht_uvlc_encode_table: checked_shared_buffer_with_slice(device, &ht_uvlc_encode_rows)?,
        })
    }
}
