// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CodeBlockPacketData, PacketDescriptor, ResolutionPacket, SubbandPrecinct};
use crate::j2c::codestream_write::BlockCodingMode;
use crate::{
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationPacketDescriptor,
    J2kPacketizationResolution, J2kPacketizationSubband,
};

pub(super) trait CodeBlockView {
    fn data(&self) -> &[u8];
    fn ht_cleanup_length(&self) -> u32;
    fn ht_refinement_length(&self) -> u32;
    fn num_coding_passes(&self) -> u8;
    fn classic_segment_lengths(&self) -> &[u32];
    fn num_zero_bitplanes(&self) -> u8;
    fn previously_included(&self) -> bool;
    fn l_block(&self) -> u32;
    fn block_coding_mode(&self) -> BlockCodingMode;
}

pub(super) trait SubbandView {
    type CodeBlock: CodeBlockView;

    fn code_blocks(&self) -> &[Self::CodeBlock];
    fn num_cbs_x(&self) -> u32;
    fn num_cbs_y(&self) -> u32;
}

pub(super) trait ResolutionView {
    type Subband: SubbandView;

    fn subbands(&self) -> &[Self::Subband];
}

pub(super) trait DescriptorView {
    fn packet_index(&self) -> u32;
    fn state_index(&self) -> u32;
    fn layer(&self) -> u8;
}

impl CodeBlockView for CodeBlockPacketData {
    fn data(&self) -> &[u8] {
        &self.data
    }

    fn ht_cleanup_length(&self) -> u32 {
        self.ht_cleanup_length
    }

    fn ht_refinement_length(&self) -> u32 {
        self.ht_refinement_length
    }

    fn num_coding_passes(&self) -> u8 {
        self.num_coding_passes
    }

    fn classic_segment_lengths(&self) -> &[u32] {
        &self.classic_segment_lengths
    }

    fn num_zero_bitplanes(&self) -> u8 {
        self.num_zero_bitplanes
    }

    fn previously_included(&self) -> bool {
        self.previously_included
    }

    fn l_block(&self) -> u32 {
        self.l_block
    }

    fn block_coding_mode(&self) -> BlockCodingMode {
        self.block_coding_mode
    }
}

impl SubbandView for SubbandPrecinct {
    type CodeBlock = CodeBlockPacketData;

    fn code_blocks(&self) -> &[Self::CodeBlock] {
        &self.code_blocks
    }

    fn num_cbs_x(&self) -> u32 {
        self.num_cbs_x
    }

    fn num_cbs_y(&self) -> u32 {
        self.num_cbs_y
    }
}

impl ResolutionView for ResolutionPacket {
    type Subband = SubbandPrecinct;

    fn subbands(&self) -> &[Self::Subband] {
        &self.subbands
    }
}

impl DescriptorView for PacketDescriptor {
    fn packet_index(&self) -> u32 {
        self.packet_index
    }

    fn state_index(&self) -> u32 {
        self.state_index
    }

    fn layer(&self) -> u8 {
        self.layer
    }
}

impl CodeBlockView for J2kPacketizationCodeBlock<'_> {
    fn data(&self) -> &[u8] {
        self.data
    }

    fn ht_cleanup_length(&self) -> u32 {
        self.ht_cleanup_length
    }

    fn ht_refinement_length(&self) -> u32 {
        self.ht_refinement_length
    }

    fn num_coding_passes(&self) -> u8 {
        self.num_coding_passes
    }

    fn classic_segment_lengths(&self) -> &[u32] {
        &[]
    }

    fn num_zero_bitplanes(&self) -> u8 {
        self.num_zero_bitplanes
    }

    fn previously_included(&self) -> bool {
        self.previously_included
    }

    fn l_block(&self) -> u32 {
        self.l_block
    }

    fn block_coding_mode(&self) -> BlockCodingMode {
        match self.block_coding_mode {
            J2kPacketizationBlockCodingMode::Classic => BlockCodingMode::Classic,
            J2kPacketizationBlockCodingMode::HighThroughput => BlockCodingMode::HighThroughput,
        }
    }
}

impl<'a> SubbandView for J2kPacketizationSubband<'a> {
    type CodeBlock = J2kPacketizationCodeBlock<'a>;

    fn code_blocks(&self) -> &[Self::CodeBlock] {
        &self.code_blocks
    }

    fn num_cbs_x(&self) -> u32 {
        self.num_cbs_x
    }

    fn num_cbs_y(&self) -> u32 {
        self.num_cbs_y
    }
}

impl DescriptorView for J2kPacketizationPacketDescriptor {
    fn packet_index(&self) -> u32 {
        self.packet_index
    }

    fn state_index(&self) -> u32 {
        self.state_index
    }

    fn layer(&self) -> u8 {
        self.layer
    }
}

impl<'a> ResolutionView for J2kPacketizationResolution<'a> {
    type Subband = J2kPacketizationSubband<'a>;

    fn subbands(&self) -> &[Self::Subband] {
        &self.subbands
    }
}
