// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packetization input values and jobs.

use alloc::vec::Vec;

use super::J2kPacketizationProgressionOrder;

/// One encoded Tier-1 code-block contribution to a packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationCodeBlock<'a> {
    /// Encoded Tier-1 bitstream bytes for this packet contribution.
    pub data: &'a [u8],
    /// HTJ2K cleanup segment length in bytes when using high-throughput coding.
    pub ht_cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes when using high-throughput coding.
    pub ht_refinement_length: u32,
    /// Number of coding passes in this contribution.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
    /// Whether this code block was included in a previous packet.
    pub previously_included: bool,
    /// L-block value used for segment length coding.
    pub l_block: u32,
    /// Block coder used for this contribution.
    pub block_coding_mode: J2kPacketizationBlockCodingMode,
}

/// Packetization block coding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationBlockCodingMode {
    /// Classic JPEG 2000 Part 1 EBCOT block coding.
    Classic,
    /// High-throughput JPEG 2000 Part 15 block coding.
    HighThroughput,
}

/// One packetization subband precinct.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kPacketizationSubband<'a> {
    /// Code-block contributions in row-major order.
    pub code_blocks: Vec<J2kPacketizationCodeBlock<'a>>,
    /// Number of code blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code blocks in the y direction.
    pub num_cbs_y: u32,
}

/// One packetization resolution packet.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kPacketizationResolution<'a> {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<J2kPacketizationSubband<'a>>,
}

/// Explicit packet descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationPacketDescriptor {
    /// Index into the packet contribution array.
    pub packet_index: u32,
    /// Persistent packet-state index for repeated layer/precinct packets.
    pub state_index: u32,
    /// Quality layer for inclusion tag-tree thresholds.
    pub layer: u8,
    /// Resolution index in the output progression.
    pub resolution: u32,
    /// Component index in the output progression.
    pub component: u16,
    /// Precinct index in the output progression.
    pub precinct: u64,
}

/// Packetization encode job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kPacketizationEncodeJob<'a> {
    /// Number of resolution packets prepared for packetization.
    pub resolution_count: u32,
    /// Number of layers to write.
    pub num_layers: u8,
    /// Number of image components.
    pub num_components: u16,
    /// Total number of code-block contributions.
    pub code_block_count: u32,
    /// Packet progression order to emit.
    pub progression_order: J2kPacketizationProgressionOrder,
    /// Explicit packet descriptors in output progression order.
    pub packet_descriptors: &'a [J2kPacketizationPacketDescriptor],
    /// Packet payload prepared by Tier-1, in LRCP packet order.
    pub resolutions: &'a [J2kPacketizationResolution<'a>],
}

crate::move_only::assert_move_only!(
    J2kPacketizationSubband<'static>,
    J2kPacketizationResolution<'static>,
);
