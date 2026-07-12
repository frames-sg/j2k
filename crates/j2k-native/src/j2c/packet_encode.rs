//! Tier-2 packet formation for JPEG 2000 encoding.
//!
//! The facade keeps packet data contracts separate from state planning,
//! checked header coding, retained-input accounting, and final assembly.

use alloc::vec::Vec;

use super::codestream_write::BlockCodingMode;

mod accelerator_ownership;

/// A code-block's contribution to a packet.
#[derive(Debug)]
pub(crate) struct CodeBlockPacketData {
    pub(crate) data: Vec<u8>,
    pub(crate) ht_cleanup_length: u32,
    pub(crate) ht_refinement_length: u32,
    pub(crate) num_coding_passes: u8,
    pub(crate) classic_segment_lengths: Vec<u32>,
    pub(crate) num_zero_bitplanes: u8,
    pub(crate) previously_included: bool,
    pub(crate) l_block: u32,
    pub(crate) block_coding_mode: BlockCodingMode,
}

/// Code-blocks in one subband precinct, in row-major order.
#[derive(Debug)]
pub(crate) struct SubbandPrecinct {
    pub(crate) code_blocks: Vec<CodeBlockPacketData>,
    pub(crate) num_cbs_x: u32,
    pub(crate) num_cbs_y: u32,
}

/// One resolution packet containing LL or HL/LH/HH subband precincts.
#[derive(Debug)]
pub(crate) struct ResolutionPacket {
    pub(crate) subbands: Vec<SubbandPrecinct>,
}

/// Explicit output/state descriptor for one packet contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacketDescriptor {
    pub(crate) packet_index: u32,
    pub(crate) state_index: u32,
    pub(crate) layer: u8,
    pub(crate) resolution: u32,
    pub(crate) component: u16,
    pub(crate) precinct: u64,
}

pub(crate) struct PacketizedTileData {
    pub(crate) data: Vec<u8>,
    pub(crate) packet_lengths: Vec<u32>,
    pub(crate) packet_headers: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PacketMarkerOptions {
    pub(crate) write_sop: bool,
    pub(crate) write_eph: bool,
    pub(crate) separate_packet_headers: bool,
}

mod form;
mod header;
mod ownership;
mod state;
mod view;

pub(crate) use accelerator_ownership::{
    packet_metadata_retained_bytes, packetized_tile_retained_bytes,
};
pub(crate) use form::{
    form_borrowed_packetization_scalar,
    form_tile_bitstream_with_public_descriptors_and_retained_baseline,
};
pub(crate) use ownership::owned_packet_retained_bytes_for_public_descriptors;

#[cfg(test)]
use form::{
    form_packet, form_tile_bitstream, form_tile_bitstream_for_progression,
    form_tile_bitstream_with_descriptors,
};

#[cfg(test)]
use crate::packet_math::bits_for_ht_cleanup_length;
#[cfg(test)]
use crate::writer::BitWriter;
#[cfg(test)]
use header::{
    encode_classic_segment_lengths, encode_length, encode_num_coding_passes,
    encode_num_ht_coding_passes, ht_segment_lengths,
};

#[cfg(test)]
mod tests;
