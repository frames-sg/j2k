// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared packet-tree accounting adapter for the single-tile coordinator.

use alloc::vec::Vec;

use super::super::super::tier1_allocation::{
    prepared_packet_tree_ownership, prepared_packets_ownership,
};
use super::super::super::PreparedResolutionPacket;
use crate::EncodeResult;

pub(in crate::j2c::encode::single_tile) fn prepared_packet_tree_retained_bytes(
    packets: &[Vec<PreparedResolutionPacket>],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    prepared_packet_tree_ownership(packets, outer_capacity)?.total()
}

pub(in crate::j2c::encode::single_tile) fn prepared_packets_retained_bytes(
    packets: &[PreparedResolutionPacket],
    outer_capacity: usize,
) -> EncodeResult<usize> {
    prepared_packets_ownership(packets, outer_capacity)?.total()
}
