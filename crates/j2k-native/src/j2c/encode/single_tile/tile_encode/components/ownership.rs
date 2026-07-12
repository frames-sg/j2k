// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared-subband owner transitions for one component packet graph.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::tier1_allocation::prepared_subbands_ownership;
use crate::j2c::encode::{
    NativeEncodePipelineResult, NativeEncodeSession, PreparedEncodeSubband,
    PreparedResolutionPacket,
};

use super::super::super::ownership::{
    prepared_packet_tree_retained_bytes, prepared_packets_retained_bytes,
};

pub(super) fn subband_prepare_retained_bytes(
    retained_base_bytes: usize,
    completed: &Vec<Vec<PreparedResolutionPacket>>,
    current: &Vec<PreparedResolutionPacket>,
    pending: &[&PreparedEncodeSubband],
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = checked_add_bytes(
        retained_base_bytes,
        prepared_packet_tree_retained_bytes(completed, completed.capacity())?,
        "prepared subband retained component packets",
    )?;
    bytes = checked_add_bytes(
        bytes,
        prepared_packets_retained_bytes(current, current.capacity())?,
        "prepared subband retained resolution packets",
    )?;
    for subband in pending {
        bytes = checked_add_bytes(
            bytes,
            prepared_subbands_ownership(core::slice::from_ref(*subband), 0)?.total()?,
            "prepared subband pending owners",
        )?;
    }
    Ok(bytes)
}

pub(super) fn try_own_packet_subbands<const N: usize>(
    subbands: [PreparedEncodeSubband; N],
    retained_base_bytes: usize,
    completed: &Vec<Vec<PreparedResolutionPacket>>,
    current: &Vec<PreparedResolutionPacket>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<PreparedEncodeSubband>> {
    let prior_bytes = subband_prepare_retained_bytes(retained_base_bytes, completed, current, &[])?;
    let source_bytes = prepared_subbands_ownership(&subbands, 0)?.total()?;
    let requested_owner_bytes =
        checked_element_bytes::<PreparedEncodeSubband>(N, "prepared packet subband owners")?;
    session.checked_phase(
        checked_add_bytes(
            prior_bytes,
            checked_add_bytes(
                source_bytes,
                requested_owner_bytes,
                "prepared packet subband ownership overlap",
            )?,
            "prepared packet subband ownership overlap",
        )?,
        "prepared packet subband ownership overlap",
    )?;
    let mut owned = Vec::new();
    owned.try_reserve_exact(N).map_err(|_| {
        host_allocation_failed("prepared packet subband owners", requested_owner_bytes)
    })?;
    let actual_owner_bytes = checked_element_bytes::<PreparedEncodeSubband>(
        owned.capacity(),
        "prepared packet subband owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            prior_bytes,
            checked_add_bytes(
                source_bytes,
                actual_owner_bytes,
                "prepared packet subband ownership overlap",
            )?,
            "prepared packet subband ownership overlap",
        )?,
        "prepared packet subband ownership overlap",
    )?;
    owned.extend(subbands);
    Ok(owned)
}
