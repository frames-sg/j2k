// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::header::{form_packet_header, ht_segment_lengths};
use super::ownership::borrowed_scalar_retained_bytes;
#[cfg(test)]
use super::ownership::owned_packet_retained_bytes;
use super::state::{build_packet_states, PacketState};
use super::view::{CodeBlockView, DescriptorView, ResolutionView, SubbandView};
use super::{PacketDescriptor, PacketMarkerOptions, PacketizedTileData, ResolutionPacket};
use crate::j2c::codestream::markers;
use crate::j2c::codestream_write::BlockCodingMode;
use crate::j2c::encode::allocation::{
    checked_add_bytes, BudgetedVec, EncodeAllocationClaim, EncodeAllocationLedger,
};
use crate::{
    EncodeError, EncodeResult, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution,
};

const SOP_BYTES: usize = 6;

struct BudgetedHeaderStore<'a> {
    headers: BudgetedVec<'a, Vec<u8>>,
    payload_claim: EncodeAllocationClaim<'a>,
}

impl<'a> BudgetedHeaderStore<'a> {
    fn try_new(allocations: &'a EncodeAllocationLedger, packet_count: usize) -> EncodeResult<Self> {
        Ok(Self {
            headers: allocations
                .try_vec_with_capacity(packet_count, "packet header owner capacity exhausted")?,
            payload_claim: allocations.claim(0, "packet header payloads")?,
        })
    }

    fn try_push(&mut self, header: BudgetedVec<'a, u8>) -> EncodeResult<()> {
        let header = header.transfer_to(&mut self.payload_claim)?;
        self.headers.try_push(header)
    }

    fn get(&self, index: usize) -> EncodeResult<&[u8]> {
        self.headers
            .get(index)
            .map(Vec::as_slice)
            .ok_or(EncodeError::InternalInvariant {
                what: "packet header index exceeded planned header store",
            })
    }

    fn into_untracked(self) -> EncodeResult<Vec<Vec<u8>>> {
        let Self {
            headers,
            payload_claim,
        } = self;
        let headers = headers.into_untracked()?;
        drop(payload_claim);
        Ok(headers)
    }
}

struct TrackedPacketizedTile<'a> {
    data: BudgetedVec<'a, u8>,
    packet_lengths: BudgetedVec<'a, u32>,
    packet_headers: Option<BudgetedHeaderStore<'a>>,
}

struct PacketAssemblyPlan<'a> {
    headers: BudgetedHeaderStore<'a>,
    packet_lengths: BudgetedVec<'a, u32>,
    tile_len: usize,
}

#[cfg(test)]
pub(crate) fn form_packet(resolution: &mut ResolutionPacket) -> EncodeResult<Vec<u8>> {
    let descriptor = [PacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    }];
    let retained =
        owned_packet_retained_bytes(core::slice::from_ref(resolution), 1, descriptor.len(), 0)?;
    Ok(form_with_retained_baseline(
        core::slice::from_ref(resolution),
        &descriptor,
        PacketMarkerOptions::default(),
        retained,
    )?
    .data)
}

#[cfg(test)]
pub(crate) fn form_tile_bitstream(
    resolution_packets: &mut [ResolutionPacket],
    num_layers: u8,
    num_components: u16,
) -> EncodeResult<Vec<u8>> {
    form_tile_bitstream_for_progression(
        resolution_packets,
        num_layers,
        num_components,
        J2kPacketizationProgressionOrder::Lrcp,
    )
}

#[cfg(test)]
pub(crate) fn form_tile_bitstream_for_progression(
    resolution_packets: &mut [ResolutionPacket],
    num_layers: u8,
    num_components: u16,
    _progression_order: J2kPacketizationProgressionOrder,
) -> EncodeResult<Vec<u8>> {
    if num_layers != 1 || num_components != 1 {
        return Err(EncodeError::InvalidInput {
            what: "implicit packet progression requires exactly one layer and one component; use explicit packet descriptors for multidimensional packetization",
        });
    }
    let retained = owned_packet_retained_bytes(resolution_packets, resolution_packets.len(), 0, 0)?;
    let allocations = EncodeAllocationLedger::new(retained)?;
    let mut descriptors = allocations.try_vec_with_capacity(
        resolution_packets.len(),
        "implicit packet descriptor capacity exhausted",
    )?;
    for packet_index in 0..resolution_packets.len() {
        let packet_index_u32 =
            u32::try_from(packet_index).map_err(|_| EncodeError::InvalidInput {
                what: "implicit packet descriptor index exceeds u32",
            })?;
        descriptors.try_push(PacketDescriptor {
            packet_index: packet_index_u32,
            state_index: packet_index_u32,
            layer: 0,
            resolution: packet_index_u32,
            component: 0,
            precinct: 0,
        })?;
    }
    let tracked = form_tracked(
        resolution_packets,
        descriptors.as_slice(),
        PacketMarkerOptions::default(),
        &allocations,
    )?;
    drop(descriptors);
    Ok(finish_tracked(tracked, &allocations)?.data)
}

#[cfg(test)]
pub(crate) fn form_tile_bitstream_with_descriptors(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
) -> EncodeResult<Vec<u8>> {
    Ok(form_tile_bitstream_with_descriptors_and_lengths(resolution_packets, descriptors)?.data)
}

#[cfg(test)]
pub(crate) fn form_tile_bitstream_with_descriptors_and_lengths(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
) -> EncodeResult<PacketizedTileData> {
    form_tile_bitstream_with_descriptors_lengths_and_markers(
        resolution_packets,
        descriptors,
        PacketMarkerOptions::default(),
    )
}

#[cfg(test)]
pub(crate) fn form_tile_bitstream_with_descriptors_lengths_and_markers(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
    marker_options: PacketMarkerOptions,
) -> EncodeResult<PacketizedTileData> {
    validate_ht_segment_lengths(resolution_packets)?;
    let retained = owned_packet_retained_bytes(
        resolution_packets,
        resolution_packets.len(),
        descriptors.len(),
        0,
    )?;
    form_with_retained_baseline(resolution_packets, descriptors, marker_options, retained)
}

pub(crate) fn form_tile_bitstream_with_public_descriptors_and_retained_baseline(
    resolution_packets: &[ResolutionPacket],
    descriptors: &[J2kPacketizationPacketDescriptor],
    marker_options: PacketMarkerOptions,
    retained_baseline_bytes: usize,
) -> EncodeResult<PacketizedTileData> {
    validate_ht_segment_lengths(resolution_packets)?;
    form_with_retained_baseline(
        resolution_packets,
        descriptors,
        marker_options,
        retained_baseline_bytes,
    )
}

pub(crate) fn form_borrowed_packetization_scalar(
    job: J2kPacketizationEncodeJob<'_>,
    additional_retained_bytes: usize,
) -> EncodeResult<Vec<u8>> {
    if usize::try_from(job.resolution_count).ok() != Some(job.resolutions.len()) {
        return Err(EncodeError::InvalidInput {
            what: "packetization resolution count does not match supplied resolutions",
        });
    }
    let actual_code_blocks = job.resolutions.iter().try_fold(0u32, |count, resolution| {
        resolution
            .subbands
            .iter()
            .try_fold(count, |count, subband| {
                let subband_count = u32::try_from(subband.code_blocks.len()).map_err(|_| {
                    EncodeError::InvalidInput {
                        what: "packetization code-block count exceeds u32",
                    }
                })?;
                count
                    .checked_add(subband_count)
                    .ok_or(EncodeError::ArithmeticOverflow {
                        what: "packetization code-block count",
                    })
            })
    })?;
    if actual_code_blocks != job.code_block_count {
        return Err(EncodeError::InvalidInput {
            what: "packetization code-block count does not match supplied resolutions",
        });
    }

    let retained = borrowed_scalar_retained_bytes(
        job.resolutions,
        job.packet_descriptors,
        additional_retained_bytes,
    )?;
    if job.packet_descriptors.is_empty() {
        return form_borrowed_implicit(
            job.resolutions,
            job.num_layers,
            job.num_components,
            job.progression_order,
            retained,
        );
    }
    Ok(form_with_retained_baseline(
        job.resolutions,
        job.packet_descriptors,
        PacketMarkerOptions::default(),
        retained,
    )?
    .data)
}

fn form_borrowed_implicit(
    resolutions: &[J2kPacketizationResolution<'_>],
    num_layers: u8,
    num_components: u16,
    _progression_order: J2kPacketizationProgressionOrder,
    retained: usize,
) -> EncodeResult<Vec<u8>> {
    if num_layers != 1 || num_components != 1 {
        return Err(EncodeError::InvalidInput {
            what: "implicit packet progression requires exactly one layer and one component; use explicit packet descriptors for multidimensional packetization",
        });
    }
    let allocations = EncodeAllocationLedger::new(retained)?;
    let mut descriptors = allocations.try_vec_with_capacity(
        resolutions.len(),
        "implicit borrowed packet descriptor capacity exhausted",
    )?;
    for packet_index in 0..resolutions.len() {
        let packet_index_u32 =
            u32::try_from(packet_index).map_err(|_| EncodeError::InvalidInput {
                what: "implicit packet descriptor index exceeds u32",
            })?;
        descriptors.try_push(PacketDescriptor {
            packet_index: packet_index_u32,
            state_index: packet_index_u32,
            layer: 0,
            resolution: packet_index_u32,
            component: 0,
            precinct: 0,
        })?;
    }
    let tracked = form_tracked(
        resolutions,
        descriptors.as_slice(),
        PacketMarkerOptions::default(),
        &allocations,
    )?;
    drop(descriptors);
    Ok(finish_tracked(tracked, &allocations)?.data)
}

fn form_with_retained_baseline<R, D>(
    packets: &[R],
    descriptors: &[D],
    marker_options: PacketMarkerOptions,
    retained_baseline_bytes: usize,
) -> EncodeResult<PacketizedTileData>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let allocations = EncodeAllocationLedger::new(retained_baseline_bytes)?;
    let tracked = form_tracked(packets, descriptors, marker_options, &allocations)?;
    finish_tracked(tracked, &allocations)
}

fn form_tracked<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<TrackedPacketizedTile<'a>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let mut states = build_packet_states(packets, descriptors, allocations)?;
    let plan = plan_packet_assembly(
        packets,
        descriptors,
        &mut states,
        marker_options,
        allocations,
    )?;
    let data = assemble_tile_data(packets, descriptors, &plan, marker_options, allocations)?;
    drop(states);

    let PacketAssemblyPlan {
        headers,
        packet_lengths,
        tile_len: _,
    } = plan;
    let packet_headers = if marker_options.separate_packet_headers {
        Some(headers)
    } else {
        drop(headers);
        None
    };
    Ok(TrackedPacketizedTile {
        data,
        packet_lengths,
        packet_headers,
    })
}

fn plan_packet_assembly<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    states: &mut BudgetedVec<'a, PacketState<'a>>,
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<PacketAssemblyPlan<'a>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let mut headers = BudgetedHeaderStore::try_new(allocations, descriptors.len())?;
    let mut packet_lengths =
        allocations.try_vec_with_capacity(descriptors.len(), "packet length capacity exhausted")?;
    let mut tile_len = 0usize;

    for descriptor in descriptors {
        let packet_index =
            usize::try_from(descriptor.packet_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor packet index out of range",
            })?;
        let packet = packets.get(packet_index).ok_or(EncodeError::InvalidInput {
            what: "packet descriptor packet index out of range",
        })?;
        let state_index =
            usize::try_from(descriptor.state_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            })?;
        let state = states
            .get_mut(state_index)
            .ok_or(EncodeError::InvalidInput {
                what: "packet descriptor state index out of range",
            })?;
        let header = form_packet_header(
            packet,
            state,
            descriptor.layer(),
            marker_options,
            allocations,
        )?;
        let sop_bytes = usize::from(marker_options.write_sop) * SOP_BYTES;
        let tile_packet_len =
            checked_add_bytes(sop_bytes, header.body_len, "packet tile-data length")?;
        let signalled_packet_len = if marker_options.separate_packet_headers {
            tile_packet_len
        } else {
            checked_add_bytes(tile_packet_len, header.bytes.len(), "inline packet length")?
        };
        packet_lengths.try_push(u32::try_from(signalled_packet_len).map_err(|_| {
            EncodeError::InvalidInput {
                what: "packet length exceeds u32",
            }
        })?)?;
        tile_len = checked_add_bytes(
            tile_len,
            if marker_options.separate_packet_headers {
                tile_packet_len
            } else {
                signalled_packet_len
            },
            "packetized tile-data length",
        )?;
        headers.try_push(header.bytes)?;
    }
    Ok(PacketAssemblyPlan {
        headers,
        packet_lengths,
        tile_len,
    })
}

fn assemble_tile_data<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    plan: &PacketAssemblyPlan<'_>,
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<BudgetedVec<'a, u8>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    // This final capacity is claimed while states, tag trees, descriptors,
    // packet lengths, and all retained headers are still live.
    let mut data = allocations
        .try_vec_with_capacity(plan.tile_len, "packetized tile-data capacity exhausted")?;
    for (packet_sequence, descriptor) in descriptors.iter().enumerate() {
        let start_len = data.len();
        if marker_options.write_sop {
            append_sop(&mut data, packet_sequence)?;
        }
        if !marker_options.separate_packet_headers {
            data.try_extend_from_slice(plan.headers.get(packet_sequence)?)?;
        }
        let packet_index =
            usize::try_from(descriptor.packet_index()).map_err(|_| EncodeError::InvalidInput {
                what: "packet descriptor packet index out of range",
            })?;
        let packet = packets.get(packet_index).ok_or(EncodeError::InvalidInput {
            what: "packet descriptor packet index out of range",
        })?;
        append_packet_body(&mut data, packet)?;
        let expected_len = usize::try_from(plan.packet_lengths[packet_sequence]).map_err(|_| {
            EncodeError::InternalInvariant {
                what: "packet length does not fit host usize",
            }
        })?;
        let actual_len =
            data.len()
                .checked_sub(start_len)
                .ok_or(EncodeError::InternalInvariant {
                    what: "packetized tile-data length regressed",
                })?;
        if actual_len != expected_len {
            return Err(EncodeError::InternalInvariant {
                what: "assembled packet length differs from checked plan",
            });
        }
    }
    if data.len() != plan.tile_len {
        return Err(EncodeError::InternalInvariant {
            what: "assembled tile length differs from checked plan",
        });
    }
    Ok(data)
}

fn finish_tracked(
    tracked: TrackedPacketizedTile<'_>,
    allocations: &EncodeAllocationLedger,
) -> EncodeResult<PacketizedTileData> {
    allocations.seal()?;
    let TrackedPacketizedTile {
        data,
        packet_lengths,
        packet_headers,
    } = tracked;
    let data = data.into_untracked()?;
    let packet_lengths = packet_lengths.into_untracked()?;
    let packet_headers = match packet_headers {
        Some(headers) => headers.into_untracked()?,
        None => Vec::new(),
    };
    allocations.finalize()?;
    Ok(PacketizedTileData {
        data,
        packet_lengths,
        packet_headers,
    })
}

fn append_sop(data: &mut BudgetedVec<'_, u8>, packet_sequence: usize) -> EncodeResult<()> {
    let sequence_modulus = usize::from(u16::MAX) + 1;
    let sequence = u16::try_from(packet_sequence % sequence_modulus).map_err(|_| {
        EncodeError::InternalInvariant {
            what: "SOP packet sequence modulo 65536 did not fit u16",
        }
    })?;
    data.try_extend_from_slice(&[
        0xFF,
        markers::SOP,
        0x00,
        0x04,
        sequence.to_be_bytes()[0],
        sequence.to_be_bytes()[1],
    ])
}

fn append_packet_body<R: ResolutionView>(
    data: &mut BudgetedVec<'_, u8>,
    packet: &R,
) -> EncodeResult<()> {
    for subband in packet.subbands() {
        for code_block in subband.code_blocks() {
            if code_block.num_coding_passes() > 0 {
                data.try_extend_from_slice(code_block.data())?;
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_ht_segment_lengths(
    resolution_packets: &[ResolutionPacket],
) -> EncodeResult<()> {
    for resolution in resolution_packets {
        for subband in &resolution.subbands {
            for code_block in &subband.code_blocks {
                if code_block.block_coding_mode == BlockCodingMode::HighThroughput {
                    ht_segment_lengths(code_block)?;
                }
            }
        }
    }
    Ok(())
}
