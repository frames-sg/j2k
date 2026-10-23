// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::ops::Range;

use super::header::{form_packet_header, ht_segment_lengths, FormedHeader};
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

/// Tiles with fewer code blocks than this form their packets on one thread.
/// A pool section costs about 100 us of wake-ups and joins on ten M4 threads,
/// against about 50 ns per code block for tag trees and again for headers; at
/// 6,144 blocks the states and the headers each take about 80 us less.
#[cfg(feature = "parallel")]
const MIN_POOL_CODE_BLOCKS: usize = 4096;

/// Tile data shorter than this is copied on one thread: 2.2 MB took 137 us
/// there and 145-161 us on ten M4 threads, 12.2 MB 740 us and about 550 us.
#[cfg(feature = "parallel")]
const MIN_POOL_COPY_BYTES: usize = 4 << 20;

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
    #[cfg(feature = "parallel")]
    let pool = descriptors.len() > 1
        && rayon::current_num_threads() > 1
        && packets
            .iter()
            .flat_map(ResolutionView::subbands)
            .map(|subband| subband.code_blocks().len())
            .sum::<usize>()
            >= MIN_POOL_CODE_BLOCKS;
    #[cfg(not(feature = "parallel"))]
    let pool = false;
    let mut states = build_packet_states(packets, descriptors, allocations, pool)?;
    let plan = plan_packet_assembly(
        packets,
        descriptors,
        &mut states,
        marker_options,
        allocations,
        pool,
    )?;
    let data = assemble_tile_data(
        packets,
        descriptors,
        &plan,
        marker_options,
        allocations,
        pool,
    )?;
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
    pool: bool,
) -> EncodeResult<PacketAssemblyPlan<'a>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    let mut headers = BudgetedHeaderStore::try_new(allocations, descriptors.len())?;
    let mut packet_lengths =
        allocations.try_vec_with_capacity(descriptors.len(), "packet length capacity exhausted")?;
    let mut tile_len = 0usize;
    #[cfg(feature = "parallel")]
    let mut formed = if pool {
        form_headers_on_pool(packets, descriptors, states, marker_options, allocations)?
    } else {
        None
    };
    #[cfg(not(feature = "parallel"))]
    let mut formed: Option<BudgetedVec<'a, Option<FormedHeader<'a>>>> = {
        let _ = pool;
        None
    };

    for (sequence, descriptor) in descriptors.iter().enumerate() {
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
        let formed_header = formed
            .as_mut()
            .and_then(|headers| headers.get_mut(sequence)?.take());
        let header = match formed_header {
            Some(header) => header,
            None => form_packet_header(
                packet,
                state,
                descriptor.layer(),
                marker_options,
                allocations,
            )?,
        };
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

/// The headers of packets that each own their state, as every packet of a
/// single-layer codestream does, formed on the pool in descriptor order;
/// `None` when a state is shared, as layers of one precinct are, or an index
/// is out of range, which the serial pass reports.
#[cfg(feature = "parallel")]
fn form_headers_on_pool<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    states: &mut [PacketState<'a>],
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
) -> EncodeResult<Option<BudgetedVec<'a, Option<FormedHeader<'a>>>>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    use rayon::prelude::*;

    let mut unclaimed =
        allocations.try_vec_with_capacity(states.len(), "packet state claim capacity exhausted")?;
    for state in states.iter_mut() {
        unclaimed.try_push(Some(state))?;
    }
    let mut jobs = allocations
        .try_vec_with_capacity(descriptors.len(), "packet header job capacity exhausted")?;
    for descriptor in descriptors {
        let packet = usize::try_from(descriptor.packet_index())
            .ok()
            .and_then(|index| packets.get(index));
        let state = usize::try_from(descriptor.state_index())
            .ok()
            .and_then(|index| unclaimed.get_mut(index)?.take());
        let (Some(packet), Some(state)) = (packet, state) else {
            return Ok(None);
        };
        jobs.try_push((packet, descriptor.layer(), state, None))?;
    }
    jobs.par_iter_mut()
        .with_max_len(1)
        .try_for_each(|(packet, layer, state, header)| {
            *header = Some(form_packet_header(
                *packet,
                state,
                *layer,
                marker_options,
                allocations,
            )?);
            EncodeResult::Ok(())
        })?;
    let mut headers = allocations.try_vec_with_capacity(
        descriptors.len(),
        "parallel packet header capacity exhausted",
    )?;
    for (_, _, _, header) in jobs.iter_mut() {
        headers.try_push(header.take())?;
    }
    Ok(Some(headers))
}

fn assemble_tile_data<'a, R, D>(
    packets: &[R],
    descriptors: &[D],
    plan: &PacketAssemblyPlan<'_>,
    marker_options: PacketMarkerOptions,
    allocations: &'a EncodeAllocationLedger,
    pool: bool,
) -> EncodeResult<BudgetedVec<'a, u8>>
where
    R: ResolutionView,
    D: DescriptorView,
{
    // This final capacity is claimed while states, tag trees, descriptors,
    // packet lengths, and all retained headers are still live.
    let mut data = allocations
        .try_vec_with_capacity(plan.tile_len, "packetized tile-data capacity exhausted")?;
    // On the pool, subband bodies are zero-filled here and copied, a subband
    // per task, once all of their offsets are known.
    #[cfg(feature = "parallel")]
    let mut deferred = if pool && plan.tile_len >= MIN_POOL_COPY_BYTES {
        let count = packets.iter().map(|packet| packet.subbands().len()).sum();
        Some(allocations.try_vec_with_capacity(count, "deferred body capacity exhausted")?)
    } else {
        None
    };
    #[cfg(not(feature = "parallel"))]
    let mut deferred = {
        let _ = pool;
        None
    };
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
        match deferred.as_mut() {
            Some(deferred) => defer_packet_body(&mut data, packet, deferred)?,
            None => append_packet_body(&mut data, packet)?,
        }
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
    #[cfg(feature = "parallel")]
    if let Some(deferred) = deferred {
        copy_bodies_on_pool(&mut data, &deferred, allocations)?;
    }
    Ok(data)
}

/// Zero-fill `packet`'s subband bodies in `data` and record their ranges for
/// [`copy_bodies_on_pool`].
fn defer_packet_body<'p, R: ResolutionView>(
    data: &mut BudgetedVec<'_, u8>,
    packet: &'p R,
    deferred: &mut BudgetedVec<'_, (Range<usize>, &'p R::Subband)>,
) -> EncodeResult<()> {
    for subband in packet.subbands() {
        let body_start = data.len();
        let body_end = subband_body(subband).try_fold(body_start, |end, block| {
            checked_add_bytes(end, block.data().len(), "deferred packet body end")
        })?;
        data.try_resize(body_end, 0)?;
        deferred.try_push((body_start..body_end, subband))?;
    }
    Ok(())
}

/// Copy each deferred subband's code blocks into its range of `data`, a
/// subband per task.
#[cfg(feature = "parallel")]
fn copy_bodies_on_pool<S: SubbandView>(
    data: &mut [u8],
    deferred: &[(Range<usize>, &S)],
    allocations: &EncodeAllocationLedger,
) -> EncodeResult<()> {
    use rayon::prelude::*;

    let mut tasks = allocations
        .try_vec_with_capacity(deferred.len(), "deferred body task capacity exhausted")?;
    let (mut rest, mut rest_start) = (data, 0);
    for (range, subband) in deferred {
        let split = range
            .start
            .checked_sub(rest_start)
            .and_then(|gap| core::mem::take(&mut rest).split_at_mut_checked(gap))
            .and_then(|(_, body)| body.split_at_mut_checked(range.len()));
        let Some((body, tail)) = split else {
            return Err(EncodeError::InternalInvariant {
                what: "deferred packet bodies overlap or exceed the tile data",
            });
        };
        tasks.try_push((body, *subband))?;
        (rest, rest_start) = (tail, range.end);
    }
    tasks
        .par_iter_mut()
        .with_max_len(1)
        .for_each(|(body, subband)| {
            let mut offset = 0;
            for block in subband_body(*subband) {
                body[offset..][..block.data().len()].copy_from_slice(block.data());
                offset += block.data().len();
            }
        });
    Ok(())
}

/// The code blocks of `subband` that contribute to its packet body.
fn subband_body<S: SubbandView>(subband: &S) -> impl Iterator<Item = &S::CodeBlock> {
    subband
        .code_blocks()
        .iter()
        .filter(|block| block.num_coding_passes() > 0)
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
        required_ht_magnitude_bound: None,
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
        for code_block in subband_body(subband) {
            data.try_extend_from_slice(code_block.data())?;
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
