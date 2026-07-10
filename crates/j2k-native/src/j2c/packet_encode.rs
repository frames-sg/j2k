//! Tier-2 packet formation for JPEG 2000 encoding.
//!
//! Organizes encoded code-block bitstreams into packets according to the
//! LRCP progression order. Each packet contains code-block data for a
//! single (layer, resolution, component, precinct) tuple.
//!
//! A packet at resolution 0 has one subband (LL).
//! A packet at resolution r > 0 has three subbands (HL, LH, HH).
//! Each subband has its own tag trees for inclusion and zero bitplanes.
//!
//! See Annex B of ITU-T T.800.

use alloc::vec;
use alloc::vec::Vec;

use super::codestream::markers;
use super::codestream_write::BlockCodingMode;
use super::tag_tree_encode::TagTreeEncoder;
use crate::packet_math::{
    self, bits_for_ht_cleanup_length, bits_for_ht_refinement_only_length, bits_for_length,
    value_fits_in_bits,
};
use crate::writer::BitWriter;
use crate::J2kPacketizationProgressionOrder;

/// A code-block's contribution to a packet.
#[derive(Debug)]
pub(crate) struct CodeBlockPacketData {
    /// Encoded bitstream data.
    pub(crate) data: Vec<u8>,
    /// HTJ2K cleanup segment length in bytes.
    pub(crate) ht_cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes.
    pub(crate) ht_refinement_length: u32,
    /// Number of coding passes in this contribution.
    pub(crate) num_coding_passes: u8,
    /// Per-pass classic segment lengths when code-block pass termination is enabled.
    pub(crate) classic_segment_lengths: Vec<u32>,
    /// Number of zero bitplanes (only relevant for first inclusion).
    pub(crate) num_zero_bitplanes: u8,
    /// Whether this code-block has been included in a previous packet.
    pub(crate) previously_included: bool,
    /// L-block value (for segment length encoding, starts at 3).
    pub(crate) l_block: u32,
    /// Block coder used for this contribution.
    pub(crate) block_coding_mode: BlockCodingMode,
}

/// Information about a single subband's precinct.
#[derive(Debug)]
pub(crate) struct SubbandPrecinct {
    /// Code-blocks in this subband's precinct (row-major order).
    pub(crate) code_blocks: Vec<CodeBlockPacketData>,
    /// Number of code-blocks in the x direction.
    pub(crate) num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub(crate) num_cbs_y: u32,
}

/// A resolution-level packet containing one or more subband precincts.
///
/// Resolution 0 has 1 subband (LL).
/// Resolution r>0 has 3 subbands (HL, LH, HH).
#[derive(Debug)]
pub(crate) struct ResolutionPacket {
    /// Subbands in this resolution's precinct.
    pub(crate) subbands: Vec<SubbandPrecinct>,
}

/// Explicit packet output descriptor for progression-order packetization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacketDescriptor {
    pub(crate) packet_index: u32,
    pub(crate) state_index: u32,
    pub(crate) layer: u8,
    pub(crate) resolution: u32,
    pub(crate) component: u16,
    pub(crate) precinct: u64,
}

struct PacketCodeBlockState {
    previously_included: bool,
    l_block: u32,
}

struct PacketSubbandState {
    inclusion_tree: TagTreeEncoder,
    zero_bitplane_tree: TagTreeEncoder,
    code_blocks: Vec<PacketCodeBlockState>,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

struct PacketState {
    subbands: Vec<PacketSubbandState>,
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

struct FormedPacket {
    merged: Vec<u8>,
    header: Vec<u8>,
    body: Vec<u8>,
}

/// Form a packet from a resolution-level packet (possibly multiple subbands).
///
/// Returns the packet bytes (header + body).
pub(crate) fn form_packet(resolution: &mut ResolutionPacket) -> Result<Vec<u8>, &'static str> {
    for subband in &resolution.subbands {
        validate_packet_subband_layout(subband)?;
    }
    let mut header_writer = BitWriter::new();
    let mut body = Vec::new();

    // Check if any code-block across all subbands has data
    let any_data = resolution
        .subbands
        .iter()
        .any(|sb| sb.code_blocks.iter().any(|cb| cb.num_coding_passes > 0));

    if !any_data {
        // Empty packet: just write 0 bit
        header_writer.write_bit(0);
        return Ok(finish_packet(
            header_writer,
            &[],
            PacketMarkerOptions::default(),
            0,
        ));
    }

    // Non-empty packet indicator
    header_writer.write_bit(1);

    // Process each subband in order (LL for res 0; HL, LH, HH for res > 0)
    for subband in &mut resolution.subbands {
        // Create tag trees for this subband's code-block inclusion and zero bitplanes
        let mut inclusion_tree = TagTreeEncoder::new(subband.num_cbs_x, subband.num_cbs_y);
        let mut zbp_tree = TagTreeEncoder::new(subband.num_cbs_x, subband.num_cbs_y);

        // Set up tag tree values
        for (i, cb) in subband.code_blocks.iter().enumerate() {
            let index = u32::try_from(i).map_err(|_| "packet code-block index exceeds u32")?;
            let x = index % subband.num_cbs_x;
            let y = index / subband.num_cbs_x;

            let inclusion_val = if cb.num_coding_passes > 0 {
                0
            } else {
                u32::MAX / 2
            };
            inclusion_tree.set_value(x, y, inclusion_val);
            zbp_tree.set_value(x, y, u32::from(cb.num_zero_bitplanes));
        }

        // Encode each code-block's packet contribution
        for (i, cb) in subband.code_blocks.iter_mut().enumerate() {
            let index = u32::try_from(i).map_err(|_| "packet code-block index exceeds u32")?;
            let x = index % subband.num_cbs_x;
            let y = index / subband.num_cbs_x;

            if !cb.previously_included {
                // First inclusion: use tag tree
                inclusion_tree.encode(x, y, 1, &mut header_writer);

                if cb.num_coding_passes == 0 {
                    continue;
                }

                // Zero bitplanes: use tag tree
                zbp_tree.encode(
                    x,
                    y,
                    u32::from(cb.num_zero_bitplanes) + 1,
                    &mut header_writer,
                );
            } else if cb.num_coding_passes > 0 {
                header_writer.write_bit(1);
            } else {
                header_writer.write_bit(0);
                continue;
            }

            if cb.num_coding_passes == 0 {
                continue;
            }

            let data_len = u32::try_from(cb.data.len())
                .map_err(|_| "code-block payload length exceeds u32")?;
            match cb.block_coding_mode {
                BlockCodingMode::Classic => {
                    encode_num_coding_passes(cb.num_coding_passes, &mut header_writer);
                    encode_classic_segment_lengths(cb, data_len, &mut header_writer)?;
                }
                BlockCodingMode::HighThroughput => {
                    encode_num_ht_coding_passes(cb.num_coding_passes, &mut header_writer);
                    encode_ht_segment_lengths(cb, &mut header_writer)?;
                }
            }

            // Append code-block data to body
            body.extend_from_slice(&cb.data);
            cb.previously_included = true;
        }
    }

    Ok(finish_packet(
        header_writer,
        &body,
        PacketMarkerOptions::default(),
        0,
    ))
}

fn validate_packet_subband_layout(subband: &SubbandPrecinct) -> Result<(), &'static str> {
    let actual_code_blocks = u32::try_from(subband.code_blocks.len())
        .map_err(|_| "packet subband code-block count exceeds u32")?;
    if subband.num_cbs_x == 0 && subband.num_cbs_y == 0 && actual_code_blocks == 0 {
        return Ok(());
    }
    let expected_code_blocks = subband
        .num_cbs_x
        .checked_mul(subband.num_cbs_y)
        .ok_or("packet subband code-block grid exceeds u32")?;
    if subband.num_cbs_x == 0
        || subband.num_cbs_y == 0
        || expected_code_blocks != actual_code_blocks
    {
        return Err("invalid packet subband code-block layout");
    }
    Ok(())
}

fn packet_state_seed(packet: &ResolutionPacket) -> Result<PacketStateSeed, &'static str> {
    let mut subbands = Vec::with_capacity(packet.subbands.len());
    for subband in &packet.subbands {
        validate_packet_subband_layout(subband)?;
        subbands.push(PacketSubbandStateSeed {
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
            inclusion_values: vec![u32::MAX / 2; subband.code_blocks.len()],
            zero_bitplane_values: vec![0; subband.code_blocks.len()],
            l_blocks: subband
                .code_blocks
                .iter()
                .map(|code_block| code_block.l_block)
                .collect(),
            previously_included: subband
                .code_blocks
                .iter()
                .map(|code_block| code_block.previously_included)
                .collect(),
        });
    }
    Ok(PacketStateSeed { subbands })
}

struct PacketSubbandStateSeed {
    num_cbs_x: u32,
    num_cbs_y: u32,
    inclusion_values: Vec<u32>,
    zero_bitplane_values: Vec<u32>,
    l_blocks: Vec<u32>,
    previously_included: Vec<bool>,
}

struct PacketStateSeed {
    subbands: Vec<PacketSubbandStateSeed>,
}

fn validate_packet_state_layout(
    seed: &PacketStateSeed,
    packet: &ResolutionPacket,
) -> Result<(), &'static str> {
    if seed.subbands.len() != packet.subbands.len() {
        return Err("packet descriptor state layout mismatch");
    }
    for (seed_subband, packet_subband) in seed.subbands.iter().zip(&packet.subbands) {
        if seed_subband.num_cbs_x != packet_subband.num_cbs_x
            || seed_subband.num_cbs_y != packet_subband.num_cbs_y
            || seed_subband.inclusion_values.len() != packet_subband.code_blocks.len()
        {
            return Err("packet descriptor state layout mismatch");
        }
    }
    Ok(())
}

fn build_packet_states(
    packets: &[ResolutionPacket],
    descriptors: &[PacketDescriptor],
) -> Result<Vec<PacketState>, &'static str> {
    let state_count = descriptors
        .iter()
        .map(|descriptor| descriptor.state_index as usize)
        .max()
        .map_or(0usize, |max_state| max_state + 1);
    let mut seeds: Vec<Option<PacketStateSeed>> =
        core::iter::repeat_with(|| None).take(state_count).collect();

    for descriptor in descriptors {
        let packet = packets
            .get(descriptor.packet_index as usize)
            .ok_or("packet descriptor packet index out of range")?;
        let seed = &mut seeds[descriptor.state_index as usize];
        if let Some(existing) = seed {
            validate_packet_state_layout(existing, packet)?;
        } else {
            *seed = Some(packet_state_seed(packet)?);
        }

        let seed = seed
            .as_mut()
            .ok_or("packet descriptor state initialization failed")?;
        for (seed_subband, packet_subband) in seed.subbands.iter_mut().zip(&packet.subbands) {
            for (idx, code_block) in packet_subband.code_blocks.iter().enumerate() {
                if code_block.num_coding_passes == 0 {
                    continue;
                }
                let layer = u32::from(descriptor.layer);
                if layer < seed_subband.inclusion_values[idx] {
                    seed_subband.inclusion_values[idx] = layer;
                    seed_subband.zero_bitplane_values[idx] =
                        u32::from(code_block.num_zero_bitplanes);
                }
            }
        }
    }

    seeds
        .into_iter()
        .map(|seed| {
            let Some(seed) = seed else {
                return Ok(PacketState {
                    subbands: Vec::new(),
                });
            };
            let mut subbands = Vec::with_capacity(seed.subbands.len());
            for seed_subband in seed.subbands {
                let mut inclusion_tree =
                    TagTreeEncoder::new(seed_subband.num_cbs_x, seed_subband.num_cbs_y);
                let mut zero_bitplane_tree =
                    TagTreeEncoder::new(seed_subband.num_cbs_x, seed_subband.num_cbs_y);
                for idx in 0..seed_subband.inclusion_values.len() {
                    let index = u32::try_from(idx)
                        .map_err(|_| "packet state code-block index exceeds u32")?;
                    let x = index % seed_subband.num_cbs_x;
                    let y = index / seed_subband.num_cbs_x;
                    inclusion_tree.set_value(x, y, seed_subband.inclusion_values[idx]);
                    zero_bitplane_tree.set_value(x, y, seed_subband.zero_bitplane_values[idx]);
                }
                let code_blocks = seed_subband
                    .l_blocks
                    .into_iter()
                    .zip(seed_subband.previously_included)
                    .map(|(l_block, previously_included)| PacketCodeBlockState {
                        previously_included,
                        l_block,
                    })
                    .collect();
                subbands.push(PacketSubbandState {
                    inclusion_tree,
                    zero_bitplane_tree,
                    code_blocks,
                    num_cbs_x: seed_subband.num_cbs_x,
                    num_cbs_y: seed_subband.num_cbs_y,
                });
            }
            Ok(PacketState { subbands })
        })
        .collect()
}

fn form_packet_parts_with_state_and_options(
    packet_data: &ResolutionPacket,
    state: &mut PacketState,
    layer: u8,
    marker_options: PacketMarkerOptions,
    packet_sequence: u16,
) -> Result<FormedPacket, &'static str> {
    if state.subbands.len() != packet_data.subbands.len() {
        return Err("packet descriptor state layout mismatch");
    }

    let mut header_writer = BitWriter::new();
    let mut body = Vec::new();
    let any_data = packet_data
        .subbands
        .iter()
        .any(|sb| sb.code_blocks.iter().any(|cb| cb.num_coding_passes > 0));

    if !any_data {
        header_writer.write_bit(0);
        return Ok(finish_packet_parts(
            header_writer,
            &[],
            marker_options,
            packet_sequence,
        ));
    }

    header_writer.write_bit(1);
    for (packet_subband, state_subband) in packet_data.subbands.iter().zip(&mut state.subbands) {
        if packet_subband.num_cbs_x != state_subband.num_cbs_x
            || packet_subband.num_cbs_y != state_subband.num_cbs_y
            || packet_subband.code_blocks.len() != state_subband.code_blocks.len()
        {
            return Err("packet descriptor state layout mismatch");
        }

        for (idx, packet_block) in packet_subband.code_blocks.iter().enumerate() {
            let index =
                u32::try_from(idx).map_err(|_| "packet state code-block index exceeds u32")?;
            let x = index % state_subband.num_cbs_x;
            let y = index / state_subband.num_cbs_x;
            let state_block = &mut state_subband.code_blocks[idx];

            if !state_block.previously_included {
                state_subband
                    .inclusion_tree
                    .encode(x, y, u32::from(layer) + 1, &mut header_writer);
                if packet_block.num_coding_passes == 0 {
                    continue;
                }
                state_subband.zero_bitplane_tree.encode(
                    x,
                    y,
                    u32::from(packet_block.num_zero_bitplanes) + 1,
                    &mut header_writer,
                );
            } else if packet_block.num_coding_passes > 0 {
                header_writer.write_bit(1);
            } else {
                header_writer.write_bit(0);
                continue;
            }

            if packet_block.num_coding_passes == 0 {
                continue;
            }

            let data_len = u32::try_from(packet_block.data.len())
                .map_err(|_| "code-block payload length exceeds u32")?;
            match packet_block.block_coding_mode {
                BlockCodingMode::Classic => {
                    encode_num_coding_passes(packet_block.num_coding_passes, &mut header_writer);
                    encode_classic_segment_lengths_with_lblock(
                        packet_block,
                        data_len,
                        &mut state_block.l_block,
                        &mut header_writer,
                    )?;
                }
                BlockCodingMode::HighThroughput => {
                    encode_num_ht_coding_passes(packet_block.num_coding_passes, &mut header_writer);
                    encode_ht_segment_lengths_with_lblock(
                        packet_block,
                        &mut state_block.l_block,
                        &mut header_writer,
                    )?;
                }
            }
            body.extend_from_slice(&packet_block.data);
            state_block.previously_included = true;
        }
    }

    Ok(finish_packet_parts(
        header_writer,
        &body,
        marker_options,
        packet_sequence,
    ))
}

fn finish_packet(
    header_writer: BitWriter,
    body: &[u8],
    marker_options: PacketMarkerOptions,
    packet_sequence: u16,
) -> Vec<u8> {
    finish_packet_parts(header_writer, body, marker_options, packet_sequence).merged
}

fn finish_packet_parts(
    header_writer: BitWriter,
    body: &[u8],
    marker_options: PacketMarkerOptions,
    packet_sequence: u16,
) -> FormedPacket {
    let mut body_prefix = Vec::new();
    if marker_options.write_sop {
        body_prefix.push(0xFF);
        body_prefix.push(markers::SOP);
        body_prefix.extend_from_slice(&4u16.to_be_bytes());
        body_prefix.extend_from_slice(&packet_sequence.to_be_bytes());
    }

    let mut header = header_writer.finish();
    if header.last().copied() == Some(0xff) {
        header.push(0x00);
    }

    if marker_options.write_eph {
        header.push(0xFF);
        header.push(markers::EPH);
    }

    let mut merged = Vec::with_capacity(body_prefix.len() + header.len() + body.len());
    merged.extend_from_slice(&body_prefix);
    merged.extend_from_slice(&header);
    merged.extend_from_slice(body);

    let mut separated_body = body_prefix;
    separated_body.extend_from_slice(body);

    FormedPacket {
        merged,
        header,
        body: separated_body,
    }
}

/// Encode the number of coding passes using the variable-length code from Table B.4.
fn encode_num_coding_passes(num_passes: u8, writer: &mut BitWriter) {
    match num_passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3 => writer.write_bits(0b1100, 4),
        4 => writer.write_bits(0b1101, 4),
        5 => writer.write_bits(0b1110, 4),
        6..=36 => {
            writer.write_bits(0b1111, 4);
            writer.write_bits(u32::from(num_passes - 6), 5);
        }
        37..=164 => {
            writer.write_bits(0b1_1111_1111, 9);
            writer.write_bits(u32::from(num_passes - 37), 7);
        }
        _ => unreachable!("JPEG 2000 supports 1..=164 coding passes per contribution"),
    }
}

fn encode_num_ht_coding_passes(num_passes: u8, writer: &mut BitWriter) {
    match num_passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3..=5 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 3), 2);
        }
        6..=36 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(u32::from(num_passes - 6), 5);
        }
        37..=164 => {
            writer.write_bits(0b11, 2);
            writer.write_bits(0b11, 2);
            writer.write_bits(31, 5);
            writer.write_bits(u32::from(num_passes - 37), 7);
        }
        _ => unreachable!("JPEG 2000 supports 1..=164 coding passes per contribution"),
    }
}

fn encode_length(
    length: u32,
    l_block: &mut u32,
    mut num_bits: u32,
    writer: &mut BitWriter,
) -> Result<(), &'static str> {
    while !value_fits_in_bits(length, num_bits) {
        writer.write_bit(1);
        *l_block = l_block
            .checked_add(1)
            .ok_or("packet length L-block overflow")?;
        num_bits = num_bits
            .checked_add(1)
            .ok_or("packet length bit count overflow")?;
    }
    writer.write_bit(0);
    let num_bits = u8::try_from(num_bits).map_err(|_| "packet length bit count exceeds u8")?;
    writer.write_bits(length, num_bits);
    Ok(())
}

fn encode_classic_segment_lengths(
    code_block: &mut CodeBlockPacketData,
    data_len: u32,
    writer: &mut BitWriter,
) -> Result<(), &'static str> {
    let mut l_block = code_block.l_block;
    encode_classic_segment_lengths_with_lblock(code_block, data_len, &mut l_block, writer)?;
    code_block.l_block = l_block;
    Ok(())
}

fn encode_classic_segment_lengths_with_lblock(
    code_block: &CodeBlockPacketData,
    data_len: u32,
    l_block: &mut u32,
    writer: &mut BitWriter,
) -> Result<(), &'static str> {
    if *l_block > u32::from(u8::MAX) {
        return Err("classic packet L-block exceeds u8");
    }
    if code_block.classic_segment_lengths.is_empty() {
        let num_bits = bits_for_length(*l_block, code_block.num_coding_passes);
        return encode_length(data_len, l_block, num_bits, writer);
    }

    if code_block.classic_segment_lengths.len() != usize::from(code_block.num_coding_passes) {
        return Err("classic pass-terminated contribution segment count mismatch");
    }
    let segment_sum = code_block
        .classic_segment_lengths
        .iter()
        .try_fold(0u32, |acc, segment_len| acc.checked_add(*segment_len))
        .ok_or("classic packet contribution segment length overflow")?;
    if segment_sum != data_len {
        return Err("classic packet contribution segment length mismatch");
    }

    let mut required_l_block = *l_block;
    while code_block
        .classic_segment_lengths
        .iter()
        .any(|&segment_len| !value_fits_in_bits(segment_len, bits_for_length(required_l_block, 1)))
    {
        writer.write_bit(1);
        required_l_block = required_l_block
            .checked_add(1)
            .ok_or("classic packet L-block overflow")?;
    }
    writer.write_bit(0);
    *l_block = required_l_block;

    let length_bits = bits_for_length(*l_block, 1);
    let length_bits =
        u8::try_from(length_bits).map_err(|_| "classic segment length bit count exceeds u8")?;
    for &segment_len in &code_block.classic_segment_lengths {
        writer.write_bits(segment_len, length_bits);
    }

    Ok(())
}

fn encode_ht_segment_lengths(
    code_block: &mut CodeBlockPacketData,
    writer: &mut BitWriter,
) -> Result<(), &'static str> {
    let mut l_block = code_block.l_block;
    encode_ht_segment_lengths_with_lblock(code_block, &mut l_block, writer)?;
    code_block.l_block = l_block;
    Ok(())
}

fn encode_ht_segment_lengths_with_lblock(
    code_block: &CodeBlockPacketData,
    l_block: &mut u32,
    writer: &mut BitWriter,
) -> Result<(), &'static str> {
    if *l_block > u32::from(u8::MAX) {
        return Err("HT packet L-block exceeds u8");
    }
    let (cleanup_length, refinement_length) = ht_segment_lengths(code_block)?;
    if cleanup_length == 0 && refinement_length != 0 {
        let mut refinement_bits =
            bits_for_ht_refinement_only_length(*l_block, code_block.num_coding_passes);
        while !value_fits_in_bits(refinement_length, refinement_bits) {
            writer.write_bit(1);
            *l_block = l_block.checked_add(1).ok_or("HT packet L-block overflow")?;
            refinement_bits = refinement_bits
                .checked_add(1)
                .ok_or("HT refinement length bit count overflow")?;
        }
        writer.write_bit(0);
        let refinement_bits = u8::try_from(refinement_bits)
            .map_err(|_| "HT refinement length bit count exceeds u8")?;
        writer.write_bits(refinement_length, refinement_bits);
        return Ok(());
    }

    let mut cleanup_bits = bits_for_ht_cleanup_length(*l_block, code_block.num_coding_passes);
    let refinement_extra_bits = u32::from(code_block.num_coding_passes > 2);

    while !value_fits_in_bits(cleanup_length, cleanup_bits)
        || (code_block.num_coding_passes > 1
            && !value_fits_in_bits(
                refinement_length,
                l_block
                    .checked_add(refinement_extra_bits)
                    .ok_or("HT refinement length bit count overflow")?,
            ))
    {
        writer.write_bit(1);
        *l_block = l_block.checked_add(1).ok_or("HT packet L-block overflow")?;
        cleanup_bits = cleanup_bits
            .checked_add(1)
            .ok_or("HT cleanup length bit count overflow")?;
    }
    writer.write_bit(0);
    let cleanup_bits =
        u8::try_from(cleanup_bits).map_err(|_| "HT cleanup length bit count exceeds u8")?;
    writer.write_bits(cleanup_length, cleanup_bits);

    if code_block.num_coding_passes > 1 {
        let refinement_bits = l_block
            .checked_add(refinement_extra_bits)
            .ok_or("HT refinement length bit count overflow")?;
        let refinement_bits = u8::try_from(refinement_bits)
            .map_err(|_| "HT refinement length bit count exceeds u8")?;
        writer.write_bits(refinement_length, refinement_bits);
    }

    Ok(())
}

fn ht_segment_lengths(code_block: &CodeBlockPacketData) -> Result<(u32, u32), &'static str> {
    packet_math::ht_segment_lengths(
        code_block.num_coding_passes,
        code_block.data.len(),
        code_block.ht_cleanup_length,
        code_block.ht_refinement_length,
    )
}

/// Form tile bitstream from resolution packets in LRCP order.
///
/// `resolution_packets` contains one `ResolutionPacket` per resolution level:
/// - Index 0: LL band (resolution 0)
/// - Index 1..N: higher resolutions (each with HL, LH, HH subbands)
pub(crate) fn form_tile_bitstream(
    resolution_packets: &mut [ResolutionPacket],
    _num_layers: u8,
    _num_components: u16,
) -> Result<Vec<u8>, &'static str> {
    let mut tile_data = Vec::new();

    // LRCP: Layer → Resolution → Component → Position
    // For single layer, single component, this is just resolution order
    for resolution in resolution_packets.iter_mut() {
        let packet = form_packet(resolution)?;
        tile_data.extend_from_slice(&packet);
    }

    Ok(tile_data)
}

pub(crate) fn form_tile_bitstream_with_descriptors(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
) -> Result<Vec<u8>, &'static str> {
    Ok(form_tile_bitstream_with_descriptors_and_lengths(resolution_packets, descriptors)?.data)
}

pub(crate) fn form_tile_bitstream_with_descriptors_and_lengths(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
) -> Result<PacketizedTileData, &'static str> {
    form_tile_bitstream_with_descriptors_lengths_and_markers(
        resolution_packets,
        descriptors,
        PacketMarkerOptions::default(),
    )
}

pub(crate) fn form_tile_bitstream_with_descriptors_lengths_and_markers(
    resolution_packets: &mut [ResolutionPacket],
    descriptors: &[PacketDescriptor],
    marker_options: PacketMarkerOptions,
) -> Result<PacketizedTileData, &'static str> {
    if descriptors.is_empty() {
        return Ok(PacketizedTileData {
            data: Vec::new(),
            packet_lengths: Vec::new(),
            packet_headers: Vec::new(),
        });
    }

    let mut states = build_packet_states(resolution_packets, descriptors)?;
    let mut tile_data = Vec::new();
    let mut packet_lengths = Vec::with_capacity(descriptors.len());
    let mut packet_headers = if marker_options.separate_packet_headers {
        Vec::with_capacity(descriptors.len())
    } else {
        Vec::new()
    };
    for (packet_sequence, descriptor) in descriptors.iter().enumerate() {
        let packet = resolution_packets
            .get(descriptor.packet_index as usize)
            .ok_or("packet descriptor packet index out of range")?;
        let state = states
            .get_mut(descriptor.state_index as usize)
            .ok_or("packet descriptor state index out of range")?;
        let packet = form_packet_parts_with_state_and_options(
            packet,
            state,
            descriptor.layer,
            marker_options,
            u16::try_from(packet_sequence % (usize::from(u16::MAX) + 1))
                .expect("SOP packet sequence modulo 65536 fits u16"),
        )?;
        if marker_options.separate_packet_headers {
            packet_lengths
                .push(u32::try_from(packet.body.len()).map_err(|_| "packet length exceeds u32")?);
            tile_data.extend_from_slice(&packet.body);
            packet_headers.push(packet.header);
        } else {
            packet_lengths
                .push(u32::try_from(packet.merged.len()).map_err(|_| "packet length exceeds u32")?);
            tile_data.extend_from_slice(&packet.merged);
        }
    }
    Ok(PacketizedTileData {
        data: tile_data,
        packet_lengths,
        packet_headers,
    })
}

pub(crate) fn form_tile_bitstream_for_progression(
    resolution_packets: &mut [ResolutionPacket],
    num_layers: u8,
    num_components: u16,
    progression_order: J2kPacketizationProgressionOrder,
) -> Result<Vec<u8>, &'static str> {
    match progression_order {
        J2kPacketizationProgressionOrder::Lrcp
        | J2kPacketizationProgressionOrder::Rlcp
        | J2kPacketizationProgressionOrder::Rpcl
        | J2kPacketizationProgressionOrder::Pcrl
        | J2kPacketizationProgressionOrder::Cprl => {
            form_tile_bitstream(resolution_packets, num_layers, num_components)
        }
    }
}

pub(crate) fn validate_ht_segment_lengths(
    resolution_packets: &[ResolutionPacket],
) -> Result<(), &'static str> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::tag_tree::{TagNode, TagTree};
    use crate::reader::BitReader;

    fn decode_num_ht_coding_passes_for_test(data: &[u8]) -> Option<u8> {
        let mut reader = BitReader::new(data);
        decode_num_ht_coding_passes_from_reader_for_test(&mut reader)
    }

    fn decode_num_ht_coding_passes_from_reader_for_test(reader: &mut BitReader<'_>) -> Option<u8> {
        let mut num_passes = 1u32;

        if reader.read_bits_with_stuffing(1)? == 1 {
            num_passes = 2;

            if reader.read_bits_with_stuffing(1)? == 1 {
                let extension = reader.read_bits_with_stuffing(2)?;
                num_passes = 3 + extension;

                if extension == 3 {
                    let extension = reader.read_bits_with_stuffing(5)?;
                    num_passes = 6 + extension;

                    if extension == 31 {
                        num_passes = 37 + reader.read_bits_with_stuffing(7)?;
                    }
                }
            }
        }

        u8::try_from(num_passes).ok()
    }

    fn decode_num_coding_passes_for_test(data: &[u8]) -> Option<u8> {
        let mut reader = BitReader::new(data);
        decode_num_coding_passes_from_reader_for_test(&mut reader)
    }

    fn decode_num_coding_passes_from_reader_for_test(reader: &mut BitReader<'_>) -> Option<u8> {
        let passes = if reader.peak_bits_with_stuffing(9) == Some(0x1ff) {
            reader.read_bits_with_stuffing(9)?;
            reader.read_bits_with_stuffing(7)? + 37
        } else if reader.peak_bits_with_stuffing(4) == Some(0x0f) {
            reader.read_bits_with_stuffing(4)?;
            reader.read_bits_with_stuffing(5)? + 6
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1110) {
            reader.read_bits_with_stuffing(4)?;
            5
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1101) {
            reader.read_bits_with_stuffing(4)?;
            4
        } else if reader.peak_bits_with_stuffing(4) == Some(0b1100) {
            reader.read_bits_with_stuffing(4)?;
            3
        } else if reader.peak_bits_with_stuffing(2) == Some(0b10) {
            reader.read_bits_with_stuffing(2)?;
            2
        } else if reader.peak_bits_with_stuffing(1) == Some(0) {
            reader.read_bits_with_stuffing(1)?;
            1
        } else {
            return None;
        };
        u8::try_from(passes).ok()
    }

    #[test]
    fn test_empty_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: Vec::new(),
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 0,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 31,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        assert!(!packet.is_empty());
    }

    #[test]
    fn malformed_packet_layout_returns_an_error() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: Vec::new(),
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 0,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 0,
                num_cbs_y: 1,
            }],
        };

        assert_eq!(
            form_packet(&mut resolution),
            Err("invalid packet subband code-block layout")
        );
    }

    #[test]
    fn packet_length_bit_count_overflow_returns_an_error() {
        let mut writer = BitWriter::new();
        let mut l_block = u32::from(u8::MAX) + 1;
        let num_bits = l_block;
        assert_eq!(
            encode_length(0, &mut l_block, num_bits, &mut writer),
            Err("packet length bit count exceeds u8")
        );
    }

    #[test]
    fn test_non_empty_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x12, 0x34, 0x56],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 20,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        assert!(packet.len() >= 3);
    }

    #[test]
    fn packet_header_round_trips_varied_8x8_codeblock_lengths() {
        let zero_bitplanes = [
            2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 2, 2, 1,
            1, 1, 1, 2, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2,
            2, 2, 2, 1, 1, 1,
        ];
        let lengths = [
            1901, 2062, 1895, 2329, 2860, 2842, 2852, 2836, 2174, 2121, 1878, 2197, 2877, 2870,
            2854, 2862, 2097, 2143, 1906, 2059, 2724, 2879, 2860, 2847, 1928, 1967, 2105, 2318,
            2605, 2911, 2892, 2860, 1998, 1995, 2073, 2075, 2339, 2935, 2896, 2897, 1877, 1938,
            1841, 2000, 2271, 2877, 2826, 2828, 2098, 1899, 1953, 2061, 2135, 2886, 2869, 2909,
            2168, 1921, 1966, 2048, 2159, 2792, 2853, 2815,
        ];
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: zero_bitplanes
                    .iter()
                    .copied()
                    .zip(lengths.iter().copied())
                    .map(|(num_zero_bitplanes, len)| CodeBlockPacketData {
                        data: vec![0; len],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1 + 3 * (8 - num_zero_bitplanes) - 2,
                        classic_segment_lengths: Vec::new(),
                        num_zero_bitplanes,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    })
                    .collect(),
                num_cbs_x: 8,
                num_cbs_y: 8,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        let body_len: usize = lengths.iter().sum();
        let header_len = packet.len() - body_len;
        let mut reader = BitReader::new(&packet[..header_len]);
        assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

        let mut inclusion_nodes = Vec::<TagNode>::new();
        let mut inclusion_tree = TagTree::new(8, 8, &mut inclusion_nodes);
        let mut zbp_nodes = Vec::<TagNode>::new();
        let mut zbp_tree = TagTree::new(8, 8, &mut zbp_nodes);

        for (idx, (&expected_zbp, &expected_len)) in
            zero_bitplanes.iter().zip(lengths.iter()).enumerate()
        {
            let index = u32::try_from(idx).expect("8x8 test code-block index fits u32");
            let x = index % 8;
            let y = index / 8;
            let included = inclusion_tree
                .read(x, y, &mut reader, 1, &mut inclusion_nodes)
                .expect("inclusion tag")
                == 0;
            assert!(included, "inclusion at index {idx}");

            let actual_zbp = zbp_tree
                .read(x, y, &mut reader, u32::MAX, &mut zbp_nodes)
                .expect("zero bitplane tag");
            assert_eq!(actual_zbp, u32::from(expected_zbp), "zbp at index {idx}");

            let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
                .expect("number of coding passes");
            let mut l_block = 3u32;
            while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
                l_block += 1;
            }
            let length_bits = l_block + u32::from(passes).ilog2();
            let actual_len = reader
                .read_bits_with_stuffing(
                    u8::try_from(length_bits).expect("packet length bit count fits u8"),
                )
                .expect("code-block length");
            assert_eq!(
                actual_len,
                u32::try_from(expected_len).expect("test payload length fits u32"),
                "length at index {idx}"
            );
        }
    }

    #[test]
    fn packet_header_trailing_ff_stuffs_zero_before_body() {
        for len in 1..4096 {
            let mut resolution = ResolutionPacket {
                subbands: vec![SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x80; len],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        classic_segment_lengths: Vec::new(),
                        num_zero_bitplanes: 0,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                }],
            };

            let packet = form_packet(&mut resolution).expect("valid test packet");
            let header_len = packet.len() - len;
            let has_boundary_ff = packet[header_len - 1] == 0xff
                || (header_len >= 2
                    && packet[header_len - 2] == 0xff
                    && packet[header_len - 1] == 0x00);

            if !has_boundary_ff {
                continue;
            }

            let mut reader = BitReader::new(&packet);
            assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

            let mut inclusion_nodes = Vec::<TagNode>::new();
            let mut inclusion_tree = TagTree::new(1, 1, &mut inclusion_nodes);
            let included = inclusion_tree
                .read(0, 0, &mut reader, 1, &mut inclusion_nodes)
                .expect("inclusion tag")
                == 0;
            assert!(included);

            let mut zbp_nodes = Vec::<TagNode>::new();
            let mut zbp_tree = TagTree::new(1, 1, &mut zbp_nodes);
            assert_eq!(
                zbp_tree
                    .read(0, 0, &mut reader, u32::MAX, &mut zbp_nodes)
                    .expect("zero bitplane tag"),
                0
            );

            let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
                .expect("number of coding passes");
            assert_eq!(passes, 1);

            let mut l_block = 3u32;
            while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
                l_block += 1;
            }
            let actual_len = reader
                .read_bits_with_stuffing(
                    u8::try_from(l_block).expect("packet length bit count fits u8"),
                )
                .expect("code-block length");
            assert_eq!(
                actual_len,
                u32::try_from(len).expect("test payload length fits u32")
            );

            reader.align();
            let expected_body = vec![0x80; len];
            assert_eq!(reader.offset(), header_len);
            assert_eq!(reader.read_bytes(len), Some(expected_body.as_slice()));
            return;
        }

        panic!("did not find a packet header ending in 0xff");
    }

    #[test]
    fn classic_pass_terminated_lengths_share_one_lblock_increment() {
        let lengths = [1u32, 9, 17];
        let mut code_block = CodeBlockPacketData {
            data: vec![
                0;
                usize::try_from(lengths.iter().sum::<u32>())
                    .expect("test payload length fits usize")
            ],
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: u8::try_from(lengths.len()).expect("pass count fits u8"),
            classic_segment_lengths: lengths.to_vec(),
            num_zero_bitplanes: 0,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::Classic,
        };
        let mut writer = BitWriter::new();
        let data_len = u32::try_from(code_block.data.len()).expect("test payload length fits u32");

        encode_num_coding_passes(code_block.num_coding_passes, &mut writer);
        encode_classic_segment_lengths(&mut code_block, data_len, &mut writer)
            .expect("classic segment lengths encode");

        let bytes = writer.finish();
        let mut reader = BitReader::new(&bytes);
        let passes = decode_num_coding_passes_from_reader_for_test(&mut reader)
            .expect("number of coding passes");
        assert_eq!(
            passes,
            u8::try_from(lengths.len()).expect("pass count fits u8")
        );

        let mut l_block = 3u32;
        while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
            l_block += 1;
        }

        let decoded_lengths: Vec<_> = lengths
            .iter()
            .map(|_| {
                reader
                    .read_bits_with_stuffing(
                        u8::try_from(l_block).expect("packet length bit count fits u8"),
                    )
                    .expect("terminated pass segment length")
            })
            .collect();
        assert_eq!(decoded_lengths, lengths);
    }

    #[test]
    fn test_multi_subband_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x10, 0x20],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        classic_segment_lengths: Vec::new(),
                        num_zero_bitplanes: 20,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x30, 0x40],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        classic_segment_lengths: Vec::new(),
                        num_zero_bitplanes: 22,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
                SubbandPrecinct {
                    code_blocks: vec![CodeBlockPacketData {
                        data: vec![0x50],
                        ht_cleanup_length: 0,
                        ht_refinement_length: 0,
                        num_coding_passes: 1,
                        classic_segment_lengths: Vec::new(),
                        num_zero_bitplanes: 24,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: BlockCodingMode::Classic,
                    }],
                    num_cbs_x: 1,
                    num_cbs_y: 1,
                },
            ],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        // Should contain all 5 bytes of code-block data
        assert!(packet.len() >= 5);
    }

    #[test]
    fn test_encode_num_passes() {
        let mut w = BitWriter::new();
        encode_num_coding_passes(1, &mut w);
        let d = w.finish();
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_encode_num_passes_round_trip() {
        for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
            let mut w = BitWriter::new();
            encode_num_coding_passes(num_passes, &mut w);
            let data = w.finish();
            assert_eq!(decode_num_coding_passes_for_test(&data), Some(num_passes));
        }
    }

    #[test]
    fn test_encode_num_ht_passes_round_trip() {
        for num_passes in [1u8, 2, 3, 4, 5, 6, 19, 37, 38, 100, 164] {
            let mut w = BitWriter::new();
            encode_num_ht_coding_passes(num_passes, &mut w);
            let data = w.finish();
            assert_eq!(
                decode_num_ht_coding_passes_for_test(&data),
                Some(num_passes)
            );
        }
    }

    #[test]
    fn test_non_empty_ht_packet() {
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: vec![0x12, 0x34, 0x56],
                    ht_cleanup_length: 3,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 20,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::HighThroughput,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        assert!(packet.len() >= 3);
    }

    #[test]
    fn ht_packet_header_round_trips_refinement_pass_count_and_length() {
        let payload = vec![0x12, 0x34, 0x56, 0x78, 0x9a];
        let mut resolution = ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data: payload.clone(),
                    ht_cleanup_length: 3,
                    ht_refinement_length: 2,
                    num_coding_passes: 3,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 2,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::HighThroughput,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        };

        let packet = form_packet(&mut resolution).expect("valid test packet");
        let header_len = packet.len() - payload.len();
        let mut reader = BitReader::new(&packet[..header_len]);
        assert_eq!(reader.read_bits_with_stuffing(1), Some(1));

        let mut inclusion_nodes = Vec::<TagNode>::new();
        let mut inclusion_tree = TagTree::new(1, 1, &mut inclusion_nodes);
        assert_eq!(
            inclusion_tree.read(0, 0, &mut reader, 1, &mut inclusion_nodes),
            Some(0)
        );

        let mut zbp_nodes = Vec::<TagNode>::new();
        let mut zbp_tree = TagTree::new(1, 1, &mut zbp_nodes);
        assert_eq!(
            zbp_tree.read(0, 0, &mut reader, u32::MAX, &mut zbp_nodes),
            Some(2)
        );

        let passes = decode_num_ht_coding_passes_from_reader_for_test(&mut reader)
            .expect("HT coding pass count");
        assert_eq!(passes, 3);

        let mut l_block = 3u32;
        let mut length_bits = bits_for_ht_cleanup_length(l_block, passes);
        while reader.read_bits_with_stuffing(1).expect("lblock increment") == 1 {
            l_block += 1;
            length_bits += 1;
        }
        assert_eq!(
            reader.read_bits_with_stuffing(
                u8::try_from(length_bits).expect("cleanup length bit count fits u8")
            ),
            Some(3)
        );
        let refinement_bits = l_block + 1;
        assert_eq!(
            reader.read_bits_with_stuffing(
                u8::try_from(refinement_bits).expect("refinement length bit count fits u8")
            ),
            Some(2)
        );
        assert_eq!(&packet[header_len..], payload.as_slice());
    }

    #[test]
    fn ht_packet_segment_lengths_reject_overflowing_refinement_sum() {
        let code_block = CodeBlockPacketData {
            data: vec![0x12],
            ht_cleanup_length: u32::MAX,
            ht_refinement_length: 1,
            num_coding_passes: 3,
            classic_segment_lengths: Vec::new(),
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::HighThroughput,
        };

        let err = ht_segment_lengths(&code_block).expect_err("overflowing HT lengths rejected");

        assert_eq!(err, "multi-pass HTJ2K packet contribution length overflow");
    }

    fn single_block_packet(data: Vec<u8>, previously_included: bool) -> ResolutionPacket {
        ResolutionPacket {
            subbands: vec![SubbandPrecinct {
                code_blocks: vec![CodeBlockPacketData {
                    data,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    classic_segment_lengths: Vec::new(),
                    num_zero_bitplanes: 0,
                    previously_included,
                    l_block: 3,
                    block_coding_mode: BlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        }
    }

    #[test]
    fn explicit_packet_descriptors_control_packet_order() {
        let first = single_block_packet(vec![0xA0], false);
        let second = single_block_packet(vec![0xB0], false);
        let mut expected_second = single_block_packet(vec![0xB0], false);
        let mut expected_first = single_block_packet(vec![0xA0], false);
        let expected = [
            form_packet(&mut expected_second).expect("valid second test packet"),
            form_packet(&mut expected_first).expect("valid first test packet"),
        ]
        .concat();

        let actual = form_tile_bitstream_with_descriptors(
            &mut [first, second],
            &[
                PacketDescriptor {
                    packet_index: 1,
                    state_index: 1,
                    layer: 0,
                    resolution: 0,
                    component: 0,
                    precinct: 0,
                },
                PacketDescriptor {
                    packet_index: 0,
                    state_index: 0,
                    layer: 0,
                    resolution: 1,
                    component: 0,
                    precinct: 0,
                },
            ],
        )
        .expect("descriptor packetization");

        assert_eq!(actual, expected);
    }

    #[test]
    fn explicit_packet_descriptors_reuse_packet_state_across_layers() {
        let first = single_block_packet(vec![0x11], false);
        let second = single_block_packet(vec![0x22], false);

        let mut expected_first = single_block_packet(vec![0x11], false);
        let first_bytes = form_packet(&mut expected_first).expect("valid first test packet");
        let l_block_after_first = expected_first.subbands[0].code_blocks[0].l_block;
        let mut expected_second = single_block_packet(vec![0x22], true);
        expected_second.subbands[0].code_blocks[0].l_block = l_block_after_first;
        let expected = [
            first_bytes,
            form_packet(&mut expected_second).expect("valid second test packet"),
        ]
        .concat();

        let actual = form_tile_bitstream_with_descriptors(
            &mut [first, second],
            &[
                PacketDescriptor {
                    packet_index: 0,
                    state_index: 0,
                    layer: 0,
                    resolution: 0,
                    component: 0,
                    precinct: 0,
                },
                PacketDescriptor {
                    packet_index: 1,
                    state_index: 0,
                    layer: 1,
                    resolution: 0,
                    component: 0,
                    precinct: 0,
                },
            ],
        )
        .expect("stateful descriptor packetization");

        assert_eq!(actual, expected);
    }
}
