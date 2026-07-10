// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget, bitplane_encode,
    classic_layer_contributions, classic_multilayer_code_block_style,
    classic_unbudgeted_segment_layers, encode_all_ht_code_blocks, encode_prepared_subbands,
    enforce_classic_segment_layer_monotonicity, enforce_ht_segment_layer_monotonicity,
    ht_layer_contributions, ht_segment_count, ht_segment_rate, ht_unbudgeted_segment_layers, vec,
    BlockCodingMode, ClassicSegmentAssignmentCandidate, ClassicSegmentLocation,
    EncodeProgressionOrder, HtSegmentAssignmentCandidate, HtSegmentLocation,
    J2kEncodeStageAccelerator, J2kPacketizationPacketDescriptor, LayeredPreparedBlock,
    LayeredPreparedPacket, LayeredPreparedSubband, PreparedResolutionPacket, ResolutionPacket,
    SubbandPrecinct, Vec,
};

pub(super) fn encode_prepared_resolution_packets(
    prepared_packets: Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<ResolutionPacket>, &'static str> {
    let subband_counts: Vec<_> = prepared_packets
        .iter()
        .map(|packet| packet.subbands.len())
        .collect();
    let prepared_subbands: Vec<_> = prepared_packets
        .into_iter()
        .flat_map(|packet| packet.subbands)
        .collect();
    let mut encoded_subbands =
        encode_prepared_subbands(prepared_subbands, accelerator)?.into_iter();

    subband_counts
        .into_iter()
        .map(|subband_count| {
            let mut subbands = Vec::with_capacity(subband_count);
            for _ in 0..subband_count {
                subbands.push(
                    encoded_subbands
                        .next()
                        .ok_or("encoded subband count mismatch")?,
                );
            }
            Ok(ResolutionPacket { subbands })
        })
        .collect()
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn encode_prepared_resolution_packets_layered(
    prepared_packets: Vec<PreparedResolutionPacket>,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    quality_layer_byte_targets: &[u64],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<(Vec<ResolutionPacket>, Vec<J2kPacketizationPacketDescriptor>), &'static str> {
    let layer_count = usize::from(num_layers);
    let mut layered_packets = Vec::with_capacity(prepared_packets.len());
    let mut classic_candidates = Vec::new();
    let mut classic_locations = Vec::new();
    let mut classic_block_index = 0usize;
    let mut ht_candidates = Vec::new();
    let mut ht_locations = Vec::new();
    let mut ht_block_index = 0usize;

    for prepared_packet in prepared_packets {
        let packet_idx = layered_packets.len();
        let mut layered_packet = LayeredPreparedPacket {
            component: prepared_packet.component,
            resolution: prepared_packet.resolution,
            precinct: prepared_packet.precinct,
            subbands: Vec::with_capacity(prepared_packet.subbands.len()),
        };

        for subband in prepared_packet.subbands {
            let subband_idx = layered_packet.subbands.len();
            let mut layered_subband = LayeredPreparedSubband {
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
                blocks: Vec::with_capacity(subband.code_blocks.len()),
            };

            match subband.block_coding_mode {
                BlockCodingMode::Classic => {
                    for block in subband.code_blocks {
                        let block_idx = layered_subband.blocks.len();
                        let encoded = bitplane_encode::encode_code_block_segments_with_style_i64(
                            &block.coefficients,
                            block.width,
                            block.height,
                            subband.sub_band_type,
                            subband.total_bitplanes,
                            &classic_multilayer_code_block_style(),
                        );
                        let segment_layers = if quality_layer_byte_targets.is_empty() {
                            classic_unbudgeted_segment_layers(&encoded, num_layers)?
                        } else {
                            for (segment_idx, segment) in encoded.segments.iter().enumerate() {
                                classic_candidates.push(ClassicSegmentAssignmentCandidate {
                                    block_index: classic_block_index,
                                    segment_index: segment_idx,
                                    rate: u64::from(segment.data_length),
                                    distortion_delta: segment.distortion_delta,
                                });
                                classic_locations.push(ClassicSegmentLocation {
                                    packet_idx,
                                    subband_idx,
                                    block_idx,
                                    segment_idx,
                                });
                            }
                            vec![layer_count.saturating_sub(1); encoded.segments.len()]
                        };
                        layered_subband.blocks.push(LayeredPreparedBlock::Classic {
                            encoded,
                            segment_layers,
                        });
                        classic_block_index = classic_block_index
                            .checked_add(1)
                            .ok_or("classic PCRD block index overflow")?;
                    }
                }
                BlockCodingMode::HighThroughput => {
                    let encoded_blocks =
                        encode_all_ht_code_blocks(core::slice::from_ref(&subband), accelerator)?;
                    let block_count = encoded_blocks.len();
                    for (block_idx, encoded) in encoded_blocks.into_iter().enumerate() {
                        let segment_layers = if quality_layer_byte_targets.is_empty() {
                            ht_unbudgeted_segment_layers(
                                &encoded,
                                num_layers,
                                block_idx,
                                block_count,
                            )?
                        } else {
                            let segment_count = ht_segment_count(&encoded);
                            let mut segment_layers = Vec::with_capacity(segment_count);
                            for segment_idx in 0..segment_count {
                                ht_candidates.push(HtSegmentAssignmentCandidate {
                                    block_index: ht_block_index,
                                    segment_index: segment_idx,
                                    rate: ht_segment_rate(&encoded, segment_idx)?,
                                });
                                ht_locations.push(HtSegmentLocation {
                                    packet_idx,
                                    subband_idx,
                                    block_idx: layered_subband.blocks.len(),
                                    segment_idx,
                                });
                                segment_layers.push(layer_count.saturating_sub(1));
                            }
                            segment_layers
                        };
                        layered_subband
                            .blocks
                            .push(LayeredPreparedBlock::HighThroughput {
                                encoded,
                                segment_layers,
                            });
                        ht_block_index = ht_block_index
                            .checked_add(1)
                            .ok_or("HTJ2K segment block index overflow")?;
                    }
                }
            }

            layered_packet.subbands.push(layered_subband);
        }

        layered_packets.push(layered_packet);
    }

    if !quality_layer_byte_targets.is_empty() {
        let assignments = assign_classic_segment_layers_by_slope(
            &classic_candidates,
            layer_count,
            quality_layer_byte_targets,
        )?;
        for (assignment_idx, layer) in assignments.into_iter().enumerate() {
            let location = classic_locations
                .get(assignment_idx)
                .ok_or("classic PCRD assignment location mismatch")?;
            let block = layered_packets
                .get_mut(location.packet_idx)
                .ok_or("classic PCRD packet index mismatch")?
                .subbands
                .get_mut(location.subband_idx)
                .ok_or("classic PCRD subband index mismatch")?
                .blocks
                .get_mut(location.block_idx)
                .ok_or("classic PCRD block index mismatch")?;
            let LayeredPreparedBlock::Classic { segment_layers, .. } = block else {
                return Err("classic PCRD assignment referenced HT block");
            };
            let segment_layer = segment_layers
                .get_mut(location.segment_idx)
                .ok_or("classic PCRD segment index mismatch")?;
            *segment_layer = layer;
        }
        enforce_classic_segment_layer_monotonicity(&mut layered_packets);
    }
    if !quality_layer_byte_targets.is_empty() {
        let assignments = assign_ht_segment_layers_by_budget(
            &ht_candidates,
            layer_count,
            quality_layer_byte_targets,
        )?;
        for (assignment_idx, layer) in assignments.into_iter().enumerate() {
            let location = ht_locations
                .get(assignment_idx)
                .ok_or("HTJ2K segment assignment location mismatch")?;
            let block = layered_packets
                .get_mut(location.packet_idx)
                .ok_or("HTJ2K packet index mismatch")?
                .subbands
                .get_mut(location.subband_idx)
                .ok_or("HTJ2K subband index mismatch")?
                .blocks
                .get_mut(location.block_idx)
                .ok_or("HTJ2K block index mismatch")?;
            let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block else {
                return Err("HTJ2K segment assignment referenced classic block");
            };
            let segment_layer = segment_layers
                .get_mut(location.segment_idx)
                .ok_or("HTJ2K segment index mismatch")?;
            *segment_layer = layer;
        }
        enforce_ht_segment_layer_monotonicity(&mut layered_packets);
    }

    let mut resolution_packets = Vec::with_capacity(layered_packets.len() * layer_count);
    let mut descriptors = Vec::with_capacity(layered_packets.len() * layer_count);
    for (state_index, layered_packet) in layered_packets.into_iter().enumerate() {
        let mut layer_packets: Vec<_> = (0..layer_count)
            .map(|_| ResolutionPacket {
                subbands: Vec::with_capacity(layered_packet.subbands.len()),
            })
            .collect();

        for subband in layered_packet.subbands {
            let mut layer_subbands: Vec<_> = (0..layer_count)
                .map(|_| SubbandPrecinct {
                    code_blocks: Vec::with_capacity(subband.blocks.len()),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect();

            for block in subband.blocks {
                let contributions = match block {
                    LayeredPreparedBlock::Classic {
                        encoded,
                        segment_layers,
                    } => classic_layer_contributions(&encoded, num_layers, &segment_layers)?,
                    LayeredPreparedBlock::HighThroughput {
                        encoded,
                        segment_layers,
                    } => ht_layer_contributions(&encoded, num_layers, &segment_layers)?,
                };
                for (layer_idx, contribution) in contributions.into_iter().enumerate() {
                    layer_subbands[layer_idx].code_blocks.push(contribution);
                }
            }

            for (layer_packet, layer_subband) in layer_packets.iter_mut().zip(layer_subbands) {
                layer_packet.subbands.push(layer_subband);
            }
        }

        let state_index =
            u32::try_from(state_index).map_err(|_| "packet descriptor state index exceeds u32")?;
        for (layer_idx, layer_packet) in layer_packets.into_iter().enumerate() {
            let packet_index = u32::try_from(resolution_packets.len())
                .map_err(|_| "packet descriptor index exceeds u32")?;
            resolution_packets.push(layer_packet);
            descriptors.push(J2kPacketizationPacketDescriptor {
                packet_index,
                state_index,
                layer: u8::try_from(layer_idx).map_err(|_| "quality layer index exceeds u8")?,
                resolution: layered_packet.resolution,
                component: layered_packet.component,
                precinct: layered_packet.precinct,
            });
        }
    }

    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );

    Ok((resolution_packets, descriptors))
}
