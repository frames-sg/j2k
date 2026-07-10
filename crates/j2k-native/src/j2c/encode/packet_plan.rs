// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    packet_encode, BlockCodingMode, EncodeOptions, EncodeParams, EncodeProgressionOrder,
    J2kEncodeStageAccelerator, J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
    J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor, J2kPacketizationResolution,
    J2kPacketizationSubband, PreparedCompactResolutionPacket, PreparedEncodeSubband,
    PreparedResolutionPacket, ResolutionPacket, Vec,
};

pub(super) fn count_code_blocks(
    resolution_packets: &[ResolutionPacket],
) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

pub(super) fn count_compact_code_blocks(
    resolution_packets: &[PreparedCompactResolutionPacket<'_>],
) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

pub(super) fn split_component_resolution_packets_by_precinct(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
) -> Result<Vec<Vec<PreparedResolutionPacket>>, &'static str> {
    if precinct_exponents.is_empty() {
        return Ok(component_resolution_packets);
    }

    component_resolution_packets
        .into_iter()
        .map(|component_packets| {
            let mut split_packets = Vec::new();
            for packet in component_packets {
                split_packets.extend(split_prepared_resolution_packet_by_precinct(
                    packet,
                    width,
                    height,
                    num_decomposition_levels,
                    precinct_exponents,
                )?);
            }
            Ok(split_packets)
        })
        .collect()
}

pub(super) fn split_prepared_resolution_packet_by_precinct(
    packet: PreparedResolutionPacket,
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution =
        usize::try_from(packet.resolution).map_err(|_| "resolution index exceeds usize")?;
    let &(ppx, ppy) = precinct_exponents
        .get(resolution)
        .ok_or("missing precinct exponents for resolution")?;
    let (precincts_x, precincts_y) = resolution_precinct_grid(
        width,
        height,
        num_decomposition_levels,
        packet.resolution,
        ppx,
        ppy,
    )?;
    let packet_count = (precincts_x as usize)
        .checked_mul(precincts_y as usize)
        .ok_or("precinct packet count overflow")?;
    let component = packet.component;
    let resolution = packet.resolution;
    let subbands = packet.subbands;
    let mut packets = Vec::with_capacity(packet_count);

    for precinct_y in 0..precincts_y {
        for precinct_x in 0..precincts_x {
            let precinct = u64::from(precinct_y)
                .checked_mul(u64::from(precincts_x))
                .and_then(|value| value.checked_add(u64::from(precinct_x)))
                .ok_or("precinct index overflow")?;
            let split_subbands = subbands
                .iter()
                .map(|subband| {
                    split_prepared_subband_by_precinct(
                        subband, resolution, ppx, ppy, precinct_x, precinct_y,
                    )
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            packets.push(PreparedResolutionPacket {
                component,
                resolution,
                precinct,
                subbands: split_subbands,
            });
        }
    }

    Ok(packets)
}

pub(super) fn resolution_precinct_grid(
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    resolution: u32,
    ppx: u8,
    ppy: u8,
) -> Result<(u32, u32), &'static str> {
    let resolution_shift = u32::from(num_decomposition_levels)
        .checked_sub(resolution)
        .ok_or("resolution exceeds decomposition level count")?;
    let resolution_scale = pow2_u32(resolution_shift)?;
    let resolution_width = width.div_ceil(resolution_scale);
    let resolution_height = height.div_ceil(resolution_scale);
    let precinct_width = pow2_u32(u32::from(ppx))?;
    let precinct_height = pow2_u32(u32::from(ppy))?;

    Ok((
        if resolution_width == 0 {
            0
        } else {
            resolution_width.div_ceil(precinct_width)
        },
        if resolution_height == 0 {
            0
        } else {
            resolution_height.div_ceil(precinct_height)
        },
    ))
}

pub(super) fn split_prepared_subband_by_precinct(
    subband: &PreparedEncodeSubband,
    resolution: u32,
    ppx: u8,
    ppy: u8,
    precinct_x: u32,
    precinct_y: u32,
) -> Result<PreparedEncodeSubband, &'static str> {
    if subband.code_blocks.is_empty() || subband.width == 0 || subband.height == 0 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let subband_ppx = if resolution > 0 {
        ppx.checked_sub(1)
            .ok_or("nonzero resolution precinct exponent underflow")?
    } else {
        ppx
    };
    let subband_ppy = if resolution > 0 {
        ppy.checked_sub(1)
            .ok_or("nonzero resolution precinct exponent underflow")?
    } else {
        ppy
    };
    let precinct_width = pow2_u32(u32::from(subband_ppx))?;
    let precinct_height = pow2_u32(u32::from(subband_ppy))?;
    let precinct_x0 = precinct_x
        .checked_mul(precinct_width)
        .ok_or("precinct x coordinate overflow")?;
    let precinct_y0 = precinct_y
        .checked_mul(precinct_height)
        .ok_or("precinct y coordinate overflow")?;
    let x0 = precinct_x0.min(subband.width);
    let y0 = precinct_y0.min(subband.height);
    let x1 = precinct_x0
        .checked_add(precinct_width)
        .ok_or("precinct x extent overflow")?
        .min(subband.width);
    let y1 = precinct_y0
        .checked_add(precinct_height)
        .ok_or("precinct y extent overflow")?
        .min(subband.height);

    if x0 >= x1 || y0 >= y1 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let cb_width = subband.code_block_width;
    let cb_height = subband.code_block_height;
    if cb_width == 0 || cb_height == 0 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let cb_x0 = (x0 / cb_width) * cb_width;
    let cb_y0 = (y0 / cb_height) * cb_height;
    let cb_x1 = x1.div_ceil(cb_width) * cb_width;
    let cb_y1 = y1.div_ceil(cb_height) * cb_height;
    let cbx_start = cb_x0 / cb_width;
    let cby_start = cb_y0 / cb_height;
    let cbx_end = cb_x1 / cb_width;
    let cby_end = cb_y1 / cb_height;
    let num_cbs_x = cbx_end.saturating_sub(cbx_start);
    let num_cbs_y = cby_end.saturating_sub(cby_start);
    let mut indices = Vec::with_capacity((num_cbs_x as usize).saturating_mul(num_cbs_y as usize));

    for cby in cby_start..cby_end {
        for cbx in cbx_start..cbx_end {
            let index = cby
                .checked_mul(subband.num_cbs_x)
                .and_then(|value| value.checked_add(cbx))
                .ok_or("precinct code-block index overflow")?;
            indices.push(usize::try_from(index).map_err(|_| "code-block index exceeds usize")?);
        }
    }

    let code_blocks = indices
        .iter()
        .map(|&idx| {
            subband
                .code_blocks
                .get(idx)
                .cloned()
                .ok_or("precinct code-block index out of range")
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let preencoded_ht_code_blocks = subband
        .preencoded_ht_code_blocks
        .as_ref()
        .map(|blocks| {
            indices
                .iter()
                .map(|&idx| {
                    blocks
                        .get(idx)
                        .cloned()
                        .ok_or("precinct preencoded code-block index out of range")
                })
                .collect::<Result<Vec<_>, &'static str>>()
        })
        .transpose()?;

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width: x1 - x0,
        height: y1 - y0,
        sub_band_type: subband.sub_band_type,
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: subband.block_coding_mode,
        ht_target_coding_passes: subband.ht_target_coding_passes,
    })
}

pub(super) fn empty_prepared_subband_precinct(
    subband: &PreparedEncodeSubband,
) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: Vec::new(),
        preencoded_ht_code_blocks: subband
            .preencoded_ht_code_blocks
            .as_ref()
            .map(|_| Vec::new()),
        num_cbs_x: 0,
        num_cbs_y: 0,
        code_block_width: subband.code_block_width,
        code_block_height: subband.code_block_height,
        width: 0,
        height: 0,
        sub_band_type: subband.sub_band_type,
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: subband.block_coding_mode,
        ht_target_coding_passes: subband.ht_target_coding_passes,
    }
}

pub(super) fn pow2_u32(exponent: u32) -> Result<u32, &'static str> {
    1_u32
        .checked_shl(exponent)
        .ok_or("precinct exponent exceeds u32 shift width")
}

pub(super) fn packet_descriptors_for_order(
    packets: &[PreparedResolutionPacket],
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let mut descriptors = packets
        .iter()
        .enumerate()
        .map(|(packet_index, packet)| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: packet.resolution,
                component: packet.component,
                precinct: packet.precinct,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );
    Ok(descriptors)
}

pub(super) fn packet_descriptors_for_compact_order(
    packets: &[PreparedCompactResolutionPacket<'_>],
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let mut descriptors = packets
        .iter()
        .enumerate()
        .map(|(packet_index, packet)| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: packet.resolution,
                component: packet.component,
                precinct: packet.precinct,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );
    Ok(descriptors)
}

pub(super) fn ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_resolution_packets(component_resolution_packets)
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            component_ordered_prepared_resolution_packets(component_resolution_packets)
        }
    }
}

pub(super) fn ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_compact_resolution_packets(component_resolution_packets)
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            component_ordered_prepared_compact_resolution_packets(component_resolution_packets)
        }
    }
}

pub(super) fn lrcp_ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(alloc::vec::Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("component packet resolution count mismatch")?,
            );
        }
    }

    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("component packet resolution count mismatch");
    }

    Ok(resolution_packets)
}

pub(super) fn lrcp_ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(alloc::vec::Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("component packet resolution count mismatch")?,
            );
        }
    }

    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("component packet resolution count mismatch");
    }

    Ok(resolution_packets)
}

pub(super) fn component_ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_resolution_packets.len()));

    for component in component_resolution_packets {
        if component.len() != resolution_count {
            return Err("component packet resolution count mismatch");
        }
        resolution_packets.extend(component);
    }

    Ok(resolution_packets)
}

pub(super) fn component_ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_resolution_packets.len()));

    for component in component_resolution_packets {
        if component.len() != resolution_count {
            return Err("component packet resolution count mismatch");
        }
        resolution_packets.extend(component);
    }

    Ok(resolution_packets)
}

pub(super) fn public_packetization_progression_order(
    progression_order: EncodeProgressionOrder,
) -> crate::J2kPacketizationProgressionOrder {
    progression_order.packetization_order()
}

pub(super) fn scalar_packet_descriptors(
    descriptors: &[J2kPacketizationPacketDescriptor],
) -> Vec<packet_encode::PacketDescriptor> {
    descriptors
        .iter()
        .map(|descriptor| packet_encode::PacketDescriptor {
            packet_index: descriptor.packet_index,
            state_index: descriptor.state_index,
            layer: descriptor.layer,
            resolution: descriptor.resolution,
            component: descriptor.component,
            precinct: descriptor.precinct,
        })
        .collect()
}

pub(super) fn public_packetization_resolutions(
    resolution_packets: &[ResolutionPacket],
) -> Vec<J2kPacketizationResolution<'_>> {
    resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| J2kPacketizationSubband {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| J2kPacketizationCodeBlock {
                            data: &code_block.data,
                            ht_cleanup_length: code_block.ht_cleanup_length,
                            ht_refinement_length: code_block.ht_refinement_length,
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: code_block.previously_included,
                            l_block: code_block.l_block,
                            block_coding_mode: public_packetization_block_coding_mode(
                                code_block.block_coding_mode,
                            ),
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect()
}

pub(super) fn packetize_resolution_packets_with_options(
    resolution_packets: &mut [ResolutionPacket],
    packet_descriptors: &[J2kPacketizationPacketDescriptor],
    num_layers: u8,
    num_components: u16,
    progression_order: EncodeProgressionOrder,
    marker_options: packet_encode::PacketMarkerOptions,
    allow_packetization_accelerator: bool,
    force_scalar_packetization: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<packet_encode::PacketizedTileData, &'static str> {
    let packetization_resolutions = public_packetization_resolutions(resolution_packets);
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: resolution_packets.len() as u32,
        num_layers,
        num_components,
        code_block_count: count_code_blocks(resolution_packets)?,
        progression_order: public_packetization_progression_order(progression_order),
        packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    if allow_packetization_accelerator && !force_scalar_packetization {
        if let Some(data) = accelerator.encode_packetization(packetization_job)? {
            return Ok(packet_encode::PacketizedTileData {
                data,
                packet_lengths: Vec::new(),
                packet_headers: Vec::new(),
            });
        }
    }

    let scalar_packet_descriptors = scalar_packet_descriptors(packet_descriptors);
    packet_encode::form_tile_bitstream_with_descriptors_lengths_and_markers(
        resolution_packets,
        &scalar_packet_descriptors,
        marker_options,
    )
}

pub(super) fn packetization_requires_scalar(
    params: &EncodeParams,
    tile_part_packet_limit: Option<u16>,
) -> bool {
    params.write_plt
        || params.write_plm
        || params.write_ppm
        || params.write_ppt
        || params.write_sop
        || params.write_eph
        || tile_part_packet_limit.is_some()
}

pub(super) fn public_packetization_resolutions_from_compact<'a>(
    resolution_packets: &'a [PreparedCompactResolutionPacket<'a>],
) -> Vec<J2kPacketizationResolution<'a>> {
    resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| J2kPacketizationSubband {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| J2kPacketizationCodeBlock {
                            data: code_block.data,
                            ht_cleanup_length: code_block.cleanup_length,
                            ht_refinement_length: code_block.refinement_length,
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect()
}

pub(super) fn public_packetization_block_coding_mode(
    block_coding_mode: BlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match block_coding_mode {
        BlockCodingMode::Classic => J2kPacketizationBlockCodingMode::Classic,
        BlockCodingMode::HighThroughput => J2kPacketizationBlockCodingMode::HighThroughput,
    }
}
