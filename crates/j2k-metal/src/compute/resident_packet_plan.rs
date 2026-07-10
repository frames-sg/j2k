// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use j2k_native::J2kPacketizationPacketDescriptor;
use metal::{Buffer, CommandBuffer};

use super::{
    encode_capacity::{
        codestream_progression_order_code, lossless_codestream_assembly_capacity,
        lossless_codestream_payload_offset, packet_tree_node_count,
    },
    J2kBatchedCodestreamAssemblyJob, J2kBatchedPacketEncodeJob, J2kLosslessCodestreamAssemblyJob,
    J2kLosslessDeviceCodeBlock, J2kPacketDescriptor, J2kPacketResolution, J2kPacketStateBlock,
    J2kPacketSubband, J2kPreparedLosslessDeviceCodeBlocks, J2kResidentBatchEncodeItem,
    J2kResidentPacketBlock, J2kResidentPacketizationResolution,
};
use crate::Error;

pub(super) struct PreparedLosslessBatchTile {
    pub(super) coefficient_buffer: Buffer,
    pub(super) coefficient_byte_offset: usize,
    pub(super) coefficient_byte_len: usize,
    pub(super) coefficient_buffer_is_batch_shared: bool,
    pub(super) code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
    pub(super) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(super) prepare_command_buffer: CommandBuffer,
    pub(super) prepare_deinterleave_rct_command_buffer: Option<CommandBuffer>,
    pub(super) prepare_dwt53_command_buffer: Option<CommandBuffer>,
    pub(super) prepare_dwt53_vertical_command_buffers: Vec<CommandBuffer>,
    pub(super) prepare_dwt53_horizontal_command_buffers: Vec<CommandBuffer>,
    pub(super) prepare_coefficient_extract_command_buffer: Option<CommandBuffer>,
    pub(super) deinterleave_status_buffer: Buffer,
    pub(super) plane_buffers: Vec<Buffer>,
    pub(super) scratch_buffers: Vec<Buffer>,
    pub(super) coefficient_job_buffer: Buffer,
    pub(super) resolution_count: u32,
    pub(super) num_layers: u8,
    pub(super) component_count: u8,
    pub(super) code_block_count: u32,
    pub(super) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(super) resolutions: Vec<J2kResidentPacketizationResolution>,
    pub(super) codestream: J2kLosslessCodestreamAssemblyJob,
}

/// Moves resident batch encode items into the family-neutral per-tile form
/// shared by the classic and HT batch drivers.
pub(super) fn prepared_lossless_batch_tiles(
    items: Vec<J2kResidentBatchEncodeItem>,
) -> Vec<PreparedLosslessBatchTile> {
    let mut prepared_tiles = Vec::with_capacity(items.len());
    for item in items {
        let J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            _prepare_command_buffer: prepare_command_buffer,
            _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
            _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
            _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
            _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
            _prepare_coefficient_extract_command_buffer: prepare_coefficient_extract_command_buffer,
            _deinterleave_status_buffer: deinterleave_status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        } = item.prepared;
        prepared_tiles.push(PreparedLosslessBatchTile {
            coefficient_buffer,
            coefficient_byte_offset,
            coefficient_byte_len,
            coefficient_buffer_is_batch_shared,
            code_blocks,
            recyclable_private_buffers,
            prepare_command_buffer,
            prepare_deinterleave_rct_command_buffer,
            prepare_dwt53_command_buffer,
            prepare_dwt53_vertical_command_buffers,
            prepare_dwt53_horizontal_command_buffers,
            prepare_coefficient_extract_command_buffer,
            deinterleave_status_buffer,
            plane_buffers,
            scratch_buffers,
            coefficient_job_buffer,
            resolution_count: item.resolution_count,
            num_layers: item.num_layers,
            component_count: item.component_count,
            code_block_count: item.code_block_count,
            packet_descriptors: item.packet_descriptors,
            resolutions: item.resolutions,
            codestream: item.codestream,
        });
    }
    prepared_tiles
}

/// Per-family constants for the shared resident batch packet planner; values
/// reproduce each family's original literals so diagnostics and GPU job
/// fields stay byte-identical.
#[derive(Clone, Copy)]
pub(super) struct ResidentBatchPacketPlanParams {
    pub(super) family_name: &'static str,
    pub(super) block_coding_mode: u32,
    pub(super) high_throughput: u32,
    pub(super) code_block_style: u32,
}

pub(super) struct ResidentBatchPacketPlan {
    pub(super) packet_resolutions: Vec<J2kPacketResolution>,
    pub(super) packet_subbands: Vec<J2kPacketSubband>,
    pub(super) resident_blocks: Vec<J2kResidentPacketBlock>,
    pub(super) packet_descriptors: Vec<J2kPacketDescriptor>,
    pub(super) state_blocks: Vec<J2kPacketStateBlock>,
    pub(super) packet_jobs: Vec<J2kBatchedPacketEncodeJob>,
    pub(super) assembly_jobs: Vec<J2kBatchedCodestreamAssemblyJob>,
    pub(super) packet_output_capacity_total: usize,
    pub(super) packet_payload_copy_job_capacity_total: usize,
    pub(super) max_payload_copy_jobs_per_tile: usize,
    pub(super) header_capacity_total: usize,
    pub(super) scratch_words_total: usize,
    pub(super) codestream_capacity_total: usize,
    pub(super) codestream_offsets: Vec<usize>,
    pub(super) codestream_capacities: Vec<usize>,
}

/// Builds the per-tile packet/assembly plan shared by the classic and HT
/// resident batch encode drivers (the packet-plan stage of both was
/// token-identical apart from the values now carried in `params` and the
/// per-family packet output capacity rule).
#[expect(
    clippy::too_many_lines,
    reason = "single-pass packet planning keeps descriptor offsets and capacities consistent"
)]
pub(super) fn build_resident_batch_packet_plan(
    prepared_tiles: &[PreparedLosslessBatchTile],
    tile_tier1_job_bases: &[usize],
    params: ResidentBatchPacketPlanParams,
    tile_packet_output_capacity: impl Fn(
        usize,
        &PreparedLosslessBatchTile,
        usize,
    ) -> Result<usize, Error>,
) -> Result<ResidentBatchPacketPlan, Error> {
    let batch_err = |suffix: &str| Error::MetalKernel {
        message: format!("{} Metal batch {}", params.family_name, suffix),
    };
    let mut packet_resolutions = Vec::<J2kPacketResolution>::new();
    let mut packet_subbands = Vec::<J2kPacketSubband>::new();
    let mut resident_blocks = Vec::<J2kResidentPacketBlock>::new();
    let mut packet_descriptors = Vec::<J2kPacketDescriptor>::new();
    let mut state_blocks = Vec::<J2kPacketStateBlock>::new();
    let mut packet_jobs = Vec::<J2kBatchedPacketEncodeJob>::with_capacity(prepared_tiles.len());
    let mut assembly_jobs =
        Vec::<J2kBatchedCodestreamAssemblyJob>::with_capacity(prepared_tiles.len());
    let mut packet_output_capacity_total = 0usize;
    let mut packet_payload_copy_job_capacity_total = 0usize;
    let mut max_payload_copy_jobs_per_tile = 0usize;
    let mut header_capacity_total = 0usize;
    let mut scratch_words_total = 0usize;
    let mut codestream_capacity_total = 0usize;
    let mut codestream_offsets = Vec::<usize>::with_capacity(prepared_tiles.len());
    let mut codestream_capacities = Vec::<usize>::with_capacity(prepared_tiles.len());

    for (tile_index, tile) in prepared_tiles.iter().enumerate() {
        let local_resolution_offset = packet_resolutions.len();
        let local_subband_offset = packet_subbands.len();
        let local_block_offset = resident_blocks.len();
        let local_descriptor_offset = packet_descriptors.len();
        let local_state_block_offset = state_blocks.len();
        let tier1_job_base = tile_tier1_job_bases[tile_index];
        let mut max_tree_nodes = 1usize;
        let mut local_subband_count = 0usize;
        let mut local_resident_block_count = 0usize;
        let mut local_payload_copy_job_capacity = 0usize;

        for resolution in &tile.resolutions {
            let subband_offset = u32::try_from(local_subband_count)
                .map_err(|_| batch_err("packet subband offset exceeds u32"))?;
            for subband in &resolution.subbands {
                let block_offset = u32::try_from(local_resident_block_count)
                    .map_err(|_| batch_err("packet block offset exceeds u32"))?;
                max_tree_nodes = max_tree_nodes.max(packet_tree_node_count(
                    subband.num_cbs_x,
                    subband.num_cbs_y,
                )?);
                let code_block_start = usize::try_from(subband.code_block_start)
                    .map_err(|_| batch_err("packet code-block offset exceeds usize"))?;
                let code_block_count = usize::try_from(subband.code_block_count)
                    .map_err(|_| batch_err("packet code-block count exceeds usize"))?;
                let code_block_end = code_block_start
                    .checked_add(code_block_count)
                    .ok_or_else(|| batch_err("packet code-block range overflow"))?;
                if code_block_end > tile.code_blocks.len() {
                    return Err(batch_err("packet code-block range out of bounds"));
                }
                for tier1_job_index in code_block_start..code_block_end {
                    resident_blocks.push(J2kResidentPacketBlock {
                        tier1_job_index: u32::try_from(
                            tier1_job_base
                                .checked_add(tier1_job_index)
                                .ok_or_else(|| batch_err("Tier-1 index overflow"))?,
                        )
                        .map_err(|_| batch_err("Tier-1 index exceeds u32"))?,
                        previously_included: 0,
                        l_block: 3,
                        block_coding_mode: params.block_coding_mode,
                    });
                }
                packet_subbands.push(J2kPacketSubband {
                    block_offset,
                    block_count: subband.code_block_count,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                });
                local_subband_count = local_subband_count
                    .checked_add(1)
                    .ok_or_else(|| batch_err("subband count overflow"))?;
                local_resident_block_count = local_resident_block_count
                    .checked_add(code_block_count)
                    .ok_or_else(|| batch_err("resident block count overflow"))?;
            }
            packet_resolutions.push(J2kPacketResolution {
                subband_offset,
                subband_count: u32::try_from(resolution.subbands.len())
                    .map_err(|_| batch_err("resolution subband count exceeds u32"))?,
            });
        }

        if tile.resolutions.len()
            != usize::try_from(tile.resolution_count)
                .map_err(|_| batch_err("resolution count exceeds usize"))?
        {
            return Err(batch_err("resolution count mismatch"));
        }
        if local_resident_block_count
            != usize::try_from(tile.code_block_count)
                .map_err(|_| batch_err("code-block count exceeds usize"))?
        {
            return Err(batch_err("code-block count mismatch"));
        }

        let mut state_block_offsets = HashMap::<u32, (u32, usize)>::new();
        for descriptor in &tile.packet_descriptors {
            let packet_index = usize::try_from(descriptor.packet_index)
                .map_err(|_| batch_err("descriptor packet index exceeds usize"))?;
            let resolution = packet_resolutions
                .get(local_resolution_offset + packet_index)
                .ok_or_else(|| batch_err("descriptor packet index out of range"))?;
            let subband_start = usize::try_from(resolution.subband_offset)
                .map_err(|_| batch_err("descriptor subband offset exceeds usize"))?;
            let subband_count = usize::try_from(resolution.subband_count)
                .map_err(|_| batch_err("descriptor subband count exceeds usize"))?;
            let mut packet_block_count = 0usize;
            for subband in &packet_subbands[local_subband_offset + subband_start
                ..local_subband_offset + subband_start + subband_count]
            {
                let subband_block_count = usize::try_from(subband.block_count)
                    .map_err(|_| batch_err("descriptor block count exceeds usize"))?;
                packet_block_count = packet_block_count
                    .checked_add(subband_block_count)
                    .ok_or_else(|| batch_err("descriptor block count overflow"))?;
            }
            let (state_block_offset, existing_count) = if let Some(&(offset, count)) =
                state_block_offsets.get(&descriptor.state_index)
            {
                (offset, count)
            } else {
                let offset = u32::try_from(state_blocks.len() - local_state_block_offset)
                    .map_err(|_| batch_err("state block offset exceeds u32"))?;
                for subband in &packet_subbands[local_subband_offset + subband_start
                    ..local_subband_offset + subband_start + subband_count]
                {
                    for _ in 0..subband.block_count {
                        state_blocks.push(J2kPacketStateBlock {
                            previously_included: 0,
                            l_block: 3,
                        });
                    }
                }
                state_block_offsets.insert(descriptor.state_index, (offset, packet_block_count));
                (offset, packet_block_count)
            };
            if existing_count != packet_block_count {
                return Err(batch_err("descriptor state layout mismatch"));
            }
            local_payload_copy_job_capacity = local_payload_copy_job_capacity
                .checked_add(packet_block_count)
                .ok_or_else(|| batch_err("packet payload-copy job count overflow"))?;
            packet_descriptors.push(J2kPacketDescriptor {
                packet_index: descriptor.packet_index,
                state_index: descriptor.state_index,
                layer: u32::from(descriptor.layer),
                resolution: descriptor.resolution,
                component: u32::from(descriptor.component),
                precinct_lo: u32::try_from(descriptor.precinct & u64::from(u32::MAX))
                    .expect("masked precinct low word fits u32"),
                precinct_hi: u32::try_from(descriptor.precinct >> 32)
                    .expect("precinct high word fits u32"),
                state_block_offset,
            });
        }

        let header_capacity = local_resident_block_count
            .checked_mul(256)
            .and_then(|bytes| bytes.checked_add(4096))
            .map(|bytes| bytes.max(4096))
            .ok_or_else(|| batch_err("packet header capacity overflow"))?;
        let packet_output_capacity =
            tile_packet_output_capacity(tile_index, tile, header_capacity)?;
        let codestream_capacity =
            lossless_codestream_assembly_capacity(packet_output_capacity, tile.codestream)?;
        let codestream_payload_offset = lossless_codestream_payload_offset(tile.codestream)?;
        let scratch_words = max_tree_nodes
            .checked_mul(6)
            .ok_or_else(|| batch_err("scratch size overflow"))?;

        let header_offset = header_capacity_total;
        let scratch_offset = scratch_words_total;
        if tile.packet_descriptors.is_empty() {
            local_payload_copy_job_capacity = local_resident_block_count;
        }
        let payload_copy_offset = packet_payload_copy_job_capacity_total;
        let codestream_offset = codestream_capacity_total;
        let packet_output_offset = codestream_offset
            .checked_add(codestream_payload_offset)
            .ok_or_else(|| batch_err("direct packet output offset overflow"))?;
        packet_jobs.push(J2kBatchedPacketEncodeJob {
            resolution_offset: u32::try_from(local_resolution_offset)
                .map_err(|_| batch_err("resolution offset exceeds u32"))?,
            subband_offset: u32::try_from(local_subband_offset)
                .map_err(|_| batch_err("subband offset exceeds u32"))?,
            block_offset: u32::try_from(local_block_offset)
                .map_err(|_| batch_err("block offset exceeds u32"))?,
            descriptor_offset: u32::try_from(local_descriptor_offset)
                .map_err(|_| batch_err("descriptor offset exceeds u32"))?,
            state_block_offset: u32::try_from(local_state_block_offset)
                .map_err(|_| batch_err("state block offset exceeds u32"))?,
            output_offset: u32::try_from(packet_output_offset)
                .map_err(|_| batch_err("packet output offset exceeds u32"))?,
            header_offset: u32::try_from(header_offset)
                .map_err(|_| batch_err("header offset exceeds u32"))?,
            scratch_offset: u32::try_from(scratch_offset)
                .map_err(|_| batch_err("scratch offset exceeds u32"))?,
            payload_copy_offset: u32::try_from(payload_copy_offset)
                .map_err(|_| batch_err("packet payload-copy offset exceeds u32"))?,
            payload_copy_capacity: u32::try_from(local_payload_copy_job_capacity)
                .map_err(|_| batch_err("packet payload-copy capacity exceeds u32"))?,
            resolution_count: tile.resolution_count,
            num_layers: u32::from(tile.num_layers),
            num_components: u32::from(tile.component_count),
            code_block_count: tile.code_block_count,
            subband_count: u32::try_from(local_subband_count)
                .map_err(|_| batch_err("local subband count exceeds u32"))?,
            descriptor_count: u32::try_from(tile.packet_descriptors.len())
                .map_err(|_| batch_err("descriptor count exceeds u32"))?,
            output_capacity: u32::try_from(packet_output_capacity)
                .map_err(|_| batch_err("packet output capacity exceeds u32"))?,
            header_capacity: u32::try_from(header_capacity)
                .map_err(|_| batch_err("header capacity exceeds u32"))?,
            scratch_node_capacity: u32::try_from(max_tree_nodes)
                .map_err(|_| batch_err("scratch node capacity exceeds u32"))?,
        });
        assembly_jobs.push(J2kBatchedCodestreamAssemblyJob {
            tile_data_offset: u32::try_from(packet_output_offset)
                .map_err(|_| batch_err("assembly packet offset exceeds u32"))?,
            codestream_offset: u32::try_from(codestream_offset)
                .map_err(|_| batch_err("codestream offset exceeds u32"))?,
            width: tile.codestream.width,
            height: tile.codestream.height,
            num_components: u32::from(tile.codestream.component_count),
            bit_depth: u32::from(tile.codestream.bit_depth),
            signed_samples: u32::from(tile.codestream.signed),
            num_decomposition_levels: u32::from(tile.codestream.num_decomposition_levels),
            use_mct: u32::from(tile.codestream.use_mct),
            guard_bits: u32::from(tile.codestream.guard_bits),
            progression_order: codestream_progression_order_code(tile.codestream.progression_order),
            write_tlm: u32::from(tile.codestream.write_tlm),
            high_throughput: params.high_throughput,
            code_block_style: params.code_block_style,
            code_block_width_exp: u32::from(tile.codestream.code_block_width_exp),
            code_block_height_exp: u32::from(tile.codestream.code_block_height_exp),
            output_capacity: u32::try_from(codestream_capacity)
                .map_err(|_| batch_err("codestream capacity exceeds u32"))?,
        });
        codestream_offsets.push(codestream_offset);
        codestream_capacities.push(codestream_capacity);
        packet_output_capacity_total = packet_output_capacity_total
            .checked_add(packet_output_capacity)
            .ok_or_else(|| batch_err("packet output total overflow"))?;
        packet_payload_copy_job_capacity_total = packet_payload_copy_job_capacity_total
            .checked_add(local_payload_copy_job_capacity)
            .ok_or_else(|| batch_err("packet payload-copy job total overflow"))?;
        max_payload_copy_jobs_per_tile =
            max_payload_copy_jobs_per_tile.max(local_payload_copy_job_capacity);
        header_capacity_total = header_capacity_total
            .checked_add(header_capacity)
            .ok_or_else(|| batch_err("header total overflow"))?;
        scratch_words_total = scratch_words_total
            .checked_add(scratch_words)
            .ok_or_else(|| batch_err("scratch total overflow"))?;
        codestream_capacity_total = codestream_capacity_total
            .checked_add(codestream_capacity)
            .ok_or_else(|| batch_err("codestream total overflow"))?;
    }

    Ok(ResidentBatchPacketPlan {
        packet_resolutions,
        packet_subbands,
        resident_blocks,
        packet_descriptors,
        state_blocks,
        packet_jobs,
        assembly_jobs,
        packet_output_capacity_total,
        packet_payload_copy_job_capacity_total,
        max_payload_copy_jobs_per_tile,
        header_capacity_total,
        scratch_words_total,
        codestream_capacity_total,
        codestream_offsets,
        codestream_capacities,
    })
}
