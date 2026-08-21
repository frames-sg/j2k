// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    abi::{
        J2kHtPacketBlock, J2kHtPacketJob, J2kHtPacketStatus, J2kHtPacketSubband,
        J2kHtPacketSubbandTagState, J2kHtPacketTagNodeState,
    },
    constants::{J2K_ENCODE_STATUS_FAIL, J2K_ENCODE_STATUS_OK, J2K_ENCODE_STATUS_UNSUPPORTED},
    helpers::{load_job, load_u8, store_u8},
    packet_writer::{
        j2k_packet_encode_ht_segment_lengths, j2k_packet_encode_num_ht_passes, j2k_packet_finish,
        j2k_packet_write_bit, j2k_packet_writer_init,
    },
    tag_tree::{j2k_packet_build_tag_trees, j2k_packet_tag_tree_encode, j2k_packet_tag_tree_new},
};
use cuda_device::thread;

pub(crate) struct J2kPacketHeaderResult {
    pub(crate) code: u32,
    pub(crate) detail: u32,
    pub(crate) header_len: u32,
    pub(crate) body_len: u32,
    pub(crate) output_len: u32,
}

#[inline(always)]
pub(crate) fn j2k_packet_status(
    status: *mut J2kHtPacketStatus,
    code: u32,
    detail: u32,
    output_len: u32,
) {
    unsafe {
        (*status).code = code;
        (*status).detail = detail;
        (*status).output_len = output_len;
        (*status).reserved0 = 0;
    }
}

#[inline(always)]
fn j2k_packet_header_result(
    code: u32,
    detail: u32,
    header_len: u32,
    body_len: u32,
    output_len: u32,
) -> J2kPacketHeaderResult {
    J2kPacketHeaderResult {
        code,
        detail,
        header_len,
        body_len,
        output_len,
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_build_header_serial(
    payload_len: u64,
    packet: J2kHtPacketJob,
    subbands: *const J2kHtPacketSubband,
    blocks: *const J2kHtPacketBlock,
    tag_states: *const J2kHtPacketSubbandTagState,
    tag_nodes: *const J2kHtPacketTagNodeState,
    tag_state_count: u64,
    tag_node_count: u64,
    packet_out: *mut u8,
) -> J2kPacketHeaderResult {
    let mut writer = j2k_packet_writer_init(packet_out, packet.output_capacity);

    let mut any_data = 0_u32;
    let mut subband_idx = 0_u32;
    while subband_idx < packet.subband_count {
        let subband = load_job(subbands, packet.subband_start + subband_idx);
        if subband.num_cbs_x == 0
            || subband.num_cbs_y == 0
            || subband.num_cbs_x.checked_mul(subband.num_cbs_y) != Some(subband.block_count)
        {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 7, 0, 0, 0);
        }
        let mut idx = 0_u32;
        while idx < subband.block_count {
            let block = load_job(blocks, subband.block_start + idx);
            if block.num_coding_passes > 0 {
                any_data = 1;
            }
            if block.num_coding_passes > 164 {
                return j2k_packet_header_result(J2K_ENCODE_STATUS_UNSUPPORTED, 1, 0, 0, 0);
            }
            let Some(data_end) = (block.data_offset as u64).checked_add(block.data_len as u64)
            else {
                return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 2, 0, 0, 0);
            };
            if data_end > payload_len {
                return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 2, 0, 0, 0);
            }
            if block.num_coding_passes == 0 {
                if block.data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0
                {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 10, 0, 0, 0);
                }
            } else if block.num_coding_passes == 1 {
                let cleanup_length = if block.cleanup_length == 0 {
                    block.data_len
                } else {
                    block.cleanup_length
                };
                if cleanup_length != block.data_len || block.refinement_length != 0 {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 11, 0, 0, 0);
                }
            } else {
                let segment_len = block.cleanup_length as u64 + block.refinement_length as u64;
                if block.cleanup_length == 0
                    || block.refinement_length == 0
                    || segment_len != block.data_len as u64
                    || block.cleanup_length < 2
                    || block.cleanup_length >= 65_535
                    || block.refinement_length >= 2047
                {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 12, 0, 0, 0);
                }
            }
            idx += 1;
        }
        subband_idx += 1;
    }

    if any_data == 0 {
        j2k_packet_write_bit(&mut writer, 0);
        j2k_packet_finish(&mut writer);
        if writer.failed != 0 {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 3, 0, 0, 0);
        }
        return j2k_packet_header_result(J2K_ENCODE_STATUS_OK, 0, writer.pos, 0, writer.pos);
    }

    j2k_packet_write_bit(&mut writer, 1);
    subband_idx = 0;
    while subband_idx < packet.subband_count {
        let subband_meta_idx = packet.subband_start + subband_idx;
        let subband = load_job(subbands, subband_meta_idx);
        let mut inclusion_tree = j2k_packet_tag_tree_new();
        let mut zbp_tree = j2k_packet_tag_tree_new();
        j2k_packet_build_tag_trees(
            &mut inclusion_tree,
            &mut zbp_tree,
            subband,
            blocks,
            tag_states,
            tag_nodes,
            tag_state_count,
            tag_node_count,
            subband_meta_idx,
        );
        if inclusion_tree.failed != 0 || zbp_tree.failed != 0 {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_UNSUPPORTED, 8, 0, 0, 0);
        }

        let mut idx = 0_u32;
        while idx < subband.block_count {
            let block = load_job(blocks, subband.block_start + idx);
            let x = idx % subband.num_cbs_x;
            let y = idx / subband.num_cbs_x;
            if block.previously_included == 0 {
                if block.num_coding_passes > 0 && block.inclusion_layer != packet.layer {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 9, 0, 0, 0);
                }
                j2k_packet_tag_tree_encode(
                    &mut inclusion_tree,
                    x,
                    y,
                    packet.layer + 1,
                    &mut writer,
                );
                if block.num_coding_passes == 0 {
                    idx += 1;
                    continue;
                }
                j2k_packet_tag_tree_encode(
                    &mut zbp_tree,
                    x,
                    y,
                    block.num_zero_bitplanes + 1,
                    &mut writer,
                );
            } else if block.num_coding_passes > 0 {
                j2k_packet_write_bit(&mut writer, 1);
            } else {
                j2k_packet_write_bit(&mut writer, 0);
                idx += 1;
                continue;
            }
            j2k_packet_encode_num_ht_passes(&mut writer, block.num_coding_passes);
            j2k_packet_encode_ht_segment_lengths(&mut writer, block);
            idx += 1;
        }
        subband_idx += 1;
    }

    j2k_packet_finish(&mut writer);
    if writer.failed != 0 {
        return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 4, 0, 0, 0);
    }
    if writer.pos > 0 && load_u8(packet_out.cast_const(), (writer.pos - 1) as u64) == 0xff {
        if writer.pos >= packet.output_capacity {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 5, 0, 0, 0);
        }
        store_u8(packet_out, writer.pos as u64, 0);
        writer.pos += 1;
    }

    let header_len = writer.pos;
    let mut body_len = 0_u32;
    subband_idx = 0;
    while subband_idx < packet.subband_count {
        let subband = load_job(subbands, packet.subband_start + subband_idx);
        let mut idx = 0_u32;
        while idx < subband.block_count {
            let block = load_job(blocks, subband.block_start + idx);
            if block.num_coding_passes != 0 && block.data_len != 0 {
                let Some(next_body_len) = body_len.checked_add(block.data_len) else {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 6, 0, 0, 0);
                };
                let Some(total_len) = header_len.checked_add(next_body_len) else {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 6, 0, 0, 0);
                };
                if total_len > packet.output_capacity {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 6, 0, 0, 0);
                }
                body_len = next_body_len;
            }
            idx += 1;
        }
        subband_idx += 1;
    }

    j2k_packet_header_result(
        J2K_ENCODE_STATUS_OK,
        0,
        header_len,
        body_len,
        header_len + body_len,
    )
}

#[inline(always)]
pub(crate) fn j2k_packet_copy_body_cooperative(
    payload: *const u8,
    packet: J2kHtPacketJob,
    subbands: *const J2kHtPacketSubband,
    blocks: *const J2kHtPacketBlock,
    packet_out: *mut u8,
    header_len: u32,
    body_len: u32,
) {
    let mut body_byte = thread::threadIdx_x();
    let step = thread::blockDim_x();
    while body_byte < body_len {
        let mut remaining = body_byte;
        let mut copied = false;
        let mut subband_idx = 0_u32;
        while subband_idx < packet.subband_count && !copied {
            let subband = load_job(subbands, packet.subband_start + subband_idx);
            let mut idx = 0_u32;
            while idx < subband.block_count {
                let block = load_job(blocks, subband.block_start + idx);
                if block.num_coding_passes != 0 && block.data_len != 0 {
                    if remaining < block.data_len {
                        store_u8(
                            packet_out,
                            (header_len + body_byte) as u64,
                            load_u8(payload, (block.data_offset + remaining) as u64),
                        );
                        copied = true;
                        break;
                    }
                    remaining -= block.data_len;
                }
                idx += 1;
            }
            subband_idx += 1;
        }
        body_byte += step;
    }
}
