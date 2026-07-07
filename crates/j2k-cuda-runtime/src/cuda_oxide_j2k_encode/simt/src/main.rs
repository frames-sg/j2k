#![allow(
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::too_many_arguments,
    static_mut_refs
)]

use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;

include!("../../../cuda_oxide_simt_prelude.rs");

const J2K_FDWT97_ALPHA: f32 = j2k_codec_math::dwt::DWT97_ALPHA_F32;
const J2K_FDWT97_BETA: f32 = j2k_codec_math::dwt::DWT97_BETA_F32;
const J2K_FDWT97_GAMMA: f32 = j2k_codec_math::dwt::DWT97_GAMMA_F32;
const J2K_FDWT97_DELTA: f32 = j2k_codec_math::dwt::DWT97_DELTA_F32;
const J2K_FDWT97_KAPPA: f32 = j2k_codec_math::dwt::DWT97_KAPPA_F32;
const J2K_FDWT97_INV_KAPPA: f32 = j2k_codec_math::dwt::DWT97_INV_KAPPA_F32;
const J2K_HT_MEL_SIZE: u32 = 192;
const J2K_HT_VLC_SIZE: u32 = 3072 - J2K_HT_MEL_SIZE;
const J2K_HT_MS_SIZE: u32 = ((16384 * 16) + 14) / 15;
const J2K_HT_MEL_OFFSET: u32 = J2K_HT_MS_SIZE;
const J2K_HT_VLC_OFFSET: u32 = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;
const J2K_HT_COMPACT_ASSEMBLE_FLAG: u32 = 0x8000_0000;
const J2K_HT_COMPACT_LENGTH_MASK: u32 = 0x7fff;
const J2K_ENCODE_STATUS_OK: u32 = 0;
const J2K_ENCODE_STATUS_FAIL: u32 = 1;
const J2K_ENCODE_STATUS_UNSUPPORTED: u32 = 2;
const J2K_PACKET_TAG_INF: u32 = 0x7fff_ffff;
const J2K_PACKET_MAX_TAG_NODES: usize = 2048;
const J2K_PACKET_MAX_TAG_LEVELS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtEncodeCompactJob {
    source_offset: u32,
    compact_offset: u32,
    data_len: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtPacketJob {
    block_start: u32,
    block_count: u32,
    subband_start: u32,
    subband_count: u32,
    output_offset: u32,
    output_capacity: u32,
    layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtPacketSubband {
    block_start: u32,
    block_count: u32,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtPacketBlock {
    data_offset: u32,
    data_len: u32,
    cleanup_length: u32,
    refinement_length: u32,
    num_coding_passes: u32,
    num_zero_bitplanes: u32,
    l_block: u32,
    previously_included: u32,
    inclusion_layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtPacketSubbandTagState {
    inclusion_node_start: u32,
    zero_bitplane_node_start: u32,
    node_count: u32,
    reserved0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct J2kHtPacketTagNodeState {
    current: u32,
    known: u32,
}

#[repr(C)]
struct J2kHtPacketStatus {
    code: u32,
    detail: u32,
    output_len: u32,
    reserved0: u32,
}

struct J2kPacketBitWriter {
    out: *mut u8,
    pos: u32,
    capacity: u32,
    buffer: u32,
    bits_in_buffer: u32,
    last_byte_was_ff: u32,
    failed: u32,
}

struct J2kPacketTagTree {
    values: [u32; J2K_PACKET_MAX_TAG_NODES],
    current: [u32; J2K_PACKET_MAX_TAG_NODES],
    known: [u32; J2K_PACKET_MAX_TAG_NODES],
    widths: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    heights: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    offsets: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    levels: u32,
    total_nodes: u32,
    failed: u32,
}

struct J2kPacketHeaderResult {
    code: u32,
    detail: u32,
    header_len: u32,
    body_len: u32,
    output_len: u32,
}

#[inline(always)]
fn load_u8(ptr: *const u8, index: u64) -> u8 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_u32(ptr: *const u32, index: u64) -> u32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_f32(ptr: *const f32, index: u32) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn load_f32_u64(ptr: *const f32, index: u64) -> f32 {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u32, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_f32_u64(ptr: *mut f32, index: u64, value: f32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_i32(ptr: *mut i32, index: u64, value: i32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_u8(ptr: *mut u8, index: u64, value: u8) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn store_u32(ptr: *mut u32, index: u64, value: u32) {
    simt_store(ptr, index as usize, value);
}

#[inline(always)]
fn load_job<T: Copy>(ptr: *const T, index: u32) -> T {
    simt_load(ptr, index as usize)
}

#[inline(always)]
fn j2k_packet_status(status: *mut J2kHtPacketStatus, code: u32, detail: u32, output_len: u32) {
    unsafe {
        (*status).code = code;
        (*status).detail = detail;
        (*status).output_len = output_len;
        (*status).reserved0 = 0;
    }
}

#[inline(always)]
fn j2k_packet_writer_init(out: *mut u8, capacity: u32) -> J2kPacketBitWriter {
    J2kPacketBitWriter {
        out,
        pos: 0,
        capacity,
        buffer: 0,
        bits_in_buffer: 0,
        last_byte_was_ff: 0,
        failed: 0,
    }
}

#[inline(always)]
fn j2k_packet_push_byte(writer: &mut J2kPacketBitWriter, byte: u8) {
    if writer.pos >= writer.capacity {
        writer.failed = 1;
        return;
    }
    store_u8(writer.out, writer.pos as u64, byte);
    writer.pos += 1;
    writer.last_byte_was_ff = if byte == 0xff { 1 } else { 0 };
}

#[inline(always)]
fn j2k_packet_flush_byte(writer: &mut J2kPacketBitWriter) {
    let limit = if writer.last_byte_was_ff != 0 { 7 } else { 8 };
    let byte = ((writer.buffer >> (writer.bits_in_buffer - limit)) & 0xff) as u8;
    j2k_packet_push_byte(writer, byte);
    writer.bits_in_buffer -= limit;
    writer.buffer = if writer.bits_in_buffer == 0 {
        0
    } else {
        writer.buffer & ((1_u32 << writer.bits_in_buffer) - 1)
    };
}

#[inline(always)]
fn j2k_packet_write_bit(writer: &mut J2kPacketBitWriter, bit: u32) {
    writer.buffer = (writer.buffer << 1) | (bit & 1);
    writer.bits_in_buffer += 1;
    let limit = if writer.last_byte_was_ff != 0 { 7 } else { 8 };
    if writer.bits_in_buffer >= limit {
        j2k_packet_flush_byte(writer);
    }
}

#[inline(always)]
fn j2k_packet_write_bits(writer: &mut J2kPacketBitWriter, value: u32, mut count: u32) {
    while count > 0 {
        count -= 1;
        j2k_packet_write_bit(writer, (value >> count) & 1);
    }
}

#[inline(always)]
fn j2k_packet_finish(writer: &mut J2kPacketBitWriter) {
    if writer.bits_in_buffer == 0 {
        return;
    }
    let limit = if writer.last_byte_was_ff != 0 { 7 } else { 8 };
    let shift = limit - writer.bits_in_buffer;
    let byte = ((writer.buffer << shift) & 0xff) as u8;
    j2k_packet_push_byte(writer, byte);
    writer.buffer = 0;
    writer.bits_in_buffer = 0;
}

#[inline(always)]
fn j2k_packet_encode_num_ht_passes(writer: &mut J2kPacketBitWriter, num_passes: u32) {
    if num_passes == 1 {
        j2k_packet_write_bit(writer, 0);
    } else if num_passes == 2 {
        j2k_packet_write_bits(writer, 0b10, 2);
    } else if num_passes <= 5 {
        j2k_packet_write_bits(writer, 0b11, 2);
        j2k_packet_write_bits(writer, num_passes - 3, 2);
    } else if num_passes <= 36 {
        j2k_packet_write_bits(writer, 0b11, 2);
        j2k_packet_write_bits(writer, 0b11, 2);
        j2k_packet_write_bits(writer, num_passes - 6, 5);
    } else {
        j2k_packet_write_bits(writer, 0b11, 2);
        j2k_packet_write_bits(writer, 0b11, 2);
        j2k_packet_write_bits(writer, 31, 5);
        j2k_packet_write_bits(writer, num_passes - 37, 7);
    }
}

#[inline(always)]
fn j2k_packet_value_fits(value: u32, bits: u32) -> bool {
    bits >= 32 || value < (1_u32 << bits)
}

#[inline(always)]
fn j2k_packet_ht_length_bits(l_block: u32, num_passes: u32) -> u32 {
    let placeholder_groups = (if num_passes > 0 { num_passes - 1 } else { 0 }) / 3;
    let placeholder_passes = placeholder_groups * 3;
    let mut value = placeholder_passes + 1;
    let mut log2_value = 0;
    while value > 1 {
        value >>= 1;
        log2_value += 1;
    }
    l_block + log2_value
}

#[inline(always)]
fn j2k_packet_encode_ht_segment_lengths(writer: &mut J2kPacketBitWriter, block: J2kHtPacketBlock) {
    let cleanup_length = if block.num_coding_passes == 1 && block.cleanup_length == 0 {
        block.data_len
    } else {
        block.cleanup_length
    };
    let mut l_block = block.l_block;
    let mut cleanup_bits = j2k_packet_ht_length_bits(l_block, block.num_coding_passes);
    let refinement_extra_bits = if block.num_coding_passes > 2 { 1 } else { 0 };
    while !j2k_packet_value_fits(cleanup_length, cleanup_bits)
        || (block.num_coding_passes > 1
            && !j2k_packet_value_fits(block.refinement_length, l_block + refinement_extra_bits))
    {
        j2k_packet_write_bit(writer, 1);
        l_block += 1;
        cleanup_bits += 1;
    }
    j2k_packet_write_bit(writer, 0);
    j2k_packet_write_bits(writer, cleanup_length, cleanup_bits);

    if block.num_coding_passes > 1 {
        j2k_packet_write_bits(
            writer,
            block.refinement_length,
            l_block + refinement_extra_bits,
        );
    }
}

#[inline(always)]
fn j2k_packet_tag_tree_new() -> J2kPacketTagTree {
    J2kPacketTagTree {
        values: [0; J2K_PACKET_MAX_TAG_NODES],
        current: [0; J2K_PACKET_MAX_TAG_NODES],
        known: [0; J2K_PACKET_MAX_TAG_NODES],
        widths: [0; J2K_PACKET_MAX_TAG_LEVELS],
        heights: [0; J2K_PACKET_MAX_TAG_LEVELS],
        offsets: [0; J2K_PACKET_MAX_TAG_LEVELS],
        levels: 0,
        total_nodes: 0,
        failed: 0,
    }
}

#[inline(always)]
fn j2k_packet_tag_tree_fail(tree: &mut J2kPacketTagTree) -> bool {
    tree.failed = 1;
    false
}

#[inline(always)]
fn j2k_packet_tag_tree_init(tree: &mut J2kPacketTagTree, width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return j2k_packet_tag_tree_fail(tree);
    }

    let mut w = width;
    let mut h = height;
    let mut total = 0_u32;
    let mut levels = 0_u32;
    loop {
        if levels as usize >= J2K_PACKET_MAX_TAG_LEVELS {
            return j2k_packet_tag_tree_fail(tree);
        }
        let Some(nodes) = w.checked_mul(h) else {
            return j2k_packet_tag_tree_fail(tree);
        };
        let Some(next_total) = total.checked_add(nodes) else {
            return j2k_packet_tag_tree_fail(tree);
        };
        if next_total as usize > J2K_PACKET_MAX_TAG_NODES {
            return j2k_packet_tag_tree_fail(tree);
        }

        let level = levels as usize;
        tree.widths[level] = w;
        tree.heights[level] = h;
        tree.offsets[level] = total;
        total = next_total;
        levels += 1;
        if w <= 1 && h <= 1 {
            break;
        }
        w = (w + 1) >> 1;
        h = (h + 1) >> 1;
    }

    tree.levels = levels;
    tree.total_nodes = total;
    tree.failed = 0;
    let mut idx = 0_u32;
    while idx < total {
        let slot = idx as usize;
        tree.values[slot] = 0;
        tree.current[slot] = 0;
        tree.known[slot] = 0;
        idx += 1;
    }
    true
}

#[inline(always)]
fn j2k_packet_tag_tree_propagate(tree: &mut J2kPacketTagTree) {
    let mut level = 1_u32;
    while level < tree.levels {
        let prev_w = tree.widths[(level - 1) as usize];
        let prev_h = tree.heights[(level - 1) as usize];
        let curr_w = tree.widths[level as usize];
        let curr_h = tree.heights[level as usize];
        let mut cy = 0_u32;
        while cy < curr_h {
            let mut cx = 0_u32;
            while cx < curr_w {
                let mut min_value = u32::MAX;
                let child_x0 = cx << 1;
                let child_y0 = cy << 1;
                let child_x1 = (child_x0 + 2).min(prev_w);
                let child_y1 = (child_y0 + 2).min(prev_h);
                let mut y = child_y0;
                while y < child_y1 {
                    let mut x = child_x0;
                    while x < child_x1 {
                        let child_idx = tree.offsets[(level - 1) as usize] + y * prev_w + x;
                        min_value = min_value.min(tree.values[child_idx as usize]);
                        x += 1;
                    }
                    y += 1;
                }
                let node_idx = tree.offsets[level as usize] + cy * curr_w + cx;
                tree.values[node_idx as usize] = min_value;
                cx += 1;
            }
            cy += 1;
        }
        level += 1;
    }
}

#[inline(always)]
fn j2k_packet_build_tag_trees(
    inclusion_tree: &mut J2kPacketTagTree,
    zbp_tree: &mut J2kPacketTagTree,
    subband: J2kHtPacketSubband,
    blocks: *const J2kHtPacketBlock,
    tag_states: *const J2kHtPacketSubbandTagState,
    tag_nodes: *const J2kHtPacketTagNodeState,
    tag_state_count: u64,
    tag_node_count: u64,
    subband_meta_idx: u32,
) {
    if !j2k_packet_tag_tree_init(inclusion_tree, subband.num_cbs_x, subband.num_cbs_y)
        || !j2k_packet_tag_tree_init(zbp_tree, subband.num_cbs_x, subband.num_cbs_y)
    {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }

    let mut idx = 0_u32;
    while idx < subband.block_count {
        let block = load_job(blocks, subband.block_start + idx);
        let x = idx % subband.num_cbs_x;
        let y = idx / subband.num_cbs_x;
        let leaf_idx = y * subband.num_cbs_x + x;
        inclusion_tree.values[leaf_idx as usize] = if block.previously_included == 0 {
            block.inclusion_layer
        } else {
            J2K_PACKET_TAG_INF
        };
        zbp_tree.values[leaf_idx as usize] = block.num_zero_bitplanes;
        idx += 1;
    }
    j2k_packet_tag_tree_propagate(inclusion_tree);
    j2k_packet_tag_tree_propagate(zbp_tree);

    if tag_state_count == 0 {
        return;
    }
    if subband_meta_idx as u64 >= tag_state_count {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }
    let state = load_job(tag_states, subband_meta_idx);
    if state.node_count != inclusion_tree.total_nodes {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }
    let Some(inclusion_end) =
        (state.inclusion_node_start as u64).checked_add(state.node_count as u64)
    else {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    };
    let Some(zbp_end) =
        (state.zero_bitplane_node_start as u64).checked_add(state.node_count as u64)
    else {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    };
    if inclusion_end > tag_node_count || zbp_end > tag_node_count {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }

    let mut node_idx = 0_u32;
    while node_idx < state.node_count {
        let inclusion_node = load_job(tag_nodes, state.inclusion_node_start + node_idx);
        let zbp_node = load_job(tag_nodes, state.zero_bitplane_node_start + node_idx);
        let slot = node_idx as usize;
        inclusion_tree.current[slot] = inclusion_node.current;
        inclusion_tree.known[slot] = inclusion_node.known;
        zbp_tree.current[slot] = zbp_node.current;
        zbp_tree.known[slot] = zbp_node.known;
        node_idx += 1;
    }
}

#[inline(always)]
fn j2k_packet_tag_tree_encode(
    tree: &mut J2kPacketTagTree,
    x: u32,
    y: u32,
    max_value: u32,
    writer: &mut J2kPacketBitWriter,
) {
    let mut path = [0_u32; J2K_PACKET_MAX_TAG_LEVELS];
    let mut cx = x;
    let mut cy = y;
    let mut level = 0_u32;
    while level < tree.levels {
        path[level as usize] = tree.offsets[level as usize] + cy * tree.widths[level as usize] + cx;
        cx >>= 1;
        cy >>= 1;
        level += 1;
    }

    let mut parent_value = 0_u32;
    let mut reverse = tree.levels;
    while reverse > 0 {
        let node_idx = path[(reverse - 1) as usize];
        let slot = node_idx as usize;
        let mut start = tree.current[slot].max(parent_value);
        if tree.known[slot] == 0 {
            let target = tree.values[slot].min(max_value);
            while start < target {
                j2k_packet_write_bit(writer, 0);
                start += 1;
            }
            if tree.values[slot] < max_value {
                j2k_packet_write_bit(writer, 1);
                tree.known[slot] = 1;
            }
            tree.current[slot] = target;
        }
        parent_value = tree.current[slot];
        reverse -= 1;
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
fn j2k_packet_build_header_serial(
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
fn j2k_packet_copy_body_cooperative(
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

#[inline(always)]
fn floor_f32(value: f32) -> f32 {
    // f32::floor routes through libdevice in cuda-oxide, which emits NVVM IR
    // instead of the PTX loaded by this runtime path.
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

#[inline(always)]
fn j2k_fdwt53_predict_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let left = load_f32(src, row_base + odd - 1);
    let right = if odd + 1 < width {
        load_f32(src, row_base + odd + 1)
    } else {
        load_f32(src, row_base + last_even)
    };
    load_f32(src, row_base + odd) - floor_f32((left + right) * 0.5)
}

#[inline(always)]
fn j2k_fdwt53_predict_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 {
        height - 2
    } else {
        height - 1
    };
    let top = load_f32(src, (odd - 1) * full_width + x);
    let bottom = if odd + 1 < height {
        load_f32(src, (odd + 1) * full_width + x)
    } else {
        load_f32(src, last_even * full_width + x)
    };
    load_f32(src, odd * full_width + x) - floor_f32((top + bottom) * 0.5)
}

#[inline(always)]
fn j2k_fdwt97_high1_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let left = load_f32(src, row_base + odd - 1);
    let right = if odd + 1 < width {
        load_f32(src, row_base + odd + 1)
    } else {
        load_f32(src, row_base + last_even)
    };
    load_f32(src, row_base + odd) + J2K_FDWT97_ALPHA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_low1_row(src: *const f32, row_base: u32, width: u32, low_index: u32) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high1_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high1_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high1_row(src, row_base, width, low_index)
    } else {
        left
    };
    load_f32(src, row_base + even) + J2K_FDWT97_BETA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_high2_row(src: *const f32, row_base: u32, width: u32, high_index: u32) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let last_low = last_even / 2;
    let left = j2k_fdwt97_low1_row(src, row_base, width, high_index);
    let right = if odd + 1 < width {
        j2k_fdwt97_low1_row(src, row_base, width, high_index + 1)
    } else {
        j2k_fdwt97_low1_row(src, row_base, width, last_low)
    };
    j2k_fdwt97_high1_row(src, row_base, width, high_index) + J2K_FDWT97_GAMMA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_low2_row(src: *const f32, row_base: u32, width: u32, low_index: u32) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high2_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high2_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high2_row(src, row_base, width, low_index)
    } else {
        left
    };
    j2k_fdwt97_low1_row(src, row_base, width, low_index) + J2K_FDWT97_DELTA * (left + right)
}

#[inline(always)]
fn j2k_fdwt97_high1_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 {
        height - 2
    } else {
        height - 1
    };
    let top = load_f32(src, (odd - 1) * full_width + x);
    let bottom = if odd + 1 < height {
        load_f32(src, (odd + 1) * full_width + x)
    } else {
        load_f32(src, last_even * full_width + x)
    };
    load_f32(src, odd * full_width + x) + J2K_FDWT97_ALPHA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_low1_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high1_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    load_f32(src, even * full_width + x) + J2K_FDWT97_BETA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_high2_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 {
        height - 2
    } else {
        height - 1
    };
    let last_low = last_even / 2;
    let top = j2k_fdwt97_low1_col(src, x, full_width, height, high_index);
    let bottom = if odd + 1 < height {
        j2k_fdwt97_low1_col(src, x, full_width, height, high_index + 1)
    } else {
        j2k_fdwt97_low1_col(src, x, full_width, height, last_low)
    };
    j2k_fdwt97_high1_col(src, x, full_width, height, high_index) + J2K_FDWT97_GAMMA * (top + bottom)
}

#[inline(always)]
fn j2k_fdwt97_low2_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high2_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    j2k_fdwt97_low1_col(src, x, full_width, height, low_index) + J2K_FDWT97_DELTA * (top + bottom)
}

#[inline(always)]
fn ldexp_one_f32(exponent: i32) -> f32 {
    if exponent < -149 {
        0.0
    } else if exponent < -126 {
        f32::from_bits(1_u32 << ((exponent + 149) as u32))
    } else if exponent <= 127 {
        f32::from_bits(((exponent + 127) as u32) << 23)
    } else {
        f32::INFINITY
    }
}

#[inline(always)]
fn j2k_quantize_sample(
    sample: f32,
    step_exponent: u32,
    step_mantissa: u32,
    range_bits: u32,
    reversible: u32,
) -> i32 {
    if reversible != 0 {
        let rounded = if sample >= 0.0 {
            floor_f32(sample + 0.5)
        } else {
            -floor_f32(-sample + 0.5)
        };
        return rounded as i32;
    }

    let exponent = range_bits as i32 - step_exponent as i32;
    let base = ldexp_one_f32(exponent);
    let delta = base * (1.0 + step_mantissa as f32 / 2048.0);
    if delta <= 0.0 {
        return 0;
    }

    let sign = if sample < 0.0 { -1 } else { 1 };
    let magnitude = floor_f32(abs_f32(sample) / delta) as i32;
    sign * magnitude
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn j2k_deinterleave_to_f32(
        pixels: *const u8,
        components: *mut f32,
        num_pixels: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let pixel_base = idx * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_deinterleave_strided_to_f32(
        pixels: *const u8,
        components: *mut f32,
        width: u64,
        height: u64,
        byte_offset: u64,
        pitch_bytes: u64,
        num_components: u32,
        bit_depth: u32,
        is_signed: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        let num_pixels = width * height;
        if idx >= num_pixels || num_components == 0 || num_components > 4 {
            return;
        }

        let bytes_per_sample = if bit_depth <= 8 { 1_u32 } else { 2_u32 };
        let unsigned_offset = if is_signed != 0 {
            0.0
        } else {
            (1_u32 << (bit_depth - 1)) as f32
        };
        let y = idx / width;
        let x = idx - y * width;
        let pixel_base =
            byte_offset + y * pitch_bytes + x * num_components as u64 * bytes_per_sample as u64;
        let mut component = 0_u32;
        while component < num_components {
            let sample_base = pixel_base + component as u64 * bytes_per_sample as u64;
            let sample = if bit_depth <= 8 {
                let raw = load_u8(pixels, sample_base);
                if is_signed != 0 {
                    (raw as i8) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            } else {
                let raw = load_u8(pixels, sample_base) as u16
                    | ((load_u8(pixels, sample_base + 1) as u16) << 8);
                if is_signed != 0 {
                    (raw as i16) as f32
                } else {
                    raw as f32 - unsigned_offset
                }
            };
            store_f32_u64(components, component as u64 * num_pixels + idx, sample);
            component += 1;
        }
    }

    #[kernel]
    pub unsafe fn j2k_forward_rct(plane0: *mut f32, plane1: *mut f32, plane2: *mut f32, len: u64) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, floor_f32((r + 2.0 * g + b) * 0.25));
        store_f32_u64(plane1, idx, b - g);
        store_f32_u64(plane2, idx, r - g);
    }

    #[kernel]
    pub unsafe fn j2k_forward_ict(plane0: *mut f32, plane1: *mut f32, plane2: *mut f32, len: u64) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let r = load_f32_u64(plane0.cast_const(), idx);
        let g = load_f32_u64(plane1.cast_const(), idx);
        let b = load_f32_u64(plane2.cast_const(), idx);
        store_f32_u64(plane0, idx, 0.299 * r + 0.587 * g + 0.114 * b);
        store_f32_u64(plane1, idx, -0.16875 * r - 0.33126 * g + 0.5 * b);
        store_f32_u64(plane2, idx, 0.5 * r - 0.41869 * g - 0.08131 * b);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        if x < low_width {
            let even = x * 2;
            let left = if x > 0 {
                j2k_fdwt53_predict_row(src, row_base, current_width, x - 1)
            } else {
                j2k_fdwt53_predict_row(src, row_base, current_width, 0)
            };
            let right = if even + 1 < current_width {
                j2k_fdwt53_predict_row(src, row_base, current_width, x)
            } else {
                left
            };
            let value = load_f32(src, row_base + even) + floor_f32((left + right) * 0.25 + 0.5);
            store_f32(dst, row_base + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_row(src, row_base, current_width, x - low_width);
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt53_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        if y < low_height {
            let even = y * 2;
            let top = if y > 0 {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y - 1)
            } else {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, 0)
            };
            let bottom = if even + 1 < current_height {
                j2k_fdwt53_predict_col(src, x, full_width, current_height, y)
            } else {
                top
            };
            let value =
                load_f32(src, even * full_width + x) + floor_f32((top + bottom) * 0.25 + 0.5);
            store_f32(dst, y * full_width + x, value);
            return;
        }

        let value = j2k_fdwt53_predict_col(src, x, full_width, current_height, y - low_height);
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_horizontal(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_width: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let row_base = y * full_width;
        let value = if x < low_width {
            j2k_fdwt97_low2_row(src, row_base, current_width, x) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_row(src, row_base, current_width, x - low_width) * J2K_FDWT97_KAPPA
        };
        store_f32(dst, row_base + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_forward_dwt97_vertical(
        src: *const f32,
        dst: *mut f32,
        full_width: u32,
        current_width: u32,
        current_height: u32,
        low_height: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= current_width || y >= current_height {
            return;
        }

        let value = if y < low_height {
            j2k_fdwt97_low2_col(src, x, full_width, current_height, y) * J2K_FDWT97_INV_KAPPA
        } else {
            j2k_fdwt97_high2_col(src, x, full_width, current_height, y - low_height)
                * J2K_FDWT97_KAPPA
        };
        store_f32(dst, y * full_width + x, value);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband(
        samples: *const f32,
        coefficients: *mut i32,
        len: u64,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let idx = thread::index_1d().get() as u64;
        if idx >= len {
            return;
        }

        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, idx),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, idx, coefficient);
    }

    #[kernel]
    pub unsafe fn j2k_quantize_subband_strided(
        samples: *const f32,
        coefficients: *mut i32,
        x0: u32,
        y0: u32,
        width: u32,
        height: u32,
        stride: u32,
        step_exponent: u32,
        step_mantissa: u32,
        range_bits: u32,
        reversible: u32,
    ) {
        let x = thread::index_2d_col() as u32;
        let y = thread::index_2d_row() as u32;
        if x >= width || y >= height {
            return;
        }

        let source_index = (y0 + y) as u64 * stride as u64 + (x0 + x) as u64;
        let output_index = y as u64 * width as u64 + x as u64;
        let coefficient = j2k_quantize_sample(
            load_f32_u64(samples, source_index),
            step_exponent,
            step_mantissa,
            range_bits,
            reversible,
        );
        store_i32(coefficients, output_index, coefficient);
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_compact_codeblocks(
        scratch: *const u8,
        compact: *mut u8,
        jobs: *const J2kHtEncodeCompactJob,
        job_count: u64,
    ) {
        let job_idx = thread::blockIdx_x();
        if job_idx as u64 >= job_count {
            return;
        }

        let job = load_job(jobs, job_idx);
        let mut idx = thread::threadIdx_x();
        let step = thread::blockDim_x();
        if (job.reserved & J2K_HT_COMPACT_ASSEMBLE_FLAG) != 0 {
            let mel_len = job.reserved & J2K_HT_COMPACT_LENGTH_MASK;
            let vlc_len = (job.reserved >> 15) & J2K_HT_COMPACT_LENGTH_MASK;
            let locator_bytes = mel_len + vlc_len;
            if locator_bytes > job.data_len {
                return;
            }
            let ms_len = job.data_len - locator_bytes;
            let vlc_start = J2K_HT_VLC_SIZE - vlc_len;
            while idx < job.data_len {
                let mut value = if idx < ms_len {
                    load_u8(scratch, (job.source_offset + idx) as u64)
                } else if idx < ms_len + mel_len {
                    load_u8(
                        scratch,
                        (job.source_offset + J2K_HT_MEL_OFFSET + idx - ms_len) as u64,
                    )
                } else {
                    load_u8(
                        scratch,
                        (job.source_offset + J2K_HT_VLC_OFFSET + vlc_start + idx - ms_len - mel_len)
                            as u64,
                    )
                };
                if job.data_len >= 2 {
                    if idx == job.data_len - 1 {
                        value = (locator_bytes >> 4) as u8;
                    } else if idx == job.data_len - 2 {
                        value = ((u32::from(value) & 0xf0) | (locator_bytes & 0x0f)) as u8;
                    }
                }
                store_u8(compact, (job.compact_offset + idx) as u64, value);
                idx += step;
            }
            return;
        }

        while idx < job.data_len {
            store_u8(
                compact,
                (job.compact_offset + idx) as u64,
                load_u8(scratch, (job.source_offset + idx) as u64),
            );
            idx += step;
        }
    }

    #[kernel]
    pub unsafe fn j2k_htj2k_packetize_cleanup(
        payload: *const u8,
        payload_len: u64,
        packets: *const J2kHtPacketJob,
        subbands: *const J2kHtPacketSubband,
        blocks: *const J2kHtPacketBlock,
        tag_states: *const J2kHtPacketSubbandTagState,
        tag_nodes: *const J2kHtPacketTagNodeState,
        tag_state_count: u64,
        tag_node_count: u64,
        out: *mut u8,
        statuses: *mut J2kHtPacketStatus,
        packet_count: u64,
    ) {
        static mut HEADER_RESULT: SharedArray<u32, 3> = SharedArray::UNINIT;

        let packet_idx = thread::blockIdx_x() as u64;
        if packet_idx >= packet_count {
            return;
        }

        let packet = load_job(packets, packet_idx as u32);
        let status = unsafe { statuses.add(packet_idx as usize) };
        let packet_out = unsafe { out.add(packet.output_offset as usize) };
        let header_result = unsafe { HEADER_RESULT.as_mut_ptr() };

        if thread::threadIdx_x() == 0 {
            let result = j2k_packet_build_header_serial(
                payload_len,
                packet,
                subbands,
                blocks,
                tag_states,
                tag_nodes,
                tag_state_count,
                tag_node_count,
                packet_out,
            );
            store_u32(header_result, 0, result.code);
            store_u32(header_result, 1, result.header_len);
            store_u32(header_result, 2, result.body_len);
            j2k_packet_status(status, result.code, result.detail, result.output_len);
        }
        thread::sync_threads();

        let shared_code = load_u32(header_result.cast_const(), 0);
        let shared_body_len = load_u32(header_result.cast_const(), 2);
        if shared_code != J2K_ENCODE_STATUS_OK || shared_body_len == 0 {
            return;
        }
        j2k_packet_copy_body_cooperative(
            payload,
            packet,
            subbands,
            blocks,
            packet_out,
            load_u32(header_result.cast_const(), 1),
            shared_body_len,
        );
    }
}

fn main() {}
