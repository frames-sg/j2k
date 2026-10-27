// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packet headers built by a whole thread block. With every block coded for
//! the first time and fresh tag-tree state, a tag-tree node emits bits only on
//! its first visit, and those bits depend only on node values: an inclusion
//! node is first visited by the first block of its region in raster order, its
//! top-left one, and a zero bit-plane node by the first block with coding
//! passes. So each thread computes one block's bits on its own and thread 0
//! only appends them, which writes the bytes the serial builder writes.
//!
//! Locals that live in memory (arrays and structs passed by reference) are
//! declared before any branch: the device toolchain cannot allocate stack
//! inside loops.

use crate::{
    abi::{
        J2kHtPacketBlock, J2kHtPacketJob, J2kHtPacketStatus, J2kHtPacketSubband,
        J2kHtPacketSubbandTagState, J2kHtPacketTagNodeState,
    },
    constants::{J2K_ENCODE_STATUS_OK, J2K_PACKET_BLOCK_HEADER_WORDS, J2K_PACKET_HEADER_THREADS},
    helpers::{load_job, load_u8, load_u32, store_u8, store_u32},
    packet_writer::{
        J2kPacketBitSink, J2kPacketBitWriter, J2kPacketBlockBits, j2k_packet_block_bits_new,
        j2k_packet_encode_ht_segment_lengths, j2k_packet_encode_num_ht_passes, j2k_packet_finish,
        j2k_packet_put_zeros, j2k_packet_write_bit, j2k_packet_write_bits_bytewise,
        j2k_packet_writer_init,
    },
    packetization::{j2k_packet_block_error, j2k_packet_build_header_serial, j2k_packet_status},
    tag_tree::{J2kPacketTagTreeShape, j2k_packet_tag_tree_shape, j2k_packet_tag_tree_shape_new},
};
use cuda_device::thread;

/// Largest `l_block` whose segment lengths fit the 32-bit bit writes.
const J2K_PACKET_PARALLEL_MAX_L_BLOCK: u32 = 20;

/// Shared memory one thread block uses to build a packet header.
pub(crate) struct J2kPacketHeaderScratch {
    /// Per tag-tree node of the current sub-band: the minimum inclusion layer,
    /// the minimum zero bit-planes, and the first block with coding passes.
    pub(crate) inclusion: *mut u32,
    pub(crate) zero_bitplanes: *mut u32,
    pub(crate) first_coded: *mut u32,
    /// Each thread's block header bits. Before header emission, `bit_lens`
    /// holds per-thread eligibility flags instead of bit counts.
    pub(crate) bits: *mut u64,
    pub(crate) bit_lens: *mut u32,
    /// Each thread's body bytes during eligibility, then its header overflow flag.
    pub(crate) body_lens: *mut u64,
    /// Use the serial builder, any block has data, and the body length.
    pub(crate) flags: *mut u32,
}

const FLAG_SERIAL: u64 = 0;
const FLAG_ANY_DATA: u64 = 1;
const FLAG_BODY_LEN: u64 = 2;
const THREAD_SERIAL: u32 = 1;
const THREAD_ANY_DATA: u32 = 2;

/// Packet geometry and buffers shared by both header builders.
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketHeaderInput {
    pub(crate) payload_len: u64,
    pub(crate) packet: J2kHtPacketJob,
    pub(crate) subbands: *const J2kHtPacketSubband,
    pub(crate) blocks: *const J2kHtPacketBlock,
    pub(crate) tag_states: *const J2kHtPacketSubbandTagState,
    pub(crate) tag_nodes: *const J2kHtPacketTagNodeState,
    pub(crate) tag_state_count: u64,
    pub(crate) tag_node_count: u64,
    pub(crate) packet_out: *mut u8,
}

/// Builds the header of `input.packet` with the whole thread block and stores
/// its code, header length and body length in `result` and its status in
/// `status`. Packets the parallel builder does not cover use the serial
/// builder, so their bytes and errors are unchanged.
#[inline(always)]
pub(crate) fn j2k_packet_build_header(
    input: J2kPacketHeaderInput,
    scratch: &J2kPacketHeaderScratch,
    result: *mut u32,
    status: *mut J2kHtPacketStatus,
) {
    let built = thread::blockDim_x() <= J2K_PACKET_HEADER_THREADS
        && j2k_packet_build_header_parallel(input, scratch, result, status);
    if !built {
        // Finish all reads of the parallel layout before reusing its storage.
        thread::sync_threads();
        if thread::threadIdx_x() == 0 {
            let serial = j2k_packet_build_header_serial(
                input.payload_len,
                input.packet,
                input.subbands,
                input.blocks,
                input.tag_states,
                input.tag_nodes,
                input.tag_state_count,
                input.tag_node_count,
                input.packet_out,
                scratch.inclusion,
            );
            store_u32(result, 0, serial.code);
            store_u32(result, 1, serial.header_len);
            store_u32(result, 2, serial.body_len);
            j2k_packet_status(status, serial.code, serial.detail, serial.output_len);
        }
    }
}

/// Builds the header in parallel and returns `true`, or returns `false`, with
/// nothing reported, when the packet needs the serial builder. Every thread
/// returns the same value.
#[inline(always)]
fn j2k_packet_build_header_parallel(
    input: J2kPacketHeaderInput,
    scratch: &J2kPacketHeaderScratch,
    result: *mut u32,
    status: *mut J2kHtPacketStatus,
) -> bool {
    let mut writer = j2k_packet_writer_init(input.packet_out, input.packet.output_capacity);
    let mut shape = j2k_packet_tag_tree_shape_new();
    if thread::threadIdx_x() == 0 {
        store_u32(scratch.flags, FLAG_SERIAL, 0);
        store_u32(scratch.flags, FLAG_ANY_DATA, 0);
    }
    thread::sync_threads();
    if !j2k_packet_parallel_eligible(input, scratch) {
        return false;
    }
    let body_len = load_u32(scratch.flags.cast_const(), FLAG_BODY_LEN);

    if thread::threadIdx_x() == 0 {
        j2k_packet_write_bit(&mut writer, 1);
    }
    let mut subband_idx = 0_u32;
    while subband_idx < input.packet.subband_count {
        let subband = load_job(input.subbands, input.packet.subband_start + subband_idx);
        if j2k_packet_tag_tree_shape(&mut shape, subband.num_cbs_x, subband.num_cbs_y) {
            j2k_packet_build_subband_trees(input, subband, &shape, scratch);
            j2k_packet_append_subband_bits(input, subband, &shape, scratch, &mut writer);
        }
        subband_idx += 1;
    }

    if thread::threadIdx_x() == 0 {
        j2k_packet_finish_parallel(input, body_len, &mut writer, scratch, result, status);
    }
    thread::sync_threads();
    load_u32(scratch.flags.cast_const(), FLAG_SERIAL) == 0
}

/// Checks every block and the tag-tree state and sums the body bytes. Returns
/// `false` when the serial builder must build the header.
#[inline(always)]
fn j2k_packet_parallel_eligible(
    input: J2kPacketHeaderInput,
    scratch: &J2kPacketHeaderScratch,
) -> bool {
    let mut shape = j2k_packet_tag_tree_shape_new();
    let tid = thread::threadIdx_x();
    let threads = thread::blockDim_x();
    let mut serial = false;
    let mut any_data = false;
    let mut body_len = 0_u64;
    let mut subband_idx = 0_u32;
    while subband_idx < input.packet.subband_count {
        let subband_meta_idx = input.packet.subband_start + subband_idx;
        let subband = load_job(input.subbands, subband_meta_idx);
        if subband.num_cbs_x.checked_mul(subband.num_cbs_y) != Some(subband.block_count) {
            serial = true;
        } else if subband.block_count != 0 {
            serial |= !j2k_packet_fresh_tag_state(input, subband, subband_meta_idx, &mut shape);
        }
        let mut idx = tid;
        while idx < subband.block_count {
            let block = load_job(input.blocks, subband.block_start + idx);
            serial |= j2k_packet_block_error(block, input.payload_len) != 0
                || block.previously_included != 0
                || block.l_block > J2K_PACKET_PARALLEL_MAX_L_BLOCK
                || (block.num_coding_passes > 0 && block.inclusion_layer != input.packet.layer);
            if block.num_coding_passes > 0 {
                any_data = true;
                body_len += u64::from(block.data_len);
            }
            idx += threads;
        }
        subband_idx += 1;
    }
    // Each thread owns one slot; only thread 0 writes the combined flags.
    let flags = if serial { THREAD_SERIAL } else { 0 } | if any_data { THREAD_ANY_DATA } else { 0 };
    store_u32(scratch.bit_lens, u64::from(tid), flags);
    unsafe { *scratch.body_lens.add(tid as usize) = body_len };
    thread::sync_threads();

    if tid == 0 {
        let mut total = 0_u64;
        let mut flags = 0_u32;
        let mut thread_idx = 0_u32;
        while thread_idx < threads {
            total += unsafe { *scratch.body_lens.add(thread_idx as usize) };
            flags |= load_u32(scratch.bit_lens.cast_const(), u64::from(thread_idx));
            thread_idx += 1;
        }
        store_u32(
            scratch.flags,
            FLAG_SERIAL,
            u32::from(flags & THREAD_SERIAL != 0),
        );
        store_u32(
            scratch.flags,
            FLAG_ANY_DATA,
            u32::from(flags & THREAD_ANY_DATA != 0),
        );
        // An empty packet is one bit; the serial builder writes it.
        if load_u32(scratch.flags.cast_const(), FLAG_ANY_DATA) == 0 || total > u64::from(u32::MAX) {
            store_u32(scratch.flags, FLAG_SERIAL, 1);
        }
        store_u32(scratch.flags, FLAG_BODY_LEN, total as u32);
    }
    thread::sync_threads();
    load_u32(scratch.flags.cast_const(), FLAG_SERIAL) == 0
}

/// Whether the sub-band's tag trees start fresh, as at its first layer, and
/// its tag-tree state is one the serial builder accepts. The geometry checks
/// give the same answer on every thread; the node states are split across
/// threads and only those a thread read can make it answer `false`.
#[inline(always)]
fn j2k_packet_fresh_tag_state(
    input: J2kPacketHeaderInput,
    subband: J2kHtPacketSubband,
    subband_meta_idx: u32,
    shape: &mut J2kPacketTagTreeShape,
) -> bool {
    if !j2k_packet_tag_tree_shape(shape, subband.num_cbs_x, subband.num_cbs_y) {
        return false;
    }
    if input.tag_state_count == 0 {
        return true;
    }
    if u64::from(subband_meta_idx) >= input.tag_state_count {
        return false;
    }
    let state = load_job(input.tag_states, subband_meta_idx);
    if state.node_count != shape.total_nodes
        || u64::from(state.inclusion_node_start) + u64::from(state.node_count) > input.tag_node_count
        || u64::from(state.zero_bitplane_node_start) + u64::from(state.node_count)
            > input.tag_node_count
    {
        return false;
    }
    let mut node_idx = thread::threadIdx_x();
    while node_idx < state.node_count {
        let inclusion = load_job(input.tag_nodes, state.inclusion_node_start + node_idx);
        let zero_bitplanes = load_job(input.tag_nodes, state.zero_bitplane_node_start + node_idx);
        if (inclusion.current | inclusion.known | zero_bitplanes.current | zero_bitplanes.known) != 0
        {
            return false;
        }
        node_idx += thread::blockDim_x();
    }
    true
}

/// Fills the shared tag trees of `subband`: leaves first, then each level as
/// the minimum of its children.
#[inline(always)]
fn j2k_packet_build_subband_trees(
    input: J2kPacketHeaderInput,
    subband: J2kHtPacketSubband,
    shape: &J2kPacketTagTreeShape,
    scratch: &J2kPacketHeaderScratch,
) {
    let tid = thread::threadIdx_x();
    let threads = thread::blockDim_x();
    let mut idx = tid;
    while idx < subband.block_count {
        let block = load_job(input.blocks, subband.block_start + idx);
        let leaf = u64::from(idx);
        store_u32(scratch.inclusion, leaf, block.inclusion_layer);
        store_u32(scratch.zero_bitplanes, leaf, block.num_zero_bitplanes);
        let first_coded = if block.num_coding_passes > 0 { idx } else { u32::MAX };
        store_u32(scratch.first_coded, leaf, first_coded);
        idx += threads;
    }
    thread::sync_threads();

    let mut level = 1_u32;
    while level < shape.levels {
        let prev_w = shape.widths[(level - 1) as usize];
        let prev_h = shape.heights[(level - 1) as usize];
        let prev_offset = shape.offsets[(level - 1) as usize];
        let curr_w = shape.widths[level as usize];
        let nodes = curr_w * shape.heights[level as usize];
        let mut node = tid;
        while node < nodes {
            let child_x0 = (node % curr_w) << 1;
            let child_y0 = (node / curr_w) << 1;
            let child_x1 = (child_x0 + 2).min(prev_w);
            let child_y1 = (child_y0 + 2).min(prev_h);
            let mut inclusion = u32::MAX;
            let mut zero_bitplanes = u32::MAX;
            let mut first_coded = u32::MAX;
            let mut y = child_y0;
            while y < child_y1 {
                let mut x = child_x0;
                while x < child_x1 {
                    let child = u64::from(prev_offset + y * prev_w + x);
                    inclusion = inclusion.min(load_u32(scratch.inclusion.cast_const(), child));
                    zero_bitplanes =
                        zero_bitplanes.min(load_u32(scratch.zero_bitplanes.cast_const(), child));
                    first_coded = first_coded.min(load_u32(scratch.first_coded.cast_const(), child));
                    x += 1;
                }
                y += 1;
            }
            let slot = u64::from(shape.offsets[level as usize] + node);
            store_u32(scratch.inclusion, slot, inclusion);
            store_u32(scratch.zero_bitplanes, slot, zero_bitplanes);
            store_u32(scratch.first_coded, slot, first_coded);
            node += threads;
        }
        thread::sync_threads();
        level += 1;
    }
}

/// Computes the header bits of `subband`'s blocks a thread block's worth at a
/// time, and has thread 0 append them in raster order.
#[inline(always)]
fn j2k_packet_append_subband_bits(
    input: J2kPacketHeaderInput,
    subband: J2kHtPacketSubband,
    shape: &J2kPacketTagTreeShape,
    scratch: &J2kPacketHeaderScratch,
    writer: &mut J2kPacketBitWriter,
) {
    let mut bits = j2k_packet_block_bits_new();
    let tid = thread::threadIdx_x();
    let threads = thread::blockDim_x();
    let mut chunk_start = 0_u32;
    while chunk_start < subband.block_count {
        let idx = chunk_start + tid;
        if idx < subband.block_count {
            let block = load_job(input.blocks, subband.block_start + idx);
            j2k_packet_block_header_bits(&mut bits, input.packet, block, idx, subband, shape, scratch);
            // Eligibility already consumed the body lengths. Reuse those
            // disjoint slots so overflow has one writer per thread too.
            unsafe { *scratch.body_lens.add(tid as usize) = u64::from(bits.overflow) };
            let mut word = 0_usize;
            while word < J2K_PACKET_BLOCK_HEADER_WORDS {
                unsafe {
                    *scratch.bits.add(tid as usize * J2K_PACKET_BLOCK_HEADER_WORDS + word) =
                        bits.words[word];
                }
                word += 1;
            }
            store_u32(scratch.bit_lens, u64::from(tid), bits.len);
        }
        thread::sync_threads();

        if tid == 0 {
            let count = (subband.block_count - chunk_start).min(threads);
            let mut slot = 0_u32;
            while slot < count {
                if unsafe { *scratch.body_lens.add(slot as usize) } != 0 {
                    store_u32(scratch.flags, FLAG_SERIAL, 1);
                }
                j2k_packet_append_block_bits(writer, scratch, slot);
                slot += 1;
            }
        }
        thread::sync_threads();
        chunk_start += threads;
    }
}

/// Sets `bits` to one block's header bits: its inclusion and zero bit-plane
/// tag-tree bits, then its pass count and segment lengths.
#[inline(always)]
fn j2k_packet_block_header_bits(
    bits: &mut J2kPacketBlockBits,
    packet: J2kHtPacketJob,
    block: J2kHtPacketBlock,
    idx: u32,
    subband: J2kHtPacketSubband,
    shape: &J2kPacketTagTreeShape,
    scratch: &J2kPacketHeaderScratch,
) {
    let x = idx % subband.num_cbs_x;
    let y = idx / subband.num_cbs_x;
    bits.words = [0; J2K_PACKET_BLOCK_HEADER_WORDS];
    bits.len = 0;
    bits.overflow = 0;

    // Inclusion, against threshold `layer + 1`; a node's value stays at its
    // threshold-capped value once coded.
    let threshold = packet.layer + 1;
    let mut parent_current = 0_u32;
    let mut level = shape.levels;
    while level > 0 {
        level -= 1;
        let (node_x, node_y) = (x >> level, y >> level);
        let node = shape.offsets[level as usize] + node_y * shape.widths[level as usize] + node_x;
        let value = load_u32(scratch.inclusion.cast_const(), u64::from(node));
        let target = value.min(threshold);
        if node_x << level == x && node_y << level == y {
            j2k_packet_put_zeros(bits, target.saturating_sub(parent_current));
            if value < threshold {
                bits.put_bit(1);
            }
        }
        parent_current = target;
    }
    if block.num_coding_passes == 0 {
        return;
    }

    // Zero bit-planes, each node coded in full on its first visit.
    let mut parent_value = 0_u32;
    level = shape.levels;
    while level > 0 {
        level -= 1;
        let node = shape.offsets[level as usize]
            + (y >> level) * shape.widths[level as usize]
            + (x >> level);
        let value = load_u32(scratch.zero_bitplanes.cast_const(), u64::from(node));
        if load_u32(scratch.first_coded.cast_const(), u64::from(node)) == idx {
            j2k_packet_put_zeros(bits, value - parent_value);
            bits.put_bit(1);
        }
        parent_value = value;
    }
    j2k_packet_encode_num_ht_passes(bits, block.num_coding_passes);
    j2k_packet_encode_ht_segment_lengths(bits, block);
}

/// Appends the bits thread `slot` computed to `writer`.
#[inline(always)]
fn j2k_packet_append_block_bits(
    writer: &mut J2kPacketBitWriter,
    scratch: &J2kPacketHeaderScratch,
    slot: u32,
) {
    let mut remaining = load_u32(scratch.bit_lens.cast_const(), u64::from(slot));
    let mut word = 0_usize;
    while remaining > 0 {
        let bits = unsafe { *scratch.bits.add(slot as usize * J2K_PACKET_BLOCK_HEADER_WORDS + word) };
        let count = remaining.min(64);
        let high = count.min(32);
        j2k_packet_write_bits_bytewise(writer, ((bits >> 32) as u32) >> (32 - high), high);
        if count > 32 {
            j2k_packet_write_bits_bytewise(writer, (bits as u32) >> (64 - count), count - 32);
        }
        remaining -= count;
        word += 1;
    }
}

/// Thread 0: flushes the header, then records the result, or asks for the
/// serial builder where it would report an error.
#[inline(always)]
fn j2k_packet_finish_parallel(
    input: J2kPacketHeaderInput,
    body_len: u32,
    writer: &mut J2kPacketBitWriter,
    scratch: &J2kPacketHeaderScratch,
    result: *mut u32,
    status: *mut J2kHtPacketStatus,
) {
    j2k_packet_finish(writer);
    let capacity = input.packet.output_capacity;
    let mut serial = writer.failed != 0;
    if !serial && writer.pos > 0 && load_u8(input.packet_out.cast_const(), u64::from(writer.pos - 1)) == 0xff
    {
        if writer.pos >= capacity {
            serial = true;
        } else {
            store_u8(input.packet_out, u64::from(writer.pos), 0);
            writer.pos += 1;
        }
    }
    serial |= load_u32(scratch.flags.cast_const(), FLAG_SERIAL) != 0
        || u64::from(writer.pos) + u64::from(body_len) > u64::from(capacity);
    if serial {
        store_u32(scratch.flags, FLAG_SERIAL, 1);
        return;
    }
    let header_len = writer.pos;
    store_u32(result, 0, J2K_ENCODE_STATUS_OK);
    store_u32(result, 1, header_len);
    store_u32(result, 2, body_len);
    j2k_packet_status(status, J2K_ENCODE_STATUS_OK, 0, header_len + body_len);
}
