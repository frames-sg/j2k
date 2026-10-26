// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    abi::J2kHtPacketBlock, constants::J2K_PACKET_BLOCK_HEADER_WORDS, helpers::store_u8,
};

pub(crate) struct J2kPacketBitWriter {
    pub(crate) out: *mut u8,
    pub(crate) pos: u32,
    pub(crate) capacity: u32,
    pub(crate) buffer: u32,
    pub(crate) bits_in_buffer: u32,
    pub(crate) last_byte_was_ff: u32,
    pub(crate) failed: u32,
}

#[inline(always)]
pub(crate) fn j2k_packet_writer_init(out: *mut u8, capacity: u32) -> J2kPacketBitWriter {
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
pub(crate) fn j2k_packet_push_byte(writer: &mut J2kPacketBitWriter, byte: u8) {
    if writer.pos >= writer.capacity {
        writer.failed = 1;
        return;
    }
    store_u8(writer.out, writer.pos as u64, byte);
    writer.pos += 1;
    writer.last_byte_was_ff = if byte == 0xff { 1 } else { 0 };
}

#[inline(always)]
pub(crate) fn j2k_packet_flush_byte(writer: &mut J2kPacketBitWriter) {
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
pub(crate) fn j2k_packet_write_bit(writer: &mut J2kPacketBitWriter, bit: u32) {
    writer.buffer = (writer.buffer << 1) | (bit & 1);
    writer.bits_in_buffer += 1;
    let limit = if writer.last_byte_was_ff != 0 { 7 } else { 8 };
    if writer.bits_in_buffer >= limit {
        j2k_packet_flush_byte(writer);
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_write_bits(writer: &mut J2kPacketBitWriter, value: u32, mut count: u32) {
    while count > 0 {
        count -= 1;
        j2k_packet_write_bit(writer, (value >> count) & 1);
    }
}

/// Writes the `count` (at most 32) low bits of `value`, most significant
/// first, exactly as `count` calls of `j2k_packet_write_bit` would, a byte at a
/// time.
#[inline(always)]
pub(crate) fn j2k_packet_write_bits_bytewise(
    writer: &mut J2kPacketBitWriter,
    value: u32,
    mut count: u32,
) {
    while count > 0 {
        let limit = if writer.last_byte_was_ff != 0 { 7 } else { 8 };
        let take = (limit - writer.bits_in_buffer).min(count);
        count -= take;
        writer.buffer = (writer.buffer << take) | ((value >> count) & ((1_u32 << take) - 1));
        writer.bits_in_buffer += take;
        if writer.bits_in_buffer >= limit {
            j2k_packet_flush_byte(writer);
        }
    }
}

/// Destination for packet header bits, most significant bit first.
pub(crate) trait J2kPacketBitSink {
    fn put_bit(&mut self, bit: u32);
    fn put_bits(&mut self, value: u32, count: u32);
}

impl J2kPacketBitSink for J2kPacketBitWriter {
    #[inline(always)]
    fn put_bit(&mut self, bit: u32) {
        j2k_packet_write_bit(self, bit);
    }

    #[inline(always)]
    fn put_bits(&mut self, value: u32, count: u32) {
        j2k_packet_write_bits(self, value, count);
    }
}

/// One code block's header bits before bit stuffing, most significant first.
pub(crate) struct J2kPacketBlockBits {
    pub(crate) words: [u64; J2K_PACKET_BLOCK_HEADER_WORDS],
    pub(crate) len: u32,
    pub(crate) overflow: u32,
}

#[inline(always)]
pub(crate) fn j2k_packet_block_bits_new() -> J2kPacketBlockBits {
    J2kPacketBlockBits {
        words: [0; J2K_PACKET_BLOCK_HEADER_WORDS],
        len: 0,
        overflow: 0,
    }
}

impl J2kPacketBitSink for J2kPacketBlockBits {
    #[inline(always)]
    fn put_bit(&mut self, bit: u32) {
        self.put_bits(bit, 1);
    }

    /// `count` is at most 32.
    #[inline(always)]
    fn put_bits(&mut self, value: u32, count: u32) {
        let mut remaining = count;
        while remaining > 0 {
            if self.len as usize >= J2K_PACKET_BLOCK_HEADER_WORDS * 64 {
                self.overflow = 1;
                return;
            }
            let used = self.len % 64;
            let take = (64 - used).min(remaining);
            remaining -= take;
            let chunk = (u64::from(value) >> remaining) & ((1_u64 << take) - 1);
            self.words[(self.len / 64) as usize] |= chunk << (64 - used - take);
            self.len += take;
        }
    }
}

/// Writes `count` zero bits.
#[inline(always)]
pub(crate) fn j2k_packet_put_zeros(sink: &mut impl J2kPacketBitSink, mut count: u32) {
    while count > 0 {
        let take = count.min(32);
        sink.put_bits(0, take);
        count -= take;
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_finish(writer: &mut J2kPacketBitWriter) {
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
pub(crate) fn j2k_packet_encode_num_ht_passes(
    writer: &mut impl J2kPacketBitSink,
    num_passes: u32,
) {
    if num_passes == 1 {
        writer.put_bit(0);
    } else if num_passes == 2 {
        writer.put_bits(0b10, 2);
    } else if num_passes <= 5 {
        writer.put_bits(0b11, 2);
        writer.put_bits(num_passes - 3, 2);
    } else if num_passes <= 36 {
        writer.put_bits(0b11, 2);
        writer.put_bits(0b11, 2);
        writer.put_bits(num_passes - 6, 5);
    } else {
        writer.put_bits(0b11, 2);
        writer.put_bits(0b11, 2);
        writer.put_bits(31, 5);
        writer.put_bits(num_passes - 37, 7);
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_value_fits(value: u32, bits: u32) -> bool {
    bits >= 32 || value < (1_u32 << bits)
}

#[inline(always)]
pub(crate) fn j2k_packet_ht_length_bits(l_block: u32, num_passes: u32) -> u32 {
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
pub(crate) fn j2k_packet_encode_ht_segment_lengths(
    writer: &mut impl J2kPacketBitSink,
    block: J2kHtPacketBlock,
) {
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
        writer.put_bit(1);
        l_block += 1;
        cleanup_bits += 1;
    }
    writer.put_bit(0);
    writer.put_bits(cleanup_length, cleanup_bits);

    if block.num_coding_passes > 1 {
        writer.put_bits(block.refinement_length, l_block + refinement_extra_bits);
    }
}
