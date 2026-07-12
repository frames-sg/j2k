//! Bit-level writer with JPEG 2000 byte-stuffing support.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{
    try_reserve_untracked_bounded, try_untracked_vec, BudgetedVec, EncodeAllocationLedger,
};
use crate::{EncodeError, EncodeResult};

/// Fallible bit sink used by checked packet-header encoders.
pub(crate) trait FallibleBitWriter {
    fn try_write_bit(&mut self, bit: u32) -> EncodeResult<()>;
    fn try_write_bits(&mut self, value: u32, count: u8) -> EncodeResult<()>;
}

#[derive(Debug, Clone, Default)]
struct BitAccumulator {
    /// Current partial byte being assembled.
    buffer: u32,
    /// Number of valid bits in `buffer` (MSB-first).
    bits_in_buffer: u8,
    /// Whether the last completed byte was 0xFF (triggers bit-stuffing).
    last_byte_was_ff: bool,
}

impl BitAccumulator {
    #[inline]
    fn write_bit(&mut self, bit: u32) -> Option<u8> {
        self.buffer = (self.buffer << 1) | (bit & 1);
        self.bits_in_buffer += 1;

        let limit = if self.last_byte_was_ff { 7 } else { 8 };
        (self.bits_in_buffer >= limit).then(|| self.take_full_byte(limit))
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the bit-buffer invariant limits the extracted value to one byte"
    )]
    fn take_full_byte(&mut self, limit: u8) -> u8 {
        let byte = (self.buffer >> (self.bits_in_buffer - limit)) as u8;
        self.last_byte_was_ff = byte == 0xFF;
        self.bits_in_buffer -= limit;
        self.buffer &= (1 << self.bits_in_buffer) - 1;
        byte
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the bit-buffer invariant limits the shifted value to one byte"
    )]
    fn flush(&mut self) -> Option<u8> {
        if self.bits_in_buffer == 0 {
            return None;
        }
        let limit = if self.last_byte_was_ff { 7 } else { 8 };
        let shift = limit - self.bits_in_buffer;
        let byte = (self.buffer << shift) as u8;
        self.last_byte_was_ff = byte == 0xFF;
        self.buffer = 0;
        self.bits_in_buffer = 0;
        Some(byte)
    }

    #[cfg(test)]
    fn reset_marker_stuffing(&mut self) {
        self.last_byte_was_ff = false;
    }
}

/// A writer that outputs bits to a byte buffer, supporting JPEG 2000 byte-stuffing.
///
/// After writing a 0xFF byte, the next byte is restricted to 7 bits
/// (a 0-bit is inserted before the first data bit), preventing false marker codes.
#[derive(Debug)]
pub(crate) struct BitWriter {
    data: Vec<u8>,
    bits: BitAccumulator,
    byte_limit: Option<usize>,
    allocation_error: Option<EncodeError>,
}

impl BitWriter {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::try_with_capacity(0).expect("test bit-writer allocation")
    }

    pub(crate) fn try_with_capacity(capacity: usize) -> EncodeResult<Self> {
        Ok(Self {
            data: try_untracked_vec(capacity, "raw Tier-1 segment output")?,
            bits: BitAccumulator::default(),
            byte_limit: None,
            allocation_error: None,
        })
    }

    pub(crate) fn try_with_byte_limit(payload_bytes: usize) -> EncodeResult<Self> {
        let mut writer = Self::try_with_capacity(payload_bytes.min(256))?;
        writer.byte_limit = Some(payload_bytes);
        Ok(writer)
    }

    /// Write a single bit (0 or 1).
    #[inline]
    pub(crate) fn write_bit(&mut self, bit: u32) {
        if self.allocation_error.is_some() {
            return;
        }
        if let Some(byte) = self.bits.write_bit(bit) {
            self.try_push(byte);
        }
    }

    /// Write `count` bits from `value` (MSB first).
    #[cfg(test)]
    #[inline]
    pub(crate) fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Write a big-endian u16 directly to the output buffer.
    /// Flushes any pending bits first.
    #[cfg(test)]
    pub(crate) fn write_u16_raw(&mut self, value: u16) {
        self.flush();
        self.data.extend_from_slice(&value.to_be_bytes());
        self.bits.reset_marker_stuffing();
    }

    /// Write a JPEG 2000 marker (0xFF followed by the marker code).
    #[cfg(test)]
    pub(crate) fn write_marker(&mut self, marker: u8) {
        self.flush();
        self.data.push(0xFF);
        self.data.push(marker);
        self.bits.reset_marker_stuffing();
    }

    /// Flush the partial byte, padding with zero bits.
    pub(crate) fn flush(&mut self) {
        if self.allocation_error.is_some() {
            return;
        }
        if let Some(byte) = self.bits.flush() {
            self.try_push(byte);
        }
    }

    /// Flush and return the assembled byte buffer.
    #[cfg(test)]
    pub(crate) fn finish(self) -> Vec<u8> {
        self.finish_checked().expect("legacy bit-writer output")
    }

    pub(crate) fn finish_checked(mut self) -> EncodeResult<Vec<u8>> {
        self.flush();
        if let Some(error) = self.allocation_error.take() {
            return Err(error);
        }
        Ok(self.data)
    }

    fn try_push(&mut self, byte: u8) {
        if self
            .byte_limit
            .is_some_and(|limit| self.data.len() >= limit)
        {
            self.allocation_error = Some(EncodeError::InternalInvariant {
                what: "raw Tier-1 writer exceeded its checked payload plan",
            });
            return;
        }
        let byte_limit = self.byte_limit.unwrap_or(usize::MAX);
        if let Err(error) = try_reserve_untracked_bounded(
            &mut self.data,
            1,
            byte_limit,
            "raw Tier-1 segment output",
        ) {
            self.allocation_error = Some(error);
            return;
        }
        self.data.push(byte);
    }
}

#[cfg(test)]
impl FallibleBitWriter for BitWriter {
    #[inline]
    fn try_write_bit(&mut self, bit: u32) -> EncodeResult<()> {
        self.write_bit(bit);
        Ok(())
    }

    #[inline]
    fn try_write_bits(&mut self, value: u32, count: u8) -> EncodeResult<()> {
        self.write_bits(value, count);
        Ok(())
    }
}

/// Packet-header writer backed by one preplanned, fallibly allocated buffer.
///
/// Unlike [`BitWriter`], this type never grows its vector. Every write returns
/// a typed error if the packet header exceeds its checked plan.
#[derive(Debug)]
pub(crate) struct CheckedBitWriter<'a> {
    data: BudgetedVec<'a, u8>,
    bits: BitAccumulator,
    planned_bytes: usize,
}

impl<'a> CheckedBitWriter<'a> {
    pub(crate) fn try_with_capacity(
        allocations: &'a EncodeAllocationLedger,
        planned_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<Self> {
        let data = allocations.try_vec_with_capacity(planned_bytes, what)?;
        Ok(Self {
            data,
            bits: BitAccumulator::default(),
            planned_bytes,
        })
    }

    #[inline]
    pub(crate) fn try_write_bit(&mut self, bit: u32) -> EncodeResult<()> {
        if let Some(byte) = self.bits.write_bit(bit) {
            self.try_push(byte)?;
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn try_write_bits(&mut self, value: u32, count: u8) -> EncodeResult<()> {
        for index in (0..count).rev() {
            self.try_write_bit((value >> index) & 1)?;
        }
        Ok(())
    }

    pub(crate) fn try_finish(mut self) -> EncodeResult<BudgetedVec<'a, u8>> {
        if let Some(byte) = self.bits.flush() {
            self.try_push(byte)?;
        }
        Ok(self.data)
    }

    fn try_push(&mut self, byte: u8) -> EncodeResult<()> {
        if self.data.len() >= self.planned_bytes || self.data.len() >= self.data.capacity() {
            return Err(EncodeError::InternalInvariant {
                what: "packet header exceeded its checked bit-writer plan",
            });
        }
        self.data.try_push(byte)
    }
}

impl FallibleBitWriter for CheckedBitWriter<'_> {
    #[inline]
    fn try_write_bit(&mut self, bit: u32) -> EncodeResult<()> {
        Self::try_write_bit(self, bit)
    }

    #[inline]
    fn try_write_bits(&mut self, value: u32, count: u8) -> EncodeResult<()> {
        Self::try_write_bits(self, value, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_write_bits_basic() {
        let mut w = BitWriter::new();
        w.write_bits(0b1011_0011, 8);
        let data = w.finish();
        assert_eq!(data, vec![0b1011_0011]);
    }

    #[test]
    fn test_write_bits_partial() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b11001, 5);
        let data = w.finish();
        assert_eq!(data, vec![0b1011_1001]);
    }

    #[test]
    fn test_byte_stuffing() {
        let mut w = BitWriter::new();
        // Write 0xFF
        w.write_bits(0xFF, 8);
        // Next byte should be limited to 7 bits due to stuffing
        w.write_bits(0b101_0101, 7);
        let data = w.finish();
        assert_eq!(data[0], 0xFF);
        // After 0xFF, the 7 bits are written with a leading 0 (stuffed bit)
        assert_eq!(data[1], 0b101_0101);
    }

    #[test]
    fn test_marker_write() {
        let mut w = BitWriter::new();
        w.write_marker(0x51); // SIZ marker
        let data = w.finish();
        assert_eq!(data, vec![0xFF, 0x51]);
    }

    #[test]
    fn test_round_trip_u16() {
        let mut w = BitWriter::new();
        w.write_u16_raw(0x1234);
        let data = w.finish();
        assert_eq!(data, vec![0x12, 0x34]);
    }
}
