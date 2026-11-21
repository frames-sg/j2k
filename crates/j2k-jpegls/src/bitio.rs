// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-coded bit streams with JPEG-LS bit stuffing (T.87 A.1): after a
//! `0xFF` byte, the next byte carries a zero bit and seven data bits.

/// Why a bit read failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadError {
    /// The segment ended before the bits were available.
    Truncated,
    /// A unary prefix was longer than the code allows.
    InvalidCode,
}

/// Reads bits from one entropy-coded segment: the bytes of a scan or restart
/// interval, up to (not including) the next marker.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// The low `bits` bits are unread, most significant first.
    cache: u64,
    bits: u32,
    last_ff: bool,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            cache: 0,
            bits: 0,
            last_ff: false,
        }
    }

    fn refill(&mut self) {
        while self.bits <= 55 {
            let Some(&byte) = self.data.get(self.pos) else {
                break;
            };
            self.pos += 1;
            if self.last_ff {
                self.cache = (self.cache << 7) | u64::from(byte & 0x7F);
                self.bits += 7;
            } else {
                self.cache = (self.cache << 8) | u64::from(byte);
                self.bits += 8;
            }
            self.last_ff = byte == 0xFF;
        }
    }

    fn consume(&mut self, count: u32) {
        self.bits -= count;
        self.cache &= low_mask(self.bits);
    }

    /// Read `count` bits (at most 32), most significant first.
    pub(crate) fn read_bits(&mut self, count: u32) -> Result<u32, ReadError> {
        if count == 0 {
            return Ok(0);
        }
        if self.bits < count {
            self.refill();
            if self.bits < count {
                return Err(ReadError::Truncated);
            }
        }
        let value = (self.cache >> (self.bits - count)) & low_mask(count);
        self.consume(count);
        u32::try_from(value).map_err(|_| ReadError::InvalidCode)
    }

    pub(crate) fn read_bit(&mut self) -> Result<bool, ReadError> {
        Ok(self.read_bits(1)? == 1)
    }

    /// Count zero bits up to and including the next one bit. Fails if more
    /// than `max_zeros` zeros precede it.
    pub(crate) fn read_unary(&mut self, max_zeros: u32) -> Result<u32, ReadError> {
        let mut zeros = 0_u32;
        loop {
            if self.bits == 0 {
                self.refill();
                if self.bits == 0 {
                    return Err(ReadError::Truncated);
                }
            }
            let leading = (self.cache << (64 - self.bits)).leading_zeros();
            if leading < self.bits {
                zeros += leading;
                if zeros > max_zeros {
                    return Err(ReadError::InvalidCode);
                }
                self.consume(leading + 1);
                return Ok(zeros);
            }
            zeros += self.bits;
            self.consume(self.bits);
            if zeros > max_zeros {
                return Err(ReadError::InvalidCode);
            }
        }
    }

    /// Unread bits left in the segment, counting a byte after `0xFF` as seven
    /// bits. A complete segment leaves only its final padding, under 8 bits.
    pub(crate) fn remaining_bits(&self) -> usize {
        let mut remaining = self.bits as usize;
        let mut last_ff = self.last_ff;
        for &byte in self.data.get(self.pos..).unwrap_or_default() {
            remaining += if last_ff { 7 } else { 8 };
            last_ff = byte == 0xFF;
        }
        remaining
    }
}

/// Writes bits with JPEG-LS bit stuffing.
pub(crate) struct BitWriter {
    out: Vec<u8>,
    /// The low `bits` bits are pending, most significant first.
    pending: u64,
    bits: u32,
    last_ff: bool,
}

impl BitWriter {
    /// Append entropy-coded data to `out`.
    pub(crate) const fn new(out: Vec<u8>) -> Self {
        Self {
            out,
            pending: 0,
            bits: 0,
            last_ff: false,
        }
    }

    const fn capacity(&self) -> u32 {
        if self.last_ff {
            7
        } else {
            8
        }
    }

    /// Append the low `count` bits of `value` (`count` at most 32).
    pub(crate) fn put(&mut self, value: u32, count: u32) {
        if count == 0 {
            return;
        }
        self.pending = (self.pending << count) | (u64::from(value) & low_mask(count));
        self.bits += count;
        loop {
            let capacity = self.capacity();
            if self.bits < capacity {
                break;
            }
            self.bits -= capacity;
            let byte = (self.pending >> self.bits) & low_mask(capacity);
            self.pending &= low_mask(self.bits);
            self.emit(byte);
        }
    }

    pub(crate) fn put_zeros(&mut self, mut count: u32) {
        while count > 0 {
            let chunk = count.min(32);
            self.put(0, chunk);
            count -= chunk;
        }
    }

    pub(crate) fn put_bit(&mut self, bit: bool) {
        self.put(u32::from(bit), 1);
    }

    fn emit(&mut self, byte: u64) {
        let byte = u8::try_from(byte).unwrap_or(u8::MAX);
        self.out.push(byte);
        self.last_ff = byte == 0xFF;
    }

    /// End a segment: pad the last byte with zero bits, and follow a final
    /// `0xFF` with a zero byte so no marker can be misread.
    pub(crate) fn finish_segment(&mut self) {
        if self.bits > 0 {
            let capacity = self.capacity();
            let byte = (self.pending << (capacity - self.bits)) & low_mask(capacity);
            self.emit(byte);
        }
        if self.last_ff {
            self.emit(0);
        }
        self.pending = 0;
        self.bits = 0;
        self.last_ff = false;
    }

    /// The output so far. Call [`Self::finish_segment`] first.
    pub(crate) const fn bytes_mut(&mut self) -> &mut Vec<u8> {
        &mut self.out
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.out
    }
}

const fn low_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter, ReadError};

    #[test]
    fn writer_stuffs_a_zero_bit_after_ff() {
        let mut writer = BitWriter::new(Vec::new());
        writer.put(0xFF, 8);
        writer.put(0x7F, 7);
        writer.put(0b1, 1);
        writer.finish_segment();
        // FF, then 0 + 1111111, then 1 padded with zeros.
        assert_eq!(writer.into_bytes(), [0xFF, 0x7F, 0x80]);
    }

    #[test]
    fn writer_follows_a_final_ff_with_a_zero_byte() {
        let mut writer = BitWriter::new(Vec::new());
        writer.put(0xFF, 8);
        writer.finish_segment();
        assert_eq!(writer.into_bytes(), [0xFF, 0x00]);
    }

    #[test]
    fn reader_inverts_writer() {
        let mut writer = BitWriter::new(Vec::new());
        let values: Vec<(u32, u32)> = (0..200_u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> (i % 23), 1 + i % 32))
            .collect();
        for &(value, count) in &values {
            writer.put(value, count);
        }
        writer.put_zeros(40);
        writer.put_bit(true);
        writer.finish_segment();
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        for &(value, count) in &values {
            let mask = if count == 32 {
                u32::MAX
            } else {
                (1 << count) - 1
            };
            assert_eq!(reader.read_bits(count), Ok(value & mask));
        }
        assert_eq!(reader.read_unary(64), Ok(40));
        assert!(reader.remaining_bits() < 8);
    }

    #[test]
    fn reader_reports_truncation_and_overlong_codes() {
        let mut reader = BitReader::new(&[0x00, 0x00]);
        assert_eq!(reader.read_unary(8), Err(ReadError::InvalidCode));
        let mut reader = BitReader::new(&[0x00]);
        assert_eq!(reader.read_unary(64), Err(ReadError::Truncated));
        let mut reader = BitReader::new(&[0xAB]);
        assert_eq!(reader.read_bits(9), Err(ReadError::Truncated));
    }
}
