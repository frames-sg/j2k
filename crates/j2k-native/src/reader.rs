//! Combined byte and bit reader utilities.

use crate::error::{bail, MarkerError, Result};
use alloc::sync::Arc;
use core::fmt::Debug;
use core::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    cur_pos: usize,
    cancellation: Option<Arc<AtomicBool>>,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self::with_cancellation(data, None)
    }

    pub(crate) fn with_cancellation(data: &'a [u8], cancellation: Option<Arc<AtomicBool>>) -> Self {
        Self {
            data,
            cur_pos: 0,
            cancellation,
        }
    }

    pub(crate) fn child(&self, data: &'a [u8]) -> Self {
        Self::with_cancellation(data, self.cancellation.clone())
    }

    pub(crate) fn check_cancelled(&self) -> Result<()> {
        if self
            .cancellation
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(crate::DecodingError::Cancelled.into());
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn align(&mut self) {
        let bit_pos = self.bit_pos();

        if !bit_pos.is_multiple_of(8) {
            self.cur_pos += 8 - bit_pos;
        }
    }

    #[inline]
    pub(crate) fn at_end(&self) -> bool {
        self.byte_pos() >= self.data.len()
    }

    #[inline]
    pub(crate) fn jump_to_end(&mut self) {
        self.cur_pos = self.data.len() * 8;
    }

    #[inline]
    pub(crate) fn tail(&self) -> Option<&'a [u8]> {
        self.data.get(self.byte_pos()..)
    }

    #[inline]
    pub(crate) fn offset(&self) -> usize {
        self.byte_pos()
    }

    #[inline]
    pub(crate) fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        debug_assert_eq!(self.bit_pos(), 0);

        let bytes = self.peek_bytes(len)?;
        self.cur_pos += len * 8;

        Some(bytes)
    }

    #[inline]
    pub(crate) fn read_byte(&mut self) -> Option<u8> {
        debug_assert_eq!(self.bit_pos(), 0);

        let byte = self.peek_byte()?;
        self.cur_pos += 8;

        Some(byte)
    }

    #[inline]
    pub(crate) fn skip_bytes(&mut self, len: usize) -> Option<()> {
        self.read_bytes(len).map(|_| ())
    }

    #[inline]
    pub(crate) fn peek_bytes(&self, len: usize) -> Option<&'a [u8]> {
        let start = self.byte_pos();
        let end = start.checked_add(len)?;
        self.data.get(start..end)
    }

    #[inline]
    pub(crate) fn peek_byte(&self) -> Option<u8> {
        self.data.get(self.byte_pos()).copied()
    }

    #[inline]
    pub(crate) fn read_u16(&mut self) -> Option<u16> {
        let bytes = self.read_bytes(2)?;

        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    #[inline]
    pub(crate) fn read_u32(&mut self) -> Option<u32> {
        let bytes = self.read_bytes(4)?;

        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    #[inline]
    pub(crate) fn read_u64(&mut self) -> Option<u64> {
        let bytes = self.read_bytes(8)?;

        Some(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    #[inline]
    pub(crate) fn read_bit(&mut self) -> Option<u32> {
        let byte_pos = self.byte_pos();
        let byte = u32::from(*self.data.get(byte_pos)?);
        let shift = 7 - self.bit_pos();
        self.cur_pos += 1;
        Some((byte >> shift) & 1)
    }

    #[inline]
    pub(crate) fn byte_pos(&self) -> usize {
        self.cur_pos / 8
    }

    #[inline]
    pub(crate) fn bit_pos(&self) -> usize {
        self.cur_pos % 8
    }

    /// Like the normal `read_bits` method, but accounts for stuffing bits
    /// in addition.
    #[inline]
    pub(crate) fn read_bits_with_stuffing(&mut self, bit_size: u8) -> Option<u32> {
        // Whole runs of a byte's remaining bits at once. Reading the last bit
        // of a 0xFF byte also reads the stuff bit that starts the next byte,
        // which must be zero, as bit by bit reads do (B.10.1). Values keep
        // their low 32 bits, as shifting one bit at a time keeps them.
        let mut value = 0_u32;
        let mut remaining = usize::from(bit_size);
        while remaining > 0 {
            let byte = *self.data.get(self.byte_pos())?;
            let available = 8 - self.bit_pos();
            let take = available.min(remaining);
            let bits = (u32::from(byte) >> (available - take)) & ((1_u32 << take) - 1);
            value = (value << take) | bits;
            self.cur_pos += take;
            remaining -= take;
            if take == available && byte == 0xff {
                self.read_stuff_bit()?;
            }
        }

        Some(value)
    }

    /// Whether the next bit read ends a 0xFF byte, so a stuff bit follows:
    /// the one-bit-at-a-time rule `read_bits_with_stuffing` applies to runs.
    #[cfg(test)]
    pub(crate) fn needs_to_read_stuff_bit(&mut self) -> bool {
        // B.10.1: "If the value of the byte is 0xFF, the next byte includes an extra zero bit
        // stuffed into the MSB."
        self.bit_pos() == 7 && self.data.get(self.byte_pos()) == Some(&0xff)
    }

    #[inline]
    pub(crate) fn read_stuff_bit(&mut self) -> Option<()> {
        let stuff_bit = self.read_bit()?;

        if stuff_bit != 0 {
            return None;
        }

        Some(())
    }

    #[cfg(test)]
    pub(crate) fn peak_bits_with_stuffing(&mut self, bit_size: u8) -> Option<u32> {
        self.clone().read_bits_with_stuffing(bit_size)
    }

    #[inline]
    pub(crate) fn read_marker(&mut self) -> Result<u8> {
        self.check_cancelled()?;
        if self.peek_byte().ok_or(MarkerError::Invalid)? != 0xFF {
            bail!(MarkerError::Invalid);
        }

        let _marker_prefix = self.read_byte().ok_or(MarkerError::Invalid)?;
        Ok(self.read_byte().ok_or(MarkerError::Invalid)?)
    }

    #[inline]
    pub(crate) fn peek_marker(&mut self) -> Option<u8> {
        self.clone().read_marker().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;
    use crate::{DecodeError, MarkerError};

    #[test]
    fn stuffing_check_at_eof_returns_none_instead_of_panicking() {
        let mut reader = BitReader {
            data: &[],
            cur_pos: 7,
            cancellation: None,
        };

        assert_eq!(reader.read_bits_with_stuffing(1), None);
    }

    /// The one-bit-at-a-time reference for `read_bits_with_stuffing`.
    fn read_bits_one_at_a_time(reader: &mut BitReader<'_>, bit_size: u8) -> Option<u32> {
        let mut bit = 0_u32;
        for _ in 0..bit_size {
            let needs_stuff_bit = reader.needs_to_read_stuff_bit();
            bit = (bit << 1) | reader.read_bit()?;
            if needs_stuff_bit {
                reader.read_stuff_bit()?;
            }
        }
        Some(bit)
    }

    #[test]
    fn stuffed_reads_match_one_bit_at_a_time_reads() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..2000 {
            let len = usize::try_from(next() % 24).expect("small length");
            let data: alloc::vec::Vec<u8> = (0..len)
                .map(|_| match next() % 4 {
                    0 => 0xff,
                    1 => 0x7f,
                    _ => u8::try_from(next() >> 56).expect("top byte"),
                })
                .collect();
            let mut fast = BitReader::new(&data);
            let mut reference = BitReader::new(&data);
            loop {
                let bits = u8::try_from(next() % 40).expect("small width");
                let (a, b) = (
                    fast.read_bits_with_stuffing(bits),
                    read_bits_one_at_a_time(&mut reference, bits),
                );
                assert_eq!(a, b, "{data:02x?} width {bits}");
                if a.is_none() {
                    break;
                }
                assert_eq!(fast.cur_pos, reference.cur_pos);
            }
        }
    }

    #[test]
    fn truncated_marker_returns_error_instead_of_panicking() {
        let mut reader = BitReader::new(&[0xff]);

        assert_eq!(
            reader.read_marker(),
            Err(DecodeError::Marker(MarkerError::Invalid))
        );
    }
}
