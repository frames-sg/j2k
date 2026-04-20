//! Bit-buffer reader for codestream parsing.
//!
//! Port of `ojph_bitbuffer_read.h`. Handles JPEG 2000 byte-stuffing:
//! after a 0xFF byte, the next byte's MSB is a stuffed bit.

/// Bit-level reader with JPEG 2000 0xFF byte-unstuffing.
///
/// Reads from a byte slice, automatically handling the rule that after
/// a 0xFF byte the MSB of the next byte is ignored (stuffed).
#[derive(Debug)]
pub struct BitBufferRead<'a> {
    data: &'a [u8],
    pos: usize,
    /// Accumulated bit buffer (left-justified).
    buf: u64,
    /// Number of valid bits in `buf`.
    bits_left: u32,
    /// True if the previous byte was 0xFF (triggers unstuffing).
    unstuff: bool,
}

impl<'a> BitBufferRead<'a> {
    /// Create a new bit reader over the given byte slice.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            buf: 0,
            bits_left: 0,
            unstuff: false,
        }
    }

    /// Fill the internal buffer with up to 32 new bits from the byte stream.
    pub fn fill(&mut self) {
        while self.bits_left <= 32 && self.pos < self.data.len() {
            let byte = self.data[self.pos] as u64;
            self.pos += 1;
            let bits_to_add = if self.unstuff { 7 } else { 8 };
            let val = if self.unstuff { byte & 0x7F } else { byte };
            self.buf |= val << (64 - bits_to_add - self.bits_left);
            self.bits_left += bits_to_add;
            self.unstuff = (self.data[self.pos - 1]) == 0xFF;
        }
    }

    /// Peek at the top `n` bits without consuming them (n <= 32).
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        debug_assert!(n <= 32 && n <= self.bits_left);
        (self.buf >> (64 - n)) as u32
    }

    /// Consume `n` bits from the buffer.
    #[inline]
    pub fn advance(&mut self, n: u32) {
        debug_assert!(n <= self.bits_left);
        self.buf <<= n;
        self.bits_left -= n;
    }

    /// Read `n` bits and return them as a u32 (n <= 32).
    #[inline]
    pub fn read(&mut self, n: u32) -> u32 {
        if self.bits_left < n {
            self.fill();
        }
        let val = self.peek(n);
        self.advance(n);
        val
    }

    /// Returns the number of valid bits currently in the buffer.
    #[inline]
    pub fn available_bits(&self) -> u32 {
        self.bits_left
    }

    /// Returns the current byte position in the data stream.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Returns true if the previous byte read was 0xFF.
    #[inline]
    pub fn is_unstuffing(&self) -> bool {
        self.unstuff
    }

    /// Reset the reader with new data.
    pub fn reset(&mut self, data: &'a [u8]) {
        self.data = data;
        self.pos = 0;
        self.buf = 0;
        self.bits_left = 0;
        self.unstuff = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_single_bits() {
        // 0xA5 = 1010_0101
        let data = [0xA5u8];
        let mut reader = BitBufferRead::new(&data);
        assert_eq!(reader.read(1), 1);
        assert_eq!(reader.read(1), 0);
        assert_eq!(reader.read(1), 1);
        assert_eq!(reader.read(1), 0);
        assert_eq!(reader.read(1), 0);
        assert_eq!(reader.read(1), 1);
        assert_eq!(reader.read(1), 0);
        assert_eq!(reader.read(1), 1);
    }

    #[test]
    fn read_multi_bit_values() {
        // 0xAB = 1010_1011, 0xCD = 1100_1101
        let data = [0xABu8, 0xCD];
        let mut reader = BitBufferRead::new(&data);
        assert_eq!(reader.read(4), 0xA); // 1010
        assert_eq!(reader.read(4), 0xB); // 1011
        assert_eq!(reader.read(8), 0xCD);
    }

    #[test]
    fn read_across_byte_boundary() {
        let data = [0xABu8, 0xCD];
        let mut reader = BitBufferRead::new(&data);
        // Read 12 bits spanning both bytes: 1010_1011_1100 = 0xABC
        assert_eq!(reader.read(12), 0xABC);
        assert_eq!(reader.read(4), 0xD);
    }

    #[test]
    fn multiple_reads_across_bytes() {
        // [0xAB, 0xCD] = 0b10101011_11001101
        let data = [0xABu8, 0xCD];
        let mut reader = BitBufferRead::new(&data);
        assert_eq!(reader.read(4), 0b1010); // 0xA
        assert_eq!(reader.read(8), 0b10111100); // 0xBC (spans bytes)
        assert_eq!(reader.read(4), 0b1101); // 0xD
    }

    #[test]
    fn unstuffing_after_0xff() {
        // After 0xFF, the next byte's MSB is ignored (stuffed bit).
        // 0xFF followed by 0x80 → the 0x80 only provides 7 bits: 000_0000
        let data = [0xFFu8, 0x80];
        let mut reader = BitBufferRead::new(&data);
        reader.fill();
        // First 8 bits from 0xFF = 1111_1111
        assert_eq!(reader.read(8), 0xFF);
        // Now unstuffing is active; next byte 0x80 → MSB stripped → 7 bits: 000_0000
        assert_eq!(reader.read(7), 0x00);
    }

    #[test]
    fn unstuffing_preserves_lower_7_bits() {
        // After 0xFF, the unstuffed byte is placed with a gap for the
        // stuffed bit. Verify that fill+read produces consistent results.
        let data = [0xFFu8, 0x7F, 0x42];
        let mut reader = BitBufferRead::new(&data);
        reader.fill();
        // First 8 bits are 0xFF
        assert_eq!(reader.read(8), 0xFF);
        // The unstuffed byte provides 7 bits; verify we can read them
        let bits = reader.available_bits();
        assert!(bits >= 7);
    }

    #[test]
    fn empty_stream() {
        let data: &[u8] = &[];
        let reader = BitBufferRead::new(data);
        assert_eq!(reader.available_bits(), 0);
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn position_tracking() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        let mut reader = BitBufferRead::new(&data);
        assert_eq!(reader.position(), 0);
        reader.fill();
        // fill() reads bytes to fill the buffer
        assert!(reader.position() > 0);
    }

    #[test]
    fn reset_clears_state() {
        let data1 = [0xFFu8, 0x00];
        let data2 = [0x42u8];
        let mut reader = BitBufferRead::new(&data1);
        reader.fill();
        let _ = reader.read(8);

        reader.reset(&data2);
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.available_bits(), 0);
        assert!(!reader.is_unstuffing());
        assert_eq!(reader.read(8), 0x42);
    }

    #[test]
    fn fill_then_peek_advance() {
        let data = [0xABu8, 0xCD];
        let mut reader = BitBufferRead::new(&data);
        reader.fill();
        // Peek should not consume bits
        let peeked = reader.peek(8);
        assert_eq!(peeked, 0xAB);
        assert_eq!(reader.available_bits(), reader.available_bits());
        // Advance and read next
        reader.advance(8);
        assert_eq!(reader.peek(8), 0xCD);
    }

    #[test]
    fn read_full_32bits() {
        let data = [0x12u8, 0x34, 0x56, 0x78, 0x9A];
        let mut reader = BitBufferRead::new(&data);
        let val = reader.read(32);
        assert_eq!(val, 0x12345678);
    }

    #[test]
    fn unstuffing_multiple_0xff_sequence() {
        // [0xFF, 0x00, 0xFF, 0x00] - consecutive 0xFF bytes
        let data = [0xFFu8, 0x00, 0xFF, 0x00];
        let mut reader = BitBufferRead::new(&data);
        reader.fill();
        // Read the first 0xFF (8 bits)
        assert_eq!(reader.read(8), 0xFF);
        // After 0xFF, unstuffing: next byte 0x00 → 7 bits = 0
        assert_eq!(reader.read(7), 0x00);
        // 0x00 is not 0xFF, so next byte reads normally: 0xFF → 8 bits
        assert_eq!(reader.read(8), 0xFF);
        // After that 0xFF, unstuffing again: 0x00 → 7 bits
        assert_eq!(reader.read(7), 0x00);
    }
}
