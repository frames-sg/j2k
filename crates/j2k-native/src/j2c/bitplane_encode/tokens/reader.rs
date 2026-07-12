// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked MSB-first bit reader for externally generated Tier-1 tokens.

pub(super) struct ClassicTier1TokenReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> ClassicTier1TokenReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    fn total_bits(&self) -> Result<usize, &'static str> {
        self.bytes
            .len()
            .checked_mul(8)
            .ok_or("classic Tier-1 token buffer bit length overflows")
    }

    pub(super) fn seek(&mut self, bit_pos: usize) -> Result<(), &'static str> {
        if bit_pos > self.total_bits()? {
            return Err("classic Tier-1 token offset exceeds token buffer");
        }
        self.bit_pos = bit_pos;
        Ok(())
    }

    pub(super) fn read_bits(&mut self, count: u8) -> Result<u32, &'static str> {
        let end = self
            .bit_pos
            .checked_add(usize::from(count))
            .ok_or("classic Tier-1 token bit range overflows")?;
        if end > self.total_bits()? {
            return Err("classic Tier-1 token read exceeds token buffer");
        }
        let mut value = 0u32;
        for _ in 0..count {
            let byte = self.bytes[self.bit_pos / 8];
            let shift = 7 - (self.bit_pos % 8);
            value = (value << 1) | u32::from((byte >> shift) & 1);
            self.bit_pos += 1;
        }
        Ok(value)
    }
}
