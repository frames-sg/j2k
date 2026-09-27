// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) struct MelDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    remaining: usize,
    unstuff: bool,
    current_byte: u8,
    bits_left: u8,
    k: usize,
    num_runs: usize,
    runs: u64,
}

impl<'a> MelDecoder<'a> {
    pub(super) fn new(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        Self {
            data,
            pos: lcup - scup,
            remaining: scup - 1,
            unstuff: false,
            current_byte: 0,
            bits_left: 0,
            k: 0,
            num_runs: 0,
            runs: 0,
        }
    }

    fn read_bit(&mut self) -> Option<u32> {
        if self.bits_left == 0 {
            let mut byte = if self.remaining > 0 {
                let byte = self.data.get(self.pos).copied()?;
                self.pos += 1;
                self.remaining -= 1;
                byte
            } else {
                0xFF
            };

            if self.remaining == 0 {
                byte |= 0x0F;
            }

            self.current_byte = byte;
            self.bits_left = 8 - u8::from(self.unstuff);
            self.unstuff = byte == 0xFF;
        }

        self.bits_left -= 1;
        Some(u32::from((self.current_byte >> self.bits_left) & 1))
    }

    fn read_bits(&mut self, count: usize) -> Option<u32> {
        let mut value = 0;

        for _ in 0..count {
            value = (value << 1) | self.read_bit()?;
        }

        Some(value)
    }

    fn decode_more_runs(&mut self) -> Option<()> {
        const MEL_EXP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

        while self.num_runs < 8 {
            let eval = MEL_EXP[self.k];
            let first = self.read_bit()?;
            let run = if first == 1 {
                self.k = (self.k + 1).min(12);
                ((1usize << eval) - 1) << 1
            } else {
                self.k = self.k.saturating_sub(1);
                (self.read_bits(eval)? as usize) << 1 | 1
            };

            self.runs |= (run as u64) << (self.num_runs * 7);
            self.num_runs += 1;

            if eval == 5 && first == 0 && self.num_runs >= 8 {
                break;
            }
        }

        Some(())
    }

    pub(super) fn get_run(&mut self) -> Option<i32> {
        if self.num_runs == 0 {
            self.decode_more_runs()?;
        }

        let run = (self.runs & 0x7F) as i32;
        self.runs >>= 7;
        self.num_runs -= 1;
        Some(run)
    }
}

/// `Clone` (not `Copy`) keeps the by-value refill hand-offs explicit.
#[derive(Clone)]
pub(super) struct ForwardBitReader<'a, const PAD: u8> {
    data: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a, const PAD: u8> ForwardBitReader<'a, PAD> {
    pub(super) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            tmp: 0,
            bits: 0,
            unstuff: false,
        }
    }

    /// Top the reservoir up to at least 33 bits, as the byte loop does. A byte
    /// after `0xFF` carries seven bits, so four bytes with no `0xFF` before
    /// or among their first three are added as one word; stuffing, an empty
    /// reservoir, and the padded tail take the byte loop.
    #[expect(clippy::inline_always, reason = "per-sample refill check")]
    #[inline(always)]
    fn fill(&mut self) {
        if !self.unstuff && self.bits > 0 {
            if let Some(&[b0, b1, b2, b3]) = self.data.get(self.pos..self.pos + 4) {
                let word = u32::from_le_bytes([b0, b1, b2, b3]);
                if b0 != 0xFF && b1 != 0xFF && b2 != 0xFF {
                    self.tmp |= u64::from(word) << self.bits;
                    self.bits += 32;
                    self.pos += 4;
                    self.unstuff = b3 == 0xFF;
                    return;
                }
            }
        }
        *self = self.clone().fill_bytes();
    }

    /// Taking and returning the reader by value keeps the caller's local
    /// reader out of memory across this cold call.
    #[cold]
    #[inline(never)]
    fn fill_bytes(mut self) -> Self {
        while self.bits <= 32 {
            let byte = if self.pos < self.data.len() {
                let byte = self.data[self.pos];
                self.pos += 1;
                byte
            } else {
                PAD
            };

            let valid_bits = 8 - u32::from(self.unstuff);
            let next_unstuff = byte == 0xFF;
            let byte = if self.unstuff { byte & 0x7F } else { byte };
            self.tmp |= u64::from(byte) << self.bits;
            self.bits += valid_bits;
            self.unstuff = next_unstuff;
        }
        self
    }

    #[expect(clippy::cast_possible_truncation, reason = "low reservoir word")]
    pub(super) fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    pub(super) fn advance(&mut self, count: u32) {
        debug_assert!(count <= self.bits);
        self.tmp >>= count;
        self.bits -= count;
    }
}

/// `Clone` (not `Copy`) keeps the by-value refill hand-offs explicit.
#[derive(Clone)]
pub(super) struct ReverseBitReader<'a> {
    data: &'a [u8],
    pos: isize,
    remaining: usize,
    tmp: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a> ReverseBitReader<'a> {
    #[expect(clippy::cast_possible_wrap, reason = "validated signed cursor")]
    pub(super) fn new_vlc(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        let d = data[lcup - 2];
        let tmp = u64::from(d >> 4);
        let bits = 4 - u32::from((tmp & 0x7) == 0x7);

        Self {
            data,
            pos: lcup as isize - 3,
            remaining: scup - 2,
            tmp,
            bits,
            unstuff: (d | 0x0F) > 0x8F,
        }
    }

    #[expect(clippy::cast_possible_wrap, reason = "validated signed cursor")]
    pub(super) fn new_mrp(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: data.len() as isize - 1,
            remaining: data.len(),
            tmp: 0,
            bits: 0,
            unstuff: true,
        }
    }

    /// Top the reservoir up to at least 33 bits, as the byte loop does. Read
    /// backward, a byte is stuffed when it ends in seven ones after a byte
    /// above 0x8F; four bytes with no such pair are added as one word, and
    /// stuffing, an empty reservoir, and the padded tail take the byte loop.
    #[expect(
        clippy::cast_sign_loss,
        clippy::inline_always,
        reason = "nonnegative live cursor; per-quad refill check"
    )]
    #[inline(always)]
    fn fill(&mut self) {
        if self.bits > 0 && self.remaining >= 4 && self.pos >= 3 {
            let last = self.pos as usize;
            if let Some(&[b3, b2, b1, b0]) = self.data.get(last - 3..=last) {
                // `b0` is consumed first, so it lands in the low byte. Each
                // per-byte flag below sits in that byte's bit 7.
                let word = u32::from_le_bytes([b0, b1, b2, b3]);
                // Above 0x8F: bit 7 set and bits 4-6 nonzero.
                let above_8f = word & 0x8080_8080 & ((word & 0x7070_7070) + 0x7070_7070);
                // Low seven bits all ones.
                let low_ones = ((word & 0x7F7F_7F7F) + 0x0101_0101) & 0x8080_8080;
                let after_above_8f = (above_8f << 8) | (u32::from(self.unstuff) << 7);
                if after_above_8f & low_ones == 0 {
                    self.tmp |= u64::from(word) << self.bits;
                    self.bits += 32;
                    self.pos -= 4;
                    self.remaining -= 4;
                    self.unstuff = above_8f >> 31 != 0;
                    return;
                }
            }
        }
        *self = self.clone().fill_bytes();
    }

    /// By value for the same reason as [`ForwardBitReader::fill_bytes`].
    #[expect(clippy::cast_sign_loss, reason = "nonnegative live cursor")]
    #[cold]
    #[inline(never)]
    fn fill_bytes(mut self) -> Self {
        while self.bits <= 32 {
            let byte = if self.remaining > 0 {
                let byte = self.data[self.pos as usize];
                self.pos -= 1;
                self.remaining -= 1;
                byte
            } else {
                0
            };

            let stuffed = self.unstuff && (byte & 0x7F) == 0x7F;
            let d_bits = 8 - u32::from(stuffed);
            let next_unstuff = byte > 0x8F;
            let byte = if stuffed { byte & 0x7F } else { byte };
            self.tmp |= u64::from(byte) << self.bits;
            self.bits += d_bits;
            self.unstuff = next_unstuff;
        }
        self
    }

    #[expect(clippy::cast_possible_truncation, reason = "low reservoir word")]
    pub(super) fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    #[expect(clippy::cast_possible_truncation, reason = "low reservoir word")]
    pub(super) fn advance(&mut self, count: u32) -> u32 {
        debug_assert!(count <= self.bits);
        self.tmp >>= count;
        self.bits -= count;
        self.tmp as u32
    }
}

#[expect(clippy::inline_always, reason = "inline two loads in refinement scans")]
#[inline(always)]
pub(super) fn read_u32_pair(values: &[u16], index: usize) -> u32 {
    u32::from(values[index]) | (u32::from(values[index + 1]) << 16)
}

#[cfg(test)]
mod tests;
