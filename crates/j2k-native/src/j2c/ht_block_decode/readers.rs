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

    fn fill(&mut self) {
        while self.bits <= 32 {
            let byte = if self.pos < self.data.len() {
                let byte = self.data[self.pos];
                self.pos += 1;
                byte
            } else {
                PAD
            };

            self.tmp |= u64::from(byte) << self.bits;
            self.bits += 8 - u32::from(self.unstuff);
            self.unstuff = byte == 0xFF;
        }
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

    #[expect(clippy::cast_sign_loss, reason = "nonnegative live cursor")]
    fn fill(&mut self) {
        while self.bits <= 32 {
            let byte = if self.remaining > 0 {
                let byte = self.data[self.pos as usize];
                self.pos -= 1;
                self.remaining -= 1;
                byte
            } else {
                0
            };

            let d_bits = 8 - u32::from(self.unstuff && (byte & 0x7F) == 0x7F);
            self.tmp |= u64::from(byte) << self.bits;
            self.bits += d_bits;
            self.unstuff = byte > 0x8F;
        }
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
mod tests {
    use super::{ForwardBitReader, MelDecoder, ReverseBitReader};

    #[test]
    fn reader_state_and_bit_consumption_match_pre_split_goldens() {
        let data = [0xAA, 0xFF, 0x01, 0x7F, 0x80];
        let mut forward = ForwardBitReader::<0xFF>::new(&data);
        assert_eq!(forward.fetch(), 0x3F81_FFAA);
        forward.advance(5);
        assert_eq!(forward.fetch(), 0x01FC_0FFD);
        forward.advance(19);
        assert_eq!(forward.fetch(), 0xFFFF_C03F);
        assert_eq!(
            (forward.pos, forward.bits, forward.tmp, forward.unstuff),
            (5, 37, 0x0000_003F_FFFF_C03F, true)
        );

        let mut reverse = ReverseBitReader::new_mrp(&data);
        assert_eq!(reverse.fetch(), 0xFF01_7F80);
        assert_eq!(reverse.advance(7), 0x55FE_02FF);
        assert_eq!(reverse.fetch(), 0x55FE_02FF);
        assert_eq!(
            (
                reverse.pos,
                reverse.remaining,
                reverse.bits,
                reverse.tmp,
                reverse.unstuff,
            ),
            (-1, 0, 33, 0x0000_0001_55FE_02FF, true)
        );

        let mel_data = [0x12, 0x34, 0x56, 0x78];
        let mut mel = MelDecoder::new(&mel_data, mel_data.len(), 2);
        let mut runs = [0i32; 8];
        for run in &mut runs {
            *run = mel.get_run().expect("MEL run");
        }
        assert_eq!(runs, [1, 0, 1, 0, 0, 0, 2, 2]);
        assert_eq!(
            (
                mel.pos,
                mel.remaining,
                mel.bits_left,
                mel.k,
                mel.num_runs,
                mel.runs,
                mel.unstuff,
            ),
            (3, 0, 0, 5, 0, 0, false)
        );

        let vlc_data = [0x12, 0x34, 0x56, 0x78, 0x9A];
        let mut vlc = ReverseBitReader::new_vlc(&vlc_data, vlc_data.len(), 3);
        assert_eq!(vlc.fetch(), 0x0000_02B7);
        assert_eq!(vlc.advance(9), 0x0000_0001);
        assert_eq!(vlc.fetch(), 0x0000_0001);
        assert_eq!(
            (vlc.pos, vlc.remaining, vlc.bits, vlc.tmp, vlc.unstuff),
            (1, 0, 34, 1, false)
        );
    }
}
