// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

const MEL_EXP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
const MEL_SIZE: usize = 192;
const VLC_SIZE: usize = 3072 - MEL_SIZE;
const MS_SIZE: usize = (16384usize * 16).div_ceil(15);

pub(super) struct MelEncoder {
    pub(super) buffer: Vec<u8>,
    pub(super) pos: usize,
    pub(super) remaining_bits: u8,
    pub(super) tmp: u8,
    pub(super) run: usize,
    pub(super) k: usize,
    pub(super) threshold: usize,
}

impl MelEncoder {
    pub(super) fn new() -> Self {
        Self {
            buffer: vec![0; MEL_SIZE],
            pos: 0,
            remaining_bits: 8,
            tmp: 0,
            run: 0,
            k: 0,
            threshold: 1,
        }
    }

    pub(super) fn emit_bit(&mut self, bit: bool) -> Result<(), &'static str> {
        self.tmp = (self.tmp << 1) | u8::from(bit);
        self.remaining_bits -= 1;

        if self.remaining_bits == 0 {
            if self.pos >= self.buffer.len() {
                return Err("HTJ2K MEL encoder buffer is full");
            }

            self.buffer[self.pos] = self.tmp;
            self.pos += 1;
            self.remaining_bits = if self.tmp == 0xFF { 7 } else { 8 };
            self.tmp = 0;
        }

        Ok(())
    }

    pub(super) fn encode(&mut self, bit: bool) -> Result<(), &'static str> {
        if !bit {
            self.run += 1;
            if self.run >= self.threshold {
                self.emit_bit(true)?;
                self.run = 0;
                self.k = (self.k + 1).min(MEL_EXP.len() - 1);
                self.threshold = 1 << MEL_EXP[self.k];
            }
        } else {
            self.emit_bit(false)?;
            let mut t = MEL_EXP[self.k];
            while t > 0 {
                t -= 1;
                self.emit_bit(((self.run >> t) & 1) != 0)?;
            }
            self.run = 0;
            self.k = self.k.saturating_sub(1);
            self.threshold = 1 << MEL_EXP[self.k];
        }

        Ok(())
    }
}

pub(super) struct VlcEncoder {
    pub(super) buffer: Vec<u8>,
    pub(super) pos: usize,
    pub(super) used_bits: u8,
    pub(super) tmp: u8,
    pub(super) last_greater_than_8f: bool,
}

impl VlcEncoder {
    pub(super) fn new() -> Self {
        let mut buffer = vec![0; VLC_SIZE];
        let last = buffer.len() - 1;
        buffer[last] = 0xFF;

        Self {
            buffer,
            pos: 1,
            used_bits: 4,
            tmp: 0x0F,
            last_greater_than_8f: true,
        }
    }

    pub(super) fn encode(
        &mut self,
        mut codeword: u32,
        mut codeword_len: u8,
    ) -> Result<(), &'static str> {
        while codeword_len > 0 {
            if self.pos >= self.buffer.len() {
                return Err("HTJ2K VLC encoder buffer is full");
            }

            let mut available_bits = 8 - u8::from(self.last_greater_than_8f) - self.used_bits;
            let take = available_bits.min(codeword_len);
            let mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            self.tmp |= ((codeword & mask) as u8) << self.used_bits;
            self.used_bits += take;
            available_bits -= take;
            codeword_len -= take;
            codeword >>= take;

            if available_bits == 0 {
                if self.last_greater_than_8f && self.tmp != 0x7F {
                    self.last_greater_than_8f = false;
                    continue;
                }

                let write_index = self.buffer.len() - 1 - self.pos;
                self.buffer[write_index] = self.tmp;
                self.pos += 1;
                self.last_greater_than_8f = self.tmp > 0x8F;
                self.tmp = 0;
                self.used_bits = 0;
            }
        }

        Ok(())
    }
}

pub(super) struct MagSgnEncoder {
    pub(super) buffer: Vec<u8>,
    pub(super) pos: usize,
    pub(super) max_bits: u8,
    pub(super) used_bits: u8,
    pub(super) tmp: u32,
}

impl MagSgnEncoder {
    pub(super) fn new() -> Self {
        Self {
            buffer: vec![0; MS_SIZE],
            pos: 0,
            max_bits: 8,
            used_bits: 0,
            tmp: 0,
        }
    }

    #[inline(always)]
    pub(super) fn encode(
        &mut self,
        mut codeword: u32,
        mut codeword_len: u32,
    ) -> Result<(), &'static str> {
        while codeword_len > 0 {
            if self.pos >= self.buffer.len() {
                return Err("HTJ2K magnitude/sign encoder buffer is full");
            }

            let take = u32::from(self.max_bits - self.used_bits).min(codeword_len);
            let mask = if take == 32 {
                u32::MAX
            } else {
                (1u32 << take) - 1
            };
            self.tmp |= (codeword & mask) << self.used_bits;
            self.used_bits += take as u8;
            codeword >>= take;
            codeword_len -= take;

            if self.used_bits >= self.max_bits {
                self.buffer[self.pos] = self.tmp as u8;
                self.pos += 1;
                self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
                self.tmp = 0;
                self.used_bits = 0;
            }
        }

        Ok(())
    }

    pub(super) fn terminate(&mut self) -> Result<(), &'static str> {
        if self.used_bits > 0 {
            let unused = self.max_bits - self.used_bits;
            self.tmp |= (0xFF & ((1u32 << unused) - 1)) << self.used_bits;
            self.used_bits += unused;

            if self.tmp != 0xFF {
                if self.pos >= self.buffer.len() {
                    return Err("HTJ2K magnitude/sign encoder buffer is full");
                }

                self.buffer[self.pos] = self.tmp as u8;
                self.pos += 1;
            }
        } else if self.max_bits == 7 {
            self.pos = self.pos.saturating_sub(1);
        }

        Ok(())
    }
}

pub(super) fn terminate_mel_vlc(
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
) -> Result<(), &'static str> {
    if mel.run > 0 {
        mel.emit_bit(true)?;
    }

    mel.tmp = (u16::from(mel.tmp) << mel.remaining_bits) as u8;
    let mel_mask = ((0xFFu16 << mel.remaining_bits) & 0xFF) as u8;
    let vlc_mask = if vlc.used_bits == 0 {
        0
    } else {
        ((1u16 << vlc.used_bits) - 1) as u8
    };

    if (mel_mask | vlc_mask) == 0 {
        return Ok(());
    }

    let fused = mel.tmp | vlc.tmp;
    let fused_ok =
        (((fused ^ mel.tmp) & mel_mask) | ((fused ^ vlc.tmp) & vlc_mask)) == 0 && fused != 0xFF;

    if fused_ok && vlc.pos > 1 {
        if mel.pos >= mel.buffer.len() {
            return Err("HTJ2K MEL encoder buffer is full");
        }

        mel.buffer[mel.pos] = fused;
        mel.pos += 1;
    } else {
        if mel.pos >= mel.buffer.len() {
            return Err("HTJ2K MEL encoder buffer is full");
        }
        if vlc.pos >= vlc.buffer.len() {
            return Err("HTJ2K VLC encoder buffer is full");
        }

        mel.buffer[mel.pos] = mel.tmp;
        mel.pos += 1;
        let write_index = vlc.buffer.len() - 1 - vlc.pos;
        vlc.buffer[write_index] = vlc.tmp;
        vlc.pos += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{terminate_mel_vlc, MagSgnEncoder, MelEncoder, VlcEncoder};

    #[test]
    fn writer_state_and_termination_match_pre_split_goldens() {
        let mut mel = MelEncoder::new();
        for bit in [
            false, false, true, false, true, true, false, false, false, true,
        ] {
            mel.encode(bit).expect("MEL bit");
        }
        let mut vlc = VlcEncoder::new();
        vlc.encode(0b101101, 6).expect("VLC bits");
        vlc.encode(0x1ff, 9).expect("VLC bits");
        terminate_mel_vlc(&mut mel, &mut vlc).expect("terminate MEL/VLC");

        assert_eq!(
            (
                mel.pos,
                mel.remaining_bits,
                mel.tmp,
                mel.run,
                mel.k,
                mel.threshold,
            ),
            (2, 5, 0x80, 0, 2, 1)
        );
        assert_eq!(&mel.buffer[..mel.pos], &[0xD3, 0x87]);
        assert_eq!(
            (vlc.pos, vlc.used_bits, vlc.tmp, vlc.last_greater_than_8f),
            (3, 3, 0x07, true)
        );
        assert_eq!(
            &vlc.buffer[vlc.buffer.len() - vlc.pos..],
            &[0xFE, 0xDF, 0xFF]
        );

        let mut magsgn = MagSgnEncoder::new();
        magsgn.encode(0xff, 8).expect("MagSgn bits");
        magsgn.encode(0x55, 7).expect("MagSgn bits");
        magsgn.encode(0x3, 2).expect("MagSgn bits");
        magsgn.terminate().expect("terminate MagSgn");

        assert_eq!(
            (magsgn.pos, magsgn.max_bits, magsgn.used_bits, magsgn.tmp),
            (2, 8, 8, 0xFF)
        );
        assert_eq!(&magsgn.buffer[..magsgn.pos], &[0xFF, 0x55]);
    }
}
