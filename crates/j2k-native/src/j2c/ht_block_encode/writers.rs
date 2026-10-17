// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::j2c::encode::allocation::try_untracked_vec_filled;
use crate::EncodeResult;

const MEL_EXP: [usize; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
pub(super) const MEL_SIZE: usize = 192;
pub(super) const VLC_SIZE: usize = 3072 - MEL_SIZE;
pub(super) const MS_SIZE: usize = (16384usize * 16).div_ceil(15);

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
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::try_new().expect("test HTJ2K MEL allocation")
    }

    pub(super) fn try_new() -> EncodeResult<Self> {
        Ok(Self {
            buffer: try_untracked_vec_filled(MEL_SIZE, 0_u8, "HTJ2K MEL reservoir")?,
            pos: 0,
            remaining_bits: 8,
            tmp: 0,
            run: 0,
            k: 0,
            threshold: 1,
        })
    }

    pub(super) fn reset(&mut self) {
        self.pos = 0;
        self.remaining_bits = 8;
        self.tmp = 0;
        self.run = 0;
        self.k = 0;
        self.threshold = 1;
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::inline_always,
        reason = "MEL run lengths are at most six bits and this is the byte-packing hot path"
    )]
    #[inline(always)]
    fn emit_bits(&mut self, bits: u8, count: u8) -> Result<(), &'static str> {
        let combined = (u16::from(self.tmp) << count) | u16::from(bits);
        let remaining = i16::from(self.remaining_bits) - i16::from(count);

        if remaining <= 0 {
            if self.pos >= self.buffer.len() {
                return Err("HTJ2K MEL encoder buffer is full");
            }

            let excess = (-remaining) as u8;
            let byte = (combined >> excess) as u8;
            self.buffer[self.pos] = byte;
            self.pos += 1;
            self.tmp = (combined & ((1u16 << excess) - 1)) as u8;
            self.remaining_bits = (remaining + i16::from(8 - u8::from(byte == 0xFF))) as u8;
        } else {
            self.tmp = combined as u8;
            self.remaining_bits = remaining as u8;
        }

        Ok(())
    }

    pub(super) fn emit_bit(&mut self, bit: bool) -> Result<(), &'static str> {
        self.emit_bits(u8::from(bit), 1)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "MEL exponents are at most five, so the packed run and count fit u8"
    )]
    pub(super) fn encode(&mut self, bit: bool) -> Result<(), &'static str> {
        if bit {
            let t = MEL_EXP[self.k];
            self.emit_bits((self.run & ((1 << t) - 1)) as u8, (t + 1) as u8)?;
            self.run = 0;
            self.k = self.k.saturating_sub(1);
            self.threshold = 1 << MEL_EXP[self.k];
        } else {
            self.run += 1;
            if self.run >= self.threshold {
                self.emit_bit(true)?;
                self.run = 0;
                self.k = (self.k + 1).min(MEL_EXP.len() - 1);
                self.threshold = 1 << MEL_EXP[self.k];
            }
        }

        Ok(())
    }
}

pub(super) struct VlcEncoder {
    pub(super) buffer: Vec<u8>,
    pub(super) pos: usize,
    pub(super) used_bits: u8,
    pub(super) tmp: u64,
    pub(super) last_greater_than_8f: bool,
}

impl VlcEncoder {
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::try_new().expect("test HTJ2K VLC allocation")
    }

    pub(super) fn try_new() -> EncodeResult<Self> {
        let mut buffer = try_untracked_vec_filled(VLC_SIZE, 0_u8, "HTJ2K VLC reservoir")?;
        let last = buffer.len() - 1;
        buffer[last] = 0xFF;

        Ok(Self {
            buffer,
            pos: 1,
            used_bits: 4,
            tmp: 0x0F,
            last_greater_than_8f: true,
        })
    }

    pub(super) fn reset(&mut self) {
        self.pos = 1;
        self.used_bits = 4;
        self.tmp = 0x0F;
        self.last_greater_than_8f = true;
        let last = self.buffer.len() - 1;
        self.buffer[last] = 0xFF;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the mask is bounded by the available byte bits before packing into the VLC reservoir"
    )]
    pub(super) fn encode(
        &mut self,
        codeword: u32,
        mut codeword_len: u8,
    ) -> Result<(), &'static str> {
        if codeword_len == 0 {
            return Ok(());
        }
        let mut codeword = u64::from(codeword);
        if codeword_len < u32::BITS as u8 {
            codeword &= (1u64 << codeword_len) - 1;
        }
        loop {
            let available = 64 - self.used_bits;
            if codeword_len <= available {
                self.tmp |= codeword << self.used_bits;
                self.used_bits += codeword_len;
                return Ok(());
            }
            if available > 0 {
                self.tmp |= codeword << self.used_bits;
                self.used_bits = 64;
                codeword >>= available;
                codeword_len -= available;
            }
            self.drain()?;
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "the reservoir mask bounds the emitted value to one byte in this hot path"
    )]
    #[inline(always)]
    fn drain(&mut self) -> Result<(), &'static str> {
        while self.used_bits >= 8 {
            if self.pos >= self.buffer.len() {
                return Err("HTJ2K VLC encoder buffer is full");
            }

            let needs_stuffing = self.last_greater_than_8f && (self.tmp & 0x7F) == 0x7F;
            let bits = 8 - u8::from(needs_stuffing);
            let byte = (self.tmp & ((1u64 << bits) - 1)) as u8;
            let write_index = self.buffer.len() - 1 - self.pos;
            self.buffer[write_index] = byte;
            self.pos += 1;
            self.tmp >>= bits;
            self.used_bits -= bits;
            self.last_greater_than_8f = byte > 0x8F;
        }

        Ok(())
    }
}

pub(super) struct MagSgnEncoder {
    pub(super) buffer: Vec<u8>,
    pub(super) pos: usize,
    pub(super) used_bits: u8,
    pub(super) tmp: u64,
    pub(super) last_was_ff: bool,
}

impl MagSgnEncoder {
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::try_new().expect("test HTJ2K magnitude/sign allocation")
    }

    pub(super) fn try_new() -> EncodeResult<Self> {
        Ok(Self {
            buffer: try_untracked_vec_filled(MS_SIZE, 0_u8, "HTJ2K magnitude/sign reservoir")?,
            pos: 0,
            used_bits: 0,
            tmp: 0,
            last_was_ff: false,
        })
    }

    pub(super) fn reset(&mut self) {
        self.pos = 0;
        self.used_bits = 0;
        self.tmp = 0;
        self.last_was_ff = false;
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "take and the flushed reservoir are byte-bounded in this magnitude/sign hot path"
    )]
    #[inline(always)]
    pub(super) fn encode(
        &mut self,
        codeword: u32,
        mut codeword_len: u32,
    ) -> Result<(), &'static str> {
        if codeword_len == 0 {
            return Ok(());
        }
        let mut codeword = u64::from(codeword);
        if codeword_len < u32::BITS {
            codeword &= (1u64 << codeword_len) - 1;
        }
        loop {
            let available = 64 - u32::from(self.used_bits);
            if codeword_len <= available {
                self.tmp |= codeword << self.used_bits;
                self.used_bits += codeword_len as u8;
                return Ok(());
            }
            if available > 0 {
                self.tmp |= codeword << self.used_bits;
                self.used_bits = 64;
                codeword >>= available;
                codeword_len -= available;
            }
            self.drain()?;
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "drained chunks are at most eight bytes and this is the MagSgn packing hot path"
    )]
    #[inline(always)]
    fn drain(&mut self) -> Result<(), &'static str> {
        const ONES: u64 = 0x0101_0101_0101_0101;
        const HIGHS: u64 = 0x8080_8080_8080_8080;

        if self.last_was_ff {
            if self.used_bits < 7 {
                return Ok(());
            }
            self.push_byte((self.tmp & 0x7F) as u8)?;
            self.tmp >>= 7;
            self.used_bits -= 7;
            self.last_was_ff = false;
        }

        while self.used_bits >= 8 {
            let byte_count = usize::from((self.used_bits / 8).min(8));
            let valid_mask = if byte_count == 8 {
                u64::MAX
            } else {
                (1u64 << (byte_count * 8)) - 1
            };
            let inverted = !self.tmp;
            let ff_bytes = inverted.wrapping_sub(ONES) & !inverted & HIGHS & valid_mask;
            if ff_bytes == 0 {
                self.push_word(self.tmp, byte_count)?;
                let drained_bits = byte_count * 8;
                self.tmp = if drained_bits == 64 {
                    0
                } else {
                    self.tmp >> drained_bits
                };
                self.used_bits -= drained_bits as u8;
                continue;
            }

            let ff_position = (ff_bytes.trailing_zeros() / 8) as usize;
            let safe_bytes = ff_position + 1;
            self.push_word(self.tmp, safe_bytes)?;
            let drained_bits = safe_bytes * 8;
            self.tmp = if drained_bits == 64 {
                0
            } else {
                self.tmp >> drained_bits
            };
            self.used_bits -= drained_bits as u8;
            if self.used_bits >= 7 {
                self.push_byte((self.tmp & 0x7F) as u8)?;
                self.tmp >>= 7;
                self.used_bits -= 7;
            } else {
                self.last_was_ff = true;
                return Ok(());
            }
        }
        Ok(())
    }

    #[expect(
        clippy::inline_always,
        reason = "single-byte capacity handling stays in the MagSgn drain hot path"
    )]
    #[inline(always)]
    fn push_byte(&mut self, byte: u8) -> Result<(), &'static str> {
        if self.pos >= self.buffer.len() {
            return Err("HTJ2K magnitude/sign encoder buffer is full");
        }
        self.buffer[self.pos] = byte;
        self.pos += 1;
        Ok(())
    }

    #[expect(
        clippy::inline_always,
        reason = "short bulk copies stay in the MagSgn drain hot path"
    )]
    #[inline(always)]
    fn push_word(&mut self, word: u64, byte_count: usize) -> Result<(), &'static str> {
        let end = self
            .pos
            .checked_add(byte_count)
            .ok_or("HTJ2K magnitude/sign encoder buffer is full")?;
        if end > self.buffer.len() {
            return Err("HTJ2K magnitude/sign encoder buffer is full");
        }
        self.buffer[self.pos..end].copy_from_slice(&word.to_le_bytes()[..byte_count]);
        self.pos = end;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "termination flushes a reservoir whose used-bit count is bounded to one byte"
    )]
    pub(super) fn terminate(&mut self) -> Result<(), &'static str> {
        self.drain()?;
        if self.used_bits > 0 {
            let max_bits = if self.last_was_ff { 7 } else { 8 };
            let unused = max_bits - self.used_bits;
            let mut byte = self.tmp & ((1u64 << self.used_bits) - 1);
            byte |= (0xFF & ((1u64 << unused) - 1)) << self.used_bits;

            if byte != 0xFF {
                self.push_byte(byte as u8)?;
            }
        } else if self.last_was_ff {
            self.pos = self.pos.saturating_sub(1);
        }

        Ok(())
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "MEL and VLC termination masks and reservoirs are explicitly bounded to one byte"
)]
pub(super) fn terminate_mel_vlc(
    mel: &mut MelEncoder,
    vlc: &mut VlcEncoder,
) -> Result<(), &'static str> {
    if mel.run > 0 {
        mel.emit_bit(true)?;
    }

    vlc.drain()?;
    if vlc.last_greater_than_8f && (vlc.tmp & 0x7F) == 0x7F {
        if vlc.pos >= vlc.buffer.len() {
            return Err("HTJ2K VLC encoder buffer is full");
        }
        let write_index = vlc.buffer.len() - 1 - vlc.pos;
        vlc.buffer[write_index] = 0x7F;
        vlc.pos += 1;
        vlc.tmp >>= 7;
        vlc.used_bits -= 7;
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

    let vlc_tmp = vlc.tmp as u8;
    let fused = mel.tmp | vlc_tmp;
    let fused_ok =
        (((fused ^ mel.tmp) & mel_mask) | ((fused ^ vlc_tmp) & vlc_mask)) == 0 && fused != 0xFF;

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
        vlc.buffer[write_index] = vlc_tmp;
        vlc.pos += 1;
    }

    Ok(())
}

#[cfg(test)]
mod tests;
