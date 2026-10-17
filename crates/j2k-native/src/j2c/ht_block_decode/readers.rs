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

    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "low reservoir word; per-sample hot path"
    )]
    #[inline(always)]
    pub(super) fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    #[expect(clippy::inline_always, reason = "per-sample hot path")]
    #[inline(always)]
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
            unstuff: d > 0x8F,
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

    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "low reservoir word; per-quad hot path"
    )]
    #[inline(always)]
    pub(super) fn fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }

        self.tmp as u32
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "low reservoir word; per-quad hot path"
    )]
    #[inline(always)]
    pub(super) fn advance(&mut self, count: u32) -> u32 {
        debug_assert!(count <= self.bits);
        self.tmp >>= count;
        self.bits -= count;
        self.tmp as u32
    }
}

/// Padding after the unstuffed `MagSgn` bytes. It is at least one fetch window,
/// so a clamped fetch past the end sees only padding.
const FLAT_MAGSGN_PADDING: usize = 16;

/// Bytes of unstuffed `MagSgn` data a block of `samples` can consume, plus
/// padding. Each sample consumes at most 32 bits and peeks one more.
pub(super) fn flat_magsgn_capacity(samples: usize) -> crate::Result<usize> {
    samples
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(8 + FLAT_MAGSGN_PADDING))
        .ok_or(crate::ValidationError::ImageTooLarge.into())
}

/// Unstuff a forward `MagSgn` segment (Clause 7.3.2: a byte after `0xFF`
/// carries seven bits) into `out`, then pad it with ones. The padding is what
/// [`ForwardBitReader`] reads past the end of the segment, so every fetch
/// returns the same bits. Only the first `consumable` bytes can be read by a
/// block, so a longer (malformed) segment stops there.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the low byte of the bit accumulator is emitted on purpose"
)]
pub(super) fn unstuff_magsgn<'a>(
    src: &[u8],
    out: &'a mut alloc::vec::Vec<u8>,
    consumable: usize,
) -> Option<&'a [u8]> {
    const LOW_SEVEN: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let required = consumable.checked_add(8 + FLAT_MAGSGN_PADDING)?;
    if out.capacity() < required {
        return None;
    }
    out.clear();
    let mut acc = 0_u64;
    let mut bits = 0_u32;
    let mut unstuff = false;
    let mut index = 0usize;
    while index < src.len() && out.len() < consumable {
        if !unstuff {
            if let Some(chunk) = src.get(index..index + 8) {
                let word = u64::from_le_bytes(chunk.try_into().ok()?);
                // Bit 7 of each byte is set where that byte is 0xFF.
                let ff = ((word & LOW_SEVEN) + ONES) & word & HIGH;
                if ff & (HIGH >> 8) == 0 {
                    // No stuffing inside the first seven bytes: all 64 bits
                    // are data, appended after the `bits` pending bits.
                    out.extend_from_slice(&(acc | (word << bits)).to_le_bytes());
                    acc = if bits == 0 { 0 } else { word >> (64 - bits) };
                    unstuff = ff >> 63 != 0;
                    index += 8;
                    continue;
                }
            }
        }
        let byte = src[index];
        index += 1;
        let data = if unstuff { byte & 0x7F } else { byte };
        acc |= u64::from(data) << bits;
        bits += 8 - u32::from(unstuff);
        unstuff = byte == 0xFF;
        while bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
    if bits > 0 {
        out.push((acc | (u64::MAX << bits)) as u8);
    }
    out.truncate(consumable);
    out.resize(out.len() + 8 + FLAT_MAGSGN_PADDING, 0xFF);
    Some(out.as_slice())
}

/// `MagSgn` bits read at absolute bit positions from an unstuffed buffer.
///
/// Unlike a reservoir reader, consecutive fetches depend only on the running
/// position, so the per-sample critical path is one addition.
pub(super) struct FlatBitReader<'a> {
    data: &'a [u8],
    last_window: usize,
    #[cfg(feature = "simd")]
    last_window16: usize,
    position: usize,
}

impl<'a> FlatBitReader<'a> {
    pub(super) fn new(data: &'a [u8]) -> Option<Self> {
        Some(Self {
            last_window: data.len().checked_sub(8 + FLAT_MAGSGN_PADDING)? + FLAT_MAGSGN_PADDING,
            // At least 24 bytes are present here, so this cannot underflow.
            #[cfg(feature = "simd")]
            last_window16: data.len() - 16,
            data,
            position: 0,
        })
    }

    /// Absolute bit position of the next unread bit.
    #[cfg(feature = "simd")]
    #[expect(clippy::inline_always, reason = "per-quad hot path")]
    #[inline(always)]
    pub(super) fn position(&self) -> usize {
        self.position
    }

    #[cfg(feature = "simd")]
    #[expect(clippy::inline_always, reason = "per-quad hot path")]
    #[inline(always)]
    pub(super) fn advance_to(&mut self, position: usize) {
        self.position = position;
    }

    /// The 16 bytes starting at the byte that holds bit `position`, and that
    /// bit's offset within the byte. A window past the data is clamped to the
    /// last 16 bytes, which are padding (all ones) like everything after the
    /// data, so the bits read are unchanged.
    #[cfg(feature = "simd")]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "bit offset below 8; per-quad hot path"
    )]
    #[inline(always)]
    pub(super) fn window16(&self, position: usize) -> (&[u8], u32) {
        let byte = (position >> 3).min(self.last_window16);
        (&self.data[byte..byte + 16], (position & 7) as u32)
    }

    /// The next 32 bits, least significant first.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "low window word; per-sample hot path"
    )]
    #[inline(always)]
    pub(super) fn fetch(&self) -> u32 {
        // Past the data every window is padding, which is all ones at any
        // bit offset, so clamping the window is exact.
        let byte = (self.position >> 3).min(self.last_window);
        let window = self.data[byte..byte + 8]
            .try_into()
            .map_or(u64::MAX, u64::from_le_bytes);
        (window >> (self.position & 7)) as u32
    }

    #[expect(clippy::inline_always, reason = "per-sample hot path")]
    #[inline(always)]
    pub(super) fn advance(&mut self, count: u32) {
        self.position += count as usize;
    }
}

#[expect(clippy::inline_always, reason = "inline two loads in refinement scans")]
#[inline(always)]
pub(super) fn read_u32_pair(values: &[u16], index: usize) -> u32 {
    u32::from(values[index]) | (u32::from(values[index + 1]) << 16)
}

#[cfg(test)]
mod tests;
