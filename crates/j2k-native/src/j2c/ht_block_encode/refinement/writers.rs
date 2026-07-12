// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{try_reserve_untracked_bounded, try_untracked_vec};
use crate::{EncodeError, EncodeResult};

pub(super) struct ForwardRefinementBitWriter {
    data: Vec<u8>,
    byte_limit: usize,
    used_bits: u8,
    max_bits: u8,
    tmp: u8,
}

impl ForwardRefinementBitWriter {
    pub(super) fn try_new(byte_limit: usize) -> EncodeResult<Self> {
        Ok(Self {
            data: try_untracked_vec(byte_limit, "HTJ2K SigProp refinement")?,
            byte_limit,
            used_bits: 0,
            max_bits: 8,
            tmp: 0,
        })
    }

    pub(super) fn push_bit(&mut self, bit: bool) -> EncodeResult<()> {
        if bit {
            self.tmp |= 1 << self.used_bits;
        }
        self.used_bits += 1;
        if self.used_bits == self.max_bits {
            self.flush_full_byte()?;
        }
        Ok(())
    }

    fn flush_full_byte(&mut self) -> EncodeResult<()> {
        self.try_push(self.tmp)?;
        self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
        self.tmp = 0;
        self.used_bits = 0;
        Ok(())
    }

    pub(super) fn finish(mut self) -> EncodeResult<Vec<u8>> {
        if self.used_bits > 0 {
            self.try_push(self.tmp)?;
        }
        if self.data.is_empty() {
            self.try_push(0)?;
        }
        Ok(self.data)
    }

    fn try_push(&mut self, byte: u8) -> EncodeResult<()> {
        if self.data.len() >= self.byte_limit {
            return Err(EncodeError::InternalInvariant {
                what: "HTJ2K SigProp output exceeded its checked bound",
            });
        }
        try_reserve_untracked_bounded(
            &mut self.data,
            1,
            self.byte_limit,
            "HTJ2K SigProp refinement",
        )?;
        self.data.push(byte);
        Ok(())
    }
}

pub(super) struct ReverseRefinementBitWriter {
    bits: Vec<bool>,
    bit_limit: usize,
    byte_limit: usize,
}

impl ReverseRefinementBitWriter {
    pub(super) fn try_new(bit_limit: usize, byte_limit: usize) -> EncodeResult<Self> {
        Ok(Self {
            bits: try_untracked_vec(bit_limit, "HTJ2K MagRef bit staging")?,
            bit_limit,
            byte_limit,
        })
    }

    pub(super) fn push_bit(&mut self, bit: bool) -> EncodeResult<()> {
        if self.bits.len() >= self.bit_limit {
            return Err(EncodeError::InternalInvariant {
                what: "HTJ2K MagRef bits exceeded their checked bound",
            });
        }
        self.bits.push(bit);
        Ok(())
    }

    pub(super) fn finish(self) -> EncodeResult<Vec<u8>> {
        let mut read_order = try_untracked_vec(self.byte_limit, "HTJ2K MagRef refinement")?;
        let mut offset = 0usize;
        let mut unstuff = true;

        while offset < self.bits.len() {
            let remaining = self.bits.len() - offset;
            let first_seven_are_ones =
                remaining >= 7 && self.bits[offset..offset + 7].iter().all(|bit| *bit);
            let capacity = if unstuff && first_seven_are_ones {
                7
            } else {
                8
            };
            let take = capacity.min(remaining);
            let mut byte = 0u8;
            for bit_idx in 0..take {
                if self.bits[offset + bit_idx] {
                    byte |= 1 << bit_idx;
                }
            }
            if read_order.len() >= self.byte_limit {
                return Err(EncodeError::InternalInvariant {
                    what: "HTJ2K MagRef output exceeded its checked bound",
                });
            }
            read_order.push(byte);
            offset += take;
            unstuff = byte > 0x8F;
        }

        if read_order.is_empty() {
            if self.byte_limit == 0 {
                return Err(EncodeError::InternalInvariant {
                    what: "HTJ2K MagRef empty output has no planned terminator",
                });
            }
            read_order.push(0);
        }
        read_order.reverse();
        Ok(read_order)
    }
}

#[cfg(test)]
mod tests {
    use super::{ForwardRefinementBitWriter, ReverseRefinementBitWriter};

    #[test]
    fn refinement_writer_stuffing_matches_pre_split_goldens() {
        let mut forward = ForwardRefinementBitWriter::try_new(3).expect("forward allocation");
        for bit in [
            true, true, true, true, true, true, true, true, true, false, true, false, true, false,
            true, true, false,
        ] {
            forward.push_bit(bit).expect("forward bit");
        }
        assert_eq!(
            forward.finish().expect("forward finish"),
            [0xFF, 0x55, 0x01]
        );

        let mut reverse = ReverseRefinementBitWriter::try_new(16, 3).expect("reverse allocation");
        for bit in [
            true, true, true, true, true, true, true, true, false, true, false, true, false, true,
            true, false,
        ] {
            reverse.push_bit(bit).expect("reverse bit");
        }
        assert_eq!(
            reverse.finish().expect("reverse finish"),
            [0x00, 0xD5, 0x7F]
        );
    }
}
