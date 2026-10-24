//! MQ arithmetic encoder for JPEG 2000 (ITU-T T.800 Annex C).
//!
//! This is the encoding counterpart of `arithmetic_decoder.rs`.
//! It uses the same QE probability table and context state machine.

use alloc::vec::Vec;
use core::hint::select_unpredictable;

use super::encode::allocation::{try_reserve_untracked_bounded, try_untracked_vec};
use super::mq::PACKED_DECODER_STATES;
use crate::{EncodeError, EncodeResult};

/// MQ arithmetic encoder context (identical layout to decoder context).
///
/// Bits 0-6: state index (0-46) into the QE table.
/// Bit 7: MPS (Most Probable Symbol, 0 or 1).
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct ArithmeticEncoderContext(u8);

impl ArithmeticEncoderContext {
    #[cfg(test)]
    pub(crate) fn index(self) -> u32 {
        u32::from(self.0 & 0x7F)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn mps(self) -> u32 {
        u32::from(self.0 >> 7)
    }

    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    pub(crate) fn reset_with_index(&mut self, index: u8) {
        self.0 = index;
    }
}

/// MQ arithmetic encoder (ITU-T T.800 Annex C).
///
/// Uses the direct buffer approach with proper 7-bit stuffing after 0xFF
/// to match the decoder's Annex G byte input convention.
pub(crate) struct ArithmeticEncoder {
    /// Output byte stream. Index 0 is a sentinel byte (0x00).
    data: Vec<u8>,
    /// A-register (interval size), 16-bit precision.
    a: u32,
    /// C-register (code value), 28-bit + carry at bit 27.
    c: u32,
    /// Bit shift counter (12 initially, then 7 after 0xFF, 8 otherwise).
    ct: u32,
    byte_limit: Option<usize>,
    allocation_error: Option<EncodeError>,
}

impl ArithmeticEncoder {
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_capacity(1)
    }

    #[cfg(test)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).expect("legacy MQ encoder allocation")
    }

    pub(crate) fn try_with_capacity(capacity: usize) -> EncodeResult<Self> {
        let mut data = try_untracked_vec(capacity.max(1), "MQ encoder output")?;
        data.push(0x00);

        Ok(Self {
            data, // Sentinel byte at index 0
            a: 0x8000,
            c: 0,
            ct: 12,
            byte_limit: None,
            allocation_error: None,
        })
    }

    /// Construct an MQ encoder that cannot grow beyond the checked payload
    /// plan. The internal sentinel is included in the allocated limit but not
    /// in `payload_bytes` returned to the caller.
    pub(crate) fn try_with_byte_limit(payload_bytes: usize) -> EncodeResult<Self> {
        let byte_limit = payload_bytes
            .checked_add(1)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "MQ encoder payload plus sentinel",
            })?;
        let mut encoder = Self::try_with_capacity(byte_limit.clamp(1, 256))?;
        encoder.byte_limit = Some(byte_limit);
        Ok(encoder)
    }

    /// Encode a single symbol (0 or 1) with the given context (C.2.6 ENCODE).
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the packed MQ state holds Qe in 16 bits and each next state in 8"
    )]
    #[inline(always)]
    pub(crate) fn encode(&mut self, bit: u32, context: &mut ArithmeticEncoderContext) {
        // CODEMPS and CODELPS, T.800 C.2.6, as selects rather than branches, since
        // near-equiprobable contexts make the MPS test unpredictable: with
        // the conditional exchange the MPS keeps the larger sub-interval and
        // the LPS the smaller, C adds Qe exactly when the coded symbol takes
        // the upper one, and the MPS state moves only when A renormalizes.
        let state = PACKED_DECODER_STATES[usize::from(context.0)];
        let qe = (state & 0xFFFF) as u32;
        let a = self.a - qe;
        let is_mps = bit == context.mps();
        self.c += select_unpredictable(is_mps == (a >= qe), qe, 0);
        let next_a = select_unpredictable(is_mps, a.max(qe), a.min(qe));
        let next_mps = select_unpredictable(next_a < 0x8000, (state >> 24) as u8, context.0);
        context.0 = select_unpredictable(is_mps, next_mps, (state >> 32) as u8);
        self.a = next_a;
        self.renormalize(next_a.leading_zeros() - 16);
    }

    /// Renormalize the encoder, T.800 C.2.7 RENORME, by the `shifts` that bring A
    /// back to at least 0x8000, emitting a byte whenever CT runs out.
    #[expect(
        clippy::inline_always,
        reason = "MQ state transitions are measured per-symbol hot paths"
    )]
    #[inline(always)]
    fn renormalize(&mut self, mut shifts: u32) {
        self.a <<= shifts;
        while shifts >= self.ct {
            self.c <<= self.ct;
            shifts -= self.ct;
            self.byte_out();
        }
        self.c <<= shifts;
        self.ct -= shifts;
    }

    /// Output a byte with carry propagation and bit stuffing (C.2.8 BYTEOUT).
    ///
    /// After 0xFF, only 7 bits are extracted (bit stuffing) to prevent
    /// marker-like byte sequences in the output. After an allocation error
    /// the registers keep advancing but nothing more is stored; the error is
    /// returned when the encoder finishes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "MQ register masks bound every emitted chunk to one byte"
    )]
    fn byte_out(&mut self) {
        let last_byte = *self.data.last().unwrap();
        if last_byte == 0xFF {
            // 7-bit mode after 0xFF (bit stuffing)
            let b = (self.c >> 20) as u8;
            self.try_push_byte(b);
            self.c &= 0x000F_FFFF;
            self.ct = 7;
        } else if self.c & 0x0800_0000 == 0 {
            // No carry: normal 8-bit output
            let b = (self.c >> 19) as u8;
            self.try_push_byte(b);
            self.c &= 0x0007_FFFF;
            self.ct = 8;
        } else {
            // Carry occurred (bit 27 set): propagate into last byte
            let last = self.data.last_mut().unwrap();
            *last += 1;
            self.c &= 0x07FF_FFFF; // Clear carry bit
            if *last == 0xFF {
                // Carry made last byte 0xFF: switch to 7-bit mode
                let b = (self.c >> 20) as u8;
                self.try_push_byte(b);
                self.c &= 0x000F_FFFF;
                self.ct = 7;
            } else {
                let b = (self.c >> 19) as u8;
                self.try_push_byte(b);
                self.c &= 0x0007_FFFF;
                self.ct = 8;
            }
        }
    }

    /// SETBITS procedure (C.2.9).
    fn set_bits(&mut self) {
        let temp = self.c + self.a;
        self.c |= 0xFFFF;
        if self.c >= temp {
            self.c -= 0x8000;
        }
    }

    /// Flush the encoder state (C.2.9 FLUSH).
    pub(crate) fn flush(&mut self) {
        self.set_bits();
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= self.ct;
        self.byte_out();
    }

    /// Return the encoded data (excluding sentinel), consuming the encoder.
    #[cfg(test)]
    pub(crate) fn finish(self) -> Vec<u8> {
        self.finish_checked().expect("legacy MQ encoder output")
    }

    pub(crate) fn finish_checked(mut self) -> EncodeResult<Vec<u8>> {
        self.flush();
        if let Some(error) = self.allocation_error.take() {
            return Err(error);
        }
        // T.800 B.10.7: code-block bytes must not end in 0xFF. A pending
        // terminal 0xFF is omitted; the decoder supplies 0xFF past the end.
        if self.data.len() > 1 && self.data.last() == Some(&0xFF) {
            self.data.pop();
        }
        // Remove sentinel byte at index 0
        self.data.drain(..1);
        Ok(self.data)
    }

    fn try_push_byte(&mut self, byte: u8) {
        if self.allocation_error.is_some() {
            return;
        }
        if self
            .byte_limit
            .is_some_and(|limit| self.data.len() >= limit)
        {
            self.allocation_error = Some(EncodeError::InternalInvariant {
                what: "MQ encoder exceeded its checked payload plan",
            });
            return;
        }
        let byte_limit = self.byte_limit.unwrap_or(usize::MAX);
        if let Err(error) =
            try_reserve_untracked_bounded(&mut self.data, 1, byte_limit, "MQ encoder output")
        {
            self.allocation_error = Some(error);
            return;
        }
        self.data.push(byte);
    }
}

#[cfg(test)]
mod tests;
